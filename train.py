import torch
import warnings
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, BitsAndBytesConfig
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import logging as hf_logging
import gc
from tqdm import tqdm

from encoder import DocVLMEncoder
from model import DocVLM
from train_dataset import DocVQADataset , collate_fn

warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

def load_qwen(quantize=True):
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if quantize else None


    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        quantization_config=cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={
            0: "10GiB",   # GPU gets 10GB
            "cpu": "32GiB"  # rest spills to CPU RAM
        }
    )

    for param in model.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    return model, processor

def train():
    torch.cuda.empty_cache()
    gc.collect()

    qwen, processor = load_qwen()
    
    encoder = DocVLMEncoder(
        num_queries=64,
        qwen_hidden_dim=qwen.config.hidden_size,
    ).cuda().to(torch.float32)


    docvlm = DocVLM(qwen, encoder)

    dataset = DocVQADataset(processor, split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    optimizer = AdamW([
        {"params": encoder.learnable_queries,        "lr": 1e-6},
        {"params": encoder.projection.parameters(),  "lr": 1e-6},
        {"params": encoder.ocr_encoder.parameters(), "lr": 1e-7},
    ], weight_decay=0.01)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=140_000,
    )

    scaler = torch.amp.GradScaler()


    for p in encoder.ocr_encoder.parameters():
        p.requires_grad = False

    step = 0
    docvlm.train()

    for batch in tqdm(dataloader, desc=f"Stage 1", total=140_000):
        if step == 10_000:
            print("Unfreezing OCR encoder")
            for p in encoder.ocr_encoder.parameters():
                p.requires_grad = True

        batch = {k: v.cuda() for k,v in batch.items()}

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = docvlm(
                ocr_input_ids=batch["ocr_input_ids"],
                ocr_attention_mask=batch["ocr_attention_mask"],
                bbox=batch["bbox"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=None
            )

            loss = outputs.loss

            if torch.isnan(loss):
                print("NaN loss detected, skipping step")
                optimizer.zero_grad()
                step += 1
                continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        tqdm.write(f"Step {step} | Loss: {loss.item():.4f}")

        if step % 5_000 == 0 and step > 0:
            torch.save({
                "step": step,
                "learnable_queries": encoder.learnable_queries,
                "projection": encoder.projection.state_dict(),
                "ocr_encoder": encoder.ocr_encoder.state_dict(),
            }, f"checkpoints/stage1_step{step}.pt")

        step += 1
        if step >= 140_000:
            break

    # Save final stage 1
    torch.save({
        "step": step,
        "learnable_queries": encoder.learnable_queries,
        "projection": encoder.projection.state_dict(),
        "ocr_encoder": encoder.ocr_encoder.state_dict(),
    }, "checkpoints/stage1_final.pt")
    print("Stage 1 done")


if __name__ == "__main__":
    import os
    os.makedirs("checkpoints", exist_ok=True)
    train()