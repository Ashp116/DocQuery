import torch
import torch.nn as nn
import warnings
import os
import gc
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import logging as hf_logging
from tqdm import tqdm

from encoder import DocVLMEncoder
from model import DocVLM
from train_dataset import DocVQADatasetV2, collate_fn_v2, precache_ocr

warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()


def load_qwen():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    for param in model.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    return model, processor


def train_stage2(ocr_cache_dir=None):
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("checkpoints", exist_ok=True)

    qwen, processor = load_qwen()

    encoder = DocVLMEncoder(
        num_queries=64,
        qwen_hidden_dim=qwen.config.hidden_size,
    ).cuda()

    ckpt = torch.load("checkpoints/stage1_final.pt", map_location="cuda")
    encoder.learnable_queries = nn.Parameter(ckpt["learnable_queries"].detach().clone())
    encoder.projection.load_state_dict(ckpt["projection"])
    encoder.ocr_encoder.load_state_dict(ckpt["ocr_encoder"])
    del ckpt
    gc.collect()
    print("Loaded Stage 1 checkpoint")

    docvlm = DocVLM(qwen, encoder)

    dataset = DocVQADatasetV2(processor, split="train", ocr_cache_dir=ocr_cache_dir)
    precache_ocr(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn_v2,
    )

    optimizer = AdamW([
        {"params": encoder.learnable_queries,        "lr": 5e-6},
        {"params": encoder.projection.parameters(),  "lr": 5e-6},
        {"params": encoder.ocr_encoder.parameters(), "lr": 5e-6},
    ], weight_decay=0.01)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=100_000,
    )

    step = 0
    docvlm.train()
    qwen.eval()  # Qwen is frozen — eval keeps attention deterministic

    pbar = tqdm(dataloader, desc="Stage 2", total=100_000)
    for batch in pbar:
        batch = {k: v.cuda() for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = docvlm(
                ocr_input_ids=batch["ocr_input_ids"],
                ocr_attention_mask=batch["ocr_attention_mask"],
                bbox=batch["bbox"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
            )
            loss = outputs.loss

            if torch.isnan(loss):
                optimizer.zero_grad()
                step += 1
                continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", step=step)

        if step % 5_000 == 0 and step > 0:
            torch.save({
                "step": step,
                "learnable_queries": encoder.learnable_queries,
                "projection": encoder.projection.state_dict(),
                "ocr_encoder": encoder.ocr_encoder.state_dict(),
            }, f"checkpoints/stage2_step{step}.pt")

        step += 1
        if step >= 100_000:
            break

    torch.save({
        "step": step,
        "learnable_queries": encoder.learnable_queries,
        "projection": encoder.projection.state_dict(),
        "ocr_encoder": encoder.ocr_encoder.state_dict(),
    }, "checkpoints/stage2_final.pt")
    print("Stage 2 done")


if __name__ == "__main__":
    train_stage2()
