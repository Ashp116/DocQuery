import sys
import json
import torch
import torch.nn as nn
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from encoder import DocVLMEncoder
from model import DocVLM
from ocr import extract_ocr

SYSTEM_PROMPT = (
    "You are an invoice parsing assistant. "
    "Extract all invoice information and return it as a single valid JSON object "
    "with exactly these fields: vendor_name, invoice_number, invoice_date, due_date, "
    "line_items (array of objects each with description, quantity, unit_price, total), "
    "subtotal, tax, total. "
    "Use null for any field not present in the document. "
    "Return ONLY the JSON object — no explanation, no markdown, no other text."
)


def load_model():
    print("Loading Qwen2-VL...")
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    qwen.eval()
    for param in qwen.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    print("Loading encoder from checkpoints/stage2_final.pt...")
    encoder = DocVLMEncoder(
        num_queries=64,
        qwen_hidden_dim=qwen.config.hidden_size,
    ).cuda()
    encoder.eval()

    ckpt = torch.load("checkpoints/stage2_final.pt", map_location="cuda")
    encoder.learnable_queries = nn.Parameter(ckpt["learnable_queries"].detach().clone())
    encoder.projection.load_state_dict(ckpt["projection"])
    encoder.ocr_encoder.load_state_dict(ckpt["ocr_encoder"])

    docvlm = DocVLM(qwen, encoder)
    docvlm.eval()
    return docvlm, processor


def main():
    if len(sys.argv) < 2:
        print("Usage: python infer.py <invoice_image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    docvlm, processor = load_model()

    # --- OCR ---
    print("Running OCR...")
    ocr = extract_ocr(image_path, max_length=512)
    # extract_ocr returns input_ids/attention_mask as (1, 256), bbox as (256, 4)
    ocr_input_ids      = ocr["input_ids"].cuda()                       # (1, 256)
    ocr_attention_mask = ocr["attention_mask"].cuda()                  # (1, 256)
    bbox               = ocr["bbox"].clamp(0, 1000).unsqueeze(0).cuda()  # (1, 256, 4)

    # --- Vision ---
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all invoice fields from this document as JSON."},
            ],
        },
    ]
    image_inputs, _ = process_vision_info(messages)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_enc = processor(
        text=[prompt_text],
        images=image_inputs,
        return_tensors="pt",
    )
    input_ids      = text_enc["input_ids"].cuda()
    attention_mask = text_enc["attention_mask"].cuda()
    pixel_values   = text_enc["pixel_values"].cuda()
    image_grid_thw = text_enc["image_grid_thw"].cuda()

    # --- Generate ---
    print("Generating...")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        generated_ids = docvlm.generate(
            ocr_input_ids=ocr_input_ids,
            ocr_attention_mask=ocr_attention_mask,
            bbox=bbox,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    output_text = processor.decode(generated_ids[0], skip_special_tokens=True)

    # --- Parse JSON ---
    try:
        start = output_text.find('{')
        end   = output_text.rfind('}') + 1
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in output")
        data = json.loads(output_text[start:end])
        print(json.dumps(data, indent=2))
    except (json.JSONDecodeError, ValueError):
        print(output_text)


if __name__ == "__main__":
    main()
