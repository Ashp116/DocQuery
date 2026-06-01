import easyocr
import torch
from PIL import Image
from transformers import AutoTokenizer

reader = easyocr.Reader(['en'], gpu=False)
ocr_tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

def extract_ocr(image_path: str, max_length: int = 512):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    results = reader.readtext(image_path)

    words = []
    boxes = []

    for (bbox_pts, text, conf) in results:
        if conf < 0.3 or text.strip():
            continue

        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), min(ys)


        norm_box = [
            int(x0 / W * 1000),
            int(y0 / H * 1000),
            int(x1 / W * 1000),
            int(y1 / H * 1000),
        ]
        words.append(text)
        boxes.append(norm_box)

    if not words:
        words = ["[PAD]"]
        boxes = [[0,0,0,0]]

    encoding = ocr_tokenizer(
        words,
        boxes=boxes,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

     # Force bbox to always be (1, seq_len, 4)
    bbox = encoding["bbox"]
    if bbox.dim() == 2:
        pass  # already (seq_len, 4) — correct
    elif bbox.dim() == 3:
        bbox = bbox.squeeze(0)  # remove extra dim if (1, seq_len, 4)

    # Verify shape is exactly (max_length, 4)
    assert bbox.shape == (256, 4), f"Unexpected bbox shape: {bbox.shape}"
    encoding["bbox"] = bbox

    return encoding