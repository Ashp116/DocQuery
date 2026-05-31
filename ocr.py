import easyocr
import torch
from PIL import Image
from transformers import AutoTokenizer

reader = easyocr.Reader(['en'], gpu=True)
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
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        is_split_into_words=True,
    )

    return encoding