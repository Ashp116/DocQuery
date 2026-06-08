import easyocr
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

ocr_tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

_MAX_OCR_SIDE = 1024  # resize before OCR; saves ~50% detection time on large doc images
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=True)
    return _reader


def extract_ocr(image_or_path, max_length: int = 512):
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path).convert("RGB")
    else:
        img = image_or_path.convert("RGB")

    W, H = img.size
    if max(W, H) > _MAX_OCR_SIDE:
        scale = _MAX_OCR_SIDE / max(W, H)
        img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        W, H = img.size

    # Pass numpy array so EasyOCR skips its own disk read.
    # batch_size=4 speeds up the recognition phase on batched crops.
    results = _get_reader().readtext(np.array(img), batch_size=4)

    words = []
    boxes = []

    for (bbox_pts, text, conf) in results:
        if conf < 0.3 or not text.strip():
            continue

        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)

        norm_box = [
            max(0, min(1000, int(x0 / W * 1000))),
            max(0, min(1000, int(y0 / H * 1000))),
            max(0, min(1000, int(x1 / W * 1000))),
            max(0, min(1000, int(y1 / H * 1000))),
        ]
        words.append(text)
        boxes.append(norm_box)

    if not words:
        words = ["[PAD]"]
        boxes = [[0, 0, 0, 0]]

    encoding = ocr_tokenizer(
        words,
        boxes=boxes,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    bbox = encoding["bbox"]
    if bbox.dim() == 3:
        bbox = bbox.squeeze(0)

    assert bbox.shape == (max_length, 4), f"Unexpected bbox shape: {bbox.shape}"
    encoding["bbox"] = bbox

    return encoding
