import os
import pickle
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from ocr import extract_ocr

def collate_fn(batch):
    ocr_input_ids      = torch.stack([b["ocr_input_ids"] for b in batch])
    ocr_attention_mask = torch.stack([b["ocr_attention_mask"] for b in batch])
    bbox               = torch.stack([b["bbox"] for b in batch])

    # Pad input_ids to same length within batch
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True,
        padding_value=-100  # -100 is ignored by cross entropy loss
    )

    return {
        "ocr_input_ids":      ocr_input_ids,
        "ocr_attention_mask": ocr_attention_mask,
        "bbox":               bbox,
        "input_ids":          input_ids,
        "attention_mask":     attention_mask,
        "labels":             labels,
    }


def precache_ocr(dataset, num_samples=None):
    n = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    for idx in tqdm(range(n), desc="Pre-caching OCR"):
        cache_path = os.path.join(dataset.cache_dir, f"{idx}.pkl")
        if os.path.exists(cache_path):
            continue
        try:
            sample = dataset.data[idx]
            tmp = f"/tmp/docvlm_{idx}.jpg"
            sample["image"].save(tmp)
            ocr = extract_ocr(tmp, max_length=dataset.max_ocr_length)
            with open(cache_path, "wb") as f:
                pickle.dump(ocr, f)
        except Exception as e:
            print(f"Warning: OCR failed for sample {idx}: {e}")


class DocVQADataset(Dataset):
    def __init__(self, processor, split="train", max_ocr_length=512):
        self.data = load_dataset(
            "HuggingFaceM4/DocumentVQA",
            split=split
        )
        self.processor = processor
        self.max_ocr_length = max_ocr_length
        self.cache_dir = f"/tmp/docvqa_ocr_cache_{split}"
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image: Image.Image = sample["image"]
        question: str = sample["question"]
        answer: str = sample["answers"][0]

        cache_path = os.path.join(self.cache_dir, f"{idx}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                ocr = pickle.load(f)
            ocr["bbox"] = ocr["bbox"].clamp(0, 1000)
        else:
            tmp = f"/tmp/docvlm_{idx}.jpg"
            image.save(tmp)
            ocr = extract_ocr(tmp, max_length=self.max_ocr_length)
            with open(cache_path, "wb") as f:
                pickle.dump(ocr, f)

        prompt = f"Question: {question}\nAnswer:"
        text_enc = self.processor(
            text=prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,   # no padding here, collate_fn handles it
        )

        label_enc = self.processor(
            text=answer,
            return_tensors="pt",
            truncation=True,
            max_length=32,
            padding=False,
        )

        return {
            "ocr_input_ids": ocr["input_ids"].squeeze(0),
            "ocr_attention_mask": ocr["attention_mask"].squeeze(0),
            "bbox": ocr["bbox"].squeeze(0),
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "labels": label_enc["input_ids"].squeeze(0)
        }
