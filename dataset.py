import os
import torch
from torch.utils.data import Dataset
from dataset import load_dataset
from PIL import Image
from .ocr import extract_ocr

class DocVQADataset(Dataset):
    def __init__(self, processor, split="train", max_ocr_length=512):
        self.data = load_dataset(
            "HuggingFaceM4/DocumentVQA",
            split=split
        )
        self.processor = processor
        self.max_ocr_length = max_ocr_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image: Image.Image = sample["image"]
        question: str = sample["question"]
        answer: str = sample["answers"][0]

        tmp = f"/tmp/docvlm_{idx}.jpg"
        image.save(tmp)

        ocr = extract_ocr(tmp, max_length=self.max_ocr_length)

        prompt = f"Question: {question}\nAnswer:"
        text_enc = self.processor(
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True
        )

        label_enc = self.processor(
            text=answer,
            return_tensors="pt",
            max_length=64,
            truncation=True
        )

        return {
            "ocr_input_ids": ocr["input_ids"].squeeze(0),
            "ocr_attention_mask": ocr["attention_mask"].squeeze(0),
            "bbox": ocr["bbox"].squeeze(0),
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "labels": label_enc["input_ids"].squeeze(0)
        }