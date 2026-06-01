import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from encoder import DocVLMEncoder

class DocVLM(nn.Module):
    def __init__(self, qwen_model, encoder):
        super().__init__()
        self.qwen = qwen_model
        self.encoder = encoder


    def forward(self, ocr_input_ids, ocr_attention_mask, bbox, input_ids, attention_mask, labels=None, pixel_values=None, image_grid_thw=None):
        ocr_tokens = self.encoder(
            ocr_input_ids,
            ocr_attention_mask,
            bbox,
        ).to(torch.float16)

        with torch.no_grad():
            text_embeds = self.qwen.model.embed_tokens(input_ids)

        text_embeds = text_embeds.detach()
        combined_embeds = torch.cat([ocr_tokens, text_embeds], dim=1)

        B = input_ids.shape[0]
        ocr_mask = torch.ones(B, 64, device=attention_mask.device)
        combined_mask = torch.cat([ocr_mask, attention_mask], dim=1)

        if labels is not None:
            ignore = torch.full((B, 64), -100, device=labels.device)
            labels = torch.cat([ignore, labels], dim=1)

            text_len = input_ids.shape[1]
            label_len = labels.shape[1] - 64
            if label_len < text_len:
                pad = torch.full(
                    (B, text_len - label_len),
                    -100,
                    device=labels.device
                )
                labels = torch.cat([labels, pad], dim=1)
            elif label_len > text_len:
                labels = labels[:, :64 + text_len]

        outputs = self.qwen(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels
        )

        return outputs
    

    @torch.no_grad()
    def generate(self, ocr_input_ids, ocr_attention_mask, bbox, input_ids, attention_mask, **generate_kwargs):
        ocr_tokens = self.encoder(
            ocr_input_ids, ocr_attention_mask, bbox
        ).to(torch.bfloat16)

        text_embeds = self.qwen.model.embed_tokens(input_ids).to(torch.bfloat16)
        combined_embeds = torch.cat([ocr_tokens, text_embeds], dim=1)

        B = input_ids.shape[0]
        ocr_mask = torch.ones(B, 64, device=attention_mask.device)
        combined_masks = torch.cat([ocr_mask, attention_mask], dim=1)

        return self.qwen.generate(
            input_embeds=combined_embeds,
            attention_mask=combined_masks,
            **generate_kwargs
        )