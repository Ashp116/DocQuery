import torch
import torch.nn as nn

class DocVLM(nn.Module):
    def __init__(self, qwen_model, encoder):
        super().__init__()
        self.qwen = qwen_model
        self.encoder = encoder

    def _merge_image_embeds(self, input_ids, text_embeds, pixel_values, image_grid_thw):
        """Process pixel_values through Qwen's visual encoder and merge into text_embeds."""
        visual_dtype = next(self.qwen.visual.parameters()).dtype
        image_embeds = self.qwen.visual(pixel_values.to(dtype=visual_dtype), grid_thw=image_grid_thw)
        image_mask = (input_ids == self.qwen.config.image_token_id)
        text_embeds = text_embeds.clone()
        text_embeds[image_mask] = image_embeds.to(text_embeds.dtype)
        return text_embeds

    def forward(self, ocr_input_ids, ocr_attention_mask, bbox, input_ids, attention_mask, labels=None, pixel_values=None, image_grid_thw=None):
        ocr_tokens = self.encoder(ocr_input_ids, ocr_attention_mask, bbox)

        with torch.no_grad():
            text_embeds = self.qwen.model.embed_tokens(input_ids)
            if pixel_values is not None:
                text_embeds = self._merge_image_embeds(input_ids, text_embeds, pixel_values, image_grid_thw)

        # Align dtypes before concatenation — Qwen embeds are fp16, encoder output
        # follows the active autocast dtype (bfloat16 in stage 1, fp16 in stage 2).
        text_embeds = text_embeds.detach().to(ocr_tokens.dtype)
        combined_embeds = torch.cat([ocr_tokens, text_embeds], dim=1)

        B = input_ids.shape[0]
        num_q = self.encoder.num_queries
        ocr_mask = torch.ones(B, num_q, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([ocr_mask, attention_mask], dim=1)

        if labels is not None:
            ignore = torch.full((B, num_q), -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([ignore, labels], dim=1)

            text_len = input_ids.shape[1]
            label_len = labels.shape[1] - num_q
            if label_len < text_len:
                pad = torch.full((B, text_len - label_len), -100, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([labels, pad], dim=1)
            elif label_len > text_len:
                labels = labels[:, :num_q + text_len]

        # pixel_values already merged into inputs_embeds — do not pass again.
        outputs = self.qwen(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def generate(self, ocr_input_ids, ocr_attention_mask, bbox, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, **generate_kwargs):
        ocr_tokens = self.encoder(ocr_input_ids, ocr_attention_mask, bbox)

        text_embeds = self.qwen.model.embed_tokens(input_ids)
        if pixel_values is not None:
            text_embeds = self._merge_image_embeds(input_ids, text_embeds, pixel_values, image_grid_thw)

        text_embeds = text_embeds.to(ocr_tokens.dtype)
        combined_embeds = torch.cat([ocr_tokens, text_embeds], dim=1)

        B = input_ids.shape[0]
        num_q = self.encoder.num_queries
        ocr_mask = torch.ones(B, num_q, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_masks = torch.cat([ocr_mask, attention_mask], dim=1)

        return self.qwen.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_masks,
            **generate_kwargs,
        )
