import torch
import torch.nn as nn
from transformers import AutoModel

class DocVLMEncoder(nn.Module):
    def __init__(
        self,
        num_queries: int = 64,
        ocr_model_name: str = "microsoft/layoutlmv3-base",
        qwen_hidden_dim: int = 3584,
    ):
        super().__init__()

        self.ocr_encoder = AutoModel.from_pretrained(ocr_model_name)
        ocr_hidden_dim = self.ocr_encoder.config.hidden_size  # 768

        self.learnable_queries = nn.Parameter(
            torch.randn(num_queries, ocr_hidden_dim) * 0.002
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=ocr_hidden_dim,
            num_heads=8,
            batch_first=True,
        )

        self.projection = nn.Sequential(
            nn.LayerNorm(ocr_hidden_dim),
            nn.Linear(ocr_hidden_dim, qwen_hidden_dim),
            nn.GELU(),
            nn.Linear(qwen_hidden_dim, qwen_hidden_dim),
        )

        self.num_queries = num_queries

    def forward(self, input_ids, attention_mask, bbox):
        B = input_ids.shape[0]

        ocr_out = self.ocr_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
        ).last_hidden_state

        queries = self.learnable_queries.unsqueeze(0).expand(B, -1, -1)

        key_padding_mask = (attention_mask == 0)  
        compressed, _ = self.cross_attn(
            query=queries,
            key=ocr_out,
            value=ocr_out,
            key_padding_mask=key_padding_mask,
        )

        projected = self.projection(compressed)

        return projected