import torch
import torch.nn as nn
from transformers import AutoModel


class DocVLMEncoder(nn.Module):
    def __init__(self, num_queries: int = 64, ocr_model_name: str = "microsoft/layoutmv3-base", qwen_hidden_dim: int = 3584):
        super().__init__()

        self.ocr_reader = AutoModel.from_pretrained(ocr_model_name)
        ocr_hidden_dim = self.ocr_reader.config.hidden_size


        self.learnable_queries = nn.Parameter(
            torch.randn(num_queries, ocr_hidden_dim) * 0.02
        )

        self.projection = nn.Sequential(
            nn.LayerNorm(ocr_hidden_dim),
            nn.Linear(ocr_hidden_dim, qwen_hidden_dim),
            nn.GELU(),
            nn.Linear(qwen_hidden_dim, ocr_hidden_dim)
        )

        self.num_queries = num_queries


    def forward(self, input_ids, attention_mask, bbox):
        B = input_ids.shape[0]

        ocr_out = self.ocr_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox
        ).last_hidden_state


        queries = self.learnable_queries.unsqueeze(0).expand(B, -1, -1)

        combined = torch.cat([ocr_out, queries], dim=1)

        query_mask = torch.ones(B, self.num_queries, device=input_ids.device)
        extended_mask = torch.cat([attention_mask, query_mask], dim=1)

        compressed = self.ocr_reader(
            inputs_embeds=combined,
            attention_mask=extended_mask,
        ).last_hidden_state

        compressed_queries = compressed[:, -self.num_queries, :]

        projected = self.projection(compressed_queries)
        
        return projected