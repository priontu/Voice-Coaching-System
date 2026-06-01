from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from OnsetOffsetDetection_v1.model import ConformerStyleBlock


@dataclass(frozen=True)
class CQTModelConfig:
    n_bins: int = 120       # must match CQTFeatureConfig.n_bins
    model_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 256
    conv_kernel_size: int = 31
    dropout: float = 0.1


class OnsetOffsetConformerCQT(nn.Module):
    """Same Conformer encoder as v1 but takes CQT bins instead of mel bins as input."""

    def __init__(self, config: CQTModelConfig = CQTModelConfig()):
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.n_bins, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.Dropout(config.dropout),
        )
        self.layers = nn.ModuleList([
            ConformerStyleBlock(
                dim=config.model_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                conv_kernel_size=config.conv_kernel_size,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.head = nn.Linear(config.model_dim, 3)

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.input_proj(features)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        logits = self.head(x)
        return {
            "onset_logits": logits[..., 0],
            "offset_logits": logits[..., 1],
            "active_logits": logits[..., 2],
        }
