from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from OnsetOffsetDetection_v1.model import ConformerStyleBlock


@dataclass(frozen=True)
class CoupledModelConfig:
    n_mels: int = 80
    model_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 256
    conv_kernel_size: int = 31
    dropout: float = 0.1


class OnsetOffsetCoupledConformer(nn.Module):
    """Conformer encoder where the active-note head is conditioned on onset probabilities.

    Based on Onsets and Frames (Hawthorne et al., 2018): the frame detector receives
    onset predictions as additional context, letting it distinguish a sustained note
    from a fresh attack on the same pitch.
    """

    def __init__(self, config: CoupledModelConfig = CoupledModelConfig()):
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.n_mels, config.model_dim),
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

        self.onset_head = nn.Linear(config.model_dim, 1)
        self.offset_head = nn.Linear(config.model_dim, 1)

        # Active head sees backbone features + onset probability (the coupling)
        self.active_proj = nn.Sequential(
            nn.Linear(config.model_dim + 1, config.model_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.active_head = nn.Linear(config.model_dim, 1)

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.input_proj(features)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)

        onset_logits = self.onset_head(x).squeeze(-1)
        offset_logits = self.offset_head(x).squeeze(-1)

        # Detach onset so its gradient does not flow through the active loss path
        onset_prob = torch.sigmoid(onset_logits.detach())
        active_input = torch.cat([x, onset_prob.unsqueeze(-1)], dim=-1)
        active_logits = self.active_head(self.active_proj(active_input)).squeeze(-1)

        return {
            "onset_logits": onset_logits,
            "offset_logits": offset_logits,
            "active_logits": active_logits,
        }
