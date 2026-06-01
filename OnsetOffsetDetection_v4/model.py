from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MambaModelConfig:
    n_mels: int = 80
    model_dim: int = 128
    num_layers: int = 4
    d_state: int = 16       # SSM state size
    d_conv: int = 4         # local convolution width inside Mamba
    expand: int = 2         # inner-dimension expansion factor
    dropout: float = 0.1


class BidirectionalMambaBlock(nn.Module):
    """One Mamba layer run in both directions; outputs are concatenated then projected back."""

    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise RuntimeError(
                "mamba-ssm is required for OnsetOffsetDetection_v4. "
                "Install with: pip install mamba-ssm causal-conv1d"
            ) from exc

        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.merge = nn.Linear(dim * 2, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.norm(x)
        fwd = self.mamba_fwd(y)
        bwd = self.mamba_bwd(y.flip(1)).flip(1)
        y = self.merge(torch.cat([fwd, bwd], dim=-1))
        if attention_mask is not None:
            y = y * attention_mask.unsqueeze(-1).to(y.dtype)
        return x + self.dropout(y)


class OnsetOffsetMamba(nn.Module):
    """Bidirectional Mamba backbone for onset/offset detection.

    Replaces Conformer attention + convolution with selective state-space layers,
    giving linear sequence complexity — no 30-second cap needed.
    Requires CUDA and mamba-ssm.
    """

    def __init__(self, config: MambaModelConfig = MambaModelConfig()):
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.n_mels, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.Dropout(config.dropout),
        )
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(
                dim=config.model_dim,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
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
