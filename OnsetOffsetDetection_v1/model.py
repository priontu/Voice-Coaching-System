from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OnsetOffsetModelConfig:
    n_mels: int = 80
    model_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 256
    conv_kernel_size: int = 31
    dropout: float = 0.1


class OnsetOffsetConformer(nn.Module):
    """Compact Conformer-style encoder with onset/offset/active heads."""

    def __init__(self, config: OnsetOffsetModelConfig = OnsetOffsetModelConfig()):
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.n_mels, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.Dropout(config.dropout),
        )
        self.layers = nn.ModuleList(
            [
                ConformerStyleBlock(
                    dim=config.model_dim,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    conv_kernel_size=config.conv_kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
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


class ConformerStyleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_dim, dropout)
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(dim, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(dim, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        attn_input = self.self_attn_norm(x)
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        attn_output, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.self_attn_dropout(attn_output)
        x = x + self.conv(x, attention_mask)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd for same-length padding")
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.post_conv_norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.pointwise_in(y)
        y = torch.nn.functional.glu(y, dim=1)
        y = self.depthwise(y)
        y = y.transpose(1, 2)
        y = self.post_conv_norm(y)
        y = y.transpose(1, 2)
        y = self.activation(y)
        y = self.pointwise_out(y)
        y = y.transpose(1, 2)
        y = self.dropout(y)
        if attention_mask is not None:
            y = y * attention_mask.unsqueeze(-1).to(y.dtype)
        return y
