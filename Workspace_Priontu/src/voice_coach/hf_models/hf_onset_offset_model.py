from __future__ import annotations

import torch


class HFOnsetOffsetModel(torch.nn.Module):
    """Small, HF-friendly onset/offset model skeleton.

    This minimal model uses a 1D convolutional encoder and two simple
    linear heads for onset and offset logits. It's intentionally small
    and serves as a starting point for training.
    """

    def __init__(self, input_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.onset_head = torch.nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.offset_head = torch.nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward expects `x` shape (B, C, T) and returns logits with shape (B, T)."""
        h = self.encoder(x)
        onset = self.onset_head(h).squeeze(1)
        offset = self.offset_head(h).squeeze(1)
        return {"onset_logits": onset, "offset_logits": offset}
