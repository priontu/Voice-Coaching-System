"""
models/onset_offset/model.py - Note onset/offset detection model.

Architecture: CNN feature extractor → BiLSTM temporal encoder → two MLP heads.

Input:  log-mel spectrogram  [B, 1, n_mels, T]
Output: onset_logits         [B, T]
        offset_logits        [B, T]

Unchanged from the original Note Model/model.py.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool with Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        padding: Tuple[int, int] = (1, 1),
        pool_size: Tuple[int, int] = (2, 1),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


class MLPHead(nn.Module):
    """Two-layer MLP that collapses the feature dim to a scalar per frame."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_size] → [B, T]"""
        return self.net(x).squeeze(-1)


class OnsetOffsetModel(nn.Module):
    """
    CNN + BiLSTM model for singing note onset and offset detection.

    The CNN reduces the mel-frequency axis while preserving the time axis.
    The BiLSTM encodes temporal context. Two MLP heads predict onset and
    offset frame probabilities (returned as raw logits).

    Args:
        n_mels:           Number of mel bins.
        cnn_channels:     Output channels for each ConvBlock.
        lstm_hidden_size: Hidden size per direction in BiLSTM.
        lstm_num_layers:  Number of stacked BiLSTM layers.
        lstm_dropout:     Dropout between LSTM layers.
        head_hidden_size: Hidden size of each MLP head.
        dropout:          Dropout rate for ConvBlocks and MLP heads.
    """

    def __init__(
        self,
        n_mels: int = 80,
        cnn_channels: List[int] | None = None,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        head_hidden_size: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 128, 128]

        blocks: List[nn.Module] = []
        in_ch = 1
        for out_ch in cnn_channels:
            blocks.append(ConvBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        n_freq_out = n_mels // (2 ** len(cnn_channels))
        lstm_input_size = cnn_channels[-1] * n_freq_out

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )
        self.lstm_drop = nn.Dropout(p=dropout)

        lstm_out = lstm_hidden_size * 2  # bidirectional

        self.onset_head = MLPHead(lstm_out, head_hidden_size, dropout)
        self.offset_head = MLPHead(lstm_out, head_hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 1, n_mels, T]

        Returns:
            onset_logits:  [B, T]
            offset_logits: [B, T]
        """
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.view(B, C * F, T).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_drop(x)
        return self.onset_head(x), self.offset_head(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Like forward() but returns sigmoid probabilities with no gradients."""
        onset_logits, offset_logits = self.forward(x)
        return torch.sigmoid(onset_logits), torch.sigmoid(offset_logits)
