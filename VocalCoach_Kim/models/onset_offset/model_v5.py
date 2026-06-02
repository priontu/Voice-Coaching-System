"""
models/onset_offset/model_v5.py - Wav2Vec2-based onset/offset detector.

Wav2Vec2 backbone (facebook/wav2vec2-base) with a shared 3-class linear head
predicting onset, offset, and active (voiced) probability per frame.

Input:  raw 16 kHz waveform  [B, T_samples]
Output: dict with onset_logits, offset_logits, active_logits  [B, T_frames]
Frame rate: ~50 Hz (20 ms / frame) — fixed CNN stride of 320 samples @ 16 kHz.

Copied from archive/OnsetOffsetDetection_v5/model.py to avoid archive path dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Wav2Vec2ModelConfig:
    pretrained_model_name: str = "facebook/wav2vec2-base"
    freeze_feature_extractor: bool = True
    freeze_transformer: bool = False
    dropout: float = 0.1


class Wav2Vec2OnsetDetector(nn.Module):
    """Wav2Vec2 backbone with onset/offset/active heads.

    The pretrained CNN feature extractor is frozen by default; the transformer
    layers are fine-tuned. Raw 16 kHz waveform goes in, per-frame logits come
    out at ~50 Hz (20 ms / frame).
    """

    def __init__(self, config: Wav2Vec2ModelConfig = Wav2Vec2ModelConfig()):
        super().__init__()
        try:
            from transformers import Wav2Vec2Model
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for the v5 onset/offset model. "
                "Install with: pip install transformers"
            ) from exc

        self.config_obj = config
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            config.pretrained_model_name, use_safetensors=True
        )

        if config.freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        if config.freeze_transformer:
            for param in self.wav2vec2.encoder.parameters():
                param.requires_grad_(False)

        hidden_size = self.wav2vec2.config.hidden_size  # 768 for wav2vec2-base
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(hidden_size, 3)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # input_values: (B, T_samples) raw waveform
        # attention_mask: (B, T_samples) sample-level, 1 = valid
        out = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        x = self.dropout(out.last_hidden_state)  # (B, T_frames, 768)
        logits = self.head(x)
        return {
            "onset_logits": logits[..., 0],
            "offset_logits": logits[..., 1],
            "active_logits": logits[..., 2],
        }
