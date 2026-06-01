from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from OnsetOffsetDetection_v1.preprocessing import (
    OnsetOffsetLabelConfig,
    active_note_target,
    event_target,
    lengths_to_mask,
    pad_label_dicts,
)


# Wav2Vec2-base feature-extractor convolution config (kernel_size, stride) per layer.
_WAV2VEC2_BASE_CONV = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]


def _wav2vec2_frame_lengths(sample_lengths: torch.Tensor) -> torch.Tensor:
    lengths = sample_lengths.clone().long()
    for kernel_size, stride in _WAV2VEC2_BASE_CONV:
        lengths = (lengths - kernel_size) // stride + 1
    return lengths.clamp(min=0)


@dataclass(frozen=True)
class Wav2Vec2FeatureConfig:
    sample_rate: int = 16000
    # Wav2Vec2-base total stride: 5×2^6 = 320 samples → 20 ms per frame
    frame_seconds: float = 320 / 16000


class Wav2Vec2BatchProcessor(torch.nn.Module):
    """Passes raw waveform to Wav2Vec2 rather than extracting mel features.

    batch["input_features"] = (B, T_samples) raw waveform
    batch["attention_mask"] = (B, T_samples) sample-level mask (for Wav2Vec2 input)
    batch["frame_attention_mask"] = (B, T_frames) frame-level mask (for loss)
    batch["frame_lengths"] = (B,) number of valid Wav2Vec2 output frames
    """

    def __init__(
        self,
        feature_config: Wav2Vec2FeatureConfig = Wav2Vec2FeatureConfig(),
        label_config: OnsetOffsetLabelConfig = OnsetOffsetLabelConfig(),
        device: str | torch.device = "cpu",
        use_amp_features: bool = False,
        max_audio_samples: int | None = None,
    ):
        super().__init__()
        self.feature_config = feature_config
        self.label_config = label_config
        self.device = torch.device(device)
        self.max_audio_samples = max_audio_samples

    def forward(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        audio_items = [
            self._resample(audio.to(self.device, non_blocking=True), sr)
            for audio, sr in zip(raw_batch["audio_items"], raw_batch["audio_sample_rates"], strict=True)
        ]
        if self.max_audio_samples is not None:
            audio_items = [a[: self.max_audio_samples] for a in audio_items]

        sample_lengths = torch.tensor(
            [a.numel() for a in audio_items], dtype=torch.long, device=self.device
        )
        max_len = int(sample_lengths.max().item())
        waveforms = torch.zeros((len(audio_items), max_len), dtype=torch.float32, device=self.device)
        for i, audio in enumerate(audio_items):
            waveforms[i, : audio.numel()] = audio

        # Sample-level mask for Wav2Vec2 input
        positions = torch.arange(max_len, device=self.device)
        attention_mask = (positions.unsqueeze(0) < sample_lengths.unsqueeze(1)).long()

        # Frame-level mask for loss computation
        frame_lengths = _wav2vec2_frame_lengths(sample_lengths)
        max_frames = int(frame_lengths.max().item())
        frame_attention_mask = lengths_to_mask(frame_lengths, max_frames)

        frame_times = (
            torch.arange(max_frames, dtype=torch.float32, device=self.device)
            * self.feature_config.frame_seconds
        )
        labels = [
            self._labels_for_example(example, frame_times, int(frame_lengths[i].item()))
            for i, example in enumerate(raw_batch["examples"])
        ]
        targets = pad_label_dicts(labels, max_frames)
        return {
            "input_features": waveforms,
            "attention_mask": attention_mask,
            "frame_attention_mask": frame_attention_mask,
            "frame_lengths": frame_lengths,
            "labels": targets,
            "metadata": raw_batch["metadata"],
        }

    def _labels_for_example(self, example: Any, frame_times: torch.Tensor, num_frames: int) -> dict[str, torch.Tensor]:
        starts = torch.tensor([n.start_sec for n in example.notes], dtype=torch.float32, device=self.device)
        ends = torch.tensor([n.end_sec for n in example.notes], dtype=torch.float32, device=self.device)
        valid_times = frame_times[:num_frames]
        return {
            "onset": event_target(valid_times, starts, self.label_config.onset_sigma_sec),
            "offset": event_target(valid_times, ends, self.label_config.offset_sigma_sec),
            "active": active_note_target(valid_times, starts, ends),
        }

    def _resample(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.feature_config.sample_rate:
            return audio.float().contiguous()
        import torchaudio.functional as F_audio
        return F_audio.resample(audio.float(), sample_rate, self.feature_config.sample_rate).contiguous()
