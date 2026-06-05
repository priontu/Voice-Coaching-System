from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from OnsetOffsetDetection_v1.preprocessing import (
    OnsetOffsetLabelConfig,
    active_note_target,
    audio_lengths_to_frame_lengths,
    event_target,
    lengths_to_mask,
    pad_audio_batch,
    pad_label_dicts,
)


@dataclass(frozen=True)
class CQTFeatureConfig:
    sample_rate: int = 16000
    n_bins: int = 120           # 5 octaves × 24 bins/octave
    bins_per_octave: int = 24
    fmin: float = 65.41         # C2
    hop_length: int = 160       # 10 ms frames at 16 kHz
    top_db: float = 80.0

    @property
    def frame_seconds(self) -> float:
        return self.hop_length / self.sample_rate


class CQTBatchProcessor(torch.nn.Module):
    """Batch processor that extracts CQT features instead of log-mel."""

    def __init__(
        self,
        feature_config: CQTFeatureConfig = CQTFeatureConfig(),
        label_config: OnsetOffsetLabelConfig = OnsetOffsetLabelConfig(),
        device: str | torch.device = "cpu",
        use_amp_features: bool = False,
        max_audio_samples: int | None = None,
    ):
        super().__init__()
        try:
            from nnAudio.features import CQT1992v2
        except ImportError as exc:
            raise RuntimeError(
                "nnAudio is required for OnsetOffsetDetection_v2. "
                "Install with: pip install nnAudio"
            ) from exc

        self.feature_config = feature_config
        self.label_config = label_config
        self.device = torch.device(device)
        self.use_amp_features = use_amp_features
        self.max_audio_samples = max_audio_samples

        self.cqt = CQT1992v2(
            sr=feature_config.sample_rate,
            hop_length=feature_config.hop_length,
            fmin=feature_config.fmin,
            n_bins=feature_config.n_bins,
            bins_per_octave=feature_config.bins_per_octave,
            output_format="Magnitude",
            center=True,
            verbose=False,
            pad_mode="reflect",
        ).to(self.device)

    def forward(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        audio_items = [
            self._resample(audio.to(self.device, non_blocking=True), sr)
            for audio, sr in zip(raw_batch["audio_items"], raw_batch["audio_sample_rates"], strict=True)
        ]
        if self.max_audio_samples is not None:
            audio_items = [a[: self.max_audio_samples] for a in audio_items]

        audio_batch, audio_lengths = pad_audio_batch(audio_items)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp_features and self.device.type == "cuda"):
            # nnAudio returns (B, n_bins, T) → transpose to (B, T, n_bins)
            mag = self.cqt(audio_batch).transpose(1, 2).contiguous()

        # Convert magnitude to dB scale
        features = mag.float()
        features = 20.0 * torch.log10(features.clamp(min=1e-7))
        ref = features.amax(dim=(1, 2), keepdim=True)
        features = features.clamp(min=ref - self.feature_config.top_db)

        max_frames = features.shape[1]
        frame_lengths = audio_lengths_to_frame_lengths(audio_lengths, self.feature_config).clamp(max=max_frames)
        attention_mask = lengths_to_mask(frame_lengths, max_frames)
        features = _cmvn(features, attention_mask)

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
            "input_features": features,
            "attention_mask": attention_mask,
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


def _cmvn(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (features * m).sum(dim=1, keepdim=True) / n
    var = ((features - mean).square() * m).sum(dim=1, keepdim=True) / n
    return (features - mean) / (var + 1e-5).sqrt()
