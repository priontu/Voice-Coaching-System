from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class OnsetOffsetFeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 160
    win_length: int = 1024
    fmin: float = 30.0
    fmax: float | None = 8000.0
    top_db: float = 80.0

    @property
    def frame_seconds(self) -> float:
        return self.hop_length / self.sample_rate


@dataclass(frozen=True)
class OnsetOffsetLabelConfig:
    onset_sigma_sec: float = 0.025
    offset_sigma_sec: float = 0.025


class OnsetOffsetBatchProcessor(torch.nn.Module):
    """Convert raw batches into log-mel features and onset/offset labels on device."""

    def __init__(
        self,
        feature_config: OnsetOffsetFeatureConfig = OnsetOffsetFeatureConfig(),
        label_config: OnsetOffsetLabelConfig = OnsetOffsetLabelConfig(),
        device: str | torch.device = "cpu",
        use_amp_features: bool = False,
        max_audio_samples: int | None = None,
    ):
        super().__init__()
        self.feature_config = feature_config
        self.label_config = label_config
        self.device = torch.device(device)
        self.use_amp_features = use_amp_features
        self.max_audio_samples = max_audio_samples
        self.feature_extractor = self._build_feature_extractor().to(self.device)

    def forward(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        audio_items = [
            self._resample(audio.to(self.device, non_blocking=True), sample_rate)
            for audio, sample_rate in zip(raw_batch["audio_items"], raw_batch["audio_sample_rates"], strict=True)
        ]
        if self.max_audio_samples is not None:
            audio_items = [a[: self.max_audio_samples] for a in audio_items]
        audio_batch, audio_lengths = pad_audio_batch(audio_items)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp_features and self.device.type == "cuda"):
            features = self.feature_extractor(audio_batch)

        max_frames = features.shape[1]
        frame_lengths = torch.clamp(audio_lengths_to_frame_lengths(audio_lengths, self.feature_config), max=max_frames)
        attention_mask = lengths_to_mask(frame_lengths, max_frames)
        features = _cmvn(features.float(), attention_mask)
        frame_times = (
            torch.arange(max_frames, dtype=torch.float32, device=self.device)
            * self.feature_config.frame_seconds
        )
        labels = [
            self._labels_for_example(example, frame_times, int(frame_lengths[index].item()))
            for index, example in enumerate(raw_batch["examples"])
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
        starts = torch.tensor([note.start_sec for note in example.notes], dtype=torch.float32, device=self.device)
        ends = torch.tensor([note.end_sec for note in example.notes], dtype=torch.float32, device=self.device)
        valid_times = frame_times[:num_frames]
        onset = event_target(valid_times, starts, self.label_config.onset_sigma_sec)
        offset = event_target(valid_times, ends, self.label_config.offset_sigma_sec)
        active = active_note_target(valid_times, starts, ends)
        return {"onset": onset, "offset": offset, "active": active}

    def _build_feature_extractor(self) -> torch.nn.Module:
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for OnsetOffsetDetection_v1") from exc

        return torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.feature_config.sample_rate,
                n_fft=self.feature_config.n_fft,
                win_length=self.feature_config.win_length,
                hop_length=self.feature_config.hop_length,
                f_min=self.feature_config.fmin,
                f_max=self.feature_config.fmax,
                n_mels=self.feature_config.n_mels,
                power=2.0,
                center=True,
            ),
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=self.feature_config.top_db),
            TransposeMel(),
        )

    def _resample(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.feature_config.sample_rate:
            return audio.float().contiguous()
        try:
            import torchaudio.functional as F_audio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for OnsetOffsetDetection_v1") from exc
        return F_audio.resample(audio.float(), sample_rate, self.feature_config.sample_rate).contiguous()


class TransposeMel(torch.nn.Module):
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return mel.transpose(1, 2).contiguous()


def event_target(frame_times: torch.Tensor, events: torch.Tensor, sigma_sec: float) -> torch.Tensor:
    if events.numel() == 0:
        return torch.zeros_like(frame_times)
    distances = frame_times[:, None] - events[None, :]
    return torch.exp(-(distances.square()) / (2.0 * sigma_sec**2)).amax(dim=1)


def active_note_target(frame_times: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    if starts.numel() == 0:
        return torch.zeros_like(frame_times)
    active = (frame_times[:, None] >= starts[None, :]) & (frame_times[:, None] < ends[None, :])
    return active.any(dim=1).float()


def pad_label_dicts(label_dicts: list[dict[str, torch.Tensor]], max_frames: int) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key in ("onset", "offset", "active"):
        values = [torch.nn.functional.pad(item[key], (0, max_frames - item[key].numel())) for item in label_dicts]
        output[key] = torch.stack(values, dim=0)
    return output


def pad_audio_batch(audio_items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([item.numel() for item in audio_items], dtype=torch.long, device=audio_items[0].device)
    max_len = int(lengths.max().item())
    batch = torch.zeros((len(audio_items), max_len), dtype=torch.float32, device=audio_items[0].device)
    for index, audio in enumerate(audio_items):
        batch[index, : audio.numel()] = audio.float()
    return batch, lengths


def audio_lengths_to_frame_lengths(lengths: torch.Tensor, config: OnsetOffsetFeatureConfig) -> torch.Tensor:
    return torch.div(lengths, config.hop_length, rounding_mode="floor") + 1


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _cmvn(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-utterance cepstral mean-variance normalisation over valid frames.

    features : (B, T, n_mels)  float32
    mask     : (B, T)          bool
    Returns  : (B, T, n_mels)  float32, normalised per mel-bin
    """
    m = mask.float().unsqueeze(-1)                         # (B, T, 1)
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)          # (B, 1, 1)
    mean = (features * m).sum(dim=1, keepdim=True) / n     # (B, 1, n_mels)
    var = ((features - mean).square() * m).sum(dim=1, keepdim=True) / n
    return (features - mean) / (var + 1e-5).sqrt()
