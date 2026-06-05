from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from voice_coach.data.runtime_dataset import RuntimeExample
from voice_coach.preprocessing.torch_features import (
    TorchAudioFeatureConfig,
    TorchLogMelExtractor,
    audio_lengths_to_frame_lengths,
    frame_times,
    pad_audio_batch,
    resample_audio,
)
from voice_coach.preprocessing.torch_labels import TorchLabelConfig, build_training_labels, pad_label_dicts


@dataclass
class VoiceCoachRawBatchCollator:
    """Keep raw audio and references ragged for later device conversion."""

    def __call__(self, examples: list[RuntimeExample]) -> dict[str, Any]:
        return {
            "audio_items": [example.audio for example in examples],
            "audio_sample_rates": [example.audio_sample_rate for example in examples],
            "examples": examples,
            "metadata": [self._metadata(example) for example in examples],
        }

    def _metadata(self, example: RuntimeExample) -> dict[str, str | None]:
        recording = example.recording
        return {
            "recording_id": recording.recording_id,
            "song_id": recording.song_id,
            "wav_path": _path_string(recording.wav_path),
            "musicxml_path": _path_string(recording.musicxml_path),
            "textgrid_path": _path_string(recording.textgrid_path),
        }


@dataclass
class VoiceCoachDeviceBatchConverter:
    """Convert raw batches into model-ready tensors on CPU or GPU."""

    feature_config: TorchAudioFeatureConfig = field(default_factory=TorchAudioFeatureConfig)
    label_config: TorchLabelConfig = field(default_factory=TorchLabelConfig)
    device: str | torch.device = "cpu"
    use_amp_features: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.feature_extractor = TorchLogMelExtractor(self.feature_config).to(self.device)

    def __call__(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        audio_items = [
            resample_audio(
                audio.to(self.device, non_blocking=True),
                original_sample_rate=sample_rate,
                target_sample_rate=self.feature_config.sample_rate,
            )
            for audio, sample_rate in zip(raw_batch["audio_items"], raw_batch["audio_sample_rates"], strict=True)
        ]
        audio_batch, audio_lengths = pad_audio_batch(audio_items)
        examples: list[RuntimeExample] = raw_batch["examples"]

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.use_amp_features and self.device.type == "cuda",
        ):
            input_features = self.feature_extractor(audio_batch)

        max_frames = input_features.shape[1]
        frame_lengths = torch.clamp(
            audio_lengths_to_frame_lengths(audio_lengths, self.feature_config),
            max=max_frames,
        )
        attention_mask = _lengths_to_mask(frame_lengths, max_frames)

        label_dicts = [
            self._labels_for_example(example, int(frame_lengths[index].item()))
            for index, example in enumerate(examples)
        ]
        labels = pad_label_dicts(label_dicts, max_frames)

        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "labels": labels,
            "frame_lengths": frame_lengths,
            "audio_lengths": audio_lengths,
            "metadata": raw_batch["metadata"],
        }

    def _labels_for_example(self, example: RuntimeExample, num_frames: int) -> dict[str, torch.Tensor]:
        times = frame_times(num_frames, self.feature_config, self.device)
        starts, ends, freqs = _note_tensors(example, self.device)
        boundaries = _phoneme_boundary_tensor(example, self.device)
        return build_training_labels(
            times,
            starts,
            ends,
            freqs,
            boundaries,
            self.label_config,
        )


@dataclass
class VoiceCoachBatchCollator:
    """Convenience collator returning model-ready tensors directly.

    For higher-throughput CUDA training, prefer VoiceCoachRawBatchCollator in
    the DataLoader and call VoiceCoachDeviceBatchConverter in the training loop.
    """

    feature_config: TorchAudioFeatureConfig = field(default_factory=TorchAudioFeatureConfig)
    label_config: TorchLabelConfig = field(default_factory=TorchLabelConfig)
    device: str | torch.device = "cpu"
    use_amp_features: bool = False

    def __post_init__(self) -> None:
        self.raw_collator = VoiceCoachRawBatchCollator()
        self.converter = VoiceCoachDeviceBatchConverter(
            feature_config=self.feature_config,
            label_config=self.label_config,
            device=self.device,
            use_amp_features=self.use_amp_features,
        )

    def __call__(self, examples: list[RuntimeExample]) -> dict[str, Any]:
        return self.converter(self.raw_collator(examples))


def _note_tensors(example: RuntimeExample, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    starts = torch.tensor([note.start_sec for note in example.notes], dtype=torch.float32, device=device)
    ends = torch.tensor([note.end_sec for note in example.notes], dtype=torch.float32, device=device)
    freqs = torch.tensor([note.frequency_hz for note in example.notes], dtype=torch.float32, device=device)
    return starts, ends, freqs


def _phoneme_boundary_tensor(example: RuntimeExample, device: torch.device) -> torch.Tensor:
    boundaries: list[float] = []
    for phoneme in example.phonemes:
        if _is_scored_phone(phoneme.text):
            boundaries.extend([phoneme.start_sec, phoneme.end_sec])
    return torch.tensor(boundaries, dtype=torch.float32, device=device)


def _is_scored_phone(value: str) -> bool:
    normalized = value.lower().strip().replace("ː", "").replace(":", "")
    if not normalized:
        return False
    return normalized not in {"<sp>", "<ap>", "sp", "ap", "sil", "silence", "pau", "pause"}


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _path_string(path: Path | None) -> str | None:
    return str(path) if path is not None else None
