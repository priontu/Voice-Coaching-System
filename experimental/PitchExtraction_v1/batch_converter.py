from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hf_models.data.collator import VoiceCoachDeviceBatchConverter
from hf_models.preprocessing.torch_features import TorchAudioFeatureConfig
from hf_models.preprocessing.torch_labels import TorchLabelConfig, build_reference_f0_and_voicing

from .nanopitch_runtime import NanoPitchBatchExtractor, NanoPitchRuntimeConfig


@dataclass
class PitchExtractionBatchConverter:
    """Build model features/labels and attach NanoPitch pitch extraction.

    Output batch keys:
      input_features, labels, attention_mask, ...  from hf_models
      nanopitch.f0_hz, nanopitch.voiced, ...       native 100 Hz NanoPitch
      labels.nanopitch_f0, labels.nanopitch_voicing aligned to input_features
    """

    feature_config: TorchAudioFeatureConfig = field(default_factory=TorchAudioFeatureConfig)
    label_config: TorchLabelConfig = field(default_factory=TorchLabelConfig)
    nanopitch_config: NanoPitchRuntimeConfig = field(default_factory=NanoPitchRuntimeConfig)
    device: str | torch.device = "cpu"
    use_amp_features: bool = False
    align_nanopitch_to_training_frames: bool = True

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.base_converter = VoiceCoachDeviceBatchConverter(
            feature_config=self.feature_config,
            label_config=self.label_config,
            device=self.device,
            use_amp_features=self.use_amp_features,
        )
        self.nanopitch = NanoPitchBatchExtractor(
            config=self.nanopitch_config,
            device=self.device,
        )

    def load_weights(self, weights_path: str | Path) -> None:
        self.nanopitch.load_weights(weights_path)

    def __call__(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        batch = self.base_converter(raw_batch)
        nanopitch_outputs = self.nanopitch(raw_batch)
        batch["nanopitch"] = nanopitch_outputs
        ref_f0, ref_voicing = _reference_at_nanopitch_times(
            raw_batch["examples"],
            nanopitch_outputs["times"],
            nanopitch_outputs["frame_lengths"],
            self.device,
        )
        batch["labels"]["nanopitch_ref_f0"] = ref_f0
        batch["labels"]["nanopitch_ref_voicing"] = ref_voicing

        if self.align_nanopitch_to_training_frames:
            target_frames = batch["input_features"].shape[1]
            batch["labels"]["nanopitch_f0"] = _resize_1d(nanopitch_outputs["f0_hz"], target_frames)
            batch["labels"]["nanopitch_voicing"] = _resize_1d(nanopitch_outputs["voiced"], target_frames)

        return batch


def _resize_1d(values: torch.Tensor, target_frames: int) -> torch.Tensor:
    if values.shape[1] == target_frames:
        return values
    resized = F.interpolate(
        values.unsqueeze(1),
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(1)


def _reference_at_nanopitch_times(
    examples: list[Any],
    times: torch.Tensor,
    frame_lengths: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    f0_parts = []
    voicing_parts = []
    for index, example in enumerate(examples):
        starts = torch.tensor([note.start_sec for note in example.notes], dtype=torch.float32, device=device)
        ends = torch.tensor([note.end_sec for note in example.notes], dtype=torch.float32, device=device)
        freqs = torch.tensor([note.frequency_hz for note in example.notes], dtype=torch.float32, device=device)
        f0, voicing = build_reference_f0_and_voicing(times, starts, ends, freqs)
        valid = torch.arange(times.numel(), device=device) < frame_lengths[index]
        f0_parts.append(torch.where(valid, f0, torch.zeros_like(f0)))
        voicing_parts.append(torch.where(valid, voicing, torch.zeros_like(voicing)))
    return torch.stack(f0_parts, dim=0), torch.stack(voicing_parts, dim=0)
