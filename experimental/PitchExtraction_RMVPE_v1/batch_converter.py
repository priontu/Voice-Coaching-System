from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hf_models.data.collator import VoiceCoachDeviceBatchConverter
from hf_models.preprocessing.torch_features import TorchAudioFeatureConfig
from hf_models.preprocessing.torch_labels import TorchLabelConfig, build_reference_f0_and_voicing

from .runtime import RMVPEBatchExtractor, RMVPERuntimeConfig
from .scoring import RMVPECoachingScoreConfig, note_median_pitch_events


@dataclass
class RMVPEBatchConverter:
    """Build training features/labels and attach RMVPE pitch extraction."""

    feature_config: TorchAudioFeatureConfig = field(default_factory=TorchAudioFeatureConfig)
    label_config: TorchLabelConfig = field(default_factory=TorchLabelConfig)
    rmvpe_config: RMVPERuntimeConfig = field(default_factory=RMVPERuntimeConfig)
    coaching_score_config: RMVPECoachingScoreConfig = field(default_factory=RMVPECoachingScoreConfig)
    device: str | torch.device = "cpu"
    use_amp_features: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.base_converter = VoiceCoachDeviceBatchConverter(
            feature_config=self.feature_config,
            label_config=self.label_config,
            device=self.device,
            use_amp_features=self.use_amp_features,
        )
        self.rmvpe = RMVPEBatchExtractor(
            config=self.rmvpe_config,
            device=self.device,
        )

    def __call__(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        batch = self.base_converter(raw_batch)
        rmvpe_outputs = self.rmvpe(raw_batch)
        batch["rmvpe"] = rmvpe_outputs

        ref_f0, ref_voicing = _reference_at_rmvpe_times(
            raw_batch["examples"],
            rmvpe_outputs["times"],
            rmvpe_outputs["frame_lengths"],
            self.device,
        )
        batch["labels"]["rmvpe_ref_f0"] = ref_f0
        batch["labels"]["rmvpe_ref_voicing"] = ref_voicing
        batch["rmvpe_recommended"] = _recommended_note_scores(
            raw_batch["examples"],
            rmvpe_outputs,
            self.coaching_score_config,
        )
        return batch


def _reference_at_rmvpe_times(
    examples: list[Any],
    times: torch.Tensor,
    frame_lengths: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    f0_parts = []
    voicing_parts = []
    for index, example in enumerate(examples):
        current_times = times[index]
        starts = torch.tensor([note.start_sec for note in example.notes], dtype=torch.float32, device=device)
        ends = torch.tensor([note.end_sec for note in example.notes], dtype=torch.float32, device=device)
        freqs = torch.tensor([note.frequency_hz for note in example.notes], dtype=torch.float32, device=device)
        f0, voicing = build_reference_f0_and_voicing(current_times, starts, ends, freqs)
        valid = torch.arange(current_times.numel(), device=device) < frame_lengths[index]
        f0_parts.append(torch.where(valid, f0, torch.zeros_like(f0)))
        voicing_parts.append(torch.where(valid, voicing, torch.zeros_like(voicing)))
    return torch.stack(f0_parts, dim=0), torch.stack(voicing_parts, dim=0)


def _recommended_note_scores(
    examples: list[Any],
    rmvpe_outputs: dict[str, Any],
    config: RMVPECoachingScoreConfig,
) -> list[list[dict[str, float | int]]]:
    scores = []
    for index, example in enumerate(examples):
        length = int(rmvpe_outputs["frame_lengths"][index].item())
        scores.append(
            note_median_pitch_events(
                times_sec=rmvpe_outputs["times"][index, :length].detach().float().cpu(),
                f0_hz=rmvpe_outputs["f0_hz"][index, :length].detach().float().cpu(),
                confidence=rmvpe_outputs["confidence"][index, :length].detach().float().cpu(),
                notes=example.notes,
                config=config,
            )
        )
    return scores
