from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import lightning as L
except ImportError as exc:
    raise RuntimeError("lightning is required for PitchExtractionDataModule") from exc

from voice_coach.data.collator import VoiceCoachRawBatchCollator
from voice_coach.data.manifest import Recording, load_manifest
from voice_coach.data.runtime_dataset import VoiceCoachRuntimeDataset
from voice_coach.preprocessing.torch_features import TorchAudioFeatureConfig
from voice_coach.preprocessing.torch_labels import TorchLabelConfig

from .batch_converter import PitchExtractionBatchConverter
from .nanopitch_runtime import NanoPitchRuntimeConfig


@dataclass
class PitchExtractionDataConfig:
    train_manifest: Path
    val_manifest: Path | None = None
    test_manifest: Path | None = None
    batch_size: int = 8
    num_workers: int = 4
    val_fraction: float = 0.1
    seed: int = 1234
    pin_memory: bool = True
    persistent_workers: bool = True
    cache_references: bool = True
    feature: TorchAudioFeatureConfig = field(default_factory=TorchAudioFeatureConfig)
    labels: TorchLabelConfig = field(default_factory=TorchLabelConfig)
    nanopitch: NanoPitchRuntimeConfig = field(default_factory=NanoPitchRuntimeConfig)


class PitchExtractionDataModule(L.LightningDataModule):
    """Lightning DataModule that adds NanoPitch extraction to runtime batches."""

    def __init__(self, config: PitchExtractionDataConfig):
        super().__init__()
        self.config = config
        self.raw_collator = VoiceCoachRawBatchCollator()
        self.device_converters: dict[str, PitchExtractionBatchConverter] = {}

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self.config.val_manifest is not None:
                self.train_dataset = self._dataset_from_manifest(self.config.train_manifest)
                self.val_dataset = self._dataset_from_manifest(self.config.val_manifest)
            else:
                recordings = load_manifest(self.config.train_manifest)
                if len(recordings) < 2:
                    raise ValueError("At least two recordings are required to auto-split train/val datasets")
                val_size = max(1, int(len(recordings) * self.config.val_fraction))
                val_size = min(val_size, len(recordings) - 1)
                train_size = len(recordings) - val_size
                generator = torch.Generator().manual_seed(self.config.seed)
                train_subset, val_subset = random_split(recordings, [train_size, val_size], generator=generator)
                self.train_dataset = self._dataset_from_recordings(list(train_subset))
                self.val_dataset = self._dataset_from_recordings(list(val_subset))

        if stage in (None, "test") and self.config.test_manifest is not None:
            self.test_dataset = self._dataset_from_manifest(self.config.test_manifest)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)

    def transfer_batch_to_device(
        self,
        batch: dict[str, Any],
        device: torch.device,
        dataloader_idx: int,
    ) -> dict[str, Any]:
        key = str(device)
        if key not in self.device_converters:
            self.device_converters[key] = PitchExtractionBatchConverter(
                feature_config=self.config.feature,
                label_config=self.config.labels,
                nanopitch_config=self.config.nanopitch,
                device=device,
                use_amp_features=device.type == "cuda",
            )
        return self.device_converters[key](batch)

    def _dataset_from_manifest(self, manifest: Path) -> VoiceCoachRuntimeDataset:
        return VoiceCoachRuntimeDataset(
            manifest_path=manifest,
            target_sample_rate=self.config.feature.sample_rate,
            cache_references=self.config.cache_references,
        )

    def _dataset_from_recordings(self, recordings: list[Recording]) -> VoiceCoachRuntimeDataset:
        return VoiceCoachRuntimeDataset(
            recordings=recordings,
            target_sample_rate=self.config.feature.sample_rate,
            cache_references=self.config.cache_references,
        )

    def _loader(self, dataset: VoiceCoachRuntimeDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers and self.config.num_workers > 0,
            collate_fn=self.raw_collator,
        )

