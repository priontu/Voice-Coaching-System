from __future__ import annotations

import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from voice_coach.data.collator import VoiceCoachRawBatchCollator
from voice_coach.data.manifest import Recording, save_manifest
from voice_coach.data.runtime_dataset import VoiceCoachRuntimeDataset


def discover_gtsinger_english_recordings(
    dataset_root: str | Path,
    limit: int | None = None,
    group_contains: str | None = None,
) -> list[Recording]:
    root = Path(dataset_root)
    recordings: list[Recording] = []
    for wav_path in sorted(root.rglob("*.wav")):
        if group_contains and group_contains not in str(wav_path.parent):
            continue
        if not wav_path.with_suffix(".json").exists():
            continue
        musicxml_path = wav_path.with_suffix(".musicxml")
        if not musicxml_path.exists():
            musicxml_path = wav_path.with_suffix(".xml")
        if not musicxml_path.exists():
            continue
        textgrid_path = wav_path.with_suffix(".TextGrid")
        if not textgrid_path.exists():
            textgrid_path = wav_path.with_suffix(".textgrid")
        relative = wav_path.relative_to(root).with_suffix("")
        recordings.append(
            Recording(
                recording_id=sanitize_id(str(relative)),
                song_id=sanitize_id(str(relative.parent)),
                wav_path=wav_path,
                musicxml_path=musicxml_path,
                textgrid_path=textgrid_path if textgrid_path.exists() else None,
            )
        )
    return recordings[:limit] if limit is not None else recordings


def split_recordings(
    recordings: list[Recording],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Recording], list[Recording], list[Recording]]:
    if len(recordings) < 3:
        raise ValueError("At least three recordings are required for train/val/test split")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(recordings), generator=generator).tolist()
    test_size = max(1, int(len(recordings) * test_fraction))
    val_size = max(1, int(len(recordings) * val_fraction))
    if test_size + val_size >= len(recordings):
        test_size = 1
        val_size = 1
    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]
    return (
        [recordings[index] for index in train_indices],
        [recordings[index] for index in val_indices],
        [recordings[index] for index in test_indices],
    )


def make_raw_loader(
    recordings: list[Recording],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = VoiceCoachRuntimeDataset(recordings=recordings, cache_references=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=VoiceCoachRawBatchCollator(),
    )


def save_split_manifests(
    output_dir: Path,
    train: list[Recording],
    val: list[Recording],
    test: list[Recording],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_manifest": output_dir / "train_manifest.json",
        "val_manifest": output_dir / "val_manifest.json",
        "test_manifest": output_dir / "test_manifest.json",
    }
    save_manifest(train, paths["train_manifest"])
    save_manifest(val, paths["val_manifest"])
    save_manifest(test, paths["test_manifest"])
    return {key: str(value) for key, value in paths.items()}


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
