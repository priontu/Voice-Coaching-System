#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from PitchExtraction_v1.batch_converter import PitchExtractionBatchConverter
from PitchExtraction_v1.nanopitch_runtime import NanoPitchRuntimeConfig
from hf_models.data.collator import VoiceCoachRawBatchCollator
from hf_models.data.runtime_dataset import VoiceCoachRuntimeDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect runtime preprocessing plus NanoPitch extraction.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent / "weights.pth")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decoder", choices=["argmax", "offline", "realtime"], default="argmax")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    dataset = VoiceCoachRuntimeDataset(manifest_path=args.manifest, cache_references=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=VoiceCoachRawBatchCollator(),
    )
    converter = PitchExtractionBatchConverter(
        nanopitch_config=NanoPitchRuntimeConfig(weights_path=args.weights, decoder=args.decoder),
        device=args.device,
        use_amp_features=args.device.startswith("cuda"),
    )

    batch = converter(next(iter(loader)))
    print(f"device: {batch['input_features'].device}")
    print(f"input_features: {tuple(batch['input_features'].shape)}")
    print(f"labels.f0: {tuple(batch['labels']['f0'].shape)}")
    print(f"labels.nanopitch_f0: {tuple(batch['labels']['nanopitch_f0'].shape)}")
    print(f"nanopitch.f0_hz: {tuple(batch['nanopitch']['f0_hz'].shape)}")
    print(f"nanopitch voiced frames: {int(batch['nanopitch']['voiced'].sum().item())}")
    print(f"recordings: {[item['recording_id'] for item in batch['metadata']]}")


if __name__ == "__main__":
    main()
