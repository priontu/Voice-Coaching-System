#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hf_models.data.collator import VoiceCoachDeviceBatchConverter, VoiceCoachRawBatchCollator
from hf_models.data.runtime_dataset import VoiceCoachRuntimeDataset
from hf_models.preprocessing.torch_features import TorchAudioFeatureConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one runtime training/evaluation batch.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    dataset = VoiceCoachRuntimeDataset(
        manifest_path=args.manifest,
        target_sample_rate=args.sample_rate,
        cache_references=True,
    )
    raw_collator = VoiceCoachRawBatchCollator()
    converter = VoiceCoachDeviceBatchConverter(
        feature_config=TorchAudioFeatureConfig(
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            hop_length=args.hop_length,
        ),
        device=args.device,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=raw_collator,
    )

    batch = converter(next(iter(loader)))
    print(f"device: {batch['input_features'].device}")
    print(f"input_features: {tuple(batch['input_features'].shape)}")
    print(f"attention_mask: {tuple(batch['attention_mask'].shape)}")
    for name, value in batch["labels"].items():
        print(f"labels.{name}: {tuple(value.shape)} {value.dtype} {value.device}")
    print(f"recordings: {[item['recording_id'] for item in batch['metadata']]}")


if __name__ == "__main__":
    main()
