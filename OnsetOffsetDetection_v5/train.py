#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError as exc:
    raise RuntimeError("lightning is required") from exc

from OnsetOffsetDetection_v1.data import discover_gtsinger_english_recordings, make_raw_loader, save_split_manifests, split_recordings
from OnsetOffsetDetection_v1.preprocessing import OnsetOffsetLabelConfig
from OnsetOffsetDetection_v1.train import _DirectSaveCheckpointIO
from OnsetOffsetDetection_v1.train_utils import make_early_stopping, plot_training_curves
from OnsetOffsetDetection_v5.lightning_module import OnsetOffsetLightningModule, OnsetOffsetTrainingConfig
from OnsetOffsetDetection_v5.model import Wav2Vec2ModelConfig
from OnsetOffsetDetection_v5.preprocessing import Wav2Vec2FeatureConfig

DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "OnsetOffsetDetection_v5" / "runs" / "default"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OnsetOffsetDetection_v5 (Wav2Vec2 backbone).")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--group-contains", default=None)
    parser.add_argument("--batch-size", type=int, default=2)   # Wav2Vec2 is large; smaller batch
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-4)  # lower for fine-tuning
    # Wav2Vec2-specific
    parser.add_argument("--pretrained-model-name", default="facebook/wav2vec2-base")
    parser.add_argument("--freeze-feature-extractor", action="store_true", default=True)
    parser.add_argument("--freeze-transformer", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--max-audio-sec", type=float, default=30.0)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs). 0 disables early stopping.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    L.seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_gtsinger_english_recordings(args.dataset_root, args.limit, args.group_contains)
    if len(recordings) < 3:
        raise RuntimeError(f"Need at least 3 recordings, found {len(recordings)}")
    train, val, test = split_recordings(recordings, args.val_fraction, args.test_fraction, args.seed)
    manifests = save_split_manifests(args.output_dir, train, val, test)

    feature_config = Wav2Vec2FeatureConfig()
    model_config = Wav2Vec2ModelConfig(
        pretrained_model_name=args.pretrained_model_name,
        freeze_feature_extractor=args.freeze_feature_extractor,
        freeze_transformer=args.freeze_transformer,
        dropout=args.dropout,
    )
    training_config = OnsetOffsetTrainingConfig(
        model=model_config,
        features=feature_config,
        labels=OnsetOffsetLabelConfig(),
        learning_rate=args.learning_rate,
        max_audio_sec=args.max_audio_sec,
    )
    module = OnsetOffsetLightningModule(training_config)

    train_loader = make_raw_loader(train, args.batch_size, args.num_workers, True, pin_memory=True)
    val_loader = make_raw_loader(val, args.batch_size, args.num_workers, False, pin_memory=True)

    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="epoch={epoch:03d}-val_loss={val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint]
    if args.patience > 0:
        callbacks.append(make_early_stopping(args.patience))
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        default_root_dir=args.output_dir,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(args.output_dir), name="logs"),
        plugins=[_DirectSaveCheckpointIO()],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    (args.output_dir / "train_config.json").write_text(json.dumps({
        "args": vars(args) | {"dataset_root": str(args.dataset_root), "output_dir": str(args.output_dir)},
        "recordings": {"train": len(train), "val": len(val), "test": len(test)},
        "manifests": manifests,
        "training_config": asdict(training_config),
    }, indent=2))
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"output_dir: {args.output_dir}")
    print(f"best_checkpoint: {checkpoint.best_model_path}")
    plot_training_curves(args.output_dir)


if __name__ == "__main__":
    main()
