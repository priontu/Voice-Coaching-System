"""
Training script for the note onset/offset detection model.

Requires a CUDA GPU. Training will not run on CPU.
The active GPU name is logged at startup.

Usage::

    python train.py \\
        --config configs/default.yaml \\
        --train_manifest data/train.json \\
        --val_manifest data/val.json

Optional overrides::

    --checkpoint_dir checkpoints/run_01
    --resume checkpoints/run_01/epoch_050.pt
    --epochs 20
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_dataloaders
from metrics import compute_frame_metrics
from model import OnsetOffsetModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Triggers when validation loss fails to improve for *patience* epochs."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        """Return True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# Epoch runners
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: OnsetOffsetModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    onset_weight: float,
    offset_weight: float,
) -> Dict[str, float]:
    """Run one full training epoch and return average loss metrics."""
    model.train()
    total = on_sum = off_sum = 0.0
    n = 0

    with tqdm(loader, desc="  train", leave=False, unit="batch") as bar:
        for batch in bar:
            specs = batch["spectrogram"].to(device)
            on_tgt = batch["onset_labels"].to(device)
            off_tgt = batch["offset_labels"].to(device)

            optimizer.zero_grad()
            on_logits, off_logits = model(specs)
            l_on = criterion(on_logits, on_tgt)
            l_off = criterion(off_logits, off_tgt)
            loss = onset_weight * l_on + offset_weight * l_off
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += loss.item()
            on_sum += l_on.item()
            off_sum += l_off.item()
            n += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total / n,
        "onset_loss": on_sum / n,
        "offset_loss": off_sum / n,
    }


@torch.no_grad()
def validate_epoch(
    model: OnsetOffsetModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    onset_weight: float,
    offset_weight: float,
) -> Dict[str, float]:
    """Run one validation epoch; also computes frame-level F1."""
    model.eval()
    total = on_sum = off_sum = 0.0
    n = 0
    all_on_p, all_on_t, all_off_p, all_off_t = [], [], [], []

    with tqdm(loader, desc="    val", leave=False, unit="batch") as bar:
        for batch in bar:
            specs = batch["spectrogram"].to(device)
            on_tgt = batch["onset_labels"].to(device)
            off_tgt = batch["offset_labels"].to(device)

            on_logits, off_logits = model(specs)
            l_on = criterion(on_logits, on_tgt)
            l_off = criterion(off_logits, off_tgt)
            loss = onset_weight * l_on + offset_weight * l_off

            total += loss.item()
            on_sum += l_on.item()
            off_sum += l_off.item()
            n += 1

            all_on_p.append(torch.sigmoid(on_logits).cpu().numpy().ravel())
            all_on_t.append(on_tgt.cpu().numpy().ravel())
            all_off_p.append(torch.sigmoid(off_logits).cpu().numpy().ravel())
            all_off_t.append(off_tgt.cpu().numpy().ravel())

    frame_m = compute_frame_metrics(
        np.concatenate(all_on_p),
        np.concatenate(all_on_t),
        np.concatenate(all_off_p),
        np.concatenate(all_off_t),
    )

    return {
        "loss": total / n,
        "onset_loss": on_sum / n,
        "offset_loss": off_sum / n,
        **frame_m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: OnsetOffsetModel,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str,
    filename: str,
) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    out = Path(checkpoint_dir) / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        out,
    )
    logger.info("  checkpoint → %s", out)


def load_checkpoint(
    path: str,
    model: OnsetOffsetModel,
    optimizer: optim.Optimizer | None = None,
) -> int:
    """Load checkpoint into model (and optionally optimizer); return start epoch."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    logger.info("Resumed from %s  (epoch %d)", path, epoch)
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, device: torch.device) -> OnsetOffsetModel:
    mc, ac = cfg["model"], cfg["audio"]
    return OnsetOffsetModel(
        n_mels=ac["n_mels"],
        cnn_channels=mc["cnn_channels"],
        lstm_hidden_size=mc["lstm_hidden_size"],
        lstm_num_layers=mc["lstm_num_layers"],
        lstm_dropout=mc["lstm_dropout"],
        head_hidden_size=mc["head_hidden_size"],
        dropout=mc["dropout"],
    ).to(device)


def train(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.checkpoint_dir:
        cfg["paths"]["checkpoint_dir"] = args.checkpoint_dir
    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Training requires a GPU.\n"
            "Check your PyTorch installation: https://pytorch.org/get-started"
        )
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    logger.info("GPU: %s", torch.cuda.get_device_name(0))

    ac = cfg["audio"]
    tc = cfg["training"]
    dc = cfg["data"]

    train_loader, val_loader = create_dataloaders(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        batch_size=tc["batch_size"],
        num_workers=dc["num_workers"],
        sample_rate=ac["sample_rate"],
        n_fft=ac["n_fft"],
        hop_length=ac["hop_length"],
        n_mels=ac["n_mels"],
        fmin=ac["fmin"],
        fmax=ac["fmax"],
        label_sigma=tc["label_sigma"],
        segment_length=dc.get("segment_length"),
    )

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss()
    stopper = EarlyStopping(patience=tc["patience"])

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)

    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    best_val_loss = float("inf")

    for epoch in range(start_epoch + 1, tc["num_epochs"] + 1):
        tr = train_epoch(
            model, train_loader, optimizer, criterion, device,
            tc["grad_clip"], tc["onset_weight"], tc["offset_weight"],
        )
        va = validate_epoch(
            model, val_loader, criterion, device,
            tc["onset_weight"], tc["offset_weight"],
        )
        scheduler.step(va["loss"])

        logger.info(
            "Epoch %03d | tr_loss=%.4f | va_loss=%.4f | "
            "on_f1=%.3f | off_f1=%.3f | lr=%.2e",
            epoch,
            tr["loss"],
            va["loss"],
            va.get("onset_f1", 0.0),
            va.get("offset_f1", 0.0),
            optimizer.param_groups[0]["lr"],
        )

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            save_checkpoint(model, optimizer, epoch, va, ckpt_dir, "best.pt")

        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, va, ckpt_dir, f"epoch_{epoch:03d}.pt"
            )

        if stopper(va["loss"]):
            logger.info("Early stopping at epoch %d.", epoch)
            break

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train note onset/offset detection model"
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--train_manifest", required=True, help="Training manifest JSON")
    parser.add_argument("--val_manifest", required=True, help="Validation manifest JSON")
    parser.add_argument("--checkpoint_dir", default=None, help="Override checkpoint dir")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
