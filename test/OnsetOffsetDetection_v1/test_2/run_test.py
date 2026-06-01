#!/usr/bin/env python
"""Train OnsetOffsetDetection_v1 on the full dataset, then evaluate the best checkpoint.

Usage
-----
python test/OnsetOffsetDetection_v1/test_2/run_test.py \
    --dataset-root /mnt/archive/GTSinger/English

Add --device cpu to run without a GPU (slow).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = REPO_ROOT / "OnsetOffsetDetection_v1" / "train.py"
EVALUATE_SCRIPT = REPO_ROOT / "OnsetOffsetDetection_v1" / "evaluate.py"
DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate OnsetOffsetDetection_v1.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.5)
    parser.add_argument("--tolerance-sec", type=float, default=0.05)
    parser.add_argument("--min-distance-sec", type=float, default=0.05)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_run_dir = args.output_dir / "train_run"

    precision = "bf16-mixed" if args.device.startswith("cuda") else "32"
    accelerator = "gpu" if args.device.startswith("cuda") else "cpu"

    # -------------------------------------------------------------------------
    # Step 1 – train
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: train")
    print("=" * 60)

    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--dataset-root", str(args.dataset_root),
        "--output-dir", str(train_run_dir),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--max-epochs", str(args.max_epochs),
        "--accelerator", accelerator,
        "--devices", "2",
        "--precision", precision,
    ]
    print("command:", " ".join(train_cmd))
    print()

    train_result = subprocess.run(train_cmd, capture_output=False, text=True)
    if train_result.returncode != 0:
        print(f"\nTraining failed (exit {train_result.returncode})", file=sys.stderr)
        sys.exit(train_result.returncode)

    checkpoint = _find_best_checkpoint(train_run_dir)
    if checkpoint is None:
        print("Could not locate a checkpoint under", train_run_dir, file=sys.stderr)
        sys.exit(1)
    print(f"\nBest checkpoint: {checkpoint}")

    # -------------------------------------------------------------------------
    # Step 2 – evaluate
    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 2: evaluate")
    print("=" * 60)

    eval_output_dir = args.output_dir / "eval_outputs"
    eval_cmd = [
        sys.executable, str(EVALUATE_SCRIPT),
        "--dataset-root", str(args.dataset_root),
        "--checkpoint", str(checkpoint),
        "--output-dir", str(eval_output_dir),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--onset-threshold", str(args.onset_threshold),
        "--offset-threshold", str(args.offset_threshold),
        "--tolerance-sec", str(args.tolerance_sec),
        "--min-distance-sec", str(args.min_distance_sec),
    ]
    print("command:", " ".join(eval_cmd))
    print()

    eval_result = subprocess.run(eval_cmd, capture_output=False, text=True)
    if eval_result.returncode != 0:
        print(f"\nEvaluation failed (exit {eval_result.returncode})", file=sys.stderr)
        sys.exit(eval_result.returncode)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    summary_path = eval_output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        combined = {
            "checkpoint": str(checkpoint),
            "max_epochs": args.max_epochs,
            "precision": precision,
            "aggregate": summary.get("aggregate"),
        }
        combined_path = args.output_dir / "combined_summary.json"
        combined_path.write_text(json.dumps(combined, indent=2))
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(json.dumps(combined, indent=2))
        print(f"\ncombined summary: {combined_path}")


def _find_best_checkpoint(run_dir: Path) -> Path | None:
    """Return the checkpoint with the lowest val_loss, or last.ckpt as fallback."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    candidates = []
    for ckpt in ckpt_dir.glob("epoch=*-val_loss=*.ckpt"):
        m = re.search(r"val_loss=([0-9]+\.[0-9]+)", ckpt.name)
        if m:
            candidates.append((float(m.group(1)), ckpt))
    if candidates:
        candidates.sort()
        return candidates[0][1]

    last = ckpt_dir / "last.ckpt"
    return last if last.exists() else None


if __name__ == "__main__":
    main()
