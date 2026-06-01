#!/usr/bin/env python
"""Train then evaluate a single OnsetOffsetDetection version.

Usage
-----
python test/run_version.py --version v1 --dataset-root /mnt/archive/GTSinger/English
python test/run_version.py --version v2 --dataset-root /mnt/archive/GTSinger/English --batch-size 64 --max-epochs 200
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

VERSION_SCRIPTS: dict[str, tuple[Path, Path, list[str]]] = {
    "v1": (
        REPO_ROOT / "OnsetOffsetDetection_v1" / "train.py",
        REPO_ROOT / "OnsetOffsetDetection_v1" / "evaluate.py",
        [],
    ),
    "v2": (
        REPO_ROOT / "OnsetOffsetDetection_v2" / "train.py",
        REPO_ROOT / "OnsetOffsetDetection_v2" / "evaluate.py",
        [],
    ),
    "v3": (
        REPO_ROOT / "OnsetOffsetDetection_v3" / "train.py",
        REPO_ROOT / "OnsetOffsetDetection_v3" / "evaluate.py",
        [],
    ),
    "v4": (
        REPO_ROOT / "OnsetOffsetDetection_v4" / "train.py",
        REPO_ROOT / "OnsetOffsetDetection_v4" / "evaluate.py",
        ["--accelerator", "gpu"],
    ),
    "v5": (
        REPO_ROOT / "OnsetOffsetDetection_v5" / "train.py",
        REPO_ROOT / "OnsetOffsetDetection_v5" / "evaluate.py",
        ["--batch-size", "2"],
    ),
}

DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate one OnsetOffsetDetection version.")
    parser.add_argument("--version", required=True, choices=list(VERSION_SCRIPTS),
                        help="Which version to run: v1, v2, v3, v4, or v5.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: test/runs/<version>).")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience. 0 disables.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot-limit", type=int, default=8,
                        help="Recordings to plot during eval. 0 skips plotting.")
    args = parser.parse_args()

    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.version)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_run_dir = output_dir / "train_run"
    eval_output_dir = output_dir / "eval"

    train_script, eval_script, extra_train = VERSION_SCRIPTS[args.version]
    precision = "bf16-mixed" if args.device.startswith("cuda") else "32"
    accelerator = "gpu" if args.device.startswith("cuda") else "cpu"

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"{args.version}: TRAIN")
    print(f"{'='*60}")

    uses_own_accelerator = "--accelerator" in extra_train
    train_cmd = [
        sys.executable, str(train_script),
        "--dataset-root", str(args.dataset_root),
        "--output-dir", str(train_run_dir),
        "--max-epochs", str(args.max_epochs),
        "--num-workers", str(args.num_workers),
        "--precision", precision,
        "--patience", str(args.patience),
    ]
    if not uses_own_accelerator:
        train_cmd += ["--accelerator", accelerator]
    if "--batch-size" not in extra_train:
        train_cmd += ["--batch-size", str(args.batch_size)]
    train_cmd += extra_train

    print("command:", " ".join(train_cmd))
    result = subprocess.run(train_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n{args.version} training failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    checkpoint = _find_best_checkpoint(train_run_dir)
    if checkpoint is None:
        print(f"\nNo checkpoint found under {train_run_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"\nBest checkpoint: {checkpoint}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"{args.version}: EVALUATE")
    print(f"{'='*60}")

    eval_batch = args.batch_size if "--batch-size" not in extra_train else int(extra_train[extra_train.index("--batch-size") + 1])
    eval_cmd = [
        sys.executable, str(eval_script),
        "--dataset-root", str(args.dataset_root),
        "--checkpoint", str(checkpoint),
        "--output-dir", str(eval_output_dir),
        "--batch-size", str(eval_batch),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--plot-limit", str(args.plot_limit),
    ]
    print("command:", " ".join(eval_cmd))
    result = subprocess.run(eval_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n{args.version} evaluation failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    summary_path = eval_output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        agg = summary.get("aggregate", {})
        print(f"\n{'='*60}")
        print(f"{args.version} RESULTS")
        print(f"{'='*60}")
        print(json.dumps(agg, indent=2))


def _find_best_checkpoint(run_dir: Path) -> Path | None:
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
