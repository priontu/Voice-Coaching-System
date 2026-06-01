#!/usr/bin/env python
"""Train and evaluate all OnsetOffsetDetection versions (v1–v5) in sequence.

Each version is trained on the full GTSinger English dataset for 20 epochs, then
evaluated on the full test set. Results are collected and printed as a comparison
table at the end.

Usage
-----
python test/OnsetOffsetDetection_all/test_1/run_test.py \
    --dataset-root /mnt/archive/GTSinger/English

Pass --versions v1 v3 v5 to run a subset of versions.
Pass --skip-existing to skip any version whose combined_summary.json already exists.

Required packages per version:
  v2  nnAudio          pip install nnAudio
  v4  mamba-ssm        pip install mamba-ssm causal-conv1d   (CUDA only)
  v5  transformers     pip install transformers
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

# Maps version tag → (train_script, evaluate_script, extra_train_flags)
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
        ["--accelerator", "gpu"],   # mamba-ssm requires CUDA
    ),
    "v5": (
        REPO_ROOT / "OnsetOffsetDetection_v5" / "train.py",
        REPO_ROOT / "OnsetOffsetDetection_v5" / "evaluate.py",
        ["--batch-size", "2"],      # Wav2Vec2 is larger; smaller batch
    ),
}

DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate all OnsetOffsetDetection versions.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--versions", nargs="+", default=list(VERSION_SCRIPTS), choices=list(VERSION_SCRIPTS),
                        help="Which versions to run (default: all).")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience passed to each train.py. 0 disables.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a version if its combined_summary.json already exists.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    precision = "bf16-mixed" if args.device.startswith("cuda") else "32"
    accelerator = "gpu" if args.device.startswith("cuda") else "cpu"

    results: dict[str, dict] = {}

    for version in args.versions:
        train_script, eval_script, extra_train = VERSION_SCRIPTS[version]
        version_dir = args.output_dir / version
        combined_path = version_dir / "combined_summary.json"

        if args.skip_existing and combined_path.exists():
            print(f"\n{'='*60}")
            print(f"{version}: skipping (combined_summary.json exists)")
            print(f"{'='*60}")
            results[version] = json.loads(combined_path.read_text())
            continue

        version_dir.mkdir(parents=True, exist_ok=True)
        train_run_dir = version_dir / "train_run"
        eval_output_dir = version_dir / "eval_outputs"

        # ------------------------------------------------------------------
        # Train
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"{version}: TRAIN")
        print(f"{'='*60}")

        # v4 forces --accelerator gpu via extra_train; skip generic accelerator
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
        # Only add --batch-size if extra_train doesn't override it
        if "--batch-size" not in extra_train:
            train_cmd += ["--batch-size", str(args.batch_size)]
        train_cmd += extra_train

        print("command:", " ".join(train_cmd))
        result = subprocess.run(train_cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"\n{version} training failed (exit {result.returncode})", file=sys.stderr)
            results[version] = {"error": f"training failed (exit {result.returncode})"}
            continue

        checkpoint = _find_best_checkpoint(train_run_dir)
        if checkpoint is None:
            print(f"\n{version}: no checkpoint found under {train_run_dir}", file=sys.stderr)
            results[version] = {"error": "no checkpoint found"}
            continue
        print(f"\n{version} best checkpoint: {checkpoint}")

        # ------------------------------------------------------------------
        # Evaluate
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"{version}: EVALUATE")
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
        ]
        print("command:", " ".join(eval_cmd))
        result = subprocess.run(eval_cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"\n{version} evaluation failed (exit {result.returncode})", file=sys.stderr)
            results[version] = {"error": f"evaluation failed (exit {result.returncode})"}
            continue

        # Collect results
        summary_path = eval_output_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            combined = {
                "checkpoint": str(checkpoint),
                "max_epochs": args.max_epochs,
                "precision": precision,
                "aggregate": summary.get("aggregate"),
            }
            combined_path.write_text(json.dumps(combined, indent=2))
            results[version] = combined

    # ----------------------------------------------------------------------
    # Comparison table
    # ----------------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    _print_table(results)

    table_path = args.output_dir / "comparison.json"
    table_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {table_path}")


def _print_table(results: dict[str, dict]) -> None:
    header = f"{'Version':<8} {'Onset F1':>9} {'Onset MAE':>10} {'Offset F1':>10} {'Offset MAE':>11} {'Note'}"
    print(header)
    print("-" * len(header))
    for version, data in results.items():
        if "error" in data:
            print(f"{version:<8}  {'ERROR: ' + data['error']}")
            continue
        agg = data.get("aggregate") or {}
        onset = agg.get("onset") or {}
        offset = agg.get("offset") or {}
        f1_on = f"{onset['f1']:.3f}" if onset.get("f1") is not None else "  n/a"
        mae_on = f"{onset['mae_sec']:.4f}s" if onset.get("mae_sec") is not None else "     n/a"
        f1_off = f"{offset['f1']:.3f}" if offset.get("f1") is not None else "  n/a"
        mae_off = f"{offset['mae_sec']:.4f}s" if offset.get("mae_sec") is not None else "     n/a"
        note = _version_note(version)
        print(f"{version:<8} {f1_on:>9} {mae_on:>10} {f1_off:>10} {mae_off:>11}  {note}")


def _version_note(version: str) -> str:
    notes = {
        "v1": "Conformer + mel",
        "v2": "Conformer + CQT",
        "v3": "Conformer + mel + onset→active coupling",
        "v4": "Mamba SSM + mel (no length cap)",
        "v5": "Wav2Vec2 fine-tune",
    }
    return notes.get(version, "")


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
