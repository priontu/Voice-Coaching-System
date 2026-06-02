"""
Evaluation script for the note onset/offset detection model.

Runs the model on one or more GTSinger WAV files and reports
onset, offset, and duration metrics against the ground truth JSON.

Usage
-----
# Single file:
    python evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --config     configs/default.yaml \\
        --audio      "path/to/0000.wav" \\
        --label      "path/to/0000.json"

# Whole manifest (e.g. the test split):
    python evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --config     configs/default.yaml \\
        --manifest   data/manifests/test.json

# Change timing tolerance (default 50 ms):
    python evaluate.py ... --tolerance_ms 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from inference import NoteDetector
from metrics import evaluate_file
from utils import parse_gtsinger_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_single(
    detector: NoteDetector,
    audio_path: str,
    label_path: str,
    tolerance_ms: float,
) -> Dict:
    """Run detection + evaluation on one audio/label pair."""
    predicted = detector.detect(audio_path)

    ref_notes = parse_gtsinger_json(label_path)
    reference = [{"onset_time": n.onset, "offset_time": n.offset} for n in ref_notes]

    return evaluate_file(predicted, reference, tolerance_ms=tolerance_ms)


def average_metrics(results: List[Dict]) -> Dict:
    """Compute mean of each metric across multiple files."""
    keys = [k for k in results[0] if isinstance(results[0][k], float)]
    return {k: round(sum(r[k] for r in results) / len(results), 4) for k in keys}


def print_metrics(metrics: Dict, label: str = "Results") -> None:
    width = 28
    print(f"\n{'─' * 44}")
    print(f"  {label}")
    print(f"{'─' * 44}")
    print(f"  {'Metric':<{width}} Value")
    print(f"  {'──────':<{width}} ─────")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<{width}} {v:.4f}")
        else:
            print(f"  {k:<{width}} {v}")
    print(f"{'─' * 44}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate note onset/offset detection against GTSinger ground truth"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--audio",      default=None,  help="Single WAV file to evaluate")
    parser.add_argument("--label",      default=None,  help="GTSinger JSON label for --audio")
    parser.add_argument("--manifest",   default=None,  help="Manifest JSON for batch evaluation")
    parser.add_argument("--tolerance_ms", type=float, default=50.0,
                        help="Timing tolerance for P/R/F1 matching in ms (default: 50)")
    parser.add_argument("--output",     default=None,
                        help="Optional path to save results as JSON")
    args = parser.parse_args()

    if args.audio is None and args.manifest is None:
        parser.error("Provide either --audio + --label, or --manifest.")
    if args.audio is not None and args.label is None:
        parser.error("--label is required when using --audio.")

    detector = NoteDetector.from_checkpoint(args.checkpoint, args.config)

    # ── Single file ───────────────────────────────────────────────────────
    if args.audio:
        metrics = evaluate_single(detector, args.audio, args.label, args.tolerance_ms)
        print_metrics(metrics, label=Path(args.audio).name)
        all_results = [{"file": args.audio, **metrics}]

    # ── Manifest (batch) ──────────────────────────────────────────────────
    else:
        with open(args.manifest, "r", encoding="utf-8") as f:
            samples = json.load(f)

        logger.info("Evaluating %d files from %s", len(samples), args.manifest)
        all_results = []

        for i, sample in enumerate(samples, 1):
            audio_path = sample["audio"]
            label_path = sample["label"]
            logger.info("[%d/%d] %s", i, len(samples), Path(audio_path).name)

            try:
                metrics = evaluate_single(detector, audio_path, label_path, args.tolerance_ms)
                all_results.append({"file": audio_path, **metrics})
            except Exception as e:
                logger.warning("Failed on %s: %s", audio_path, e)

        if not all_results:
            logger.error("No files evaluated successfully.")
            return

        avg = average_metrics(all_results)
        print_metrics(avg, label=f"Average over {len(all_results)} files")

    # ── Save output ───────────────────────────────────────────────────────
    if args.output:
        out = {"per_file": all_results}
        if len(all_results) > 1:
            out["average"] = average_metrics(all_results)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
