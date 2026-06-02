"""
inference/run_onset_offset.py - CLI entry point for note onset/offset detection.

Usage:
    python inference/run_onset_offset.py \\
        --checkpoint checkpoints/best.pt \\
        --audio singing.wav \\
        [--output results.json] \\
        [--config configs/onset_offset.yaml]

Options:
    --checkpoint    Path to .pt checkpoint (required)
    --audio         Path to input WAV file (required)
    --output        Output JSON path (prints to stdout if omitted)
    --config        Path to YAML config (default: configs/onset_offset.yaml)
    --verbose, -v   Enable DEBUG logging

Example:
    python inference/run_onset_offset.py \\
        --checkpoint checkpoints/best.pt --audio singing.wav --output notes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.onset_offset.detector import NoteDetector
from utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Note onset/offset detection inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument(
        "--config", default="configs/onset_offset.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--audio", required=True, help="Path to input WAV file.")
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (prints to stdout if omitted).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")

    detector = NoteDetector.from_checkpoint(args.checkpoint, args.config)
    notes = detector.detect(args.audio)

    output = json.dumps(notes, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
