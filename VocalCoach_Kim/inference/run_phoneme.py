"""
inference/run_phoneme.py - CLI entry point for phoneme boundary detection.

Usage:
    python inference/run_phoneme.py <audio.wav> [options]

Options:
    --output, -o    Save results to JSON file
    --plot, -p      Visualize phoneme boundaries
    --words, -w     Group phonemes into words
    --reference, -r Reference JSON for evaluation metrics
    --device        cpu | cuda (default: auto)
    --config        Path to YAML config (default: configs/phoneme.yaml)
    --verbose, -v   Enable DEBUG logging

Example:
    python inference/run_phoneme.py singing.wav -o out.json --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the project root: python inference/run_phoneme.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.loader import load_model_config
from models.phoneme.phoneme_model import (
    PhonemeBoundaryConfig,
    PhonemeSegment,
    compute_boundary_metrics,
    extract_phoneme_boundaries_from_audio,
)
from utils.audio import load_audio_torch
from utils.device import get_torch_device
from utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract phoneme boundaries from audio using Wav2Vec2 + CTC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_file", help="Path to audio file (.wav/.flac/.mp3/.ogg)")
    parser.add_argument("--output", "-o", help="Save results to JSON")
    parser.add_argument("--plot", "-p", action="store_true", help="Visualize boundaries")
    parser.add_argument("--words", "-w", action="store_true", help="Group phonemes into words")
    parser.add_argument("--reference", "-r", help="Reference JSON for evaluation metrics")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="auto",
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config file (default: configs/phoneme.yaml)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    return parser


def main() -> None:
    # Fix IPA encoding on Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")

    # Build config: YAML → override with CLI args
    if args.config:
        from configs.loader import load_config, merge_configs
        yaml_cfg = load_config(args.config)
    else:
        yaml_cfg = load_model_config("phoneme")

    device_pref = args.device if args.device != "auto" else "auto"
    device = get_torch_device(device_pref)
    config = PhonemeBoundaryConfig.from_yaml({**yaml_cfg, "device": {"preference": device_pref}})
    config.device = device

    # Run pipeline
    result = extract_phoneme_boundaries_from_audio(
        args.audio_file,
        config=config,
        return_segments=True,
        word_grouping=args.words,
    )

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    preview = result["phonemes"][:20]
    suffix = "..." if len(result["phonemes"]) > 20 else ""
    print(f"\nPhoneme sequence (first 20): {' '.join(preview)}{suffix}")

    if result.get("segments"):
        print("\nFirst 5 segments:")
        for seg in result["segments"][:5]:
            print(
                f"  {seg.phoneme:<8s}  "
                f"{seg.start_time:.3f}s – {seg.end_time:.3f}s  "
                f"conf={seg.confidence:.3f}"
            )

    # Save JSON
    if args.output:
        output_data = {
            "phonemes": result["phonemes"],
            "boundaries": [[s, e] for s, e in result["boundaries"]],
            "segments": [seg.to_dict() for seg in result.get("segments", [])],
            "metadata": result["metadata"],
        }
        if "words" in result:
            output_data["words"] = result["words"]

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nOutput saved → {args.output}")

    # Visualize
    if args.plot:
        try:
            audio, _ = load_audio_torch(args.audio_file)
            from visualization.phoneme_viz import plot_phoneme_boundaries
            plot_path = (
                Path(args.output).stem + "_plot.png" if args.output else "phoneme_plot.png"
            )
            plot_phoneme_boundaries(audio, result.get("segments", []), save_path=str(plot_path))
        except Exception as exc:
            print(f"[Warning] Visualization failed: {exc}")

    # Evaluate against reference
    if args.reference:
        with open(args.reference, "r") as f:
            ref_data = json.load(f)
        ref_segments = [PhonemeSegment(**seg) for seg in ref_data.get("segments", [])]
        pred_segments = result.get("segments", [])
        metrics = compute_boundary_metrics(pred_segments, ref_segments)

        print(f"\n{'='*70}")
        print("EVALUATION METRICS")
        print(f"{'='*70}")
        for key, val in metrics.items():
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
