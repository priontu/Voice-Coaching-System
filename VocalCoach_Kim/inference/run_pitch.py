"""
inference/run_pitch.py - CLI entry point for VAD + pitch estimation.

Usage:
    python inference/run_pitch.py --audio singing.wav [options]

Options:
    --audio, -a         Path to input WAV file (required)
    --output, -o        Output JSON path (default: pitch_data.json)
    --backend           torchcrepe | pyin (default: torchcrepe)
    --device            auto | cpu | cuda (default: auto)
    --vad-mode          WebRTC aggressiveness 0-3 (default: 2)
    --vad-frame-ms      VAD frame duration 10|20|30 ms (default: 20)
    --no-vad            Skip VAD; rely on pitch model's voiced detection
    --hop-length        Pitch hop in samples (default: 160)
    --fmin              Minimum F0 Hz (default: 50)
    --fmax              Maximum F0 Hz (default: 1000)
    --gap-fill          Max gap frames to interpolate (default: 10)
    --visualize         Save visualization plots
    --config            Path to YAML config (default: configs/pitch.yaml)
    --verbose, -v       Enable DEBUG logging

Example:
    python inference/run_pitch.py --audio singing.wav --output pitch_data.json --visualize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from configs.loader import load_model_config
from models.pitch.fusion import FusionConfig
from models.pitch.pitch_wrapper import PitchConfig
from models.pitch.pipeline import PipelineConfig, PitchVADPipeline
from models.pitch.vad import VADConfig
from utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VAD + Pitch extraction pipeline for singing analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", "-a", required=True, help="Path to input WAV file.")
    parser.add_argument("--output", "-o", default="pitch_data.json", help="Output JSON path.")
    parser.add_argument("--backend", choices=["torchcrepe", "pyin"], default="torchcrepe")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=2)
    parser.add_argument("--vad-frame-ms", type=int, choices=[10, 20, 30], default=20)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--fmin", type=float, default=50.0)
    parser.add_argument("--fmax", type=float, default=1000.0)
    parser.add_argument("--periodicity-threshold", type=float, default=0.21)
    parser.add_argument("--gap-fill", type=int, default=10)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")

    # Build PipelineConfig from YAML base + CLI overrides
    if args.config:
        from configs.loader import load_config
        yaml_cfg = load_config(args.config)
    else:
        yaml_cfg = load_model_config("pitch")

    config = PipelineConfig(
        use_vad=not args.no_vad,
        vad=VADConfig(
            aggressiveness=args.vad_mode,
            frame_duration_ms=args.vad_frame_ms,
            sample_rate=16000,
        ),
        pitch=PitchConfig(
            backend=args.backend,
            hop_length=args.hop_length,
            fmin=args.fmin,
            fmax=args.fmax,
            periodicity_threshold=args.periodicity_threshold,
            device=args.device,
        ),
        fusion=FusionConfig(
            max_gap_fill_frames=args.gap_fill,
        ),
        output_path=args.output,
    )

    pipeline = PitchVADPipeline(config)

    try:
        result = pipeline.run(args.audio)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Summary
    print(f"\n{'─'*50}")
    print(f"  Frames:        {len(result.timestamps)}")
    print(f"  Voiced frames: {int(np.sum(result.voiced_mask))}")
    print(f"  Voiced ratio:  {result.voiced_ratio():.1%}")
    print(f"  Voiced dur:    {result.voiced_duration():.2f}s")
    print(f"  Output JSON:   {args.output}")
    print(f"{'─'*50}\n")

    if np.sum(result.voiced_mask) > 0:
        voiced_f0 = result.f0[result.voiced_mask & (result.f0 > 0)]
        if len(voiced_f0) > 0:
            print(f"  F0 min:    {np.min(voiced_f0):.1f} Hz")
            print(f"  F0 max:    {np.max(voiced_f0):.1f} Hz")
            print(f"  F0 median: {np.median(voiced_f0):.1f} Hz")
            print(f"  F0 mean:   {np.mean(voiced_f0):.1f} Hz")

    if args.visualize:
        try:
            from visualization.pitch_viz import plot_pitch_vad_combined
            base = Path(args.output).stem
            plot_pitch_vad_combined(
                audio=result.audio,
                sr=result.sample_rate,
                pitch_times=result.timestamps,
                f0=result.f0,
                voiced_mask=result.voiced_mask,
                vad_times=result.vad_times_raw,
                vad_mask=result.vad_mask_raw,
                save_path=f"{base}_vad_pitch.png",
            )
        except Exception as exc:
            print(f"[Warning] Visualization failed: {exc}")


if __name__ == "__main__":
    main()
