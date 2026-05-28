"""
inference/run_pipeline.py - CLI for the unified inference pipeline.

Single file (basic):
    python inference/run_pipeline.py --audio singing.wav

Full pipeline with reference alignment, metrics, and scoring:
    python inference/run_pipeline.py --audio singing.wav \
        --musicxml references/song.musicxml \
        --textgrid references/song.TextGrid \
        --compute-metrics --compute-scores --export-json --plot

Batch directory:
    python inference/run_pipeline.py --input_dir ./dataset/ --export-json

Options:
    --audio, -a          Single WAV file path
    --input_dir, -d      Directory of WAV files (batch mode)
    --musicxml           Path to MusicXML reference score (enables alignment)
    --textgrid           Path to Praat TextGrid reference (enables alignment)
    --pattern            Glob pattern for batch mode (default: *.wav)
    --output_dir, -o     Directory for output JSON files (default: outputs/)
    --config             Path to pipeline YAML config
    --device             auto | cpu | cuda | cuda:N | mps
    --no-pitch           Disable pitch+VAD
    --no-phoneme         Disable phoneme detection
    --no-onset-offset    Disable onset/offset detection
    --compute-metrics    Compute PerformanceMetricsReport after alignment
    --compute-scores     Compute PerformanceScoreReport + InterpretationSummary
    --export-json        Write per-result JSON to output_dir
    --plot               Generate and save visualization plots
    --skip-errors        Continue batch on per-file errors (default: True)
    --verbose, -v        Enable DEBUG logging
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VocalCoach unified inference pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", "-a", help="Single WAV file.")
    source.add_argument("--input_dir", "-d", help="Directory of WAV files (batch).")

    parser.add_argument("--pattern", default="*.wav",
                        help="Glob pattern for batch mode.")
    parser.add_argument("--output_dir", "-o", default="outputs/",
                        help="Output directory for JSON results.")
    parser.add_argument("--config", default=None,
                        help="Path to pipeline YAML config.")
    parser.add_argument("--device", default=None,
                        help="Device override: auto | cpu | cuda.")
    parser.add_argument("--no-pitch", action="store_true",
                        help="Disable pitch+VAD module.")
    parser.add_argument("--no-phoneme", action="store_true",
                        help="Disable phoneme detection module.")
    parser.add_argument("--no-onset-offset", action="store_true",
                        help="Disable onset/offset detection module.")
    parser.add_argument("--musicxml", default=None,
                        help="Path to MusicXML reference score (enables reference alignment).")
    parser.add_argument("--textgrid", default=None,
                        help="Path to Praat TextGrid reference (enables reference alignment).")
    parser.add_argument("--compute-metrics", action="store_true",
                        help="Compute PerformanceMetricsReport after reference alignment.")
    parser.add_argument("--compute-scores", action="store_true",
                        help="Compute PerformanceScoreReport and InterpretationSummary.")
    parser.add_argument("--export-json", action="store_true",
                        help="Write per-result JSON to --output_dir.")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save visualization plots to --output_dir.")
    parser.add_argument("--skip-errors", action="store_true", default=True,
                        help="Continue batch on individual file errors.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging.")
    return parser


def build_pipeline(args):
    """Construct pipeline from CLI args, applying overrides."""
    from inference.pipeline import UnifiedInferencePipeline

    if args.config:
        pipeline = UnifiedInferencePipeline.from_config_file(args.config)
    else:
        pipeline = UnifiedInferencePipeline()

    # Apply CLI overrides
    overrides: dict = {"pipeline": {}}
    if args.no_pitch:
        overrides["pipeline"]["enable_pitch"] = False
    if args.no_phoneme:
        overrides["pipeline"]["enable_phoneme"] = False
    if getattr(args, "no_onset_offset", False):
        overrides["pipeline"]["enable_onset_offset"] = False
    if args.device:
        overrides["pipeline"]["device"] = args.device
    if overrides["pipeline"]:
        # Re-build with merged overrides (cheap — no models loaded yet)
        from configs.loader import merge_configs
        merged = merge_configs(pipeline._cfg, overrides)
        pipeline = UnifiedInferencePipeline(merged)

    return pipeline


def export_result(result, output_dir: Path) -> Path:
    """Write a UnifiedInferenceResult to a JSON file in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.audio_path).stem
    out_path = output_dir / f"{stem}_unified.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(result.to_dict(), fp, indent=2)
    return out_path


def print_summary(result) -> None:
    """Print a concise human-readable summary to stdout."""
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  File:      {Path(result.audio_path).name}")
    print(f"  Duration:  {result.duration_s:.2f}s")
    print(f"  Device:    {result.metadata.get('device', '?')}")
    print(f"  Elapsed:   {result.metadata.get('elapsed_s', 0):.2f}s")
    print(sep)

    if result.has_pitch():
        import numpy as np
        voiced_f0 = result.f0[result.voiced & (result.f0 > 0)] if result.voiced is not None else result.f0[result.f0 > 0]
        voiced_pct = float(np.mean(result.voiced)) * 100 if result.voiced is not None else 0.0
        print(f"  Pitch:     {len(result.f0)} frames, {voiced_pct:.1f}% voiced")
        if len(voiced_f0) > 0:
            print(f"             F0 {np.min(voiced_f0):.0f}–{np.max(voiced_f0):.0f} Hz "
                  f"(med {np.median(voiced_f0):.0f} Hz)")

    if result.has_phonemes():
        print(f"  Phonemes:  {len(result.phoneme_segments)} segments")

    if result.has_onset_offset():
        n_notes = len(result.note_events) if result.note_events else 0
        print(f"  Notes:     {n_notes} detected")

    if result.aligned is not None:
        print(f"  Aligned:   {result.aligned.n_frames} canonical frames (100fps)")

    if result.alignment is not None:
        n_matched = len(result.alignment.note_matches) if result.alignment.note_matches else 0
        print(f"  Ref align: {n_matched} note matches")

    if result.metrics is not None:
        m = result.metrics
        if m.pitch is not None:
            acc = m.pitch.pitch_accuracy
            mace = m.pitch.mace_cents
            acc_str  = f"{acc * 100:.1f}%" if acc is not None else "N/A"
            mace_str = f"{mace:.1f}¢"      if mace is not None else "N/A"
            print(f"  Pitch acc: {acc_str}  MACE {mace_str}")
        if m.timing is not None:
            acc = m.timing.timing_accuracy
            mae = m.timing.mean_abs_onset_error_ms
            acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
            mae_str = f"{mae:.1f}ms"      if mae is not None else "N/A"
            print(f"  Timing:    {acc_str}  MAE {mae_str}")

    if result.scores is not None:
        s = result.scores
        overall = f"{s.overall_score:.1f}" if s.overall_score is not None else "N/A"
        print(f"  Score:     {overall}/100  ", end="")
        cats = []
        for cat_name in ("pitch_score", "timing_score", "duration_score", "lyric_score"):
            cat = getattr(s, cat_name, None)
            if cat is not None:
                cats.append(f"{cat.category}={cat.score:.0f}")
        print("  ".join(cats))

    if result.interpretation is not None:
        i = result.interpretation
        print(f"  Level:     {i.overall_level.upper()}")
        if i.strengths:
            print(f"  Strengths: {i.strengths[0]}")
        if i.weaknesses:
            print(f"  Improve:   {i.weaknesses[0]}")

    print(sep)


def _save_plots(result, output_dir: Path) -> None:
    """Generate and save scoring visualization plots when scores are available."""
    try:
        from visualization.scoring_viz import plot_performance_dashboard
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(result.audio_path).stem
        save_path = output_dir / f"{stem}_dashboard.png"
        plot_performance_dashboard(result.scores, save_path=str(save_path))
        print(f"  Plot:      {save_path}")
    except Exception as exc:
        print(f"  [warn] Plot generation failed: {exc}", file=sys.stderr)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")

    pipeline = build_pipeline(args)
    output_dir = Path(args.output_dir)
    export = args.export_json

    compute_metrics = getattr(args, "compute_metrics", False)
    compute_scores = getattr(args, "compute_scores", False)
    do_plot = getattr(args, "plot", False)

    if args.audio:
        # Single-file mode
        try:
            result = pipeline.predict(
                args.audio,
                musicxml_path=args.musicxml,
                textgrid_path=args.textgrid,
                compute_metrics=compute_metrics,
                compute_scores=compute_scores,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        print_summary(result)

        if export:
            out_path = export_result(result, output_dir)
            print(f"  JSON:      {out_path}")

        if do_plot and result.scores is not None:
            _save_plots(result, output_dir)

    else:
        # Batch mode
        results = pipeline.predict_directory(
            args.input_dir,
            pattern=args.pattern,
            skip_errors=args.skip_errors,
        )
        if not results:
            print("No files processed.", file=sys.stderr)
            sys.exit(1)

        for r in results:
            print_summary(r)
            if export:
                out_path = export_result(r, output_dir)
                print(f"  JSON: {out_path}")
            if do_plot and r.scores is not None:
                _save_plots(r, output_dir)

        print(f"\nProcessed {len(results)} file(s).")


if __name__ == "__main__":
    main()
