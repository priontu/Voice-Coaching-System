"""
scripts/batch_evaluate.py - Batch evaluation with optional reference alignment.

Run from the VocalCoach root:
    py scripts/batch_evaluate.py --input-dir dataset/ --output-dir results/

With reference alignment:
    py scripts/batch_evaluate.py \
        --input-dir dataset/audio/ \
        --reference-dir dataset/references/ \
        --compute-metrics --compute-scores --export-json

Reference directory convention:
    The script looks for MusicXML and TextGrid files in --reference-dir that
    share the same stem as each audio file. Example:
        audio/  01_aria.wav         →  references/01_aria.musicxml
                                       references/01_aria.TextGrid
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def _find_reference(stem: str, ref_dir: Optional[Path]):
    """Locate MusicXML and/or TextGrid files matching audio stem in ref_dir."""
    if ref_dir is None:
        return None, None
    musicxml = None
    textgrid = None
    for ext in (".musicxml", ".xml", ".mxl"):
        p = ref_dir / f"{stem}{ext}"
        if p.exists():
            musicxml = p
            break
    for ext in (".TextGrid", ".textgrid"):
        p = ref_dir / f"{stem}{ext}"
        if p.exists():
            textgrid = p
            break
    return musicxml, textgrid


def _print_row(label: str, value: str) -> None:
    print(f"  {label:<22} {value}")


def _format_score(report) -> str:
    if report is None:
        return "—"
    if report.overall_score is None:
        return "N/A"
    cats = []
    for attr in ("pitch_score", "timing_score", "duration_score", "lyric_score"):
        cat = getattr(report, attr, None)
        if cat is not None:
            cats.append(f"{cat.category[:3]}={cat.score:.0f}")
    cat_str = "  ".join(cats)
    return f"{report.overall_score:.1f}/100  [{cat_str}]"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch evaluate a directory of audio files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input-dir",  "-i", required=True,
                    help="Directory containing WAV audio files.")
    ap.add_argument("--reference-dir", default=None,
                    help="Directory containing matching MusicXML / TextGrid files.")
    ap.add_argument("--output-dir",  "-o", default="outputs/",
                    help="Directory for output JSON files.")
    ap.add_argument("--pattern",      default="*.wav",
                    help="Glob pattern for audio files.")
    ap.add_argument("--compute-metrics", action="store_true",
                    help="Compute PerformanceMetricsReport after alignment.")
    ap.add_argument("--compute-scores",  action="store_true",
                    help="Compute PerformanceScoreReport + interpretation.")
    ap.add_argument("--export-json",    action="store_true",
                    help="Write per-file JSON results to --output-dir.")
    ap.add_argument("--config",         default=None,
                    help="Pipeline YAML config override path.")
    ap.add_argument("--device",         default=None,
                    help="Device: auto | cpu | cuda | cuda:N")
    ap.add_argument("--skip-errors",    action="store_true", default=True)
    ap.add_argument("--verbose", "-v",  action="store_true")
    args = ap.parse_args()

    from utils.logging_utils import setup_logging
    setup_logging("DEBUG" if args.verbose else "INFO")

    # Build pipeline
    from inference.pipeline import UnifiedInferencePipeline
    from configs.loader import merge_configs

    overrides: Dict = {}
    if args.device:
        overrides = {"pipeline": {"device": args.device}}
    if args.config:
        pipeline = UnifiedInferencePipeline.from_config_file(args.config)
        if overrides:
            from configs.loader import merge_configs
            pipeline = UnifiedInferencePipeline(merge_configs(pipeline._cfg, overrides))
    else:
        pipeline = UnifiedInferencePipeline.from_dict(overrides) if overrides else UnifiedInferencePipeline()

    input_dir  = Path(args.input_dir)
    ref_dir    = Path(args.reference_dir) if args.reference_dir else None
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: --input-dir '{input_dir}' is not a directory.", file=sys.stderr)
        return 1

    audio_files = sorted(input_dir.glob(args.pattern))
    if not audio_files:
        print(f"No files matching '{args.pattern}' in {input_dir}.", file=sys.stderr)
        return 1

    print(f"\nBatch evaluation: {len(audio_files)} file(s) in {input_dir}")
    sep = "─" * 60
    print(sep)

    summaries: List[Dict] = []
    t_batch = time.perf_counter()

    for i, audio_path in enumerate(audio_files, 1):
        stem = audio_path.stem
        musicxml, textgrid = _find_reference(stem, ref_dir)

        print(f"\n[{i}/{len(audio_files)}] {audio_path.name}")
        if musicxml:
            print(f"  MusicXML:  {musicxml.name}")
        if textgrid:
            print(f"  TextGrid:  {textgrid.name}")

        try:
            result = pipeline.predict(
                audio_path,
                musicxml_path=musicxml,
                textgrid_path=textgrid,
                compute_metrics=args.compute_metrics,
                compute_scores=args.compute_scores,
            )

            elapsed = result.metadata.get("elapsed_s", 0)
            print(f"  Elapsed:   {elapsed:.2f}s")

            if result.scores is not None:
                print(f"  Score:     {_format_score(result.scores)}")
            if result.interpretation is not None:
                print(f"  Level:     {result.interpretation.overall_level.upper()}")

            if args.export_json:
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f"{stem}_eval.json"
                with open(out_path, "w", encoding="utf-8") as fp:
                    json.dump(result.to_dict(), fp, indent=2)
                print(f"  JSON:      {out_path}")

            summaries.append({
                "file": audio_path.name,
                "elapsed_s": elapsed,
                "overall_score": result.scores.overall_score if result.scores else None,
                "level": result.interpretation.overall_level if result.interpretation else None,
            })

        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            if not args.skip_errors:
                raise
            summaries.append({"file": audio_path.name, "error": str(exc)})

    total_elapsed = time.perf_counter() - t_batch

    # Summary table
    print(f"\n{sep}")
    print(f"  Batch complete — {len(audio_files)} file(s) in {total_elapsed:.1f}s")
    print(sep)
    scores = [s["overall_score"] for s in summaries if s.get("overall_score") is not None]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  Average score:  {avg:.1f}/100")
        print(f"  Score range:    {min(scores):.1f} – {max(scores):.1f}")
    print(sep)

    # Write aggregate summary
    if args.export_json:
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump({"files": summaries, "total_elapsed_s": total_elapsed}, fp, indent=2)
        print(f"  Summary JSON:   {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
