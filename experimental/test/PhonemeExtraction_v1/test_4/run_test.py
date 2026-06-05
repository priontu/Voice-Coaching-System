#!/usr/bin/env python
"""test_4: Per-song phoneme timeline plots – ground truth vs. prediction.

One figure per song, four subplots (one per recording clip: 0000–0003).
Each subplot shows two horizontal lanes:
  GT   – reference phonemes from the TextGrid (ARPAbet labels)
  Pred – model-predicted phonemes (IPA labels)

Rectangles are coloured by boundary alignment quality:
  green  – boundary error < 0.10 s
  orange – boundary error 0.10–0.30 s
  red    – boundary error > 0.30 s
  grey   – phoneme beyond the aligned region (no counterpart)

No model inference required – reads existing test_1 outputs.

Usage
-----
    python run_test.py
    python run_test.py --summary-json PATH --output-dir DIR --max-plot-sec 25
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from voice_coach.data.textgrid import phoneme_intervals
from PhonemeExtraction_v1.eval import is_scored_phone

DEFAULT_SUMMARY_JSON = (
    Path(__file__).resolve().parents[1] / "test_1" / "outputs" / "summary.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_GREEN  = "#4caf50"
_ORANGE = "#ff9800"
_RED    = "#f44336"
_GREY   = "#9e9e9e"


def _color(boundary_error_sec: float) -> str:
    if boundary_error_sec < 0.10:
        return _GREEN
    if boundary_error_sec < 0.30:
        return _ORANGE
    return _RED


# ---------------------------------------------------------------------------
# Core drawing routine – one recording per axis
# ---------------------------------------------------------------------------

def _draw_recording(
    ax,
    *,
    ref_intervals: list,
    pred_phonemes: list[dict],
    matches: list[dict],
    title: str,
    max_sec: float,
    Rectangle,
) -> None:
    GT_BOT,   GT_H   = 1.05, 0.85
    PRED_BOT, PRED_H = 0.10, 0.85

    n_matched = len(matches)

    # --- GT lane ---
    for idx, interval in enumerate(ref_intervals):
        start = float(interval.start_sec)
        if start >= max_sec:
            break
        end   = min(float(interval.end_sec), max_sec)
        width = max(end - start, 1e-4)
        if idx < n_matched:
            color = _color(float(matches[idx]["boundary_error_sec"]))
        else:
            color = _GREY
        ax.add_patch(Rectangle((start, GT_BOT), width, GT_H,
                                color=color, alpha=0.55, linewidth=0))
        ax.axvline(x=start, color="white", linewidth=0.4, alpha=0.6)
        if width > 0.08:
            ax.text(start + width / 2, GT_BOT + GT_H / 2,
                    interval.text, ha="center", va="center",
                    fontsize=6, color="black", clip_on=True)

    # --- Pred lane ---
    for idx, phoneme in enumerate(pred_phonemes):
        start = float(phoneme["start_sec"])
        if start >= max_sec:
            break
        end   = min(float(phoneme["end_sec"]), max_sec)
        width = max(end - start, 1e-4)
        if idx < n_matched:
            color = _color(float(matches[idx]["boundary_error_sec"]))
        else:
            color = _GREY
        ax.add_patch(Rectangle((start, PRED_BOT), width, PRED_H,
                                color=color, alpha=0.85, linewidth=0))
        ax.axvline(x=start, color="white", linewidth=0.4, alpha=0.6)
        if width > 0.08:
            ax.text(start + width / 2, PRED_BOT + PRED_H / 2,
                    phoneme["phoneme"], ha="center", va="center",
                    fontsize=6, color="white", clip_on=True)

    # Lane labels
    ax.text(-0.01, GT_BOT   + GT_H   / 2, "GT",
            ha="right", va="center", fontsize=7,
            transform=ax.get_yaxis_transform())
    ax.text(-0.01, PRED_BOT + PRED_H / 2, "Pred",
            ha="right", va="center", fontsize=7,
            transform=ax.get_yaxis_transform())

    ax.set_xlim(0, max_sec)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(axis="x", labelsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GT vs. predicted phoneme timelines, grouped by song.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--output-dir",   type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi",          type=int,  default=150)
    parser.add_argument("--max-plot-sec", type=float, default=20.0,
                        help="Seconds of audio shown per subplot.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    if not args.summary_json.exists():
        raise FileNotFoundError(
            f"summary.json not found: {args.summary_json}\n"
            "Run test_1 first, or pass --summary-json with the correct path."
        )

    summary   = json.loads(args.summary_json.read_text())
    dataset_root = Path(summary["dataset_root"])

    plot_dir  = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Group recordings by song title (third path component under dataset_root)
    songs: dict[str, list[dict]] = defaultdict(list)
    for rec in summary["recordings"]:
        try:
            song_title = Path(rec["wav_path"]).relative_to(dataset_root).parts[1]
        except (ValueError, IndexError):
            song_title = "unknown"
        songs[song_title].append(rec)

    legend_patches = [
        Patch(facecolor=_GREEN,  alpha=0.75, label="boundary error < 0.10 s"),
        Patch(facecolor=_ORANGE, alpha=0.75, label="0.10 – 0.30 s"),
        Patch(facecolor=_RED,    alpha=0.75, label="> 0.30 s"),
        Patch(facecolor=_GREY,   alpha=0.75, label="unmatched"),
    ]

    saved: list[str] = []
    for song_title, recordings in sorted(songs.items()):
        recordings = sorted(recordings, key=lambda r: r["recording_id"])
        n = len(recordings)

        fig, axes = plt.subplots(
            n, 1, figsize=(16, 2.4 * n + 0.8),
            squeeze=False,
        )
        technique = Path(recordings[0]["wav_path"]).relative_to(dataset_root).parts[0]
        fig.suptitle(
            f'"{song_title}"  —  {technique}',
            fontsize=10, y=1.0,
        )

        for row_idx, rec in enumerate(recordings):
            ax = axes[row_idx][0]

            # Load ground truth
            ref_all = phoneme_intervals(Path(rec["textgrid_path"]))
            ref_scored = [i for i in ref_all if is_scored_phone(i.text)]

            # Load prediction
            pred_data = json.loads(Path(rec["prediction_path"]).read_text())
            pred_phonemes = pred_data["phonemes"]

            # Load matches
            matches: list[dict] = []
            matches_path = Path(rec["matches_path"])
            if matches_path.exists():
                with matches_path.open() as fh:
                    matches = list(csv.DictReader(fh))

            # Determine x-axis extent
            last_ref  = max((float(i.end_sec)  for i in ref_scored),  default=0.0)
            last_pred = max(
                (float(p["end_sec"]) for p in pred_phonemes), default=0.0
            )
            max_sec = min(args.max_plot_sec, max(last_ref, last_pred) + 0.5)

            clip_id = rec["recording_id"].split("_")[-1]  # "0000" … "0003"
            n_ref   = rec["metrics"]["reference_count"]
            n_pred  = rec["metrics"]["predicted_count"]
            n_align = rec["metrics"]["aligned_count"]
            bmae    = rec["metrics"].get("boundary_mae_sec")
            bmae_str = f"{bmae:.3f} s" if bmae is not None else "n/a"

            _draw_recording(
                ax,
                ref_intervals=ref_scored,
                pred_phonemes=pred_phonemes,
                matches=matches,
                title=(
                    f"clip {clip_id}   "
                    f"ref={n_ref}  pred={n_pred}  aligned={n_align}   "
                    f"boundary MAE={bmae_str}"
                ),
                max_sec=max_sec,
                Rectangle=Rectangle,
            )

            if row_idx == n - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

        # Legend on the last subplot
        axes[-1][0].legend(
            handles=legend_patches, fontsize=7, loc="upper right",
            ncol=2, framealpha=0.8,
        )

        plt.tight_layout()
        safe_name = song_title.replace(" ", "_").replace("'", "")
        out_path  = plot_dir / f"{safe_name}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out_path))
        print(f"  saved {out_path.name}  ({n} clips)")

    manifest = {
        "summary_json": str(args.summary_json),
        "max_plot_sec": args.max_plot_sec,
        "plots": saved,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nPlots: {plot_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
