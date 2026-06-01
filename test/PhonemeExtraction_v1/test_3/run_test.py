#!/usr/bin/env python
"""test_3: Aggregate analysis plots comparing predicted phoneme boundaries to ground truth.

Reads existing evaluation outputs (summary.json + matches CSVs) from a prior
test run. No model inference — no GPU required.

Plots produced
--------------
1. timing_scatter.png       Predicted vs. reference start/end times (scatter).
2. error_distributions.png  Histograms of start / end / boundary errors.
3. per_recording_mae.png    Boundary MAE per recording, coloured by technique.
4. phoneme_errors.png       Mean boundary error for the top-N most common
                             reference phonemes.
5. error_by_technique.png   Box plot of boundary errors grouped by singing
                             technique.

Usage
-----
    python run_test.py
    python run_test.py --summary-json PATH --output-dir DIR --dpi 150
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_SUMMARY_JSON = (
    Path(__file__).resolve().parents[1] / "test_1" / "outputs" / "summary.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _technique(wav_path: str, dataset_root: Path) -> str:
    try:
        return Path(wav_path).relative_to(dataset_root).parts[0]
    except (ValueError, IndexError):
        return "unknown"


def _normalize(phone: str) -> str:
    phone = phone.lower().strip().replace("ː", "").replace(":", "")
    return "".join(ch for ch in phone if not ch.isdigit())


def _short_id(recording_id: str) -> str:
    return recording_id.replace("_Control_Group", "")


# ---------------------------------------------------------------------------
# Plot 1: Timing scatter – predicted vs. reference start/end times
# ---------------------------------------------------------------------------

def _plot_timing_scatter(pairs: list[dict], path: Path, dpi: int, plt, np) -> None:
    match_mask = np.array([p["label_match"] for p in pairs])
    ref_starts = np.array([float(p["reference_start_sec"]) for p in pairs])
    pred_starts = np.array([float(p["predicted_start_sec"]) for p in pairs])
    ref_ends = np.array([float(p["reference_end_sec"]) for p in pairs])
    pred_ends = np.array([float(p["predicted_end_sec"]) for p in pairs])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Predicted vs. Reference Phoneme Boundaries", fontsize=11)

    for ax, ref_vals, pred_vals, title in [
        (axes[0], ref_starts, pred_starts, "Start Time"),
        (axes[1], ref_ends, pred_ends, "End Time"),
    ]:
        ax.scatter(
            ref_vals[match_mask], pred_vals[match_mask],
            color="#4caf50", alpha=0.5, s=14, rasterized=True,
            label=f"label match  (n={match_mask.sum()})",
        )
        ax.scatter(
            ref_vals[~match_mask], pred_vals[~match_mask],
            color="#f44336", alpha=0.35, s=10, rasterized=True,
            label=f"label mismatch  (n={(~match_mask).sum()})",
        )
        lo = min(ref_vals.min(), pred_vals.min()) - 0.3
        hi = max(ref_vals.max(), pred_vals.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.55, label="y = x  (perfect)")
        corr = float(np.corrcoef(ref_vals, pred_vals)[0, 1])
        ax.set_title(f"{title}  (r = {corr:.3f})", fontsize=10)
        ax.set_xlabel("Reference (s)", fontsize=9)
        ax.set_ylabel("Predicted (s)", fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 2: Error distributions – histograms of absolute errors
# ---------------------------------------------------------------------------

def _plot_error_distributions(pairs: list[dict], path: Path, dpi: int, plt, np) -> None:
    fields = [
        ("start_error_sec",    "Start Error (s)",    "#2196f3"),
        ("end_error_sec",      "End Error (s)",      "#ff9800"),
        ("boundary_error_sec", "Boundary Error (s)", "#9c27b0"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Phoneme Boundary Error Distributions (all aligned pairs)", fontsize=11)

    for ax, (field, label, color) in zip(axes, fields):
        vals = np.array([float(p[field]) for p in pairs])
        ax.hist(vals, bins=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
        mean_v = float(vals.mean())
        med_v = float(np.median(vals))
        ax.axvline(mean_v,  color="red",   linewidth=1.4, linestyle="--",
                   label=f"mean = {mean_v:.3f} s")
        ax.axvline(med_v,   color="black", linewidth=1.4, linestyle=":",
                   label=f"median = {med_v:.3f} s")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 3: Per-recording boundary MAE, coloured by singing technique
# ---------------------------------------------------------------------------

def _plot_per_recording_mae(
    recordings: list[dict], dataset_root: Path,
    path: Path, dpi: int, plt, np,
) -> None:
    techniques = sorted({_technique(r["wav_path"], dataset_root) for r in recordings})
    palette = plt.cm.tab10.colors
    tech_color = {t: palette[i % len(palette)] for i, t in enumerate(techniques)}

    recs = sorted(recordings, key=lambda r: r["metrics"].get("boundary_mae_sec") or 0, reverse=True)
    labels  = [_short_id(r["recording_id"]) for r in recs]
    b_maes  = [r["metrics"].get("boundary_mae_sec") or 0.0 for r in recs]
    colors  = [tech_color[_technique(r["wav_path"], dataset_root)] for r in recs]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.42 + 1.2)))
    fig.suptitle("Boundary MAE per Recording (sorted)", fontsize=11)

    y = range(n)
    bars = ax.barh(y, b_maes, color=colors, alpha=0.85, edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, b_maes):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Boundary MAE (s)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=tech_color[t], alpha=0.85, label=t) for t in techniques]
    ax.legend(handles=legend_patches, fontsize=8, title="Technique", loc="lower right")

    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 4: Mean boundary error per reference phoneme (top N by frequency)
# ---------------------------------------------------------------------------

def _plot_phoneme_errors(
    pairs: list[dict], path: Path, dpi: int, top_n: int, plt, np,
) -> None:
    phone_errors: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        phone = _normalize(p["reference_phoneme"])
        if phone and phone not in {"<sp>", "<ap>", "sp", "ap", "sil", "silence"}:
            phone_errors[phone].append(float(p["boundary_error_sec"]))

    # Keep the top_n most frequent reference phonemes
    sorted_by_count = sorted(phone_errors.items(), key=lambda kv: len(kv[1]), reverse=True)
    top = sorted_by_count[:top_n]
    # Sort by mean error (descending) for display
    top_sorted = sorted(top, key=lambda kv: float(np.mean(kv[1])), reverse=True)

    phones = [kv[0] for kv in top_sorted]
    means  = [float(np.mean(kv[1])) for kv in top_sorted]
    stds   = [float(np.std(kv[1]))  for kv in top_sorted]
    counts = [len(kv[1])            for kv in top_sorted]

    n = len(phones)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.38 + 1.2)))
    fig.suptitle(
        f"Mean Boundary Error by Reference Phoneme  (top {top_n} most frequent)",
        fontsize=10,
    )

    y = range(n)
    ax.barh(y, means, xerr=stds, color="#5c6bc0", alpha=0.8,
            error_kw={"linewidth": 0.8, "capsize": 2}, edgecolor="white", linewidth=0.3)
    for i, (mean_v, count) in enumerate(zip(means, counts)):
        ax.text(mean_v + max(stds) * 0.05, i,
                f"  n={count}", va="center", fontsize=7, color="#333")

    ax.set_yticks(list(y))
    ax.set_yticklabels(phones, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Mean Boundary Error (s)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 5: Boundary error distribution by singing technique (box plot)
# ---------------------------------------------------------------------------

def _plot_error_by_technique(
    pairs: list[dict], path: Path, dpi: int, plt, np,
) -> None:
    tech_errors: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        tech_errors[p["technique"]].append(float(p["boundary_error_sec"]))

    techniques = sorted(tech_errors)
    data = [tech_errors[t] for t in techniques]
    counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(max(7, len(techniques) * 1.6), 5))
    fig.suptitle("Boundary Error Distribution by Singing Technique", fontsize=11)

    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    palette = plt.cm.tab10.colors
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(techniques) + 1))
    ax.set_xticklabels(
        [f"{t}\n(n={counts[i]})" for i, t in enumerate(techniques)],
        fontsize=8,
    )
    ax.set_ylabel("Boundary Error (s)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)

    # Annotate mean above each box
    for i, d in enumerate(data, start=1):
        mean_v = float(np.mean(d))
        ax.text(i, ax.get_ylim()[1] * 0.97, f"μ={mean_v:.3f}",
                ha="center", va="top", fontsize=7, color="#333")

    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GT-vs-prediction analysis plots for PhonemeExtraction_v1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON,
        help="Path to summary.json produced by a prior test run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots/ and manifest.json are written.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--top-n-phonemes", type=int, default=20,
        help="Number of most-frequent reference phonemes shown in plot 4.",
    )
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not args.summary_json.exists():
        raise FileNotFoundError(
            f"summary.json not found: {args.summary_json}\n"
            "Run test_1 first, or pass --summary-json with the correct path."
        )

    summary = json.loads(args.summary_json.read_text())
    dataset_root = Path(summary["dataset_root"])
    recordings = summary["recordings"]

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load all aligned phoneme pairs from the matches CSVs
    all_pairs: list[dict] = []
    missing = 0
    for recording in recordings:
        matches_path = Path(recording["matches_path"])
        if not matches_path.exists():
            print(f"  [warn] missing {matches_path.name}")
            missing += 1
            continue
        technique = _technique(recording["wav_path"], dataset_root)
        with matches_path.open() as fh:
            for row in csv.DictReader(fh):
                row["recording_id"] = recording["recording_id"]
                row["technique"] = technique
                row["label_match"] = (
                    _normalize(row["predicted_phoneme"]) == _normalize(row["reference_phoneme"])
                )
                all_pairs.append(row)

    if not all_pairs:
        raise RuntimeError("No match rows loaded. Check --summary-json path.")

    n_recs = len(recordings) - missing
    print(f"Loaded {len(all_pairs)} aligned pairs from {n_recs}/{len(recordings)} recordings.\n")

    plots: dict[str, str] = {}

    print("Generating plots ...")
    p = plot_dir / "timing_scatter.png"
    _plot_timing_scatter(all_pairs, p, args.dpi, plt, np)
    plots["timing_scatter"] = str(p)

    p = plot_dir / "error_distributions.png"
    _plot_error_distributions(all_pairs, p, args.dpi, plt, np)
    plots["error_distributions"] = str(p)

    p = plot_dir / "per_recording_mae.png"
    _plot_per_recording_mae(recordings, dataset_root, p, args.dpi, plt, np)
    plots["per_recording_mae"] = str(p)

    p = plot_dir / "phoneme_errors.png"
    _plot_phoneme_errors(all_pairs, p, args.dpi, args.top_n_phonemes, plt, np)
    plots["phoneme_errors"] = str(p)

    p = plot_dir / "error_by_technique.png"
    _plot_error_by_technique(all_pairs, p, args.dpi, plt, np)
    plots["error_by_technique"] = str(p)

    manifest = {
        "summary_json": str(args.summary_json),
        "num_recordings": len(recordings),
        "num_aligned_pairs": len(all_pairs),
        "plots": plots,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nAll plots saved to: {plot_dir}")
    print(f"Manifest:           {manifest_path}")


if __name__ == "__main__":
    main()
