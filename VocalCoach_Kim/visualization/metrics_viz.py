"""
visualization/metrics_viz.py - Metric visualization for Phase 6 evaluation results.

All plots use matplotlib with the non-interactive Agg backend so they can
run headlessly on any machine (no display required). Call save_path to write
PNG/PDF; omit it to get a Figure object for interactive inspection.

Entry points:
  plot_pitch_deviation      Per-note pitch deviation bar chart
  plot_onset_deviation      Per-note onset timing error bar chart
  plot_duration_comparison  Predicted vs. reference duration scatter
  plot_phoneme_timing       Per-phoneme boundary error bar chart
  plot_metric_heatmap       Pitch + timing + duration heatmap across notes
  plot_metrics_summary      Five-panel summary figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.types import (
        AlignmentResult,
        FusedPerformanceRepresentation,
        PerformanceMetricsReport,
        ReferencePerformanceRepresentation,
    )


# ---------------------------------------------------------------------------
# Backend guard
# ---------------------------------------------------------------------------

def _require_mpl():
    """Import matplotlib, force non-interactive Agg backend, return pyplot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_pitch_deviation(
    report: "PerformanceMetricsReport",
    tolerance_cents: Optional[float] = None,
    figsize: tuple = (10, 4),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Bar chart of per-note pitch deviation in cents.

    Bars are colored green (within tolerance) / red (outside) when tolerance_cents
    is provided. A dashed horizontal zero line is drawn for reference.
    """
    plt = _require_mpl()

    if report.pitch is None or not report.pitch.per_note:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No pitch data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(title or "Pitch Deviation")
        _save_or_return(fig, save_path)
        return fig

    breakdowns = report.pitch.per_note
    indices = [b.event_idx for b in breakdowns]
    values = [b.value if b.value is not None else 0.0 for b in breakdowns]
    tol = tolerance_cents or report.pitch.tolerance_cents

    colors = [
        "seagreen" if abs(v) <= tol else "crimson"
        for v in values
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(tol, color="gray", linewidth=0.8, linestyle="--", label=f"+{tol:.0f}¢")
    ax.axhline(-tol, color="gray", linewidth=0.8, linestyle="--", label=f"−{tol:.0f}¢")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(i) for i in indices], fontsize=7, rotation=45)
    ax.set_xlabel("Predicted note index")
    ax.set_ylabel("Pitch deviation (cents)")
    ax.set_title(title or "Per-Note Pitch Deviation")
    ax.legend(fontsize=8)

    mace = report.pitch.mace_cents
    if mace is not None:
        ax.text(0.02, 0.96, f"MACE={mace:.1f}¢", transform=ax.transAxes,
                fontsize=9, va="top", color="navy")

    fig.tight_layout()
    _save_or_return(fig, save_path)
    return fig


def plot_onset_deviation(
    report: "PerformanceMetricsReport",
    tolerance_ms: Optional[float] = None,
    figsize: tuple = (10, 4),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Bar chart of per-note onset timing deviation in milliseconds.

    Blue = early (negative), red = late (positive). Gray dashed lines show
    tolerance bounds.
    """
    plt = _require_mpl()

    if report.timing is None or not report.timing.per_note:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(title or "Onset Deviation")
        _save_or_return(fig, save_path)
        return fig

    breakdowns = report.timing.per_note
    indices = [b.event_idx for b in breakdowns]
    values = [b.value if b.value is not None else 0.0 for b in breakdowns]
    tol = tolerance_ms or report.timing.tolerance_ms

    colors = ["steelblue" if v < 0 else "tomato" for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(tol, color="gray", linewidth=0.8, linestyle="--", label=f"+{tol:.0f} ms")
    ax.axhline(-tol, color="gray", linewidth=0.8, linestyle="--", label=f"−{tol:.0f} ms")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(i) for i in indices], fontsize=7, rotation=45)
    ax.set_xlabel("Predicted note index")
    ax.set_ylabel("Onset deviation (ms)  [+ = late]")
    ax.set_title(title or "Per-Note Onset Timing Deviation")
    ax.legend(fontsize=8)

    mae = report.timing.mean_abs_onset_error_ms
    if mae is not None:
        ax.text(0.02, 0.96, f"MAE={mae:.1f} ms", transform=ax.transAxes,
                fontsize=9, va="top", color="navy")

    fig.tight_layout()
    _save_or_return(fig, save_path)
    return fig


def plot_duration_comparison(
    report: "PerformanceMetricsReport",
    figsize: tuple = (6, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Scatter plot of predicted duration vs. reference duration (in seconds).

    The identity line y = x represents perfect duration reproduction.
    """
    plt = _require_mpl()

    if report.duration is None or not report.duration.per_note:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No duration data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(title or "Duration Comparison")
        _save_or_return(fig, save_path)
        return fig

    breakdowns = report.duration.per_note
    pred_durs = [b.metadata.get("pred_dur_s", 0.0) for b in breakdowns]
    ref_durs = [b.metadata.get("ref_dur_s", 0.0) for b in breakdowns]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(ref_durs, pred_durs, alpha=0.7, s=40, color="steelblue",
               edgecolors="white", linewidth=0.5)

    max_dur = max(max(ref_durs, default=1.0), max(pred_durs, default=1.0)) * 1.05
    ax.plot([0, max_dur], [0, max_dur], "k--", linewidth=0.8, label="y = x (perfect)")
    ax.set_xlim(0, max_dur)
    ax.set_ylim(0, max_dur)
    ax.set_xlabel("Reference duration (s)")
    ax.set_ylabel("Predicted duration (s)")
    ax.set_title(title or "Duration Comparison: Predicted vs. Reference")
    ax.legend(fontsize=8)

    ratio = report.duration.mean_duration_ratio
    if ratio is not None:
        ax.text(0.02, 0.96, f"Mean ratio={ratio:.2f}", transform=ax.transAxes,
                fontsize=9, va="top", color="navy")

    fig.tight_layout()
    _save_or_return(fig, save_path)
    return fig


def plot_phoneme_timing(
    report: "PerformanceMetricsReport",
    tolerance_ms: Optional[float] = None,
    figsize: tuple = (10, 4),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Bar chart of per-phoneme boundary timing deviation in milliseconds.
    """
    plt = _require_mpl()

    if report.lyric is None or not report.lyric.per_phoneme:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No lyric/phoneme data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(title or "Phoneme Timing")
        _save_or_return(fig, save_path)
        return fig

    breakdowns = report.lyric.per_phoneme
    indices = [b.event_idx for b in breakdowns]
    values = [b.value if b.value is not None else 0.0 for b in breakdowns]
    tol = tolerance_ms or report.lyric.tolerance_ms

    label_colors = []
    for b, v in zip(breakdowns, values):
        if b.metadata.get("label_match", False):
            label_colors.append("seagreen" if abs(v) <= tol else "gold")
        else:
            label_colors.append("crimson")

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(values)), values, color=label_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(tol, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(-tol, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(i) for i in indices], fontsize=6, rotation=45)
    ax.set_xlabel("Predicted phoneme index")
    ax.set_ylabel("Boundary deviation (ms)  [+ = late]")
    ax.set_title(title or "Per-Phoneme Boundary Timing Deviation")

    mae = report.lyric.mean_abs_phoneme_boundary_error_ms
    if mae is not None:
        ax.text(0.02, 0.96, f"MAE={mae:.1f} ms", transform=ax.transAxes,
                fontsize=9, va="top", color="navy")

    fig.tight_layout()
    _save_or_return(fig, save_path)
    return fig


def plot_metric_heatmap(
    report: "PerformanceMetricsReport",
    figsize: tuple = (10, 4),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Heatmap with one row per metric category (pitch / timing / duration)
    and one column per matched note.

    Cell values are normalized to [−1, 1] within each row for visual
    comparability across different scales.
    """
    plt = _require_mpl()
    import math

    rows = []
    row_labels = []

    # Pitch deviation row (cents)
    if report.pitch and report.pitch.per_note:
        vals = [b.value or 0.0 for b in report.pitch.per_note]
        rows.append(vals)
        row_labels.append("Pitch (¢)")

    # Timing deviation row (ms)
    if report.timing and report.timing.per_note:
        vals = [b.value or 0.0 for b in report.timing.per_note]
        rows.append(vals)
        row_labels.append("Onset (ms)")

    # Duration error row (s)
    if report.duration and report.duration.per_note:
        vals = [b.value or 0.0 for b in report.duration.per_note]
        rows.append(vals)
        row_labels.append("Duration (s)")

    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No metric data for heatmap", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(title or "Metric Heatmap")
        _save_or_return(fig, save_path)
        return fig

    # Normalise each row to [−1, 1] by its max abs value
    n_cols = max(len(r) for r in rows)
    import numpy as np
    grid = np.zeros((len(rows), n_cols))
    for i, row in enumerate(rows):
        arr = np.array(row, dtype=float)
        max_abs = np.max(np.abs(arr)) if arr.size > 0 else 1.0
        if max_abs == 0:
            max_abs = 1.0
        grid[i, :len(arr)] = arr / max_abs

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Note index")
    ax.set_title(title or "Per-Note Metric Heatmap (normalised)")
    plt.colorbar(im, ax=ax, label="Normalised deviation")

    fig.tight_layout()
    _save_or_return(fig, save_path)
    return fig


def plot_metrics_summary(
    report: "PerformanceMetricsReport",
    figsize: tuple = (14, 12),
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Five-panel summary figure combining all metric plots.

    Panels (top→bottom, left→right):
      1. Pitch deviation bar chart
      2. Onset deviation bar chart
      3. Duration comparison scatter
      4. Phoneme timing bar chart
      5. Metric heatmap
    """
    plt = _require_mpl()

    fig = plt.figure(figsize=figsize)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Pitch deviation ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_pitch_deviation(report, ax1)

    # ── Onset deviation ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_onset_deviation(report, ax2)

    # ── Duration scatter ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _draw_duration_scatter(report, ax3)

    # ── Phoneme timing ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _draw_phoneme_timing(report, ax4)

    # ── Metric heatmap (full width) ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    _draw_heatmap(report, ax5)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.01)

    _save_or_return(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Private axis-level drawing helpers (reused by both individual and summary)
# ---------------------------------------------------------------------------

def _draw_pitch_deviation(report, ax):
    if report.pitch is None or not report.pitch.per_note:
        ax.text(0.5, 0.5, "No pitch data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Pitch Deviation")
        return
    tol = report.pitch.tolerance_cents
    values = [b.value or 0.0 for b in report.pitch.per_note]
    colors = ["seagreen" if abs(v) <= tol else "crimson" for v in values]
    ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(tol, color="gray", linewidth=0.7, linestyle="--")
    ax.axhline(-tol, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Note idx", fontsize=8)
    ax.set_ylabel("Deviation (¢)", fontsize=8)
    ax.set_title("Pitch Deviation", fontsize=9)
    mace = report.pitch.mace_cents
    if mace is not None:
        ax.text(0.02, 0.96, f"MACE={mace:.1f}¢", transform=ax.transAxes,
                fontsize=8, va="top", color="navy")


def _draw_onset_deviation(report, ax):
    if report.timing is None or not report.timing.per_note:
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Onset Deviation")
        return
    tol = report.timing.tolerance_ms
    values = [b.value or 0.0 for b in report.timing.per_note]
    colors = ["steelblue" if v < 0 else "tomato" for v in values]
    ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(tol, color="gray", linewidth=0.7, linestyle="--")
    ax.axhline(-tol, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Note idx", fontsize=8)
    ax.set_ylabel("Deviation (ms)", fontsize=8)
    ax.set_title("Onset Timing Deviation", fontsize=9)
    mae = report.timing.mean_abs_onset_error_ms
    if mae is not None:
        ax.text(0.02, 0.96, f"MAE={mae:.1f} ms", transform=ax.transAxes,
                fontsize=8, va="top", color="navy")


def _draw_duration_scatter(report, ax):
    if report.duration is None or not report.duration.per_note:
        ax.text(0.5, 0.5, "No duration data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Duration Comparison")
        return
    pred = [b.metadata.get("pred_dur_s", 0.0) for b in report.duration.per_note]
    ref = [b.metadata.get("ref_dur_s", 0.0) for b in report.duration.per_note]
    max_d = max(max(ref, default=1.0), max(pred, default=1.0)) * 1.05
    ax.scatter(ref, pred, alpha=0.7, s=25, color="steelblue")
    ax.plot([0, max_d], [0, max_d], "k--", linewidth=0.7)
    ax.set_xlim(0, max_d)
    ax.set_ylim(0, max_d)
    ax.set_xlabel("Reference (s)", fontsize=8)
    ax.set_ylabel("Predicted (s)", fontsize=8)
    ax.set_title("Duration Comparison", fontsize=9)


def _draw_phoneme_timing(report, ax):
    if report.lyric is None or not report.lyric.per_phoneme:
        ax.text(0.5, 0.5, "No phoneme data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Phoneme Timing")
        return
    tol = report.lyric.tolerance_ms
    values = [b.value or 0.0 for b in report.lyric.per_phoneme]
    colors = ["seagreen" if abs(v) <= tol else "crimson" for v in values]
    ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(tol, color="gray", linewidth=0.7, linestyle="--")
    ax.axhline(-tol, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Phoneme idx", fontsize=8)
    ax.set_ylabel("Deviation (ms)", fontsize=8)
    ax.set_title("Phoneme Boundary Deviation", fontsize=9)


def _draw_heatmap(report, ax):
    try:
        import numpy as np
    except ImportError:
        ax.text(0.5, 0.5, "numpy required for heatmap", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    rows, labels = [], []
    if report.pitch and report.pitch.per_note:
        rows.append([b.value or 0.0 for b in report.pitch.per_note])
        labels.append("Pitch (¢)")
    if report.timing and report.timing.per_note:
        rows.append([b.value or 0.0 for b in report.timing.per_note])
        labels.append("Onset (ms)")
    if report.duration and report.duration.per_note:
        rows.append([b.value or 0.0 for b in report.duration.per_note])
        labels.append("Duration (s)")

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Metric Heatmap")
        return

    n_cols = max(len(r) for r in rows)
    grid = np.zeros((len(rows), n_cols))
    for i, row in enumerate(rows):
        arr = np.array(row, dtype=float)
        mx = np.max(np.abs(arr)) if arr.size else 1.0
        grid[i, :len(arr)] = arr / (mx if mx != 0 else 1.0)

    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Note index", fontsize=8)
    ax.set_title("Per-Note Metric Heatmap (normalised)", fontsize=9)
    import matplotlib.pyplot as plt_mod
    plt_mod.colorbar(im, ax=ax, label="Norm. deviation", fraction=0.02)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_or_return(fig, save_path: Optional[str]):
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
