"""
visualization/scoring_viz.py - Score visualization utilities.

All functions use matplotlib with the Agg backend (headless, deterministic).
No GUI windows are created; figures are returned and optionally saved to disk.

Functions:
    plot_category_radar       Spider/radar chart of category scores
    plot_score_breakdown      Horizontal bar chart of category + component scores
    plot_timing_penalty       Per-note onset error colored by severity
    plot_pitch_scoring_overlay Pitch deviation bars colored by score range
    plot_performance_dashboard Combined 4-panel summary figure
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Backend guard
# ---------------------------------------------------------------------------

def _require_mpl():
    """Switch to the Agg backend (headless) and return pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_or_return(fig, save_path: Optional[str]):
    """Save figure to disk (creating parent dirs) or return it."""
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        _require_mpl().close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Internal draw helpers
# ---------------------------------------------------------------------------

def _draw_radar(ax, labels: List[str], values: List[float], title: str) -> None:
    """Draw a radar (spider) chart on a polar Axes."""
    n = len(labels)
    if n < 3:
        ax.text(0.5, 0.5, "Not enough categories", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        return

    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]  # close the polygon
    vals = list(values) + [values[0]]

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], size=7, color="gray")
    ax.plot(angles, vals, linewidth=2, color="#2196F3")
    ax.fill(angles, vals, alpha=0.2, color="#2196F3")
    ax.set_title(title, size=11, pad=14)


def _draw_breakdown_bars(ax, categories: Dict[str, Optional[float]], title: str) -> None:
    """Horizontal bar chart of category scores."""
    names = list(categories.keys())
    scores = [v if v is not None else 0.0 for v in categories.values()]

    colors = []
    for s in scores:
        if s >= 90:
            colors.append("#4CAF50")   # green
        elif s >= 75:
            colors.append("#8BC34A")   # light green
        elif s >= 55:
            colors.append("#FFC107")   # amber
        else:
            colors.append("#F44336")   # red

    y_pos = range(len(names))
    ax.barh(list(y_pos), scores, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 100)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names)
    ax.set_xlabel("Score [0–100]")
    ax.set_title(title, fontsize=11)
    ax.axvline(x=90, color="#4CAF50", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(x=75, color="#FFC107", linestyle="--", linewidth=0.8, alpha=0.7)
    for i, s in enumerate(scores):
        ax.text(min(s + 1, 98), i, f"{s:.1f}", va="center", ha="left", fontsize=8)


def _draw_component_bars(ax, components, cat_label: str) -> None:
    """Horizontal bar chart of ScoreBreakdown components for one category."""
    names  = [c.component for c in components]
    scores = [c.score for c in components]
    colors = ["#2196F3" if s >= 75 else "#FF5722" for s in scores]

    y_pos = range(len(names))
    ax.barh(list(y_pos), scores, color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 100)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title(f"{cat_label} components", fontsize=10)
    for i, s in enumerate(scores):
        ax.text(min(s + 1, 98), i, f"{s:.1f}", va="center", ha="left", fontsize=7)


def _draw_timing_penalty(ax, per_note, tolerance_ms: float, title: str) -> None:
    """Bar chart of per-note onset deviations coloured by severity."""
    if not per_note:
        ax.text(0.5, 0.5, "No per-note data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(title, fontsize=10)
        return

    indices = [b.event_idx for b in per_note]
    devs = [b.value if b.value is not None else 0.0 for b in per_note]
    colors = ["#F44336" if abs(d) > tolerance_ms else "#2196F3" for d in devs]

    ax.bar(indices, devs, color=colors, width=0.7)
    ax.axhline(y=tolerance_ms, color="#F44336", linestyle="--", linewidth=0.8, label=f"+{tolerance_ms:.0f} ms")
    ax.axhline(y=-tolerance_ms, color="#F44336", linestyle="--", linewidth=0.8, label=f"-{tolerance_ms:.0f} ms")
    ax.axhline(y=0, color="gray", linewidth=0.6)
    ax.set_xlabel("Note index")
    ax.set_ylabel("Onset deviation (ms)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)


def _draw_pitch_overlay(ax, per_note, tolerance_cents: float, title: str) -> None:
    """Bar chart of per-note pitch deviations coloured by within/outside tolerance."""
    if not per_note:
        ax.text(0.5, 0.5, "No per-note data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(title, fontsize=10)
        return

    indices = [b.event_idx for b in per_note]
    devs = [b.value if b.value is not None else 0.0 for b in per_note]
    colors = ["#4CAF50" if abs(d) <= tolerance_cents else "#F44336" for d in devs]

    ax.bar(indices, devs, color=colors, width=0.7)
    ax.axhline(y=tolerance_cents, color="#4CAF50", linestyle="--", linewidth=0.8, label=f"+{tolerance_cents:.0f}¢")
    ax.axhline(y=-tolerance_cents, color="#4CAF50", linestyle="--", linewidth=0.8, label=f"-{tolerance_cents:.0f}¢")
    ax.axhline(y=0, color="gray", linewidth=0.6)
    ax.set_xlabel("Note index")
    ax.set_ylabel("Pitch deviation (cents)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_category_radar(
    report: Any,
    figsize: Tuple[float, float] = (6, 6),
    title: str = "Performance Category Scores",
    save_path: Optional[str] = None,
):
    """
    Radar (spider) chart of the four category scores.

    Args:
        report:    PerformanceScoreReport.
        figsize:   Figure dimensions in inches.
        title:     Chart title.
        save_path: If given, save to this path instead of returning the figure.

    Returns:
        matplotlib Figure, or None if save_path was provided.
    """
    plt = _require_mpl()

    categories: Dict[str, Optional[float]] = {}
    for name, cat in [
        ("Pitch",    report.pitch_score),
        ("Timing",   report.timing_score),
        ("Duration", report.duration_score),
        ("Lyric",    report.lyric_score),
    ]:
        if cat is not None:
            categories[name] = cat.score

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    labels = list(categories.keys())
    values = [categories[l] for l in labels]

    overall = report.overall_score
    full_title = (
        f"{title}\nOverall: {overall:.1f}" if overall is not None else title
    )
    _draw_radar(ax, labels, values, full_title)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_score_breakdown(
    report: Any,
    figsize: Tuple[float, float] = (8, 5),
    title: str = "Score Breakdown by Category",
    save_path: Optional[str] = None,
):
    """
    Horizontal bar chart showing all category scores and the overall score.

    Args:
        report:    PerformanceScoreReport.
        figsize:   Figure dimensions in inches.
        title:     Chart title.
        save_path: If given, save to this path instead of returning the figure.

    Returns:
        matplotlib Figure, or None if save_path was provided.
    """
    plt = _require_mpl()

    categories: Dict[str, Optional[float]] = {}
    for name, cat in [
        ("Pitch",    report.pitch_score),
        ("Timing",   report.timing_score),
        ("Duration", report.duration_score),
        ("Lyric",    report.lyric_score),
    ]:
        if cat is not None:
            categories[name] = cat.score
    if report.overall_score is not None:
        categories["Overall"] = report.overall_score

    fig, ax = plt.subplots(figsize=figsize)
    _draw_breakdown_bars(ax, categories, title)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_timing_penalty(
    report: Any,
    tolerance_ms: float = 50.0,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Per-Note Onset Deviation",
    save_path: Optional[str] = None,
):
    """
    Bar chart of per-note onset deviations coloured by within/outside tolerance.

    Args:
        report:       PerformanceScoreReport.
        tolerance_ms: Threshold line drawn at ±tolerance_ms.
        figsize:      Figure dimensions in inches.
        title:        Chart title.
        save_path:    If given, save to this path instead of returning the figure.

    Returns:
        matplotlib Figure, or None if save_path was provided.
    """
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=figsize)

    per_note = []
    if report.timing_score is not None:
        for comp in report.timing_score.components:
            if comp.component == "onset_mae" and hasattr(comp, "_per_note"):
                per_note = comp._per_note
                break

    # Fallback: try to get per_note from score_metadata
    meta = report.score_metadata or {}
    if not per_note and "timing_per_note" in meta:
        per_note = meta["timing_per_note"]

    _draw_timing_penalty(ax, per_note, tolerance_ms, title)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_pitch_scoring_overlay(
    report: Any,
    tolerance_cents: float = 50.0,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Per-Note Pitch Deviation",
    save_path: Optional[str] = None,
):
    """
    Bar chart of per-note pitch deviations coloured by within/outside tolerance.

    Args:
        report:          PerformanceScoreReport.
        tolerance_cents: Threshold lines drawn at ±tolerance_cents.
        figsize:         Figure dimensions in inches.
        title:           Chart title.
        save_path:       If given, save to this path instead of returning the figure.

    Returns:
        matplotlib Figure, or None if save_path was provided.
    """
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=figsize)

    per_note = []
    meta = report.score_metadata or {}
    if "pitch_per_note" in meta:
        per_note = meta["pitch_per_note"]

    _draw_pitch_overlay(ax, per_note, tolerance_cents, title)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_performance_dashboard(
    report: Any,
    figsize: Tuple[float, float] = (14, 10),
    suptitle: str = "Performance Score Dashboard",
    save_path: Optional[str] = None,
):
    """
    4-panel dashboard: radar + category bars + pitch breakdown + timing breakdown.

    Args:
        report:    PerformanceScoreReport.
        figsize:   Figure dimensions in inches.
        suptitle:  Figure-level title.
        save_path: If given, save to this path instead of returning the figure.

    Returns:
        matplotlib Figure, or None if save_path was provided.
    """
    import matplotlib.gridspec as gridspec
    plt = _require_mpl()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel 1: radar ────────────────────────────────────────────────
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    cat_scores: Dict[str, float] = {}
    for name, attr in [("Pitch", "pitch_score"), ("Timing", "timing_score"),
                        ("Duration", "duration_score"), ("Lyric", "lyric_score")]:
        cs = getattr(report, attr, None)
        if cs is not None:
            cat_scores[name] = cs.score
    overall = report.overall_score
    radar_title = f"Category Scores\nOverall: {overall:.1f}" if overall is not None else "Category Scores"
    _draw_radar(ax_radar, list(cat_scores.keys()), list(cat_scores.values()), radar_title)

    # ── Panel 2: category bar chart ───────────────────────────────────
    ax_bars = fig.add_subplot(gs[0, 1])
    all_cats: Dict[str, Optional[float]] = {
        name: (cs.score if cs is not None else None)
        for name, cs in [
            ("Pitch", report.pitch_score), ("Timing", report.timing_score),
            ("Duration", report.duration_score), ("Lyric", report.lyric_score),
        ]
    }
    if overall is not None:
        all_cats["Overall"] = overall
    _draw_breakdown_bars(ax_bars, all_cats, "Category Scores")

    # ── Panel 3: pitch components ─────────────────────────────────────
    ax_pitch = fig.add_subplot(gs[1, 0])
    if report.pitch_score is not None and report.pitch_score.components:
        _draw_component_bars(ax_pitch, report.pitch_score.components, "Pitch")
    else:
        ax_pitch.text(0.5, 0.5, "No pitch data", ha="center", va="center",
                      transform=ax_pitch.transAxes, color="gray")
        ax_pitch.set_title("Pitch components", fontsize=10)

    # ── Panel 4: timing components ────────────────────────────────────
    ax_timing = fig.add_subplot(gs[1, 1])
    if report.timing_score is not None and report.timing_score.components:
        _draw_component_bars(ax_timing, report.timing_score.components, "Timing")
    else:
        ax_timing.text(0.5, 0.5, "No timing data", ha="center", va="center",
                       transform=ax_timing.transAxes, color="gray")
        ax_timing.set_title("Timing components", fontsize=10)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    return _save_or_return(fig, save_path)
