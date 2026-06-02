"""
visualization/reference_viz.py - Predicted vs. reference comparison plots.

All functions use the Agg backend and return matplotlib Figure objects. They
are safe to call in headless environments (no display required).

Available plots:
  plot_note_alignment      - timeline overlay of predicted vs. reference notes
  plot_phoneme_alignment   - phoneme segment comparison with match highlights
  plot_onset_deviation     - bar chart of onset timing errors per matched note
  plot_duration_comparison - scatter plot of predicted vs. reference durations
  plot_pitch_vs_reference  - F0 contour overlaid with reference MIDI pitch steps
  plot_alignment_summary   - composite 4-panel figure (main entry point)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy matplotlib import with Agg backend
# ---------------------------------------------------------------------------

def _require_mpl():
    """Import and configure matplotlib. Raises ImportError if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for reference visualization. "
            "Install it with: pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_note_alignment(fused, reference, alignment=None, ax=None, figsize=(14, 3),
                        title="Predicted vs. Reference Notes") -> "Figure":
    """
    Draw predicted (blue) and reference (orange) note timelines on a shared axis.

    Matched pairs are connected with a thin grey line. Unmatched predicted
    notes are shown in red; unmatched reference notes in dark orange.

    Args:
        fused:      FusedPerformanceRepresentation (predicted).
        reference:  ReferencePerformanceRepresentation (ground truth).
        alignment:  AlignmentResult; if None, no match highlighting.
        ax:         Optional matplotlib Axes.
        figsize:    Figure size when ax is None.
        title:      Figure title.

    Returns:
        matplotlib Figure.
    """
    plt = _require_mpl()
    import matplotlib.patches as mpatches

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    matched_pred = set()
    matched_ref = set()
    if alignment is not None:
        for m in alignment.note_matches:
            matched_pred.add(m.pred_idx)
            matched_ref.add(m.ref_idx)

    # Reference notes (row 0)
    for i, n in enumerate(reference.notes):
        if n.is_rest:
            continue
        color = "darkorange" if i not in matched_ref else "orange"
        ax.barh(0, n.offset_time - n.onset_time, left=n.onset_time,
                height=0.4, color=color, alpha=0.7)

    # Predicted notes (row 1)
    for i, n in enumerate(fused.note_events):
        off = n.offset_time or (n.onset_time + (n.duration or 0.0))
        color = "red" if i not in matched_pred else "steelblue"
        ax.barh(1, off - n.onset_time, left=n.onset_time,
                height=0.4, color=color, alpha=0.7)

    # Connection lines for matched pairs
    if alignment is not None:
        ref_list = reference.notes
        pred_list = fused.note_events
        for m in alignment.note_matches:
            pred_n = pred_list[m.pred_idx]
            ref_n = ref_list[m.ref_idx]
            ax.plot(
                [pred_n.onset_time, ref_n.onset_time],
                [1.2, -0.2],
                color="grey", lw=0.7, alpha=0.5,
            )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Reference", "Predicted"])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

    legend_handles = [
        mpatches.Patch(color="orange", label="Reference (matched)"),
        mpatches.Patch(color="darkorange", label="Reference (unmatched)"),
        mpatches.Patch(color="steelblue", label="Predicted (matched)"),
        mpatches.Patch(color="red", label="Predicted (unmatched)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def plot_phoneme_alignment(fused, reference, alignment=None, ax=None, figsize=(14, 3),
                           title="Phoneme Alignment") -> "Figure":
    """
    Draw predicted (LyricEvent) and reference (ReferencePhoneme) segments as
    colour-coded horizontal bars. Matched pairs share the same hue.
    """
    plt = _require_mpl()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    matched_pred = set()
    matched_ref = set()
    if alignment is not None:
        for m in alignment.phoneme_matches:
            matched_pred.add(m.pred_idx)
            matched_ref.add(m.ref_idx)

    for i, ph in enumerate(reference.phonemes):
        color = "darkorange" if i not in matched_ref else "orange"
        ax.barh(0, ph.end_time - ph.start_time, left=ph.start_time,
                height=0.4, color=color, alpha=0.75)
        ax.text(ph.start_time + (ph.end_time - ph.start_time) / 2, 0,
                ph.phoneme, ha="center", va="center", fontsize=6)

    for i, le in enumerate(fused.lyric_events):
        color = "red" if i not in matched_pred else "steelblue"
        ax.barh(1, le.end_time - le.start_time, left=le.start_time,
                height=0.4, color=color, alpha=0.75)
        ax.text(le.start_time + (le.end_time - le.start_time) / 2, 1,
                le.phoneme, ha="center", va="center", fontsize=6)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Reference", "Predicted"])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_onset_deviation(alignment, ax=None, figsize=(10, 3),
                         title="Onset Timing Deviation (Predicted − Reference)") -> "Figure":
    """
    Bar chart of signed onset deviation (seconds) per matched note, sorted by
    reference onset time. Positive = predicted is later (behind the beat).
    """
    plt = _require_mpl()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if not alignment or not alignment.note_matches:
        ax.text(0.5, 0.5, "No note matches to display",
                ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    matches = sorted(alignment.note_matches, key=lambda m: m.pred_idx)
    x = list(range(len(matches)))
    devs = [m.onset_deviation_s for m in matches]
    colors = ["firebrick" if d > 0 else "steelblue" for d in devs]

    ax.bar(x, devs, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Matched Note Index")
    ax.set_ylabel("Deviation (s)")
    ax.set_title(title)
    ax.set_xticks(x)
    fig.tight_layout()
    return fig


def plot_duration_comparison(fused, reference, alignment, ax=None, figsize=(6, 5),
                             title="Duration: Predicted vs. Reference (s)") -> "Figure":
    """
    Scatter plot of predicted duration (y) vs. reference duration (x) for
    matched notes. The identity line y=x is shown in grey.
    """
    plt = _require_mpl()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if not alignment or not alignment.note_matches:
        ax.text(0.5, 0.5, "No note matches to display",
                ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    pred_durs, ref_durs = [], []
    for m in alignment.note_matches:
        pred_n = fused.note_events[m.pred_idx]
        ref_n = reference.notes[m.ref_idx]
        pred_off = pred_n.offset_time or (pred_n.onset_time + (pred_n.duration or 0.0))
        pred_durs.append(pred_off - pred_n.onset_time)
        ref_durs.append(ref_n.offset_time - ref_n.onset_time)

    ax.scatter(ref_durs, pred_durs, alpha=0.7, color="steelblue", s=25)
    lim = max(max(ref_durs, default=1), max(pred_durs, default=1)) * 1.1
    ax.plot([0, lim], [0, lim], color="grey", lw=0.8, linestyle="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Reference Duration (s)")
    ax.set_ylabel("Predicted Duration (s)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_pitch_vs_reference(fused, reference, ax=None, figsize=(14, 4),
                            title="F0 Contour vs. Reference Pitch") -> "Figure":
    """
    Predicted F0 contour overlaid with horizontal step segments showing the
    reference MIDI pitch (converted to Hz) for each reference note.
    """
    plt = _require_mpl()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # F0 contour
    if fused.timestamps is not None and fused.f0 is not None:
        import numpy as np
        ts = fused.timestamps
        f0 = fused.f0.copy().astype(float)
        if fused.voiced is not None:
            f0[~fused.voiced] = float("nan")
        ax.plot(ts, f0, color="steelblue", lw=0.8, label="Predicted F0")

    # Reference pitch steps
    for n in reference.notes:
        if n.is_rest or n.pitch_hz is None:
            continue
        ax.hlines(n.pitch_hz, n.onset_time, n.offset_time,
                  color="darkorange", lw=2.5, alpha=0.8, label="_nolegend_")

    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite summary figure
# ---------------------------------------------------------------------------

def plot_alignment_summary(
    fused,
    reference,
    alignment=None,
    figsize=(14, 12),
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
) -> "Figure":
    """
    4-panel alignment summary figure:
      1. Note timeline overlay (predicted vs. reference)
      2. Onset deviation bar chart
      3. Duration scatter plot
      4. F0 contour vs. reference pitch steps

    Args:
        fused:      FusedPerformanceRepresentation.
        reference:  ReferencePerformanceRepresentation.
        alignment:  AlignmentResult; optional.
        figsize:    Overall figure size.
        suptitle:   Optional super-title.
        save_path:  If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    plt = _require_mpl()
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 2, figure=fig)

    ax_notes = fig.add_subplot(gs[0, :])
    ax_onset = fig.add_subplot(gs[1, :])
    ax_dur = fig.add_subplot(gs[2, 0])
    ax_pitch = fig.add_subplot(gs[2:, 1])
    ax_phones = fig.add_subplot(gs[3, 0])

    plot_note_alignment(fused, reference, alignment, ax=ax_notes)
    plot_onset_deviation(alignment, ax=ax_onset)
    plot_duration_comparison(fused, reference, alignment, ax=ax_dur)
    plot_pitch_vs_reference(fused, reference, ax=ax_pitch)
    plot_phoneme_alignment(fused, reference, alignment, ax=ax_phones)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("[reference_viz] Saved alignment summary to %s", save_path)

    return fig
