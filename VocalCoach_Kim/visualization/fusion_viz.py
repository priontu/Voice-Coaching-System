"""
visualization/fusion_viz.py - Matplotlib visualizations for fused event timelines.

All functions accept a FusedPerformanceRepresentation and produce matplotlib
figures suitable for debugging and analysis. No GUI dependencies beyond
matplotlib; use fig.savefig() for file output.

Usage:
    from visualization.fusion_viz import plot_fused_timeline
    fig = plot_fused_timeline(fused, figsize=(14, 10))
    fig.savefig("fusion_debug.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Lazy-import matplotlib so the module can be imported without a display
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for all envs)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_PALETTE = {
    "pitch":      "#2196F3",   # blue
    "voiced":     "#4CAF50",   # green
    "unvoiced":   "#F44336",   # red
    "note":       "#FF9800",   # orange
    "phoneme":    "#9C27B0",   # purple
    "word":       "#00BCD4",   # cyan
    "phrase":     "#795548",   # brown
    "onset":      "#E91E63",   # pink
    "offset":     "#607D8B",   # blue-grey
    "f0_voiced":  "#1565C0",   # dark blue (voiced F0)
    "f0_unvoiced":"#BBDEFB",   # light blue (unvoiced F0)
}


# ---------------------------------------------------------------------------
# Sub-plot components
# ---------------------------------------------------------------------------

def plot_pitch_contour(
    fused,
    ax=None,
    show_voiced: bool = True,
    show_f0_line: bool = True,
    figsize: Tuple[int, int] = (12, 3),
    title: str = "Pitch Contour (F0)",
):
    """
    Plot the F0 contour, coloured by voicing state.

    Args:
        fused:        FusedPerformanceRepresentation.
        ax:           Existing matplotlib Axes. Created if None.
        show_voiced:  Shade voiced/unvoiced regions.
        show_f0_line: Draw the F0 line.
        figsize:      Figure size (used only when ax is None).
        title:        Axes title.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if fused.timestamps is None or fused.f0 is None:
        ax.set_title(f"{title} (no data)")
        return fig

    ts = np.asarray(fused.timestamps, dtype=np.float64)
    f0 = np.asarray(fused.f0, dtype=np.float64)

    if show_voiced and fused.voiced is not None:
        voiced = np.asarray(fused.voiced, dtype=bool)
        v_f0 = np.where(voiced, f0, np.nan)
        u_f0 = np.where(~voiced, f0, np.nan)
        if show_f0_line:
            ax.plot(ts, v_f0, color=_PALETTE["f0_voiced"], lw=1.2, label="Voiced F0")
            ax.plot(ts, u_f0, color=_PALETTE["f0_unvoiced"], lw=0.8, alpha=0.5, label="Unvoiced F0")
    else:
        if show_f0_line:
            ax.plot(ts, f0, color=_PALETTE["pitch"], lw=1.2, label="F0")

    # Shade voiced regions
    for region in fused.voiced_regions:
        color = _PALETTE["voiced"] if region.label == "voiced" else _PALETTE["unvoiced"]
        ax.axvspan(region.start_time, region.end_time, alpha=0.10, color=color, linewidth=0)

    ax.set_ylabel("F0 (Hz)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, fused.duration_s)

    if standalone:
        plt.tight_layout()
    return fig


def plot_note_timeline(
    fused,
    ax=None,
    show_pitch: bool = True,
    figsize: Tuple[int, int] = (12, 3),
    title: str = "Note Events",
):
    """
    Draw horizontal bars for each detected note, colour-coded by mean F0 (MIDI).

    Args:
        fused:       FusedPerformanceRepresentation.
        ax:          Existing Axes. Created if None.
        show_pitch:  Colour bars by MIDI pitch (uses viridis scale, 40–80 MIDI).
        figsize:     Figure size (standalone only).
        title:       Axes title.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for idx, note in enumerate(fused.note_events):
        onset = note.onset_time
        offset = note.offset_time if note.offset_time is not None else onset + (note.duration or 0.3)
        dur = offset - onset
        if show_pitch and note.pitch_midi is not None:
            # Map MIDI 40–80 to [0, 1]
            t = np.clip((note.pitch_midi - 40) / 40.0, 0, 1)
            color = plt.cm.viridis(t)
        else:
            color = _PALETTE["note"]
        ax.barh(0, dur, left=onset, height=0.6, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        if dur > 0.15:
            label = note.lyric_text or str(idx)
            ax.text(onset + dur / 2, 0, label, ha="center", va="center",
                    fontsize=7, color="white", clip_on=True)

    ax.set_yticks([])
    ax.set_ylabel("Notes")
    ax.set_title(title)
    ax.set_xlim(0, fused.duration_s)

    if standalone:
        plt.tight_layout()
    return fig


def plot_phoneme_overlay(
    fused,
    ax=None,
    figsize: Tuple[int, int] = (12, 3),
    title: str = "Phoneme Segments",
):
    """
    Draw per-phoneme segment bars with label text.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for le in fused.lyric_events:
        dur = le.duration
        alpha = max(0.4, min(0.95, le.confidence))
        ax.barh(0, dur, left=le.start_time, height=0.6,
                color=_PALETTE["phoneme"], alpha=alpha,
                edgecolor="white", linewidth=0.4)
        if dur > 0.04:
            ax.text(le.start_time + dur / 2, 0, le.phoneme,
                    ha="center", va="center", fontsize=7, color="white", clip_on=True)

    ax.set_yticks([])
    ax.set_ylabel("Phonemes")
    ax.set_title(title)
    ax.set_xlim(0, fused.duration_s)

    if standalone:
        plt.tight_layout()
    return fig


def plot_lyric_timeline(
    fused,
    ax=None,
    figsize: Tuple[int, int] = (12, 2),
    title: str = "Word Events",
):
    """
    Draw per-word segment bars with word text.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for we in fused.word_events:
        dur = we.duration
        alpha = max(0.4, min(0.95, we.confidence))
        ax.barh(0, dur, left=we.start_time, height=0.6,
                color=_PALETTE["word"], alpha=alpha,
                edgecolor="white", linewidth=0.4)
        if dur > 0.05:
            ax.text(we.start_time + dur / 2, 0, we.text,
                    ha="center", va="center", fontsize=7, color="white", clip_on=True)

    ax.set_yticks([])
    ax.set_ylabel("Words")
    ax.set_title(title)
    ax.set_xlim(0, fused.duration_s)

    if standalone:
        plt.tight_layout()
    return fig


def plot_onset_offset_curves(
    fused,
    ax=None,
    figsize: Tuple[int, int] = (12, 2),
    title: str = "Onset / Offset Probabilities",
):
    """
    Plot the raw onset and offset probability curves from aligned features.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ts = fused.timestamps
    if ts is None:
        ax.set_title(f"{title} (no data)")
        return fig

    ts = np.asarray(ts, dtype=np.float64)

    # We don't store onset/offset probs on FusedPerformanceRepresentation directly,
    # but we can show note boundaries as vertical lines
    for note in fused.note_events:
        ax.axvline(note.onset_time, color=_PALETTE["onset"], alpha=0.7, lw=1.2)
        if note.offset_time:
            ax.axvline(note.offset_time, color=_PALETTE["offset"], alpha=0.7, lw=1.2, ls="--")

    onset_patch = mpatches.Patch(color=_PALETTE["onset"], label="Onset")
    offset_patch = mpatches.Patch(color=_PALETTE["offset"], label="Offset", linestyle="--")
    ax.legend(handles=[onset_patch, offset_patch], fontsize=8, loc="upper right")
    ax.set_ylabel("Note boundaries")
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlim(0, fused.duration_s)

    if standalone:
        plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite plot
# ---------------------------------------------------------------------------

def plot_fused_timeline(
    fused,
    figsize: Tuple[int, int] = (14, 10),
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Multi-panel overview figure combining all event streams.

    Panels (top → bottom):
      1. Pitch contour (F0 + voiced shading)
      2. Note events (horizontal bars, MIDI-coloured)
      3. Phoneme segments
      4. Word events

    Args:
        fused:      FusedPerformanceRepresentation.
        figsize:    Overall figure size.
        suptitle:   Super-title (defaults to audio filename).
        save_path:  If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()

    n_panels = 4
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    plot_pitch_contour(fused, ax=axes[0])
    plot_note_timeline(fused, ax=axes[1])
    plot_phoneme_overlay(fused, ax=axes[2])
    plot_lyric_timeline(fused, ax=axes[3])

    axes[-1].set_xlabel("Time (s)")

    title = suptitle or fused.audio_path
    fig.suptitle(title, fontsize=11, y=1.01)

    # Summary annotation
    summary = (
        f"{fused.n_notes} notes  |  "
        f"{len(fused.word_events)} words  |  "
        f"{len(fused.phrase_events)} phrases  |  "
        f"{fused.duration_s:.2f}s"
    )
    fig.text(0.5, -0.01, summary, ha="center", fontsize=9, color="gray")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_note_phoneme_alignment(
    fused,
    figsize: Tuple[int, int] = (14, 5),
    title: str = "Note–Phoneme Alignment",
    save_path: Optional[str] = None,
):
    """
    Two-row plot: notes on top, phoneme segments below, with shared time axis.
    Vertical dashed lines connect note boundaries to the phoneme row.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()

    fig, (ax_note, ax_phon) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                             gridspec_kw={"height_ratios": [1, 1]})

    plot_note_timeline(fused, ax=ax_note, title="Notes")
    plot_phoneme_overlay(fused, ax=ax_phon, title="Phonemes")

    # Draw onset/offset boundary lines crossing both panels
    for note in fused.note_events:
        ax_note.axvline(note.onset_time, color="gray", alpha=0.4, lw=0.8, ls=":")
        ax_phon.axvline(note.onset_time, color="gray", alpha=0.4, lw=0.8, ls=":")
        if note.offset_time:
            ax_note.axvline(note.offset_time, color="gray", alpha=0.4, lw=0.8, ls="--")
            ax_phon.axvline(note.offset_time, color="gray", alpha=0.4, lw=0.8, ls="--")

    ax_phon.set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
