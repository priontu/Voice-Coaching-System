"""
visualization.py - Plotting utilities for waveform, VAD mask, and pitch contour.

All functions accept optional save_path arguments. When provided the figure
is saved and closed (non-interactive, suitable for batch runs). When omitted
the figure is returned so callers can display or further customize it.

Color conventions used throughout:
  Blue        — waveform
  Green shading — voiced regions (VAD output)
  Red line    — raw or cleaned pitch contour
  Orange line — reference pitch (when provided)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — override with matplotlib.use() before import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _save_or_return(fig: Figure, save_path: Optional[str]) -> Optional[Figure]:
    """Save and close the figure if save_path is given, else return it."""
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Viz] Saved → {p}")
        return None
    return fig


def _shade_voiced_regions(
    ax: plt.Axes,
    voiced_mask: np.ndarray,
    timestamps: np.ndarray,
    color: str = "green",
    alpha: float = 0.15,
    label: str = "Voiced (VAD)",
) -> None:
    """Shade contiguous voiced regions as semi-transparent spans on an axis."""
    if len(timestamps) < 2:
        return

    frame_sec = float(timestamps[1] - timestamps[0])
    half = frame_sec / 2.0

    in_seg = False
    seg_start = 0.0
    first = True

    for i, (v, t) in enumerate(zip(voiced_mask, timestamps)):
        if v and not in_seg:
            seg_start = t - half
            in_seg = True
        elif not v and in_seg:
            ax.axvspan(
                seg_start, t - half,
                color=color, alpha=alpha,
                label=label if first else "_nolegend_",
            )
            in_seg = False
            first = False

    if in_seg:
        ax.axvspan(
            seg_start, float(timestamps[-1]) + half,
            color=color, alpha=alpha,
            label=label if first else "_nolegend_",
        )


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_waveform(
    audio: np.ndarray,
    sr: int,
    title: str = "Waveform",
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot the raw audio waveform.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate in Hz.
        title: Plot title.
        save_path: If given, save and close the figure.

    Returns:
        Figure if save_path is None, else None.
    """
    times = np.arange(len(audio)) / sr

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(times, audio, color="steelblue", linewidth=0.6, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(times[0], times[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return _save_or_return(fig, save_path)


def plot_waveform_with_vad(
    audio: np.ndarray,
    sr: int,
    voiced_mask: np.ndarray,
    vad_times: np.ndarray,
    title: str = "Waveform + VAD",
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot waveform with green shading over voiced (VAD-detected) regions.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate in Hz.
        voiced_mask: Boolean VAD mask, shape (n_vad_frames,).
        vad_times: VAD frame center timestamps, shape (n_vad_frames,).
        title: Plot title.
        save_path: If given, save and close.

    Returns:
        Figure if save_path is None, else None.
    """
    audio_times = np.arange(len(audio)) / sr

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(audio_times, audio, color="steelblue", linewidth=0.5, alpha=0.8, label="Waveform")
    _shade_voiced_regions(ax, voiced_mask, vad_times, color="green", alpha=0.25, label="Voiced (VAD)")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(audio_times[0], audio_times[-1])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return _save_or_return(fig, save_path)


def plot_pitch_contour(
    times: np.ndarray,
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    reference_f0: Optional[np.ndarray] = None,
    title: str = "Pitch Contour",
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot the cleaned pitch contour (voiced frames only).

    Unvoiced frames (f0 == 0) are omitted from the plot to avoid jagged
    drops to zero. An optional reference F0 is overlaid in orange.

    Args:
        times: Frame timestamps, shape (T,).
        f0: Predicted F0 in Hz, shape (T,). Unvoiced = 0.
        voiced_mask: Boolean voiced mask, shape (T,).
        reference_f0: Optional reference F0 in Hz, shape (T,).
        title: Plot title.
        save_path: If given, save and close.

    Returns:
        Figure if save_path is None, else None.
    """
    voiced_t = times[voiced_mask & (f0 > 0)]
    voiced_f = f0[voiced_mask & (f0 > 0)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(voiced_t, voiced_f, s=2, color="crimson", label="Predicted F0", zorder=3)

    if reference_f0 is not None:
        ref_voiced = reference_f0 > 0
        ax.plot(
            times[ref_voiced], reference_f0[ref_voiced],
            color="darkorange", linewidth=1.5, alpha=0.8, label="Reference F0", zorder=2,
        )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlim(times[0], times[-1])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return _save_or_return(fig, save_path)


def plot_pitch_vad_combined(
    audio: np.ndarray,
    sr: int,
    pitch_times: np.ndarray,
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    vad_times: Optional[np.ndarray] = None,
    vad_mask: Optional[np.ndarray] = None,
    reference_f0: Optional[np.ndarray] = None,
    title: str = "Waveform · VAD · Pitch",
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Three-panel combined visualization: waveform, VAD mask, and pitch contour.

    Panel 1 (top)   — blue waveform with green voiced regions shaded
    Panel 2 (middle)— binary VAD mask over time
    Panel 3 (bottom)— red pitch contour (+ orange reference if provided)

    Args:
        audio: 1-D float32 audio.
        sr: Sample rate.
        pitch_times: Pitch frame timestamps, shape (T,).
        f0: Cleaned F0 in Hz, shape (T,). Unvoiced = 0.
        voiced_mask: Final voiced boolean mask, shape (T,).
        vad_times: VAD frame timestamps (at VAD frame rate), shape (N,). Optional.
        vad_mask: Raw VAD binary mask, shape (N,). Optional.
        reference_f0: Optional reference F0 array aligned to pitch_times, shape (T,).
        title: Overall figure title.
        save_path: If given, save and close.

    Returns:
        Figure if save_path is None, else None.
    """
    audio_times = np.arange(len(audio)) / sr

    use_vad_panel = vad_times is not None and vad_mask is not None
    n_panels = 3 if use_vad_panel else 2
    heights = [2, 1, 3] if use_vad_panel else [2, 3]

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 2 * n_panels + 2),
        gridspec_kw={"height_ratios": heights},
        sharex=False,
    )

    if n_panels == 2:
        ax_wave, ax_pitch = axes
        ax_vad = None
    else:
        ax_wave, ax_vad, ax_pitch = axes

    t_min = float(audio_times[0])
    t_max = float(audio_times[-1])

    # --- Panel 1: Waveform + voiced shading ---
    ax_wave.plot(audio_times, audio, color="steelblue", linewidth=0.5, alpha=0.9)
    _shade_voiced_regions(
        ax_wave, voiced_mask, pitch_times,
        color="green", alpha=0.2, label="Voiced (fused)",
    )
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_xlim(t_min, t_max)
    ax_wave.legend(loc="upper right", fontsize=7)
    ax_wave.grid(True, alpha=0.25)
    ax_wave.set_title(title)

    # --- Panel 2: Raw VAD mask (optional) ---
    if ax_vad is not None and use_vad_panel:
        ax_vad.fill_between(
            vad_times,
            vad_mask.astype(float),
            step="mid",
            color="green",
            alpha=0.7,
            label="VAD voiced",
        )
        ax_vad.set_yticks([0, 1])
        ax_vad.set_yticklabels(["Unvoiced", "Voiced"], fontsize=7)
        ax_vad.set_xlim(t_min, t_max)
        ax_vad.set_ylim(-0.05, 1.15)
        ax_vad.set_ylabel("VAD")
        ax_vad.legend(loc="upper right", fontsize=7)
        ax_vad.grid(True, alpha=0.25)

    # --- Panel 3: Pitch contour ---
    voiced_t = pitch_times[voiced_mask & (f0 > 0)]
    voiced_f = f0[voiced_mask & (f0 > 0)]

    ax_pitch.scatter(voiced_t, voiced_f, s=2, color="crimson", label="Predicted F0", zorder=3)

    if reference_f0 is not None:
        ref_voiced = reference_f0 > 0
        ax_pitch.plot(
            pitch_times[ref_voiced], reference_f0[ref_voiced],
            color="darkorange", linewidth=1.5, alpha=0.8, label="Reference F0", zorder=2,
        )

    ax_pitch.set_xlabel("Time (s)")
    ax_pitch.set_ylabel("F0 (Hz)")
    ax_pitch.set_xlim(t_min, t_max)
    ax_pitch.legend(loc="upper right", fontsize=7)
    ax_pitch.grid(True, alpha=0.25)

    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_cent_error(
    times: np.ndarray,
    predicted_f0: np.ndarray,
    reference_f0: np.ndarray,
    threshold_cents: float = 50.0,
    title: str = "Pitch Error (cents)",
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot per-frame cent error between predicted and reference F0.

    Dashed lines at ±threshold_cents show the accuracy boundary.

    Args:
        times: Frame timestamps, shape (T,).
        predicted_f0: Predicted F0 in Hz, shape (T,).
        reference_f0: Reference F0 in Hz, shape (T,).
        threshold_cents: Accuracy threshold to draw as dashed lines.
        title: Plot title.
        save_path: If given, save and close.

    Returns:
        Figure if save_path is None, else None.
    """
    voiced = (predicted_f0 > 0) & (reference_f0 > 0)
    if not np.any(voiced):
        logger.warning("[Viz] No voiced frames to plot cent error.")
        return None

    t_voiced = times[voiced]
    cents = 1200 * np.log2(predicted_f0[voiced] / reference_f0[voiced])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_voiced, cents, color="steelblue", linewidth=1.0, label="Cent error")
    ax.axhline(threshold_cents, color="red", linestyle="--", linewidth=1, label=f"+{threshold_cents:.0f} cents")
    ax.axhline(-threshold_cents, color="red", linestyle="--", linewidth=1, label=f"-{threshold_cents:.0f} cents")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (cents)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return _save_or_return(fig, save_path)


def plot_note_level_errors(
    note_results: list,
    threshold_cents: float = 50.0,
    title: str = "Note-Level Median Pitch Error",
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Bar chart of per-note median cent errors (mirrors pitch_score.py's note chart).

    Args:
        note_results: List of note result dicts from pitch_score.py's
                      compute_note_level_pitch_correctness().
        threshold_cents: Accuracy threshold drawn as dashed lines.
        title: Plot title.
        save_path: If given, save and close.

    Returns:
        Figure if save_path is None, else None.
    """
    scored = [n for n in note_results if n.get("scored")]
    if not scored:
        logger.warning("[Viz] No scored notes to plot.")
        return None

    indices = [n["note_index"] for n in scored]
    errors = [n["median_cent_error"] for n in scored]
    colors = ["green" if abs(e) <= threshold_cents else "red" for e in errors]

    fig, ax = plt.subplots(figsize=(max(10, len(indices) * 0.5), 5))
    ax.bar(indices, errors, color=colors, alpha=0.7)
    ax.axhline(threshold_cents, linestyle="--", color="gray", linewidth=1, label=f"+{threshold_cents:.0f} cents")
    ax.axhline(-threshold_cents, linestyle="--", color="gray", linewidth=1, label=f"-{threshold_cents:.0f} cents")
    ax.axhline(0, linestyle="-", color="black", linewidth=0.8, alpha=0.5)

    green_patch = mpatches.Patch(color="green", alpha=0.7, label="Correct (±50¢)")
    red_patch = mpatches.Patch(color="red", alpha=0.7, label="Incorrect")
    ax.legend(handles=[green_patch, red_patch], loc="upper right", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Note Index")
    ax.set_ylabel("Median Cent Error")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    return _save_or_return(fig, save_path)
