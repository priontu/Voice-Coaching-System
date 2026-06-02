"""
Visualization utilities for onset/offset detection.

All plot functions return a matplotlib Figure and optionally save to disk.
Call plt.show() after the function if you want interactive display.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Waveform + boundaries
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform_with_boundaries(
    waveform: np.ndarray,
    sample_rate: int,
    predicted_notes: List[Dict],
    reference_notes: Optional[List[Dict]] = None,
    title: str = "Waveform — Note Boundaries",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot audio waveform with predicted (and optionally reference) onset/offset markers.

    Args:
        waveform:        [N] mono audio samples.
        sample_rate:     Sample rate in Hz.
        predicted_notes: Output of NoteDetector.detect().
        reference_notes: Ground truth notes for comparison.
        title:           Figure title.
        save_path:       If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    times = np.arange(len(waveform)) / sample_rate

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(times, waveform, color="steelblue", lw=0.5, alpha=0.8, label="Waveform")

    for note in predicted_notes:
        if note.get("onset_time") is not None:
            ax.axvline(note["onset_time"], color="#27ae60", lw=1.2, alpha=0.75)
        if note.get("offset_time") is not None:
            ax.axvline(note["offset_time"], color="#e74c3c", lw=1.2, alpha=0.75)

    if reference_notes:
        for note in reference_notes:
            if note.get("onset_time") is not None:
                ax.axvline(note["onset_time"], color="#2980b9", lw=1.0, ls="--", alpha=0.6)
            if note.get("offset_time") is not None:
                ax.axvline(note["offset_time"], color="#e67e22", lw=1.0, ls="--", alpha=0.6)

    handles = [
        mpatches.Patch(color="#27ae60", label="Pred onset"),
        mpatches.Patch(color="#e74c3c", label="Pred offset"),
    ]
    if reference_notes:
        handles += [
            mpatches.Patch(color="#2980b9", label="Ref onset"),
            mpatches.Patch(color="#e67e22", label="Ref offset"),
        ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(times[0], times[-1])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram + boundaries
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrogram_with_boundaries(
    log_mel: np.ndarray,
    hop_length: int,
    sample_rate: int,
    predicted_notes: List[Dict],
    reference_notes: Optional[List[Dict]] = None,
    title: str = "Log-Mel Spectrogram — Note Boundaries",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Display a log-mel spectrogram with onset/offset overlays.

    Args:
        log_mel:         [n_mels, T] spectrogram array.
        hop_length:      Hop size in samples (for x-axis scaling).
        sample_rate:     Sample rate in Hz (for x-axis scaling).
        predicted_notes: Detected note boundaries.
        reference_notes: Ground truth boundaries for comparison.
        title:           Figure title.
        save_path:       If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    n_mels, T = log_mel.shape
    duration = T * hop_length / sample_rate

    fig, ax = plt.subplots(figsize=(14, 5))
    img = ax.imshow(
        log_mel,
        origin="lower",
        aspect="auto",
        extent=[0.0, duration, 0, n_mels],
        cmap="magma",
    )
    plt.colorbar(img, ax=ax, label="Log power", shrink=0.8)

    for note in predicted_notes:
        if note.get("onset_time") is not None:
            ax.axvline(note["onset_time"], color="#2ecc71", lw=1.5, alpha=0.85)
        if note.get("offset_time") is not None:
            ax.axvline(note["offset_time"], color="#e74c3c", lw=1.5, alpha=0.85)

    if reference_notes:
        for note in reference_notes:
            if note.get("onset_time") is not None:
                ax.axvline(note["onset_time"], color="cyan", lw=1.0, ls="--", alpha=0.65)
            if note.get("offset_time") is not None:
                ax.axvline(note["offset_time"], color="yellow", lw=1.0, ls="--", alpha=0.65)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Probability curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_probability_curves(
    onset_probs: np.ndarray,
    offset_probs: np.ndarray,
    frame_times: np.ndarray,
    onset_threshold: float = 0.3,
    offset_threshold: float = 0.3,
    reference_onsets: Optional[List[float]] = None,
    reference_offsets: Optional[List[float]] = None,
    title: str = "Onset / Offset Probability Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot onset and offset probability curves with threshold lines.

    Vertical dotted lines show ground-truth boundary positions when provided.

    Args:
        onset_probs:      [T] onset probability array.
        offset_probs:     [T] offset probability array.
        frame_times:      [T] time in seconds per frame.
        onset_threshold:  Decision threshold drawn as a horizontal line.
        offset_threshold: Decision threshold drawn as a horizontal line.
        reference_onsets: Optional ground-truth onset times for reference.
        reference_offsets:Optional ground-truth offset times for reference.
        title:            Figure title.
        save_path:        If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(title)

    # ── Onset ──
    ax1.plot(frame_times, onset_probs, color="#27ae60", lw=1.0, label="Onset prob")
    ax1.axhline(
        onset_threshold, color="#27ae60", ls="--", alpha=0.6,
        label=f"Threshold ({onset_threshold})"
    )
    ax1.fill_between(frame_times, onset_probs, alpha=0.15, color="#27ae60")
    if reference_onsets:
        for t in reference_onsets:
            ax1.axvline(t, color="#2980b9", lw=0.9, ls=":", alpha=0.5)
    ax1.set_ylabel("Onset probability")
    ax1.set_ylim(-0.02, 1.08)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.25)

    # ── Offset ──
    ax2.plot(frame_times, offset_probs, color="#e74c3c", lw=1.0, label="Offset prob")
    ax2.axhline(
        offset_threshold, color="#e74c3c", ls="--", alpha=0.6,
        label=f"Threshold ({offset_threshold})"
    )
    ax2.fill_between(frame_times, offset_probs, alpha=0.15, color="#e74c3c")
    if reference_offsets:
        for t in reference_offsets:
            ax2.axvline(t, color="#e67e22", lw=0.9, ls=":", alpha=0.5)
    ax2.set_ylabel("Offset probability")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(-0.02, 1.08)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.25)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metrics bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_summary(
    metrics: Dict[str, float],
    title: str = "Detection Metrics",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of onset / offset precision, recall, and F1.

    Args:
        metrics:   Dict returned by evaluate_file() or similar.
        title:     Figure title.
        save_path: If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    keys = [
        "onset_precision", "onset_recall", "onset_f1",
        "offset_precision", "offset_recall", "offset_f1",
    ]
    labels = [k.replace("_", "\n") for k in keys if k in metrics]
    values = [metrics[k] for k in keys if k in metrics]
    colors = ["#27ae60"] * 3 + ["#e74c3c"] * 3

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(labels)), values,
        color=colors[: len(labels)], edgecolor="black", alpha=0.85
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.012,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    # Add MAE annotations if present
    mae_text_parts = []
    if "onset_mae_ms" in metrics:
        mae_text_parts.append(f"Onset MAE: {metrics['onset_mae_ms']:.1f} ms")
    if "offset_mae_ms" in metrics:
        mae_text_parts.append(f"Offset MAE: {metrics['offset_mae_ms']:.1f} ms")
    if "duration_mae_ms" in metrics:
        mae_text_parts.append(f"Duration MAE: {metrics['duration_mae_ms']:.1f} ms")
    if mae_text_parts:
        ax.text(
            0.98, 0.97, "\n".join(mae_text_parts),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Composite overview (all panels in one figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_overview(
    waveform: np.ndarray,
    log_mel: np.ndarray,
    onset_probs: np.ndarray,
    offset_probs: np.ndarray,
    frame_times: np.ndarray,
    sample_rate: int,
    hop_length: int,
    predicted_notes: List[Dict],
    reference_notes: Optional[List[Dict]] = None,
    onset_threshold: float = 0.3,
    offset_threshold: float = 0.3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Four-panel overview: waveform, spectrogram, onset curve, offset curve.

    Args:
        waveform:         [N] mono audio samples.
        log_mel:          [n_mels, T] spectrogram.
        onset_probs:      [T] onset probabilities.
        offset_probs:     [T] offset probabilities.
        frame_times:      [T] time in seconds per frame.
        sample_rate:      Sample rate in Hz.
        hop_length:       Hop size in samples.
        predicted_notes:  Detected boundaries.
        reference_notes:  Optional ground truth.
        onset_threshold:  Detection threshold line.
        offset_threshold: Detection threshold line.
        save_path:        If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    ref_on = (
        [n["onset_time"] for n in reference_notes if n.get("onset_time") is not None]
        if reference_notes else None
    )
    ref_off = (
        [n["offset_time"] for n in reference_notes if n.get("offset_time") is not None]
        if reference_notes else None
    )
    duration = len(waveform) / sample_rate
    n_mels = log_mel.shape[0]

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=False)

    # Panel 1: waveform
    t_wav = np.arange(len(waveform)) / sample_rate
    axes[0].plot(t_wav, waveform, color="steelblue", lw=0.4, alpha=0.8)
    for note in predicted_notes:
        if note.get("onset_time") is not None:
            axes[0].axvline(note["onset_time"], color="#27ae60", lw=1.0, alpha=0.7)
        if note.get("offset_time") is not None:
            axes[0].axvline(note["offset_time"], color="#e74c3c", lw=1.0, alpha=0.7)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform")
    axes[0].set_xlim(0, duration)

    # Panel 2: spectrogram
    axes[1].imshow(
        log_mel, origin="lower", aspect="auto",
        extent=[0.0, duration, 0, n_mels], cmap="magma",
    )
    for note in predicted_notes:
        if note.get("onset_time") is not None:
            axes[1].axvline(note["onset_time"], color="#2ecc71", lw=1.2, alpha=0.8)
        if note.get("offset_time") is not None:
            axes[1].axvline(note["offset_time"], color="#e74c3c", lw=1.2, alpha=0.8)
    axes[1].set_ylabel("Mel bin")
    axes[1].set_title("Log-Mel Spectrogram")

    # Panel 3: onset probs
    axes[2].plot(frame_times, onset_probs, color="#27ae60", lw=1.0)
    axes[2].fill_between(frame_times, onset_probs, alpha=0.2, color="#27ae60")
    axes[2].axhline(onset_threshold, color="#27ae60", ls="--", alpha=0.6)
    if ref_on:
        for t in ref_on:
            axes[2].axvline(t, color="#2980b9", lw=0.8, ls=":", alpha=0.55)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Onset prob")
    axes[2].set_title("Onset Probability")
    axes[2].grid(alpha=0.2)

    # Panel 4: offset probs
    axes[3].plot(frame_times, offset_probs, color="#e74c3c", lw=1.0)
    axes[3].fill_between(frame_times, offset_probs, alpha=0.2, color="#e74c3c")
    axes[3].axhline(offset_threshold, color="#e74c3c", ls="--", alpha=0.6)
    if ref_off:
        for t in ref_off:
            axes[3].axvline(t, color="#e67e22", lw=0.8, ls=":", alpha=0.55)
    axes[3].set_ylim(0, 1.05)
    axes[3].set_ylabel("Offset prob")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Offset Probability")
    axes[3].grid(alpha=0.2)

    fig.suptitle("Note Detection Overview", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
