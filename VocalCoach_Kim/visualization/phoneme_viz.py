"""
visualization/phoneme_viz.py - Phoneme boundary visualization.

Waveform + colour-coded phoneme segment timeline.
Moved from the original Phoneme Model/phoneme_model.py visualization section.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def plot_phoneme_boundaries(
    audio,
    segments: List,
    sample_rate: int = 16000,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
) -> None:
    """
    Waveform + colour-coded phoneme segment timeline.

    Args:
        audio:       1-D audio array or torch.Tensor.
        segments:    List of PhonemeSegment objects.
        sample_rate: Audio sample rate in Hz.
        save_path:   Save figure to this path instead of displaying.
        figsize:     Matplotlib figure size.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping visualization")
        return

    # Normalize to numpy
    try:
        import torch
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
    except ImportError:
        pass

    time_axis = np.arange(len(audio)) / sample_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax1.plot(time_axis, audio, linewidth=0.5, color="steelblue", alpha=0.7)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title("Waveform with Phoneme Boundaries", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    if segments:
        unique_phonemes = list(dict.fromkeys(s.phoneme for s in segments))
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(unique_phonemes), 1)))
        color_map = {p: palette[i] for i, p in enumerate(unique_phonemes)}

        for seg in segments:
            color = color_map[seg.phoneme]
            ax2.axvspan(seg.start_time, seg.end_time, alpha=0.3, color=color)
            mid = (seg.start_time + seg.end_time) / 2
            ax2.text(mid, 0.5, seg.phoneme, ha="center", va="center",
                     fontsize=9, fontweight="bold")

    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Phonemes", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=150, bbox_inches="tight")
        logger.info(f"[Viz] Phoneme boundary plot saved → {p}")
    else:
        plt.show()

    plt.close()
