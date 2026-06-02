"""
preprocessing/framing.py - Generic signal framing utilities.

Provides sliding-window framing for arbitrary signals and helpers for
computing frame counts across models with different hop sizes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from preprocessing.timestamps import HOP_LENGTH, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Frame count helpers
# ---------------------------------------------------------------------------

def n_frames_from_samples(
    n_samples: int,
    hop_length: int = HOP_LENGTH,
    frame_length: Optional[int] = None,
    center: bool = False,
) -> int:
    """
    Compute the number of frames produced by a signal of n_samples.

    Matches librosa / torch STFT conventions.

    Args:
        n_samples:    Total number of audio samples.
        hop_length:   Hop size in samples.
        frame_length: Window length in samples. If None, uses hop_length.
        center:       If True, signal is padded by frame_length//2 on each side
                      before framing (librosa default). If False, no padding.

    Returns:
        Integer number of complete frames.
    """
    fl = frame_length if frame_length is not None else hop_length
    if center:
        n_samples = n_samples + 2 * (fl // 2)
    return max(0, (n_samples - fl) // hop_length + 1)


def n_frames_from_duration(
    duration_s: float,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    frame_length: Optional[int] = None,
    center: bool = False,
) -> int:
    """Convenience wrapper: duration in seconds → frame count."""
    n_samples = int(np.round(duration_s * sample_rate))
    return n_frames_from_samples(n_samples, hop_length, frame_length, center)


# ---------------------------------------------------------------------------
# Signal framing
# ---------------------------------------------------------------------------

def frame_signal(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int = HOP_LENGTH,
    center: bool = False,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Slice a 1-D signal into overlapping frames.

    Args:
        signal:       1-D array of length N.
        frame_length: Number of samples per frame.
        hop_length:   Hop size in samples.
        center:       If True, pad signal by frame_length//2 on each side so
                      frame centers align to sample indices (librosa convention).
        pad_mode:     NumPy pad mode used when center=True.

    Returns:
        2-D array of shape (n_frames, frame_length).
    """
    if center:
        pad = frame_length // 2
        signal = np.pad(signal, pad, mode=pad_mode)

    n = len(signal)
    n_fr = max(0, (n - frame_length) // hop_length + 1)
    out = np.empty((n_fr, frame_length), dtype=signal.dtype)
    for i in range(n_fr):
        start = i * hop_length
        out[i] = signal[start : start + frame_length]
    return out


def vad_frame_boundaries(
    n_samples: int,
    sample_rate: int = SAMPLE_RATE,
    frame_duration_ms: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute start/end sample indices for non-overlapping VAD frames.

    WebRTC VAD requires non-overlapping frames of exactly 10, 20, or 30 ms.

    Args:
        n_samples:         Total audio length in samples.
        sample_rate:       Audio sample rate.
        frame_duration_ms: Frame duration: must be 10, 20, or 30.

    Returns:
        starts: 1-D int array of frame start sample indices.
        ends:   1-D int array of frame end sample indices (exclusive).
    """
    if frame_duration_ms not in (10, 20, 30):
        raise ValueError(
            f"frame_duration_ms must be 10, 20, or 30; got {frame_duration_ms}"
        )
    frame_samples = int(sample_rate * frame_duration_ms / 1000)
    n_frames = n_samples // frame_samples
    starts = np.arange(n_frames, dtype=np.int64) * frame_samples
    ends = starts + frame_samples
    return starts, ends


def canonical_frame_count(n_samples: int, hop_length: int = HOP_LENGTH) -> int:
    """
    Number of canonical (non-overlapping, non-padded) frames.

    Equivalent to n_frames_from_samples with center=False and
    frame_length=hop_length.
    """
    return max(0, (n_samples - 1) // hop_length + 1)
