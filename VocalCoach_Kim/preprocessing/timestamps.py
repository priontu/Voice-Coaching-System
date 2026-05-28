"""
preprocessing/timestamps.py - Canonical frame-time conversion engine.

All temporal alignment in VocalCoach uses this module as a single source of
truth. The canonical timeline is defined by HOP_LENGTH=160 at SAMPLE_RATE=16000
(10 ms frames, 100 fps).
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

# ---------------------------------------------------------------------------
# Canonical constants — the single source of truth for timing across all modules
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000          # Hz
HOP_LENGTH: int = 160             # samples per frame → 10 ms
FRAME_DURATION: float = 0.01      # seconds = HOP_LENGTH / SAMPLE_RATE

# Per-model native hop sizes (for documentation / alignment callers)
_PHONEME_HOP: int = 320   # Wav2Vec2 conv stride product → 20 ms
_ONSET_HOP: int = 256     # CNN+BiLSTM log-mel hop → ~16 ms
_VAD_HOP: int = 320       # WebRTC VAD at 20 ms frames → 320 samples


# ---------------------------------------------------------------------------
# Frame ↔ time conversion
# ---------------------------------------------------------------------------

def frames_to_times(
    frame_indices: Union[int, np.ndarray],
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    center: bool = True,
) -> np.ndarray:
    """
    Convert frame indices to time in seconds.

    Args:
        frame_indices: Scalar or array of integer frame indices.
        hop_length:    Hop size in samples.
        sample_rate:   Audio sample rate in Hz.
        center:        If True, timestamps point to frame centers;
                       if False, to frame starts.

    Returns:
        Float64 array of timestamps in seconds.
    """
    indices = np.asarray(frame_indices, dtype=np.float64)
    hop_sec = hop_length / sample_rate
    times = indices * hop_sec
    if center:
        times += hop_sec / 2.0
    return times


def times_to_frames(
    times: Union[float, np.ndarray],
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    center: bool = True,
) -> np.ndarray:
    """
    Convert times in seconds to nearest frame indices.

    Args:
        times:       Scalar or array of time values in seconds.
        hop_length:  Hop size in samples.
        sample_rate: Audio sample rate in Hz.
        center:      If True, shift by half a hop before quantizing
                     (matches center=True in frames_to_times).

    Returns:
        Int64 array of frame indices (clipped to >= 0).
    """
    t = np.asarray(times, dtype=np.float64)
    hop_sec = hop_length / sample_rate
    if center:
        t = t - hop_sec / 2.0
    frames = np.round(t / hop_sec).astype(np.int64)
    return np.clip(frames, 0, None)


def canonical_timestamps(
    n_frames: int,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    center: bool = True,
) -> np.ndarray:
    """
    Build the complete timestamp array for n_frames canonical frames.

    This is the preferred way to generate a uniform timeline for any module
    that outputs frame-level features.

    Returns:
        Float32 array of shape (n_frames,) with timestamps in seconds.
    """
    return frames_to_times(
        np.arange(n_frames), hop_length=hop_length, sample_rate=sample_rate, center=center
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

def duration_to_frames(
    duration_s: float,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> int:
    """Return the number of canonical frames that span a given duration."""
    return int(np.ceil(duration_s * sample_rate / hop_length))


def frames_to_duration(
    n_frames: int,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """Return the duration in seconds covered by n_frames at the given hop."""
    return n_frames * hop_length / sample_rate


def samples_to_frames(
    n_samples: int,
    hop_length: int = HOP_LENGTH,
) -> int:
    """Number of complete frames for a signal of n_samples."""
    return max(0, (n_samples - 1) // hop_length + 1)


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def align_to_grid(
    times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
    kind: Literal["nearest", "linear", "zero"] = "linear",
    fill_value: Union[float, str] = 0.0,
) -> np.ndarray:
    """
    Resample a (times, values) sequence onto target_times.

    For boolean masks use kind='nearest'; for continuous signals use 'linear'.

    Args:
        times:        Source timestamps, shape (N,), monotonically increasing.
        values:       Source values, shape (N,) or (N, D).
        target_times: Target timestamps, shape (M,).
        kind:         Interpolation kind passed to scipy.interpolate.interp1d,
                      or 'nearest' for nearest-neighbour (boolean-safe).
        fill_value:   Value to use outside the source range.
                      Pass 'extrapolate' only with kind='linear'.

    Returns:
        Resampled values at target_times, shape (M,) or (M, D).
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values)
    target_times = np.asarray(target_times, dtype=np.float64)

    if len(times) == 0:
        out_shape = (len(target_times),) + values.shape[1:]
        return np.full(out_shape, fill_value, dtype=values.dtype)

    if len(times) == 1:
        out_shape = (len(target_times),) + values.shape[1:]
        return np.full(out_shape, values[0], dtype=values.dtype)

    if kind == "nearest":
        indices = np.searchsorted(times, target_times, side="left")
        indices = np.clip(indices, 0, len(times) - 1)
        left = np.clip(indices - 1, 0, len(times) - 1)
        right = indices
        left_dist = np.abs(target_times - times[left])
        right_dist = np.abs(target_times - times[right])
        chosen = np.where(left_dist <= right_dist, left, right)
        return values[chosen]

    from scipy.interpolate import interp1d  # type: ignore

    if np.isscalar(fill_value):
        fv = (fill_value, fill_value)
    else:
        fv = fill_value  # type: ignore

    interpolator = interp1d(
        times,
        values,
        axis=0,
        kind=kind,
        bounds_error=False,
        fill_value=fv,
    )
    return interpolator(target_times).astype(values.dtype)


def snap_to_frame(
    time_s: float,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    center: bool = True,
) -> float:
    """
    Snap a continuous timestamp to the nearest canonical frame center/start.

    Useful for aligning phoneme segment boundaries to the frame grid.
    """
    frame = int(times_to_frames(time_s, hop_length=hop_length,
                                 sample_rate=sample_rate, center=center))
    return float(frames_to_times(frame, hop_length=hop_length,
                                  sample_rate=sample_rate, center=center))


def frame_overlap_ratio(
    seg_start: float,
    seg_end: float,
    frame_idx: int,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """
    Fraction of a segment that overlaps frame frame_idx.

    Used to assign per-frame phoneme labels: a frame gets the phoneme whose
    segment has the largest overlap ratio.
    """
    hop_sec = hop_length / sample_rate
    f_start = frame_idx * hop_sec
    f_end = f_start + hop_sec
    overlap = max(0.0, min(seg_end, f_end) - max(seg_start, f_start))
    return overlap / hop_sec
