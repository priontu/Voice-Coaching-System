"""
alignment.py - Frame timing alignment between VAD and pitch estimation.

VAD operates at a fixed frame rate (e.g. 20ms → 50 fps) while the pitch
model operates at its own hop rate (e.g. 10ms → 100 fps). This module
provides robust utilities to resample and synchronize these two timelines
before the fusion step.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core alignment
# ---------------------------------------------------------------------------

def align_vad_to_pitch(
    vad_mask: np.ndarray,
    vad_times: np.ndarray,
    pitch_times: np.ndarray,
) -> np.ndarray:
    """
    Align a VAD voiced mask from VAD frame timestamps to pitch frame timestamps.

    This is the primary function called during the inference pipeline.
    Uses nearest-neighbor mapping, which is correct for binary masks because
    linear interpolation on 0/1 values produces meaningless fractional results.

    Args:
        vad_mask: Boolean voiced mask at VAD frame rate, shape (N,).
        vad_times: Center timestamps of VAD frames in seconds, shape (N,).
        pitch_times: Center timestamps of pitch frames in seconds, shape (M,).

    Returns:
        aligned_mask: Boolean voiced mask at pitch frame rate, shape (M,).
    """
    if len(vad_times) == 0 or len(pitch_times) == 0:
        logger.warning("[Align] Empty input — returning all-unvoiced mask.")
        return np.zeros(len(pitch_times), dtype=bool)

    logger.debug(
        f"[Align] VAD: {len(vad_times)} frames @ "
        f"{1/(vad_times[1]-vad_times[0]):.0f}fps → "
        f"pitch: {len(pitch_times)} frames @ "
        f"{1/(pitch_times[1]-pitch_times[0]) if len(pitch_times)>1 else 0:.0f}fps"
    )

    return resample_mask(
        source_mask=vad_mask,
        source_times=vad_times,
        target_times=pitch_times,
        method="nearest",
    )


def resample_mask(
    source_mask: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
    method: str = "nearest",
) -> np.ndarray:
    """
    Resample a boolean mask from one set of frame timestamps to another.

    Out-of-range target timestamps are filled with the boundary value of the
    source mask (no extrapolation artifacts).

    Args:
        source_mask: Boolean mask at source frame rate, shape (N,).
        source_times: Source frame center timestamps in seconds, shape (N,).
        target_times: Target frame center timestamps in seconds, shape (M,).
        method: "nearest" (binary-safe, default) or "linear" (soft threshold).

    Returns:
        Resampled boolean mask, shape (M,).
    """
    float_mask = source_mask.astype(np.float32)

    # Boundary fill values — use the first/last source frame's value so that
    # target frames that extend slightly beyond the source range stay stable.
    fill_left = float(float_mask[0])
    fill_right = float(float_mask[-1])

    interp_fn = interp1d(
        source_times,
        float_mask,
        kind="nearest" if method == "nearest" else "linear",
        bounds_error=False,
        fill_value=(fill_left, fill_right),
    )

    resampled = interp_fn(target_times).astype(np.float32)

    # Threshold back to binary (handles both "nearest" and "linear" cases)
    return resampled >= 0.5


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------

def compute_timestamps(
    n_frames: int,
    hop_length_samples: int,
    sample_rate: int,
    center: bool = True,
) -> np.ndarray:
    """
    Build a frame-center timestamp array for a given hop size.

    Matches the convention used by torchcrepe and librosa: frame index i
    corresponds to audio samples [i*hop, (i+1)*hop).

    Args:
        n_frames: Number of frames.
        hop_length_samples: Hop size in samples.
        sample_rate: Audio sample rate in Hz.
        center: If True, timestamp is the center of the frame window;
                if False, it is the start of the frame window.

    Returns:
        Timestamps array, shape (n_frames,), dtype float32, units: seconds.
    """
    hop_sec = hop_length_samples / sample_rate
    offsets = np.arange(n_frames, dtype=np.float32) * hop_sec
    if center:
        offsets += hop_sec / 2.0
    return offsets


# ---------------------------------------------------------------------------
# Array synchronization
# ---------------------------------------------------------------------------

def synchronize_arrays(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Truncate multiple arrays to the length of the shortest one.

    Ensures timestamps, f0, and voiced_mask are all perfectly aligned
    despite minor off-by-one differences between different extraction steps.

    Args:
        *arrays: One or more 1-D numpy arrays to synchronize.

    Returns:
        Tuple of truncated arrays, all with the same length.

    Example:
        times, f0, voiced = synchronize_arrays(times, f0, voiced)
    """
    if not arrays:
        return ()

    min_len = min(len(a) for a in arrays)

    if len(set(len(a) for a in arrays)) > 1:
        logger.debug(
            f"[Align] Synchronizing arrays from lengths "
            f"{[len(a) for a in arrays]} → {min_len}"
        )

    return tuple(a[:min_len] for a in arrays)


def vad_frame_to_time(frame_index: int, frame_duration_ms: int) -> float:
    """Convert a VAD frame index to its center timestamp in seconds."""
    return (frame_index + 0.5) * frame_duration_ms / 1000.0


def time_to_vad_frame(time_sec: float, frame_duration_ms: int) -> int:
    """Convert a timestamp in seconds to the nearest VAD frame index."""
    return int(time_sec * 1000.0 / frame_duration_ms)
