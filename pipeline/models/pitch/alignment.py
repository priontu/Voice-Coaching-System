"""
models/pitch/alignment.py - Frame timing alignment between VAD and pitch.

VAD operates at a fixed frame rate (e.g. 20ms → 50 fps) while the pitch
model operates at its own hop rate (e.g. 10ms → 100 fps). This module
provides utilities to resample and synchronize these two timelines before
the fusion step.

Unchanged from the original Pitch Model w VAD/alignment.py.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d  # type: ignore

logger = logging.getLogger(__name__)


def align_vad_to_pitch(
    vad_mask: np.ndarray,
    vad_times: np.ndarray,
    pitch_times: np.ndarray,
) -> np.ndarray:
    """
    Align a VAD voiced mask from VAD timestamps to pitch frame timestamps.

    Uses nearest-neighbor mapping (correct for binary masks).

    Args:
        vad_mask:    Boolean voiced mask at VAD frame rate, shape (N,).
        vad_times:   Center timestamps of VAD frames in seconds, shape (N,).
        pitch_times: Center timestamps of pitch frames in seconds, shape (M,).

    Returns:
        aligned_mask: Boolean voiced mask at pitch frame rate, shape (M,).
    """
    if len(vad_times) == 0 or len(pitch_times) == 0:
        logger.warning("[Align] Empty input — returning all-unvoiced mask.")
        return np.zeros(len(pitch_times), dtype=bool)

    logger.debug(
        f"[Align] VAD: {len(vad_times)} frames → pitch: {len(pitch_times)} frames"
    )
    return resample_mask(vad_mask, vad_times, pitch_times, method="nearest")


def resample_mask(
    source_mask: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
    method: str = "nearest",
) -> np.ndarray:
    """
    Resample a boolean mask from one set of frame timestamps to another.

    Out-of-range target timestamps are filled with the boundary source value.

    Args:
        source_mask:  Boolean mask at source frame rate, shape (N,).
        source_times: Source frame center timestamps in seconds, shape (N,).
        target_times: Target frame center timestamps in seconds, shape (M,).
        method:       "nearest" (binary-safe) or "linear" (soft threshold).

    Returns:
        Resampled boolean mask, shape (M,).
    """
    float_mask = source_mask.astype(np.float32)
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
    return resampled >= 0.5


def compute_timestamps(
    n_frames: int,
    hop_length_samples: int,
    sample_rate: int,
    center: bool = True,
) -> np.ndarray:
    """
    Build a frame-center timestamp array for a given hop size.

    Args:
        n_frames:           Number of frames.
        hop_length_samples: Hop size in samples.
        sample_rate:        Audio sample rate in Hz.
        center:             When True, timestamps point to frame centers.

    Returns:
        Timestamps array, shape (n_frames,), float32, in seconds.
    """
    hop_sec = hop_length_samples / sample_rate
    offsets = np.arange(n_frames, dtype=np.float32) * hop_sec
    if center:
        offsets += hop_sec / 2.0
    return offsets


def synchronize_arrays(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Truncate multiple arrays to the length of the shortest one.

    Guards against minor off-by-one differences between different
    extraction steps (VAD vs. pitch frame counts).
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
