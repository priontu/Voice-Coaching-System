"""
preprocessing/normalization.py - Signal and spectrogram normalization.

Pure-numpy functions for peak normalization, RMS normalization, and
spectrogram standardization. No audio I/O.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Waveform normalization
# ---------------------------------------------------------------------------

def peak_normalize(
    audio: np.ndarray,
    target_peak: float = 1.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Scale audio so the absolute maximum equals target_peak.

    Silences (peak < epsilon) are returned unchanged to avoid divide-by-zero.
    """
    peak = np.max(np.abs(audio))
    if peak < epsilon:
        return audio.copy()
    return (audio * target_peak / peak).astype(audio.dtype)


def rms_normalize(
    audio: np.ndarray,
    target_rms: float = 0.1,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Scale audio so its RMS equals target_rms.

    Less sensitive to transient peaks than peak normalization; useful when
    comparing RMS-based energy features across recordings.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < epsilon:
        return audio.copy()
    return (audio * target_rms / rms).astype(audio.dtype)


# ---------------------------------------------------------------------------
# Spectrogram normalization
# ---------------------------------------------------------------------------

def normalize_log_mel(
    spec: np.ndarray,
    strategy: str = "global_mean_var",
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Normalize a log-mel spectrogram.

    Args:
        spec:     2-D array of shape (n_mels, n_frames) or (n_frames, n_mels).
        strategy: One of:
                  'global_mean_var' — zero-mean unit-variance across all values,
                  'per_channel'     — per-mel-bin mean-var normalization,
                  'min_max'         — rescale to [0, 1].
        epsilon:  Stability term to avoid divide-by-zero.

    Returns:
        Normalized spectrogram, same shape and dtype as input.
    """
    spec = spec.astype(np.float32)

    if strategy == "global_mean_var":
        mean = spec.mean()
        std = spec.std()
        return (spec - mean) / (std + epsilon)

    if strategy == "per_channel":
        # Normalize each mel bin independently; axis=1 if shape is (n_mels, T)
        mean = spec.mean(axis=1, keepdims=True)
        std = spec.std(axis=1, keepdims=True)
        return (spec - mean) / (std + epsilon)

    if strategy == "min_max":
        lo, hi = spec.min(), spec.max()
        if hi - lo < epsilon:
            return np.zeros_like(spec)
        return (spec - lo) / (hi - lo)

    raise ValueError(
        f"Unknown normalization strategy '{strategy}'. "
        "Choose 'global_mean_var', 'per_channel', or 'min_max'."
    )


def clip_log_mel(
    spec: np.ndarray,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Clip a log-mel spectrogram from below so the dynamic range is at most top_db.

    Equivalent to librosa's top_db clipping. Modifies nothing if all values are
    within range.
    """
    spec = spec.astype(np.float32)
    return np.maximum(spec, spec.max() - top_db)
