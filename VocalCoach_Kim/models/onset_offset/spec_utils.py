"""
models/onset_offset/spec_utils.py - Spectrogram and peak-picking utilities.

Domain-specific utilities for the onset/offset detection module. These live
here (not in utils/) because they are tightly coupled to the model's input
format (log-mel spectrogram) and output interpretation (peak-picking).

Unchanged from the original Note Model/utils.py (audio loading section removed
— now handled by utils.audio.load_audio).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torchaudio
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0,
) -> torch.Tensor:
    """
    Compute log-mel spectrogram.

    Args:
        waveform: [1, N] mono audio tensor.

    Returns:
        log_mel: [1, n_mels, T]
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
    )
    mel_spec = mel_transform(waveform)  # [1, n_mels, T]
    return torch.log(mel_spec + 1e-9)


def normalize_spectrogram(spec: torch.Tensor) -> torch.Tensor:
    """Per-sample zero-mean unit-variance normalization."""
    return (spec - spec.mean()) / (spec.std() + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Frame times
# ─────────────────────────────────────────────────────────────────────────────

def frames_to_time(
    n_frames: int,
    hop_length: int = 256,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Return centre time (seconds) for each spectrogram frame."""
    return np.arange(n_frames) * hop_length / sample_rate


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian soft-label construction (used by training)
# ─────────────────────────────────────────────────────────────────────────────

def build_onset_labels(
    onsets: List[float],
    frame_times: np.ndarray,
    sigma: float = 0.02,
) -> np.ndarray:
    """Frame-level Gaussian soft targets centred on note onsets."""
    return _gaussian_labels(onsets, frame_times, sigma)


def build_offset_labels(
    offsets: List[float],
    frame_times: np.ndarray,
    sigma: float = 0.02,
) -> np.ndarray:
    """Frame-level Gaussian soft targets centred on note offsets."""
    return _gaussian_labels(offsets, frame_times, sigma)


def _gaussian_labels(
    boundaries: List[float],
    frame_times: np.ndarray,
    sigma: float,
) -> np.ndarray:
    labels = np.zeros(len(frame_times), dtype=np.float32)
    for t in boundaries:
        gauss = np.exp(-0.5 * ((frame_times - t) / sigma) ** 2)
        labels = np.maximum(labels, gauss)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Peak picking
# ─────────────────────────────────────────────────────────────────────────────

def peak_pick_onsets(
    probs: np.ndarray,
    frame_times: np.ndarray,
    threshold: float = 0.3,
    min_distance_frames: int = 3,
) -> List[float]:
    """Extract onset timestamps from a probability curve."""
    peaks, _ = find_peaks(probs, height=threshold, distance=min_distance_frames)
    return sorted(float(frame_times[i]) for i in peaks)


def peak_pick_offsets(
    probs: np.ndarray,
    frame_times: np.ndarray,
    threshold: float = 0.3,
    min_distance_frames: int = 3,
) -> List[float]:
    """Extract offset timestamps from a probability curve."""
    peaks, _ = find_peaks(probs, height=threshold, distance=min_distance_frames)
    return sorted(float(frame_times[i]) for i in peaks)


def pair_onsets_offsets(
    onsets: List[float],
    offsets: List[float],
) -> List[dict]:
    """
    Greedily pair onset/offset times into note boundaries.

    Each onset is matched with the next available offset that comes after it.
    """
    notes = []
    offsets_sorted = sorted(offsets)
    off_idx = 0

    for onset in sorted(onsets):
        while off_idx < len(offsets_sorted) and offsets_sorted[off_idx] <= onset:
            off_idx += 1

        if off_idx < len(offsets_sorted):
            offset = offsets_sorted[off_idx]
            notes.append({
                "onset_time": onset,
                "offset_time": offset,
                "duration": round(offset - onset, 6),
            })
            off_idx += 1
        else:
            notes.append({"onset_time": onset, "offset_time": None, "duration": None})

    return notes
