"""
preprocessing/spectrograms.py - Log-mel spectrogram computation.

Shared spectrogram backend used by the onset/offset model and any future
models that consume mel features. Replaces the duplicate implementation in
models/onset_offset/spec_utils.py for new code; the original is kept for
backward compatibility with the existing NoteDetector checkpoint.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from preprocessing.timestamps import HOP_LENGTH, SAMPLE_RATE, canonical_timestamps
from preprocessing.normalization import normalize_log_mel, clip_log_mel


# ---------------------------------------------------------------------------
# Log-mel spectrogram
# ---------------------------------------------------------------------------

def compute_log_mel(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = 1024,
    hop_length: int = HOP_LENGTH,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    top_db: float = 80.0,
    center: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a clipped log-mel spectrogram.

    Attempts librosa first, then a pure-numpy fallback.

    Args:
        audio:       1-D float32 array (samples,).
        sample_rate: Audio sample rate in Hz.
        n_fft:       FFT size.
        hop_length:  Hop size in samples.
        n_mels:      Number of mel filter banks.
        fmin:        Minimum frequency in Hz.
        fmax:        Maximum frequency in Hz. None → sample_rate / 2.
        top_db:      Dynamic range clipping in dB.
        center:      If True, pad audio so frame centers align to sample positions
                     (librosa default). The onset/offset model trained with
                     center=False — keep that for compatibility.

    Returns:
        spec:       2-D float32 array of shape (n_mels, n_frames).
        timestamps: 1-D float32 array of frame-center times in seconds.
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    try:
        spec = _librosa_log_mel(audio, sample_rate, n_fft, hop_length, n_mels, fmin, fmax, center)
    except ImportError:
        spec = _numpy_log_mel(audio, sample_rate, n_fft, hop_length, n_mels, fmin, fmax)

    spec = clip_log_mel(spec, top_db=top_db)
    n_frames = spec.shape[1]
    timestamps = canonical_timestamps(n_frames, hop_length=hop_length, sample_rate=sample_rate)
    return spec, timestamps


def compute_log_mel_torch(
    audio,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = 1024,
    hop_length: int = HOP_LENGTH,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    top_db: float = 80.0,
):
    """
    Compute a log-mel spectrogram and return it as a 4-D torch tensor.

    Returns a tensor of shape (1, 1, n_mels, n_frames) ready for the
    CNN+BiLSTM onset/offset model.

    Args:
        audio: 1-D numpy array or torch.Tensor (CPU), shape (samples,).
    """
    import torch

    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    spec, timestamps = compute_log_mel(
        audio_np, sample_rate=sample_rate, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
        fmin=fmin, fmax=fmax, top_db=top_db,
    )
    tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, T)
    return tensor, timestamps


# ---------------------------------------------------------------------------
# Internal backends
# ---------------------------------------------------------------------------

def _librosa_log_mel(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    center: bool,
) -> np.ndarray:
    import librosa  # type: ignore

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        center=center,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def _numpy_log_mel(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Minimal numpy fallback (no padding, Hann window)."""
    window = np.hanning(n_fft)
    n_frames = max(0, (len(audio) - n_fft) // hop_length + 1)
    power = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)

    for i in range(n_frames):
        frame = audio[i * hop_length : i * hop_length + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        power[:, i] = spectrum

    mel_fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel_power = mel_fb @ power
    log_mel = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    return log_mel.astype(np.float32)


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft // 2 + 1)."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(left, center):
            if center > left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb
