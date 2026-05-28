"""
utils/audio.py - Unified audio I/O for all VocalCoach modules.

All models share this single entry point for loading, resampling, and
normalizing audio. The function supports both numpy and PyTorch outputs
so both the numpy-based pitch/onset models and the torch-based phoneme
model can use the same backend.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Canonical sample rate used throughout the project
TARGET_SAMPLE_RATE: int = 16000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_audio(
    path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return a mono float32 numpy array.

    Tries soundfile → librosa → stdlib wave in order. Resampling requires
    librosa or torchaudio; a RuntimeError is raised if neither is available.

    Args:
        path:       Path to any soundfile-readable format.
        target_sr:  Desired sample rate. Default: 16 000 Hz.
        mono:       Mix down to mono when True.
        normalize:  Peak-normalize output to [-1, 1] when True.

    Returns:
        audio: float32 array, shape (samples,).
        sr:    Actual sample rate (equals target_sr after resampling).

    Raises:
        FileNotFoundError: File does not exist.
        RuntimeError:      No suitable backend for resampling.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = _read_audio(path, mono)

    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)
        sr = target_sr

    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 1e-8:
            audio = audio / peak

    logger.info(f"[audio] Loaded {path.name}: {len(audio)/sr:.2f}s @ {sr}Hz")
    return audio.astype(np.float32), sr


def load_audio_torch(
    path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = True,
):
    """
    Load audio and return a 1-D float32 torch.Tensor (CPU).

    Wraps load_audio() and converts to tensor. Requires torch.

    Returns:
        waveform: 1-D torch.Tensor, shape (samples,).
        sr:       Sample rate.
    """
    import torch
    audio_np, sr = load_audio(path, target_sr=target_sr, mono=mono, normalize=normalize)
    return torch.from_numpy(audio_np), sr


# ---------------------------------------------------------------------------
# Frame utilities (shared by VAD and preprocessing)
# ---------------------------------------------------------------------------

def frame_audio(
    audio: np.ndarray,
    sr: int,
    frame_duration_ms: int = 20,
) -> np.ndarray:
    """
    Segment audio into fixed-length, non-overlapping frames.

    WebRTC VAD requires 10, 20, or 30 ms frames at supported sample rates.

    Args:
        audio:             1-D float32 array.
        sr:                Sample rate in Hz.
        frame_duration_ms: Frame duration: 10, 20, or 30 ms.

    Returns:
        frames: 2-D array, shape (n_frames, frame_samples).
                Trailing incomplete frames are discarded.

    Raises:
        ValueError: Invalid frame_duration_ms.
    """
    if frame_duration_ms not in (10, 20, 30):
        raise ValueError(
            f"frame_duration_ms must be 10, 20, or 30; got {frame_duration_ms}"
        )
    frame_samples = int(sr * frame_duration_ms / 1000)
    n_frames = len(audio) // frame_samples
    return audio[: n_frames * frame_samples].reshape(n_frames, frame_samples)


def audio_to_pcm16(audio: np.ndarray) -> bytes:
    """
    Convert a float32 frame in [-1, 1] to signed PCM-16 bytes.

    WebRTC VAD requires raw PCM bytes, not float arrays.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def generate_timestamps(
    n_frames: int,
    hop_length: int,
    sample_rate: int,
    center: bool = True,
) -> np.ndarray:
    """
    Build a frame-center timestamp array for a given hop size.

    Args:
        n_frames:    Number of frames.
        hop_length:  Hop size in samples.
        sample_rate: Audio sample rate in Hz.
        center:      When True, timestamps point to frame centers;
                     when False, to frame starts.

    Returns:
        Timestamps array, shape (n_frames,), float32, in seconds.
    """
    hop_sec = hop_length / sample_rate
    offsets = np.arange(n_frames, dtype=np.float32) * hop_sec
    if center:
        offsets += hop_sec / 2.0
    return offsets


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_audio(path: Path, mono: bool) -> Tuple[np.ndarray, int]:
    """Try soundfile → librosa → stdlib wave."""
    try:
        import soundfile as sf  # type: ignore
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1) if (mono and audio.shape[1] > 1) else audio[:, 0]
        return audio, sr
    except ImportError:
        pass

    try:
        import librosa  # type: ignore
        audio, sr = librosa.load(str(path), sr=None, mono=mono, dtype=np.float32)
        return audio, sr
    except ImportError:
        pass

    return _load_wav_stdlib(path)


def _load_wav_stdlib(path: Path) -> Tuple[np.ndarray, int]:
    """Read 16-bit PCM WAV with no external dependencies."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    return audio, sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample using librosa (preferred) or torchaudio."""
    try:
        import librosa  # type: ignore
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        pass

    try:
        import torch
        import torchaudio.transforms as T  # type: ignore
        wf = torch.from_numpy(audio).unsqueeze(0)
        wf = T.Resample(orig_freq=orig_sr, new_freq=target_sr)(wf)
        return wf.squeeze(0).numpy()
    except ImportError:
        pass

    raise RuntimeError(
        f"Cannot resample {orig_sr}→{target_sr}Hz. "
        "Install librosa: pip install librosa"
    )