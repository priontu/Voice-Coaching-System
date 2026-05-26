"""
utils.py - Shared audio I/O utilities for the VAD + Pitch pipeline.

Handles WAV loading, resampling, normalization, PCM conversion, and
JSON serialization in the format expected by pitch_score.py.
"""

import json
import logging
import wave
from pathlib import Path
from typing import Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# WebRTC VAD requires audio at this rate
TARGET_SAMPLE_RATE: int = 16000


def get_best_device() -> str:
    """
    Return the best available PyTorch device string.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    Falls back to CPU gracefully if torch is not installed.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            name = torch.cuda.get_device_name(device)
            logger.info(f"[utils] GPU detected: {name} → using {device}")
            return device
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            logger.info("[utils] Apple MPS detected → using mps")
            return "mps"
    except ImportError:
        pass
    logger.info("[utils] No GPU detected → using cpu")
    return "cpu"


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(
    path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file, convert to mono, resample, and optionally normalize.

    Tries soundfile first (broadest format support), falls back to librosa,
    and finally to the stdlib wave module (PCM only, no resampling).

    Args:
        path: Path to the audio file.
        target_sr: Target sample rate in Hz. Default: 16000.
        mono: Average channels to produce a mono signal.
        normalize: Peak-normalize the output to [-1, 1].

    Returns:
        audio: Float32 numpy array, shape (samples,).
        sr: Actual sample rate (equals target_sr after resampling).

    Raises:
        FileNotFoundError: Audio file does not exist.
        RuntimeError: No suitable audio backend available for resampling.
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

    logger.info(
        f"[utils] Loaded {path.name}: {len(audio)/sr:.2f}s @ {sr}Hz"
    )
    return audio.astype(np.float32), sr


def _read_audio(path: Path, mono: bool) -> Tuple[np.ndarray, int]:
    """Try soundfile → librosa → stdlib wave to read audio."""
    try:
        import soundfile as sf  # type: ignore
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if mono and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0]
        return audio, sr
    except ImportError:
        pass

    try:
        import librosa  # type: ignore
        audio, sr = librosa.load(str(path), sr=None, mono=mono, dtype=np.float32)
        return audio, sr
    except ImportError:
        pass

    # Stdlib fallback — PCM 16-bit only
    return _load_wav_stdlib(path)


def _load_wav_stdlib(path: Path) -> Tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV with no external dependencies."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using librosa (required for rate conversion)."""
    try:
        import librosa  # type: ignore
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        raise RuntimeError(
            f"Cannot resample {orig_sr}→{target_sr}Hz without librosa. "
            "Install with: pip install librosa"
        )


# ---------------------------------------------------------------------------
# Frame segmentation
# ---------------------------------------------------------------------------

def frame_audio(
    audio: np.ndarray,
    sr: int,
    frame_duration_ms: int = 20,
) -> np.ndarray:
    """
    Segment audio into fixed-duration, non-overlapping frames.

    WebRTC VAD accepts only 10, 20, or 30 ms frames at supported sample rates.

    Args:
        audio: 1-D float32 array.
        sr: Sample rate in Hz.
        frame_duration_ms: Frame length in ms. Must be 10, 20, or 30.

    Returns:
        frames: 2-D array of shape (n_frames, frame_samples). The trailing
                samples that do not fill a complete frame are discarded.

    Raises:
        ValueError: Invalid frame_duration_ms.
    """
    if frame_duration_ms not in (10, 20, 30):
        raise ValueError(
            f"frame_duration_ms must be 10, 20, or 30; got {frame_duration_ms}"
        )

    frame_samples = int(sr * frame_duration_ms / 1000)
    n_frames = len(audio) // frame_samples
    audio_trimmed = audio[: n_frames * frame_samples]
    return audio_trimmed.reshape(n_frames, frame_samples)


# ---------------------------------------------------------------------------
# PCM conversion (for WebRTC VAD)
# ---------------------------------------------------------------------------

def audio_to_pcm16(audio: np.ndarray) -> bytes:
    """
    Convert a float32 frame in [-1, 1] to raw signed PCM16 bytes.

    WebRTC VAD requires raw PCM bytes, not float arrays.

    Args:
        audio: 1-D float32 array, values in [-1, 1].

    Returns:
        bytes: Raw little-endian int16 PCM data.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# JSON I/O — compatible with pitch_score.py's load_pitch_data()
# ---------------------------------------------------------------------------

def save_pitch_json(
    timestamps: np.ndarray,
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    output_path: Union[str, Path],
    audio_path: str = "",
    sample_rate: int = TARGET_SAMPLE_RATE,
    hop_length: int = 160,
) -> None:
    """
    Write pitch data to JSON in the format expected by pitch_score.py.

    The output is a strict superset of the existing schema:
        {"frames": [{"time": float, "f0": float, "voiced": bool, ...}]}

    The extra "voiced" and "midi" fields are ignored by the existing
    pitch_score.py load_pitch_data(), so this is fully backward-compatible.

    Args:
        timestamps: Frame timestamps in seconds, shape (T,).
        f0: F0 values in Hz (0.0 for unvoiced), shape (T,).
        voiced_mask: Boolean voiced flags, shape (T,).
        output_path: Destination JSON file path.
        audio_path: Optional source audio path to embed in metadata.
        sample_rate: Sample rate for metadata.
        hop_length: Hop length for metadata.
    """
    frames = []
    for t, freq, v in zip(timestamps, f0, voiced_mask):
        frames.append({
            "time": float(t),
            "f0": float(freq),
            "voiced": bool(v),
            "midi": float(69 + 12 * np.log2(freq / 440.0)) if freq > 0 else None,
        })

    output = {
        "audio_path": audio_path,
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "num_frames": len(frames),
        "frames": frames,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fp:
        json.dump(output, fp, indent=2)

    logger.info(f"[utils] Saved {len(frames)} frames → {output_path}")


def load_pitch_json(
    path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pitch data written by save_pitch_json().

    Falls back gracefully if "voiced" is absent (legacy files produced
    by test_pitch.py will have f0 > 0 treated as voiced).

    Returns:
        timestamps: shape (T,) float32
        f0: shape (T,) float32
        voiced_mask: shape (T,) bool
    """
    with open(path, "r") as fp:
        data = json.load(fp)

    frames = data["frames"]
    timestamps = np.array([f["time"] for f in frames], dtype=np.float32)
    f0 = np.array([f["f0"] for f in frames], dtype=np.float32)
    voiced_mask = np.array(
        [f.get("voiced", f["f0"] > 0) for f in frames], dtype=bool
    )
    return timestamps, f0, voiced_mask
