"""
preprocessing/audio_pipeline.py - Centralized audio preprocessing orchestrator.

AudioPreprocessor is the single entry point for turning a raw audio file into
the numeric arrays each model expects. It delegates to utils.audio for I/O,
framing.py for frame bookkeeping, and spectrograms.py for mel features.

Models should call AudioPreprocessor.process_for_<model>() rather than
importing utils.audio directly, so timing constants stay consistent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from preprocessing.timestamps import (
    FRAME_DURATION,
    HOP_LENGTH,
    SAMPLE_RATE,
    canonical_timestamps,
)
from preprocessing.framing import canonical_frame_count, n_frames_from_samples
from preprocessing.spectrograms import compute_log_mel, compute_log_mel_torch
from preprocessing.normalization import peak_normalize
from utils.audio import load_audio, load_audio_torch
from utils.types import AudioFeatures

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Stateless (no model weights) preprocessing coordinator.

    Instantiate once with a config dict (matching preprocessing.yaml) and
    reuse for all files. Each process_for_* method returns the exact array
    shape and dtype that the corresponding model expects.
    """

    def __init__(self, cfg: Optional[Dict] = None) -> None:
        cfg = cfg or {}
        audio_cfg = cfg.get("audio", {})
        spec_cfg = cfg.get("spectrogram", {})

        self.sample_rate: int = int(audio_cfg.get("sample_rate", SAMPLE_RATE))
        self.hop_length: int = int(spec_cfg.get("hop_length", HOP_LENGTH))
        self.n_fft: int = int(spec_cfg.get("n_fft", 1024))
        self.n_mels: int = int(spec_cfg.get("n_mels", 80))
        self.fmin: float = float(spec_cfg.get("fmin", 0.0))
        self.fmax: Optional[float] = spec_cfg.get("fmax", None)
        self.normalize: bool = bool(audio_cfg.get("normalize", True))

    # ------------------------------------------------------------------
    # General-purpose
    # ------------------------------------------------------------------

    def process(self, path: Union[str, Path]) -> AudioFeatures:
        """
        Load and normalize audio. Returns an AudioFeatures container.

        Use this when you need access to the raw waveform rather than
        model-specific pre-processed arrays.
        """
        audio, sr = load_audio(
            path, target_sr=self.sample_rate, mono=True, normalize=self.normalize
        )
        return AudioFeatures(
            audio=audio,
            sample_rate=sr,
            duration_s=len(audio) / sr,
            source_path=str(path),
        )

    # ------------------------------------------------------------------
    # Model-specific entry points
    # ------------------------------------------------------------------

    def process_for_phoneme(self, path: Union[str, Path]):
        """
        Load audio for the phoneme model.

        Returns a 1-D CPU float32 torch.Tensor, the exact shape that
        PhonemeInferenceModel.predict() expects.
        """
        waveform, _ = load_audio_torch(
            path, target_sr=self.sample_rate, mono=True, normalize=self.normalize
        )
        logger.debug(f"[preprocess] phoneme input shape: {tuple(waveform.shape)}")
        return waveform  # shape: (samples,)

    def process_for_pitch(self, path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio for the pitch/VAD pipeline.

        Returns (audio, sample_rate) where audio is a float32 numpy array.
        The pitch pipeline (torchcrepe) and VAD both consume numpy arrays.
        """
        audio, sr = load_audio(
            path, target_sr=self.sample_rate, mono=True, normalize=self.normalize
        )
        logger.debug(f"[preprocess] pitch input length: {len(audio)} @ {sr}Hz")
        return audio, sr

    def process_for_onset_offset(
        self,
        path: Union[str, Path],
        hop_length: Optional[int] = None,
    ):
        """
        Load audio and compute a log-mel spectrogram for the onset/offset model.

        The onset/offset model was trained with hop_length=256; this is preserved
        for inference. Pass hop_length explicitly to override.

        Returns:
            tensor: 4-D torch.Tensor of shape (1, 1, n_mels, n_frames).
            timestamps: float32 array of frame-center times, shape (n_frames,).
        """
        # Preserve the trained hop for this specific model
        hop = hop_length if hop_length is not None else 256

        audio, sr = load_audio(
            path, target_sr=self.sample_rate, mono=True, normalize=self.normalize
        )
        tensor, timestamps = compute_log_mel_torch(
            audio,
            sample_rate=sr,
            n_fft=self.n_fft,
            hop_length=hop,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        logger.debug(f"[preprocess] onset/offset spectrogram: {tuple(tensor.shape)}")
        return tensor, timestamps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def canonical_frame_count(self, path: Union[str, Path]) -> int:
        """
        Number of canonical 10-ms frames for an audio file, without loading it.

        Uses AudioFeatures.duration_s to avoid holding the waveform in memory.
        """
        feats = self.process(path)
        n_samples = int(feats.duration_s * self.sample_rate)
        return canonical_frame_count(n_samples, hop_length=HOP_LENGTH)

    def canonical_timestamps_for(self, path: Union[str, Path]) -> np.ndarray:
        """Build the canonical timestamp array for a given audio file."""
        n_frames = self.canonical_frame_count(path)
        return canonical_timestamps(n_frames, hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Module-level convenience instance (uses config defaults)
# ---------------------------------------------------------------------------

_default_preprocessor: Optional[AudioPreprocessor] = None


def get_preprocessor(cfg: Optional[Dict] = None) -> AudioPreprocessor:
    """
    Return the shared default AudioPreprocessor, or a new one with cfg.

    Using a shared instance avoids re-reading the config YAML on every call
    while remaining thread-safe (preprocessors have no mutable state).
    """
    global _default_preprocessor
    if cfg is not None:
        return AudioPreprocessor(cfg)
    if _default_preprocessor is None:
        _default_preprocessor = AudioPreprocessor()
    return _default_preprocessor
