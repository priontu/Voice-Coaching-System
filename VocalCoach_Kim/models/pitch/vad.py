"""
models/pitch/vad.py - Voice Activity Detection using py-webrtcvad.

Produces a frame-level voiced/unvoiced binary mask for audio signals.
Runs entirely on CPU.

Changes from the original Pitch Model w VAD/vad.py:
  - frame_audio() and audio_to_pcm16() imported from utils.audio (shared)
  - All VAD logic is UNCHANGED
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import webrtcvad  # type: ignore
    _HAS_WEBRTCVAD = True
except ImportError:
    _HAS_WEBRTCVAD = False
    logging.getLogger(__name__).warning(
        "py-webrtcvad not found. Energy-based VAD will be used as fallback.\n"
        "Install with: pip install webrtcvad-wheels"
    )

from utils.audio import audio_to_pcm16, frame_audio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VADConfig:
    """All tunable parameters for the WebRTC VAD module."""

    aggressiveness: int = 2
    frame_duration_ms: int = 20
    sample_rate: int = 16000
    smoothing_window: int = 5
    energy_threshold_db: float = -40.0

    @classmethod
    def from_yaml(cls, cfg: dict) -> "VADConfig":
        v = cfg.get("vad", {})
        return cls(
            aggressiveness=v.get("aggressiveness", 2),
            frame_duration_ms=v.get("frame_duration_ms", 20),
            sample_rate=cfg.get("audio", {}).get("sample_rate", 16000),
            smoothing_window=v.get("smoothing_window", 5),
            energy_threshold_db=v.get("energy_threshold_db", -40.0),
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WebRTCVAD:
    """
    Frame-level VAD wrapper with automatic fallback to energy-based detection.

    Accepts float32 audio and returns a boolean voiced mask aligned to
    fixed-duration frames, plus the center timestamp of each frame.
    """

    def __init__(self, config: Optional[VADConfig] = None) -> None:
        self.config = config or VADConfig()
        self._validate()

        if _HAS_WEBRTCVAD:
            self._vad = webrtcvad.Vad(self.config.aggressiveness)
            logger.info(
                f"[VAD] WebRTC VAD ready "
                f"(aggressiveness={self.config.aggressiveness}, "
                f"frame={self.config.frame_duration_ms}ms)"
            )
        else:
            self._vad = None
            logger.warning(
                f"[VAD] Energy-based fallback active "
                f"(threshold={self.config.energy_threshold_db}dBFS)"
            )

    def run(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect voiced frames in a 1-D audio array.

        Returns:
            voiced_mask: Boolean array, shape (n_vad_frames,).
            timestamps:  Center timestamps of each VAD frame (seconds).
        """
        if sr != self.config.sample_rate:
            raise ValueError(
                f"Audio sr={sr} does not match VAD sr={self.config.sample_rate}. "
                "Use utils.audio.load_audio() which resamples to the target rate."
            )

        frames = frame_audio(audio, sr, self.config.frame_duration_ms)
        n_frames = len(frames)
        frame_sec = self.config.frame_duration_ms / 1000.0
        timestamps = (np.arange(n_frames) + 0.5) * frame_sec

        raw_mask = self._run_webrtc(frames) if _HAS_WEBRTCVAD else self._run_energy(frames)
        smoothed = _smooth_mask(raw_mask, self.config.smoothing_window)

        n_voiced = int(np.sum(smoothed))
        logger.info(
            f"[VAD] {n_frames} frames, {n_voiced} voiced "
            f"({100*n_voiced/max(n_frames,1):.1f}%)"
        )
        return smoothed.astype(bool), timestamps

    def get_voiced_segments(
        self,
        voiced_mask: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """Extract contiguous voiced regions as (start_sec, end_sec) pairs."""
        half_frame = self.config.frame_duration_ms / 2000.0
        segments: List[Tuple[float, float]] = []
        in_seg = False
        seg_start = 0.0

        for is_voiced, t in zip(voiced_mask, timestamps):
            if is_voiced and not in_seg:
                seg_start = t - half_frame
                in_seg = True
            elif not is_voiced and in_seg:
                segments.append((seg_start, t - half_frame))
                in_seg = False

        if in_seg:
            segments.append((seg_start, float(timestamps[-1]) + half_frame))

        return segments

    def _run_webrtc(self, frames: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(frames), dtype=bool)
        for i, frame in enumerate(frames):
            pcm = audio_to_pcm16(frame)
            try:
                mask[i] = self._vad.is_speech(pcm, self.config.sample_rate)
            except Exception as exc:
                logger.debug(f"[VAD] Frame {i} rejected by WebRTC: {exc}")
                mask[i] = False
        return mask

    def _run_energy(self, frames: np.ndarray) -> np.ndarray:
        rms_energy = np.mean(frames ** 2, axis=1)
        energy_db = 10.0 * np.log10(np.maximum(rms_energy, 1e-12))
        return energy_db > self.config.energy_threshold_db

    def _validate(self) -> None:
        if self.config.aggressiveness not in (0, 1, 2, 3):
            raise ValueError("aggressiveness must be 0, 1, 2, or 3")
        if self.config.frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        if self.config.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("sample_rate must be one of 8000, 16000, 32000, 48000")


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _smooth_mask(mask: np.ndarray, window: int) -> np.ndarray:
    """Apply a sliding median filter to remove isolated voiced/unvoiced flips."""
    if window <= 1:
        return mask.copy()

    smoothed = np.empty_like(mask)
    half = window // 2

    for i in range(len(mask)):
        lo = max(0, i - half)
        hi = min(len(mask), i + half + 1)
        smoothed[i] = np.median(mask[lo:hi].astype(np.float32)) >= 0.5

    return smoothed


def run_vad(
    audio: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_duration_ms: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function: create a WebRTCVAD with default settings and run it."""
    config = VADConfig(
        aggressiveness=aggressiveness,
        frame_duration_ms=frame_duration_ms,
        sample_rate=sr,
    )
    return WebRTCVAD(config).run(audio, sr)
