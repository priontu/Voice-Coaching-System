"""
vad.py - Voice Activity Detection using py-webrtcvad.

Produces a frame-level voiced/unvoiced binary mask for audio signals.
Runs entirely on CPU — GPU is not needed or used here.

Fallback: energy-based VAD when py-webrtcvad is not installed.
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
        "Install the real VAD with: pip install webrtcvad-wheels"
    )

from utils import audio_to_pcm16, frame_audio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VADConfig:
    """All tunable parameters for the WebRTC VAD module."""

    aggressiveness: int = 2
    """
    WebRTC aggressiveness mode.
      0 = least aggressive (keeps more speech, tolerates more noise)
      3 = most aggressive (filters most noise, may cut some speech)
    Recommended for singing: 1 or 2.
    """

    frame_duration_ms: int = 20
    """Frame length in ms fed to WebRTC VAD. Must be 10, 20, or 30."""

    sample_rate: int = 16000
    """
    Sample rate expected by WebRTC VAD.
    Supported values: 8000, 16000, 32000, 48000 Hz.
    """

    smoothing_window: int = 5
    """
    Median filter window (in frames) applied to the raw binary mask.
    Eliminates isolated voiced/unvoiced flips. Set to 1 to disable.
    """

    energy_threshold_db: float = -40.0
    """
    Energy threshold (dBFS) for the energy-based fallback VAD.
    Only used when py-webrtcvad is not installed.
    """


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WebRTCVAD:
    """
    Frame-level VAD wrapper with automatic fallback to energy-based detection.

    Accepts float32 audio and returns a boolean voiced mask aligned to
    fixed-duration frames, plus the center timestamp of each frame.

    Usage:
        config = VADConfig(aggressiveness=2, frame_duration_ms=20)
        vad = WebRTCVAD(config)
        voiced_mask, vad_times = vad.run(audio, sr=16000)
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect voiced frames in a 1-D audio array.

        Args:
            audio: Float32 mono audio, shape (samples,).
            sr: Sample rate — must match config.sample_rate.

        Returns:
            voiced_mask: Boolean array, shape (n_vad_frames,).
                         True = frame contains speech/singing.
            timestamps: Center timestamps of each VAD frame, shape (n_vad_frames,).
                        Units: seconds.

        Raises:
            ValueError: Sample rate mismatch.
        """
        if sr != self.config.sample_rate:
            raise ValueError(
                f"Audio sr={sr} does not match VAD sr={self.config.sample_rate}. "
                "Use utils.load_audio() which resamples to the target rate."
            )

        frames = frame_audio(audio, sr, self.config.frame_duration_ms)
        n_frames = len(frames)
        frame_sec = self.config.frame_duration_ms / 1000.0
        timestamps = (np.arange(n_frames) + 0.5) * frame_sec

        if _HAS_WEBRTCVAD:
            raw_mask = self._run_webrtc(frames)
        else:
            raw_mask = self._run_energy(frames)

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
        """
        Extract contiguous voiced regions as (start_sec, end_sec) pairs.

        Useful for visualization and for calculating total voiced duration.

        Args:
            voiced_mask: Boolean frame mask, shape (n_frames,).
            timestamps: Frame center timestamps, shape (n_frames,).

        Returns:
            List of (start_sec, end_sec) tuples in chronological order.
        """
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

    # ------------------------------------------------------------------
    # Internal backends
    # ------------------------------------------------------------------

    def _run_webrtc(self, frames: np.ndarray) -> np.ndarray:
        """Run each frame through the WebRTC VAD and return a raw binary mask."""
        mask = np.zeros(len(frames), dtype=bool)
        for i, frame in enumerate(frames):
            pcm = audio_to_pcm16(frame)
            try:
                mask[i] = self._vad.is_speech(pcm, self.config.sample_rate)
            except Exception as exc:
                # Malformed frame (e.g. unexpected length) — treat as unvoiced
                logger.debug(f"[VAD] Frame {i} rejected by WebRTC: {exc}")
                mask[i] = False
        return mask

    def _run_energy(self, frames: np.ndarray) -> np.ndarray:
        """Energy-based fallback VAD: mark frames above an energy threshold."""
        rms_energy = np.mean(frames ** 2, axis=1)
        energy_db = 10.0 * np.log10(np.maximum(rms_energy, 1e-12))
        return energy_db > self.config.energy_threshold_db

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.config.aggressiveness not in (0, 1, 2, 3):
            raise ValueError("aggressiveness must be 0, 1, 2, or 3")
        if self.config.frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        if self.config.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(
                "sample_rate must be one of 8000, 16000, 32000, 48000"
            )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _smooth_mask(mask: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a sliding median filter to remove isolated voiced/unvoiced flips.

    A majority vote over `window` consecutive frames determines whether
    the center frame is voiced. This prevents rapid on/off switching caused
    by momentary noise bursts or breath sounds.

    Args:
        mask: Raw boolean mask, shape (N,).
        window: Sliding window size in frames. Odd values work best.

    Returns:
        Smoothed boolean mask, shape (N,).
    """
    if window <= 1:
        return mask.copy()

    smoothed = np.empty_like(mask)
    half = window // 2

    for i in range(len(mask)):
        lo = max(0, i - half)
        hi = min(len(mask), i + half + 1)
        # Majority vote via median of 0/1 values
        smoothed[i] = np.median(mask[lo:hi].astype(np.float32)) >= 0.5

    return smoothed


def run_vad(
    audio: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_duration_ms: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: create a WebRTCVAD with default settings and run it.

    Args:
        audio: Float32 mono audio, shape (samples,).
        sr: Sample rate (must be 16000 after load_audio()).
        aggressiveness: WebRTC mode 0–3.
        frame_duration_ms: Frame length: 10, 20, or 30 ms.

    Returns:
        voiced_mask: Boolean array, shape (n_vad_frames,).
        timestamps: Frame center timestamps in seconds, shape (n_vad_frames,).
    """
    config = VADConfig(
        aggressiveness=aggressiveness,
        frame_duration_ms=frame_duration_ms,
        sample_rate=sr,
    )
    vad = WebRTCVAD(config)
    return vad.run(audio, sr)
