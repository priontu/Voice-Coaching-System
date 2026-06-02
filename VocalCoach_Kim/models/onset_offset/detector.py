"""
models/onset_offset/detector.py - NoteDetector and BaseInferenceModel wrapper.

NoteDetector handles: audio loading → spectrogram → model inference →
peak picking → onset/offset pairing.

Changes from the original Note Model/inference.py:
  - load_audio() replaced by utils.audio.load_audio (shared utility)
  - BaseInferenceModel interface added (OnsetOffsetInferenceModel)
  - All inference logic is UNCHANGED
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from models.base import BaseInferenceModel
from models.onset_offset.model import OnsetOffsetModel
from models.onset_offset.spec_utils import (
    compute_log_mel_spectrogram,
    frames_to_time,
    normalize_spectrogram,
    pair_onsets_offsets,
    peak_pick_offsets,
    peak_pick_onsets,
)
from utils.audio import load_audio
from utils.checkpoints import load_checkpoint
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class NoteDetector:
    """
    End-to-end inference pipeline: WAV → note boundaries.

    Attributes:
        model:               Loaded OnsetOffsetModel in eval mode.
        device:              Torch device used for inference.
        onset_threshold:     Minimum peak height for onset detection.
        offset_threshold:    Minimum peak height for offset detection.
        min_distance_frames: Minimum frames between consecutive peaks.
    """

    def __init__(
        self,
        model: OnsetOffsetModel,
        device: torch.device,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        onset_threshold: float = 0.3,
        offset_threshold: float = 0.3,
        min_distance_frames: int = 3,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_distance_frames = min_distance_frames

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config_path: str) -> "NoteDetector":
        """Instantiate NoteDetector from a saved checkpoint and YAML config."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mc, ac, pp = cfg["model"], cfg["audio"], cfg["peak_picking"]

        model = OnsetOffsetModel(
            n_mels=ac["n_mels"],
            cnn_channels=mc["cnn_channels"],
            lstm_hidden_size=mc["lstm_hidden_size"],
            lstm_num_layers=mc["lstm_num_layers"],
            lstm_dropout=mc["lstm_dropout"],
            head_hidden_size=mc["head_hidden_size"],
            dropout=mc["dropout"],
        )

        # Use shared checkpoint loader
        load_checkpoint(checkpoint_path, model, device=str(device))

        return cls(
            model=model,
            device=device,
            sample_rate=ac["sample_rate"],
            n_fft=ac["n_fft"],
            hop_length=ac["hop_length"],
            n_mels=ac["n_mels"],
            fmin=ac["fmin"],
            fmax=ac["fmax"],
            onset_threshold=pp["onset_threshold"],
            offset_threshold=pp["offset_threshold"],
            min_distance_frames=pp["min_distance_frames"],
        )

    # ── Core inference ────────────────────────────────────────────────────

    def predict_probs(
        self, audio_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run model inference on a WAV file.

        Returns:
            onset_probs:  [T] float32 in [0, 1]
            offset_probs: [T] float32 in [0, 1]
            frame_times:  [T] seconds per frame
        """
        # Uses shared audio loader; returns numpy array
        audio_np, sr = load_audio(audio_path, target_sr=self.sample_rate)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # [1, N]

        log_mel = compute_log_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        log_mel = normalize_spectrogram(log_mel)

        x = log_mel.unsqueeze(0).to(self.device)  # [1, 1, n_mels, T]
        on_probs, off_probs = self.model.predict(x)

        on_probs = on_probs.squeeze(0).cpu().numpy()   # [T]
        off_probs = off_probs.squeeze(0).cpu().numpy()  # [T]
        frame_times = frames_to_time(log_mel.shape[-1], self.hop_length, self.sample_rate)

        return on_probs, off_probs, frame_times

    def predict_probs_from_array(
        self,
        audio_np: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run model inference on a pre-loaded numpy array.

        Used by UnifiedInferencePipeline to avoid loading the audio twice.
        The spectrogram logic is identical to predict_probs(); only the I/O
        step is skipped.

        Args:
            audio_np: 1-D float32 numpy array @ sr Hz.
            sr:       Sample rate (must match self.sample_rate).

        Returns:
            onset_probs:  [T] float32
            offset_probs: [T] float32
            frame_times:  [T] seconds per frame
        """
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # [1, N]

        log_mel = compute_log_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        log_mel = normalize_spectrogram(log_mel)

        x = log_mel.unsqueeze(0).to(self.device)  # [1, 1, n_mels, T]
        with torch.no_grad():
            on_probs, off_probs = self.model.predict(x)

        on_probs = on_probs.squeeze(0).cpu().numpy()
        off_probs = off_probs.squeeze(0).cpu().numpy()
        frame_times = frames_to_time(log_mel.shape[-1], self.hop_length, self.sample_rate)
        return on_probs, off_probs, frame_times

    def detect(self, audio_path: str) -> List[Dict]:
        """
        Detect note boundaries in a WAV file.

        Returns:
            List of dicts: {"onset_time", "offset_time", "duration"}
        """
        on_probs, off_probs, frame_times = self.predict_probs(audio_path)

        onsets = peak_pick_onsets(on_probs, frame_times, self.onset_threshold, self.min_distance_frames)
        offsets = peak_pick_offsets(off_probs, frame_times, self.offset_threshold, self.min_distance_frames)

        notes = pair_onsets_offsets(onsets, offsets)
        logger.info("Detected %d notes in '%s'.", len(notes), audio_path)
        return notes


# ---------------------------------------------------------------------------
# BaseInferenceModel wrapper
# ---------------------------------------------------------------------------

class OnsetOffsetInferenceModel(BaseInferenceModel):
    """
    BaseInferenceModel implementation for the note onset/offset detector.

    Example:
        model = OnsetOffsetInferenceModel.from_checkpoint(
            "checkpoints/best.pt", "configs/onset_offset.yaml"
        )
        notes = model.run("singing.wav")
        # [{"onset_time": 1.23, "offset_time": 1.57, "duration": 0.34}, ...]
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._checkpoint_path = checkpoint_path
        self._config_path = config_path
        self._detector: Optional[NoteDetector] = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "configs/onset_offset.yaml",
    ) -> "OnsetOffsetInferenceModel":
        instance = cls(checkpoint_path=checkpoint_path, config_path=config_path)
        instance.load_model()
        return instance

    def load_model(self) -> None:
        if self._checkpoint_path is None:
            raise ValueError("checkpoint_path must be set before calling load_model()")
        config_path = self._config_path or "configs/onset_offset.yaml"
        self._detector = NoteDetector.from_checkpoint(self._checkpoint_path, config_path)
        self._is_loaded = True

    def predict(self, audio: Any) -> List[Dict]:
        raise NotImplementedError(
            "Use run(audio_path) for the onset/offset model. "
            "The spectrogram computation requires an audio file path."
        )

    def run(self, audio_path) -> List[Dict]:
        if not self._is_loaded:
            self.load_model()
        return self._detector.detect(str(audio_path))
