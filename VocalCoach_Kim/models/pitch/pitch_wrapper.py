"""
models/pitch/pitch_wrapper.py - Model-agnostic pitch estimation wrapper.

Supported backends: "torchcrepe" | "pyin" | "custom".

Changes from the original Pitch Model w VAD/pitch_wrapper.py:
  - get_best_device() replaced by utils.device.get_device (shared utility)
  - All prediction logic is UNCHANGED
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np

from utils.device import get_device

logger = logging.getLogger(__name__)

Backend = Literal["torchcrepe", "pyin", "custom"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PitchConfig:
    """All tunable parameters for pitch estimation."""

    backend: Backend = "torchcrepe"
    hop_length: int = 160
    fmin: float = 50.0
    fmax: float = 1000.0
    model_capacity: str = "full"
    batch_size: int = 1024
    use_viterbi: bool = True
    periodicity_threshold: float = 0.21
    silence_threshold_db: float = -60.0
    pyin_fmin: float = 65.0
    pyin_fmax: float = 2093.0
    pyin_frame_length: int = 2048
    pyin_hop_length: int = 512
    device: str = "auto"

    @classmethod
    def from_yaml(cls, cfg: Dict) -> "PitchConfig":
        p = cfg.get("pitch", {})
        return cls(
            backend=p.get("backend", "torchcrepe"),
            hop_length=p.get("hop_length", 160),
            fmin=p.get("fmin", 50.0),
            fmax=p.get("fmax", 1000.0),
            model_capacity=p.get("model_capacity", "full"),
            batch_size=p.get("batch_size", 1024),
            use_viterbi=p.get("use_viterbi", True),
            periodicity_threshold=p.get("periodicity_threshold", 0.21),
            silence_threshold_db=p.get("silence_threshold_db", -60.0),
            pyin_fmin=p.get("pyin_fmin", 65.0),
            pyin_fmax=p.get("pyin_fmax", 2093.0),
            pyin_frame_length=p.get("pyin_frame_length", 2048),
            pyin_hop_length=p.get("pyin_hop_length", 512),
            device=cfg.get("device", {}).get("preference", "auto"),
        )


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class PitchModelWrapper:
    """
    Unified pitch estimation interface over multiple backends.

    The underlying models are used as-is — this wrapper only handles
    data marshaling and device placement. No model weights are modified.
    """

    def __init__(self, config: Optional[PitchConfig] = None) -> None:
        self.config = config or PitchConfig()
        self._custom_fn: Optional[Callable] = None
        logger.info(f"[Pitch] Backend: {self.config.backend} | device: {self.config.device}")

    @classmethod
    def from_callable(
        cls,
        fn: Callable,
        config: Optional[PitchConfig] = None,
    ) -> "PitchModelWrapper":
        """Wrap a user-supplied pitch function under the standard interface."""
        cfg = config or PitchConfig()
        cfg.backend = "custom"
        instance = cls(cfg)
        instance._custom_fn = fn
        return instance

    def predict(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Run pitch estimation and return normalized (times, f0, confidence).

        All backends return the same triple for backend-agnostic downstream code.
        """
        dispatch = {
            "torchcrepe": self._predict_torchcrepe,
            "pyin": self._predict_pyin,
            "custom": self._predict_custom,
        }
        if self.config.backend not in dispatch:
            raise ValueError(f"Unknown backend: {self.config.backend!r}")

        times, f0, conf = dispatch[self.config.backend](audio, sr)

        f0 = np.where(np.isnan(f0), 0.0, f0).astype(np.float32)
        times = times.astype(np.float32)

        n_voiced = int(np.sum(f0 > 0))
        logger.info(
            f"[Pitch/{self.config.backend}] {len(times)} frames, "
            f"{n_voiced} voiced ({100*n_voiced/max(len(times),1):.1f}%)"
        )
        return times, f0, conf

    # ------------------------------------------------------------------
    # torchcrepe backend
    # ------------------------------------------------------------------

    def _predict_torchcrepe(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            import torch
            import torchcrepe  # type: ignore
        except ImportError as e:
            raise ImportError(
                "torchcrepe not installed. Install with:\n"
                "  pip install torchcrepe\n"
                "  pip install torch"
            ) from e

        cfg = self.config
        device = get_device(cfg.device)

        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        decoder = torchcrepe.decode.viterbi if cfg.use_viterbi else torchcrepe.decode.weighted_argmax

        logger.info(
            f"[Pitch/torchcrepe] predicting "
            f"(hop={cfg.hop_length}, fmin={cfg.fmin}, fmax={cfg.fmax}, "
            f"model={cfg.model_capacity}, viterbi={cfg.use_viterbi}, device={device})"
        )

        f0_tensor, periodicity = torchcrepe.predict(
            audio_tensor, sr, cfg.hop_length, cfg.fmin, cfg.fmax,
            model=cfg.model_capacity,
            return_periodicity=True,
            batch_size=cfg.batch_size,
            device=device,
            decoder=decoder,
        )

        periodicity = torchcrepe.threshold.Silence(cfg.silence_threshold_db)(
            periodicity, audio_tensor, sr, cfg.hop_length
        )
        f0_tensor = torchcrepe.threshold.At(cfg.periodicity_threshold)(f0_tensor, periodicity)

        n_frames = f0_tensor.shape[-1]
        times = (torch.arange(n_frames, dtype=torch.float32) * cfg.hop_length / sr).cpu().numpy()
        f0_np = f0_tensor.squeeze().cpu().detach().numpy().astype(np.float32)
        conf_np = periodicity.squeeze().cpu().detach().numpy().astype(np.float32)
        f0_np = np.where(np.isnan(f0_np), 0.0, f0_np)

        return times, f0_np, conf_np

    # ------------------------------------------------------------------
    # pYIN backend
    # ------------------------------------------------------------------

    def _predict_pyin(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            import librosa  # type: ignore
        except ImportError as e:
            raise ImportError("librosa is not installed: pip install librosa") from e

        cfg = self.config
        logger.info(f"[Pitch/pYIN] predicting (fmin={cfg.pyin_fmin}Hz, fmax={cfg.pyin_fmax}Hz)")

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio, fmin=cfg.pyin_fmin, fmax=cfg.pyin_fmax, sr=sr,
            frame_length=cfg.pyin_frame_length, hop_length=cfg.pyin_hop_length,
            fill_na=0.0,
        )

        n_frames = len(f0)
        times = librosa.frames_to_time(
            np.arange(n_frames), sr=sr, hop_length=cfg.pyin_hop_length,
        ).astype(np.float32)
        f0 = np.where(np.isnan(f0), 0.0, f0).astype(np.float32)

        return times, f0, voiced_prob.astype(np.float32)

    # ------------------------------------------------------------------
    # Custom callable backend
    # ------------------------------------------------------------------

    def _predict_custom(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self._custom_fn is None:
            raise RuntimeError(
                "No custom pitch function registered. "
                "Use PitchModelWrapper.from_callable(fn) to register one."
            )

        result = self._custom_fn(audio, sr)

        if len(result) == 2:
            times, f0 = result
            confidence = None
        elif len(result) == 3:
            times, f0, confidence = result
            confidence = np.asarray(confidence, dtype=np.float32) if confidence is not None else None
        else:
            raise ValueError(
                "Custom pitch function must return (times, f0) or (times, f0, confidence)"
            )

        return (
            np.asarray(times, dtype=np.float32),
            np.asarray(f0, dtype=np.float32),
            confidence,
        )
