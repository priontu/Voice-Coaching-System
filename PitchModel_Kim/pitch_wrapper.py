"""
pitch_wrapper.py - Model-agnostic pitch estimation wrapper.

Wraps the existing torchcrepe pipeline (from test_pitch.py) and alternative
backends under a unified interface WITHOUT modifying the upstream model.

Supported backends:
  "torchcrepe" : torchcrepe (PyTorch CREPE) — primary backend, GPU-capable.
                 Matches the existing test_pitch.py configuration exactly.
  "pyin"       : librosa pYIN — CPU-only, good fallback.
  "custom"     : Any callable f(audio, sr) → (times, f0) or (times, f0, conf).

The wrapper never re-trains or modifies the underlying model.
It only handles audio → tensor conversion, device placement, and output
normalization so the downstream pipeline always sees the same (times, f0, conf)
triple regardless of which backend is active.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Backend = Literal["torchcrepe", "pyin", "custom"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PitchConfig:
    """All tunable parameters for pitch estimation."""

    backend: Backend = "torchcrepe"
    """Which pitch estimation backend to use."""

    # ---- torchcrepe settings (mirror of test_pitch.py defaults) ----
    hop_length: int = 160
    """
    Hop size in samples. 160 @ 16 kHz = 10 ms per frame.
    Must match the value used when generating pitch_data.json so that
    timestamps align with pitch_score.py expectations.
    """
    fmin: float = 50.0
    """Minimum detectable frequency in Hz."""
    fmax: float = 1000.0
    """Maximum detectable frequency in Hz (covers full singing range)."""
    model_capacity: str = "full"
    """torchcrepe model size: 'tiny', 'small', 'medium', 'large', 'full'."""
    batch_size: int = 1024
    """Batch size for torchcrepe GPU inference."""
    use_viterbi: bool = True
    """
    Use Viterbi decoding for smoother, more temporally consistent F0.
    Slightly slower but significantly better for sustained notes.
    """
    periodicity_threshold: float = 0.21
    """
    Frames with periodicity below this value are set to NaN then zeroed.
    0.21 is the empirically recommended threshold for torchcrepe on singing.
    """
    silence_threshold_db: float = -60.0
    """
    Silence gating: frames quieter than this (dBFS) are treated as unvoiced
    before periodicity thresholding. Prevents low-energy noise from leaking in.
    """

    # ---- pYIN settings ----
    pyin_fmin: float = 65.0
    """pYIN minimum frequency in Hz (C2 ≈ 65 Hz)."""
    pyin_fmax: float = 2093.0
    """pYIN maximum frequency in Hz (C7 ≈ 2093 Hz)."""
    pyin_frame_length: int = 2048
    """pYIN analysis window length in samples."""
    pyin_hop_length: int = 512
    """pYIN hop length in samples. Determines output frame rate."""

    # ---- Device ----
    device: str = "auto"
    """
    PyTorch device for GPU-capable backends.
    'auto' (default) selects CUDA if available, then MPS, then CPU.
    Override with 'cpu', 'cuda', 'cuda:0', or 'mps'.
    """


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class PitchModelWrapper:
    """
    Unified pitch estimation interface over multiple backends.

    The underlying models are used as-is — this wrapper only handles
    data marshaling (numpy ↔ tensor), device placement, and output
    normalization. No model weights are modified.

    Example — torchcrepe (GPU):
        cfg = PitchConfig(backend="torchcrepe", device="cuda")
        wrapper = PitchModelWrapper(cfg)
        times, f0, confidence = wrapper.predict(audio, sr=16000)

    Example — custom model:
        def my_model(audio, sr):
            return my_times, my_f0          # or (my_times, my_f0, my_conf)
        wrapper = PitchModelWrapper.from_callable(my_model)
        times, f0, _ = wrapper.predict(audio, sr=16000)
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
        """
        Wrap a user-supplied pitch function under the standard interface.

        The function signature must be:
            fn(audio: np.ndarray, sr: int) → (times, f0)
          or
            fn(audio: np.ndarray, sr: int) → (times, f0, confidence)

        Args:
            fn: Callable pitch estimator.
            config: Optional config (the backend field will be set to 'custom').

        Returns:
            Configured PitchModelWrapper instance.
        """
        cfg = config or PitchConfig()
        cfg.backend = "custom"
        instance = cls(cfg)
        instance._custom_fn = fn
        return instance

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Run pitch estimation and return normalized outputs.

        All backends return the same triple so downstream code is backend-agnostic.

        Args:
            audio: Float32 mono audio, shape (samples,).
            sr: Sample rate in Hz (should be 16000 after load_audio()).

        Returns:
            times: Frame center timestamps in seconds, shape (T,) float32.
            f0: Fundamental frequency in Hz, shape (T,) float32.
                Unvoiced / silent frames are set to 0.0.
            confidence: Per-frame confidence in [0, 1], shape (T,) float32.
                        None when the backend does not provide confidence scores.

        Raises:
            ImportError: Required backend library is not installed.
            RuntimeError: Custom function not registered when backend='custom'.
        """
        dispatch = {
            "torchcrepe": self._predict_torchcrepe,
            "pyin": self._predict_pyin,
            "custom": self._predict_custom,
        }
        if self.config.backend not in dispatch:
            raise ValueError(f"Unknown backend: {self.config.backend!r}")

        times, f0, conf = dispatch[self.config.backend](audio, sr)

        # Replace any remaining NaN with 0.0 for downstream compatibility
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
        """
        Run torchcrepe pitch estimation — mirrors test_pitch.py exactly.

        torchcrepe expects a (1, samples) float tensor. The decoder can be
        'viterbi' (smooth, slower) or 'weighted_argmax' (fast, slightly noisier).
        """
        try:
            import torch
            import torchcrepe  # type: ignore
        except ImportError as e:
            raise ImportError(
                "torchcrepe not installed. Install with:\n"
                "  pip install torchcrepe\n"
                "  pip install torch  (if not already installed)"
            ) from e

        cfg = self.config

        # Resolve "auto" to the best available device at inference time
        from utils import get_best_device
        device = get_best_device() if cfg.device == "auto" else cfg.device

        # Convert to (1, samples) tensor — same as test_pitch.py
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio_tensor = audio_tensor.to(device)

        decoder = torchcrepe.decode.viterbi if cfg.use_viterbi else torchcrepe.decode.weighted_argmax

        logger.info(
            f"[Pitch/torchcrepe] predicting "
            f"(hop={cfg.hop_length}, fmin={cfg.fmin}, fmax={cfg.fmax}, "
            f"model={cfg.model_capacity}, viterbi={cfg.use_viterbi}, "
            f"device={device})"
        )

        f0_tensor, periodicity = torchcrepe.predict(
            audio_tensor,
            sr,
            cfg.hop_length,
            cfg.fmin,
            cfg.fmax,
            model=cfg.model_capacity,
            return_periodicity=True,
            batch_size=cfg.batch_size,
            device=device,
            decoder=decoder,
        )
        # f0_tensor: (1, n_frames), periodicity: (1, n_frames)

        # Apply silence gating first (removes very quiet frames)
        periodicity = torchcrepe.threshold.Silence(cfg.silence_threshold_db)(
            periodicity, audio_tensor, sr, cfg.hop_length
        )

        # Threshold by periodicity confidence
        f0_tensor = torchcrepe.threshold.At(cfg.periodicity_threshold)(
            f0_tensor, periodicity
        )
        # Frames below threshold are now NaN in f0_tensor

        n_frames = f0_tensor.shape[-1]

        # Build timestamps matching test_pitch.py's convention:
        #   times[i] = i * hop_length / sr  (frame start, not center)
        # We keep this convention for full backward compatibility with
        # the pitch_data.json format consumed by pitch_score.py.
        times = (torch.arange(n_frames, dtype=torch.float32) *
                 cfg.hop_length / sr).cpu().numpy()

        f0_np = f0_tensor.squeeze().cpu().detach().numpy().astype(np.float32)
        conf_np = periodicity.squeeze().cpu().detach().numpy().astype(np.float32)

        # NaN → 0.0 for unvoiced frames (consistent with test_pitch.py output)
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
        """Run librosa pYIN probabilistic pitch estimation (CPU only)."""
        try:
            import librosa  # type: ignore
        except ImportError as e:
            raise ImportError(
                "librosa is not installed. Install with: pip install librosa"
            ) from e

        cfg = self.config
        logger.info(
            f"[Pitch/pYIN] predicting "
            f"(fmin={cfg.pyin_fmin}Hz, fmax={cfg.pyin_fmax}Hz, "
            f"hop={cfg.pyin_hop_length})"
        )

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=cfg.pyin_fmin,
            fmax=cfg.pyin_fmax,
            sr=sr,
            frame_length=cfg.pyin_frame_length,
            hop_length=cfg.pyin_hop_length,
            fill_na=0.0,
        )

        n_frames = len(f0)
        times = librosa.frames_to_time(
            np.arange(n_frames),
            sr=sr,
            hop_length=cfg.pyin_hop_length,
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
        """Call the user-supplied pitch function and normalize its output."""
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
            confidence = (
                np.asarray(confidence, dtype=np.float32)
                if confidence is not None else None
            )
        else:
            raise ValueError(
                "Custom pitch function must return (times, f0) or (times, f0, confidence)"
            )

        return (
            np.asarray(times, dtype=np.float32),
            np.asarray(f0, dtype=np.float32),
            confidence,
        )
