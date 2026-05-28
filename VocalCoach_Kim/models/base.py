"""
models/base.py - Abstract base interface for all VocalCoach inference models.

Every model module exposes a class that inherits BaseInferenceModel so the
scoring and fusion layers can call a consistent API regardless of the
underlying model architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union


class BaseInferenceModel(ABC):
    """
    Minimal interface contract for VocalCoach inference modules.

    Subclasses must implement load_model() and predict(). Everything else
    (config parsing, device selection, post-processing) is module-specific.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self._is_loaded: bool = False

    # ── Required interface ────────────────────────────────────────────────

    @abstractmethod
    def load_model(self) -> None:
        """
        Initialize model weights and move to the target device.

        Must set self._is_loaded = True on success.
        """

    @abstractmethod
    def predict(self, audio: Any) -> Any:
        """
        Run inference on a pre-loaded audio array or tensor.

        Args:
            audio: Float32 audio data (numpy array or torch Tensor).

        Returns:
            Module-specific result object or dict.
        """

    # ── Optional lifecycle hooks ──────────────────────────────────────────

    def run(self, audio_path: Union[str, Path]) -> Any:
        """
        Convenience end-to-end runner: load audio → ensure model → predict.

        Subclasses may override to add custom pre/post-processing.
        """
        from utils.audio import load_audio

        audio, sr = load_audio(str(audio_path))
        if not self._is_loaded:
            self.load_model()
        return self.predict(audio)

    # ── Helpers ───────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True after load_model() has completed successfully."""
        return self._is_loaded

    def __repr__(self) -> str:
        name = type(self).__name__
        status = "loaded" if self._is_loaded else "unloaded"
        return f"{name}({status})"
