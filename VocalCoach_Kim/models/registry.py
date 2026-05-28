"""
models/registry.py - Centralized model registry with lazy loading.

The registry maps model names to factory functions. Models are not instantiated
until load() is called, avoiding heavyweight imports (transformers, torch) until
they are actually needed.

Usage:
    from models.registry import ModelRegistry

    registry = ModelRegistry()
    registry.load("pitch")
    registry.load("phoneme")

    pitch_model = registry.get("pitch")
    result = pitch_model.run("singing.wav")

Models registered by default:
    "phoneme"       → PhonemeInferenceModel
    "pitch"         → PitchInferenceModel
    "onset_offset"  → OnsetOffsetInferenceModel (requires checkpoint)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from models.base import BaseInferenceModel

logger = logging.getLogger(__name__)

# Type alias for factory callables
_Factory = Callable[..., BaseInferenceModel]


class ModelRegistry:
    """
    Registry of model factory functions with lazy instantiation.

    Models are registered by name with a factory callable. The factory is
    called on the first load() for that name, and the resulting model instance
    is cached for subsequent get() calls.

    Factories must accept **kwargs so that callers can pass model-specific
    configuration (e.g. checkpoint_path, config dict).
    """

    def __init__(self) -> None:
        self._factories: Dict[str, _Factory] = {}
        self._instances: Dict[str, BaseInferenceModel] = {}
        self._register_defaults()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, factory: _Factory) -> None:
        """
        Register a model factory function under the given name.

        Args:
            name:    Unique model identifier (e.g. "phoneme").
            factory: Callable(**kwargs) → BaseInferenceModel.
                     The factory should NOT call load_model() itself —
                     that is done lazily by load().
        """
        if name in self._factories:
            logger.warning(f"[registry] Overwriting existing factory for '{name}'")
        self._factories[name] = factory
        logger.debug(f"[registry] Registered factory: '{name}'")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, name: str, force_reload: bool = False, **kwargs: Any) -> BaseInferenceModel:
        """
        Instantiate and load a model by name (lazy + cached).

        If the model has already been loaded, the cached instance is returned
        unless force_reload=True.

        Args:
            name:         Model name as registered.
            force_reload: If True, re-instantiate and reload even if cached.
            **kwargs:     Passed to the factory function (e.g. checkpoint_path,
                          config_path, pipeline_config).

        Returns:
            Loaded BaseInferenceModel instance.

        Raises:
            KeyError:   Unknown model name.
            ValueError: Model cannot be loaded (e.g. missing checkpoint).
        """
        if name not in self._factories:
            raise KeyError(
                f"Unknown model '{name}'. "
                f"Registered: {sorted(self._factories)}"
            )

        if name in self._instances and not force_reload:
            logger.debug(f"[registry] Returning cached model '{name}'")
            return self._instances[name]

        logger.info(f"[registry] Loading model: '{name}'")
        factory = self._factories[name]
        instance = factory(**kwargs)

        if not instance.is_loaded:
            instance.load_model()

        self._instances[name] = instance
        logger.info(f"[registry] '{name}' ready ({instance!r})")
        return instance

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[BaseInferenceModel]:
        """
        Return a loaded model by name, or None if not yet loaded.

        Use load() to trigger loading; get() never loads.
        """
        return self._instances.get(name)

    def is_loaded(self, name: str) -> bool:
        """True if the model has been loaded and is cached."""
        inst = self._instances.get(name)
        return inst is not None and inst.is_loaded

    def unload(self, name: str) -> None:
        """
        Remove a model instance from the cache.

        The model weights remain in GPU/CPU memory until Python's garbage
        collector runs. To free memory immediately, call del on the return
        value of get() before calling unload().
        """
        if name in self._instances:
            del self._instances[name]
            logger.info(f"[registry] Unloaded '{name}' from registry cache")

    def loaded_names(self):
        """Return the names of all currently loaded models."""
        return list(self._instances.keys())

    def registered_names(self):
        """Return all registered model names (loaded or not)."""
        return sorted(self._factories.keys())

    def __repr__(self) -> str:
        loaded = [k for k, v in self._instances.items() if v.is_loaded]
        return (
            f"ModelRegistry("
            f"registered={self.registered_names()}, "
            f"loaded={loaded})"
        )

    # ------------------------------------------------------------------
    # Default registrations
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        """Register the three built-in model factories."""

        def _phoneme_factory(**kwargs):
            from models.phoneme.phoneme_model import PhonemeInferenceModel, PhonemeBoundaryConfig
            cfg = kwargs.get("config")
            if cfg is None:
                try:
                    from configs.loader import load_model_config
                    cfg = load_model_config("phoneme")
                except Exception:
                    cfg = {}
            bc = PhonemeBoundaryConfig.from_yaml(cfg) if cfg else PhonemeBoundaryConfig()
            return PhonemeInferenceModel(config=bc)

        def _pitch_factory(**kwargs):
            from models.pitch.pipeline import PitchInferenceModel, PipelineConfig
            pipeline_cfg = kwargs.get("pipeline_config")
            if pipeline_cfg is None:
                try:
                    from configs.loader import load_model_config
                    yaml_cfg = load_model_config("pitch")
                    pipeline_cfg = PipelineConfig.from_yaml(yaml_cfg)
                except Exception:
                    pipeline_cfg = PipelineConfig()
            return PitchInferenceModel(config=pipeline_cfg)

        def _onset_factory(**kwargs):
            from models.onset_offset.detector import OnsetOffsetInferenceModel
            checkpoint_path = kwargs.get("checkpoint_path") or _resolve_checkpoint()
            config_path = kwargs.get("config_path", "configs/onset_offset.yaml")
            inst = OnsetOffsetInferenceModel(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
            )
            return inst

        self.register("phoneme", _phoneme_factory)
        self.register("pitch", _pitch_factory)
        self.register("onset_offset", _onset_factory)


def _resolve_checkpoint() -> Optional[str]:
    """
    Search the checkpoints/ directory for a .pt file to use as the onset/offset
    checkpoint. Returns None if not found (load will fail gracefully).
    """
    candidates = [
        Path("checkpoints/best.pt"),
        Path("checkpoints/onset_offset.pt"),
        Path("checkpoints/onset_offset_best.pt"),
    ]
    for c in candidates:
        if c.exists():
            logger.info(f"[registry] Auto-resolved checkpoint: {c}")
            return str(c)
    logger.warning(
        "[registry] No onset/offset checkpoint found in checkpoints/. "
        "Pass checkpoint_path= explicitly to registry.load('onset_offset')."
    )
    return None


# ---------------------------------------------------------------------------
# Module-level singleton — shared by UnifiedInferencePipeline
# ---------------------------------------------------------------------------

_global_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Return the shared global registry, creating it if necessary."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry
