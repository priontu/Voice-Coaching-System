"""
configs/loader.py - Centralized YAML configuration loader.

Usage:
    from configs.loader import load_config, load_model_config, merge_configs

    cfg = load_model_config("pitch")          # loads configs/pitch.yaml
    sys_cfg = load_config("configs/system.yaml")
    merged = merge_configs(sys_cfg, overrides)
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default configs directory (relative to this file's location)
_CONFIGS_DIR = Path(__file__).parent


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a nested dict.

    Args:
        path: Path to YAML file (absolute or relative to cwd).

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: File does not exist.
        ImportError:       PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    logger.debug(f"[config] Loaded: {p}")
    return cfg


def load_model_config(model_name: str) -> Dict[str, Any]:
    """
    Load the canonical config for a named model.

    Merges system.yaml (base) with <model_name>.yaml (overrides).

    Args:
        model_name: "phoneme" | "pitch" | "onset_offset"

    Returns:
        Merged config dict.

    Raises:
        ValueError: Unknown model name.
    """
    known = {"phoneme", "pitch", "onset_offset"}
    if model_name not in known:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {known}")

    system = load_config(_CONFIGS_DIR / "system.yaml")
    model = load_config(_CONFIGS_DIR / f"{model_name}.yaml")
    return merge_configs(system, model)


def load_preprocessing_config() -> Dict[str, Any]:
    """Load shared preprocessing parameters."""
    return load_config(_CONFIGS_DIR / "preprocessing.yaml")


def load_pipeline_config() -> Dict[str, Any]:
    """
    Load the unified pipeline configuration.

    Merges system.yaml (base) with pipeline.yaml (pipeline-specific overrides).
    """
    system = load_config(_CONFIGS_DIR / "system.yaml")
    pipeline = load_config(_CONFIGS_DIR / "pipeline.yaml")
    return merge_configs(system, pipeline)


def merge_configs(
    base: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deep-merge *overrides* into *base* and return the result.

    Nested dicts are merged recursively; scalar values in *overrides*
    replace those in *base*. Neither input is modified.

    Args:
        base:      Base configuration dict.
        overrides: Override values (typically model-specific config).

    Returns:
        New merged dict.
    """
    result = copy.deepcopy(base)
    _deep_merge(result, overrides)
    return result


def get_value(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested value using dot-path keys.

    Example:
        get_value(cfg, "pitch", "hop_length", default=160)
    """
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_merge(target: dict, source: dict) -> None:
    """Recursively merge source into target in-place."""
    for key, val in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(val, dict):
            _deep_merge(target[key], val)
        else:
            target[key] = copy.deepcopy(val)
