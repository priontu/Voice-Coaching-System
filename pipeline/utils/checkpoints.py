"""
utils/checkpoints.py - Checkpoint save/load utilities for PyTorch models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_checkpoint(
    path: str,
    model,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a .pt checkpoint into model and return the full checkpoint dict.

    Args:
        path:   Path to the .pt checkpoint file.
        model:  PyTorch Module. state_dict is loaded in-place.
        device: Target device string for map_location.
        strict: Passed to model.load_state_dict().

    Returns:
        Full checkpoint dict (includes "epoch", "optimizer_state_dict", etc.).

    Raises:
        FileNotFoundError: Checkpoint file does not exist.
    """
    import torch

    ckpt_path = validate_checkpoint_path(path)
    ckpt: Dict[str, Any] = torch.load(str(ckpt_path), map_location=device)

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=strict)

    epoch = ckpt.get("epoch", "?")
    logger.info(f"[ckpt] Loaded '{ckpt_path.name}' (epoch {epoch}) → {device}")
    return ckpt


def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    epoch: Optional[int] = None,
    **metadata,
) -> None:
    """
    Save model (and optionally optimizer) state to a .pt file.

    Args:
        path:      Destination file path (.pt).
        model:     PyTorch Module to save.
        optimizer: Optional optimizer to save alongside the model.
        epoch:     Current training epoch (embedded in checkpoint dict).
        **metadata: Extra key/value pairs stored in the checkpoint dict.
    """
    import torch

    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        **metadata,
    }
    if epoch is not None:
        payload["epoch"] = epoch
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, str(ckpt_path))
    logger.info(f"[ckpt] Saved → {ckpt_path}")


def validate_checkpoint_path(path: str) -> Path:
    """
    Resolve and validate a checkpoint path.

    Raises:
        FileNotFoundError: File does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p
