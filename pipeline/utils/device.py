"""
utils/device.py - Centralized PyTorch device management.

All models use get_device() / get_torch_device() so device selection
logic is defined in one place and consistently logged.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> str:
    """
    Return the best available PyTorch device string.

    Priority when preference == "auto":  CUDA → MPS → CPU.

    Args:
        preference: "auto", "cpu", "cuda", "cuda:N", or "mps".

    Returns:
        Device string suitable for torch.device().

    Raises:
        ValueError: Explicit device requested but not available.
    """
    if preference not in ("auto",):
        _validate_explicit(preference)
        return preference

    try:
        import torch
        if torch.cuda.is_available():
            dev = f"cuda:{torch.cuda.current_device()}"
            name = torch.cuda.get_device_name(dev)
            logger.info(f"[device] GPU: {name} → {dev}")
            return dev
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            logger.info("[device] Apple MPS detected → mps")
            return "mps"
    except ImportError:
        pass

    logger.info("[device] No GPU detected → cpu")
    return "cpu"


def get_torch_device(preference: str = "auto"):
    """
    Return a torch.device for the best available device.

    Wraps get_device() and constructs a torch.device object.
    """
    import torch
    return torch.device(get_device(preference))


def get_device_info() -> dict:
    """Return a summary dict of available compute devices."""
    info: dict = {"cpu": True, "cuda": False, "mps": False, "cuda_name": None}
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["cuda_name"] = torch.cuda.get_device_name(0)
        info["mps"] = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_explicit(device: str) -> None:
    """Raise if an explicitly requested device is not available."""
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA device '{device}' requested but CUDA is not available.")
        if device == "mps":
            mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            if not mps_ok:
                raise ValueError("MPS device requested but not available.")
    except ImportError:
        raise RuntimeError("PyTorch is required for device management.")
