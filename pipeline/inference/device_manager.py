"""
inference/device_manager.py - Centralized device orchestration.

A single DeviceManager instance is shared across all models in the unified
pipeline so device selection logic runs exactly once and every model ends up
on the same device without cross-device tensor copies.
"""

from __future__ import annotations

import logging
from typing import Optional

from utils.device import get_device, get_device_info, get_torch_device

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Stateless (after construction) device coordinator.

    Instantiate once at pipeline startup; pass to any model that needs a
    device.  The same instance can be shared across threads — it contains no
    mutable state after __init__.

    Args:
        preference: "auto", "cpu", "cuda", "cuda:N", or "mps".
                    "auto" selects CUDA → MPS → CPU in that order.
    """

    def __init__(self, preference: str = "auto") -> None:
        self._preference = preference
        self._device_str: str = get_device(preference)

        info = get_device_info()
        logger.info(
            f"[DeviceManager] Selected device: {self._device_str} "
            f"(cuda={info['cuda']}, mps={info['mps']})"
        )
        if info["cuda"] and info.get("cuda_name"):
            logger.info(f"[DeviceManager] GPU: {info['cuda_name']}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def device_str(self) -> str:
        """Device string suitable for torch.device() or model config."""
        return self._device_str

    @property
    def torch_device(self):
        """torch.device object for the selected device."""
        import torch
        return torch.device(self._device_str)

    @property
    def is_cuda(self) -> bool:
        return self._device_str.startswith("cuda")

    @property
    def is_cpu(self) -> bool:
        return self._device_str == "cpu"

    def move(self, tensor):
        """
        Move a tensor to the managed device if it is not already there.

        Returns the tensor (possibly the same object if already on device).
        """
        if tensor.device.type != self.torch_device.type:
            return tensor.to(self.torch_device)
        return tensor

    def __repr__(self) -> str:
        return f"DeviceManager(device={self._device_str!r})"
