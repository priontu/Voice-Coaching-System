from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RMVPERuntimeConfig:
    """Runtime settings for RMVPE pitch extraction.

    RMVPE is loaded through the ``rmvpe-onnx`` package. That package downloads
    the Hugging Face ONNX checkpoint on first use unless ``model_path`` points
    to a local ONNX file.
    """

    model_path: Path | None = None
    confidence_threshold: float = 0.35
    providers: tuple[str, ...] | None = None
    return_activation: bool = False


class RMVPEBatchExtractor:
    """Batch wrapper around RMVPE ONNX inference.

    The ONNX package works item-by-item with NumPy audio. This wrapper keeps
    the DataLoader interface consistent with the rest of the project and pads
    the returned contours into tensors.
    """

    def __init__(
        self,
        config: RMVPERuntimeConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        self.config = config or RMVPERuntimeConfig()
        self.device = torch.device(device)
        self.model = self._load_model()

    def __call__(self, raw_batch: dict[str, Any]) -> dict[str, torch.Tensor | list[Any]]:
        predictions = []
        for audio, sample_rate in zip(raw_batch["audio_items"], raw_batch["audio_sample_rates"], strict=True):
            audio_np = _to_mono_numpy(audio)
            time, frequency, confidence, activation = self.model.predict(audio=audio_np, sr=int(sample_rate))
            frequency = np.asarray(frequency, dtype=np.float32)
            confidence = np.asarray(confidence, dtype=np.float32)
            time = np.asarray(time, dtype=np.float32)
            voiced = confidence >= self.config.confidence_threshold
            frequency = np.where(voiced, frequency, 0.0).astype(np.float32, copy=False)
            predictions.append(
                {
                    "times": time,
                    "f0_hz": frequency,
                    "confidence": confidence,
                    "voiced": voiced.astype(np.float32),
                    "activation": activation if self.config.return_activation else None,
                }
            )

        return _pad_predictions(predictions, self.device)

    def _load_model(self):
        os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/voice_coach_numba_cache")
        try:
            try:
                from rmvpe_onnx import RMVPE
            except ImportError:
                from rmvpe import RMVPE
        except ImportError as exc:
            raise RuntimeError(
                "RMVPE extraction requires rmvpe-onnx. Install it in torch_it with:\n"
                "  pip install rmvpe-onnx\n"
                "For CUDA ONNX Runtime acceleration, install a compatible onnxruntime-gpu build."
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.config.model_path is not None:
            kwargs["model_path"] = str(self.config.model_path)
        if self.config.providers is not None:
            kwargs["providers"] = list(self.config.providers)

        try:
            return RMVPE(**kwargs)
        except TypeError:
            # Older/newer package versions may not expose all constructor args.
            if self.config.model_path is not None:
                try:
                    return RMVPE(str(self.config.model_path))
                except TypeError:
                    pass
            return RMVPE()


def _to_mono_numpy(audio: torch.Tensor) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    audio = audio.detach().float().cpu().contiguous()
    return audio.numpy().astype(np.float32, copy=False)


def _pad_predictions(
    predictions: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor | list[Any]]:
    lengths = torch.tensor([len(item["f0_hz"]) for item in predictions], dtype=torch.long, device=device)
    max_len = int(lengths.max().item()) if len(predictions) else 0
    batch_size = len(predictions)

    f0 = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    confidence = torch.zeros_like(f0)
    voiced = torch.zeros_like(f0)
    times = torch.zeros_like(f0)
    activations = []

    for index, item in enumerate(predictions):
        length = len(item["f0_hz"])
        if length == 0:
            activations.append(None)
            continue
        f0[index, :length] = torch.as_tensor(item["f0_hz"], dtype=torch.float32, device=device)
        confidence[index, :length] = torch.as_tensor(item["confidence"], dtype=torch.float32, device=device)
        voiced[index, :length] = torch.as_tensor(item["voiced"], dtype=torch.float32, device=device)
        times[index, :length] = torch.as_tensor(item["times"], dtype=torch.float32, device=device)
        activations.append(item["activation"])

    attention_mask = _lengths_to_mask(lengths, max_len)
    return {
        "f0_hz": f0,
        "voiced": voiced,
        "confidence": confidence,
        "times": times,
        "frame_lengths": lengths,
        "attention_mask": attention_mask,
        "activation": activations,
    }


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)
