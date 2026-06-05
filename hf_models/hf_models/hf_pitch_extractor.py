from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class HFPitchExtractor:
    """Wrapper for an RMVPE-style pitch extractor.

    This class attempts to use `rmvpe_onnx.RMVPE` (or `rmvpe.RMVPE`) when
    available. It expects a WAV file path and returns a dictionary with
    NumPy arrays for `times`, `f0_hz`, and `voiced`.
    """

    model_path: str | None = None

    def predict(self, wav_path: str) -> dict[str, np.ndarray]:
        try:
            import torchaudio
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("torchaudio is required to read audio files") from exc

        try:
            try:
                from rmvpe_onnx import RMVPE
            except ImportError:
                from rmvpe import RMVPE
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RMVPE pitch extraction requires rmvpe-onnx (optional)."
                " Install with: pip install rmvpe-onnx"
            ) from exc

        waveform, sr = torchaudio.load(wav_path)
        # convert to mono numpy
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        audio_np = waveform.detach().cpu().numpy().astype(np.float32)

        kwargs: dict[str, Any] = {}
        if self.model_path is not None:
            kwargs["model_path"] = self.model_path

        model = RMVPE(**kwargs) if kwargs else RMVPE()
        time, frequency, confidence, activation = model.predict(audio=audio_np, sr=int(sr))
        frequency = np.asarray(frequency, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        time = np.asarray(time, dtype=np.float32)
        voiced = (confidence >= 0.35).astype(np.int64)

        return {"times": time, "f0_hz": frequency, "voiced": voiced}
