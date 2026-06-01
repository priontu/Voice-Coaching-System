from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .parser import VocalParseOutput, parse_vocalparse_text


@dataclass(frozen=True)
class VocalParseConfig:
    checkpoint: str = "pymaster/VocalParse"
    max_duration_sec: float | None = 30.0
    sample_rate: int = 16000
    max_new_tokens: int = 512
    attn_implementation: str = "sdpa"


class VocalParseTranscriber:
    """VocalParse transcriber with HF snapshot resolution and model caching."""

    def __init__(self, config: VocalParseConfig | None = None):
        self.config = config or VocalParseConfig()
        self._checkpoint_path: Path | None = None
        self._model = None
        self._processor = None
        self._device = None

    def transcribe(self, audio_path: str | Path) -> VocalParseOutput:
        try:
            from vocalparse.model import load_audio
            from vocalparse.prompts import build_prefix_text
        except ImportError as exc:
            raise RuntimeError(
                "The vocalparse package is required. Install it with:\n"
                "  pip install git+https://github.com/pymaster17/VocalParse.git"
            ) from exc

        prepared_audio = self._prepare_audio(audio_path)
        model, processor, device = self._load_model()
        wav = load_audio(str(prepared_audio), sr=self.config.sample_rate).astype(np.float32)

        tokenizer = processor.tokenizer
        prefix_text = build_prefix_text(processor)
        inputs = processor(
            text=[prefix_text],
            audio=[wav],
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        model_dtype = next(model.parameters()).dtype
        inputs = {
            key: _move_input(value, device, model_dtype)
            for key, value in inputs.items()
        }
        prefix_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )
        gen_seq = outputs.sequences[0] if hasattr(outputs, "sequences") else outputs[0]
        pred_tokens = gen_seq[prefix_len:]
        raw_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)
        return parse_vocalparse_text(raw_text)

    def _load_model(self):
        if self._model is not None:
            return self._model, self._processor, self._device

        try:
            from vocalparse.model import load_model
        except ImportError as exc:
            raise RuntimeError(
                "The vocalparse package is required. Install it with:\n"
                "  pip install git+https://github.com/pymaster17/VocalParse.git"
            ) from exc

        checkpoint = self._resolve_checkpoint()
        self._model, self._processor, self._device = load_model(
            {
                "checkpoint": str(checkpoint),
                "attn_implementation": self.config.attn_implementation,
            }
        )
        return self._model, self._processor, self._device

    def _resolve_checkpoint(self) -> Path:
        if self._checkpoint_path is not None:
            return self._checkpoint_path

        checkpoint = Path(self.config.checkpoint)
        if checkpoint.exists():
            self._checkpoint_path = checkpoint.resolve()
            return self._checkpoint_path

        if "/" not in self.config.checkpoint:
            raise FileNotFoundError(f"VocalParse checkpoint not found: {self.config.checkpoint}")

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to download VocalParse checkpoints") from exc

        local_dir = snapshot_download(repo_id=self.config.checkpoint)
        self._checkpoint_path = Path(local_dir).resolve()
        return self._checkpoint_path

    def _prepare_audio(self, audio_path: str | Path) -> Path:
        audio_path = Path(audio_path)
        if self.config.max_duration_sec is None:
            return audio_path

        try:
            import torch
            import torchaudio
            import torchaudio.functional as F_audio
        except ImportError as exc:
            raise RuntimeError("torch and torchaudio are required for audio clipping/resampling") from exc

        audio, sample_rate = torchaudio.load(audio_path)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        max_samples = int(self.config.max_duration_sec * sample_rate)
        if audio.shape[-1] > max_samples:
            audio = audio[..., :max_samples]
        if sample_rate != self.config.sample_rate:
            audio = F_audio.resample(audio, sample_rate, self.config.sample_rate)
            sample_rate = self.config.sample_rate

        temp_dir = Path(tempfile.mkdtemp(prefix="vocalparse_audio_"))
        prepared = temp_dir / f"{audio_path.stem}_prepared.wav"
        torchaudio.save(str(prepared), audio.cpu(), sample_rate)
        return prepared


def _move_input(value: torch.Tensor, device: torch.device, model_dtype: torch.dtype) -> torch.Tensor:
    value = value.to(device)
    if torch.is_floating_point(value):
        return value.to(model_dtype)
    return value
