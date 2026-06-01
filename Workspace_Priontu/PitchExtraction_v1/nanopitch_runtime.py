from __future__ import annotations

import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights.pth"
DEFAULT_NANOPITCH_ROOT = REPO_ROOT / "NanoPitch_v2"
NanoPitchDecoder = Literal["argmax", "offline", "realtime"]


@dataclass(frozen=True)
class NanoPitchRuntimeConfig:
    weights_path: Path = DEFAULT_WEIGHTS
    nanopitch_root: Path = DEFAULT_NANOPITCH_ROOT
    sample_rate: int = 16000
    n_mels: int = 40
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    fmin: float = 30.0
    fmax: float | None = 8000.0
    top_db: float = 80.0
    vad_threshold: float = 0.5
    pitch_threshold: float = 0.3
    decoder: NanoPitchDecoder = "argmax"

    @property
    def frame_rate_hz(self) -> float:
        return self.sample_rate / self.hop_length


class NanoPitchBatchExtractor(torch.nn.Module):
    """GPU-oriented NanoPitch extractor for raw DataLoader batches.

    The fast path decodes pitch with per-frame argmax on the device. This is
    suitable for training-time pseudo-labels/features. Offline Viterbi can be
    added for evaluation, but it is CPU/numpy-based in NanoPitch_v2.
    """

    def __init__(
        self,
        config: NanoPitchRuntimeConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.config = config or NanoPitchRuntimeConfig()
        self.device = torch.device(device)
        self.nanopitch_module = _load_nanopitch_module(self.config.nanopitch_root)
        self.feature_extractor = self._build_feature_extractor().to(self.device)
        self.model = self._load_model(self.config.weights_path).to(self.device)
        self.model.eval()

    def load_weights(self, weights_path: str | Path) -> None:
        self.model = self._load_model(Path(weights_path)).to(self.device)
        self.model.eval()

    def forward(self, raw_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        audio_items = [
            self._resample(audio.to(self.device, non_blocking=True), sample_rate)
            for audio, sample_rate in zip(raw_batch["audio_items"], raw_batch["audio_sample_rates"], strict=True)
        ]
        audio_batch, audio_lengths = _pad_audio_batch(audio_items)

        with torch.inference_mode():
            mel = self.feature_extractor(audio_batch)
            vad_prob, posteriorgram, _ = self.model(mel)

        vad_prob = vad_prob.squeeze(-1)
        pitch_confidence, bins = posteriorgram.max(dim=-1)
        f0_hz = self._decode_f0(posteriorgram, bins, pitch_confidence).to(vad_prob.device)
        voiced = (vad_prob >= self.config.vad_threshold) & (f0_hz > 0)
        if self.config.decoder == "argmax":
            voiced = voiced & (pitch_confidence >= self.config.pitch_threshold)
        f0_hz = torch.where(voiced, f0_hz, torch.zeros_like(f0_hz))

        frame_lengths = torch.div(audio_lengths, self.config.hop_length, rounding_mode="floor") + 1
        max_frames = f0_hz.shape[1]
        frame_lengths = torch.clamp(frame_lengths, max=max_frames)
        attention_mask = _lengths_to_mask(frame_lengths, max_frames)
        times = (
            torch.arange(max_frames, device=self.device, dtype=torch.float32)
            * self.config.hop_length
            / self.config.sample_rate
        )

        return {
            "f0_hz": f0_hz,
            "voiced": voiced.float(),
            "vad_prob": vad_prob,
            "pitch_confidence": pitch_confidence,
            "posteriorgram": posteriorgram,
            "attention_mask": attention_mask,
            "frame_lengths": frame_lengths,
            "times": times,
            "decoder": self.config.decoder,
        }

    def _decode_f0(
        self,
        posteriorgram: torch.Tensor,
        bins: torch.Tensor,
        pitch_confidence: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.decoder == "argmax":
            f0_hz = _bin_to_f0_torch(bins.float()).to(posteriorgram.dtype)
            return torch.where(
                pitch_confidence >= self.config.pitch_threshold,
                f0_hz,
                torch.zeros_like(f0_hz),
            )

        posterior_np = posteriorgram.detach().float().cpu().numpy()
        decoded = []
        for item in posterior_np:
            if self.config.decoder == "offline":
                decoded.append(self.nanopitch_module.viterbi_decode(item))
            elif self.config.decoder == "realtime":
                decoded.append(self.nanopitch_module.viterbi_decode_realtime(item))
            else:
                raise ValueError(f"Unsupported NanoPitch decoder: {self.config.decoder}")
        decoded_np = np.stack(decoded, axis=0).astype("float32", copy=False)
        return torch.from_numpy(decoded_np).to(device=posteriorgram.device, dtype=posteriorgram.dtype)

    def _load_model(self, weights_path: Path) -> torch.nn.Module:
        if not weights_path.exists():
            raise FileNotFoundError(f"NanoPitch weights not found: {weights_path}")
        warnings.warn(
            f"Loading NanoPitch weights from {weights_path}. Only load trusted checkpoints.",
            RuntimeWarning,
        )
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        model_kwargs = checkpoint.get("model_kwargs", {}) if isinstance(checkpoint, dict) else {}
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model = self.nanopitch_module.NanoPitch(**model_kwargs)
        model.load_state_dict(state_dict)
        return model

    def _build_feature_extractor(self) -> torch.nn.Module:
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for PitchExtraction_v1") from exc

        return torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                win_length=self.config.win_length,
                hop_length=self.config.hop_length,
                f_min=self.config.fmin,
                f_max=self.config.fmax,
                n_mels=self.config.n_mels,
                power=2.0,
                center=True,
            ),
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=self.config.top_db),
            _TransposeMel(),
        )

    def _resample(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.config.sample_rate:
            return audio.float().contiguous()
        try:
            import torchaudio.functional as F_audio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for PitchExtraction_v1") from exc
        return F_audio.resample(audio.float(), sample_rate, self.config.sample_rate).contiguous()


class _TransposeMel(torch.nn.Module):
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return mel.transpose(1, 2).contiguous()


def _bin_to_f0_torch(bins: torch.Tensor) -> torch.Tensor:
    return 31.7 * torch.pow(2.0, bins * 20.0 / 1200.0)


def _pad_audio_batch(audio_items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([item.numel() for item in audio_items], dtype=torch.long, device=audio_items[0].device)
    max_len = int(lengths.max().item())
    batch = torch.zeros((len(audio_items), max_len), dtype=torch.float32, device=audio_items[0].device)
    for index, audio in enumerate(audio_items):
        batch[index, : audio.numel()] = audio.float()
    return batch, lengths


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _load_nanopitch_module(nanopitch_root: Path):
    model_path = nanopitch_root / "training" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"NanoPitch model.py not found: {model_path}")

    module_name = "_pitch_extraction_v1_nanopitch_model"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import NanoPitch model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
