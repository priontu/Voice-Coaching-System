from __future__ import annotations

import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch


DecoderName = Literal["offline", "realtime", "argmax"]


@dataclass(frozen=True)
class NanoPitchFeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 40
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    fmin: float = 30.0
    fmax: float | None = 8000.0
    top_db: float = 80.0

    @property
    def frame_rate_hz(self) -> float:
        return self.sample_rate / self.hop_length


@dataclass(frozen=True)
class NanoPitchPrediction:
    times: np.ndarray
    f0_hz: np.ndarray
    voiced: np.ndarray
    vad_prob: np.ndarray
    pitch_confidence: np.ndarray
    posteriorgram: np.ndarray
    checkpoint_path: str
    decoder: str

    def to_dict(self) -> dict:
        return {
            "times": self.times,
            "f0_hz": self.f0_hz,
            "voiced": self.voiced,
            "vad_prob": self.vad_prob,
            "pitch_confidence": self.pitch_confidence,
            "posteriorgram": self.posteriorgram,
            "checkpoint_path": self.checkpoint_path,
            "decoder": self.decoder,
        }


class NanoPitchExtractor:
    """Runtime-swappable NanoPitch pitch/F0 extractor."""

    def __init__(
        self,
        checkpoint_path: str | Path = "priontu_chowdhury",
        nanopitch_root: str | Path = "NanoPitch_v2",
        decoder: DecoderName = "offline",
        device: str | torch.device | None = None,
        feature_config: NanoPitchFeatureConfig | None = None,
        vad_threshold: float = 0.5,
    ):
        self.nanopitch_root = Path(nanopitch_root)
        self.decoder = decoder
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_config = feature_config or NanoPitchFeatureConfig()
        self.vad_threshold = vad_threshold

        self._nanopitch = _load_nanopitch_module(self.nanopitch_root)
        self.feature_extractor = self._build_feature_extractor().to(self.device)
        self.model: torch.nn.Module | None = None
        self.checkpoint_path: Path | None = None
        self.load_weights(checkpoint_path)

    def load_weights(self, checkpoint_path: str | Path) -> None:
        resolved = resolve_nanopitch_checkpoint(checkpoint_path, self.nanopitch_root)
        warnings.warn(
            "Loading NanoPitch checkpoint via torch.load(). Only load checkpoints from trusted sources.",
            RuntimeWarning,
        )
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model_kwargs = checkpoint.get("model_kwargs", {}) if isinstance(checkpoint, dict) else {}

        model = self._nanopitch.NanoPitch(**model_kwargs)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)

        self.model = model
        self.checkpoint_path = resolved

    def predict(self, wav_path: str | Path) -> dict:
        if self.model is None or self.checkpoint_path is None:
            raise RuntimeError("NanoPitch weights have not been loaded")

        audio = self._load_audio(wav_path)
        with torch.inference_mode():
            mel = self.feature_extractor(audio.unsqueeze(0))
            vad, posterior, _ = self.model(mel)

        vad_np = vad.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
        posterior_np = posterior.squeeze(0).detach().cpu().numpy().astype(np.float32)
        f0_hz = self._decode(posterior_np)
        pitch_confidence = posterior_np.max(axis=1).astype(np.float32)
        voiced = ((vad_np >= self.vad_threshold) & (f0_hz > 0)).astype(np.float32)
        f0_hz = np.where(voiced > 0, f0_hz, 0.0).astype(np.float32)
        times = (
            np.arange(len(f0_hz), dtype=np.float32)
            * self.feature_config.hop_length
            / self.feature_config.sample_rate
        )

        return NanoPitchPrediction(
            times=times,
            f0_hz=f0_hz,
            voiced=voiced,
            vad_prob=vad_np,
            pitch_confidence=pitch_confidence,
            posteriorgram=posterior_np,
            checkpoint_path=str(self.checkpoint_path),
            decoder=self.decoder,
        ).to_dict()

    def _decode(self, posteriorgram: np.ndarray) -> np.ndarray:
        if self.decoder == "offline":
            return self._nanopitch.viterbi_decode(posteriorgram).astype(np.float32)
        if self.decoder == "realtime":
            return self._nanopitch.viterbi_decode_realtime(posteriorgram).astype(np.float32)
        if self.decoder == "argmax":
            bins = posteriorgram.argmax(axis=1)
            confidence = posteriorgram.max(axis=1)
            f0 = self._nanopitch.bin_to_f0(bins).astype(np.float32)
            return np.where(confidence >= 0.3, f0, 0.0).astype(np.float32)
        raise ValueError(f"Unsupported decoder: {self.decoder}")

    def _load_audio(self, wav_path: str | Path) -> torch.Tensor:
        try:
            import torchaudio
            import torchaudio.functional as F_audio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for NanoPitch extraction") from exc

        audio, sample_rate = torchaudio.load(wav_path)
        audio = audio.mean(dim=0) if audio.ndim == 2 else audio
        audio = audio.to(self.device)
        if sample_rate != self.feature_config.sample_rate:
            audio = F_audio.resample(audio, sample_rate, self.feature_config.sample_rate)
        return audio.float().contiguous()

    def _build_feature_extractor(self) -> torch.nn.Module:
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for NanoPitch feature extraction") from exc

        return torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.feature_config.sample_rate,
                n_fft=self.feature_config.n_fft,
                win_length=self.feature_config.win_length,
                hop_length=self.feature_config.hop_length,
                f_min=self.feature_config.fmin,
                f_max=self.feature_config.fmax,
                n_mels=self.feature_config.n_mels,
                power=2.0,
                center=True,
            ),
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=self.feature_config.top_db),
            _TransposeMel(),
        )


class _TransposeMel(torch.nn.Module):
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return mel.transpose(1, 2).contiguous()


def resolve_nanopitch_checkpoint(checkpoint_path: str | Path, nanopitch_root: str | Path = "NanoPitch_v2") -> Path:
    root = Path(nanopitch_root)
    key = str(checkpoint_path)
    aliases = {
        "exp1": root / "test/runs/exp1/checkpoints/best.pth",
        "exp2": root / "test/runs/exp2/checkpoints/best.pth",
        "priontu": root / "submissions/Priontu_Chowdhury/weights.pth",
        "priontu_chowdhury": root / "submissions/Priontu_Chowdhury/weights.pth",
        "default": root / "submissions/Priontu_Chowdhury/weights.pth",
    }
    candidate = aliases.get(key.lower(), Path(checkpoint_path))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.exists():
        for name in ("weights.pth", "best.pth", "checkpoint.pth"):
            submission_candidate = root / "submissions" / key / name
            if submission_candidate.exists():
                candidate = submission_candidate
                break
    if not candidate.exists():
        raise FileNotFoundError(f"NanoPitch checkpoint not found: {checkpoint_path}")
    return candidate.resolve()


def _load_nanopitch_module(nanopitch_root: Path):
    model_path = nanopitch_root / "training" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"NanoPitch model.py not found: {model_path}")

    module_name = "_voice_coach_nanopitch_model"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import NanoPitch model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
