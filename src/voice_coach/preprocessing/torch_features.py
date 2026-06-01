from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchAudioFeatureConfig:
    sample_rate: int = 48000
    n_mels: int = 80
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    fmin: float = 30.0
    fmax: float | None = 16000.0
    top_db: float = 80.0

    @property
    def frame_rate_hz(self) -> float:
        return self.sample_rate / self.hop_length


class TorchLogMelExtractor(torch.nn.Module):
    """GPU-friendly log-mel extractor backed by torchaudio."""

    def __init__(self, config: TorchAudioFeatureConfig):
        super().__init__()
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for GPU audio feature extraction") from exc

        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.fmin,
            f_max=config.fmax,
            n_mels=config.n_mels,
            power=2.0,
            center=True,
            normalized=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=config.top_db)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Return log-mel features shaped [batch, frames, n_mels]."""

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        audio = audio.float()
        mel = self.mel(audio)
        log_mel = self.to_db(mel)
        return log_mel.transpose(1, 2).contiguous()


def resample_audio(
    audio: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    if original_sample_rate == target_sample_rate:
        return audio

    try:
        import torchaudio.functional as F_audio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required for GPU audio resampling") from exc

    return F_audio.resample(
        audio,
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
    )


def pad_audio_batch(audio_items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor(
        [item.numel() for item in audio_items],
        dtype=torch.long,
        device=audio_items[0].device if audio_items else None,
    )
    max_len = int(lengths.max().item()) if len(audio_items) else 0
    device = audio_items[0].device if audio_items else torch.device("cpu")
    batch = torch.zeros((len(audio_items), max_len), dtype=torch.float32, device=device)
    for index, audio in enumerate(audio_items):
        batch[index, : audio.numel()] = audio.float()
    return batch, lengths


def audio_lengths_to_frame_lengths(lengths: torch.Tensor, config: TorchAudioFeatureConfig) -> torch.Tensor:
    return torch.div(lengths, config.hop_length, rounding_mode="floor") + 1


def frame_times(num_frames: int, config: TorchAudioFeatureConfig, device: torch.device | str) -> torch.Tensor:
    frames = torch.arange(num_frames, dtype=torch.float32, device=device)
    return frames * (config.hop_length / config.sample_rate)
