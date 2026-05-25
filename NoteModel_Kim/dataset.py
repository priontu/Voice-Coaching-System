"""
Dataset and DataLoader construction for note onset/offset detection.

Expects either:
  (a) a manifest JSON file — list of {"audio": "...", "label": "..."} dicts, or
  (b) a data_dir with audio/ and labels/ subdirectories.

Label files may be .json or .xml (MusicXML).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from utils import (
    NoteAnnotation,
    build_offset_labels,
    build_onset_labels,
    compute_log_mel_spectrogram,
    frames_to_time,
    load_audio,
    normalize_spectrogram,
    parse_gtsinger_json,
    parse_json_notes,
    parse_musicxml_notes,
)

logger = logging.getLogger(__name__)


class NoteDetectionDataset(Dataset):
    """
    PyTorch Dataset for note boundary detection.

    Each item yields::

        {
            "spectrogram":    [1, n_mels, T]   log-mel, normalised
            "onset_labels":   [T]               Gaussian soft targets
            "offset_labels":  [T]               Gaussian soft targets
        }

    When *segment_length* is set every item is randomly cropped (or
    zero-padded) to exactly *segment_frames* frames so batches have a
    uniform time dimension.

    Args:
        manifest_path:  Path to JSON manifest.  Either this or *data_dir*
                        must be provided.
        data_dir:       Root directory containing audio/ and labels/.
        sample_rate:    Target sample rate in Hz.
        n_fft:          FFT window size.
        hop_length:     Spectrogram hop size in samples.
        n_mels:         Number of mel bins.
        fmin:           Minimum mel frequency.
        fmax:           Maximum mel frequency.
        label_sigma:    Gaussian soft-label sigma in seconds.
        segment_length: Fixed segment length in seconds (None = variable).
        augment:        Apply SpecAugment-style augmentation.
    """

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        label_sigma: float = 0.02,
        segment_length: Optional[float] = 10.0,
        augment: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.label_sigma = label_sigma
        self.augment = augment

        self.segment_frames: Optional[int] = (
            int(segment_length * sample_rate / hop_length)
            if segment_length is not None
            else None
        )

        self._freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self._time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

        self.samples: List[Dict[str, str]] = []

        if manifest_path is not None:
            self._load_manifest(manifest_path)
        elif data_dir is not None:
            self._discover_data(data_dir)
        else:
            raise ValueError("Provide either manifest_path or data_dir.")

    # ── Setup ─────────────────────────────────────────────────────────────

    def _load_manifest(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        logger.info("Loaded %d samples from manifest %s", len(self.samples), path)

    def _discover_data(self, data_dir: str) -> None:
        root = Path(data_dir)
        audio_dir = root / "audio"
        label_dir = root / "labels"

        for wav in sorted(audio_dir.glob("*.wav")):
            stem = wav.stem
            label = label_dir / f"{stem}.json"
            if not label.exists():
                label = label_dir / f"{stem}.xml"
            if label.exists():
                self.samples.append({"audio": str(wav), "label": str(label)})
            else:
                logger.warning("No label found for %s — skipping.", wav.name)

        logger.info("Discovered %d samples in %s", len(self.samples), data_dir)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_notes(self, label_path: str) -> List[NoteAnnotation]:
        if label_path.endswith((".xml", ".musicxml")):
            return parse_musicxml_notes(label_path)
        # Detect GTSinger format by presence of "note_start" key in first entry
        if label_path.endswith(".json"):
            import json as _json
            with open(label_path, "r", encoding="utf-8") as f:
                peek = _json.load(f)
            if peek and "note_start" in peek[0]:
                return parse_gtsinger_json(label_path)
        return parse_json_notes(label_path)

    def _segment_or_pad(
        self,
        log_mel: torch.Tensor,
        onset_lbl: torch.Tensor,
        offset_lbl: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random crop or zero-pad to self.segment_frames."""
        T = log_mel.shape[-1]
        target = self.segment_frames  # type: ignore[assignment]

        if T >= target:
            start = int(torch.randint(0, T - target + 1, (1,)).item())
            log_mel = log_mel[:, :, start : start + target]
            onset_lbl = onset_lbl[start : start + target]
            offset_lbl = offset_lbl[start : start + target]
        else:
            pad = target - T
            log_mel = torch.nn.functional.pad(log_mel, (0, pad))
            onset_lbl = torch.nn.functional.pad(onset_lbl, (0, pad))
            offset_lbl = torch.nn.functional.pad(offset_lbl, (0, pad))

        return log_mel, onset_lbl, offset_lbl

    def _apply_augmentation(self, log_mel: torch.Tensor) -> torch.Tensor:
        log_mel = self._freq_mask(log_mel)
        log_mel = self._time_mask(log_mel)
        return log_mel

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        waveform, _ = load_audio(sample["audio"], target_sr=self.sample_rate)
        log_mel = compute_log_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )  # [1, n_mels, T]

        n_frames = log_mel.shape[-1]
        frame_times = frames_to_time(n_frames, self.hop_length, self.sample_rate)

        notes = self._load_notes(sample["label"])
        onset_lbl = torch.from_numpy(
            build_onset_labels(notes, frame_times, self.label_sigma)
        )
        offset_lbl = torch.from_numpy(
            build_offset_labels(notes, frame_times, self.label_sigma)
        )

        log_mel = normalize_spectrogram(log_mel)

        if self.augment:
            log_mel = self._apply_augmentation(log_mel)

        if self.segment_frames is not None:
            log_mel, onset_lbl, offset_lbl = self._segment_or_pad(
                log_mel, onset_lbl, offset_lbl
            )

        return {
            "spectrogram": log_mel,       # [1, n_mels, T]
            "onset_labels": onset_lbl,    # [T]
            "offset_labels": offset_lbl,  # [T]
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    batch_size: int = 16,
    num_workers: int = 4,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from manifest files.

    Args:
        train_manifest: Path to training manifest JSON.
        val_manifest:   Path to validation manifest JSON.
        batch_size:     Samples per batch.
        num_workers:    DataLoader worker processes.
        **dataset_kwargs: Forwarded to NoteDetectionDataset.

    Returns:
        (train_loader, val_loader)
    """
    train_ds = NoteDetectionDataset(
        manifest_path=train_manifest, augment=True, **dataset_kwargs
    )
    val_ds = NoteDetectionDataset(
        manifest_path=val_manifest, augment=False, **dataset_kwargs
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
