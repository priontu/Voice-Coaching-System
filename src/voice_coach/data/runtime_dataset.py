from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from voice_coach.data.gtsinger import parse_gtsinger_json_notes
from voice_coach.data.manifest import Recording, load_manifest
from voice_coach.data.musicxml import NoteEvent, parse_musicxml_notes
from voice_coach.data.textgrid import TextGridInterval, phoneme_intervals


@dataclass(frozen=True)
class RuntimeExample:
    recording: Recording
    audio: torch.Tensor
    audio_sample_rate: int
    notes: list[NoteEvent]
    phonemes: list[TextGridInterval]


class VoiceCoachRuntimeDataset(Dataset):
    """Load raw examples for on-the-fly training/evaluation conversion."""

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        recordings: list[Recording] | None = None,
        target_sample_rate: int = 48000,
        cache_references: bool = True,
        prefer_aligned_json_notes: bool = True,
    ):
        if recordings is None:
            if manifest_path is None:
                raise ValueError("Either manifest_path or recordings must be provided")
            recordings = load_manifest(manifest_path)

        self.recordings = recordings
        self.target_sample_rate = target_sample_rate
        self.cache_references = cache_references
        self.prefer_aligned_json_notes = prefer_aligned_json_notes
        self._reference_cache: dict[str, tuple[list[NoteEvent], list[TextGridInterval]]] = {}

        if cache_references:
            for recording in self.recordings:
                self._reference_cache[recording.recording_id] = self._load_references(recording)

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, index: int) -> RuntimeExample:
        recording = self.recordings[index]
        audio, native_sample_rate = _load_audio(recording.wav_path)
        cached = self._reference_cache.get(recording.recording_id)
        if cached is None:
            cached = self._load_references(recording)
            self._reference_cache[recording.recording_id] = cached
        notes, phonemes = cached
        return RuntimeExample(
            recording=recording,
            audio=audio,
            audio_sample_rate=native_sample_rate,
            notes=notes,
            phonemes=phonemes,
        )

    def _load_references(self, recording: Recording) -> tuple[list[NoteEvent], list[TextGridInterval]]:
        json_path = recording.wav_path.with_suffix(".json")
        if self.prefer_aligned_json_notes and json_path.exists():
            notes = parse_gtsinger_json_notes(json_path)
            if not notes:
                notes = parse_musicxml_notes(recording.musicxml_path)
        else:
            notes = parse_musicxml_notes(recording.musicxml_path)
        phonemes: list[TextGridInterval] = []
        if recording.textgrid_path is not None and recording.textgrid_path.exists():
            phonemes = phoneme_intervals(recording.textgrid_path)
        return notes, phonemes


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required for runtime audio loading") from exc

    audio, native_sr = torchaudio.load(path)
    audio = audio.mean(dim=0) if audio.ndim == 2 else audio
    return audio.float().contiguous(), native_sr
