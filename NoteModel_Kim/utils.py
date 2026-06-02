"""
Audio processing utilities and annotation helpers.

Covers:
  - WAV loading / mono / resample
  - Log-mel spectrogram computation
  - Gaussian soft-label construction for onsets and offsets
  - Peak-picking to convert probability curves into timestamps
  - MusicXML / JSON annotation parsers
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NoteAnnotation:
    """A single note with start and end times in seconds."""
    onset: float
    offset: float

    @property
    def duration(self) -> float:
        return self.offset - self.onset


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(
    path: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """
    Load a WAV file, convert to mono, and resample to *target_sr*.

    Returns:
        waveform:    [1, N] float32 tensor
        sample_rate: target_sr
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    # data: [N] mono or [N, C] multichannel → mix to mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)  # [1, N]

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

    return waveform, target_sr


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0,
) -> torch.Tensor:
    """
    Compute log-mel spectrogram.

    Args:
        waveform: [1, N] mono audio tensor

    Returns:
        log_mel: [1, n_mels, T]
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
    )
    mel_spec = mel_transform(waveform)               # [1, n_mels, T]
    return torch.log(mel_spec + 1e-9)


def normalize_spectrogram(spec: torch.Tensor) -> torch.Tensor:
    """Per-sample zero-mean unit-variance normalization."""
    return (spec - spec.mean()) / (spec.std() + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Frame times
# ─────────────────────────────────────────────────────────────────────────────

def frames_to_time(
    n_frames: int,
    hop_length: int = 256,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Return centre time (seconds) for each spectrogram frame."""
    return np.arange(n_frames) * hop_length / sample_rate


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian soft-label construction
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_labels(
    boundaries: List[float],
    frame_times: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Build frame-level labels as max-pooled Gaussians around *boundaries*."""
    labels = np.zeros(len(frame_times), dtype=np.float32)
    for t in boundaries:
        gauss = np.exp(-0.5 * ((frame_times - t) / sigma) ** 2)
        labels = np.maximum(labels, gauss)
    return labels


def build_onset_labels(
    notes: List[NoteAnnotation],
    frame_times: np.ndarray,
    sigma: float = 0.02,
) -> np.ndarray:
    """
    Frame-level Gaussian soft targets centred on note onsets.

    Args:
        notes:       List of NoteAnnotation objects.
        frame_times: [T] centre time (s) for each frame.
        sigma:       Gaussian std deviation in seconds.

    Returns:
        labels: [T] float32 in [0, 1]
    """
    return _gaussian_labels([n.onset for n in notes], frame_times, sigma)


def build_offset_labels(
    notes: List[NoteAnnotation],
    frame_times: np.ndarray,
    sigma: float = 0.02,
) -> np.ndarray:
    """
    Frame-level Gaussian soft targets centred on note offsets.

    Args:
        notes:       List of NoteAnnotation objects.
        frame_times: [T] centre time (s) for each frame.
        sigma:       Gaussian std deviation in seconds.

    Returns:
        labels: [T] float32 in [0, 1]
    """
    return _gaussian_labels([n.offset for n in notes], frame_times, sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Peak picking
# ─────────────────────────────────────────────────────────────────────────────

def peak_pick_onsets(
    probs: np.ndarray,
    frame_times: np.ndarray,
    threshold: float = 0.3,
    min_distance_frames: int = 3,
) -> List[float]:
    """
    Extract onset timestamps from a probability curve.

    Uses scipy.signal.find_peaks with a height threshold and minimum
    inter-peak distance constraint.

    Args:
        probs:               [T] onset probability array.
        frame_times:         [T] time in seconds per frame.
        threshold:           Minimum peak height.
        min_distance_frames: Minimum frames between consecutive peaks.

    Returns:
        Sorted list of onset timestamps in seconds.
    """
    peaks, _ = find_peaks(probs, height=threshold, distance=min_distance_frames)
    return sorted(float(frame_times[i]) for i in peaks)


def peak_pick_offsets(
    probs: np.ndarray,
    frame_times: np.ndarray,
    threshold: float = 0.3,
    min_distance_frames: int = 3,
) -> List[float]:
    """
    Extract offset timestamps from a probability curve.

    Args:
        probs:               [T] offset probability array.
        frame_times:         [T] time in seconds per frame.
        threshold:           Minimum peak height.
        min_distance_frames: Minimum frames between consecutive peaks.

    Returns:
        Sorted list of offset timestamps in seconds.
    """
    peaks, _ = find_peaks(probs, height=threshold, distance=min_distance_frames)
    return sorted(float(frame_times[i]) for i in peaks)


def pair_onsets_offsets(
    onsets: List[float],
    offsets: List[float],
) -> List[dict]:
    """
    Greedily pair onset/offset times into note boundaries.

    Each onset is matched with the next available offset that comes after it.

    Returns:
        List of dicts: {"onset_time", "offset_time", "duration"}
        duration is None when no matching offset was found.
    """
    notes = []
    offsets_sorted = sorted(offsets)
    off_idx = 0

    for onset in sorted(onsets):
        while off_idx < len(offsets_sorted) and offsets_sorted[off_idx] <= onset:
            off_idx += 1

        if off_idx < len(offsets_sorted):
            offset = offsets_sorted[off_idx]
            notes.append({
                "onset_time": onset,
                "offset_time": offset,
                "duration": round(offset - onset, 6),
            })
            off_idx += 1
        else:
            notes.append({"onset_time": onset, "offset_time": None, "duration": None})

    return notes


# ─────────────────────────────────────────────────────────────────────────────
# Annotation parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_notes(json_path: Union[str, Path]) -> List[NoteAnnotation]:
    """
    Parse note annotations from a JSON file.

    Expected format::

        [{"onset": 0.5, "offset": 1.2}, ...]

    Returns:
        Sorted list of NoteAnnotation.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    notes = [NoteAnnotation(onset=float(d["onset"]), offset=float(d["offset"])) for d in data]
    return sorted(notes, key=lambda n: n.onset)


def parse_gtsinger_json(json_path: Union[str, Path]) -> List[NoteAnnotation]:
    """
    Parse note timings from a GTSinger annotation JSON file.

    GTSinger JSON format — each entry is a word::

        [
          {
            "word": "let",
            "note_start": [1.59],
            "note_end":   [1.71],
            ...
          },
          ...
        ]

    ``note_start`` / ``note_end`` are lists because one word can span
    multiple notes (e.g. melismas).  Rest/silence entries (MIDI note == 0
    or word == "<AP>") are skipped.

    Returns:
        Sorted list of NoteAnnotation with times in seconds.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    notes: List[NoteAnnotation] = []
    for entry in data:
        note_vals = entry.get("note", [])
        note_starts = entry.get("note_start", [])
        note_ends = entry.get("note_end", [])

        for i, (start, end) in enumerate(zip(note_starts, note_ends)):
            # Skip silence / rest markers
            midi = note_vals[i] if i < len(note_vals) else None
            if midi == 0 or entry.get("word", "") in ("<AP>", "<SP>"):
                continue
            if end <= start:
                continue
            notes.append(NoteAnnotation(onset=float(start), offset=float(end)))

    return sorted(notes, key=lambda n: n.onset)


def parse_musicxml_notes(xml_path: Union[str, Path]) -> List[NoteAnnotation]:
    """
    Parse note timings from a MusicXML file (requires *music21*).

    Handles a single constant tempo; for pieces with tempo changes the
    converter uses the first MetronomeMark found (default 120 BPM).

    Returns:
        Sorted list of NoteAnnotation with times in seconds.
    """
    try:
        import music21
        from music21 import note as m21note, chord as m21chord
    except ImportError:
        raise ImportError("music21 is required for MusicXML parsing: pip install music21")

    score = music21.converter.parse(str(xml_path))

    # Resolve tempo — fall back to 120 BPM if no mark is present.
    tempo_marks = score.flatten().getElementsByClass("MetronomeMark")
    bpm = float(tempo_marks[0].number) if len(tempo_marks) > 0 else 120.0
    spq = 60.0 / bpm  # seconds per quarter note

    notes: List[NoteAnnotation] = []
    for part in score.parts:
        for element in part.flatten().notesAndRests:
            if isinstance(element, (m21note.Note, m21chord.Chord)):
                onset_s = float(element.offset) * spq
                dur_s = float(element.duration.quarterLength) * spq
                notes.append(NoteAnnotation(onset=onset_s, offset=onset_s + dur_s))

    return sorted(notes, key=lambda n: n.onset)
