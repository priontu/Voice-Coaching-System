"""
fusion/event_alignment.py - Cross-event temporal alignment and annotation.

Provides deterministic algorithms for:
  • note ↔ phoneme overlap analysis and annotation
  • word-to-note assignment
  • voiced-region extraction from boolean masks
  • phrase segmentation from note inter-onset gaps
  • nearest-neighbour boundary snapping

All overlap computations are based on raw time intervals (no frame rounding),
which avoids quantisation error accumulating across long sequences.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from preprocessing.timestamps import HOP_LENGTH, SAMPLE_RATE, snap_to_frame
from utils.types import (
    LyricEvent,
    NoteEvent,
    PhonemeSegment,
    PhraseEvent,
    TemporalRegion,
    WordEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overlap utilities
# ---------------------------------------------------------------------------

def overlap_duration(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    """Return the duration of the overlap between two intervals in seconds."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def overlap_fraction(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
    reference: str = "a",
) -> float:
    """
    Fraction of the reference interval covered by the overlap with [b_start, b_end].

    Args:
        reference: 'a' → fraction of [a_start, a_end]; 'b' → fraction of b.
    """
    dur = overlap_duration(a_start, a_end, b_start, b_end)
    if reference == "a":
        denom = a_end - a_start
    else:
        denom = b_end - b_start
    return dur / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Phoneme ↔ note alignment
# ---------------------------------------------------------------------------

def align_phonemes_to_notes(
    note_events: List[NoteEvent],
    phoneme_segments: List[PhonemeSegment],
    min_overlap_s: float = 0.01,
) -> Dict[int, List[int]]:
    """
    Map each note to the phoneme segments that overlap it.

    Args:
        note_events:       List of NoteEvent.
        phoneme_segments:  List of PhonemeSegment.
        min_overlap_s:     Minimum overlap duration to register a mapping.

    Returns:
        {note_idx: [phoneme_indices]} — keys for all notes, even if empty list.
    """
    mapping: Dict[int, List[int]] = {i: [] for i in range(len(note_events))}
    for pi, seg in enumerate(phoneme_segments):
        for ni, note in enumerate(note_events):
            note_end = note.offset_time if note.offset_time is not None else (
                note.onset_time + (note.duration or 0.0)
            )
            ov = overlap_duration(note.onset_time, note_end, seg.start_time, seg.end_time)
            if ov >= min_overlap_s:
                mapping[ni].append(pi)
    return mapping


def annotate_notes_with_phonemes(
    note_events: List[NoteEvent],
    phoneme_segments: List[PhonemeSegment],
    min_overlap_s: float = 0.01,
) -> List[NoteEvent]:
    """
    Return a new NoteEvent list with phoneme_labels and lyric_text populated.

    Phoneme labels preserve temporal order and are deduplicated while keeping
    the first occurrence of each label (maintains left-to-right reading order).
    """
    mapping = align_phonemes_to_notes(note_events, phoneme_segments, min_overlap_s)
    annotated: List[NoteEvent] = []
    for ni, note in enumerate(note_events):
        indices = mapping.get(ni, [])
        if indices:
            ordered = sorted(indices, key=lambda i: phoneme_segments[i].start_time)
            labels: List[str] = []
            seen: set = set()
            for i in ordered:
                label = phoneme_segments[i].phoneme
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
            note.phoneme_labels = labels
            note.lyric_text = "-".join(labels)
        annotated.append(note)
    return annotated


# ---------------------------------------------------------------------------
# Word ↔ note alignment
# ---------------------------------------------------------------------------

def align_words_to_notes(
    word_events: List[WordEvent],
    note_events: List[NoteEvent],
    min_overlap_s: float = 0.01,
) -> Dict[int, List[int]]:
    """
    Map each note to the word events that overlap it.

    Returns:
        {note_idx: [word_indices]}
    """
    mapping: Dict[int, List[int]] = {i: [] for i in range(len(note_events))}
    for wi, we in enumerate(word_events):
        for ni, note in enumerate(note_events):
            note_end = note.offset_time if note.offset_time is not None else (
                note.onset_time + (note.duration or 0.0)
            )
            ov = overlap_duration(note.onset_time, note_end, we.start_time, we.end_time)
            if ov >= min_overlap_s:
                mapping[ni].append(wi)
    return mapping


def annotate_words_with_notes(
    word_events: List[WordEvent],
    note_events: List[NoteEvent],
    min_overlap_s: float = 0.01,
) -> List[WordEvent]:
    """
    Assign the dominant note_idx to each WordEvent in-place and return the list.

    The dominant note is the one with the largest overlap with the word interval.
    """
    for we in word_events:
        best_overlap = 0.0
        best_note_idx: Optional[int] = None
        for ni, note in enumerate(note_events):
            note_end = note.offset_time if note.offset_time is not None else (
                note.onset_time + (note.duration or 0.0)
            )
            ov = overlap_duration(note.onset_time, note_end, we.start_time, we.end_time)
            if ov > best_overlap and ov >= min_overlap_s:
                best_overlap = ov
                best_note_idx = ni
        we.note_idx = best_note_idx
    return word_events


def annotate_lyrics_with_notes(
    lyric_events: List[LyricEvent],
    note_events: List[NoteEvent],
    min_overlap_s: float = 0.005,
) -> List[LyricEvent]:
    """
    Assign the dominant note_idx to each LyricEvent in-place and return the list.
    """
    for le in lyric_events:
        best_overlap = 0.0
        best_note_idx: Optional[int] = None
        for ni, note in enumerate(note_events):
            note_end = note.offset_time if note.offset_time is not None else (
                note.onset_time + (note.duration or 0.0)
            )
            ov = overlap_duration(note.onset_time, note_end, le.start_time, le.end_time)
            if ov > best_overlap and ov >= min_overlap_s:
                best_overlap = ov
                best_note_idx = ni
        le.note_idx = best_note_idx
    return lyric_events


# ---------------------------------------------------------------------------
# Voiced region extraction
# ---------------------------------------------------------------------------

def build_voiced_regions(
    timestamps: np.ndarray,
    voiced: np.ndarray,
    min_duration_s: float = 0.02,
    label_voiced: str = "voiced",
    label_unvoiced: str = "unvoiced",
) -> List[TemporalRegion]:
    """
    Convert a boolean voiced mask into contiguous TemporalRegion objects.

    Consecutive frames with the same voiced state are merged. Regions shorter
    than min_duration_s are dropped.

    Args:
        timestamps:    Canonical frame timestamps (N,) seconds.
        voiced:        Boolean mask (N,).
        min_duration_s: Minimum region length to retain.

    Returns:
        List of TemporalRegion sorted by start_time.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)

    if len(timestamps) == 0:
        return []

    hop_s = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0.01

    regions: List[TemporalRegion] = []
    current_state = bool(voiced[0])
    region_start = float(timestamps[0])

    for i in range(1, len(timestamps)):
        state = bool(voiced[i])
        if state != current_state:
            region_end = float(timestamps[i - 1]) + hop_s
            dur = region_end - region_start
            if dur >= min_duration_s:
                label = label_voiced if current_state else label_unvoiced
                regions.append(TemporalRegion(label=label, start_time=region_start, end_time=region_end))
            current_state = state
            region_start = float(timestamps[i])

    # Final region
    region_end = float(timestamps[-1]) + hop_s
    dur = region_end - region_start
    if dur >= min_duration_s:
        label = label_voiced if current_state else label_unvoiced
        regions.append(TemporalRegion(label=label, start_time=region_start, end_time=region_end))

    return regions


# ---------------------------------------------------------------------------
# Phrase segmentation
# ---------------------------------------------------------------------------

def build_phrase_events(
    note_events: List[NoteEvent],
    max_gap_s: float = 0.5,
    word_events: Optional[List[WordEvent]] = None,
) -> List[PhraseEvent]:
    """
    Group consecutive notes into phrases based on inter-note gap.

    A new phrase starts whenever the gap between the end of the previous note
    and the onset of the current note exceeds max_gap_s.

    Args:
        note_events:   NoteEvent list sorted by onset_time.
        max_gap_s:     Maximum silence gap within a phrase (seconds).
        word_events:   Optional WordEvent list; if provided, phrase word_indices
                       are populated.

    Returns:
        List of PhraseEvent sorted by start_time.
    """
    if not note_events:
        return []

    def _note_end(n: NoteEvent) -> float:
        if n.offset_time is not None:
            return n.offset_time
        return n.onset_time + (n.duration or 0.3)

    phrases: List[PhraseEvent] = []
    current_indices: List[int] = [0]

    for ni in range(1, len(note_events)):
        prev_end = _note_end(note_events[ni - 1])
        gap = note_events[ni].onset_time - prev_end
        if gap <= max_gap_s:
            current_indices.append(ni)
        else:
            phrases.append(_make_phrase(current_indices, note_events, len(phrases), word_events))
            current_indices = [ni]

    phrases.append(_make_phrase(current_indices, note_events, len(phrases), word_events))
    return phrases


def _make_phrase(
    note_indices: List[int],
    note_events: List[NoteEvent],
    phrase_idx: int,
    word_events: Optional[List[WordEvent]],
) -> PhraseEvent:
    first = note_events[note_indices[0]]
    last = note_events[note_indices[-1]]

    def _note_end(n: NoteEvent) -> float:
        if n.offset_time is not None:
            return n.offset_time
        return n.onset_time + (n.duration or 0.3)

    phrase_end = _note_end(last)

    word_indices: List[int] = []
    if word_events:
        for wi, we in enumerate(word_events):
            ov = overlap_duration(first.onset_time, phrase_end, we.start_time, we.end_time)
            if ov > 0:
                word_indices.append(wi)

    return PhraseEvent(
        start_time=first.onset_time,
        end_time=phrase_end,
        note_indices=list(note_indices),
        word_indices=word_indices,
        phrase_idx=phrase_idx,
    )


# ---------------------------------------------------------------------------
# Boundary snapping
# ---------------------------------------------------------------------------

def snap_event_boundaries(
    events: List,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> List:
    """
    Snap start_time / end_time of a list of events to the nearest canonical frame.

    Works with NoteEvent, LyricEvent, WordEvent, PhraseEvent, TemporalRegion —
    any object that has start_time / end_time attributes. Modifies in-place.

    Args:
        events:       List of event objects with start_time / end_time.
        hop_length:   Canonical hop size.
        sample_rate:  Canonical sample rate.

    Returns:
        The same list (mutated in-place) for chaining.
    """
    for ev in events:
        if hasattr(ev, "start_time"):
            ev.start_time = snap_to_frame(ev.start_time, hop_length=hop_length, sample_rate=sample_rate)
        if hasattr(ev, "end_time"):
            ev.end_time = snap_to_frame(ev.end_time, hop_length=hop_length, sample_rate=sample_rate)
        if hasattr(ev, "onset_time"):
            ev.onset_time = snap_to_frame(ev.onset_time, hop_length=hop_length, sample_rate=sample_rate)
        if hasattr(ev, "offset_time") and ev.offset_time is not None:
            ev.offset_time = snap_to_frame(ev.offset_time, hop_length=hop_length, sample_rate=sample_rate)
    return events


# ---------------------------------------------------------------------------
# Temporal overlap scoring
# ---------------------------------------------------------------------------

def score_note_phoneme_alignment(
    note_events: List[NoteEvent],
    phoneme_segments: List[PhonemeSegment],
) -> Dict[str, float]:
    """
    Compute aggregate overlap statistics between note and phoneme timelines.

    Returns a dict with:
        covered_fraction:   fraction of total phoneme duration covered by notes
        mean_overlap:       mean per-phoneme overlap fraction (reference = phoneme)
        n_unmatched:        number of phonemes with zero note overlap
    """
    if not note_events or not phoneme_segments:
        return {"covered_fraction": 0.0, "mean_overlap": 0.0, "n_unmatched": len(phoneme_segments)}

    total_phoneme_dur = sum(s.duration for s in phoneme_segments)
    covered_dur = 0.0
    fractions: List[float] = []
    n_unmatched = 0

    for seg in phoneme_segments:
        best_frac = 0.0
        for note in note_events:
            note_end = note.offset_time or (note.onset_time + (note.duration or 0.0))
            frac = overlap_fraction(seg.start_time, seg.end_time, note.onset_time, note_end, reference="a")
            best_frac = max(best_frac, frac)
        fractions.append(best_frac)
        covered_dur += best_frac * seg.duration
        if best_frac == 0.0:
            n_unmatched += 1

    covered_fraction = covered_dur / total_phoneme_dur if total_phoneme_dur > 0 else 0.0
    mean_overlap = float(np.mean(fractions)) if fractions else 0.0

    return {
        "covered_fraction": round(covered_fraction, 4),
        "mean_overlap": round(mean_overlap, 4),
        "n_unmatched": n_unmatched,
    }
