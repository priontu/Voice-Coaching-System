"""
fusion/lyric_events.py - Lyric and word event construction from phoneme segments.

Transforms the flat list of PhonemeSegment objects produced by the phoneme
model into two levels of lyric representation:

  PhonemeSegment list
        ↓
  LyricEvent  (1:1 with segments — adds lyrical context fields)
        ↓
  WordEvent   (groups of consecutive LyricEvents separated by a silence gap)

Word grouping is purely proximity-based because the phoneme model does not
produce explicit word boundaries. The word_gap_s threshold controls grouping
granularity; the default (100 ms) works well for sung syllables.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from utils.types import LyricEvent, PhonemeSegment, WordEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_word_event(group: List[LyricEvent], word_idx: int) -> WordEvent:
    """Assemble a WordEvent from a contiguous group of LyricEvents."""
    text = "-".join(le.phoneme for le in group)
    start = group[0].start_time
    end = group[-1].end_time
    confidence = float(np.mean([le.confidence for le in group]))
    return WordEvent(
        text=text,
        start_time=start,
        end_time=end,
        phoneme_events=list(group),
        confidence=confidence,
        word_idx=word_idx,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_lyric_events(
    phoneme_segments: List[PhonemeSegment],
    timestamps: Optional[np.ndarray] = None,
    word_gap_s: float = 0.10,
    min_duration_s: float = 0.0,
) -> Tuple[List[LyricEvent], List[WordEvent]]:
    """
    Build LyricEvent and WordEvent lists from phoneme segment boundaries.

    LyricEvents are created 1:1 from PhonemeSegments, preserving confidence
    and timing. WordEvents are built by grouping consecutive LyricEvents whose
    inter-segment gap is smaller than word_gap_s.

    After construction, word_idx is back-annotated on each LyricEvent so
    callers can navigate between the two levels directly.

    Args:
        phoneme_segments: List of PhonemeSegment objects (any order; sorted internally).
        timestamps:       Canonical timestamp array (unused here, reserved for
                          future frame-anchoring of lyric events).
        word_gap_s:       Maximum gap between consecutive phonemes that are still
                          grouped into the same word (seconds). Default 100 ms.
        min_duration_s:   Phoneme segments shorter than this are discarded.

    Returns:
        Tuple (lyric_events, word_events), both sorted by start_time.
        word_idx on each LyricEvent points into the word_events list.
    """
    if not phoneme_segments:
        return [], []

    # Sort by start time; filter zero-length segments
    segs = sorted(phoneme_segments, key=lambda s: s.start_time)
    if min_duration_s > 0:
        segs = [s for s in segs if s.duration > min_duration_s]
    if not segs:
        return [], []

    # ── 1. Build LyricEvents (1:1 from PhonemeSegments) ────────────────────
    lyric_events: List[LyricEvent] = []
    for idx, seg in enumerate(segs):
        le = LyricEvent(
            phoneme=seg.phoneme,
            start_time=seg.start_time,
            end_time=seg.end_time,
            confidence=seg.confidence,
            lyric_idx=idx,
        )
        lyric_events.append(le)

    # ── 2. Group into WordEvents by silence gap ─────────────────────────────
    word_events: List[WordEvent] = []
    current_group: List[LyricEvent] = [lyric_events[0]]

    for le in lyric_events[1:]:
        gap = le.start_time - current_group[-1].end_time
        if gap < word_gap_s:
            current_group.append(le)
        else:
            word_events.append(_make_word_event(current_group, len(word_events)))
            current_group = [le]

    word_events.append(_make_word_event(current_group, len(word_events)))

    # ── 3. Back-annotate word_idx on LyricEvents ────────────────────────────
    for wi, we in enumerate(word_events):
        for le in we.phoneme_events:
            le.word_idx = wi

    logger.debug(
        "[lyric_events] Built %d lyric events → %d word events (gap=%.3fs)",
        len(lyric_events), len(word_events), word_gap_s,
    )
    return lyric_events, word_events


def merge_word_events(
    word_events: List[WordEvent],
    max_gap_s: float = 0.05,
) -> List[WordEvent]:
    """
    Merge word events that are separated by less than max_gap_s.

    Useful when the word_gap_s threshold used in build_lyric_events() was too
    aggressive and split a word across multiple WordEvent objects.

    Returns a new list of merged WordEvents with re-indexed word_idx values.
    """
    if not word_events:
        return []

    merged: List[WordEvent] = [word_events[0]]
    for we in word_events[1:]:
        prev = merged[-1]
        gap = we.start_time - prev.end_time
        if gap < max_gap_s:
            combined_phonemes = list(prev.phoneme_events) + list(we.phoneme_events)
            prev_conf = prev.confidence * len(prev.phoneme_events)
            we_conf = we.confidence * len(we.phoneme_events)
            n_total = len(combined_phonemes)
            merged[-1] = WordEvent(
                text=f"{prev.text}-{we.text}",
                start_time=prev.start_time,
                end_time=we.end_time,
                phoneme_events=combined_phonemes,
                confidence=(prev_conf + we_conf) / n_total if n_total > 0 else 1.0,
                word_idx=prev.word_idx,
            )
        else:
            merged.append(WordEvent(
                text=we.text,
                start_time=we.start_time,
                end_time=we.end_time,
                phoneme_events=list(we.phoneme_events),
                confidence=we.confidence,
                word_idx=len(merged),
            ))

    # Re-index word_idx
    for wi, we in enumerate(merged):
        we.word_idx = wi
        for le in we.phoneme_events:
            le.word_idx = wi

    return merged
