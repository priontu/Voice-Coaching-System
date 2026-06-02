"""
tests/test_lyric_events.py - Unit tests for fusion/lyric_events.py

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_lyric_events.py -v
"""

from __future__ import annotations

import pytest

from fusion.lyric_events import build_lyric_events, merge_word_events
from utils.types import LyricEvent, PhonemeSegment, WordEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _segs(*args):
    """Build a list of PhonemeSegment from (phoneme, start, end) triples."""
    return [PhonemeSegment(phoneme=p, start_time=s, end_time=e) for p, s, e in args]


def _three_segs():
    return _segs(
        ("AH", 0.0, 0.3),
        ("EH", 0.3, 0.7),
        ("IH", 0.7, 1.0),
    )


def _two_words():
    """Two phonemes, then a gap, then one more — should form two words."""
    return _segs(
        ("AH", 0.0, 0.2),
        ("EH", 0.2, 0.4),   # gap < 100ms → same word
        ("IH", 0.6, 0.9),   # gap 200ms → new word
    )


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestBuildLyricEventsBasic:
    def test_empty_input(self):
        lyric, word = build_lyric_events([])
        assert lyric == []
        assert word == []

    def test_returns_two_lists(self):
        result = build_lyric_events(_three_segs())
        assert isinstance(result, tuple) and len(result) == 2

    def test_lyric_count_equals_phoneme_count(self):
        segs = _three_segs()
        lyric, _ = build_lyric_events(segs)
        assert len(lyric) == len(segs)

    def test_lyric_event_types(self):
        lyric, word = build_lyric_events(_three_segs())
        for le in lyric:
            assert isinstance(le, LyricEvent)
        for we in word:
            assert isinstance(we, WordEvent)

    def test_phoneme_preserved(self):
        segs = _three_segs()
        lyric, _ = build_lyric_events(segs)
        phonemes = [le.phoneme for le in lyric]
        assert phonemes == ["AH", "EH", "IH"]

    def test_timestamps_preserved(self):
        segs = _three_segs()
        lyric, _ = build_lyric_events(segs)
        for le, seg in zip(lyric, segs):
            assert le.start_time == seg.start_time
            assert le.end_time == seg.end_time

    def test_confidence_preserved(self):
        segs = [PhonemeSegment("AH", 0.0, 0.3, confidence=0.75)]
        lyric, _ = build_lyric_events(segs)
        assert lyric[0].confidence == pytest.approx(0.75)

    def test_lyric_idx_assigned(self):
        lyric, _ = build_lyric_events(_three_segs())
        for i, le in enumerate(lyric):
            assert le.lyric_idx == i


# ---------------------------------------------------------------------------
# Duration calculation
# ---------------------------------------------------------------------------

class TestDuration:
    def test_positive_duration(self):
        lyric, _ = build_lyric_events(_three_segs())
        for le in lyric:
            assert le.duration > 0

    def test_correct_duration(self):
        segs = _segs(("AH", 0.0, 0.3))
        lyric, _ = build_lyric_events(segs)
        assert lyric[0].duration == pytest.approx(0.3)

    def test_word_duration_spans_phonemes(self):
        segs = _segs(("AH", 0.1, 0.4), ("EH", 0.4, 0.8))
        _, word = build_lyric_events(segs, word_gap_s=0.1)
        assert word[0].start_time == pytest.approx(0.1)
        assert word[0].end_time == pytest.approx(0.8)
        assert word[0].duration == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Word grouping
# ---------------------------------------------------------------------------

class TestWordGrouping:
    def test_all_close_form_one_word(self):
        segs = _segs(("AH", 0.0, 0.1), ("EH", 0.1, 0.2), ("IH", 0.2, 0.3))
        _, word = build_lyric_events(segs, word_gap_s=0.1)
        assert len(word) == 1

    def test_gap_splits_into_two_words(self):
        segs = _segs(
            ("AH", 0.0, 0.2),
            ("EH", 0.5, 0.8),   # 300ms gap > 100ms default
        )
        _, word = build_lyric_events(segs, word_gap_s=0.1)
        assert len(word) == 2

    def test_two_word_grouping(self):
        segs = _two_words()
        _, word = build_lyric_events(segs, word_gap_s=0.1)
        assert len(word) == 2

    def test_word_text_concatenates_phonemes(self):
        segs = _segs(("AH", 0.0, 0.2), ("EH", 0.2, 0.4))
        _, word = build_lyric_events(segs, word_gap_s=0.1)
        assert word[0].text == "AH-EH"

    def test_word_confidence_is_mean(self):
        segs = [
            PhonemeSegment("AH", 0.0, 0.2, confidence=0.8),
            PhonemeSegment("EH", 0.2, 0.4, confidence=0.4),
        ]
        _, word = build_lyric_events(segs, word_gap_s=0.1)
        assert word[0].confidence == pytest.approx(0.6)

    def test_word_idx_assigned(self):
        segs = _two_words()
        _, word = build_lyric_events(segs, word_gap_s=0.1)
        for i, we in enumerate(word):
            assert we.word_idx == i

    def test_word_phoneme_events_populated(self):
        segs = _three_segs()
        lyric, word = build_lyric_events(segs, word_gap_s=1.0)  # force single word
        assert len(word) == 1
        assert len(word[0].phoneme_events) == 3


# ---------------------------------------------------------------------------
# Back-annotation
# ---------------------------------------------------------------------------

class TestBackAnnotation:
    def test_word_idx_back_annotated_on_lyric(self):
        segs = _two_words()
        lyric, word = build_lyric_events(segs, word_gap_s=0.1)
        # lyric[0] and lyric[1] → word 0; lyric[2] → word 1
        assert lyric[0].word_idx == 0
        assert lyric[1].word_idx == 0
        assert lyric[2].word_idx == 1

    def test_lyric_events_reference_shared_objects(self):
        segs = _three_segs()
        lyric, word = build_lyric_events(segs, word_gap_s=1.0)
        assert all(le in word[0].phoneme_events for le in lyric)


# ---------------------------------------------------------------------------
# min_duration_s filtering
# ---------------------------------------------------------------------------

class TestMinDuration:
    def test_zero_duration_phoneme_excluded(self):
        segs = _segs(("AH", 0.0, 0.0), ("EH", 0.1, 0.4))
        lyric, _ = build_lyric_events(segs, min_duration_s=0.001)
        phonemes = [le.phoneme for le in lyric]
        assert "AH" not in phonemes
        assert "EH" in phonemes

    def test_empty_after_filtering(self):
        segs = _segs(("AH", 0.0, 0.0))  # zero-length
        lyric, word = build_lyric_events(segs, min_duration_s=0.001)
        assert lyric == []
        assert word == []


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_sorts_by_start_time(self):
        # Input out of order
        segs = _segs(("EH", 0.3, 0.6), ("AH", 0.0, 0.3))
        lyric, _ = build_lyric_events(segs)
        starts = [le.start_time for le in lyric]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# merge_word_events
# ---------------------------------------------------------------------------

class TestMergeWordEvents:
    def test_no_merge_when_gap_large(self):
        segs = _two_words()
        _, word = build_lyric_events(segs, word_gap_s=0.1)
        merged = merge_word_events(word, max_gap_s=0.01)
        assert len(merged) == len(word)

    def test_merge_when_gap_small(self):
        segs = _segs(("AH", 0.0, 0.2), ("EH", 0.22, 0.5))  # gap 20ms
        _, word = build_lyric_events(segs, word_gap_s=0.01)  # force two words
        merged = merge_word_events(word, max_gap_s=0.05)
        assert len(merged) == 1

    def test_empty_input(self):
        assert merge_word_events([]) == []

    def test_reindexed_after_merge(self):
        segs = _segs(("AH", 0.0, 0.2), ("EH", 0.22, 0.5))
        _, word = build_lyric_events(segs, word_gap_s=0.01)
        merged = merge_word_events(word, max_gap_s=0.05)
        for i, we in enumerate(merged):
            assert we.word_idx == i
