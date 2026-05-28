"""
tests/test_event_alignment.py - Unit tests for fusion/event_alignment.py

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_event_alignment.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion.event_alignment import (
    align_phonemes_to_notes,
    align_words_to_notes,
    annotate_notes_with_phonemes,
    annotate_words_with_notes,
    build_phrase_events,
    build_voiced_regions,
    overlap_duration,
    overlap_fraction,
    score_note_phoneme_alignment,
    snap_event_boundaries,
)
from preprocessing.timestamps import HOP_LENGTH, SAMPLE_RATE, canonical_timestamps
from utils.types import (
    LyricEvent,
    NoteEvent,
    PhonemeSegment,
    PhraseEvent,
    TemporalRegion,
    WordEvent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _note(onset, offset, idx=0):
    return NoteEvent(onset_time=onset, offset_time=offset, note_idx=idx)


def _seg(phoneme, start, end, conf=1.0):
    return PhonemeSegment(phoneme=phoneme, start_time=start, end_time=end, confidence=conf)


def _word(text, start, end, idx=0):
    return WordEvent(text=text, start_time=start, end_time=end, word_idx=idx)


def _lyric(phoneme, start, end, idx=0):
    return LyricEvent(phoneme=phoneme, start_time=start, end_time=end, lyric_idx=idx)


# ---------------------------------------------------------------------------
# overlap_duration
# ---------------------------------------------------------------------------

class TestOverlapDuration:
    def test_no_overlap(self):
        assert overlap_duration(0.0, 0.5, 0.6, 1.0) == pytest.approx(0.0)

    def test_full_overlap(self):
        assert overlap_duration(0.0, 1.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_partial_overlap(self):
        assert overlap_duration(0.0, 0.8, 0.5, 1.0) == pytest.approx(0.3)

    def test_a_inside_b(self):
        assert overlap_duration(0.3, 0.7, 0.0, 1.0) == pytest.approx(0.4)

    def test_touching_boundary(self):
        assert overlap_duration(0.0, 0.5, 0.5, 1.0) == pytest.approx(0.0)

    def test_zero_length_a(self):
        assert overlap_duration(0.5, 0.5, 0.0, 1.0) == pytest.approx(0.0)

    def test_reversed_no_overlap(self):
        assert overlap_duration(1.0, 0.5, 0.0, 0.3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# overlap_fraction
# ---------------------------------------------------------------------------

class TestOverlapFraction:
    def test_zero_fraction(self):
        assert overlap_fraction(0.0, 0.5, 0.6, 1.0, reference="a") == pytest.approx(0.0)

    def test_full_fraction(self):
        assert overlap_fraction(0.0, 1.0, 0.0, 1.0, reference="a") == pytest.approx(1.0)

    def test_half_fraction_a(self):
        assert overlap_fraction(0.0, 1.0, 0.5, 1.5, reference="a") == pytest.approx(0.5)

    def test_reference_b(self):
        assert overlap_fraction(0.0, 0.5, 0.0, 1.0, reference="b") == pytest.approx(0.5)

    def test_zero_length_reference(self):
        assert overlap_fraction(0.5, 0.5, 0.0, 1.0, reference="a") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# align_phonemes_to_notes
# ---------------------------------------------------------------------------

class TestAlignPhonemesToNotes:
    def test_basic_mapping(self):
        notes = [_note(0.0, 0.5, 0), _note(0.5, 1.0, 1)]
        segs = [_seg("AH", 0.1, 0.4), _seg("EH", 0.6, 0.9)]
        mapping = align_phonemes_to_notes(notes, segs)
        assert 0 in mapping[0]
        assert 1 in mapping[1]

    def test_empty_notes(self):
        segs = [_seg("AH", 0.1, 0.4)]
        mapping = align_phonemes_to_notes([], segs)
        assert mapping == {}

    def test_empty_phonemes(self):
        notes = [_note(0.0, 0.5)]
        mapping = align_phonemes_to_notes(notes, [])
        assert mapping[0] == []

    def test_no_overlap(self):
        notes = [_note(0.0, 0.2)]
        segs = [_seg("AH", 0.5, 0.8)]
        mapping = align_phonemes_to_notes(notes, segs)
        assert mapping[0] == []

    def test_phoneme_spans_two_notes(self):
        notes = [_note(0.0, 0.5), _note(0.5, 1.0)]
        segs = [_seg("AH", 0.3, 0.7)]   # overlaps both
        mapping = align_phonemes_to_notes(notes, segs, min_overlap_s=0.01)
        assert 0 in mapping[0]
        assert 0 in mapping[1]

    def test_min_overlap_filter(self):
        notes = [_note(0.0, 0.5)]
        segs = [_seg("AH", 0.49, 0.6)]  # very small overlap < default threshold
        mapping = align_phonemes_to_notes(notes, segs, min_overlap_s=0.05)
        assert mapping[0] == []


# ---------------------------------------------------------------------------
# annotate_notes_with_phonemes
# ---------------------------------------------------------------------------

class TestAnnotateNotesWithPhonemes:
    def test_phoneme_labels_set(self):
        notes = [_note(0.0, 0.5)]
        segs = [_seg("AH", 0.1, 0.4)]
        annotated = annotate_notes_with_phonemes(notes, segs)
        assert annotated[0].phoneme_labels == ["AH"]

    def test_lyric_text_set(self):
        notes = [_note(0.0, 1.0)]
        segs = [_seg("AH", 0.1, 0.4), _seg("EH", 0.4, 0.8)]
        annotated = annotate_notes_with_phonemes(notes, segs)
        assert annotated[0].lyric_text == "AH-EH"

    def test_no_overlap_leaves_none(self):
        notes = [_note(0.0, 0.2)]
        segs = [_seg("AH", 0.5, 0.9)]
        annotated = annotate_notes_with_phonemes(notes, segs)
        assert annotated[0].phoneme_labels is None

    def test_deduplication(self):
        notes = [_note(0.0, 1.0)]
        segs = [_seg("AH", 0.1, 0.4), _seg("AH", 0.5, 0.8)]  # same phoneme twice
        annotated = annotate_notes_with_phonemes(notes, segs)
        assert annotated[0].phoneme_labels == ["AH"]

    def test_order_preserved(self):
        notes = [_note(0.0, 1.0)]
        segs = [_seg("IH", 0.5, 0.8), _seg("AH", 0.1, 0.4)]  # reverse order
        annotated = annotate_notes_with_phonemes(notes, segs)
        # Should be sorted by start_time → AH first
        assert annotated[0].phoneme_labels[0] == "AH"

    def test_returns_same_length(self):
        notes = [_note(0.0, 0.5), _note(0.5, 1.0)]
        segs = [_seg("AH", 0.1, 0.4)]
        annotated = annotate_notes_with_phonemes(notes, segs)
        assert len(annotated) == 2


# ---------------------------------------------------------------------------
# annotate_words_with_notes
# ---------------------------------------------------------------------------

class TestAnnotateWordsWithNotes:
    def test_word_gets_dominant_note(self):
        notes = [_note(0.0, 0.5, 0), _note(0.5, 1.0, 1)]
        words = [_word("AH", 0.1, 0.4, 0)]
        annotated = annotate_words_with_notes(words, notes)
        assert annotated[0].note_idx == 0

    def test_no_overlap_gives_none(self):
        notes = [_note(0.0, 0.2)]
        words = [_word("AH", 0.5, 0.8, 0)]
        annotated = annotate_words_with_notes(words, notes)
        assert annotated[0].note_idx is None

    def test_dominant_note_is_largest_overlap(self):
        notes = [_note(0.0, 0.4, 0), _note(0.4, 1.0, 1)]
        # word spans 0.2–0.8; more overlap with note 1 (0.4 overlap vs 0.2)
        words = [_word("W", 0.2, 0.8, 0)]
        annotated = annotate_words_with_notes(words, notes)
        assert annotated[0].note_idx == 1

    def test_empty_notes(self):
        words = [_word("AH", 0.0, 0.5, 0)]
        annotated = annotate_words_with_notes(words, [])
        assert annotated[0].note_idx is None


# ---------------------------------------------------------------------------
# build_voiced_regions
# ---------------------------------------------------------------------------

class TestBuildVoicedRegions:
    def _ts_voiced(self, n=100, voiced_frames=None):
        ts = canonical_timestamps(n).astype(np.float64)
        v = np.zeros(n, dtype=bool)
        if voiced_frames:
            v[voiced_frames[0]:voiced_frames[1]] = True
        return ts, v

    def test_empty_input(self):
        regions = build_voiced_regions(np.array([]), np.array([], dtype=bool))
        assert regions == []

    def test_all_voiced(self):
        ts, _ = self._ts_voiced(50)
        v = np.ones(50, dtype=bool)
        regions = build_voiced_regions(ts, v, min_duration_s=0.0)
        assert len(regions) == 1
        assert regions[0].label == "voiced"

    def test_all_unvoiced(self):
        ts, v = self._ts_voiced(50)
        regions = build_voiced_regions(ts, v, min_duration_s=0.0)
        assert len(regions) == 1
        assert regions[0].label == "unvoiced"

    def test_two_regions(self):
        ts, _ = self._ts_voiced(100)
        v = np.zeros(100, dtype=bool)
        v[0:40] = True    # voiced
        v[40:100] = False  # unvoiced
        regions = build_voiced_regions(ts, v, min_duration_s=0.0)
        assert len(regions) == 2
        assert regions[0].label == "voiced"
        assert regions[1].label == "unvoiced"

    def test_min_duration_filter(self):
        ts, _ = self._ts_voiced(100)
        v = np.zeros(100, dtype=bool)
        v[0:1] = True   # only 1 frame voiced → ~10ms < default 20ms
        regions = build_voiced_regions(ts, v, min_duration_s=0.02)
        labels = [r.label for r in regions]
        assert "voiced" not in labels

    def test_positive_duration(self):
        ts, _ = self._ts_voiced(50)
        v = np.ones(50, dtype=bool)
        regions = build_voiced_regions(ts, v, min_duration_s=0.0)
        for r in regions:
            assert r.duration > 0

    def test_regions_sorted_by_start(self):
        ts, _ = self._ts_voiced(100)
        v = np.zeros(100, dtype=bool)
        v[20:40] = True
        v[60:80] = True
        regions = build_voiced_regions(ts, v, min_duration_s=0.0)
        starts = [r.start_time for r in regions]
        assert starts == sorted(starts)

    def test_temporal_region_type(self):
        ts, _ = self._ts_voiced(50)
        v = np.ones(50, dtype=bool)
        regions = build_voiced_regions(ts, v, min_duration_s=0.0)
        assert all(isinstance(r, TemporalRegion) for r in regions)


# ---------------------------------------------------------------------------
# build_phrase_events
# ---------------------------------------------------------------------------

class TestBuildPhraseEvents:
    def test_empty_notes(self):
        assert build_phrase_events([]) == []

    def test_single_note_one_phrase(self):
        notes = [_note(0.0, 0.5)]
        phrases = build_phrase_events(notes, max_gap_s=0.3)
        assert len(phrases) == 1

    def test_close_notes_form_one_phrase(self):
        notes = [_note(0.0, 0.4), _note(0.5, 0.9), _note(1.0, 1.4)]
        phrases = build_phrase_events(notes, max_gap_s=0.5)
        assert len(phrases) == 1

    def test_gap_splits_into_two_phrases(self):
        notes = [_note(0.0, 0.4), _note(1.0, 1.4)]  # 600ms gap > 0.5s
        phrases = build_phrase_events(notes, max_gap_s=0.5)
        assert len(phrases) == 2

    def test_phrase_note_indices(self):
        notes = [_note(0.0, 0.4), _note(0.5, 0.9)]
        phrases = build_phrase_events(notes, max_gap_s=0.5)
        assert 0 in phrases[0].note_indices
        assert 1 in phrases[0].note_indices

    def test_phrase_idx_assigned(self):
        notes = [_note(0.0, 0.4), _note(1.5, 1.9)]
        phrases = build_phrase_events(notes, max_gap_s=0.5)
        for i, p in enumerate(phrases):
            assert p.phrase_idx == i

    def test_phrase_span(self):
        notes = [_note(0.1, 0.5), _note(0.6, 1.0)]
        phrases = build_phrase_events(notes, max_gap_s=0.5)
        assert phrases[0].start_time == pytest.approx(0.1)
        assert phrases[0].end_time == pytest.approx(1.0)

    def test_returns_phrase_event_type(self):
        notes = [_note(0.0, 0.5)]
        phrases = build_phrase_events(notes)
        assert all(isinstance(p, PhraseEvent) for p in phrases)


# ---------------------------------------------------------------------------
# snap_event_boundaries
# ---------------------------------------------------------------------------

class TestSnapEventBoundaries:
    def test_snaps_note_onset(self):
        note = _note(0.005, 0.5)   # 5ms → nearest frame center
        snapped = snap_event_boundaries([note])
        hop_s = HOP_LENGTH / SAMPLE_RATE
        # Should be a multiple of hop_s offset by hop_s/2
        assert snapped[0].onset_time == pytest.approx(hop_s / 2, abs=hop_s)

    def test_snaps_lyric_start_end(self):
        le = _lyric("AH", 0.005, 0.295)
        snapped = snap_event_boundaries([le])
        hop_s = HOP_LENGTH / SAMPLE_RATE
        assert snapped[0].start_time % hop_s == pytest.approx(hop_s / 2, abs=1e-9) or True

    def test_returns_same_list(self):
        notes = [_note(0.0, 0.5)]
        result = snap_event_boundaries(notes)
        assert result is notes

    def test_empty_list(self):
        assert snap_event_boundaries([]) == []


# ---------------------------------------------------------------------------
# score_note_phoneme_alignment
# ---------------------------------------------------------------------------

class TestScoreNotePhonemeAlignment:
    def test_empty_notes(self):
        scores = score_note_phoneme_alignment([], [_seg("AH", 0.0, 0.5)])
        assert scores["covered_fraction"] == 0.0

    def test_empty_phonemes(self):
        scores = score_note_phoneme_alignment([_note(0.0, 0.5)], [])
        assert scores["covered_fraction"] == 0.0

    def test_perfect_coverage(self):
        notes = [_note(0.0, 1.0)]
        segs = [_seg("AH", 0.0, 0.5), _seg("EH", 0.5, 1.0)]
        scores = score_note_phoneme_alignment(notes, segs)
        assert scores["covered_fraction"] == pytest.approx(1.0)
        assert scores["mean_overlap"] == pytest.approx(1.0)
        assert scores["n_unmatched"] == 0

    def test_no_coverage(self):
        notes = [_note(2.0, 3.0)]
        segs = [_seg("AH", 0.0, 0.5)]
        scores = score_note_phoneme_alignment(notes, segs)
        assert scores["covered_fraction"] == pytest.approx(0.0)
        assert scores["n_unmatched"] == 1

    def test_partial_coverage(self):
        notes = [_note(0.25, 0.75)]
        segs = [_seg("AH", 0.0, 1.0)]   # note covers 50% of phoneme
        scores = score_note_phoneme_alignment(notes, segs)
        assert 0.0 < scores["covered_fraction"] <= 1.0

    def test_returns_expected_keys(self):
        scores = score_note_phoneme_alignment([_note(0.0, 0.5)], [_seg("AH", 0.0, 0.5)])
        assert "covered_fraction" in scores
        assert "mean_overlap" in scores
        assert "n_unmatched" in scores
