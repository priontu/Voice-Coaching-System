"""
tests/test_reference_alignment.py - Unit tests for alignment/ modules.

Tests cover:
  alignment_utils  - overlap, deviation, nearest-match, pitch utilities
  reference_alignment - align_notes, align_phonemes, align_words, align_performance

All tests use synthetic Python objects; no model weights or files required.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_reference_alignment.py -v
"""

from __future__ import annotations

import math

import pytest

from alignment.alignment_utils import (
    iou,
    nearest_match,
    offset_deviation,
    onset_deviation,
    overlap_duration,
    overlap_fraction_of_a,
    pitch_deviation_cents,
    pitch_deviation_semitones,
)
from alignment.reference_alignment import (
    align_notes,
    align_performance,
    align_phonemes,
    align_words,
)
from utils.types import (
    AlignmentResult,
    FusedPerformanceRepresentation,
    LyricEvent,
    NoteAlignmentMatch,
    NoteEvent,
    PhonemeAlignmentMatch,
    ReferenceNote,
    ReferencePerformanceRepresentation,
    ReferencePhoneme,
    ReferenceWord,
    WordAlignmentMatch,
    WordEvent,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _note(onset, offset, pitch_hz=None, pitch_midi=None, note_idx=None):
    n = NoteEvent(
        onset_time=onset,
        offset_time=offset,
        duration=offset - onset,
        pitch_hz=pitch_hz,
        pitch_midi=pitch_midi,
        note_idx=note_idx,
    )
    return n


def _ref_note(onset, offset, pitch_midi=None, pitch_hz=None, is_rest=False, note_idx=None):
    from utils.types import ReferenceNote
    return ReferenceNote(
        onset_time=onset,
        offset_time=offset,
        pitch_midi=pitch_midi,
        pitch_hz=pitch_hz,
        is_rest=is_rest,
        note_idx=note_idx,
    )


def _lyric(start, end, phoneme="AH"):
    return LyricEvent(phoneme=phoneme, start_time=start, end_time=end)


def _ref_phoneme(start, end, phoneme="AH"):
    return ReferencePhoneme(phoneme=phoneme, start_time=start, end_time=end)


def _word(start, end, text="HELLO"):
    return WordEvent(text=text, start_time=start, end_time=end)


def _ref_word(start, end, text="HELLO"):
    return ReferenceWord(text=text, start_time=start, end_time=end)


# ---------------------------------------------------------------------------
# TestOverlapDuration
# ---------------------------------------------------------------------------

class TestOverlapDuration:
    def test_no_overlap(self):
        assert overlap_duration(0.0, 1.0, 1.5, 2.0) == pytest.approx(0.0)

    def test_full_overlap(self):
        assert overlap_duration(0.0, 1.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_partial_overlap(self):
        assert overlap_duration(0.0, 1.0, 0.5, 1.5) == pytest.approx(0.5)

    def test_contained(self):
        assert overlap_duration(0.0, 2.0, 0.5, 1.5) == pytest.approx(1.0)

    def test_adjacent(self):
        assert overlap_duration(0.0, 1.0, 1.0, 2.0) == pytest.approx(0.0)

    def test_negative_args_clipped(self):
        assert overlap_duration(1.0, 0.0, 0.5, 1.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestOverlapFractionOfA
# ---------------------------------------------------------------------------

class TestOverlapFractionOfA:
    def test_zero_a_duration(self):
        assert overlap_fraction_of_a(1.0, 1.0, 0.5, 1.5) == pytest.approx(0.0)

    def test_full_coverage(self):
        assert overlap_fraction_of_a(0.0, 1.0, 0.0, 2.0) == pytest.approx(1.0)

    def test_half_coverage(self):
        assert overlap_fraction_of_a(0.0, 1.0, 0.5, 2.0) == pytest.approx(0.5)

    def test_no_coverage(self):
        assert overlap_fraction_of_a(0.0, 1.0, 2.0, 3.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestIOU
# ---------------------------------------------------------------------------

class TestIOU:
    def test_identical_intervals(self):
        assert iou(0.0, 1.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert iou(0.0, 1.0, 2.0, 3.0) == pytest.approx(0.0)

    def test_half_overlap(self):
        # inter=0.5, union=1.5
        assert iou(0.0, 1.0, 0.5, 1.5) == pytest.approx(0.5 / 1.5)

    def test_contained(self):
        # small [0.25, 0.75] inside [0.0, 1.0]
        # inter=0.5, union=1.0
        assert iou(0.0, 1.0, 0.25, 0.75) == pytest.approx(0.5 / 1.0)


# ---------------------------------------------------------------------------
# TestDeviations
# ---------------------------------------------------------------------------

class TestDeviations:
    def test_onset_positive_when_late(self):
        assert onset_deviation(1.1, 1.0) == pytest.approx(0.1)

    def test_onset_negative_when_early(self):
        assert onset_deviation(0.9, 1.0) == pytest.approx(-0.1)

    def test_onset_zero(self):
        assert onset_deviation(1.0, 1.0) == pytest.approx(0.0)

    def test_offset_deviation(self):
        assert offset_deviation(2.1, 2.0) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# TestNearestMatch
# ---------------------------------------------------------------------------

class TestNearestMatch:
    def test_empty_returns_minus_one(self):
        idx, dist = nearest_match(0.5, [])
        assert idx == -1
        assert dist == float("inf")

    def test_exact_match(self):
        idx, dist = nearest_match(1.0, [0.0, 1.0, 2.0])
        assert idx == 1
        assert dist == pytest.approx(0.0)

    def test_nearest_left(self):
        idx, dist = nearest_match(0.4, [0.0, 1.0])
        assert idx == 0

    def test_nearest_right(self):
        idx, dist = nearest_match(0.6, [0.0, 1.0])
        assert idx == 1

    def test_single_candidate(self):
        idx, dist = nearest_match(5.0, [3.0])
        assert idx == 0
        assert dist == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TestPitchUtilities
# ---------------------------------------------------------------------------

class TestPitchUtilities:
    def test_cents_same_pitch(self):
        assert pitch_deviation_cents(440.0, 440.0) == pytest.approx(0.0)

    def test_cents_octave_up(self):
        assert pitch_deviation_cents(880.0, 440.0) == pytest.approx(1200.0)

    def test_cents_semitone_up(self):
        # One semitone = 100 cents
        assert pitch_deviation_cents(440.0 * 2 ** (1 / 12), 440.0) == pytest.approx(100.0, abs=0.01)

    def test_semitones_octave(self):
        assert pitch_deviation_semitones(880.0, 440.0) == pytest.approx(12.0)

    def test_none_when_input_none(self):
        assert pitch_deviation_cents(None, 440.0) is None
        assert pitch_deviation_cents(440.0, None) is None

    def test_none_when_zero_hz(self):
        assert pitch_deviation_cents(0.0, 440.0) is None


# ---------------------------------------------------------------------------
# TestAlignNotes
# ---------------------------------------------------------------------------

class TestAlignNotes:
    def test_empty_predicted(self):
        matches, up, ur = align_notes([], [_ref_note(0.0, 0.5)])
        assert matches == []
        assert up == []

    def test_empty_reference(self):
        matches, up, ur = align_notes([_note(0.0, 0.5)], [])
        assert matches == []
        assert up == [0]

    def test_single_match(self):
        matches, up, ur = align_notes(
            [_note(0.0, 0.5, note_idx=0)],
            [_ref_note(0.0, 0.5, note_idx=0)],
        )
        assert len(matches) == 1
        assert matches[0].pred_idx == 0
        assert matches[0].ref_idx == 0

    def test_returns_note_alignment_match(self):
        matches, _, _ = align_notes(
            [_note(0.0, 0.5)], [_ref_note(0.0, 0.5)]
        )
        assert isinstance(matches[0], NoteAlignmentMatch)

    def test_onset_deviation_in_match(self):
        matches, _, _ = align_notes(
            [_note(0.1, 0.6)], [_ref_note(0.0, 0.5)]
        )
        assert matches[0].onset_deviation_s == pytest.approx(0.1, abs=0.001)

    def test_rests_not_matched(self):
        matches, up, ur = align_notes(
            [_note(0.0, 0.5)],
            [_ref_note(0.0, 0.5, is_rest=True)],
        )
        assert matches == []

    def test_no_overlap_no_match(self):
        matches, up, ur = align_notes(
            [_note(0.0, 0.3)],
            [_ref_note(1.0, 1.5)],
            min_overlap_s=0.01,
        )
        assert matches == []
        assert up == [0]

    def test_one_to_one_matching(self):
        # 3 predicted, 3 reference — should produce 3 matches
        pred = [_note(i * 1.0, i * 1.0 + 0.8, note_idx=i) for i in range(3)]
        ref = [_ref_note(i * 1.0, i * 1.0 + 0.8, note_idx=i) for i in range(3)]
        matches, up, ur = align_notes(pred, ref)
        assert len(matches) == 3
        assert up == []
        assert ur == []

    def test_pitch_deviation_cents_populated(self):
        matches, _, _ = align_notes(
            [_note(0.0, 0.5, pitch_hz=440.0)],
            [_ref_note(0.0, 0.5, pitch_hz=880.0)],
        )
        assert matches[0].pitch_deviation_cents == pytest.approx(-1200.0, abs=1.0)

    def test_pitch_deviation_none_when_no_pitch(self):
        matches, _, _ = align_notes(
            [_note(0.0, 0.5)], [_ref_note(0.0, 0.5)]
        )
        assert matches[0].pitch_deviation_cents is None


# ---------------------------------------------------------------------------
# TestAlignPhonemes
# ---------------------------------------------------------------------------

class TestAlignPhonemes:
    def test_empty_predicted(self):
        matches, up, ur = align_phonemes([], [_ref_phoneme(0.0, 0.3)])
        assert matches == []

    def test_single_match(self):
        matches, up, ur = align_phonemes(
            [_lyric(0.0, 0.3)], [_ref_phoneme(0.0, 0.3)]
        )
        assert len(matches) == 1

    def test_label_match_flag(self):
        matches, _, _ = align_phonemes(
            [_lyric(0.0, 0.3, "AH")], [_ref_phoneme(0.0, 0.3, "AH")]
        )
        assert matches[0].label_match is True

    def test_label_mismatch_flag(self):
        matches, _, _ = align_phonemes(
            [_lyric(0.0, 0.3, "AH")], [_ref_phoneme(0.0, 0.3, "EH")]
        )
        assert matches[0].label_match is False

    def test_returns_phoneme_alignment_match(self):
        matches, _, _ = align_phonemes(
            [_lyric(0.0, 0.3)], [_ref_phoneme(0.0, 0.3)]
        )
        assert isinstance(matches[0], PhonemeAlignmentMatch)

    def test_no_overlap_no_match(self):
        matches, up, ur = align_phonemes(
            [_lyric(0.0, 0.1)], [_ref_phoneme(0.5, 0.8)]
        )
        assert matches == []


# ---------------------------------------------------------------------------
# TestAlignWords
# ---------------------------------------------------------------------------

class TestAlignWords:
    def test_empty_inputs(self):
        matches, up, ur = align_words([], [])
        assert matches == []

    def test_single_match(self):
        matches, up, ur = align_words(
            [_word(0.0, 0.5)], [_ref_word(0.0, 0.5)]
        )
        assert len(matches) == 1

    def test_returns_word_match(self):
        matches, _, _ = align_words(
            [_word(0.0, 0.5)], [_ref_word(0.0, 0.5)]
        )
        assert isinstance(matches[0], WordAlignmentMatch)

    def test_unmatched_when_no_overlap(self):
        matches, up, ur = align_words(
            [_word(0.0, 0.3)], [_ref_word(1.0, 1.5)]
        )
        assert matches == []
        assert 0 in up


# ---------------------------------------------------------------------------
# TestAlignPerformance
# ---------------------------------------------------------------------------

def _make_fused(notes=None, lyrics=None, words=None):
    return FusedPerformanceRepresentation(
        audio_path="test.wav",
        duration_s=5.0,
        note_events=notes or [],
        lyric_events=lyrics or [],
        word_events=words or [],
    )


def _make_reference(notes=None, phonemes=None, words=None):
    return ReferencePerformanceRepresentation(
        source_path="test.xml",
        duration_s=5.0,
        notes=notes or [],
        phonemes=phonemes or [],
        words=words or [],
    )


class TestAlignPerformance:
    def test_returns_alignment_result(self):
        fused = _make_fused()
        ref = _make_reference()
        result = align_performance(fused, ref)
        assert isinstance(result, AlignmentResult)

    def test_audio_path_set(self):
        fused = _make_fused()
        ref = _make_reference()
        result = align_performance(fused, ref)
        assert result.predicted_audio_path == "test.wav"

    def test_reference_path_set(self):
        fused = _make_fused()
        ref = _make_reference()
        result = align_performance(fused, ref)
        assert result.reference_source_path == "test.xml"

    def test_empty_inputs_no_matches(self):
        result = align_performance(_make_fused(), _make_reference())
        assert result.note_matches == []
        assert result.phoneme_matches == []
        assert result.word_matches == []

    def test_note_precision_none_when_no_notes(self):
        result = align_performance(_make_fused(), _make_reference())
        assert result.note_precision is None

    def test_matches_detected(self):
        fused = _make_fused(
            notes=[_note(0.0, 0.5, note_idx=0)],
        )
        ref = _make_reference(
            notes=[_ref_note(0.0, 0.5, note_idx=0)],
        )
        result = align_performance(fused, ref)
        assert len(result.note_matches) == 1

    def test_note_precision_and_recall(self):
        fused = _make_fused(
            notes=[_note(0.0, 0.5, note_idx=0), _note(1.0, 1.5, note_idx=1)]
        )
        ref = _make_reference(
            notes=[_ref_note(0.0, 0.5, note_idx=0)]
        )
        result = align_performance(fused, ref)
        # 1 match out of 2 predicted → precision=0.5; 1 match out of 1 ref → recall=1.0
        assert result.note_precision == pytest.approx(0.5)
        assert result.note_recall == pytest.approx(1.0)

    def test_alignment_metadata_populated(self):
        result = align_performance(_make_fused(), _make_reference())
        assert "note" in result.alignment_metadata
        assert "phoneme" in result.alignment_metadata
        assert "word" in result.alignment_metadata

    def test_custom_config_applied(self):
        fused = _make_fused(notes=[_note(0.0, 0.5)])
        ref = _make_reference(notes=[_ref_note(1.0, 1.5)])
        # With a very large max onset deviation, notes that don't overlap are still excluded
        result = align_performance(
            fused, ref, config={"min_overlap_s": 0.01, "max_onset_deviation_s": 10.0}
        )
        # No temporal overlap → no match regardless of onset deviation
        assert result.note_matches == []

    def test_deterministic(self):
        fused = _make_fused(notes=[_note(i * 0.5, i * 0.5 + 0.4, note_idx=i) for i in range(5)])
        ref = _make_reference(notes=[_ref_note(i * 0.5, i * 0.5 + 0.4, note_idx=i) for i in range(5)])
        r1 = align_performance(fused, ref)
        r2 = align_performance(fused, ref)
        assert len(r1.note_matches) == len(r2.note_matches)
        for m1, m2 in zip(r1.note_matches, r2.note_matches):
            assert m1.pred_idx == m2.pred_idx
            assert m1.ref_idx == m2.ref_idx

    def test_to_dict_serializable(self):
        fused = _make_fused(notes=[_note(0.0, 0.5, note_idx=0)])
        ref = _make_reference(notes=[_ref_note(0.0, 0.5, note_idx=0)])
        result = align_performance(fused, ref)
        d = result.to_dict()
        assert "note_matches" in d
        assert "alignment_metadata" in d
