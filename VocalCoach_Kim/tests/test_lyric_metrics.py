"""
tests/test_lyric_metrics.py - Unit tests for metrics/lyric_metrics.py.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_lyric_metrics.py -v
"""

from __future__ import annotations

import pytest

from utils.types import (
    AlignmentResult,
    LyricMetrics,
    PhonemeAlignmentMatch,
    WordAlignmentMatch,
)
from metrics.lyric_metrics import (
    build_lyric_metrics,
    compute_lyric_timing_accuracy,
    compute_phoneme_boundary_error,
    compute_word_alignment_accuracy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _al(
    phoneme_devs_ms=None,
    label_matches=None,
    overlap_fractions=None,
    n_unmatched_ref_words: int = 0,
    word_devs_ms=None,
) -> AlignmentResult:
    phoneme_devs_ms = phoneme_devs_ms or []
    label_matches = label_matches or [False] * len(phoneme_devs_ms)
    overlap_fractions = overlap_fractions or [0.8] * len(phoneme_devs_ms)
    word_devs_ms = word_devs_ms or []

    phoneme_matches = [
        PhonemeAlignmentMatch(
            pred_idx=i, ref_idx=i,
            overlap_s=0.2 * overlap_fractions[i],
            overlap_fraction=overlap_fractions[i],
            onset_deviation_s=d / 1000.0,
            label_match=label_matches[i],
        )
        for i, d in enumerate(phoneme_devs_ms)
    ]
    word_matches = [
        WordAlignmentMatch(
            pred_idx=i, ref_idx=i,
            overlap_s=0.3, overlap_fraction=0.8,
            onset_deviation_s=d / 1000.0,
        )
        for i, d in enumerate(word_devs_ms)
    ]
    unmatched_ref_words = list(range(n_unmatched_ref_words))
    return AlignmentResult(
        predicted_audio_path="p.wav",
        reference_source_path="r.xml",
        phoneme_matches=phoneme_matches,
        word_matches=word_matches,
        unmatched_ref_words=unmatched_ref_words,
    )


# ---------------------------------------------------------------------------
# compute_phoneme_boundary_error
# ---------------------------------------------------------------------------

class TestComputePhonemeBoundaryError:
    def test_single_late(self):
        r = compute_phoneme_boundary_error(_al([20.0]))
        assert r["mean_phoneme_boundary_error_ms"] == pytest.approx(20.0)
        assert r["mean_abs_phoneme_boundary_error_ms"] == pytest.approx(20.0)

    def test_single_early(self):
        r = compute_phoneme_boundary_error(_al([-15.0]))
        assert r["mean_phoneme_boundary_error_ms"] == pytest.approx(-15.0)
        assert r["mean_abs_phoneme_boundary_error_ms"] == pytest.approx(15.0)

    def test_zero_error(self):
        r = compute_phoneme_boundary_error(_al([0.0]))
        assert r["mean_phoneme_boundary_error_ms"] == pytest.approx(0.0)
        assert r["std_phoneme_boundary_error_ms"] == pytest.approx(0.0)

    def test_symmetric_zero_mean(self):
        r = compute_phoneme_boundary_error(_al([-10.0, 10.0]))
        assert r["mean_phoneme_boundary_error_ms"] == pytest.approx(0.0)
        assert r["mean_abs_phoneme_boundary_error_ms"] == pytest.approx(10.0)

    def test_std_computed(self):
        r = compute_phoneme_boundary_error(_al([0.0, 0.0]))
        assert r["std_phoneme_boundary_error_ms"] == pytest.approx(0.0)

    def test_no_matches_returns_none(self):
        r = compute_phoneme_boundary_error(_al([]))
        assert r["mean_phoneme_boundary_error_ms"] is None

    def test_abs_error_nonnegative(self):
        r = compute_phoneme_boundary_error(_al([-50.0, -30.0]))
        assert r["mean_abs_phoneme_boundary_error_ms"] >= 0.0


# ---------------------------------------------------------------------------
# compute_word_alignment_accuracy
# ---------------------------------------------------------------------------

class TestComputeWordAlignmentAccuracy:
    def test_all_matched(self):
        al = _al(word_devs_ms=[0.0, 0.0])
        assert compute_word_alignment_accuracy(al) == pytest.approx(1.0)

    def test_half_matched(self):
        al = _al(word_devs_ms=[0.0], n_unmatched_ref_words=1)
        assert compute_word_alignment_accuracy(al) == pytest.approx(0.5)

    def test_none_matched(self):
        al = _al(n_unmatched_ref_words=3)
        assert compute_word_alignment_accuracy(al) == pytest.approx(0.0)

    def test_no_words_returns_none(self):
        al = _al()
        assert compute_word_alignment_accuracy(al) is None

    def test_accuracy_in_unit_interval(self):
        al = _al(word_devs_ms=[0.0, 0.0], n_unmatched_ref_words=2)
        acc = compute_word_alignment_accuracy(al)
        assert acc is not None
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# compute_lyric_timing_accuracy
# ---------------------------------------------------------------------------

class TestComputeLyricTimingAccuracy:
    def test_all_within(self):
        al = _al([5.0, 10.0, -15.0])
        acc = compute_lyric_timing_accuracy(al, tolerance_ms=30.0)
        assert acc == pytest.approx(1.0)

    def test_none_within(self):
        al = _al([50.0, -60.0])
        acc = compute_lyric_timing_accuracy(al, tolerance_ms=30.0)
        assert acc == pytest.approx(0.0)

    def test_half_within(self):
        al = _al([10.0, 100.0])
        acc = compute_lyric_timing_accuracy(al, tolerance_ms=30.0)
        assert acc == pytest.approx(0.5)

    def test_no_matches_returns_none(self):
        assert compute_lyric_timing_accuracy(_al([])) is None

    def test_boundary_at_tolerance(self):
        al = _al([30.0])
        assert compute_lyric_timing_accuracy(al, tolerance_ms=30.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_lyric_metrics
# ---------------------------------------------------------------------------

class TestBuildLyricMetrics:
    def test_returns_lyric_metrics(self):
        al = _al([10.0, -5.0])
        lm = build_lyric_metrics(al)
        assert isinstance(lm, LyricMetrics)

    def test_n_phoneme_matches(self):
        al = _al([10.0, 20.0])
        lm = build_lyric_metrics(al)
        assert lm.n_phoneme_matches == 2

    def test_per_phoneme_populated(self):
        al = _al([10.0, 20.0])
        lm = build_lyric_metrics(al)
        assert len(lm.per_phoneme) == 2

    def test_per_phoneme_value_in_ms(self):
        al = _al([25.0])
        lm = build_lyric_metrics(al)
        assert lm.per_phoneme[0].value == pytest.approx(25.0)

    def test_label_match_rate_all_correct(self):
        al = _al([5.0, 10.0], label_matches=[True, True])
        lm = build_lyric_metrics(al)
        assert lm.label_match_rate == pytest.approx(1.0)

    def test_label_match_rate_none_correct(self):
        al = _al([5.0, 10.0], label_matches=[False, False])
        lm = build_lyric_metrics(al)
        assert lm.label_match_rate == pytest.approx(0.0)

    def test_overlap_accuracy(self):
        al = _al([5.0, 5.0], overlap_fractions=[0.8, 0.3])
        lm = build_lyric_metrics(al)
        assert lm.phoneme_overlap_accuracy == pytest.approx(0.5)

    def test_word_alignment_accuracy(self):
        al = _al(word_devs_ms=[0.0], n_unmatched_ref_words=1)
        lm = build_lyric_metrics(al)
        assert lm.word_alignment_accuracy == pytest.approx(0.5)

    def test_no_data_returns_none_fields(self):
        lm = build_lyric_metrics(_al([]))
        assert lm.mean_phoneme_boundary_error_ms is None
        assert lm.n_phoneme_matches == 0

    def test_to_dict_serializable(self):
        al = _al([15.0])
        lm = build_lyric_metrics(al)
        d = lm.to_dict()
        assert "mean_phoneme_boundary_error_ms" in d
        assert "label_match_rate" in d
        assert "per_phoneme" in d
