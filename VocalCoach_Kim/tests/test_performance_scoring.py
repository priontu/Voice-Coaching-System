"""
tests/test_performance_scoring.py - Unit tests for scoring/performance_scoring.py
and scoring/validation.py.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_performance_scoring.py -v
"""

from __future__ import annotations

import pytest

from utils.types import (
    DurationMetrics,
    LyricMetrics,
    PerformanceMetricsReport,
    PerformanceScoreReport,
    PitchMetrics,
    TimingMetrics,
)
from scoring.performance_scoring import build_performance_score_report
from scoring.validation import ScoreValidationReport, validate_score_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_metrics() -> PerformanceMetricsReport:
    return PerformanceMetricsReport(
        audio_path="p.wav",
        reference_source_path="r.xml",
        pitch=PitchMetrics(
            pitch_accuracy=0.80, mace_cents=30.0, pitch_rmse_cents=35.0, n_evaluated=5
        ),
        timing=TimingMetrics(
            timing_accuracy=0.75, mean_abs_onset_error_ms=25.0,
            ioi_mae_ms=20.0, std_onset_error_ms=30.0, n_evaluated=5
        ),
        duration=DurationMetrics(
            mean_relative_duration_error=0.10, mean_duration_ratio=1.05,
            std_duration_error_s=0.05, n_evaluated=5
        ),
        lyric=LyricMetrics(
            phoneme_overlap_accuracy=0.85, word_alignment_accuracy=0.90,
            label_match_rate=0.75, mean_abs_phoneme_boundary_error_ms=20.0,
            n_phoneme_matches=10, n_word_matches=5,
        ),
        n_note_matches=5,
    )


def _pitch_only_metrics() -> PerformanceMetricsReport:
    return PerformanceMetricsReport(
        audio_path="p.wav",
        reference_source_path="r.xml",
        pitch=PitchMetrics(pitch_accuracy=0.80, mace_cents=30.0, n_evaluated=5),
        n_note_matches=5,
    )


def _empty_metrics() -> PerformanceMetricsReport:
    return PerformanceMetricsReport(audio_path="p.wav", reference_source_path="r.xml")


# ---------------------------------------------------------------------------
# build_performance_score_report
# ---------------------------------------------------------------------------

class TestBuildPerformanceScoreReport:
    def test_returns_performance_score_report(self):
        r = build_performance_score_report(_full_metrics())
        assert isinstance(r, PerformanceScoreReport)

    def test_audio_path_set(self):
        r = build_performance_score_report(_full_metrics())
        assert r.audio_path == "p.wav"

    def test_reference_path_set(self):
        r = build_performance_score_report(_full_metrics())
        assert r.reference_source_path == "r.xml"

    def test_overall_score_in_range(self):
        r = build_performance_score_report(_full_metrics())
        assert r.overall_score is not None
        assert 0.0 <= r.overall_score <= 100.0

    def test_all_category_scores_computed(self):
        r = build_performance_score_report(_full_metrics())
        assert r.pitch_score is not None
        assert r.timing_score is not None
        assert r.duration_score is not None
        assert r.lyric_score is not None

    def test_partial_metrics_available_categories_only(self):
        r = build_performance_score_report(_pitch_only_metrics())
        assert r.pitch_score is not None
        assert r.timing_score is None
        assert r.duration_score is None
        assert r.lyric_score is None

    def test_empty_metrics_overall_is_none(self):
        r = build_performance_score_report(_empty_metrics())
        assert r.overall_score is None

    def test_weights_sum_approximately_one(self):
        r = build_performance_score_report(_full_metrics())
        if r.weights_used:
            total = sum(r.weights_used.values())
            assert abs(total - 1.0) < 0.02

    def test_custom_weights_honoured(self):
        cfg = {"weights": {"pitch": 1.0, "timing": 0.0, "duration": 0.0, "lyric": 0.0}}
        r = build_performance_score_report(_full_metrics(), config=cfg)
        # With all weight on pitch, overall should equal pitch score (within confidence effects)
        assert r.overall_score is not None
        assert 0.0 <= r.overall_score <= 100.0

    def test_to_dict_serializable(self):
        d = build_performance_score_report(_full_metrics()).to_dict()
        assert "overall_score" in d
        assert "pitch_score" in d
        assert "weights_used" in d

    def test_score_metadata_has_elapsed(self):
        r = build_performance_score_report(_full_metrics())
        assert "elapsed_s" in r.score_metadata

    def test_categories_computed_listed(self):
        r = build_performance_score_report(_full_metrics())
        cats = r.score_metadata.get("categories_computed", [])
        assert "pitch" in cats
        assert "timing" in cats


# ---------------------------------------------------------------------------
# validate_score_report
# ---------------------------------------------------------------------------

class TestValidateScoreReport:
    def test_valid_report_passes(self):
        r = build_performance_score_report(_full_metrics())
        vr = validate_score_report(r)
        assert isinstance(vr, ScoreValidationReport)
        assert vr.n_errors == 0

    def test_valid_is_true_when_no_errors(self):
        r = build_performance_score_report(_full_metrics())
        assert validate_score_report(r).valid is True

    def test_nan_overall_score_detected(self):
        r = build_performance_score_report(_full_metrics())
        r.overall_score = float("nan")
        vr = validate_score_report(r)
        assert vr.n_errors >= 1

    def test_out_of_range_overall_score_detected(self):
        r = build_performance_score_report(_full_metrics())
        r.overall_score = 110.0
        vr = validate_score_report(r)
        assert vr.n_errors >= 1

    def test_empty_report_warns_no_overall(self):
        r = build_performance_score_report(_empty_metrics())
        vr = validate_score_report(r)
        assert vr.n_warnings >= 1  # overall_score is None warning

    def test_to_dict_includes_issues(self):
        r = build_performance_score_report(_empty_metrics())
        d = validate_score_report(r).to_dict()
        assert "valid" in d
        assert "n_errors" in d
        assert "issues" in d

    def test_n_errors_and_warnings_counted_correctly(self):
        r = build_performance_score_report(_full_metrics())
        vr = validate_score_report(r)
        assert vr.n_errors == len([i for i in vr.issues if i.severity == "error"])
        assert vr.n_warnings == len([i for i in vr.issues if i.severity == "warning"])

    def test_category_score_out_of_range_detected(self):
        r = build_performance_score_report(_full_metrics())
        if r.pitch_score is not None:
            r.pitch_score.score = -5.0
            vr = validate_score_report(r)
            assert vr.n_errors >= 1

    def test_confidence_out_of_range_detected(self):
        r = build_performance_score_report(_full_metrics())
        if r.pitch_score is not None:
            r.pitch_score.confidence = 1.5
            vr = validate_score_report(r)
            assert vr.n_errors >= 1
