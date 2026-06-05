"""
tests/test_reporting.py - Unit tests for metrics/reporting.py and
metrics/validation.py.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_reporting.py -v
"""

from __future__ import annotations

import pytest

from utils.types import (
    AlignmentResult,
    FusedPerformanceRepresentation,
    NoteAlignmentMatch,
    NoteEvent,
    PerformanceMetricsReport,
    PhonemeAlignmentMatch,
    ReferenceNote,
    ReferencePerformanceRepresentation,
    WordAlignmentMatch,
)
from metrics.reporting import build_metrics_report
from metrics.validation import MetricValidationReport, validate_metrics_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _note_match(pred_idx=0, ref_idx=0, onset_dev_s=0.0, offset_dev_s=0.0,
                pitch_dev=None) -> NoteAlignmentMatch:
    return NoteAlignmentMatch(
        pred_idx=pred_idx, ref_idx=ref_idx,
        overlap_s=0.3, overlap_fraction=0.8,
        onset_deviation_s=onset_dev_s,
        offset_deviation_s=offset_dev_s,
        pitch_deviation_cents=pitch_dev,
    )


def _phoneme_match(pred_idx=0, ref_idx=0, onset_dev_s=0.0,
                   label_match=False) -> PhonemeAlignmentMatch:
    return PhonemeAlignmentMatch(
        pred_idx=pred_idx, ref_idx=ref_idx,
        overlap_s=0.1, overlap_fraction=0.7,
        onset_deviation_s=onset_dev_s,
        label_match=label_match,
    )


def _word_match(pred_idx=0, ref_idx=0) -> WordAlignmentMatch:
    return WordAlignmentMatch(
        pred_idx=pred_idx, ref_idx=ref_idx,
        overlap_s=0.3, overlap_fraction=0.8, onset_deviation_s=0.02,
    )


def _fused(n_notes=2) -> FusedPerformanceRepresentation:
    events = [
        NoteEvent(onset_time=float(i) * 0.5, offset_time=float(i) * 0.5 + 0.4,
                  pitch_stability=3.0)
        for i in range(n_notes)
    ]
    return FusedPerformanceRepresentation(
        audio_path="p.wav", duration_s=5.0, note_events=events
    )


def _ref(n_notes=2) -> ReferencePerformanceRepresentation:
    notes = [
        ReferenceNote(onset_time=float(i) * 0.5, offset_time=float(i) * 0.5 + 0.4,
                      note_idx=i)
        for i in range(n_notes)
    ]
    return ReferencePerformanceRepresentation(
        source_path="r.xml", duration_s=5.0, notes=notes
    )


def _alignment_full() -> AlignmentResult:
    return AlignmentResult(
        predicted_audio_path="p.wav",
        reference_source_path="r.xml",
        note_matches=[
            _note_match(0, 0, onset_dev_s=0.02, pitch_dev=15.0),
            _note_match(1, 1, onset_dev_s=-0.01, pitch_dev=-30.0),
        ],
        phoneme_matches=[
            _phoneme_match(0, 0, onset_dev_s=0.01, label_match=True),
            _phoneme_match(1, 1, onset_dev_s=-0.02, label_match=False),
        ],
        word_matches=[_word_match(0, 0)],
    )


# ---------------------------------------------------------------------------
# build_metrics_report
# ---------------------------------------------------------------------------

class TestBuildMetricsReport:
    def test_returns_performance_metrics_report(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        assert isinstance(report, PerformanceMetricsReport)

    def test_audio_path_set(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        assert report.audio_path == "p.wav"

    def test_reference_path_set(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        assert report.reference_source_path == "r.xml"

    def test_n_note_matches(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        assert report.n_note_matches == 2

    def test_pitch_computed_when_matches_exist(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        assert report.pitch is not None

    def test_timing_computed_when_matches_exist(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        assert report.timing is not None

    def test_lyric_computed_when_phoneme_matches_exist(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        assert report.lyric is not None

    def test_duration_computed_with_fused_and_ref(self):
        al = _alignment_full()
        f = _fused(2)
        r = _ref(2)
        report = build_metrics_report(al, fused=f, reference=r)
        assert report.duration is not None

    def test_duration_none_without_fused(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        assert report.duration is None

    def test_empty_alignment_no_pitch(self):
        al = AlignmentResult(predicted_audio_path="p.wav", reference_source_path="r.xml")
        report = build_metrics_report(al)
        assert report.pitch is None
        assert report.timing is None
        assert report.lyric is None

    def test_note_precision_set(self):
        al = AlignmentResult(
            predicted_audio_path="p.wav", reference_source_path="r.xml",
            note_matches=[_note_match()],
            unmatched_pred_notes=[],
            unmatched_ref_notes=[],
        )
        report = build_metrics_report(al)
        assert report.note_precision is not None

    def test_to_dict_serializable(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        d = report.to_dict()
        assert "audio_path" in d
        assert "pitch" in d
        assert "timing" in d

    def test_computation_metadata_has_elapsed(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        assert "elapsed_s" in report.computation_metadata

    def test_config_pitch_tolerance_honored(self):
        al = AlignmentResult(
            predicted_audio_path="p.wav", reference_source_path="r.xml",
            note_matches=[_note_match(pitch_dev=60.0)],
        )
        # With 50¢ tolerance: pitch not correct
        r_strict = build_metrics_report(al, config={"pitch": {"cents_tolerance": 50.0}})
        # With 100¢ tolerance: pitch correct
        r_loose = build_metrics_report(al, config={"pitch": {"cents_tolerance": 100.0}})
        if r_strict.pitch and r_loose.pitch:
            assert r_strict.pitch.pitch_accuracy < r_loose.pitch.pitch_accuracy


# ---------------------------------------------------------------------------
# validate_metrics_report
# ---------------------------------------------------------------------------

class TestValidateMetricsReport:
    def test_valid_report_passes(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        vr = validate_metrics_report(report)
        assert isinstance(vr, MetricValidationReport)
        assert vr.n_errors == 0

    def test_empty_report_passes_with_warning(self):
        al = AlignmentResult(predicted_audio_path="p.wav", reference_source_path="r.xml")
        report = build_metrics_report(al)
        vr = validate_metrics_report(report)
        assert vr.n_errors == 0
        assert vr.n_warnings >= 1  # zero note matches warning

    def test_valid_is_true_when_no_errors(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        vr = validate_metrics_report(report)
        assert vr.valid is True

    def test_to_dict_includes_issues(self):
        al = AlignmentResult(predicted_audio_path="p.wav", reference_source_path="r.xml")
        report = build_metrics_report(al)
        vr = validate_metrics_report(report)
        d = vr.to_dict()
        assert "valid" in d
        assert "n_errors" in d
        assert "issues" in d

    def test_nan_precision_detected(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        report.note_precision = float("nan")
        vr = validate_metrics_report(report)
        assert vr.n_errors >= 1

    def test_out_of_range_precision_detected(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        report.note_precision = 1.5
        vr = validate_metrics_report(report)
        assert vr.n_errors >= 1

    def test_negative_rmse_detected(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        if report.pitch is not None:
            report.pitch.pitch_rmse_cents = -5.0
            vr = validate_metrics_report(report)
            assert vr.n_errors >= 1

    def test_out_of_range_timing_accuracy(self):
        al = _alignment_full()
        report = build_metrics_report(al)
        if report.timing is not None:
            report.timing.timing_accuracy = -0.1
            vr = validate_metrics_report(report)
            assert vr.n_errors >= 1

    def test_n_warnings_and_errors_counted(self):
        al = AlignmentResult(predicted_audio_path="p.wav", reference_source_path="r.xml")
        report = build_metrics_report(al)
        vr = validate_metrics_report(report)
        assert vr.n_warnings == len([i for i in vr.issues if i.severity == "warning"])
        assert vr.n_errors == len([i for i in vr.issues if i.severity == "error"])
