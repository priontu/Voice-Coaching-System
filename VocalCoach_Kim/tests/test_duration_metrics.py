"""
tests/test_duration_metrics.py - Unit tests for metrics/duration_metrics.py.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_duration_metrics.py -v
"""

from __future__ import annotations

import math
import pytest

from utils.types import (
    AlignmentResult,
    DurationMetrics,
    FusedPerformanceRepresentation,
    NoteAlignmentMatch,
    NoteEvent,
    ReferenceNote,
    ReferencePerformanceRepresentation,
)
from metrics.duration_metrics import (
    build_duration_metrics,
    compute_duration_error,
    compute_duration_ratio,
    compute_relative_duration_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _al(pairs) -> AlignmentResult:
    """pairs: list of (pred_idx, ref_idx)"""
    matches = [
        NoteAlignmentMatch(
            pred_idx=p, ref_idx=r,
            overlap_s=0.3, overlap_fraction=0.8,
            onset_deviation_s=0.0, offset_deviation_s=0.0,
        )
        for p, r in pairs
    ]
    return AlignmentResult(
        predicted_audio_path="p.wav",
        reference_source_path="r.xml",
        note_matches=matches,
    )


def _fused(durations) -> FusedPerformanceRepresentation:
    events = [
        NoteEvent(onset_time=float(i) * 0.5,
                  offset_time=float(i) * 0.5 + d)
        for i, d in enumerate(durations)
    ]
    return FusedPerformanceRepresentation(
        audio_path="p.wav", duration_s=10.0, note_events=events
    )


def _ref(durations) -> ReferencePerformanceRepresentation:
    notes = [
        ReferenceNote(
            onset_time=float(i) * 0.5,
            offset_time=float(i) * 0.5 + d,
            note_idx=i,
        )
        for i, d in enumerate(durations)
    ]
    return ReferencePerformanceRepresentation(
        source_path="r.xml", duration_s=10.0, notes=notes
    )


# ---------------------------------------------------------------------------
# compute_duration_error
# ---------------------------------------------------------------------------

class TestComputeDurationError:
    def test_perfect_match(self):
        al = _al([(0, 0)])
        f = _fused([0.5])
        r = _ref([0.5])
        err = compute_duration_error(al, f.note_events, r.notes)
        assert err["mean_duration_error_s"] == pytest.approx(0.0)
        assert err["mean_abs_duration_error_s"] == pytest.approx(0.0)

    def test_too_long(self):
        al = _al([(0, 0)])
        f = _fused([0.7])
        r = _ref([0.5])
        err = compute_duration_error(al, f.note_events, r.notes)
        assert err["mean_duration_error_s"] == pytest.approx(0.2, abs=1e-6)

    def test_too_short(self):
        al = _al([(0, 0)])
        f = _fused([0.3])
        r = _ref([0.5])
        err = compute_duration_error(al, f.note_events, r.notes)
        assert err["mean_duration_error_s"] == pytest.approx(-0.2, abs=1e-6)

    def test_mean_abs_nonnegative(self):
        al = _al([(0, 0), (1, 1)])
        f = _fused([0.3, 0.7])
        r = _ref([0.5, 0.5])
        err = compute_duration_error(al, f.note_events, r.notes)
        assert err["mean_abs_duration_error_s"] >= 0.0

    def test_std_computed(self):
        al = _al([(0, 0), (1, 1)])
        f = _fused([0.6, 0.6])    # both +0.1 error → std=0
        r = _ref([0.5, 0.5])
        err = compute_duration_error(al, f.note_events, r.notes)
        assert err["std_duration_error_s"] == pytest.approx(0.0)

    def test_no_matches(self):
        err = compute_duration_error(_al([]), [], [])
        assert err["mean_duration_error_s"] is None

    def test_index_out_of_range_skipped(self):
        al = _al([(99, 0)])
        f = _fused([0.5])
        r = _ref([0.5])
        err = compute_duration_error(al, f.note_events, r.notes)
        assert err["mean_duration_error_s"] is None


# ---------------------------------------------------------------------------
# compute_duration_ratio
# ---------------------------------------------------------------------------

class TestComputeDurationRatio:
    def test_perfect_ratio(self):
        al = _al([(0, 0)])
        f = _fused([0.5])
        r = _ref([0.5])
        assert compute_duration_ratio(al, f.note_events, r.notes) == pytest.approx(1.0)

    def test_double_duration(self):
        al = _al([(0, 0)])
        f = _fused([1.0])
        r = _ref([0.5])
        assert compute_duration_ratio(al, f.note_events, r.notes) == pytest.approx(2.0)

    def test_half_duration(self):
        al = _al([(0, 0)])
        f = _fused([0.25])
        r = _ref([0.5])
        assert compute_duration_ratio(al, f.note_events, r.notes) == pytest.approx(0.5)

    def test_no_matches(self):
        assert compute_duration_ratio(_al([]), [], []) is None

    def test_mean_of_ratios(self):
        al = _al([(0, 0), (1, 1)])
        f = _fused([1.0, 0.5])    # ratios 2.0 and 1.0 → mean 1.5
        r = _ref([0.5, 0.5])
        assert compute_duration_ratio(al, f.note_events, r.notes) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# compute_relative_duration_error
# ---------------------------------------------------------------------------

class TestComputeRelativeDurationError:
    def test_perfect(self):
        al = _al([(0, 0)])
        f = _fused([0.5])
        r = _ref([0.5])
        assert compute_relative_duration_error(al, f.note_events, r.notes) == pytest.approx(0.0)

    def test_twenty_percent_error(self):
        al = _al([(0, 0)])
        f = _fused([0.6])
        r = _ref([0.5])
        assert compute_relative_duration_error(al, f.note_events, r.notes) == pytest.approx(0.2)

    def test_nonnegative(self):
        al = _al([(0, 0)])
        f = _fused([0.3])
        r = _ref([0.5])
        assert compute_relative_duration_error(al, f.note_events, r.notes) >= 0.0

    def test_no_matches(self):
        assert compute_relative_duration_error(_al([]), [], []) is None


# ---------------------------------------------------------------------------
# build_duration_metrics
# ---------------------------------------------------------------------------

class TestBuildDurationMetrics:
    def test_returns_duration_metrics(self):
        al = _al([(0, 0)])
        f = _fused([0.5])
        r = _ref([0.5])
        dm = build_duration_metrics(al, fused=f, reference=r)
        assert isinstance(dm, DurationMetrics)

    def test_per_note_populated(self):
        al = _al([(0, 0), (1, 1)])
        f = _fused([0.5, 0.6])
        r = _ref([0.5, 0.5])
        dm = build_duration_metrics(al, fused=f, reference=r)
        assert len(dm.per_note) == 2

    def test_per_note_value_is_error(self):
        al = _al([(0, 0)])
        f = _fused([0.7])
        r = _ref([0.5])
        dm = build_duration_metrics(al, fused=f, reference=r)
        assert dm.per_note[0].value == pytest.approx(0.2, abs=1e-6)

    def test_n_evaluated(self):
        al = _al([(0, 0), (1, 1)])
        f = _fused([0.5, 0.5])
        r = _ref([0.5, 0.5])
        dm = build_duration_metrics(al, fused=f, reference=r)
        assert dm.n_evaluated == 2

    def test_no_fused_empty(self):
        al = _al([(0, 0)])
        dm = build_duration_metrics(al)
        assert dm.mean_duration_error_s is None
        assert dm.n_evaluated == 0

    def test_to_dict_serializable(self):
        al = _al([(0, 0)])
        f = _fused([0.5])
        r = _ref([0.5])
        dm = build_duration_metrics(al, fused=f, reference=r)
        d = dm.to_dict()
        assert "mean_duration_error_s" in d
        assert "mean_duration_ratio" in d
        assert "per_note" in d
