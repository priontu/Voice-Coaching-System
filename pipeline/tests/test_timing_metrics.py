"""
tests/test_timing_metrics.py - Unit tests for metrics/timing_metrics.py.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_timing_metrics.py -v
"""

from __future__ import annotations

import math
import pytest

from utils.types import (
    AlignmentResult,
    FusedPerformanceRepresentation,
    NoteAlignmentMatch,
    NoteEvent,
    ReferenceNote,
    ReferencePerformanceRepresentation,
    TimingMetrics,
)
from metrics.timing_metrics import (
    build_timing_metrics,
    compute_ioi_deviation,
    compute_offset_error,
    compute_onset_error,
    compute_timing_accuracy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _al(*onset_devs_ms, offset_devs_ms=None) -> AlignmentResult:
    matches = []
    for i, od in enumerate(onset_devs_ms):
        off = (offset_devs_ms[i] if offset_devs_ms else 0.0) / 1000.0
        matches.append(NoteAlignmentMatch(
            pred_idx=i, ref_idx=i,
            overlap_s=0.3, overlap_fraction=0.8,
            onset_deviation_s=od / 1000.0,
            offset_deviation_s=off,
        ))
    return AlignmentResult(
        predicted_audio_path="p.wav",
        reference_source_path="r.xml",
        note_matches=matches,
    )


def _fused_notes(onsets) -> list:
    return [NoteEvent(onset_time=o, offset_time=o + 0.4) for o in onsets]


def _ref_notes(onsets) -> list:
    return [ReferenceNote(onset_time=o, offset_time=o + 0.4) for o in onsets]


# ---------------------------------------------------------------------------
# compute_onset_error
# ---------------------------------------------------------------------------

class TestComputeOnsetError:
    def test_single_late(self):
        r = compute_onset_error(_al(30.0))
        assert r["mean_onset_error_ms"] == pytest.approx(30.0)
        assert r["mean_abs_onset_error_ms"] == pytest.approx(30.0)

    def test_single_early(self):
        r = compute_onset_error(_al(-20.0))
        assert r["mean_onset_error_ms"] == pytest.approx(-20.0)
        assert r["mean_abs_onset_error_ms"] == pytest.approx(20.0)

    def test_symmetric_zero_mean(self):
        r = compute_onset_error(_al(-30.0, 30.0))
        assert r["mean_onset_error_ms"] == pytest.approx(0.0)
        assert r["mean_abs_onset_error_ms"] == pytest.approx(30.0)

    def test_std_computed(self):
        r = compute_onset_error(_al(-10.0, 10.0))
        assert r["std_onset_error_ms"] == pytest.approx(10.0)

    def test_median_odd(self):
        r = compute_onset_error(_al(10.0, 20.0, 30.0))
        assert r["median_onset_error_ms"] == pytest.approx(20.0)

    def test_median_even(self):
        r = compute_onset_error(_al(10.0, 30.0))
        assert r["median_onset_error_ms"] == pytest.approx(20.0)

    def test_no_matches_returns_none(self):
        r = compute_onset_error(_al())
        assert r["mean_onset_error_ms"] is None

    def test_zero_error(self):
        r = compute_onset_error(_al(0.0))
        assert r["mean_onset_error_ms"] == pytest.approx(0.0)
        assert r["std_onset_error_ms"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_offset_error
# ---------------------------------------------------------------------------

class TestComputeOffsetError:
    def test_single_late(self):
        r = compute_offset_error(_al(0.0, offset_devs_ms=[40.0]))
        assert r["mean_offset_error_ms"] == pytest.approx(40.0)
        assert r["mean_abs_offset_error_ms"] == pytest.approx(40.0)

    def test_mixed_signs(self):
        r = compute_offset_error(_al(0.0, 0.0, offset_devs_ms=[-20.0, 40.0]))
        assert r["mean_offset_error_ms"] == pytest.approx(10.0)
        assert r["mean_abs_offset_error_ms"] == pytest.approx(30.0)

    def test_no_matches(self):
        r = compute_offset_error(_al())
        assert r["mean_offset_error_ms"] is None


# ---------------------------------------------------------------------------
# compute_timing_accuracy
# ---------------------------------------------------------------------------

class TestComputeTimingAccuracy:
    def test_all_within(self):
        acc = compute_timing_accuracy(_al(10.0, 20.0, -30.0), tolerance_ms=50.0)
        assert acc == pytest.approx(1.0)

    def test_none_within(self):
        acc = compute_timing_accuracy(_al(100.0, -200.0), tolerance_ms=50.0)
        assert acc == pytest.approx(0.0)

    def test_half(self):
        acc = compute_timing_accuracy(_al(10.0, 100.0), tolerance_ms=50.0)
        assert acc == pytest.approx(0.5)

    def test_no_matches_returns_none(self):
        assert compute_timing_accuracy(_al()) is None

    def test_boundary_at_tolerance(self):
        acc = compute_timing_accuracy(_al(50.0), tolerance_ms=50.0)
        assert acc == pytest.approx(1.0)

    def test_boundary_just_over(self):
        acc = compute_timing_accuracy(_al(50.01), tolerance_ms=50.0)
        assert acc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_ioi_deviation
# ---------------------------------------------------------------------------

class TestComputeIoiDeviation:
    def test_identical_ioi(self):
        pred = _fused_notes([0.0, 0.5, 1.0])
        ref = _ref_notes([0.0, 0.5, 1.0])
        assert compute_ioi_deviation(pred, ref) == pytest.approx(0.0)

    def test_constant_offset_ioi_still_zero(self):
        # IOI is the gap between successive onsets, not absolute timing.
        pred = _fused_notes([0.1, 0.6, 1.1])
        ref = _ref_notes([0.0, 0.5, 1.0])
        assert compute_ioi_deviation(pred, ref) == pytest.approx(0.0)

    def test_different_ioi(self):
        pred = _fused_notes([0.0, 0.6, 1.0])   # IOI = [600ms, 400ms]
        ref = _ref_notes([0.0, 0.5, 1.0])       # IOI = [500ms, 500ms]
        # MAE of [|600-500|, |400-500|] = [100ms, 100ms] → 100ms
        assert compute_ioi_deviation(pred, ref) == pytest.approx(100.0)

    def test_single_note_each(self):
        pred = _fused_notes([0.0])
        ref = _ref_notes([0.0])
        assert compute_ioi_deviation(pred, ref) is None

    def test_empty_both(self):
        assert compute_ioi_deviation([], []) is None


# ---------------------------------------------------------------------------
# build_timing_metrics
# ---------------------------------------------------------------------------

class TestBuildTimingMetrics:
    def test_returns_timing_metrics(self):
        tm = build_timing_metrics(_al(20.0, -10.0))
        assert isinstance(tm, TimingMetrics)

    def test_n_evaluated(self):
        tm = build_timing_metrics(_al(10.0, 20.0))
        assert tm.n_evaluated == 2

    def test_per_note_populated(self):
        tm = build_timing_metrics(_al(10.0, 20.0))
        assert len(tm.per_note) == 2

    def test_per_note_values_in_ms(self):
        tm = build_timing_metrics(_al(30.0))
        assert tm.per_note[0].value == pytest.approx(30.0)

    def test_tolerance_stored(self):
        tm = build_timing_metrics(_al(10.0), tolerance_ms=75.0)
        assert tm.tolerance_ms == pytest.approx(75.0)

    def test_timing_accuracy_computed(self):
        tm = build_timing_metrics(_al(10.0, 100.0), tolerance_ms=50.0)
        assert tm.timing_accuracy == pytest.approx(0.5)

    def test_empty_alignment_returns_none_fields(self):
        tm = build_timing_metrics(_al())
        assert tm.mean_onset_error_ms is None
        assert tm.timing_accuracy is None
        assert tm.n_evaluated == 0

    def test_to_dict_serializable(self):
        tm = build_timing_metrics(_al(15.0))
        d = tm.to_dict()
        assert "mean_onset_error_ms" in d
        assert "timing_accuracy" in d
        assert "per_note" in d
