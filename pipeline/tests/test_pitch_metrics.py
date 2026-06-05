"""
tests/test_pitch_metrics.py - Unit tests for Phase 6 pitch metric functions.

All tests use synthetic AlignmentResult / FusedPerformanceRepresentation
objects — no real audio or model inference required.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_pitch_metrics.py -v
"""

from __future__ import annotations

import math
from typing import List, Optional

import pytest

from utils.types import (
    AlignmentResult,
    FusedPerformanceRepresentation,
    MetricBreakdown,
    NoteAlignmentMatch,
    NoteEvent,
    PitchMetrics,
    ReferenceNote,
    ReferencePerformanceRepresentation,
)
from metrics.pitch_metrics import (
    build_pitch_metrics,
    compute_mace,
    compute_note_pitch_accuracy,
    compute_pitch_accuracy,
    compute_pitch_rmse,
    compute_pitch_stability,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alignment(matches: List[NoteAlignmentMatch]) -> AlignmentResult:
    return AlignmentResult(
        predicted_audio_path="pred.wav",
        reference_source_path="ref.xml",
        note_matches=matches,
    )


def _match(
    pred_idx: int = 0,
    ref_idx: int = 0,
    pitch_dev: Optional[float] = None,
    onset_dev: float = 0.0,
    offset_dev: float = 0.0,
) -> NoteAlignmentMatch:
    return NoteAlignmentMatch(
        pred_idx=pred_idx,
        ref_idx=ref_idx,
        overlap_s=0.3,
        overlap_fraction=0.8,
        onset_deviation_s=onset_dev,
        offset_deviation_s=offset_dev,
        pitch_deviation_cents=pitch_dev,
    )


def _fused(stabilities: List[Optional[float]]) -> FusedPerformanceRepresentation:
    events = [
        NoteEvent(onset_time=float(i) * 0.5, offset_time=float(i) * 0.5 + 0.4,
                  pitch_stability=s)
        for i, s in enumerate(stabilities)
    ]
    return FusedPerformanceRepresentation(audio_path="pred.wav", duration_s=5.0,
                                          note_events=events)


def _reference(pitches_hz: List[Optional[float]]) -> ReferencePerformanceRepresentation:
    notes = []
    for i, hz in enumerate(pitches_hz):
        midi = 69.0 + 12.0 * math.log2(hz / 440.0) if hz else None
        notes.append(ReferenceNote(
            onset_time=float(i) * 0.5,
            offset_time=float(i) * 0.5 + 0.4,
            pitch_midi=midi,
            pitch_hz=hz,
            pitch_name=f"A{i}",
            note_idx=i,
        ))
    return ReferencePerformanceRepresentation(
        source_path="ref.xml", duration_s=5.0, notes=notes
    )


# ---------------------------------------------------------------------------
# compute_pitch_accuracy
# ---------------------------------------------------------------------------

class TestComputePitchAccuracy:
    def test_all_correct(self):
        al = _alignment([_match(pitch_dev=10.0), _match(pitch_dev=-20.0)])
        assert compute_pitch_accuracy(al, tolerance_cents=50.0) == pytest.approx(1.0)

    def test_none_correct(self):
        al = _alignment([_match(pitch_dev=100.0), _match(pitch_dev=-80.0)])
        assert compute_pitch_accuracy(al, tolerance_cents=50.0) == pytest.approx(0.0)

    def test_half_correct(self):
        al = _alignment([_match(pitch_dev=10.0), _match(pitch_dev=100.0)])
        assert compute_pitch_accuracy(al, tolerance_cents=50.0) == pytest.approx(0.5)

    def test_no_matches_returns_none(self):
        al = _alignment([])
        assert compute_pitch_accuracy(al) is None

    def test_no_pitch_deviation_returns_none(self):
        al = _alignment([_match(pitch_dev=None)])
        assert compute_pitch_accuracy(al) is None

    def test_boundary_exactly_at_tolerance(self):
        al = _alignment([_match(pitch_dev=50.0)])
        assert compute_pitch_accuracy(al, tolerance_cents=50.0) == pytest.approx(1.0)

    def test_boundary_just_over_tolerance(self):
        al = _alignment([_match(pitch_dev=50.01)])
        assert compute_pitch_accuracy(al, tolerance_cents=50.0) == pytest.approx(0.0)

    def test_negative_deviation_absolute(self):
        al = _alignment([_match(pitch_dev=-30.0)])
        assert compute_pitch_accuracy(al, tolerance_cents=50.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_pitch_rmse
# ---------------------------------------------------------------------------

class TestComputePitchRmse:
    def test_single_value(self):
        al = _alignment([_match(pitch_dev=30.0)])
        assert compute_pitch_rmse(al) == pytest.approx(30.0)

    def test_symmetric_values(self):
        al = _alignment([_match(pitch_dev=3.0), _match(pitch_dev=-3.0)])
        expected = math.sqrt((9 + 9) / 2)
        assert compute_pitch_rmse(al) == pytest.approx(expected)

    def test_zero_deviation(self):
        al = _alignment([_match(pitch_dev=0.0), _match(pitch_dev=0.0)])
        assert compute_pitch_rmse(al) == pytest.approx(0.0)

    def test_no_matches(self):
        assert compute_pitch_rmse(_alignment([])) is None

    def test_no_pitch_deviation(self):
        assert compute_pitch_rmse(_alignment([_match(pitch_dev=None)])) is None

    def test_rmse_nonnegative(self):
        al = _alignment([_match(pitch_dev=-50.0), _match(pitch_dev=20.0)])
        assert compute_pitch_rmse(al) >= 0

    def test_rmse_mixed_signs(self):
        devs = [10.0, -20.0, 30.0]
        al = _alignment([_match(pitch_dev=d) for d in devs])
        expected = math.sqrt(sum(d ** 2 for d in devs) / len(devs))
        assert compute_pitch_rmse(al) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_mace
# ---------------------------------------------------------------------------

class TestComputeMace:
    def test_single_positive(self):
        al = _alignment([_match(pitch_dev=40.0)])
        assert compute_mace(al) == pytest.approx(40.0)

    def test_single_negative(self):
        al = _alignment([_match(pitch_dev=-40.0)])
        assert compute_mace(al) == pytest.approx(40.0)

    def test_multiple(self):
        al = _alignment([_match(pitch_dev=10.0), _match(pitch_dev=-30.0)])
        assert compute_mace(al) == pytest.approx(20.0)

    def test_no_matches(self):
        assert compute_mace(_alignment([])) is None

    def test_zero(self):
        al = _alignment([_match(pitch_dev=0.0), _match(pitch_dev=0.0)])
        assert compute_mace(al) == pytest.approx(0.0)

    def test_mace_nonnegative(self):
        al = _alignment([_match(pitch_dev=-100.0), _match(pitch_dev=-200.0)])
        assert compute_mace(al) >= 0


# ---------------------------------------------------------------------------
# compute_pitch_stability
# ---------------------------------------------------------------------------

class TestComputePitchStability:
    def test_returns_mean(self):
        f = _fused([5.0, 10.0, 15.0])
        assert compute_pitch_stability(f) == pytest.approx(10.0)

    def test_all_none(self):
        f = _fused([None, None])
        assert compute_pitch_stability(f) is None

    def test_mixed_none(self):
        f = _fused([None, 6.0, None, 4.0])
        assert compute_pitch_stability(f) == pytest.approx(5.0)

    def test_single(self):
        f = _fused([7.5])
        assert compute_pitch_stability(f) == pytest.approx(7.5)

    def test_empty(self):
        f = _fused([])
        assert compute_pitch_stability(f) is None


# ---------------------------------------------------------------------------
# compute_note_pitch_accuracy
# ---------------------------------------------------------------------------

class TestComputeNotePitchAccuracy:
    def test_all_correct_returns_1(self):
        al = _alignment([_match(pred_idx=0, ref_idx=0, pitch_dev=10.0)])
        f = _fused([None])
        r = _reference([440.0])
        acc, bds = compute_note_pitch_accuracy(al, f, r, tolerance_cents=50.0)
        assert acc == pytest.approx(1.0)
        assert len(bds) == 1

    def test_breakdown_has_value(self):
        al = _alignment([_match(pred_idx=0, ref_idx=0, pitch_dev=25.0)])
        f = _fused([None])
        r = _reference([440.0])
        _, bds = compute_note_pitch_accuracy(al, f, r)
        assert bds[0].value == pytest.approx(25.0)

    def test_breakdown_label_set(self):
        al = _alignment([_match(pred_idx=0, ref_idx=0, pitch_dev=25.0)])
        f = _fused([None])
        r = _reference([440.0])
        _, bds = compute_note_pitch_accuracy(al, f, r)
        assert bds[0].label is not None

    def test_no_pitch_deviation_returns_none(self):
        al = _alignment([_match(pitch_dev=None)])
        f = _fused([None])
        r = _reference([440.0])
        acc, bds = compute_note_pitch_accuracy(al, f, r)
        assert acc is None
        assert bds == []


# ---------------------------------------------------------------------------
# build_pitch_metrics
# ---------------------------------------------------------------------------

class TestBuildPitchMetrics:
    def test_returns_pitch_metrics_type(self):
        al = _alignment([_match(pitch_dev=20.0)])
        result = build_pitch_metrics(al)
        assert isinstance(result, PitchMetrics)

    def test_accuracy_set(self):
        al = _alignment([_match(pitch_dev=20.0)])
        pm = build_pitch_metrics(al, tolerance_cents=50.0)
        assert pm.pitch_accuracy == pytest.approx(1.0)

    def test_rmse_set(self):
        al = _alignment([_match(pitch_dev=30.0)])
        pm = build_pitch_metrics(al)
        assert pm.pitch_rmse_cents == pytest.approx(30.0)

    def test_n_evaluated(self):
        al = _alignment([_match(pitch_dev=10.0), _match(pitch_dev=None)])
        pm = build_pitch_metrics(al)
        assert pm.n_evaluated == 1

    def test_per_note_populated(self):
        al = _alignment([_match(pitch_dev=10.0), _match(pitch_dev=20.0)])
        pm = build_pitch_metrics(al)
        assert len(pm.per_note) == 2

    def test_tolerance_cents_stored(self):
        al = _alignment([_match(pitch_dev=10.0)])
        pm = build_pitch_metrics(al, tolerance_cents=25.0)
        assert pm.tolerance_cents == pytest.approx(25.0)

    def test_empty_alignment(self):
        pm = build_pitch_metrics(_alignment([]))
        assert pm.pitch_accuracy is None
        assert pm.n_evaluated == 0
        assert pm.per_note == []

    def test_to_dict_serializable(self):
        al = _alignment([_match(pitch_dev=15.0)])
        pm = build_pitch_metrics(al)
        d = pm.to_dict()
        assert "pitch_accuracy" in d
        assert "mace_cents" in d
        assert "per_note" in d
