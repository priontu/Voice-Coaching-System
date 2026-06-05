"""
tests/test_timing_scoring.py - Unit tests for scoring/timing_scoring.py.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_timing_scoring.py -v
"""

from __future__ import annotations

import pytest

from utils.types import CategoryScore, ScoreBreakdown, TimingMetrics
from scoring.timing_scoring import (
    compute_rhythm_stability_score,
    compute_timing_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timing(
    timing_accuracy: float = 0.80,
    mean_abs: float = 20.0,
    ioi_mae: float = 15.0,
    std: float = 25.0,
    n: int = 5,
    tolerance: float = 50.0,
) -> TimingMetrics:
    return TimingMetrics(
        timing_accuracy=timing_accuracy,
        mean_abs_onset_error_ms=mean_abs,
        ioi_mae_ms=ioi_mae,
        std_onset_error_ms=std,
        n_evaluated=n,
        tolerance_ms=tolerance,
    )


def _empty_timing() -> TimingMetrics:
    return TimingMetrics()


# ---------------------------------------------------------------------------
# compute_rhythm_stability_score
# ---------------------------------------------------------------------------

class TestComputeRhythmStabilityScore:
    def test_returns_scorebreakdown(self):
        bd = compute_rhythm_stability_score(_timing())
        assert isinstance(bd, ScoreBreakdown)

    def test_component_name(self):
        assert compute_rhythm_stability_score(_timing()).component == "rhythm_stability"

    def test_zero_ioi_and_std_scores_100(self):
        t = TimingMetrics(ioi_mae_ms=0.0, std_onset_error_ms=0.0, n_evaluated=5)
        bd = compute_rhythm_stability_score(t)
        assert bd.score == pytest.approx(100.0)

    def test_high_ioi_scores_low(self):
        t = TimingMetrics(ioi_mae_ms=300.0, std_onset_error_ms=300.0, n_evaluated=5)
        bd = compute_rhythm_stability_score(t)
        assert bd.score == pytest.approx(0.0)

    def test_none_data_returns_zero_confidence(self):
        bd = compute_rhythm_stability_score(_empty_timing())
        assert bd.confidence == pytest.approx(0.0)
        assert bd.raw_value is None

    def test_only_ioi_available(self):
        t = TimingMetrics(ioi_mae_ms=30.0, n_evaluated=5)
        bd = compute_rhythm_stability_score(t)
        assert bd.score > 0.0

    def test_only_std_available(self):
        t = TimingMetrics(std_onset_error_ms=30.0, n_evaluated=5)
        bd = compute_rhythm_stability_score(t)
        assert bd.score > 0.0

    def test_score_in_range(self):
        bd = compute_rhythm_stability_score(_timing())
        assert 0.0 <= bd.score <= 100.0


# ---------------------------------------------------------------------------
# compute_timing_score
# ---------------------------------------------------------------------------

class TestComputeTimingScore:
    def test_returns_categoryscore(self):
        assert isinstance(compute_timing_score(_timing()), CategoryScore)

    def test_category_is_timing(self):
        assert compute_timing_score(_timing()).category == "timing"

    def test_score_in_range(self):
        score = compute_timing_score(_timing()).score
        assert 0.0 <= score <= 100.0

    def test_n_evaluated_propagated(self):
        cs = compute_timing_score(_timing(n=8))
        assert cs.n_evaluated == 8

    def test_three_components_present(self):
        cs = compute_timing_score(_timing())
        names = {c.component for c in cs.components}
        assert "accuracy" in names
        assert "onset_mae" in names
        assert "rhythm_stability" in names

    def test_empty_metrics_zero_score(self):
        cs = compute_timing_score(_empty_timing())
        assert cs.score == pytest.approx(0.0)

    def test_perfect_metrics_high_score(self):
        t = TimingMetrics(
            timing_accuracy=1.0,
            mean_abs_onset_error_ms=0.0,
            ioi_mae_ms=0.0,
            std_onset_error_ms=0.0,
            n_evaluated=10,
        )
        cs = compute_timing_score(t)
        assert cs.score >= 95.0

    def test_to_dict_serializable(self):
        d = compute_timing_score(_timing()).to_dict()
        assert "score" in d
        assert d["category"] == "timing"

    def test_confidence_set(self):
        cs = compute_timing_score(_timing(n=5))
        assert cs.confidence is not None
        assert 0.0 <= cs.confidence <= 1.0

    def test_worse_timing_lower_score(self):
        good = compute_timing_score(_timing(timing_accuracy=0.9, mean_abs=10.0))
        poor = compute_timing_score(_timing(timing_accuracy=0.2, mean_abs=150.0))
        assert good.score > poor.score
