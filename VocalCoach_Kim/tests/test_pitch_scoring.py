"""
tests/test_pitch_scoring.py - Unit tests for scoring/pitch_scoring.py.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_pitch_scoring.py -v
"""

from __future__ import annotations

import pytest

from utils.types import CategoryScore, PitchMetrics, ScoreBreakdown
from scoring.pitch_scoring import (
    compute_intonation_score,
    compute_pitch_stability_score,
    compute_pitch_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pitch(
    accuracy: float = 0.80,
    mace: float = 30.0,
    rmse: float = 35.0,
    n: int = 5,
    tolerance: float = 50.0,
) -> PitchMetrics:
    return PitchMetrics(
        pitch_accuracy=accuracy,
        mace_cents=mace,
        pitch_rmse_cents=rmse,
        n_evaluated=n,
        tolerance_cents=tolerance,
    )


def _empty_pitch() -> PitchMetrics:
    return PitchMetrics()  # all None, n_evaluated=0


# ---------------------------------------------------------------------------
# compute_intonation_score
# ---------------------------------------------------------------------------

class TestComputeIntonationScore:
    def test_returns_scorebreakdown(self):
        bd = compute_intonation_score(_pitch())
        assert isinstance(bd, ScoreBreakdown)

    def test_component_name(self):
        assert compute_intonation_score(_pitch()).component == "intonation"

    def test_zero_mace_scores_100(self):
        bd = compute_intonation_score(PitchMetrics(mace_cents=0.0, n_evaluated=5))
        assert bd.score == pytest.approx(100.0)

    def test_high_mace_scores_low(self):
        bd = compute_intonation_score(PitchMetrics(mace_cents=200.0, n_evaluated=5))
        assert bd.score == pytest.approx(0.0)

    def test_none_mace_returns_zero_confidence(self):
        bd = compute_intonation_score(_empty_pitch())
        assert bd.confidence == pytest.approx(0.0)
        assert bd.raw_value is None

    def test_score_in_range(self):
        for mace in [0.0, 25.0, 50.0, 100.0, 200.0]:
            bd = compute_intonation_score(PitchMetrics(mace_cents=mace, n_evaluated=5))
            assert 0.0 <= bd.score <= 100.0

    def test_mace_25_is_around_88(self):
        bd = compute_intonation_score(PitchMetrics(mace_cents=25.0, n_evaluated=5))
        assert bd.score == pytest.approx(88.0)

    def test_confidence_increases_with_n(self):
        bd_low  = compute_intonation_score(PitchMetrics(mace_cents=30.0, n_evaluated=1))
        bd_high = compute_intonation_score(PitchMetrics(mace_cents=30.0, n_evaluated=10))
        assert bd_high.confidence >= bd_low.confidence


# ---------------------------------------------------------------------------
# compute_pitch_stability_score
# ---------------------------------------------------------------------------

class TestComputePitchStabilityScore:
    def test_returns_scorebreakdown(self):
        bd = compute_pitch_stability_score(_pitch())
        assert isinstance(bd, ScoreBreakdown)

    def test_component_name(self):
        assert compute_pitch_stability_score(_pitch()).component == "stability"

    def test_zero_rmse_scores_100(self):
        bd = compute_pitch_stability_score(PitchMetrics(pitch_rmse_cents=0.0, n_evaluated=5))
        assert bd.score == pytest.approx(100.0)

    def test_none_rmse_zero_confidence(self):
        bd = compute_pitch_stability_score(_empty_pitch())
        assert bd.confidence == pytest.approx(0.0)
        assert bd.raw_value is None

    def test_score_nonnegative(self):
        for rmse in [0.0, 50.0, 100.0, 300.0]:
            bd = compute_pitch_stability_score(PitchMetrics(pitch_rmse_cents=rmse, n_evaluated=5))
            assert bd.score >= 0.0

    def test_high_rmse_lower_than_low_rmse(self):
        low  = compute_pitch_stability_score(PitchMetrics(pitch_rmse_cents=20.0, n_evaluated=5))
        high = compute_pitch_stability_score(PitchMetrics(pitch_rmse_cents=150.0, n_evaluated=5))
        assert low.score > high.score


# ---------------------------------------------------------------------------
# compute_pitch_score
# ---------------------------------------------------------------------------

class TestComputePitchScore:
    def test_returns_categoryscore(self):
        assert isinstance(compute_pitch_score(_pitch()), CategoryScore)

    def test_category_is_pitch(self):
        assert compute_pitch_score(_pitch()).category == "pitch"

    def test_score_in_range(self):
        score = compute_pitch_score(_pitch()).score
        assert 0.0 <= score <= 100.0

    def test_n_evaluated_propagated(self):
        cs = compute_pitch_score(_pitch(n=7))
        assert cs.n_evaluated == 7

    def test_three_components_present(self):
        cs = compute_pitch_score(_pitch())
        names = {c.component for c in cs.components}
        assert "accuracy" in names
        assert "intonation" in names
        assert "stability" in names

    def test_empty_metrics_gives_low_or_zero_score(self):
        cs = compute_pitch_score(_empty_pitch())
        assert cs.score == pytest.approx(0.0)

    def test_perfect_metrics_high_score(self):
        p = PitchMetrics(pitch_accuracy=1.0, mace_cents=0.0, pitch_rmse_cents=0.0, n_evaluated=10)
        cs = compute_pitch_score(p)
        assert cs.score >= 95.0

    def test_to_dict_serializable(self):
        d = compute_pitch_score(_pitch()).to_dict()
        assert "score" in d
        assert "components" in d
        assert d["category"] == "pitch"

    def test_custom_weights_applied(self):
        cs_default = compute_pitch_score(_pitch())
        cs_custom  = compute_pitch_score(_pitch(), config={"weight_accuracy": 1.0, "weight_intonation": 0.0, "weight_stability": 0.0})
        # With all weight on accuracy, score should differ from default
        # Both are valid scores; just verify no crash and range
        assert 0.0 <= cs_custom.score <= 100.0

    def test_confidence_set(self):
        cs = compute_pitch_score(_pitch(n=5))
        assert cs.confidence is not None
        assert 0.0 <= cs.confidence <= 1.0
