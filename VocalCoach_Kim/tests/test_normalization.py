"""
tests/test_normalization.py - Unit tests for scoring/normalization.py.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_normalization.py -v
"""

from __future__ import annotations

import math

import pytest

from scoring.normalization import (
    bounded_score,
    gaussian_penalty,
    normalize_metric,
    piecewise_score,
)


# ---------------------------------------------------------------------------
# bounded_score
# ---------------------------------------------------------------------------

class TestBoundedScore:
    def test_at_lower_returns_100(self):
        assert bounded_score(0.0, lower=0.0, upper=100.0) == pytest.approx(100.0)

    def test_at_upper_returns_0(self):
        assert bounded_score(100.0, lower=0.0, upper=100.0) == pytest.approx(0.0)

    def test_midpoint(self):
        assert bounded_score(50.0, lower=0.0, upper=100.0) == pytest.approx(50.0)

    def test_below_lower_clamped_to_100(self):
        assert bounded_score(-10.0, lower=0.0, upper=100.0) == pytest.approx(100.0)

    def test_above_upper_clamped_to_0(self):
        assert bounded_score(150.0, lower=0.0, upper=100.0) == pytest.approx(0.0)

    def test_equal_bounds_at_lower(self):
        assert bounded_score(5.0, lower=5.0, upper=5.0) == pytest.approx(100.0)

    def test_equal_bounds_above_lower(self):
        assert bounded_score(10.0, lower=5.0, upper=5.0) == pytest.approx(0.0)

    def test_result_in_unit_range(self):
        for v in [0.0, 25.0, 50.0, 75.0, 100.0]:
            s = bounded_score(v, lower=0.0, upper=100.0)
            assert 0.0 <= s <= 100.0


# ---------------------------------------------------------------------------
# gaussian_penalty
# ---------------------------------------------------------------------------

class TestGaussianPenalty:
    def test_perfect_zero_returns_100(self):
        assert gaussian_penalty(0.0, sigma=50.0) == pytest.approx(100.0)

    def test_one_sigma(self):
        expected = 100.0 * math.exp(-0.5)
        assert gaussian_penalty(50.0, sigma=50.0) == pytest.approx(expected, rel=1e-6)

    def test_two_sigma(self):
        expected = 100.0 * math.exp(-2.0)
        assert gaussian_penalty(100.0, sigma=50.0) == pytest.approx(expected, rel=1e-6)

    def test_zero_sigma_at_zero_returns_100(self):
        assert gaussian_penalty(0.0, sigma=0.0) == pytest.approx(100.0)

    def test_zero_sigma_nonzero_value_returns_0(self):
        assert gaussian_penalty(1.0, sigma=0.0) == pytest.approx(0.0)

    def test_result_nonnegative(self):
        for v in [0.0, 10.0, 100.0, 500.0]:
            assert gaussian_penalty(v, sigma=50.0) >= 0.0

    def test_result_at_most_100(self):
        assert gaussian_penalty(0.0, sigma=100.0) <= 100.0

    def test_monotonically_decreasing(self):
        s1 = gaussian_penalty(10.0, sigma=50.0)
        s2 = gaussian_penalty(50.0, sigma=50.0)
        s3 = gaussian_penalty(100.0, sigma=50.0)
        assert s1 > s2 > s3


# ---------------------------------------------------------------------------
# piecewise_score
# ---------------------------------------------------------------------------

_BP = [(0.0, 100.0), (50.0, 75.0), (100.0, 50.0), (200.0, 0.0)]


class TestPiecewiseScore:
    def test_at_first_breakpoint(self):
        assert piecewise_score(0.0, _BP) == pytest.approx(100.0)

    def test_at_last_breakpoint(self):
        assert piecewise_score(200.0, _BP) == pytest.approx(0.0)

    def test_at_intermediate_breakpoint(self):
        assert piecewise_score(50.0, _BP) == pytest.approx(75.0)

    def test_below_first_clamped(self):
        assert piecewise_score(-10.0, _BP) == pytest.approx(100.0)

    def test_above_last_clamped(self):
        assert piecewise_score(300.0, _BP) == pytest.approx(0.0)

    def test_interpolated_midpoint(self):
        # between (0,100) and (50,75): midpoint x=25 → y=87.5
        assert piecewise_score(25.0, _BP) == pytest.approx(87.5)

    def test_empty_breakpoints_returns_zero(self):
        assert piecewise_score(50.0, []) == pytest.approx(0.0)

    def test_single_breakpoint(self):
        assert piecewise_score(99.0, [(0.0, 80.0)]) == pytest.approx(80.0)

    def test_result_in_range(self):
        for x in [0.0, 25.0, 50.0, 100.0, 200.0, 250.0]:
            s = piecewise_score(x, _BP)
            assert 0.0 <= s <= 100.0


# ---------------------------------------------------------------------------
# normalize_metric
# ---------------------------------------------------------------------------

class TestNormalizeMetric:
    def test_bounded_mode(self):
        s = normalize_metric(50.0, mode="bounded", lower=0.0, upper=100.0)
        assert s == pytest.approx(50.0)

    def test_gaussian_mode(self):
        s = normalize_metric(0.0, mode="gaussian", sigma=50.0)
        assert s == pytest.approx(100.0)

    def test_piecewise_mode(self):
        bp = [(0.0, 100.0), (100.0, 0.0)]
        s = normalize_metric(50.0, mode="piecewise", breakpoints=bp)
        assert s == pytest.approx(50.0)

    def test_threshold_mode_below(self):
        assert normalize_metric(0.3, mode="threshold", threshold=0.5) == pytest.approx(100.0)

    def test_threshold_mode_above(self):
        assert normalize_metric(0.8, mode="threshold", threshold=0.5) == pytest.approx(0.0)

    def test_threshold_mode_at_boundary(self):
        assert normalize_metric(0.5, mode="threshold", threshold=0.5) == pytest.approx(100.0)

    def test_nonfinite_nan_returns_zero(self):
        assert normalize_metric(float("nan"), mode="bounded") == pytest.approx(0.0)

    def test_nonfinite_inf_returns_zero(self):
        assert normalize_metric(float("inf"), mode="bounded") == pytest.approx(0.0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            normalize_metric(1.0, mode="unknown_mode")

    def test_result_in_range_all_modes(self):
        bp = [(0.0, 100.0), (100.0, 0.0)]
        for mode, kwargs in [
            ("bounded",   {"lower": 0.0, "upper": 100.0}),
            ("gaussian",  {"sigma": 50.0}),
            ("piecewise", {"breakpoints": bp}),
            ("threshold", {"threshold": 0.5}),
        ]:
            s = normalize_metric(50.0, mode=mode, **kwargs)
            assert 0.0 <= s <= 100.0, f"mode={mode} returned {s} outside [0,100]"
