"""
scoring/normalization.py - Score normalization utilities for the VocalCoach scoring engine.

All functions are numerically stable and deterministic. They transform raw metric
values (cents, milliseconds, fractions) into normalized scores on the [0, 100] scale.

Functions:
    bounded_score       Linear map: [lower=best, upper=worst] → [100, 0], clamped
    gaussian_penalty    Gaussian decay: 100 * exp(-x² / (2σ²))
    piecewise_score     Piecewise-linear interpolation through (x, score) breakpoints
    normalize_metric    Generic dispatcher (bounded | gaussian | piecewise | threshold)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


def bounded_score(value: float, lower: float, upper: float) -> float:
    """
    Map value from [lower, upper] linearly to [100, 0], clamped.

    `lower` is the "best" end (maps to 100); `upper` is the "worst" end (maps to 0).
    Values outside the range are clamped to 0 or 100.

    Args:
        value:  Raw metric value.
        lower:  Best achievable value → score 100.
        upper:  Worst threshold value → score 0.

    Returns:
        float in [0.0, 100.0]
    """
    if upper == lower:
        return 100.0 if value <= lower else 0.0
    frac = (value - lower) / (upper - lower)
    return float(max(0.0, min(100.0, (1.0 - frac) * 100.0)))


def gaussian_penalty(value: float, sigma: float) -> float:
    """
    Gaussian decay score: 100 * exp(-value² / (2σ²)).

    Perfect input (value=0) → 100. Increasing deviation → score decays smoothly.
    Handles non-positive sigma gracefully.

    Args:
        value:  Deviation magnitude (typically absolute value of an error).
        sigma:  Standard deviation controlling the decay rate.

    Returns:
        float in (0.0, 100.0]
    """
    if sigma <= 0.0:
        return 100.0 if value == 0.0 else 0.0
    return float(100.0 * math.exp(-0.5 * (value / sigma) ** 2))


def piecewise_score(
    value: float,
    breakpoints: List[Tuple[float, float]],
) -> float:
    """
    Piecewise-linear score via sorted (x, score) breakpoints.

    x-values must be monotonically non-decreasing. y-values (scores) are in
    [0, 100] but are not enforced — the caller is responsible for sensible values.
    Values below x0 clamp to y0; values above the last x clamp to the last y.

    Args:
        value:       Raw metric value.
        breakpoints: List of (x, score) pairs sorted by ascending x.

    Returns:
        float in [0.0, 100.0] (result is clamped after interpolation)
    """
    if not breakpoints:
        return 0.0
    if len(breakpoints) == 1:
        return float(max(0.0, min(100.0, breakpoints[0][1])))

    if value <= breakpoints[0][0]:
        return float(max(0.0, min(100.0, breakpoints[0][1])))
    if value >= breakpoints[-1][0]:
        return float(max(0.0, min(100.0, breakpoints[-1][1])))

    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= value <= x1:
            if x1 == x0:
                return float(max(0.0, min(100.0, y0)))
            t = (value - x0) / (x1 - x0)
            return float(max(0.0, min(100.0, y0 + t * (y1 - y0))))

    return float(max(0.0, min(100.0, breakpoints[-1][1])))


def normalize_metric(
    value: float,
    mode: str = "bounded",
    lower: float = 0.0,
    upper: float = 100.0,
    sigma: float = 50.0,
    breakpoints: Optional[List[Tuple[float, float]]] = None,
    threshold: float = 0.5,
) -> float:
    """
    Dispatch a raw metric value through the selected normalization curve.

    Non-finite inputs (NaN, Inf) always return 0.0 regardless of mode.

    Args:
        value:       Raw metric value.
        mode:        Normalization strategy — one of:
                       "bounded"    linear clamp (lower=best, upper=worst)
                       "gaussian"   Gaussian decay from zero
                       "piecewise"  piecewise-linear through breakpoints
                       "threshold"  binary: 100 if value <= threshold else 0
        lower:       Used by "bounded" (best/perfect value → 100).
        upper:       Used by "bounded" (worst/max value → 0).
        sigma:       Used by "gaussian" (decay width).
        breakpoints: Used by "piecewise" [(x, score), ...] sorted by x.
        threshold:   Used by "threshold".

    Returns:
        Normalized score ∈ [0.0, 100.0].

    Raises:
        ValueError: If mode is not one of the four supported strings.
    """
    if not math.isfinite(value):
        return 0.0

    if mode == "bounded":
        return bounded_score(value, lower, upper)
    if mode == "gaussian":
        return gaussian_penalty(value, sigma)
    if mode == "piecewise":
        return piecewise_score(value, breakpoints or [])
    if mode == "threshold":
        return 100.0 if value <= threshold else 0.0

    raise ValueError(
        f"Unknown normalization mode {mode!r}. "
        "Choose one of: 'bounded', 'gaussian', 'piecewise', 'threshold'."
    )
