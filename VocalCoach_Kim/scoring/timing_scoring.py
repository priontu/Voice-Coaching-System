"""
scoring/timing_scoring.py - Rhythm and timing category score computation.

Transforms TimingMetrics (Phase 6) into a CategoryScore using configurable
piecewise normalization. All scoring is deterministic.

Functions:
    compute_rhythm_stability_score  IOI + std-dev rhythm stability → ScoreBreakdown
    compute_timing_score            Overall timing CategoryScore
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from scoring.normalization import piecewise_score
from utils.types import CategoryScore, ScoreBreakdown, TimingMetrics


# Breakpoints for mean absolute onset error (ms) and IOI MAE (ms).
# 50 ms is the default tolerance used by Phase 6 timing_accuracy.
_ONSET_MAE_BP = [(0.0, 100.0), (25.0, 88.0), (50.0, 75.0), (100.0, 50.0), (200.0, 0.0)]
_IOI_MAE_BP   = [(0.0, 100.0), (30.0, 88.0), (60.0, 75.0), (120.0, 50.0), (240.0, 0.0)]
_STD_BP       = [(0.0, 100.0), (30.0, 88.0), (60.0, 72.0), (120.0, 45.0), (240.0, 0.0)]


def _confidence_from_n(n: int) -> float:
    return min(1.0, n / 3.0)


def compute_rhythm_stability_score(
    timing: TimingMetrics,
    config: Optional[Dict[str, Any]] = None,
) -> ScoreBreakdown:
    """
    Score rhythmic consistency from IOI MAE and onset standard deviation.

    Both capture different aspects of rhythm stability:
      - ioi_mae_ms: how consistently inter-note intervals match the reference
      - std_onset_error_ms: how tightly onset deviations cluster around their mean

    When only one is available it is used alone. When neither is available
    the breakdown is a zero-confidence placeholder.

    Args:
        timing:  TimingMetrics from Phase 6.
        config:  Optional overrides; supports "ioi_breakpoints", "std_breakpoints".

    Returns:
        ScoreBreakdown with component="rhythm_stability".
    """
    cfg = config or {}
    ioi_bp = cfg.get("ioi_breakpoints", _IOI_MAE_BP)
    std_bp = cfg.get("std_breakpoints", _STD_BP)

    parts: List[float] = []
    raw_values: Dict[str, Optional[float]] = {}

    if timing.ioi_mae_ms is not None:
        parts.append(piecewise_score(timing.ioi_mae_ms, ioi_bp))
        raw_values["ioi_mae_ms"] = timing.ioi_mae_ms

    if timing.std_onset_error_ms is not None:
        parts.append(piecewise_score(timing.std_onset_error_ms, std_bp))
        raw_values["std_onset_error_ms"] = timing.std_onset_error_ms

    if not parts:
        return ScoreBreakdown(
            component="rhythm_stability",
            raw_value=None,
            score=0.0,
            confidence=0.0,
            metadata={"reason": "no_timing_data"},
        )

    score = sum(parts) / len(parts)
    return ScoreBreakdown(
        component="rhythm_stability",
        raw_value=timing.ioi_mae_ms,  # primary raw signal
        score=score,
        confidence=_confidence_from_n(timing.n_evaluated),
        metadata=raw_values,
    )


def compute_timing_score(
    timing: TimingMetrics,
    config: Optional[Dict[str, Any]] = None,
) -> CategoryScore:
    """
    Compute overall timing CategoryScore from a TimingMetrics object.

    Combines three components with configurable weights:
      - accuracy      (fraction of onsets within tolerance) × weight_accuracy
      - onset_mae     (mean absolute onset error)           × weight_mae
      - rhythm_stability (IOI + std-dev consistency)        × weight_stability

    Args:
        timing:  TimingMetrics from Phase 6.
        config:  Optional overrides:
                   weight_accuracy  (default 0.50)
                   weight_mae       (default 0.30)
                   weight_stability (default 0.20)
                   onset_mae_breakpoints / ioi_breakpoints / std_breakpoints

    Returns:
        CategoryScore with category="timing".
    """
    cfg = config or {}
    w_acc = float(cfg.get("weight_accuracy", 0.50))
    w_mae = float(cfg.get("weight_mae", 0.30))
    w_sta = float(cfg.get("weight_stability", 0.20))

    onset_mae_bp = cfg.get("onset_mae_breakpoints", _ONSET_MAE_BP)

    # Accuracy component
    if timing.timing_accuracy is not None:
        acc_score = timing.timing_accuracy * 100.0
        acc_conf = _confidence_from_n(timing.n_evaluated)
    else:
        acc_score = 0.0
        acc_conf = 0.0

    acc_bd = ScoreBreakdown(
        component="accuracy",
        raw_value=timing.timing_accuracy,
        score=acc_score,
        weight=w_acc,
        confidence=acc_conf,
        metadata={"tolerance_ms": timing.tolerance_ms},
    )

    # Mean absolute onset error component
    if timing.mean_abs_onset_error_ms is not None:
        mae_score = piecewise_score(timing.mean_abs_onset_error_ms, onset_mae_bp)
        mae_conf = _confidence_from_n(timing.n_evaluated)
    else:
        mae_score = 0.0
        mae_conf = 0.0

    mae_bd = ScoreBreakdown(
        component="onset_mae",
        raw_value=timing.mean_abs_onset_error_ms,
        score=mae_score,
        weight=w_mae,
        confidence=mae_conf,
        metadata={"mean_abs_onset_error_ms": timing.mean_abs_onset_error_ms},
    )

    stability_bd = compute_rhythm_stability_score(timing, cfg)
    stability_bd.weight = w_sta

    components: List[ScoreBreakdown] = [acc_bd, mae_bd, stability_bd]

    total_eff_weight = 0.0
    weighted_sum = 0.0
    for c in components:
        conf = c.confidence if c.confidence is not None else 0.0
        eff = c.weight * conf
        total_eff_weight += eff
        weighted_sum += c.score * eff

    if total_eff_weight > 0.0:
        overall = weighted_sum / total_eff_weight
        overall_conf = _confidence_from_n(timing.n_evaluated)
    else:
        overall = 0.0
        overall_conf = 0.0

    return CategoryScore(
        category="timing",
        score=max(0.0, min(100.0, overall)),
        confidence=overall_conf,
        components=components,
        n_evaluated=timing.n_evaluated,
        metadata={"weights": {"accuracy": w_acc, "onset_mae": w_mae, "rhythm_stability": w_sta}},
    )
