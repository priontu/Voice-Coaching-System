"""
scoring/duration_scoring.py - Note duration category score computation.

Transforms DurationMetrics (Phase 6) into a CategoryScore. Duration scoring
is tempo-independent: it uses relative error (|error| / ref_dur) as the primary
signal, making it robust across different tempos.

Functions:
    compute_phrase_duration_score  Consistency-based duration breakdown → ScoreBreakdown
    compute_duration_score         Overall duration CategoryScore
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from scoring.normalization import piecewise_score
from utils.types import CategoryScore, DurationMetrics, ScoreBreakdown


# Breakpoints for relative duration error (dimensionless fraction).
# 0.0 = perfect, 1.0 = 100% off (note twice as long or not at all).
_REL_ERROR_BP = [(0.0, 100.0), (0.1, 90.0), (0.2, 75.0), (0.5, 50.0), (1.0, 0.0)]

# Breakpoints for duration ratio deviation from 1.0.
# |ratio - 1.0| = 0 → perfect; |ratio - 1.0| > 1 → very poor.
_RATIO_DEV_BP = [(0.0, 100.0), (0.1, 90.0), (0.25, 75.0), (0.5, 50.0), (1.0, 0.0)]

# Breakpoints for std of duration error (seconds).
_STD_BP = [(0.0, 100.0), (0.05, 90.0), (0.10, 75.0), (0.20, 50.0), (0.40, 0.0)]


def _confidence_from_n(n: int) -> float:
    return min(1.0, n / 3.0)


def compute_phrase_duration_score(
    duration: DurationMetrics,
    config: Optional[Dict[str, Any]] = None,
) -> ScoreBreakdown:
    """
    Score phrase-level duration consistency from std and relative error.

    Uses `std_duration_error_s` (consistency across notes) and
    `mean_relative_duration_error` (overall tempo-independent accuracy).
    Both contribute equally when available.

    Args:
        duration:  DurationMetrics from Phase 6.
        config:    Optional overrides: "std_breakpoints", "rel_error_breakpoints".

    Returns:
        ScoreBreakdown with component="phrase_duration".
    """
    cfg = config or {}
    std_bp = cfg.get("std_breakpoints", _STD_BP)
    rel_bp = cfg.get("rel_error_breakpoints", _REL_ERROR_BP)

    parts: List[float] = []
    meta: Dict[str, Optional[float]] = {}

    if duration.std_duration_error_s is not None:
        parts.append(piecewise_score(duration.std_duration_error_s, std_bp))
        meta["std_duration_error_s"] = duration.std_duration_error_s

    if duration.mean_relative_duration_error is not None:
        parts.append(piecewise_score(duration.mean_relative_duration_error, rel_bp))
        meta["mean_relative_duration_error"] = duration.mean_relative_duration_error

    if not parts:
        return ScoreBreakdown(
            component="phrase_duration",
            raw_value=None,
            score=0.0,
            confidence=0.0,
            metadata={"reason": "no_duration_data"},
        )

    score = sum(parts) / len(parts)
    return ScoreBreakdown(
        component="phrase_duration",
        raw_value=duration.mean_relative_duration_error,
        score=score,
        confidence=_confidence_from_n(duration.n_evaluated),
        metadata=meta,
    )


def compute_duration_score(
    duration: DurationMetrics,
    config: Optional[Dict[str, Any]] = None,
) -> CategoryScore:
    """
    Compute overall duration CategoryScore from a DurationMetrics object.

    Combines three components with configurable weights:
      - relative_error  (tempo-independent |error| / ref_dur) × weight_relative
      - ratio           (deviation of pred/ref ratio from 1.0) × weight_ratio
      - phrase_duration (consistency via std + relative error) × weight_phrase

    Args:
        duration:  DurationMetrics from Phase 6.
        config:    Optional overrides:
                     weight_relative (default 0.60)
                     weight_ratio    (default 0.20)
                     weight_phrase   (default 0.20)
                     rel_error_breakpoints / ratio_breakpoints / std_breakpoints

    Returns:
        CategoryScore with category="duration".
    """
    cfg = config or {}
    w_rel    = float(cfg.get("weight_relative", 0.60))
    w_ratio  = float(cfg.get("weight_ratio", 0.20))
    w_phrase = float(cfg.get("weight_phrase", 0.20))

    rel_bp   = cfg.get("rel_error_breakpoints", _REL_ERROR_BP)
    ratio_bp = cfg.get("ratio_breakpoints", _RATIO_DEV_BP)

    # Relative error component
    if duration.mean_relative_duration_error is not None:
        rel_score = piecewise_score(duration.mean_relative_duration_error, rel_bp)
        rel_conf = _confidence_from_n(duration.n_evaluated)
    else:
        rel_score = 0.0
        rel_conf = 0.0

    rel_bd = ScoreBreakdown(
        component="relative_error",
        raw_value=duration.mean_relative_duration_error,
        score=rel_score,
        weight=w_rel,
        confidence=rel_conf,
    )

    # Ratio component: score based on |ratio - 1.0|
    if duration.mean_duration_ratio is not None:
        ratio_dev = abs(duration.mean_duration_ratio - 1.0)
        ratio_score = piecewise_score(ratio_dev, ratio_bp)
        ratio_conf = _confidence_from_n(duration.n_evaluated)
    else:
        ratio_score = 0.0
        ratio_conf = 0.0

    ratio_bd = ScoreBreakdown(
        component="ratio",
        raw_value=duration.mean_duration_ratio,
        score=ratio_score,
        weight=w_ratio,
        confidence=ratio_conf,
        metadata={"mean_duration_ratio": duration.mean_duration_ratio},
    )

    phrase_bd = compute_phrase_duration_score(duration, cfg)
    phrase_bd.weight = w_phrase

    components: List[ScoreBreakdown] = [rel_bd, ratio_bd, phrase_bd]

    total_eff_weight = 0.0
    weighted_sum = 0.0
    for c in components:
        conf = c.confidence if c.confidence is not None else 0.0
        eff = c.weight * conf
        total_eff_weight += eff
        weighted_sum += c.score * eff

    if total_eff_weight > 0.0:
        overall = weighted_sum / total_eff_weight
        overall_conf = _confidence_from_n(duration.n_evaluated)
    else:
        overall = 0.0
        overall_conf = 0.0

    return CategoryScore(
        category="duration",
        score=max(0.0, min(100.0, overall)),
        confidence=overall_conf,
        components=components,
        n_evaluated=duration.n_evaluated,
        metadata={"weights": {"relative_error": w_rel, "ratio": w_ratio, "phrase_duration": w_phrase}},
    )
