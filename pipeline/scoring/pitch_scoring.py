"""
scoring/pitch_scoring.py - Pitch category score computation.

Transforms PitchMetrics (from Phase 6) into a CategoryScore using configurable
piecewise normalization curves. All scoring is deterministic and interpretable.

Functions:
    compute_intonation_score      MACE-based intonation quality → ScoreBreakdown
    compute_pitch_stability_score RMSE-based pitch stability   → ScoreBreakdown
    compute_pitch_score           Overall pitch CategoryScore
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from scoring.normalization import piecewise_score
from utils.types import CategoryScore, PitchMetrics, ScoreBreakdown


# Default piecewise breakpoints — configurable via the `config` dict.
# Units for MACE / RMSE are cents (¢).
# 50¢ = one quarter-tone; 100¢ = one full semitone.
_ACCURACY_BP = [(0.0, 30.0), (0.25, 65.0), (0.50, 83.0), (0.67, 92.0), (0.80, 97.0), (1.0, 100.0)]
_MACE_BP = [(0.0, 100.0), (50.0, 95.0), (100.0, 88.0), (200.0, 80.0), (400.0, 70.0), (700.0, 55.0), (1200.0, 30.0)]
_RMSE_BP = [(0.0, 100.0), (50.0, 93.0), (100.0, 85.0), (200.0, 75.0), (400.0, 65.0), (700.0, 50.0), (1200.0, 30.0)]


def _confidence_from_n(n: int) -> float:
    """Linearly interpolate confidence: 0 pts → 0.0, ≥3 pts → 1.0."""
    return min(1.0, n / 3.0)


def compute_intonation_score(
    pitch: PitchMetrics,
    config: Optional[Dict[str, Any]] = None,
) -> ScoreBreakdown:
    """
    Score intonation quality from Mean Absolute Cent Error (MACE).

    Lower MACE → higher score. Falls back to zero-confidence placeholder when
    mace_cents is None (insufficient data).

    Args:
        pitch:  PitchMetrics from Phase 6.
        config: Optional overrides; supports "mace_breakpoints".

    Returns:
        ScoreBreakdown with component="intonation".
    """
    cfg = config or {}
    bp = cfg.get("mace_breakpoints", _MACE_BP)

    if pitch.mace_cents is None:
        return ScoreBreakdown(
            component="intonation",
            raw_value=None,
            score=0.0,
            confidence=0.0,
            metadata={"reason": "no_pitch_data"},
        )

    score = piecewise_score(pitch.mace_cents, bp)
    return ScoreBreakdown(
        component="intonation",
        raw_value=pitch.mace_cents,
        score=score,
        confidence=_confidence_from_n(pitch.n_evaluated),
        metadata={
            "mace_cents": pitch.mace_cents,
            "n_evaluated": pitch.n_evaluated,
        },
    )


def compute_pitch_stability_score(
    pitch: PitchMetrics,
    config: Optional[Dict[str, Any]] = None,
) -> ScoreBreakdown:
    """
    Score pitch stability from pitch RMSE in cents.

    RMSE penalises large deviations more than MACE, making it sensitive to
    occasional large jumps. Lower RMSE → higher score.

    Args:
        pitch:  PitchMetrics from Phase 6.
        config: Optional overrides; supports "rmse_breakpoints".

    Returns:
        ScoreBreakdown with component="stability".
    """
    cfg = config or {}
    bp = cfg.get("rmse_breakpoints", _RMSE_BP)

    if pitch.pitch_rmse_cents is None:
        return ScoreBreakdown(
            component="stability",
            raw_value=None,
            score=0.0,
            confidence=0.0,
            metadata={"reason": "no_pitch_data"},
        )

    score = piecewise_score(pitch.pitch_rmse_cents, bp)
    return ScoreBreakdown(
        component="stability",
        raw_value=pitch.pitch_rmse_cents,
        score=score,
        confidence=_confidence_from_n(pitch.n_evaluated),
        metadata={"rmse_cents": pitch.pitch_rmse_cents},
    )


def compute_pitch_score(
    pitch: PitchMetrics,
    config: Optional[Dict[str, Any]] = None,
) -> CategoryScore:
    """
    Compute overall pitch CategoryScore from a PitchMetrics object.

    Combines three components with configurable weights:
      - accuracy   (fraction of notes within tolerance) × weight_accuracy
      - intonation (MACE-based)                         × weight_intonation
      - stability  (RMSE-based)                         × weight_stability

    Each component is confidence-weighted so that data-poor inputs contribute
    proportionally less to the aggregate.

    Args:
        pitch:  PitchMetrics from Phase 6.
        config: Optional overrides:
                  weight_accuracy   (default 0.50)
                  weight_intonation (default 0.30)
                  weight_stability  (default 0.20)
                  mace_breakpoints / rmse_breakpoints (passed through)

    Returns:
        CategoryScore with category="pitch".
    """
    cfg = config or {}
    w_acc = float(cfg.get("weight_accuracy", 0.50))
    w_int = float(cfg.get("weight_intonation", 0.30))
    w_sta = float(cfg.get("weight_stability", 0.20))

    # Accuracy component
    if pitch.pitch_accuracy is not None:
        acc_score = piecewise_score(pitch.pitch_accuracy, _ACCURACY_BP)
        acc_conf = _confidence_from_n(pitch.n_evaluated)
    else:
        acc_score = 0.0
        acc_conf = 0.0

    acc_bd = ScoreBreakdown(
        component="accuracy",
        raw_value=pitch.pitch_accuracy,
        score=acc_score,
        weight=w_acc,
        confidence=acc_conf,
        metadata={"tolerance_cents": pitch.tolerance_cents},
    )

    intonation_bd = compute_intonation_score(pitch, cfg)
    intonation_bd.weight = w_int

    stability_bd = compute_pitch_stability_score(pitch, cfg)
    stability_bd.weight = w_sta

    components: List[ScoreBreakdown] = [acc_bd, intonation_bd, stability_bd]

    # Confidence-weighted aggregation — zero-confidence components don't contribute
    total_eff_weight = 0.0
    weighted_sum = 0.0
    for c in components:
        conf = c.confidence if c.confidence is not None else 0.0
        eff = c.weight * conf
        total_eff_weight += eff
        weighted_sum += c.score * eff

    if total_eff_weight > 0.0:
        overall = weighted_sum / total_eff_weight
        overall_conf = _confidence_from_n(pitch.n_evaluated)
    else:
        overall = 0.0
        overall_conf = 0.0

    # Dampen scores from sparse evaluations so a single lucky note can't
    # outscore a well-evaluated recording. n=1→0.71x, n≥2→1.0x.
    import math
    sparsity_factor = math.sqrt(min(pitch.n_evaluated, 2) / 2.0) if pitch.n_evaluated > 0 else 0.0
    adjusted = max(0.0, min(100.0, overall)) * sparsity_factor

    return CategoryScore(
        category="pitch",
        score=adjusted,
        confidence=overall_conf,
        components=components,
        n_evaluated=pitch.n_evaluated,
        metadata={"weights": {"accuracy": w_acc, "intonation": w_int, "stability": w_sta}},
    )
