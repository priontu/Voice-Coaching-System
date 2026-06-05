"""
scoring/interpretation.py - Deterministic musical interpretation engine.

Converts a PerformanceScoreReport into an InterpretationSummary using
threshold-based rules only. No LLM usage. No freeform generation.

Functions:
    build_interpretation_summary   PerformanceScoreReport → InterpretationSummary
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from utils.types import CategoryScore, InterpretationSummary, PerformanceScoreReport


# Level thresholds (scores are ∈ [0, 100]).
_DEFAULT_EXCELLENT = 90.0
_DEFAULT_GOOD      = 75.0
_DEFAULT_FAIR      = 55.0

# Deterministic message templates keyed by (category, level).
# Only "excellent" and above are strengths; "fair" and "needs_work" are weaknesses.
_STRENGTH_MESSAGES: Dict[Tuple[str, str], str] = {
    ("pitch",    "excellent"): "Strong pitch intonation and accuracy",
    ("timing",   "excellent"): "Excellent rhythmic timing",
    ("duration", "excellent"): "Well-controlled note durations",
    ("lyric",    "excellent"): "Excellent lyric synchronization",
    ("pitch",    "good"):      "Good pitch accuracy overall",
    ("timing",   "good"):      "Good rhythmic timing",
    ("duration", "good"):      "Good note duration control",
    ("lyric",    "good"):      "Good lyric delivery",
}

_WEAKNESS_MESSAGES: Dict[Tuple[str, str], str] = {
    ("pitch",    "fair"):       "Pitch intonation could be more accurate",
    ("pitch",    "needs_work"): "Pitch intonation needs significant improvement",
    ("timing",   "fair"):       "Minor timing inconsistencies detected",
    ("timing",   "needs_work"): "Significant timing inconsistencies detected",
    ("duration", "fair"):       "Some note duration inconsistencies detected",
    ("duration", "needs_work"): "Note duration control needs improvement",
    ("lyric",    "fair"):       "Some phoneme timing inaccuracies detected",
    ("lyric",    "needs_work"): "Lyric delivery timing needs improvement",
}


def _score_to_level(
    score: float,
    excellent: float,
    good: float,
    fair: float,
) -> str:
    """Map a score ∈ [0, 100] to a level string."""
    if score >= excellent:
        return "excellent"
    if score >= good:
        return "good"
    if score >= fair:
        return "fair"
    return "needs_work"


def build_interpretation_summary(
    score_report: PerformanceScoreReport,
    config: Optional[Dict[str, Any]] = None,
) -> InterpretationSummary:
    """
    Build a deterministic InterpretationSummary from a PerformanceScoreReport.

    Levels are assigned using configurable score thresholds. All messages are
    looked up from a static rule table — there is no LLM usage or freeform
    text generation.

    Strengths are emitted for categories at "excellent" or "good" level.
    Weaknesses are emitted for categories at "fair" or "needs_work" level.

    Args:
        score_report:  PerformanceScoreReport from performance_scoring.py.
        config:        Optional threshold overrides:
                         excellent_threshold (default 90)
                         good_threshold      (default 75)
                         fair_threshold      (default 55)

    Returns:
        InterpretationSummary with level labels and rule-triggered messages.
    """
    cfg = config or {}
    exc_thr  = float(cfg.get("excellent_threshold", _DEFAULT_EXCELLENT))
    good_thr = float(cfg.get("good_threshold",      _DEFAULT_GOOD))
    fair_thr = float(cfg.get("fair_threshold",      _DEFAULT_FAIR))

    categories: List[Tuple[str, Optional[CategoryScore]]] = [
        ("pitch",    score_report.pitch_score),
        ("timing",   score_report.timing_score),
        ("duration", score_report.duration_score),
        ("lyric",    score_report.lyric_score),
    ]

    strengths: List[str] = []
    weaknesses: List[str] = []
    category_levels: Dict[str, str] = {}

    for cat_name, cat_score in categories:
        if cat_score is None:
            continue

        level = _score_to_level(cat_score.score, exc_thr, good_thr, fair_thr)
        category_levels[cat_name] = level

        key = (cat_name, level)
        if level in ("excellent", "good"):
            msg = _STRENGTH_MESSAGES.get(key)
            if msg:
                strengths.append(msg)
        else:
            msg = _WEAKNESS_MESSAGES.get(key)
            if msg:
                weaknesses.append(msg)

    # Overall level
    if score_report.overall_score is not None:
        overall_level = _score_to_level(
            score_report.overall_score, exc_thr, good_thr, fair_thr
        )
    elif category_levels:
        # Fallback: most common level across categories
        level_counts: Dict[str, int] = {}
        for lvl in category_levels.values():
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
        overall_level = max(level_counts, key=lambda k: level_counts[k])
    else:
        overall_level = "needs_work"

    return InterpretationSummary(
        audio_path=score_report.audio_path,
        overall_level=overall_level,
        strengths=strengths,
        weaknesses=weaknesses,
        category_levels=category_levels,
        metadata={
            "thresholds": {
                "excellent": exc_thr,
                "good": good_thr,
                "fair": fair_thr,
            },
            "overall_score": score_report.overall_score,
        },
    )
