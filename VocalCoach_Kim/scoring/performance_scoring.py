"""
scoring/performance_scoring.py - Overall performance score aggregation.

Aggregates all category scores (pitch, timing, duration, lyric) into a single
PerformanceScoreReport with an overall normalized score. Weighting is fully
configurable and confidence-aware; categories without sufficient data contribute
proportionally less.

Functions:
    build_performance_score_report   PerformanceMetricsReport → PerformanceScoreReport
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from scoring.duration_scoring import compute_duration_score
from scoring.lyric_scoring import compute_lyric_clarity_score
from scoring.pitch_scoring import compute_pitch_score
from scoring.timing_scoring import compute_timing_score
from utils.types import (
    CategoryScore,
    DurationMetrics,
    LyricMetrics,
    PerformanceMetricsReport,
    PerformanceScoreReport,
    PitchMetrics,
    TimingMetrics,
)


# Default category weights — must be positive; they are normalized to sum to 1.
_DEFAULT_WEIGHTS = {
    "pitch": 0.40,
    "timing": 0.30,
    "duration": 0.15,
    "lyric": 0.15,
}


def build_performance_score_report(
    metrics: PerformanceMetricsReport,
    config: Optional[Dict[str, Any]] = None,
) -> PerformanceScoreReport:
    """
    Aggregate all category scores into a PerformanceScoreReport.

    For each available metric sub-report (pitch, timing, duration, lyric) a
    CategoryScore is computed. The overall score is a confidence-weighted
    average using the configured (or default) weights.

    Categories whose metrics sub-report is None are silently omitted from the
    aggregate; the remaining weights are re-normalised so they sum to 1.

    Args:
        metrics: PerformanceMetricsReport produced by Phase 6 metrics/reporting.py.
        config:  Optional scoring configuration dict.  Supported keys:
                   weights          dict with pitch/timing/duration/lyric weights
                   pitch            passed through to compute_pitch_score
                   timing           passed through to compute_timing_score
                   duration         passed through to compute_duration_score
                   lyric            passed through to compute_lyric_clarity_score

    Returns:
        PerformanceScoreReport with per-category and overall scores.
    """
    t_start = time.perf_counter()
    cfg = config or {}

    # ── Resolve weights ───────────────────────────────────────────────
    raw_weights: Dict[str, float] = dict(_DEFAULT_WEIGHTS)
    user_weights = cfg.get("weights", {})
    for k, v in user_weights.items():
        if k in raw_weights:
            raw_weights[k] = float(v)

    # ── Category scoring ──────────────────────────────────────────────
    pitch_cat: Optional[CategoryScore] = None
    if metrics.pitch is not None:
        pitch_cat = compute_pitch_score(metrics.pitch, cfg.get("pitch"))

    timing_cat: Optional[CategoryScore] = None
    if metrics.timing is not None:
        timing_cat = compute_timing_score(metrics.timing, cfg.get("timing"))

    duration_cat: Optional[CategoryScore] = None
    if metrics.duration is not None:
        duration_cat = compute_duration_score(metrics.duration, cfg.get("duration"))

    lyric_cat: Optional[CategoryScore] = None
    if metrics.lyric is not None:
        lyric_cat = compute_lyric_clarity_score(metrics.lyric, cfg.get("lyric"))

    # ── Overall score (confidence-weighted) ───────────────────────────
    category_map = {
        "pitch":    pitch_cat,
        "timing":   timing_cat,
        "duration": duration_cat,
        "lyric":    lyric_cat,
    }

    effective_weights: Dict[str, float] = {}
    total_eff = 0.0
    weighted_sum = 0.0

    for cat_name, cat_score in category_map.items():
        if cat_score is None:
            continue
        nominal = raw_weights.get(cat_name, 0.0)
        conf = cat_score.confidence if cat_score.confidence is not None else 0.0
        eff = nominal * conf
        effective_weights[cat_name] = eff
        total_eff += eff
        weighted_sum += cat_score.score * eff

    if total_eff > 0.0:
        overall = weighted_sum / total_eff
        # Normalize effective weights to sum to 1 for reporting
        weights_used = {k: round(v / total_eff, 6) for k, v in effective_weights.items()}
    else:
        overall = None
        weights_used = {}

    elapsed = time.perf_counter() - t_start
    return PerformanceScoreReport(
        audio_path=metrics.audio_path,
        reference_source_path=metrics.reference_source_path,
        pitch_score=pitch_cat,
        timing_score=timing_cat,
        duration_score=duration_cat,
        lyric_score=lyric_cat,
        overall_score=max(0.0, min(100.0, overall)) if overall is not None else None,
        weights_used=weights_used,
        score_metadata={
            "elapsed_s": round(elapsed, 4),
            "config_used": cfg,
            "n_note_matches": metrics.n_note_matches,
            "categories_computed": [k for k, v in category_map.items() if v is not None],
        },
    )
