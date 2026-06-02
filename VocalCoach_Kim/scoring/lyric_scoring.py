"""
scoring/lyric_scoring.py - Lyric/diction category score computation.

Transforms LyricMetrics (Phase 6) into a CategoryScore. Evaluates phoneme
boundary timing, word alignment quality, phoneme overlap, and label accuracy.

Functions:
    compute_phoneme_timing_score  Boundary error-based timing → ScoreBreakdown
    compute_lyric_clarity_score   Overall lyric CategoryScore
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from scoring.normalization import piecewise_score
from utils.types import CategoryScore, LyricMetrics, ScoreBreakdown


# Breakpoints for mean absolute phoneme boundary error (ms).
# Stricter than note-level timing: phoneme accuracy is finer-grained.
_BOUNDARY_BP = [(0.0, 100.0), (15.0, 88.0), (30.0, 75.0), (60.0, 50.0), (120.0, 0.0)]


def _confidence_from_n(n: int) -> float:
    return min(1.0, n / 3.0)


def compute_phoneme_timing_score(
    lyric: LyricMetrics,
    config: Optional[Dict[str, Any]] = None,
) -> ScoreBreakdown:
    """
    Score phoneme boundary timing from mean absolute boundary error (ms).

    Lower absolute error → higher score. Falls back to zero-confidence placeholder
    when no phoneme match data is available.

    Args:
        lyric:  LyricMetrics from Phase 6.
        config: Optional overrides; supports "boundary_breakpoints".

    Returns:
        ScoreBreakdown with component="phoneme_timing".
    """
    cfg = config or {}
    bp = cfg.get("boundary_breakpoints", _BOUNDARY_BP)

    if lyric.mean_abs_phoneme_boundary_error_ms is None:
        return ScoreBreakdown(
            component="phoneme_timing",
            raw_value=None,
            score=0.0,
            confidence=0.0,
            metadata={"reason": "no_phoneme_data"},
        )

    score = piecewise_score(lyric.mean_abs_phoneme_boundary_error_ms, bp)
    return ScoreBreakdown(
        component="phoneme_timing",
        raw_value=lyric.mean_abs_phoneme_boundary_error_ms,
        score=score,
        confidence=_confidence_from_n(lyric.n_phoneme_matches),
        metadata={"mean_abs_boundary_error_ms": lyric.mean_abs_phoneme_boundary_error_ms},
    )


def compute_lyric_clarity_score(
    lyric: LyricMetrics,
    config: Optional[Dict[str, Any]] = None,
) -> CategoryScore:
    """
    Compute overall lyric/diction CategoryScore from LyricMetrics.

    Combines four components with configurable weights:
      - word_accuracy  (fraction of reference words matched)        × weight_word
      - overlap        (fraction of phonemes with overlap ≥ 0.5)   × weight_overlap
      - label_match    (fraction of phoneme pairs with same label)  × weight_label
      - phoneme_timing (mean abs boundary error)                    × weight_timing

    Args:
        lyric:   LyricMetrics from Phase 6.
        config:  Optional overrides:
                   weight_word    (default 0.35)
                   weight_overlap (default 0.25)
                   weight_label   (default 0.25)
                   weight_timing  (default 0.15)
                   boundary_breakpoints

    Returns:
        CategoryScore with category="lyric".
    """
    cfg = config or {}
    w_word    = float(cfg.get("weight_word", 0.35))
    w_overlap = float(cfg.get("weight_overlap", 0.25))
    w_label   = float(cfg.get("weight_label", 0.25))
    w_timing  = float(cfg.get("weight_timing", 0.15))

    # Word alignment accuracy
    if lyric.word_alignment_accuracy is not None:
        word_score = lyric.word_alignment_accuracy * 100.0
        word_conf = _confidence_from_n(lyric.n_word_matches)
    else:
        word_score = 0.0
        word_conf = 0.0

    word_bd = ScoreBreakdown(
        component="word_accuracy",
        raw_value=lyric.word_alignment_accuracy,
        score=word_score,
        weight=w_word,
        confidence=word_conf,
    )

    # Phoneme overlap accuracy
    if lyric.phoneme_overlap_accuracy is not None:
        overlap_score = lyric.phoneme_overlap_accuracy * 100.0
        overlap_conf = _confidence_from_n(lyric.n_phoneme_matches)
    else:
        overlap_score = 0.0
        overlap_conf = 0.0

    overlap_bd = ScoreBreakdown(
        component="overlap",
        raw_value=lyric.phoneme_overlap_accuracy,
        score=overlap_score,
        weight=w_overlap,
        confidence=overlap_conf,
    )

    # Label match rate
    if lyric.label_match_rate is not None:
        label_score = lyric.label_match_rate * 100.0
        label_conf = _confidence_from_n(lyric.n_phoneme_matches)
    else:
        label_score = 0.0
        label_conf = 0.0

    label_bd = ScoreBreakdown(
        component="label_match",
        raw_value=lyric.label_match_rate,
        score=label_score,
        weight=w_label,
        confidence=label_conf,
    )

    timing_bd = compute_phoneme_timing_score(lyric, cfg)
    timing_bd.weight = w_timing

    components: List[ScoreBreakdown] = [word_bd, overlap_bd, label_bd, timing_bd]

    total_eff_weight = 0.0
    weighted_sum = 0.0
    for c in components:
        conf = c.confidence if c.confidence is not None else 0.0
        eff = c.weight * conf
        total_eff_weight += eff
        weighted_sum += c.score * eff

    n_eval = lyric.n_phoneme_matches + lyric.n_word_matches
    if total_eff_weight > 0.0:
        overall = weighted_sum / total_eff_weight
        overall_conf = _confidence_from_n(n_eval)
    else:
        overall = 0.0
        overall_conf = 0.0

    return CategoryScore(
        category="lyric",
        score=max(0.0, min(100.0, overall)),
        confidence=overall_conf,
        components=components,
        n_evaluated=n_eval,
        metadata={
            "weights": {
                "word_accuracy": w_word,
                "overlap": w_overlap,
                "label_match": w_label,
                "phoneme_timing": w_timing,
            }
        },
    )
