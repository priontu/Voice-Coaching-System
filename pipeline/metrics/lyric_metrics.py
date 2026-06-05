"""
metrics/lyric_metrics.py - Phoneme boundary and word alignment timing metrics.

Evaluates how well predicted phoneme and word events align with reference
TextGrid annotations, using PhonemeAlignmentMatch and WordAlignmentMatch
records produced by alignment/reference_alignment.py.

Boundary error sign: positive = predicted phoneme starts LATER than reference.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.types import (
        AlignmentResult,
        FusedPerformanceRepresentation,
        LyricMetrics,
        MetricBreakdown,
        ReferencePerformanceRepresentation,
    )


def compute_phoneme_boundary_error(
    alignment: "AlignmentResult",
) -> Dict[str, Optional[float]]:
    """
    Phoneme onset timing statistics across all phoneme matches.

    Returns dict with: mean_phoneme_boundary_error_ms,
    mean_abs_phoneme_boundary_error_ms, std_phoneme_boundary_error_ms.
    All values in milliseconds; None when no phoneme matches exist.
    """
    devs_ms = [m.onset_deviation_s * 1000.0 for m in alignment.phoneme_matches]
    if not devs_ms:
        return {
            "mean_phoneme_boundary_error_ms": None,
            "mean_abs_phoneme_boundary_error_ms": None,
            "std_phoneme_boundary_error_ms": None,
        }
    n = len(devs_ms)
    mean = sum(devs_ms) / n
    mean_abs = sum(abs(d) for d in devs_ms) / n
    variance = sum((d - mean) ** 2 for d in devs_ms) / n
    std = math.sqrt(variance)
    return {
        "mean_phoneme_boundary_error_ms": mean,
        "mean_abs_phoneme_boundary_error_ms": mean_abs,
        "std_phoneme_boundary_error_ms": std,
    }


def compute_word_alignment_accuracy(
    alignment: "AlignmentResult",
) -> Optional[float]:
    """
    Fraction of reference words that were matched to a predicted word event.

    Returns None when no word data is available.
    """
    total_ref = len(alignment.word_matches) + len(alignment.unmatched_ref_words)
    if total_ref == 0:
        return None
    return len(alignment.word_matches) / total_ref


def compute_lyric_timing_accuracy(
    alignment: "AlignmentResult",
    tolerance_ms: float = 30.0,
) -> Optional[float]:
    """
    Fraction of matched phonemes whose onset deviation is within tolerance_ms.

    Returns None when no phoneme matches exist.
    """
    if not alignment.phoneme_matches:
        return None
    within = sum(
        1 for m in alignment.phoneme_matches
        if abs(m.onset_deviation_s * 1000.0) <= tolerance_ms
    )
    return within / len(alignment.phoneme_matches)


def build_lyric_metrics(
    alignment: "AlignmentResult",
    fused: Optional["FusedPerformanceRepresentation"] = None,
    reference: Optional["ReferencePerformanceRepresentation"] = None,
    tolerance_ms: float = 30.0,
) -> "LyricMetrics":
    """Compute all lyric/phoneme metrics and return a LyricMetrics dataclass."""
    from utils.types import LyricMetrics, MetricBreakdown

    boundary_stats = compute_phoneme_boundary_error(alignment)
    word_acc = compute_word_alignment_accuracy(alignment)
    lyric_ta = compute_lyric_timing_accuracy(alignment, tolerance_ms)

    # Phoneme overlap accuracy: fraction where overlap_fraction >= 0.5
    phoneme_overlap_accuracy: Optional[float] = None
    if alignment.phoneme_matches:
        n_good = sum(1 for m in alignment.phoneme_matches if m.overlap_fraction >= 0.5)
        phoneme_overlap_accuracy = n_good / len(alignment.phoneme_matches)

    # Label match rate (from PhonemeAlignmentMatch.label_match field)
    label_match_rate: Optional[float] = None
    if alignment.phoneme_matches:
        n_match = sum(1 for m in alignment.phoneme_matches if m.label_match)
        label_match_rate = n_match / len(alignment.phoneme_matches)

    per_phoneme = [
        MetricBreakdown(
            event_idx=m.pred_idx,
            value=m.onset_deviation_s * 1000.0,
            metadata={
                "ref_idx": m.ref_idx,
                "overlap_s": m.overlap_s,
                "overlap_fraction": m.overlap_fraction,
                "label_match": m.label_match,
            },
        )
        for m in alignment.phoneme_matches
    ]

    return LyricMetrics(
        mean_phoneme_boundary_error_ms=boundary_stats["mean_phoneme_boundary_error_ms"],
        mean_abs_phoneme_boundary_error_ms=boundary_stats["mean_abs_phoneme_boundary_error_ms"],
        std_phoneme_boundary_error_ms=boundary_stats["std_phoneme_boundary_error_ms"],
        phoneme_overlap_accuracy=phoneme_overlap_accuracy,
        word_alignment_accuracy=word_acc,
        label_match_rate=label_match_rate,
        n_phoneme_matches=len(alignment.phoneme_matches),
        n_word_matches=len(alignment.word_matches),
        tolerance_ms=tolerance_ms,
        per_phoneme=per_phoneme,
    )
