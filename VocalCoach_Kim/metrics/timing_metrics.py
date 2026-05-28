"""
metrics/timing_metrics.py - Onset, offset, and inter-onset interval timing metrics.

All functions are stateless and accept typed structures from utils/types.py.
Time values are stored internally in seconds; reported outputs convert to ms
for human readability unless stated otherwise.

Deviation sign convention (matches alignment_utils.py):
  positive = predicted is LATER than reference (early singing is negative).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.types import (
        AlignmentResult,
        FusedPerformanceRepresentation,
        MetricBreakdown,
        NoteEvent,
        ReferenceNote,
        ReferencePerformanceRepresentation,
        TimingMetrics,
    )


def compute_onset_error(
    alignment: "AlignmentResult",
) -> Dict[str, Optional[float]]:
    """
    Onset timing statistics across all note matches.

    Returns dict with: mean_onset_error_ms, std_onset_error_ms,
    mean_abs_onset_error_ms, median_onset_error_ms.
    All values in milliseconds; None when no matches exist.
    """
    devs_ms = [m.onset_deviation_s * 1000.0 for m in alignment.note_matches]
    if not devs_ms:
        return {
            "mean_onset_error_ms": None,
            "std_onset_error_ms": None,
            "mean_abs_onset_error_ms": None,
            "median_onset_error_ms": None,
        }
    n = len(devs_ms)
    mean = sum(devs_ms) / n
    mean_abs = sum(abs(d) for d in devs_ms) / n
    variance = sum((d - mean) ** 2 for d in devs_ms) / n
    std = math.sqrt(variance)
    sorted_devs = sorted(devs_ms)
    if n % 2 == 0:
        median = (sorted_devs[n // 2 - 1] + sorted_devs[n // 2]) / 2.0
    else:
        median = sorted_devs[n // 2]
    return {
        "mean_onset_error_ms": mean,
        "std_onset_error_ms": std,
        "mean_abs_onset_error_ms": mean_abs,
        "median_onset_error_ms": median,
    }


def compute_offset_error(
    alignment: "AlignmentResult",
) -> Dict[str, Optional[float]]:
    """
    Offset timing statistics across all note matches.

    Returns dict with: mean_offset_error_ms, mean_abs_offset_error_ms.
    All values in milliseconds; None when no matches exist.
    """
    devs_ms = [m.offset_deviation_s * 1000.0 for m in alignment.note_matches]
    if not devs_ms:
        return {"mean_offset_error_ms": None, "mean_abs_offset_error_ms": None}
    n = len(devs_ms)
    mean = sum(devs_ms) / n
    mean_abs = sum(abs(d) for d in devs_ms) / n
    return {"mean_offset_error_ms": mean, "mean_abs_offset_error_ms": mean_abs}


def compute_timing_accuracy(
    alignment: "AlignmentResult",
    tolerance_ms: float = 50.0,
) -> Optional[float]:
    """
    Fraction of matched notes whose onset deviation is within tolerance_ms.

    Returns None when no note matches exist.
    """
    if not alignment.note_matches:
        return None
    within = sum(
        1 for m in alignment.note_matches
        if abs(m.onset_deviation_s * 1000.0) <= tolerance_ms
    )
    return within / len(alignment.note_matches)


def compute_ioi_deviation(
    fused_notes: List["NoteEvent"],
    ref_notes: List["ReferenceNote"],
) -> Optional[float]:
    """
    Mean Absolute Error of inter-onset intervals (IOI) in milliseconds.

    Aligns predicted and reference IOI sequences by position (first n-1 pairs)
    where n is the length of the shorter sequence.

    Returns None when fewer than two notes exist on either side.
    """
    pred_onsets = sorted(n.onset_time for n in fused_notes if n.onset_time is not None)
    ref_onsets = sorted(n.onset_time for n in ref_notes if not n.is_rest and n.onset_time is not None)

    pred_ioi = [pred_onsets[i + 1] - pred_onsets[i] for i in range(len(pred_onsets) - 1)]
    ref_ioi = [ref_onsets[i + 1] - ref_onsets[i] for i in range(len(ref_onsets) - 1)]

    n = min(len(pred_ioi), len(ref_ioi))
    if n == 0:
        return None

    errors_ms = [abs(pred_ioi[i] - ref_ioi[i]) * 1000.0 for i in range(n)]
    return sum(errors_ms) / n


def build_timing_metrics(
    alignment: "AlignmentResult",
    fused: Optional["FusedPerformanceRepresentation"] = None,
    reference: Optional["ReferencePerformanceRepresentation"] = None,
    tolerance_ms: float = 50.0,
) -> "TimingMetrics":
    """Compute all timing metrics and return a TimingMetrics dataclass."""
    from utils.types import MetricBreakdown, TimingMetrics

    onset_stats = compute_onset_error(alignment)
    offset_stats = compute_offset_error(alignment)
    ta = compute_timing_accuracy(alignment, tolerance_ms)

    ioi_mae: Optional[float] = None
    if fused is not None and reference is not None:
        ioi_mae = compute_ioi_deviation(fused.note_events, reference.notes)

    per_note = [
        MetricBreakdown(
            event_idx=m.pred_idx,
            value=m.onset_deviation_s * 1000.0,
            metadata={
                "ref_idx": m.ref_idx,
                "offset_dev_ms": m.offset_deviation_s * 1000.0,
            },
        )
        for m in alignment.note_matches
    ]

    return TimingMetrics(
        mean_onset_error_ms=onset_stats["mean_onset_error_ms"],
        std_onset_error_ms=onset_stats["std_onset_error_ms"],
        mean_abs_onset_error_ms=onset_stats["mean_abs_onset_error_ms"],
        median_onset_error_ms=onset_stats["median_onset_error_ms"],
        mean_offset_error_ms=offset_stats["mean_offset_error_ms"],
        mean_abs_offset_error_ms=offset_stats["mean_abs_offset_error_ms"],
        timing_accuracy=ta,
        ioi_mae_ms=ioi_mae,
        n_evaluated=len(alignment.note_matches),
        tolerance_ms=tolerance_ms,
        per_note=per_note,
    )
