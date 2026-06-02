"""
metrics/duration_metrics.py - Note duration comparison metrics.

Evaluates how well predicted note durations match reference durations.
All duration values are in seconds internally; ratios are dimensionless.

Sign convention: error = predicted_duration - reference_duration.
Positive error = singer held note longer than reference.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.types import (
        AlignmentResult,
        DurationMetrics,
        FusedPerformanceRepresentation,
        MetricBreakdown,
        NoteEvent,
        ReferenceNote,
        ReferencePerformanceRepresentation,
    )


def _gather_duration_pairs(
    alignment: "AlignmentResult",
    fused_notes: List["NoteEvent"],
    ref_notes: List["ReferenceNote"],
) -> List[Dict]:
    """Build list of (pred_idx, ref_idx, pred_dur, ref_dur) dicts for valid matches."""
    pairs = []
    for m in alignment.note_matches:
        pred = fused_notes[m.pred_idx] if m.pred_idx < len(fused_notes) else None
        ref = ref_notes[m.ref_idx] if m.ref_idx < len(ref_notes) else None
        if pred is None or ref is None:
            continue
        pred_dur = pred.duration
        ref_dur = ref.duration
        if pred_dur is None or ref_dur is None or ref_dur <= 0:
            continue
        pairs.append({
            "pred_idx": m.pred_idx,
            "ref_idx": m.ref_idx,
            "pred_dur": pred_dur,
            "ref_dur": ref_dur,
        })
    return pairs


def compute_duration_error(
    alignment: "AlignmentResult",
    fused_notes: List["NoteEvent"],
    ref_notes: List["ReferenceNote"],
) -> Dict[str, Optional[float]]:
    """
    Duration error statistics across matched notes.

    Returns dict with: mean_duration_error_s, mean_abs_duration_error_s,
    std_duration_error_s. All values in seconds; None if no valid pairs.
    """
    pairs = _gather_duration_pairs(alignment, fused_notes, ref_notes)
    if not pairs:
        return {
            "mean_duration_error_s": None,
            "mean_abs_duration_error_s": None,
            "std_duration_error_s": None,
        }
    errors = [p["pred_dur"] - p["ref_dur"] for p in pairs]
    n = len(errors)
    mean = sum(errors) / n
    mean_abs = sum(abs(e) for e in errors) / n
    variance = sum((e - mean) ** 2 for e in errors) / n
    std = math.sqrt(variance)
    return {
        "mean_duration_error_s": mean,
        "mean_abs_duration_error_s": mean_abs,
        "std_duration_error_s": std,
    }


def compute_duration_ratio(
    alignment: "AlignmentResult",
    fused_notes: List["NoteEvent"],
    ref_notes: List["ReferenceNote"],
) -> Optional[float]:
    """
    Mean ratio of predicted to reference duration (pred_dur / ref_dur).

    1.0 = perfect; > 1.0 = singer stretched notes; < 1.0 = cut short.
    Returns None if no valid pairs.
    """
    pairs = _gather_duration_pairs(alignment, fused_notes, ref_notes)
    if not pairs:
        return None
    ratios = [p["pred_dur"] / p["ref_dur"] for p in pairs]
    return sum(ratios) / len(ratios)


def compute_relative_duration_error(
    alignment: "AlignmentResult",
    fused_notes: List["NoteEvent"],
    ref_notes: List["ReferenceNote"],
) -> Optional[float]:
    """
    Mean relative duration error: mean(|pred_dur - ref_dur| / ref_dur).

    Tempo-independent: scales errors by the reference note duration so long
    and short notes contribute equally. Returns None if no valid pairs.
    """
    pairs = _gather_duration_pairs(alignment, fused_notes, ref_notes)
    if not pairs:
        return None
    rel_errors = [abs(p["pred_dur"] - p["ref_dur"]) / p["ref_dur"] for p in pairs]
    return sum(rel_errors) / len(rel_errors)


def build_duration_metrics(
    alignment: "AlignmentResult",
    fused: Optional["FusedPerformanceRepresentation"] = None,
    reference: Optional["ReferencePerformanceRepresentation"] = None,
) -> "DurationMetrics":
    """Compute all duration metrics and return a DurationMetrics dataclass."""
    from utils.types import DurationMetrics, MetricBreakdown

    fused_notes: List = fused.note_events if fused is not None else []
    ref_notes: List = reference.notes if reference is not None else []

    dur_stats = compute_duration_error(alignment, fused_notes, ref_notes)
    ratio = compute_duration_ratio(alignment, fused_notes, ref_notes)
    rel_err = compute_relative_duration_error(alignment, fused_notes, ref_notes)

    pairs = _gather_duration_pairs(alignment, fused_notes, ref_notes)
    per_note = [
        MetricBreakdown(
            event_idx=p["pred_idx"],
            value=p["pred_dur"] - p["ref_dur"],
            metadata={
                "pred_dur_s": p["pred_dur"],
                "ref_dur_s": p["ref_dur"],
                "ratio": p["pred_dur"] / p["ref_dur"],
                "ref_idx": p["ref_idx"],
            },
        )
        for p in pairs
    ]

    return DurationMetrics(
        mean_duration_error_s=dur_stats["mean_duration_error_s"],
        mean_abs_duration_error_s=dur_stats["mean_abs_duration_error_s"],
        std_duration_error_s=dur_stats["std_duration_error_s"],
        mean_duration_ratio=ratio,
        mean_relative_duration_error=rel_err,
        n_evaluated=len(pairs),
        per_note=per_note,
    )
