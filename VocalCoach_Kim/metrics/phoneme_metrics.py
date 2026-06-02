"""
metrics/phoneme_metrics.py - Boundary detection metrics for phoneme segments.

Extracted from the original Phoneme Model/phoneme_model.py and extended with
additional reporting helpers. Core algorithm is unchanged.
"""

from __future__ import annotations

from typing import Dict, List

from utils.types import PhonemeSegment


def compute_boundary_metrics(
    predicted: List[PhonemeSegment],
    reference: List[PhonemeSegment],
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """
    Compute precision / recall / F1 / MAE against reference boundaries.

    Boundaries are start times of all segments plus the end time of the last
    segment. A predicted boundary matches if it falls within tolerance_ms of
    any reference boundary.

    Args:
        predicted:     Predicted PhonemeSegment list.
        reference:     Reference PhonemeSegment list.
        tolerance_ms:  Matching tolerance in milliseconds.

    Returns:
        Dict with keys: precision, recall, f1, mae_ms, matches, total_boundaries.
    """
    if not predicted or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mae_ms": 0.0}

    tol = tolerance_ms / 1000.0
    pred_b = [s.start_time for s in predicted] + [predicted[-1].end_time]
    ref_b = [s.start_time for s in reference] + [reference[-1].end_time]

    matches = 0
    total_error = 0.0
    for pt in pred_b:
        nearest = min(abs(pt - rt) for rt in ref_b)
        if nearest <= tol:
            matches += 1
            total_error += nearest

    precision = matches / len(pred_b) if pred_b else 0.0
    recall = matches / len(ref_b) if ref_b else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    mae_ms = (total_error / matches * 1000) if matches > 0 else float("inf")

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mae_ms": round(mae_ms, 2),
        "matches": matches,
        "total_boundaries": len(pred_b),
    }


def compute_phoneme_accuracy(
    predicted: List[PhonemeSegment],
    reference: List[PhonemeSegment],
    tolerance_ms: float = 50.0,
) -> float:
    """
    Fraction of predicted phonemes that match a reference phoneme (same label
    AND within tolerance_ms of the reference boundary).

    Returns:
        Accuracy in [0, 1].
    """
    if not predicted or not reference:
        return 0.0

    tol = tolerance_ms / 1000.0
    ref_used = set()
    correct = 0

    for pred in predicted:
        for i, ref in enumerate(reference):
            if i in ref_used:
                continue
            if pred.phoneme == ref.phoneme and abs(pred.start_time - ref.start_time) <= tol:
                correct += 1
                ref_used.add(i)
                break

    return correct / len(predicted)
