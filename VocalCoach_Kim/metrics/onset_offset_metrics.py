"""
metrics/onset_offset_metrics.py - Precision/recall/F1 and duration MAE for note detection.

Unchanged from the original Note Model/metrics.py — moved here for shared access.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _match_events(
    predicted: List[float],
    reference: List[float],
    tolerance_s: float,
) -> Tuple[int, int, int]:
    """Greedy matching of predicted events to reference events within tolerance_s."""
    matched_ref: set = set()
    tp = 0

    for p in sorted(predicted):
        best_i, best_d = -1, float("inf")
        for i, r in enumerate(sorted(reference)):
            d = abs(p - r)
            if d <= tolerance_s and i not in matched_ref and d < best_d:
                best_i, best_d = i, d
        if best_i >= 0:
            tp += 1
            matched_ref.add(best_i)

    fp = len(predicted) - tp
    fn = len(reference) - tp
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _mae_ms(
    predicted: List[float],
    reference: List[float],
    tolerance_s: float,
) -> float:
    matched_ref: set = set()
    errors: List[float] = []
    ref_sorted = sorted(reference)

    for p in sorted(predicted):
        best_i, best_d = -1, float("inf")
        for i, r in enumerate(ref_sorted):
            d = abs(p - r)
            if d <= tolerance_s and i not in matched_ref and d < best_d:
                best_i, best_d = i, d
        if best_i >= 0:
            errors.append(best_d * 1000.0)
            matched_ref.add(best_i)

    return float(np.mean(errors)) if errors else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def onset_metrics(
    predicted_onsets: List[float],
    reference_onsets: List[float],
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """Precision, recall, F1, and MAE for onset detection."""
    tol = tolerance_ms / 1000.0
    tp, fp, fn = _match_events(predicted_onsets, reference_onsets, tol)
    prec, rec, f1 = _prf(tp, fp, fn)
    return {
        "onset_precision": prec,
        "onset_recall": rec,
        "onset_f1": f1,
        "onset_mae_ms": _mae_ms(predicted_onsets, reference_onsets, tol),
        "onset_tp": tp, "onset_fp": fp, "onset_fn": fn,
    }


def offset_metrics(
    predicted_offsets: List[float],
    reference_offsets: List[float],
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """Precision, recall, F1, and MAE for offset detection."""
    tol = tolerance_ms / 1000.0
    tp, fp, fn = _match_events(predicted_offsets, reference_offsets, tol)
    prec, rec, f1 = _prf(tp, fp, fn)
    return {
        "offset_precision": prec,
        "offset_recall": rec,
        "offset_f1": f1,
        "offset_mae_ms": _mae_ms(predicted_offsets, reference_offsets, tol),
        "offset_tp": tp, "offset_fp": fp, "offset_fn": fn,
    }


def duration_metrics(
    predicted_notes: List[dict],
    reference_notes: List[dict],
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """Duration MAE and relative duration MAE for notes matched on onset."""
    tol = tolerance_ms / 1000.0
    dur_errors: List[float] = []
    rel_errors: List[float] = []
    ref_used: set = set()

    for pred in predicted_notes:
        p_onset = pred.get("onset_time")
        p_dur = pred.get("duration")
        if p_onset is None or p_dur is None:
            continue

        for i, ref in enumerate(reference_notes):
            if i in ref_used:
                continue
            r_onset = ref.get("onset_time", 0.0)
            r_offset = ref.get("offset_time", 0.0)
            r_dur = (r_offset or 0.0) - (r_onset or 0.0)

            if abs(p_onset - r_onset) <= tol:
                dur_errors.append(abs(p_dur - r_dur) * 1000.0)
                if r_dur > 0:
                    rel_errors.append(abs(p_dur - r_dur) / r_dur)
                ref_used.add(i)
                break

    return {
        "duration_mae_ms": float(np.mean(dur_errors)) if dur_errors else 0.0,
        "relative_duration_mae": float(np.mean(rel_errors)) if rel_errors else 0.0,
        "n_matched_notes": len(dur_errors),
    }


def compute_frame_metrics(
    onset_preds: np.ndarray,
    onset_targets: np.ndarray,
    offset_preds: np.ndarray,
    offset_targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Binary frame-level F1 for onset and offset heads (used during training)."""

    def _f1(preds: np.ndarray, targets: np.ndarray) -> float:
        p = (preds >= threshold).astype(np.int32)
        t = (targets >= 0.5).astype(np.int32)
        tp = int(np.sum((p == 1) & (t == 1)))
        fp = int(np.sum((p == 1) & (t == 0)))
        fn = int(np.sum((p == 0) & (t == 1)))
        _, _, f1 = _prf(tp, fp, fn)
        return f1

    return {
        "onset_f1": _f1(onset_preds, onset_targets),
        "offset_f1": _f1(offset_preds, offset_targets),
    }


def evaluate_file(
    predicted_notes: List[dict],
    reference_notes: List[dict],
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """Full evaluation of detected notes against ground truth for one file."""
    pred_on = [n["onset_time"] for n in predicted_notes if n.get("onset_time") is not None]
    pred_off = [n["offset_time"] for n in predicted_notes if n.get("offset_time") is not None]
    ref_on = [n["onset_time"] for n in reference_notes]
    ref_off = [n["offset_time"] for n in reference_notes]

    out: Dict[str, float] = {}
    out.update(onset_metrics(pred_on, ref_on, tolerance_ms))
    out.update(offset_metrics(pred_off, ref_off, tolerance_ms))
    out.update(duration_metrics(predicted_notes, reference_notes, tolerance_ms))
    return out
