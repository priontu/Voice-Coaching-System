"""
metrics.py - Evaluation metrics for the VAD + pitch pipeline.

Complements pitch_score.py (which provides PitchAcc50, MACE, PitchRMSE,
NotePitchAcc50) by adding VAD-specific metrics and flexible standalone
wrappers that work with either numpy arrays or the pipeline output dict.

All frequency inputs are expected in Hz. All error outputs are in cents
unless otherwise noted.

Metrics provided here:
  voiced_unvoiced_accuracy()  — requires a reference voiced mask
  pitch_rmse_cents()          — root-mean-square cent error (voiced frames)
  mean_absolute_cents_error() — MACE over voiced frames
  pitch_accuracy_threshold()  — fraction within ±N cents
  evaluate_all()              — single call returning all metrics as a dict
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VAD / voiced-detection metrics
# ---------------------------------------------------------------------------

def voiced_unvoiced_accuracy(
    predicted_voiced: np.ndarray,
    reference_voiced: np.ndarray,
) -> Dict[str, float]:
    """
    Compute voiced/unvoiced classification accuracy against a reference mask.

    Requires a ground-truth reference voiced mask (e.g. from manual annotation
    or a high-quality baseline system).

    Args:
        predicted_voiced: Boolean array, shape (T,). Pipeline output voiced mask.
        reference_voiced: Boolean array, shape (T,). Ground-truth voiced mask.

    Returns:
        Dict with keys:
          "accuracy"   — overall frame accuracy
          "precision"  — precision of voiced detection
          "recall"     — recall of voiced detection (voiced frame recall)
          "f1"         — F1 score
          "false_alarm_rate" — fraction of unvoiced frames incorrectly flagged
          "miss_rate"        — fraction of voiced frames incorrectly missed
    """
    pred = predicted_voiced.astype(bool)
    ref = reference_voiced.astype(bool)

    T = len(pred)
    if T == 0:
        return {k: 0.0 for k in ("accuracy", "precision", "recall", "f1",
                                  "false_alarm_rate", "miss_rate")}

    tp = float(np.sum(pred & ref))
    tn = float(np.sum(~pred & ~ref))
    fp = float(np.sum(pred & ~ref))
    fn = float(np.sum(~pred & ref))

    accuracy = (tp + tn) / T
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
    }


# ---------------------------------------------------------------------------
# Pitch accuracy metrics (frame-level)
# ---------------------------------------------------------------------------

def pitch_rmse_cents(
    predicted_f0: np.ndarray,
    reference_f0: np.ndarray,
    voiced_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Root-mean-square error in cents between predicted and reference F0.

    Only frames where BOTH predicted and reference are voiced (>0) are included.

    Args:
        predicted_f0: Predicted F0 in Hz, shape (T,). Unvoiced = 0.
        reference_f0: Reference F0 in Hz, shape (T,). Unvoiced = 0.
        voiced_mask: Optional additional voiced mask. If provided, frames that
                     are unvoiced according to this mask are also excluded.

    Returns:
        RMSE in cents. Returns NaN if no voiced frames overlap.
    """
    mask = (predicted_f0 > 0) & (reference_f0 > 0)
    if voiced_mask is not None:
        mask &= voiced_mask

    if not np.any(mask):
        logger.warning("[Metrics] pitch_rmse_cents: no overlapping voiced frames.")
        return float("nan")

    cents = 1200.0 * np.log2(predicted_f0[mask] / reference_f0[mask])
    return float(np.sqrt(np.mean(cents ** 2)))


def mean_absolute_cents_error(
    predicted_f0: np.ndarray,
    reference_f0: np.ndarray,
    voiced_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Mean absolute cent error (MACE) between predicted and reference F0.

    Args:
        predicted_f0: Predicted F0 in Hz, shape (T,).
        reference_f0: Reference F0 in Hz, shape (T,).
        voiced_mask: Optional boolean mask to restrict evaluation frames.

    Returns:
        MACE in cents. Returns NaN if no voiced frames overlap.
    """
    mask = (predicted_f0 > 0) & (reference_f0 > 0)
    if voiced_mask is not None:
        mask &= voiced_mask

    if not np.any(mask):
        logger.warning("[Metrics] mean_absolute_cents_error: no overlapping voiced frames.")
        return float("nan")

    cents = 1200.0 * np.log2(predicted_f0[mask] / reference_f0[mask])
    return float(np.mean(np.abs(cents)))


def pitch_accuracy_threshold(
    predicted_f0: np.ndarray,
    reference_f0: np.ndarray,
    voiced_mask: Optional[np.ndarray] = None,
    threshold_cents: float = 50.0,
) -> float:
    """
    Fraction of voiced frames where the cent error is within ±threshold_cents.

    Equivalent to PitchAcc50 in pitch_score.py when threshold_cents=50.

    Args:
        predicted_f0: Predicted F0 in Hz, shape (T,).
        reference_f0: Reference F0 in Hz, shape (T,).
        voiced_mask: Optional additional boolean restriction mask.
        threshold_cents: Acceptance window in cents (default: 50).

    Returns:
        Accuracy in [0, 1]. Returns NaN if no voiced frames overlap.
    """
    mask = (predicted_f0 > 0) & (reference_f0 > 0)
    if voiced_mask is not None:
        mask &= voiced_mask

    if not np.any(mask):
        logger.warning("[Metrics] pitch_accuracy_threshold: no overlapping voiced frames.")
        return float("nan")

    abs_cents = np.abs(1200.0 * np.log2(predicted_f0[mask] / reference_f0[mask]))
    return float(np.mean(abs_cents <= threshold_cents))


def raw_octave_error_rate(
    predicted_f0: np.ndarray,
    reference_f0: np.ndarray,
    voiced_mask: Optional[np.ndarray] = None,
    octave_threshold_cents: float = 550.0,
) -> float:
    """
    Fraction of voiced frames with a gross octave error (>550 cents off).

    Octave errors are a common failure mode in pitch estimators. Detecting
    them separately helps distinguish tuning errors from algorithm failures.

    Args:
        predicted_f0: Predicted F0 in Hz, shape (T,).
        reference_f0: Reference F0 in Hz, shape (T,).
        voiced_mask: Optional boolean restriction mask.
        octave_threshold_cents: Minimum error size (in cents) to count as octave error.

    Returns:
        Octave error rate in [0, 1].
    """
    mask = (predicted_f0 > 0) & (reference_f0 > 0)
    if voiced_mask is not None:
        mask &= voiced_mask

    if not np.any(mask):
        return float("nan")

    abs_cents = np.abs(1200.0 * np.log2(predicted_f0[mask] / reference_f0[mask]))
    return float(np.mean(abs_cents >= octave_threshold_cents))


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    predicted_f0: np.ndarray,
    reference_f0: np.ndarray,
    predicted_voiced: np.ndarray,
    reference_voiced: Optional[np.ndarray] = None,
    thresholds_cents: Tuple[float, ...] = (25.0, 50.0, 100.0),
) -> Dict:
    """
    Run all metrics in one call and return a consolidated result dict.

    This is the recommended entry point for evaluation scripts. The returned
    dict is fully JSON-serializable.

    Args:
        predicted_f0: Predicted F0 in Hz, shape (T,).
        reference_f0: Reference F0 in Hz, shape (T,).
        predicted_voiced: Boolean voiced mask from the pipeline, shape (T,).
        reference_voiced: Optional ground-truth voiced mask for VAD evaluation.
        thresholds_cents: Accuracy thresholds to evaluate (default: 25, 50, 100 cents).

    Returns:
        Dict with keys:
          "voiced_frames"        — number of overlapping voiced frames
          "MACE"                 — mean absolute cent error
          "PitchRMSE"            — RMSE in cents
          "OctaveErrorRate"      — fraction of frames with gross octave errors
          "PitchAcc{N}"          — accuracy at each threshold N cents
          "VAD"                  — dict from voiced_unvoiced_accuracy() (if ref provided)
    """
    results: Dict = {}

    # Count overlapping voiced frames
    voiced_overlap = (predicted_f0 > 0) & (reference_f0 > 0)
    results["voiced_frames"] = int(np.sum(voiced_overlap))

    if results["voiced_frames"] == 0:
        logger.warning("[Metrics] evaluate_all: no overlapping voiced frames — all metrics NaN.")
        results["MACE"] = None
        results["PitchRMSE"] = None
        results["OctaveErrorRate"] = None
        for t in thresholds_cents:
            results[f"PitchAcc{int(t)}"] = None
        if reference_voiced is not None:
            results["VAD"] = {}
        return results

    results["MACE"] = mean_absolute_cents_error(predicted_f0, reference_f0)
    results["PitchRMSE"] = pitch_rmse_cents(predicted_f0, reference_f0)
    results["OctaveErrorRate"] = raw_octave_error_rate(predicted_f0, reference_f0)

    for t in thresholds_cents:
        key = f"PitchAcc{int(t)}"
        results[key] = pitch_accuracy_threshold(predicted_f0, reference_f0, threshold_cents=t)

    if reference_voiced is not None:
        results["VAD"] = voiced_unvoiced_accuracy(predicted_voiced, reference_voiced)

    logger.info(
        f"[Metrics] MACE={results['MACE']:.2f}¢  "
        f"RMSE={results['PitchRMSE']:.2f}¢  "
        f"PitchAcc50={results.get('PitchAcc50', float('nan')):.1%}"
    )

    return results


# ---------------------------------------------------------------------------
# Convenience: print a formatted metrics report
# ---------------------------------------------------------------------------

def print_metrics_report(results: Dict) -> None:
    """Print a formatted evaluation report to stdout."""
    print("\nEvaluation Metrics")
    print("=" * 40)
    print(f"  Voiced frames compared : {results.get('voiced_frames', 'N/A')}")

    for key in ("MACE", "PitchRMSE"):
        val = results.get(key)
        unit = "cents"
        print(f"  {key:<26}: {val:.2f} {unit}" if val is not None else f"  {key}: N/A")

    for key, val in results.items():
        if key.startswith("PitchAcc"):
            pct = f"{val*100:.2f}%" if val is not None else "N/A"
            print(f"  {key:<26}: {pct}")

    oe = results.get("OctaveErrorRate")
    if oe is not None:
        print(f"  {'OctaveErrorRate':<26}: {oe*100:.2f}%")

    vad = results.get("VAD")
    if vad:
        print("\n  VAD Classification")
        print("  " + "-" * 30)
        for k, v in vad.items():
            print(f"  {k:<26}: {v:.4f}")

    print("=" * 40)
