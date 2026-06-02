"""
metrics/pitch_metrics.py - Frame-level and note-level pitch metrics.

Phase 1 functions (compute_frame_pitch_metrics, compute_note_level_pitch_correctness)
use numpy arrays and legacy note dicts; preserved for backward compatibility.

Phase 6 functions (compute_pitch_accuracy, compute_pitch_rmse, compute_mace,
compute_pitch_stability, compute_note_pitch_accuracy, build_pitch_metrics) work
with AlignmentResult and the typed structures from utils/types.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_frame_pitch_metrics(
    times: np.ndarray,
    predicted_f0: np.ndarray,
    reference_f0: np.ndarray,
    tolerance_cents: float = 50.0,
) -> Optional[Dict]:
    """
    Frame-level pitch accuracy metrics.

    Compares predicted vs. reference F0 only on frames where both are voiced
    (> 0 Hz).

    Args:
        times:          Frame timestamps in seconds, shape (T,).
        predicted_f0:   Predicted F0 in Hz, shape (T,). 0 = unvoiced.
        reference_f0:   Reference F0 in Hz, shape (T,). 0 = unvoiced.
        tolerance_cents: Accuracy threshold in cents.

    Returns:
        Dict with PitchAcc, MACE, PitchRMSE, VoicedComparedFrames,
        WorstMismatches. None if no voiced overlap.
    """
    voiced_mask = (predicted_f0 > 0) & (reference_f0 > 0)

    if np.sum(voiced_mask) == 0:
        return None

    compared_times = times[voiced_mask]
    predicted = predicted_f0[voiced_mask]
    reference = reference_f0[voiced_mask]

    cent_error = 1200 * np.log2(predicted / reference)
    abs_cent_error = np.abs(cent_error)

    pitch_acc = np.mean(abs_cent_error <= tolerance_cents) * 100
    mace = np.mean(abs_cent_error)
    rmse = np.sqrt(np.mean(cent_error ** 2))

    # Two most spatially separated worst mismatches
    sorted_indices = np.argsort(abs_cent_error)[::-1]
    worst_mismatches = []
    min_time_gap = 0.30

    for idx in sorted_indices:
        t = compared_times[idx]
        too_close = any(abs(t - item["time"]) < min_time_gap for item in worst_mismatches)
        if too_close:
            continue

        worst_mismatches.append({
            "time": float(t),
            "predicted_hz": float(predicted[idx]),
            "reference_hz": float(reference[idx]),
            "predicted_midi": float(_hz_to_midi(predicted[idx])),
            "reference_midi": float(_hz_to_midi(reference[idx])),
            "cent_error": float(cent_error[idx]),
            "abs_cent_error": float(abs_cent_error[idx]),
            "direction": "sharp" if cent_error[idx] > 0 else "flat",
        })

        if len(worst_mismatches) == 2:
            break

    return {
        f"PitchAcc{int(tolerance_cents)}": pitch_acc,
        "MACE": mace,
        "PitchRMSE": rmse,
        "VoicedComparedFrames": int(np.sum(voiced_mask)),
        "WorstMismatches": worst_mismatches,
    }


def compute_note_level_pitch_correctness(
    times: np.ndarray,
    predicted_f0: np.ndarray,
    notes: List[Dict],
    seconds_per_beat: float,
    tolerance_cents: float = 50.0,
    min_frames_per_note: int = 3,
) -> Optional[Dict]:
    """
    Note-level pitch correctness metrics.

    Each note is scored by the median cent error of voiced frames within its
    duration window.

    Args:
        times:             Frame timestamps, shape (T,).
        predicted_f0:      Predicted F0 in Hz, shape (T,).
        notes:             List of note dicts from load_musicxml_notes().
        seconds_per_beat:  Seconds per quarter note (from tempo).
        tolerance_cents:   Correctness tolerance in cents.
        min_frames_per_note: Minimum voiced frames to score a note.

    Returns:
        Dict with NotePitchAcc, ScoredNotes, NoteResults, WorstNotes.
        None if no notes were scored.
    """
    note_results = []

    for note_index, note in enumerate(notes, start=1):
        start_sec = note["start_beat"] * seconds_per_beat
        end_sec = note["end_beat"] * seconds_per_beat

        note_mask = (times >= start_sec) & (times < end_sec) & (predicted_f0 > 0)
        note_f0 = predicted_f0[note_mask]

        if len(note_f0) < min_frames_per_note:
            note_results.append({
                "note_index": note_index,
                "scored": False,
                "reason": "not enough voiced frames",
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "reference_hz": float(note["hz"]),
                "lyric": note.get("lyric"),
            })
            continue

        cent_errors = 1200 * np.log2(note_f0 / note["hz"])
        median_cent_error = np.median(cent_errors)
        abs_error = abs(median_cent_error)
        is_correct = abs_error <= tolerance_cents

        note_results.append({
            "note_index": note_index,
            "scored": True,
            "correct": bool(is_correct),
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "duration_sec": float(end_sec - start_sec),
            "reference_hz": float(note["hz"]),
            "reference_midi": float(note["midi"]),
            "median_detected_hz": float(np.median(note_f0)),
            "median_detected_midi": float(_hz_to_midi(np.median(note_f0))),
            "median_cent_error": float(median_cent_error),
            "abs_median_cent_error": float(abs_error),
            "direction": "sharp" if median_cent_error > 0 else "flat",
            "num_voiced_frames": int(len(note_f0)),
            "lyric": note.get("lyric"),
        })

    scored = [r for r in note_results if r["scored"]]
    if not scored:
        return None

    correct = [r for r in scored if r["correct"]]
    note_acc = (len(correct) / len(scored)) * 100
    worst = sorted(scored, key=lambda r: r["abs_median_cent_error"], reverse=True)[:2]

    return {
        f"NotePitchAcc{int(tolerance_cents)}": note_acc,
        "ScoredNotes": len(scored),
        "CorrectNotes": len(correct),
        "TotalReferenceNotes": len(notes),
        "NoteResults": note_results,
        "WorstNotes": worst,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hz_to_midi(f0_hz: float) -> float:
    return 69 + 12 * np.log2(f0_hz / 440.0)


def _midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12))


# ---------------------------------------------------------------------------
# Phase 6 — AlignmentResult-based pitch metrics
# ---------------------------------------------------------------------------

def compute_pitch_accuracy(
    alignment: "AlignmentResult",
    tolerance_cents: float = 50.0,
) -> Optional[float]:
    """Fraction of matched notes whose pitch deviation is within tolerance_cents."""
    matches = [m for m in alignment.note_matches if m.pitch_deviation_cents is not None]
    if not matches:
        return None
    correct = sum(1 for m in matches if abs(m.pitch_deviation_cents) <= tolerance_cents)
    return correct / len(matches)


def compute_pitch_rmse(alignment: "AlignmentResult") -> Optional[float]:
    """Root-mean-square pitch deviation in cents across all matched notes."""
    devs = [m.pitch_deviation_cents for m in alignment.note_matches
            if m.pitch_deviation_cents is not None]
    if not devs:
        return None
    return math.sqrt(sum(d ** 2 for d in devs) / len(devs))


def compute_mace(alignment: "AlignmentResult") -> Optional[float]:
    """Mean Absolute Cent Error across all matched notes."""
    devs = [m.pitch_deviation_cents for m in alignment.note_matches
            if m.pitch_deviation_cents is not None]
    if not devs:
        return None
    return sum(abs(d) for d in devs) / len(devs)


def compute_pitch_stability(
    fused: "FusedPerformanceRepresentation",
) -> Optional[float]:
    """Mean pitch_stability (intra-note F0 std) across all fused note events."""
    stabs = [n.pitch_stability for n in fused.note_events
             if n.pitch_stability is not None]
    if not stabs:
        return None
    return sum(stabs) / len(stabs)


def compute_note_pitch_accuracy(
    alignment: "AlignmentResult",
    fused: "FusedPerformanceRepresentation",
    reference: "ReferencePerformanceRepresentation",
    tolerance_cents: float = 50.0,
) -> Tuple[Optional[float], List["MetricBreakdown"]]:
    """
    Per-note pitch accuracy with breakdown.

    Returns (accuracy, per_note_breakdowns). accuracy is None when no note
    has a valid pitch deviation.
    """
    from utils.types import MetricBreakdown
    breakdowns: List[MetricBreakdown] = []
    with_pitch = [m for m in alignment.note_matches if m.pitch_deviation_cents is not None]
    if not with_pitch:
        return None, breakdowns

    correct = 0
    for m in with_pitch:
        dev = m.pitch_deviation_cents
        is_ok = abs(dev) <= tolerance_cents
        if is_ok:
            correct += 1
        label = None
        if m.ref_idx < len(reference.notes):
            label = reference.notes[m.ref_idx].pitch_name
        breakdowns.append(MetricBreakdown(
            event_idx=m.pred_idx,
            value=dev,
            label=label,
            metadata={"correct": is_ok, "ref_idx": m.ref_idx},
        ))

    return correct / len(with_pitch), breakdowns


def build_pitch_metrics(
    alignment: "AlignmentResult",
    fused: Optional["FusedPerformanceRepresentation"] = None,
    reference: Optional["ReferencePerformanceRepresentation"] = None,
    tolerance_cents: float = 50.0,
) -> "PitchMetrics":
    """Compute all pitch metrics and return a PitchMetrics dataclass."""
    from utils.types import MetricBreakdown, PitchMetrics

    devs = [m.pitch_deviation_cents for m in alignment.note_matches
            if m.pitch_deviation_cents is not None]

    accuracy = compute_pitch_accuracy(alignment, tolerance_cents)
    rmse = compute_pitch_rmse(alignment)
    mace = compute_mace(alignment)
    mean_dev = sum(devs) / len(devs) if devs else None

    # Per-note breakdown — richer when fused + reference are available
    breakdowns: List[MetricBreakdown] = []
    if fused is not None and reference is not None:
        _, breakdowns = compute_note_pitch_accuracy(alignment, fused, reference, tolerance_cents)
    else:
        for m in alignment.note_matches:
            if m.pitch_deviation_cents is not None:
                breakdowns.append(MetricBreakdown(
                    event_idx=m.pred_idx,
                    value=m.pitch_deviation_cents,
                    metadata={"ref_idx": m.ref_idx},
                ))

    stability = compute_pitch_stability(fused) if fused is not None else None

    return PitchMetrics(
        pitch_accuracy=accuracy,
        pitch_rmse_cents=rmse,
        mace_cents=mace,
        note_pitch_accuracy=accuracy,
        mean_pitch_deviation_cents=mean_dev,
        n_evaluated=len(devs),
        tolerance_cents=tolerance_cents,
        per_note=breakdowns,
        metadata={"mean_pitch_stability": stability},
    )


# Type hints for forward references (resolved at import time via TYPE_CHECKING)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.types import (
        AlignmentResult,
        FusedPerformanceRepresentation,
        MetricBreakdown,
        PerformanceMetricsReport,
        PitchMetrics,
        ReferencePerformanceRepresentation,
    )
