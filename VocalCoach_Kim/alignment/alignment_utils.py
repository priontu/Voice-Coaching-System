"""
alignment/alignment_utils.py - Pure temporal and pitch utility functions.

All functions are stateless, numerically stable, and free from I/O or model
dependencies. They serve as building blocks for the alignment engine in
reference_alignment.py and can be reused by the scoring layer (Phase 6+).

Temporal conventions:
  - All time values are in seconds (float).
  - Overlap is the intersection of two closed intervals [start, end].
  - Deviation sign convention: positive = predicted is LATER/SHARPER than reference.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Temporal overlap
# ---------------------------------------------------------------------------

def overlap_duration(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    """Duration of the intersection of [a_start, a_end] and [b_start, b_end] in seconds."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def overlap_fraction_of_a(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    """Fraction of interval A covered by its overlap with interval B.

    Returns 0.0 when A has zero duration.
    """
    dur_a = a_end - a_start
    if dur_a <= 0.0:
        return 0.0
    return overlap_duration(a_start, a_end, b_start, b_end) / dur_a


def iou(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    """Intersection-over-Union of two intervals.

    Returns 0.0 when the union is zero (both intervals are points or empty).
    """
    inter = overlap_duration(a_start, a_end, b_start, b_end)
    if inter <= 0.0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Temporal deviations
# ---------------------------------------------------------------------------

def onset_deviation(predicted_onset: float, reference_onset: float) -> float:
    """
    Signed onset timing deviation in seconds.

    Positive → predicted onset is LATER than reference (early singing is negative).
    """
    return predicted_onset - reference_onset


def offset_deviation(predicted_offset: float, reference_offset: float) -> float:
    """
    Signed offset timing deviation in seconds.

    Positive → predicted offset is LATER than reference.
    """
    return predicted_offset - reference_offset


# ---------------------------------------------------------------------------
# Nearest-neighbour search
# ---------------------------------------------------------------------------

def nearest_match(
    query_time: float,
    candidate_times: List[float],
) -> Tuple[int, float]:
    """
    Find the index and distance of the nearest value in candidate_times.

    Args:
        query_time:      The query timestamp.
        candidate_times: A list of candidate timestamps (need not be sorted).

    Returns:
        (best_index, absolute_distance_in_seconds)
        Returns (-1, float('inf')) when candidate_times is empty.
    """
    if not candidate_times:
        return -1, float("inf")

    best_idx = 0
    best_dist = abs(candidate_times[0] - query_time)
    for i, t in enumerate(candidate_times[1:], start=1):
        d = abs(t - query_time)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist


# ---------------------------------------------------------------------------
# Pitch utilities
# ---------------------------------------------------------------------------

def pitch_deviation_cents(
    predicted_hz: Optional[float],
    reference_hz: Optional[float],
) -> Optional[float]:
    """
    Signed pitch deviation in cents (100 cents = 1 semitone).

    Positive → predicted pitch is SHARPER than reference.
    Returns None if either input is None or non-positive.
    """
    if predicted_hz is None or reference_hz is None:
        return None
    if predicted_hz <= 0.0 or reference_hz <= 0.0:
        return None
    return 1200.0 * math.log2(predicted_hz / reference_hz)


def pitch_deviation_semitones(
    predicted_hz: Optional[float],
    reference_hz: Optional[float],
) -> Optional[float]:
    """
    Signed pitch deviation in semitones (12 semitones = 1 octave).

    Positive → predicted pitch is SHARPER than reference.
    Returns None if either input is None or non-positive.
    """
    cents = pitch_deviation_cents(predicted_hz, reference_hz)
    return cents / 100.0 if cents is not None else None


def hz_to_midi(hz: float) -> Optional[float]:
    """Convert frequency in Hz to MIDI note number. Returns None for non-positive Hz."""
    if hz <= 0.0:
        return None
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
