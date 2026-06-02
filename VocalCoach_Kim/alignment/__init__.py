"""
alignment/ - Prediction ↔ reference alignment engine (Phase 5).

Provides deterministic, threshold-configurable alignment of predicted
VocalCoach events against ground-truth reference annotations.

Modules:
  alignment_utils      - Pure temporal/pitch utility functions
  reference_alignment  - High-level alignment orchestration
"""

from alignment.alignment_utils import (
    iou,
    nearest_match,
    offset_deviation,
    onset_deviation,
    overlap_duration,
    overlap_fraction_of_a,
    pitch_deviation_cents,
    pitch_deviation_semitones,
)
from alignment.reference_alignment import (
    align_notes,
    align_performance,
    align_phonemes,
    align_words,
)

__all__ = [
    # utilities
    "iou",
    "nearest_match",
    "offset_deviation",
    "onset_deviation",
    "overlap_duration",
    "overlap_fraction_of_a",
    "pitch_deviation_cents",
    "pitch_deviation_semitones",
    # alignment
    "align_notes",
    "align_performance",
    "align_phonemes",
    "align_words",
]
