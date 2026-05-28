"""
reference/validation.py - Structural validation for reference representations.

Validates ReferencePerformanceRepresentation objects before they are used in
alignment or scoring. Issues are classified as:
  'warning' — non-blocking; alignment proceeds but results may be degraded.
  'error'   — blocking; alignment should be skipped.

Checks performed:
  - Invalid timestamps (negative, NaN, start >= end)
  - Missing phoneme tier (warning)
  - Missing notes tier (warning)
  - Overlapping reference notes of the same pitch (warning)
  - Overlapping reference phonemes (error)
  - Tempo consistency (notes vs. score duration)
  - Malformed entries (zero-duration notes/phonemes)
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from fusion.validation import ValidationIssue, ValidationReport
from utils.types import (
    ReferenceNote,
    ReferencePerformanceRepresentation,
    ReferencePhoneme,
    ReferenceWord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-list validators
# ---------------------------------------------------------------------------

def validate_reference_notes(
    notes: List[ReferenceNote],
    duration_s: float = float("inf"),
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    for i, note in enumerate(notes):
        if math.isnan(note.onset_time) or math.isnan(note.offset_time):
            issues.append(ValidationIssue(
                level="error", event_type="ReferenceNote", event_idx=i,
                message="NaN timestamp",
            ))
            continue
        if note.onset_time < 0:
            issues.append(ValidationIssue(
                level="error", event_type="ReferenceNote", event_idx=i,
                message=f"Negative onset_time: {note.onset_time:.4f}s",
            ))
        if note.offset_time <= note.onset_time:
            issues.append(ValidationIssue(
                level="error", event_type="ReferenceNote", event_idx=i,
                message=f"offset_time ({note.offset_time:.4f}s) ≤ onset_time ({note.onset_time:.4f}s)",
            ))
        if note.offset_time > duration_s + 1e-3:
            issues.append(ValidationIssue(
                level="warning", event_type="ReferenceNote", event_idx=i,
                message=f"offset_time {note.offset_time:.4f}s exceeds duration {duration_s:.4f}s",
            ))
        if not note.is_rest and note.pitch_midi is None:
            issues.append(ValidationIssue(
                level="warning", event_type="ReferenceNote", event_idx=i,
                message="Non-rest note has no pitch_midi",
            ))
        if note.pitch_midi is not None and not (0.0 <= note.pitch_midi <= 127.0):
            issues.append(ValidationIssue(
                level="warning", event_type="ReferenceNote", event_idx=i,
                message=f"pitch_midi {note.pitch_midi:.1f} outside [0, 127]",
            ))

    # Check for overlapping non-rest notes (same onset region)
    pitched = [(n.onset_time, n.offset_time, i) for i, n in enumerate(notes) if not n.is_rest]
    for j in range(len(pitched) - 1):
        s1, e1, idx1 = pitched[j]
        s2, e2, idx2 = pitched[j + 1]
        if s2 < e1 - 1e-4:
            issues.append(ValidationIssue(
                level="warning", event_type="ReferenceNote", event_idx=idx2,
                message=f"Overlaps preceding note (note {idx1}): [{s1:.3f}, {e1:.3f}) ∩ [{s2:.3f}, {e2:.3f})",
            ))

    return issues


def validate_reference_phonemes(
    phonemes: List[ReferencePhoneme],
    duration_s: float = float("inf"),
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    for i, ph in enumerate(phonemes):
        if math.isnan(ph.start_time) or math.isnan(ph.end_time):
            issues.append(ValidationIssue(
                level="error", event_type="ReferencePhoneme", event_idx=i,
                message="NaN timestamp",
            ))
            continue
        if ph.start_time < 0:
            issues.append(ValidationIssue(
                level="error", event_type="ReferencePhoneme", event_idx=i,
                message=f"Negative start_time: {ph.start_time:.4f}s",
            ))
        if ph.end_time <= ph.start_time:
            issues.append(ValidationIssue(
                level="error", event_type="ReferencePhoneme", event_idx=i,
                message=f"end_time ({ph.end_time:.4f}s) ≤ start_time ({ph.start_time:.4f}s)",
            ))
        if ph.end_time > duration_s + 1e-3:
            issues.append(ValidationIssue(
                level="warning", event_type="ReferencePhoneme", event_idx=i,
                message=f"end_time {ph.end_time:.4f}s exceeds duration {duration_s:.4f}s",
            ))

    # Overlapping phonemes are an error (TextGrid should not produce these)
    for j in range(len(phonemes) - 1):
        s1, e1 = phonemes[j].start_time, phonemes[j].end_time
        s2, e2 = phonemes[j + 1].start_time, phonemes[j + 1].end_time
        if s2 < e1 - 1e-4:
            issues.append(ValidationIssue(
                level="error", event_type="ReferencePhoneme", event_idx=j + 1,
                message=f"Overlaps preceding phoneme {j}: [{s1:.3f}, {e1:.3f}) ∩ [{s2:.3f}, {e2:.3f})",
            ))

    return issues


def validate_reference_words(
    words: List[ReferenceWord],
    phonemes: List[ReferencePhoneme],
    duration_s: float = float("inf"),
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    for i, w in enumerate(words):
        if math.isnan(w.start_time) or math.isnan(w.end_time):
            issues.append(ValidationIssue(
                level="error", event_type="ReferenceWord", event_idx=i,
                message="NaN timestamp",
            ))
            continue
        if w.end_time <= w.start_time:
            issues.append(ValidationIssue(
                level="error", event_type="ReferenceWord", event_idx=i,
                message=f"end_time ({w.end_time:.4f}s) ≤ start_time ({w.start_time:.4f}s)",
            ))
        if w.end_time > duration_s + 1e-3:
            issues.append(ValidationIssue(
                level="warning", event_type="ReferenceWord", event_idx=i,
                message=f"end_time {w.end_time:.4f}s exceeds duration {duration_s:.4f}s",
            ))
        # Check that referenced phoneme indices are valid
        for ph_idx in w.phoneme_indices:
            if ph_idx < 0 or ph_idx >= len(phonemes):
                issues.append(ValidationIssue(
                    level="error", event_type="ReferenceWord", event_idx=i,
                    message=f"phoneme_indices contains out-of-range index {ph_idx}",
                ))

    return issues


def validate_tempo_consistency(
    reference: ReferencePerformanceRepresentation,
) -> List[ValidationIssue]:
    """Check that note timing is consistent with the declared tempo."""
    issues: List[ValidationIssue] = []

    if not reference.notes or reference.tempo_bpm is None:
        return issues

    # The last non-rest note's offset should roughly equal duration_s
    non_rest = [n for n in reference.notes if not n.is_rest]
    if not non_rest:
        return issues

    last_offset = max(n.offset_time for n in non_rest)
    if abs(last_offset - reference.duration_s) > 1.0:
        issues.append(ValidationIssue(
            level="warning", event_type="ReferencePerformanceRepresentation", event_idx=0,
            message=(
                f"Last note offset ({last_offset:.3f}s) differs from "
                f"declared duration ({reference.duration_s:.3f}s) by more than 1s"
            ),
        ))
    return issues


# ---------------------------------------------------------------------------
# Top-level validator
# ---------------------------------------------------------------------------

def validate_reference_representation(
    reference: ReferencePerformanceRepresentation,
    log_issues: bool = True,
) -> ValidationReport:
    """
    Run all structural checks on a ReferencePerformanceRepresentation.

    Args:
        reference:   The reference object to validate.
        log_issues:  If True, emit warnings/errors via the logging system.

    Returns:
        ValidationReport with all found issues.
    """
    issues: List[ValidationIssue] = []

    # Tier presence (warnings)
    if not reference.notes:
        issues.append(ValidationIssue(
            level="warning", event_type="ReferencePerformanceRepresentation", event_idx=0,
            message="No reference notes found (MusicXML may not have been parsed)",
        ))
    if not reference.phonemes:
        issues.append(ValidationIssue(
            level="warning", event_type="ReferencePerformanceRepresentation", event_idx=0,
            message="No reference phonemes found (TextGrid may not have been parsed)",
        ))

    issues += validate_reference_notes(reference.notes, duration_s=reference.duration_s)
    issues += validate_reference_phonemes(reference.phonemes, duration_s=reference.duration_s)
    issues += validate_reference_words(reference.words, reference.phonemes, duration_s=reference.duration_s)
    issues += validate_tempo_consistency(reference)

    report = ValidationReport(issues=issues)

    if log_issues:
        for issue in issues:
            msg = f"[reference_validation] {issue.level.upper()} [{issue.event_type}#{issue.event_idx}] {issue.message}"
            if issue.level == "error":
                logger.error(msg)
            else:
                logger.warning(msg)

    if not issues:
        logger.debug("[reference_validation] No issues found.")

    return report
