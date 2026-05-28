"""
fusion/validation.py - Structural and temporal consistency checks for fused events.

Validates the output of the fusion layer before it reaches the scoring layer.
Issues are classified as 'warning' (non-blocking) or 'error' (blocks scoring).

Typical usage:
    from fusion.validation import validate_fused_representation
    report = validate_fused_representation(fused)
    if report.has_errors():
        logger.error(report)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from utils.types import (
    FusedPerformanceRepresentation,
    LyricEvent,
    NoteEvent,
    PhonemeSegment,
    PhraseEvent,
    TemporalRegion,
    WordEvent,
)

logger = logging.getLogger(__name__)

_FLOAT_TOL = 1e-6   # tolerance for floating-point boundary comparisons


# ---------------------------------------------------------------------------
# Issue and report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """One validation finding — a warning or blocking error."""
    level: str       # 'warning' | 'error'
    event_type: str  # e.g. 'note', 'lyric', 'word', 'phrase', 'region', 'global'
    event_idx: int   # index within its list; -1 for global issues
    message: str

    def to_dict(self):
        return asdict(self)

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.event_type}[{self.event_idx}]: {self.message}"


@dataclass
class ValidationReport:
    """Aggregated result of all validation checks."""
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.level == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")

    def has_errors(self) -> bool:
        return self.n_errors > 0

    def has_warnings(self) -> bool:
        return self.n_warnings > 0

    def to_dict(self):
        return {
            "is_valid": self.is_valid,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "issues": [i.to_dict() for i in self.issues],
        }

    def __repr__(self) -> str:
        return (
            f"ValidationReport(valid={self.is_valid}, "
            f"errors={self.n_errors}, warnings={self.n_warnings})"
        )

    def __str__(self) -> str:
        if not self.issues:
            return "ValidationReport: OK (no issues)"
        lines = [repr(self)]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-event-type validators
# ---------------------------------------------------------------------------

def validate_note_events(
    note_events: List[NoteEvent],
    duration_s: float = float("inf"),
) -> List[ValidationIssue]:
    """Check NoteEvent list for common structural issues."""
    issues: List[ValidationIssue] = []
    prev_offset: Optional[float] = None

    for idx, note in enumerate(note_events):
        # Negative or zero duration
        if note.duration is not None and note.duration <= _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="error", event_type="note", event_idx=idx,
                message=f"Non-positive duration: {note.duration:.6f}s",
            ))

        # onset >= offset
        if note.offset_time is not None and note.onset_time >= note.offset_time - _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="error", event_type="note", event_idx=idx,
                message=f"onset_time ({note.onset_time:.4f}) >= offset_time ({note.offset_time:.4f})",
            ))

        # Out of audio bounds
        if note.onset_time < -_FLOAT_TOL:
            issues.append(ValidationIssue(
                level="error", event_type="note", event_idx=idx,
                message=f"onset_time {note.onset_time:.4f} is negative",
            ))
        if note.offset_time is not None and note.offset_time > duration_s + _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="warning", event_type="note", event_idx=idx,
                message=f"offset_time {note.offset_time:.4f} exceeds audio duration {duration_s:.4f}",
            ))

        # Overlapping with previous note
        if prev_offset is not None and note.onset_time < prev_offset - _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="warning", event_type="note", event_idx=idx,
                message=f"Overlaps previous note (onset {note.onset_time:.4f} < prev_offset {prev_offset:.4f})",
            ))

        # Voiced fraction out of range
        if note.voiced_fraction is not None and not (0.0 <= note.voiced_fraction <= 1.0):
            issues.append(ValidationIssue(
                level="warning", event_type="note", event_idx=idx,
                message=f"voiced_fraction {note.voiced_fraction:.4f} outside [0, 1]",
            ))

        prev_offset = note.offset_time if note.offset_time is not None else None

    return issues


def validate_lyric_events(
    lyric_events: List[LyricEvent],
    duration_s: float = float("inf"),
) -> List[ValidationIssue]:
    """Check LyricEvent list for structural issues."""
    issues: List[ValidationIssue] = []

    for idx, le in enumerate(lyric_events):
        if le.duration <= _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="warning", event_type="lyric", event_idx=idx,
                message=f"Non-positive duration: {le.duration:.6f}s",
            ))
        if le.start_time < -_FLOAT_TOL:
            issues.append(ValidationIssue(
                level="error", event_type="lyric", event_idx=idx,
                message=f"start_time {le.start_time:.4f} is negative",
            ))
        if le.end_time > duration_s + _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="warning", event_type="lyric", event_idx=idx,
                message=f"end_time {le.end_time:.4f} exceeds audio duration {duration_s:.4f}",
            ))
        if not le.phoneme:
            issues.append(ValidationIssue(
                level="warning", event_type="lyric", event_idx=idx,
                message="Empty phoneme label",
            ))

    return issues


def validate_word_events(
    word_events: List[WordEvent],
    duration_s: float = float("inf"),
) -> List[ValidationIssue]:
    """Check WordEvent list for structural and internal consistency issues."""
    issues: List[ValidationIssue] = []

    for idx, we in enumerate(word_events):
        if we.duration <= _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="warning", event_type="word", event_idx=idx,
                message=f"Non-positive duration: {we.duration:.6f}s",
            ))
        if we.start_time < -_FLOAT_TOL:
            issues.append(ValidationIssue(
                level="error", event_type="word", event_idx=idx,
                message=f"start_time {we.start_time:.4f} is negative",
            ))
        if we.end_time > duration_s + _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="warning", event_type="word", event_idx=idx,
                message=f"end_time {we.end_time:.4f} exceeds audio duration {duration_s:.4f}",
            ))
        # Internal consistency: phoneme events should span the word interval
        if we.phoneme_events:
            span_start = we.phoneme_events[0].start_time
            span_end = we.phoneme_events[-1].end_time
            if abs(span_start - we.start_time) > _FLOAT_TOL * 100:
                issues.append(ValidationIssue(
                    level="warning", event_type="word", event_idx=idx,
                    message=(f"word.start_time {we.start_time:.4f} != first phoneme "
                             f"start {span_start:.4f}"),
                ))
            if abs(span_end - we.end_time) > _FLOAT_TOL * 100:
                issues.append(ValidationIssue(
                    level="warning", event_type="word", event_idx=idx,
                    message=(f"word.end_time {we.end_time:.4f} != last phoneme "
                             f"end {span_end:.4f}"),
                ))

    return issues


def validate_phrase_events(
    phrase_events: List[PhraseEvent],
    note_events: List[NoteEvent],
) -> List[ValidationIssue]:
    """Check PhraseEvent list for index validity."""
    issues: List[ValidationIssue] = []
    n_notes = len(note_events)

    for idx, pe in enumerate(phrase_events):
        if pe.duration <= _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="warning", event_type="phrase", event_idx=idx,
                message=f"Non-positive duration: {pe.duration:.6f}s",
            ))
        for ni in pe.note_indices:
            if ni < 0 or ni >= n_notes:
                issues.append(ValidationIssue(
                    level="error", event_type="phrase", event_idx=idx,
                    message=f"note_index {ni} out of range [0, {n_notes})",
                ))

    return issues


def validate_temporal_regions(
    regions: List[TemporalRegion],
    duration_s: float = float("inf"),
) -> List[ValidationIssue]:
    """Check TemporalRegion list for overlap and boundary issues."""
    issues: List[ValidationIssue] = []
    sorted_regions = sorted(regions, key=lambda r: r.start_time)

    for idx, region in enumerate(sorted_regions):
        if region.duration <= _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="warning", event_type="region", event_idx=idx,
                message=f"Non-positive duration: {region.duration:.6f}s",
            ))
        if region.start_time < -_FLOAT_TOL:
            issues.append(ValidationIssue(
                level="error", event_type="region", event_idx=idx,
                message=f"start_time {region.start_time:.4f} is negative",
            ))
        if region.end_time > duration_s + _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="warning", event_type="region", event_idx=idx,
                message=f"end_time {region.end_time:.4f} exceeds audio duration {duration_s:.4f}",
            ))

    # Check for overlapping regions
    for i in range(len(sorted_regions) - 1):
        a = sorted_regions[i]
        b = sorted_regions[i + 1]
        if b.start_time < a.end_time - _FLOAT_TOL:
            issues.append(ValidationIssue(
                level="warning", event_type="region", event_idx=i,
                message=(f"Region [{a.start_time:.4f}, {a.end_time:.4f}] overlaps "
                         f"next [{b.start_time:.4f}, {b.end_time:.4f}]"),
            ))

    return issues


def validate_array_consistency(
    fused: FusedPerformanceRepresentation,
) -> List[ValidationIssue]:
    """Check that canonical arrays share the same length."""
    issues: List[ValidationIssue] = []
    arrays = {"timestamps": fused.timestamps, "f0": fused.f0, "voiced": fused.voiced}
    lengths = {k: len(v) for k, v in arrays.items() if v is not None}

    if len(set(lengths.values())) > 1:
        issues.append(ValidationIssue(
            level="error", event_type="global", event_idx=-1,
            message=f"Canonical array length mismatch: {lengths}",
        ))

    return issues


# ---------------------------------------------------------------------------
# Top-level validator
# ---------------------------------------------------------------------------

def validate_fused_representation(
    fused: FusedPerformanceRepresentation,
    log_issues: bool = True,
) -> ValidationReport:
    """
    Run all validation checks on a FusedPerformanceRepresentation.

    Args:
        fused:       The representation to validate.
        log_issues:  If True, emit warnings/errors via the module logger.

    Returns:
        ValidationReport with all discovered issues.
    """
    report = ValidationReport()
    d = fused.duration_s

    report.issues.extend(validate_array_consistency(fused))
    report.issues.extend(validate_note_events(fused.note_events, duration_s=d))
    report.issues.extend(validate_lyric_events(fused.lyric_events, duration_s=d))
    report.issues.extend(validate_word_events(fused.word_events, duration_s=d))
    report.issues.extend(validate_phrase_events(fused.phrase_events, fused.note_events))
    report.issues.extend(validate_temporal_regions(fused.voiced_regions, duration_s=d))

    if log_issues:
        for issue in report.issues:
            if issue.level == "error":
                logger.error("[validation] %s", issue)
            else:
                logger.warning("[validation] %s", issue)

    if report.is_valid:
        logger.debug("[validation] FusedPerformanceRepresentation is valid.")
    else:
        logger.warning(
            "[validation] %d error(s), %d warning(s)",
            report.n_errors, report.n_warnings,
        )

    return report
