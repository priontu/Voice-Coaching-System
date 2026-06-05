"""
metrics/validation.py - Sanity checks for PerformanceMetricsReport.

validate_metrics_report() verifies numerical invariants (no NaN/Inf,
values in expected ranges, non-negative absolutes) and returns a
MetricValidationReport with any issues found.

Severity levels:
  error   — value violates a hard invariant (NaN/Inf, out of [0,1], negative abs)
  warning — value is suspicious but not necessarily wrong (n_evaluated=0, etc.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.types import PerformanceMetricsReport


@dataclass
class MetricValidationIssue:
    """One validation finding."""

    field: str
    severity: str   # "error" | "warning"
    message: str

    def to_dict(self) -> Dict:
        return {"field": self.field, "severity": self.severity, "message": self.message}


@dataclass
class MetricValidationReport:
    """Aggregated result of validate_metrics_report()."""

    valid: bool
    issues: List[MetricValidationIssue] = field(default_factory=list)
    n_errors: int = 0
    n_warnings: int = 0

    def to_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan_or_inf(v: Optional[float]) -> bool:
    if v is None:
        return False
    return math.isnan(v) or math.isinf(v)


def _in_unit(v: Optional[float]) -> bool:
    if v is None:
        return True
    return 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_metrics_report(
    report: PerformanceMetricsReport,
    log_issues: bool = False,
) -> MetricValidationReport:
    """
    Validate a PerformanceMetricsReport for numerical correctness.

    Args:
        report:     The report to validate.
        log_issues: If True, emit warnings via the standard logging module.

    Returns:
        MetricValidationReport; valid=True iff n_errors == 0.
    """
    issues: List[MetricValidationIssue] = []

    def err(f: str, msg: str) -> None:
        issues.append(MetricValidationIssue(field=f, severity="error", message=msg))

    def warn(f: str, msg: str) -> None:
        issues.append(MetricValidationIssue(field=f, severity="warning", message=msg))

    # ── Coverage ─────────────────────────────────────────────────────────
    if report.n_note_matches == 0:
        warn("n_note_matches", "No note matches — most metrics will be None")

    # ── Note precision / recall ───────────────────────────────────────────
    for name, val in [
        ("note_precision", report.note_precision),
        ("note_recall", report.note_recall),
    ]:
        if val is None:
            continue
        if _nan_or_inf(val):
            err(name, f"{name} is NaN or Inf")
        elif not _in_unit(val):
            err(name, f"{name}={val:.4f} outside [0, 1]")

    # ── Pitch ─────────────────────────────────────────────────────────────
    if report.pitch is not None:
        p = report.pitch
        for fname, val in [
            ("pitch.pitch_accuracy", p.pitch_accuracy),
            ("pitch.note_pitch_accuracy", p.note_pitch_accuracy),
        ]:
            if val is None:
                continue
            if _nan_or_inf(val):
                err(fname, "NaN or Inf")
            elif not _in_unit(val):
                err(fname, f"Outside [0, 1]: {val:.4f}")

        for fname, val in [
            ("pitch.pitch_rmse_cents", p.pitch_rmse_cents),
            ("pitch.mace_cents", p.mace_cents),
        ]:
            if val is None:
                continue
            if _nan_or_inf(val):
                err(fname, "NaN or Inf")
            elif val < 0:
                err(fname, f"Negative: {val:.4f}")

        if p.n_evaluated == 0 and p.pitch_accuracy is not None:
            warn("pitch", "n_evaluated=0 but pitch_accuracy is set")

        for b in p.per_note:
            if b.value is not None and _nan_or_inf(b.value):
                warn(f"pitch.per_note[{b.event_idx}]", "NaN or Inf deviation")

    # ── Timing ────────────────────────────────────────────────────────────
    if report.timing is not None:
        t = report.timing
        for fname, val in [
            ("timing.mean_onset_error_ms", t.mean_onset_error_ms),
            ("timing.std_onset_error_ms", t.std_onset_error_ms),
            ("timing.mean_abs_onset_error_ms", t.mean_abs_onset_error_ms),
            ("timing.median_onset_error_ms", t.median_onset_error_ms),
            ("timing.mean_offset_error_ms", t.mean_offset_error_ms),
            ("timing.mean_abs_offset_error_ms", t.mean_abs_offset_error_ms),
            ("timing.ioi_mae_ms", t.ioi_mae_ms),
        ]:
            if val is not None and _nan_or_inf(val):
                err(fname, "NaN or Inf")

        for fname, val in [
            ("timing.mean_abs_onset_error_ms", t.mean_abs_onset_error_ms),
            ("timing.mean_abs_offset_error_ms", t.mean_abs_offset_error_ms),
            ("timing.std_onset_error_ms", t.std_onset_error_ms),
            ("timing.ioi_mae_ms", t.ioi_mae_ms),
        ]:
            if val is not None and val < 0:
                err(fname, f"Negative: {val:.4f}")

        if t.timing_accuracy is not None:
            if _nan_or_inf(t.timing_accuracy):
                err("timing.timing_accuracy", "NaN or Inf")
            elif not _in_unit(t.timing_accuracy):
                err("timing.timing_accuracy", f"Outside [0, 1]: {t.timing_accuracy:.4f}")

    # ── Duration ──────────────────────────────────────────────────────────
    if report.duration is not None:
        d = report.duration
        for fname, val in [
            ("duration.mean_duration_error_s", d.mean_duration_error_s),
            ("duration.mean_abs_duration_error_s", d.mean_abs_duration_error_s),
            ("duration.std_duration_error_s", d.std_duration_error_s),
            ("duration.mean_duration_ratio", d.mean_duration_ratio),
            ("duration.mean_relative_duration_error", d.mean_relative_duration_error),
        ]:
            if val is not None and _nan_or_inf(val):
                err(fname, "NaN or Inf")

        for fname, val in [
            ("duration.mean_abs_duration_error_s", d.mean_abs_duration_error_s),
            ("duration.std_duration_error_s", d.std_duration_error_s),
            ("duration.mean_duration_ratio", d.mean_duration_ratio),
            ("duration.mean_relative_duration_error", d.mean_relative_duration_error),
        ]:
            if val is not None and val < 0:
                err(fname, f"Negative: {val:.4f}")

        if d.n_evaluated == 0 and d.mean_duration_error_s is not None:
            warn("duration", "n_evaluated=0 but mean_duration_error_s is set")

    # ── Lyric ─────────────────────────────────────────────────────────────
    if report.lyric is not None:
        lyr = report.lyric
        for fname, val in [
            ("lyric.mean_abs_phoneme_boundary_error_ms", lyr.mean_abs_phoneme_boundary_error_ms),
            ("lyric.std_phoneme_boundary_error_ms", lyr.std_phoneme_boundary_error_ms),
        ]:
            if val is not None:
                if _nan_or_inf(val):
                    err(fname, "NaN or Inf")
                elif val < 0:
                    err(fname, f"Negative: {val:.4f}")

        for fname, val in [
            ("lyric.phoneme_overlap_accuracy", lyr.phoneme_overlap_accuracy),
            ("lyric.word_alignment_accuracy", lyr.word_alignment_accuracy),
            ("lyric.label_match_rate", lyr.label_match_rate),
        ]:
            if val is None:
                continue
            if _nan_or_inf(val):
                err(fname, "NaN or Inf")
            elif not _in_unit(val):
                err(fname, f"Outside [0, 1]: {val:.4f}")

        if lyr.n_phoneme_matches == 0 and lyr.mean_phoneme_boundary_error_ms is not None:
            warn("lyric", "n_phoneme_matches=0 but boundary error is set")

    # ── Log ───────────────────────────────────────────────────────────────
    if log_issues and issues:
        import logging
        _log = logging.getLogger(__name__)
        for issue in issues:
            if issue.severity == "error":
                _log.error("[MetricValidation] %s: %s", issue.field, issue.message)
            else:
                _log.warning("[MetricValidation] %s: %s", issue.field, issue.message)

    n_errors = sum(1 for i in issues if i.severity == "error")
    n_warnings = sum(1 for i in issues if i.severity == "warning")
    return MetricValidationReport(
        valid=(n_errors == 0),
        issues=issues,
        n_errors=n_errors,
        n_warnings=n_warnings,
    )
