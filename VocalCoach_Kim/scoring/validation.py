"""
scoring/validation.py - Numerical sanity checks for PerformanceScoreReport.

Validates that all scores are finite, within [0, 100], weights are consistent,
and confidence values are in [0, 1]. Emits structured issues with "error" or
"warning" severity.

Dataclasses:
    ScoreValidationIssue   One detected issue with field + severity + message.
    ScoreValidationReport  Collection of issues with counts and overall validity.

Functions:
    validate_score_report  Run all checks on a PerformanceScoreReport.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.types import CategoryScore, PerformanceScoreReport

logger = logging.getLogger(__name__)


@dataclass
class ScoreValidationIssue:
    """One detected issue from score validation."""

    field: str       # dotted field path, e.g. "pitch_score.components[0]"
    severity: str    # "error" | "warning"
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "severity": self.severity, "message": self.message}


@dataclass
class ScoreValidationReport:
    """Aggregated result of validate_score_report()."""

    valid: bool
    issues: List[ScoreValidationIssue] = field(default_factory=list)
    n_errors: int = 0
    n_warnings: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "issues": [i.to_dict() for i in self.issues],
        }


def validate_score_report(
    report: PerformanceScoreReport,
    log_issues: bool = False,
) -> ScoreValidationReport:
    """
    Run numerical and structural sanity checks on a PerformanceScoreReport.

    Checks performed:
      - overall_score: finite, ∈ [0, 100]
      - each CategoryScore.score: finite, ∈ [0, 100]
      - each CategoryScore.confidence: finite, ∈ [0, 1] (if not None)
      - each CategoryScore.n_evaluated: ≥ 0; warn when 0 with a non-zero score
      - each ScoreBreakdown.score: finite, ∈ [0, 100]
      - each ScoreBreakdown.weight: > 0
      - weights_used sum ≈ 1.0 (when non-empty)

    Args:
        report:     PerformanceScoreReport to validate.
        log_issues: If True, emit each issue as a WARNING-level log message.

    Returns:
        ScoreValidationReport with issues list and validity flag.
    """
    issues: List[ScoreValidationIssue] = []

    def _err(field: str, msg: str) -> None:
        issues.append(ScoreValidationIssue(field=field, severity="error", message=msg))

    def _warn(field: str, msg: str) -> None:
        issues.append(ScoreValidationIssue(field=field, severity="warning", message=msg))

    def _check_score_value(val: Optional[float], path: str) -> None:
        if val is None:
            return
        if not math.isfinite(val):
            _err(path, f"{path} is non-finite ({val})")
        elif not (0.0 <= val <= 100.0):
            _err(path, f"{path}={val:.4f} is outside [0, 100]")

    def _check_fraction(val: Optional[float], path: str) -> None:
        if val is None:
            return
        if not math.isfinite(val):
            _err(path, f"{path} is non-finite ({val})")
        elif not (0.0 <= val <= 1.0):
            _err(path, f"{path}={val:.4f} is outside [0, 1]")

    def _validate_category(cat: Optional[CategoryScore], name: str) -> None:
        if cat is None:
            return
        _check_score_value(cat.score, f"{name}.score")
        _check_fraction(cat.confidence, f"{name}.confidence")
        if cat.n_evaluated < 0:
            _err(f"{name}.n_evaluated", f"n_evaluated={cat.n_evaluated} is negative")
        if cat.n_evaluated == 0 and cat.score > 0.0:
            _warn(f"{name}.n_evaluated", f"n_evaluated=0 but score={cat.score:.2f} — score may be unreliable")
        for i, bd in enumerate(cat.components):
            bpath = f"{name}.components[{i}]({bd.component})"
            _check_score_value(bd.score, bpath)
            _check_fraction(bd.confidence, f"{bpath}.confidence")
            if bd.weight <= 0.0:
                _err(f"{bpath}.weight", f"weight={bd.weight} must be positive")

    # Overall score
    _check_score_value(report.overall_score, "overall_score")

    if report.overall_score is None:
        _warn("overall_score", "overall_score is None — no categories had sufficient data")

    # Category scores
    _validate_category(report.pitch_score, "pitch_score")
    _validate_category(report.timing_score, "timing_score")
    _validate_category(report.duration_score, "duration_score")
    _validate_category(report.lyric_score, "lyric_score")

    # Weight consistency
    if report.weights_used:
        total = sum(report.weights_used.values())
        if not math.isfinite(total):
            _err("weights_used", f"weights sum to non-finite value ({total})")
        elif abs(total - 1.0) > 0.02:
            _warn("weights_used", f"weights sum to {total:.4f}, expected ~1.0")

    n_errors   = sum(1 for i in issues if i.severity == "error")
    n_warnings = sum(1 for i in issues if i.severity == "warning")

    if log_issues and issues:
        for iss in issues:
            lvl = logging.ERROR if iss.severity == "error" else logging.WARNING
            logger.log(lvl, "[ScoreValidation] %s: %s", iss.field, iss.message)

    return ScoreValidationReport(
        valid=n_errors == 0,
        issues=issues,
        n_errors=n_errors,
        n_warnings=n_warnings,
    )
