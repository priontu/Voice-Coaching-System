"""
metrics/ - Evaluation metrics for all VocalCoach modules.

Submodules:
  phoneme_metrics      Boundary precision/recall/F1/MAE (Phase 1)
  pitch_metrics        Frame-level and note-level pitch accuracy (Phase 1 + 6)
  onset_offset_metrics Onset/offset P/R/F1, duration MAE (Phase 1)
  timing_metrics       Onset/offset/IOI timing metrics (Phase 6)
  duration_metrics     Note duration comparison metrics (Phase 6)
  lyric_metrics        Phoneme boundary and word alignment metrics (Phase 6)
  reporting            Metric aggregation → PerformanceMetricsReport (Phase 6)
  validation           Numerical sanity checks for PerformanceMetricsReport (Phase 6)
"""

from metrics.reporting import build_metrics_report
from metrics.validation import MetricValidationReport, validate_metrics_report

__all__ = [
    "build_metrics_report",
    "validate_metrics_report",
    "MetricValidationReport",
]
