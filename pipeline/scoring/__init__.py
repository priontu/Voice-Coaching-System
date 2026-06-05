"""
scoring/ - Scoring engine for VocalCoach.

Phase 7 modules:
  normalization       Score normalization curves (bounded, gaussian, piecewise)
  pitch_scoring       Pitch CategoryScore from PitchMetrics
  timing_scoring      Timing CategoryScore from TimingMetrics
  duration_scoring    Duration CategoryScore from DurationMetrics
  lyric_scoring       Lyric CategoryScore from LyricMetrics
  performance_scoring PerformanceScoreReport from PerformanceMetricsReport
  interpretation      InterpretationSummary from PerformanceScoreReport
  validation          Numerical sanity checks → ScoreValidationReport

Legacy (Phase 1):
  pitch_score         CLI scorer against MusicXML reference
"""

from scoring.performance_scoring import build_performance_score_report
from scoring.interpretation import build_interpretation_summary
from scoring.validation import ScoreValidationReport, validate_score_report

__all__ = [
    "build_performance_score_report",
    "build_interpretation_summary",
    "validate_score_report",
    "ScoreValidationReport",
]
