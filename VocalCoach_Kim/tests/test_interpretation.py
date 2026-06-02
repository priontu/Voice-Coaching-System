"""
tests/test_interpretation.py - Unit tests for scoring/interpretation.py.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_interpretation.py -v
"""

from __future__ import annotations

import pytest

from utils.types import (
    CategoryScore,
    InterpretationSummary,
    PerformanceScoreReport,
    ScoreBreakdown,
)
from scoring.interpretation import build_interpretation_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cat(category: str, score: float, n: int = 5) -> CategoryScore:
    return CategoryScore(category=category, score=score, confidence=1.0, n_evaluated=n)


def _report(
    overall: float = 80.0,
    pitch: float = 85.0,
    timing: float = 78.0,
    duration: float = None,
    lyric: float = None,
) -> PerformanceScoreReport:
    return PerformanceScoreReport(
        audio_path="p.wav",
        reference_source_path="r.xml",
        overall_score=overall,
        pitch_score=_cat("pitch", pitch) if pitch is not None else None,
        timing_score=_cat("timing", timing) if timing is not None else None,
        duration_score=_cat("duration", duration) if duration is not None else None,
        lyric_score=_cat("lyric", lyric) if lyric is not None else None,
    )


# ---------------------------------------------------------------------------
# build_interpretation_summary
# ---------------------------------------------------------------------------

class TestBuildInterpretationSummary:
    def test_returns_interpretation_summary(self):
        s = build_interpretation_summary(_report())
        assert isinstance(s, InterpretationSummary)

    def test_excellent_level(self):
        s = build_interpretation_summary(_report(overall=92.0))
        assert s.overall_level == "excellent"

    def test_good_level(self):
        s = build_interpretation_summary(_report(overall=80.0))
        assert s.overall_level == "good"

    def test_fair_level(self):
        s = build_interpretation_summary(_report(overall=62.0))
        assert s.overall_level == "fair"

    def test_needs_work_level(self):
        s = build_interpretation_summary(_report(overall=40.0))
        assert s.overall_level == "needs_work"

    def test_strengths_populated_for_excellent_pitch(self):
        s = build_interpretation_summary(_report(overall=85.0, pitch=93.0, timing=80.0))
        assert len(s.strengths) >= 1
        assert any("pitch" in m.lower() or "intonation" in m.lower() for m in s.strengths)

    def test_weaknesses_populated_for_poor_timing(self):
        s = build_interpretation_summary(_report(overall=50.0, pitch=75.0, timing=40.0))
        assert len(s.weaknesses) >= 1
        assert any("timing" in m.lower() for m in s.weaknesses)

    def test_category_levels_populated_for_available_categories(self):
        s = build_interpretation_summary(_report(pitch=85.0, timing=78.0))
        assert "pitch" in s.category_levels
        assert "timing" in s.category_levels

    def test_none_category_not_in_levels(self):
        s = build_interpretation_summary(_report(duration=None, lyric=None))
        assert "duration" not in s.category_levels
        assert "lyric" not in s.category_levels

    def test_audio_path_preserved(self):
        s = build_interpretation_summary(_report())
        assert s.audio_path == "p.wav"

    def test_to_dict_serializable(self):
        d = build_interpretation_summary(_report()).to_dict()
        assert "overall_level" in d
        assert "strengths" in d
        assert "weaknesses" in d
        assert "category_levels" in d

    def test_deterministic_same_input(self):
        r = _report(overall=80.0, pitch=85.0, timing=78.0)
        s1 = build_interpretation_summary(r)
        s2 = build_interpretation_summary(r)
        assert s1.overall_level == s2.overall_level
        assert s1.strengths == s2.strengths
        assert s1.weaknesses == s2.weaknesses

    def test_custom_thresholds_honoured(self):
        # Lower excellent threshold — 80 should now qualify as excellent
        s = build_interpretation_summary(
            _report(overall=80.0),
            config={"excellent_threshold": 75.0, "good_threshold": 60.0, "fair_threshold": 40.0},
        )
        assert s.overall_level == "excellent"

    def test_all_categories_excellent_no_weaknesses(self):
        r = _report(overall=95.0, pitch=95.0, timing=93.0, duration=91.0, lyric=92.0)
        s = build_interpretation_summary(r)
        assert len(s.weaknesses) == 0

    def test_all_categories_needs_work_no_strengths(self):
        r = _report(overall=30.0, pitch=35.0, timing=30.0, duration=28.0, lyric=32.0)
        s = build_interpretation_summary(r)
        assert len(s.strengths) == 0

    def test_no_overall_score_uses_category_fallback(self):
        r = PerformanceScoreReport(
            audio_path="p.wav",
            reference_source_path="r.xml",
            overall_score=None,
            pitch_score=_cat("pitch", 92.0),
        )
        s = build_interpretation_summary(r)
        assert s.overall_level in ("excellent", "good", "fair", "needs_work")
