"""
scripts/smoke_test.py - Integration smoke test for all VocalCoach subsystems.

Run from the VocalCoach root:
    py scripts/smoke_test.py
    py scripts/smoke_test.py --verbose

Uses synthetic dummy data — no real audio files or checkpoints required.

Exit code 0 = all smoke tests passed.
Exit code 1 = one or more tests failed.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

_verbose = False
_registry: List[Tuple[str, Callable]] = []


def smoke(name: str):
    """Register a smoke-test function by name."""
    def decorator(fn: Callable) -> Callable:
        _registry.append((name, fn))
        return fn
    return decorator


def run_all() -> Tuple[int, int]:
    passed = failed = 0
    for name, fn in _registry:
        t0 = time.perf_counter()
        try:
            fn()
            ms = (time.perf_counter() - t0) * 1000
            print(f"  {GREEN}PASS{RESET}  {name}  ({ms:.0f}ms)")
            passed += 1
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            print(f"  {RED}FAIL{RESET}  {name}  ({ms:.0f}ms)")
            print(f"        {RED}{exc}{RESET}")
            if _verbose:
                traceback.print_exc()
            failed += 1
    return passed, failed


# ===========================================================================
# Tier 1 — Core imports
# ===========================================================================

@smoke("import: utils.types Phase 7 types")
def _t01():
    from utils.types import (
        AudioFeatures, PhonemeSegment, UnifiedInferenceResult,
        ScoreBreakdown, CategoryScore, PerformanceScoreReport,
        InterpretationSummary, PerformanceMetricsReport,
    )


@smoke("import: configs.loader")
def _t02():
    from configs.loader import load_config, load_pipeline_config, merge_configs


@smoke("import: preprocessing.audio_pipeline")
def _t03():
    from preprocessing.audio_pipeline import AudioPreprocessor


@smoke("import: scoring.normalization")
def _t04():
    from scoring.normalization import bounded_score, gaussian_penalty, piecewise_score, normalize_metric


@smoke("import: scoring.pitch_scoring")
def _t05():
    from scoring.pitch_scoring import compute_pitch_score, compute_intonation_score


@smoke("import: scoring.timing_scoring")
def _t06():
    from scoring.timing_scoring import compute_timing_score, compute_rhythm_stability_score


@smoke("import: scoring.duration_scoring")
def _t07():
    from scoring.duration_scoring import compute_duration_score, compute_phrase_duration_score


@smoke("import: scoring.lyric_scoring")
def _t08():
    from scoring.lyric_scoring import compute_lyric_clarity_score, compute_phoneme_timing_score


@smoke("import: scoring.performance_scoring")
def _t09():
    from scoring.performance_scoring import build_performance_score_report


@smoke("import: scoring.interpretation")
def _t10():
    from scoring.interpretation import build_interpretation_summary


@smoke("import: scoring.validation")
def _t11():
    from scoring.validation import validate_score_report, ScoreValidationReport


@smoke("import: metrics.reporting")
def _t12():
    from metrics.reporting import build_metrics_report


@smoke("import: visualization.scoring_viz")
def _t13():
    from visualization.scoring_viz import (
        plot_category_radar, plot_score_breakdown, plot_performance_dashboard,
    )


@smoke("import: inference.pipeline (no model load)")
def _t14():
    from inference.pipeline import UnifiedInferencePipeline


# ===========================================================================
# Tier 2 — Config loading
# ===========================================================================

@smoke("config: pipeline.yaml loads and has required sections")
def _t20():
    from configs.loader import load_pipeline_config
    cfg = load_pipeline_config()
    assert isinstance(cfg, dict)
    assert "pipeline" in cfg, "Missing 'pipeline' section"
    assert "scoring" in cfg,  "Missing 'scoring' section"


@smoke("config: system.yaml loads")
def _t21():
    from configs.loader import load_config
    p = Path(__file__).parent.parent / "configs" / "system.yaml"
    cfg = load_config(str(p))
    assert "system" in cfg


@smoke("config: merge_configs deep-merges correctly")
def _t22():
    from configs.loader import merge_configs
    base     = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99}, "c": 4}
    merged   = merge_configs(base, override)
    assert merged["a"]["x"] == 1,  "base key must survive"
    assert merged["a"]["y"] == 99, "override must win"
    assert merged["b"] == 3
    assert merged["c"] == 4


# ===========================================================================
# Tier 3 — Normalization functions
# ===========================================================================

@smoke("normalization: bounded_score lower/upper extremes")
def _t30():
    from scoring.normalization import bounded_score
    assert bounded_score(0.0,   0.0, 100.0) == 100.0
    assert bounded_score(100.0, 0.0, 100.0) == 0.0
    assert abs(bounded_score(50.0, 0.0, 100.0) - 50.0) < 1e-9


@smoke("normalization: gaussian_penalty zero deviation → 100")
def _t31():
    from scoring.normalization import gaussian_penalty
    assert gaussian_penalty(0.0, 50.0) == 100.0
    g = gaussian_penalty(50.0, 50.0)
    assert 0.0 < g < 100.0


@smoke("normalization: piecewise_score interpolates correctly")
def _t32():
    from scoring.normalization import piecewise_score
    bp = [(0.0, 100.0), (50.0, 75.0), (100.0, 0.0)]
    assert piecewise_score(0.0,   bp) == 100.0
    assert piecewise_score(50.0,  bp) == 75.0
    assert piecewise_score(100.0, bp) == 0.0
    mid = piecewise_score(25.0, bp)
    assert 75.0 < mid < 100.0


@smoke("normalization: non-finite input → 0")
def _t33():
    from scoring.normalization import normalize_metric
    assert normalize_metric(float("nan"), mode="bounded") == 0.0
    assert normalize_metric(float("inf"), mode="bounded") == 0.0


@smoke("normalization: invalid mode raises ValueError")
def _t34():
    from scoring.normalization import normalize_metric
    try:
        normalize_metric(1.0, mode="nonexistent")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


# ===========================================================================
# Tier 4 — Scoring with synthetic metrics
# ===========================================================================

def _make_pitch_metrics(mace=20.0, rmse=15.0, accuracy=0.85, n=10):
    from utils.types import PitchMetrics
    return PitchMetrics(
        n_evaluated=n,
        mace_cents=mace,
        pitch_rmse_cents=rmse,
        pitch_accuracy=accuracy,
    )


def _make_timing_metrics(accuracy=0.80, mae=30.0, n=10):
    from utils.types import TimingMetrics
    return TimingMetrics(
        n_evaluated=n,
        timing_accuracy=accuracy,
        mean_abs_onset_error_ms=mae,
        median_onset_error_ms=mae * 0.8,
    )


def _make_metrics_report(pitch=True, timing=True):
    from utils.types import PerformanceMetricsReport
    return PerformanceMetricsReport(
        audio_path="smoke_test.wav",
        reference_source_path="smoke_test.xml",
        pitch=_make_pitch_metrics() if pitch else None,
        timing=_make_timing_metrics() if timing else None,
    )


@smoke("scoring: compute_pitch_score returns valid CategoryScore")
def _t40():
    from scoring.pitch_scoring import compute_pitch_score
    cat = compute_pitch_score(_make_pitch_metrics())
    assert 0.0 <= cat.score <= 100.0
    assert cat.n_evaluated == 10
    assert cat.confidence is not None and cat.confidence > 0.0


@smoke("scoring: compute_timing_score returns valid CategoryScore")
def _t41():
    from scoring.timing_scoring import compute_timing_score
    cat = compute_timing_score(_make_timing_metrics())
    assert 0.0 <= cat.score <= 100.0


@smoke("scoring: build_performance_score_report full metrics")
def _t42():
    from scoring.performance_scoring import build_performance_score_report
    report = build_performance_score_report(_make_metrics_report())
    assert report.pitch_score is not None
    assert report.timing_score is not None
    assert report.overall_score is not None
    assert 0.0 <= report.overall_score <= 100.0


@smoke("scoring: build_performance_score_report pitch-only")
def _t43():
    from scoring.performance_scoring import build_performance_score_report
    report = build_performance_score_report(_make_metrics_report(timing=False))
    assert report.pitch_score is not None
    assert report.timing_score is None
    assert report.overall_score is not None


@smoke("scoring: validate_score_report no errors on valid report")
def _t44():
    from scoring.performance_scoring import build_performance_score_report
    from scoring.validation import validate_score_report
    report = build_performance_score_report(_make_metrics_report())
    val = validate_score_report(report)
    assert val.n_errors == 0, f"Unexpected validation errors: {val.n_errors} error(s) in {val.issues}"


@smoke("scoring: weights_used sum to 1.0")
def _t45():
    from scoring.performance_scoring import build_performance_score_report
    report = build_performance_score_report(_make_metrics_report())
    total = sum(report.weights_used.values())
    assert abs(total - 1.0) < 1e-6, f"weights_used sums to {total}, expected 1.0"


@smoke("scoring: to_dict is JSON-serializable")
def _t46():
    import json
    from scoring.performance_scoring import build_performance_score_report
    report = build_performance_score_report(_make_metrics_report())
    json.dumps(report.to_dict())


# ===========================================================================
# Tier 5 — Interpretation
# ===========================================================================

def _make_score_report(overall=80.0, pitch=85.0, timing=78.0):
    from utils.types import PerformanceScoreReport, CategoryScore
    def _cat(name, score):
        return CategoryScore(category=name, score=score, confidence=1.0, n_evaluated=5)
    return PerformanceScoreReport(
        audio_path="smoke_test.wav",
        reference_source_path="smoke_test.xml",
        overall_score=overall,
        pitch_score=_cat("pitch", pitch),
        timing_score=_cat("timing", timing),
    )


@smoke("interpretation: overall_level assigned")
def _t50():
    from scoring.interpretation import build_interpretation_summary
    s = build_interpretation_summary(_make_score_report(overall=80.0))
    assert s.overall_level in ("excellent", "good", "fair", "needs_work")


@smoke("interpretation: excellent for score=92")
def _t51():
    from scoring.interpretation import build_interpretation_summary
    s = build_interpretation_summary(_make_score_report(overall=92.0))
    assert s.overall_level == "excellent"


@smoke("interpretation: needs_work for score=40")
def _t52():
    from scoring.interpretation import build_interpretation_summary
    s = build_interpretation_summary(_make_score_report(overall=40.0))
    assert s.overall_level == "needs_work"


@smoke("interpretation: deterministic on same input")
def _t53():
    from scoring.interpretation import build_interpretation_summary
    r = _make_score_report()
    s1 = build_interpretation_summary(r)
    s2 = build_interpretation_summary(r)
    assert s1.overall_level == s2.overall_level
    assert s1.strengths    == s2.strengths
    assert s1.weaknesses   == s2.weaknesses


@smoke("interpretation: to_dict JSON-serializable")
def _t54():
    import json
    from scoring.interpretation import build_interpretation_summary
    d = build_interpretation_summary(_make_score_report()).to_dict()
    json.dumps(d)


# ===========================================================================
# Tier 6 — Pipeline initialization (no model load)
# ===========================================================================

@smoke("pipeline: initializes from dict overrides")
def _t60():
    from inference.pipeline import UnifiedInferencePipeline
    p = UnifiedInferencePipeline.from_dict({
        "pipeline": {"enable_pitch": False, "enable_phoneme": False, "enable_onset_offset": False},
        "scoring": {"enabled": False},
        "metrics": {"enabled": False},
    })
    assert p is not None


@smoke("pipeline: from_config_file reads pipeline.yaml")
def _t61():
    from inference.pipeline import UnifiedInferencePipeline
    cfg = Path(__file__).parent.parent / "configs" / "pipeline.yaml"
    if not cfg.exists():
        raise RuntimeError("configs/pipeline.yaml not found")
    p = UnifiedInferencePipeline.from_config_file(str(cfg))
    assert p is not None


# ===========================================================================
# Entry point
# ===========================================================================

def _parse_args():
    ap = argparse.ArgumentParser(description="VocalCoach smoke test suite")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show full tracebacks on failure")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    global _verbose
    _verbose = args.verbose

    print("\n" + "=" * 60)
    print("  VocalCoach Smoke Test Suite")
    print("=" * 60)

    sections = [
        ("Tier 1 — Core imports",          range(1, 15)),
        ("Tier 2 — Config loading",         range(20, 23)),
        ("Tier 3 — Normalization",          range(30, 35)),
        ("Tier 4 — Scoring (synthetic)",    range(40, 47)),
        ("Tier 5 — Interpretation",         range(50, 55)),
        ("Tier 6 — Pipeline init",          range(60, 62)),
    ]

    fn_map = {int(k[2:]): fn for k, fn in globals().items()
              if k.startswith("_t") and callable(fn)}

    total_passed = total_failed = 0
    for section_name, rng in sections:
        print(f"\n  {YELLOW}{section_name}{RESET}")
        for idx in rng:
            fn = fn_map.get(idx)
            if fn is None:
                continue
            name = next((n for n, f in _registry if f is fn), fn.__name__)
            t0 = time.perf_counter()
            try:
                fn()
                ms = (time.perf_counter() - t0) * 1000
                print(f"    {GREEN}PASS{RESET}  {name}  ({ms:.0f}ms)")
                total_passed += 1
            except Exception as exc:
                ms = (time.perf_counter() - t0) * 1000
                print(f"    {RED}FAIL{RESET}  {name}  ({ms:.0f}ms)")
                print(f"          {RED}{exc}{RESET}")
                if _verbose:
                    traceback.print_exc()
                total_failed += 1

    print("\n" + "=" * 60)
    total = total_passed + total_failed
    if total_failed == 0:
        print(f"  {GREEN}All {total} smoke tests passed.{RESET}")
    else:
        print(f"  {RED}{total_failed}/{total} FAILED — fix issues above before running the full pipeline.{RESET}")
    print("=" * 60 + "\n")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
