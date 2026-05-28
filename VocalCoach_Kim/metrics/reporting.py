"""
metrics/reporting.py - Metric aggregation engine.

build_metrics_report() is the single entry point: it accepts an AlignmentResult
and optionally a FusedPerformanceRepresentation and ReferencePerformanceRepresentation,
delegates to each category module, and assembles a PerformanceMetricsReport.

All category metrics are Optional — they are None when the corresponding
alignment data is empty or missing, so callers can selectively enable
only the data they have.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from utils.types import (
    AlignmentResult,
    DurationMetrics,
    FusedPerformanceRepresentation,
    LyricMetrics,
    PerformanceMetricsReport,
    PitchMetrics,
    ReferencePerformanceRepresentation,
    TimingMetrics,
)
from metrics.pitch_metrics import build_pitch_metrics
from metrics.timing_metrics import build_timing_metrics
from metrics.duration_metrics import build_duration_metrics
from metrics.lyric_metrics import build_lyric_metrics


def build_metrics_report(
    alignment: AlignmentResult,
    fused: Optional[FusedPerformanceRepresentation] = None,
    reference: Optional[ReferencePerformanceRepresentation] = None,
    config: Optional[Dict[str, Any]] = None,
) -> PerformanceMetricsReport:
    """
    Aggregate all metric categories into a PerformanceMetricsReport.

    Args:
        alignment: Full prediction ↔ reference alignment result (required).
        fused:     Fused performance representation (needed for duration/IOI
                   metrics and pitch stability; optional).
        reference: Parsed reference representation (needed for duration
                   metrics and pitch label enrichment; optional).
        config:    Optional dict with sub-keys:
                     pitch.cents_tolerance   (default 50.0)
                     timing.onset_tolerance_ms (default 50.0)
                     lyric.phoneme_tolerance_ms (default 30.0)

    Returns:
        PerformanceMetricsReport with all available sub-reports populated.
    """
    cfg: Dict[str, Any] = config or {}
    pitch_cfg: Dict = cfg.get("pitch", {})
    timing_cfg: Dict = cfg.get("timing", {})
    lyric_cfg: Dict = cfg.get("lyric", {})

    t0 = time.perf_counter()

    # ── Note-level coverage ────────────────────────────────────────────────
    n_pred = len(alignment.note_matches) + len(alignment.unmatched_pred_notes)
    n_ref = len(alignment.note_matches) + len(alignment.unmatched_ref_notes)

    # ── Pitch ──────────────────────────────────────────────────────────────
    pitch: Optional[PitchMetrics] = None
    if alignment.note_matches:
        pitch = build_pitch_metrics(
            alignment,
            fused=fused,
            reference=reference,
            tolerance_cents=float(pitch_cfg.get("cents_tolerance", 50.0)),
        )

    # ── Timing ────────────────────────────────────────────────────────────
    timing: Optional[TimingMetrics] = None
    if alignment.note_matches:
        timing = build_timing_metrics(
            alignment,
            fused=fused,
            reference=reference,
            tolerance_ms=float(timing_cfg.get("onset_tolerance_ms", 50.0)),
        )

    # ── Duration ──────────────────────────────────────────────────────────
    duration: Optional[DurationMetrics] = None
    if alignment.note_matches and fused is not None and reference is not None:
        duration = build_duration_metrics(alignment, fused=fused, reference=reference)

    # ── Lyric / phoneme ───────────────────────────────────────────────────
    lyric: Optional[LyricMetrics] = None
    if alignment.phoneme_matches or alignment.word_matches:
        lyric = build_lyric_metrics(
            alignment,
            fused=fused,
            reference=reference,
            tolerance_ms=float(lyric_cfg.get("phoneme_tolerance_ms", 30.0)),
        )

    elapsed = time.perf_counter() - t0

    return PerformanceMetricsReport(
        audio_path=alignment.predicted_audio_path,
        reference_source_path=alignment.reference_source_path,
        pitch=pitch,
        timing=timing,
        duration=duration,
        lyric=lyric,
        note_precision=alignment.note_precision,
        note_recall=alignment.note_recall,
        n_note_matches=alignment.n_note_matches,
        n_reference_notes=n_ref,
        n_predicted_notes=n_pred,
        computation_metadata={
            "elapsed_s": round(elapsed, 6),
            "config_used": cfg,
            "has_fused": fused is not None,
            "has_reference": reference is not None,
        },
    )
