"""
inference/pipeline.py - Unified multi-model inference pipeline.

UnifiedInferencePipeline orchestrates all three VocalCoach models (phoneme,
pitch+VAD, onset/offset) against a single audio file. Preprocessing is shared:
the audio is loaded once, then each model receives the appropriate array format
without redundant disk I/O or GPU transfers.

Execution order (deterministic):
    Audio
      ↓ AudioPreprocessor — load once
      ↓ Pitch + VAD       — numpy array → PipelineOutput (100fps)
      ↓ Phoneme           — torch tensor → List[PhonemeSegment]
      ↓ Onset/Offset      — numpy array → onset_probs, offset_probs (~62.5fps)
      ↓ fusion/alignment  — merge onto canonical 100fps grid
      → UnifiedInferenceResult

Usage:
    pipeline = UnifiedInferencePipeline.from_config_file("configs/pipeline.yaml")
    result   = pipeline.predict("singing.wav")

    # batch
    results  = pipeline.predict_directory("dataset/")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from configs.loader import load_pipeline_config, merge_configs
from fusion.alignment import merge_model_outputs
from inference.device_manager import DeviceManager
from models.registry import ModelRegistry
from preprocessing.audio_pipeline import AudioPreprocessor
from preprocessing.timestamps import HOP_LENGTH, SAMPLE_RATE
from utils.logging_utils import get_logger
from utils.types import (
    AlignmentResult,
    FrameAlignedFeatures,
    FusedPerformanceRepresentation,
    InterpretationSummary,
    NoteEvent,
    PerformanceMetricsReport,
    PerformanceScoreReport,
    ReferencePerformanceRepresentation,
    UnifiedInferenceResult,
)

logger = get_logger(__name__)


class UnifiedInferencePipeline:
    """
    Single entry point for all VocalCoach inference.

    Each module (phoneme, pitch, onset/offset) remains fully independent;
    the pipeline only orchestrates the call sequence and hands off
    a pre-loaded audio array so preprocessing runs exactly once.

    Args:
        config: Merged config dict (pipeline.yaml + system.yaml).
                Pass None to use defaults from configs/pipeline.yaml.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            try:
                config = load_pipeline_config()
            except Exception:
                config = {}

        self._cfg = config
        self._pl = config.get("pipeline", {})

        # Flags
        self._enable_pitch: bool = bool(self._pl.get("enable_pitch", True))
        self._enable_phoneme: bool = bool(self._pl.get("enable_phoneme", True))
        self._enable_onset: bool = bool(self._pl.get("enable_onset_offset", False))

        # Fusion (Phase 4)
        self._fusion_cfg: Dict[str, Any] = config.get("fusion", {})
        self._enable_fusion: bool = bool(self._fusion_cfg.get("enabled", False))

        # Reference alignment (Phase 5)
        self._ref_cfg: Dict[str, Any] = config.get("reference", {})

        # Metric computation (Phase 6)
        self._metrics_cfg: Dict[str, Any] = config.get("metrics", {})
        self._enable_metrics: bool = bool(self._metrics_cfg.get("enabled", False))

        # Scoring engine (Phase 7)
        self._scoring_cfg: Dict[str, Any] = config.get("scoring", {})
        self._enable_scoring: bool = bool(self._scoring_cfg.get("enabled", False))

        # Device
        device_pref = self._pl.get("device", "auto")
        self._device_manager = DeviceManager(preference=device_pref)

        # Preprocessing
        self._preprocessor = AudioPreprocessor(config)

        # Model registry — models load lazily on first predict()
        self._registry = ModelRegistry()

        logger.info(
            f"[Pipeline] Initialized — "
            f"pitch={'on' if self._enable_pitch else 'off'}, "
            f"phoneme={'on' if self._enable_phoneme else 'off'}, "
            f"onset_offset={'on' if self._enable_onset else 'off'}, "
            f"fusion={'on' if self._enable_fusion else 'off'}, "
            f"device={self._device_manager.device_str}"
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config_file(cls, path: Union[str, Path]) -> "UnifiedInferencePipeline":
        """Instantiate from a YAML pipeline config file."""
        from configs.loader import load_config, merge_configs
        try:
            base = load_pipeline_config()
        except Exception:
            base = {}
        overrides = load_config(str(path))
        return cls(config=merge_configs(base, overrides))

    @classmethod
    def from_dict(cls, overrides: Dict[str, Any]) -> "UnifiedInferencePipeline":
        """Instantiate with runtime overrides merged over the default config."""
        try:
            base = load_pipeline_config()
        except Exception:
            base = {}
        return cls(config=merge_configs(base, overrides))

    # ------------------------------------------------------------------
    # Explicit preloading (optional — lazy loading is the default)
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """
        Eagerly load all enabled models.

        Call this before the first predict() to pay the model-load cost
        upfront (useful for servers that want warm startup).
        """
        if self._enable_pitch:
            self._get_pitch_model()
        if self._enable_phoneme:
            self._get_phoneme_model()
        if self._enable_onset:
            self._get_onset_model()

    # ------------------------------------------------------------------
    # Single-file inference
    # ------------------------------------------------------------------

    def predict(
        self,
        audio_path: Union[str, Path],
        musicxml_path: Optional[Union[str, Path]] = None,
        textgrid_path: Optional[Union[str, Path]] = None,
        compute_metrics: bool = True,
        compute_scores: bool = True,
    ) -> UnifiedInferenceResult:
        """
        Run all enabled models on one audio file with optional reference alignment,
        metric computation, and performance scoring.

        The audio is loaded ONCE and distributed to each model in the
        appropriate format (numpy / torch.Tensor).

        Args:
            audio_path:      Path to a WAV/MP3/FLAC file.
            musicxml_path:   Optional MusicXML score for reference alignment (Phase 5).
            textgrid_path:   Optional Praat TextGrid for reference alignment (Phase 5).
            compute_metrics: If True, compute PerformanceMetricsReport after alignment
                             (Phase 6). Requires an AlignmentResult to be available,
                             which in turn requires musicxml_path or textgrid_path and
                             fusion to be enabled.
            compute_scores:  If True, compute PerformanceScoreReport + InterpretationSummary
                             (Phase 7). Requires compute_metrics (or metrics.enabled=true)
                             to have populated result.metrics.

        Returns:
            UnifiedInferenceResult with per-model outputs, aligned view, and
            optionally fused representation, reference alignment, metrics, scores,
            and interpretation.
        """
        path = Path(audio_path)
        t_start = time.perf_counter()
        logger.info(f"[Pipeline] Processing: {path.name}")

        # ── 1. Shared preprocessing — load audio ONCE ──────────────────
        audio_np, sr = self._preprocessor.process_for_pitch(path)
        duration_s = len(audio_np) / sr
        logger.debug(f"[Pipeline] Audio loaded: {duration_s:.2f}s @ {sr}Hz")

        result = UnifiedInferenceResult(
            audio_path=str(path),
            sample_rate=sr,
            hop_length=HOP_LENGTH,
            duration_s=duration_s,
        )

        # ── 2. Pitch + VAD ─────────────────────────────────────────────
        if self._enable_pitch:
            result = self._run_pitch(audio_np, sr, result)

        # ── 3. Phoneme ─────────────────────────────────────────────────
        if self._enable_phoneme:
            result = self._run_phoneme(audio_np, result)

        # ── 4. Onset / Offset ──────────────────────────────────────────
        if self._enable_onset:
            result = self._run_onset(audio_np, sr, result)

        # ── 5. Temporal alignment ──────────────────────────────────────
        result = self._align(result)

        # ── 6. Feature fusion (Phase 4) ────────────────────────────────
        if self._enable_fusion and result.aligned is not None:
            result = self._run_fusion(result)

        # ── 7. Reference parsing + alignment (Phase 5) ─────────────────
        if musicxml_path is not None or textgrid_path is not None:
            result = self._run_reference_alignment(result, musicxml_path, textgrid_path)

        # ── 8. Metric computation (Phase 6) ────────────────────────────
        run_metrics = compute_metrics or self._enable_metrics
        if run_metrics and result.alignment is not None:
            result = self._run_metrics(result)

        # ── 9. Scoring + interpretation (Phase 7) ──────────────────────
        run_scores = compute_scores or self._enable_scoring
        if run_scores and result.metrics is not None:
            result = self._run_scoring(result)

        # ── 10. Metadata ───────────────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        result.metadata.update({
            "elapsed_s": round(elapsed, 3),
            "device": self._device_manager.device_str,
            "enabled": {
                "pitch": self._enable_pitch,
                "phoneme": self._enable_phoneme,
                "onset_offset": self._enable_onset,
                "fusion": self._enable_fusion,
                "reference_alignment": (
                    musicxml_path is not None or textgrid_path is not None
                ),
                "metrics": run_metrics,
                "scores": run_scores,
            },
        })

        logger.info(
            f"[Pipeline] Done in {elapsed:.2f}s — "
            f"pitch={'✓' if result.has_pitch() else '–'}, "
            f"phoneme={'✓' if result.has_phonemes() else '–'}, "
            f"onset={'✓' if result.has_onset_offset() else '–'}"
        )
        return result

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        paths: List[Union[str, Path]],
        skip_errors: bool = True,
    ) -> List[UnifiedInferenceResult]:
        """
        Run inference on a list of audio files.

        Args:
            paths:       Iterable of file paths.
            skip_errors: If True, log errors and continue; if False, re-raise.

        Returns:
            List of UnifiedInferenceResult (one per successful file).
        """
        results = []
        for i, p in enumerate(paths, 1):
            try:
                logger.info(f"[Pipeline] Batch {i}/{len(paths)}: {Path(p).name}")
                results.append(self.predict(p))
            except Exception as exc:
                if skip_errors:
                    logger.error(f"[Pipeline] Skipping {p}: {exc}")
                else:
                    raise
        return results

    def predict_directory(
        self,
        dir_path: Union[str, Path],
        pattern: str = "*.wav",
        skip_errors: bool = True,
    ) -> List[UnifiedInferenceResult]:
        """
        Run inference on all matching files in a directory.

        Args:
            dir_path:    Directory to scan.
            pattern:     Glob pattern (default "*.wav"). Use "**/*.wav" for recursive.
            skip_errors: Passed through to predict_batch().

        Returns:
            List of UnifiedInferenceResult sorted by filename.
        """
        root = Path(dir_path)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        paths = sorted(root.glob(pattern))
        if not paths:
            logger.warning(f"[Pipeline] No files matching '{pattern}' in {root}")
            return []

        logger.info(f"[Pipeline] Batch: {len(paths)} files in {root}")
        return self.predict_batch(paths, skip_errors=skip_errors)

    # ------------------------------------------------------------------
    # Per-model runners (private — called by predict())
    # ------------------------------------------------------------------

    def _run_pitch(
        self,
        audio_np: np.ndarray,
        sr: int,
        result: UnifiedInferenceResult,
    ) -> UnifiedInferenceResult:
        t0 = time.perf_counter()
        try:
            model = self._get_pitch_model()
            import torch
            with torch.no_grad():
                pitch_out = model._pipeline.run_from_array(audio_np, sr)
            result.pitch_timestamps = pitch_out.timestamps
            result.f0 = pitch_out.f0
            result.voiced = pitch_out.voiced_mask
            logger.debug(
                f"[Pipeline] Pitch: {len(pitch_out.timestamps)} frames "
                f"({time.perf_counter()-t0:.2f}s)"
            )
        except Exception as exc:
            logger.error(f"[Pipeline] Pitch failed: {exc}")
        return result

    def _run_phoneme(
        self,
        audio_np: np.ndarray,
        result: UnifiedInferenceResult,
    ) -> UnifiedInferenceResult:
        t0 = time.perf_counter()
        try:
            import torch
            model = self._get_phoneme_model()
            waveform = torch.from_numpy(audio_np)  # 1-D CPU float32 tensor
            with torch.no_grad():
                predict_result = model.predict(waveform)
            segments = predict_result["segments"]
            result.phoneme_segments = segments
            logger.debug(
                f"[Pipeline] Phoneme: {len(segments)} segments "
                f"({time.perf_counter()-t0:.2f}s)"
            )
        except Exception as exc:
            logger.error(f"[Pipeline] Phoneme failed: {exc}")
        return result

    def _run_onset(
        self,
        audio_np: np.ndarray,
        sr: int,
        result: UnifiedInferenceResult,
    ) -> UnifiedInferenceResult:
        t0 = time.perf_counter()
        try:
            model = self._get_onset_model()
            import torch
            with torch.no_grad():
                on_probs, off_probs, frame_times = (
                    model._detector.predict_probs_from_array(audio_np, sr)
                )
            result.onset_timestamps = frame_times
            result.onset_probs = on_probs
            result.offset_probs = off_probs

            # Also run peak-picking to populate note_events
            from models.onset_offset.spec_utils import (
                pair_onsets_offsets,
                peak_pick_offsets,
                peak_pick_onsets,
            )
            det = model._detector
            onsets = peak_pick_onsets(on_probs, frame_times, det.onset_threshold, det.min_distance_frames)
            offsets = peak_pick_offsets(off_probs, frame_times, det.offset_threshold, det.min_distance_frames)
            raw_notes = pair_onsets_offsets(onsets, offsets)
            result.note_events = [
                NoteEvent(
                    onset_time=n["onset_time"],
                    offset_time=n.get("offset_time"),
                    duration=n.get("duration"),
                )
                for n in raw_notes
            ]
            logger.debug(
                f"[Pipeline] Onset/offset: {len(on_probs)} frames, "
                f"{len(result.note_events)} notes "
                f"({time.perf_counter()-t0:.2f}s)"
            )
        except Exception as exc:
            logger.error(f"[Pipeline] Onset/offset failed: {exc}")
        return result

    def _run_fusion(self, result: UnifiedInferenceResult) -> UnifiedInferenceResult:
        """
        Fuse aligned model outputs into a FusedPerformanceRepresentation.

        Called after _align() when fusion.enabled=true. Builds note events,
        lyric/word events, phrase events, and voiced regions, then assembles
        them into result.fused.
        """
        from fusion.event_alignment import (
            annotate_lyrics_with_notes,
            annotate_notes_with_phonemes,
            annotate_words_with_notes,
            build_phrase_events,
            build_voiced_regions,
            score_note_phoneme_alignment,
        )
        from fusion.lyric_events import build_lyric_events
        from fusion.note_events import build_note_events
        from fusion.validation import validate_fused_representation

        t0 = time.perf_counter()
        aligned = result.aligned
        cfg = self._fusion_cfg

        # ── Note events from aligned onset/offset probs ─────────────────
        note_events: List[NoteEvent] = []
        if aligned.onset_probs is not None and aligned.offset_probs is not None:
            note_events = build_note_events(
                timestamps=aligned.timestamps,
                onset_probs=aligned.onset_probs,
                offset_probs=aligned.offset_probs,
                f0=aligned.f0,
                voiced=aligned.voiced,
                onset_threshold=float(cfg.get("onset_threshold", 0.5)),
                offset_threshold=float(cfg.get("offset_threshold", 0.5)),
                min_duration_s=float(cfg.get("min_note_duration_ms", 50)) / 1000.0,
                hop_length=result.hop_length,
                sample_rate=result.sample_rate,
            )
        if not note_events and result.note_events:
            # Fall back to peak-picked events from _run_onset when build_note_events
            # returns nothing (e.g. canonical-grid resampling attenuated peaks below threshold)
            note_events = list(result.note_events)
            # Enrich fallback notes with pitch stats from the aligned pitch track
            if aligned.f0 is not None and aligned.timestamps is not None:
                from fusion.note_events import _compute_pitch_stats
                enriched: List[NoteEvent] = []
                for ne in note_events:
                    off = ne.offset_time or (ne.onset_time + (ne.duration or 0.0))
                    f0_hz, f0_midi, stability, vfrac = _compute_pitch_stats(
                        ne.onset_time, off, aligned.timestamps, aligned.f0, aligned.voiced
                    )
                    enriched.append(NoteEvent(
                        onset_time=ne.onset_time,
                        offset_time=ne.offset_time,
                        duration=ne.duration,
                        pitch_hz=round(f0_hz, 4) if f0_hz is not None else None,
                        pitch_midi=round(f0_midi, 4) if f0_midi is not None else None,
                        pitch_stability=round(stability, 4) if stability is not None else None,
                        voiced_fraction=round(vfrac, 4) if vfrac is not None else None,
                    ))
                note_events = enriched

        # ── Lyric + word events from phoneme segments ───────────────────
        lyric_events = []
        word_events = []
        if result.phoneme_segments:
            lyric_events, word_events = build_lyric_events(
                result.phoneme_segments,
                timestamps=aligned.timestamps,
                word_gap_s=float(cfg.get("word_gap_ms", 100)) / 1000.0,
            )

        # ── Cross-event annotation ───────────────────────────────────────
        overlap_min = float(cfg.get("min_overlap_s", 0.01))
        if note_events and result.phoneme_segments:
            note_events = annotate_notes_with_phonemes(
                note_events, result.phoneme_segments, min_overlap_s=overlap_min
            )
        if word_events and note_events:
            word_events = annotate_words_with_notes(word_events, note_events, min_overlap_s=overlap_min)
        if lyric_events and note_events:
            lyric_events = annotate_lyrics_with_notes(lyric_events, note_events, min_overlap_s=overlap_min)

        # ── Phrase events ────────────────────────────────────────────────
        phrase_events = build_phrase_events(
            note_events,
            max_gap_s=float(cfg.get("phrase_gap_s", 0.5)),
            word_events=word_events or None,
        )

        # ── Voiced regions ───────────────────────────────────────────────
        voiced_regions = []
        if aligned.voiced is not None:
            voiced_regions = build_voiced_regions(
                aligned.timestamps,
                aligned.voiced,
                min_duration_s=float(cfg.get("min_region_duration_ms", 20)) / 1000.0,
            )

        # ── Alignment quality metadata ───────────────────────────────────
        alignment_meta: Dict[str, Any] = {}
        if note_events and result.phoneme_segments:
            alignment_meta["note_phoneme_coverage"] = score_note_phoneme_alignment(
                note_events, result.phoneme_segments
            )

        fused = FusedPerformanceRepresentation(
            audio_path=result.audio_path,
            duration_s=result.duration_s,
            sample_rate=result.sample_rate,
            hop_length=result.hop_length,
            timestamps=aligned.timestamps,
            f0=aligned.f0,
            voiced=aligned.voiced,
            note_events=note_events,
            lyric_events=lyric_events,
            word_events=word_events,
            phrase_events=phrase_events,
            voiced_regions=voiced_regions,
            phoneme_segments=result.phoneme_segments,
            alignment_metadata=alignment_meta,
            inference_metadata=result.metadata.copy(),
        )

        # ── Validation ───────────────────────────────────────────────────
        validate_fused_representation(fused, log_issues=True)

        result.fused = fused

        logger.debug(
            "[Pipeline] Fusion: %d notes, %d words, %d phrases, %d voiced regions "
            "(%.2fs)",
            len(note_events), len(word_events), len(phrase_events),
            len(voiced_regions), time.perf_counter() - t0,
        )
        return result

    def _run_scoring(self, result: UnifiedInferenceResult) -> UnifiedInferenceResult:
        """
        Compute PerformanceScoreReport and InterpretationSummary (Phase 7).

        Requires result.metrics to be populated by _run_metrics(). The scoring
        and interpretation steps are independent and both failures are caught
        separately so a scoring failure does not suppress interpretation.
        """
        t0 = time.perf_counter()
        try:
            from scoring.performance_scoring import build_performance_score_report
            from scoring.validation import validate_score_report

            score_report = build_performance_score_report(
                result.metrics,
                config=self._scoring_cfg,
            )
            validate_score_report(score_report, log_issues=True)
            result.scores = score_report
            logger.debug(
                "[Pipeline] Scoring done in %.3fs — overall=%.1f",
                time.perf_counter() - t0,
                score_report.overall_score if score_report.overall_score is not None else float("nan"),
            )
        except Exception as exc:
            logger.error("[Pipeline] Scoring failed: %s", exc)
            return result

        try:
            from scoring.interpretation import build_interpretation_summary

            interp_cfg = self._scoring_cfg.get("interpretation", {})
            result.interpretation = build_interpretation_summary(
                result.scores, config=interp_cfg
            )
        except Exception as exc:
            logger.error("[Pipeline] Interpretation failed: %s", exc)

        return result

    def _run_metrics(self, result: UnifiedInferenceResult) -> UnifiedInferenceResult:
        """
        Compute PerformanceMetricsReport from the alignment result (Phase 6).

        Requires result.alignment to be populated (i.e. both reference parsing
        and alignment must have succeeded). fused and reference are passed
        through for richer duration and pitch stability metrics.
        """
        t0 = time.perf_counter()
        try:
            from metrics.reporting import build_metrics_report
            from metrics.validation import validate_metrics_report

            metrics_report = build_metrics_report(
                alignment=result.alignment,
                fused=result.fused,
                reference=result.reference,
                config=self._metrics_cfg,
            )
            validate_metrics_report(metrics_report, log_issues=True)
            result.metrics = metrics_report
            logger.debug(
                "[Pipeline] Metrics computed in %.3fs — pitch=%s timing=%s",
                time.perf_counter() - t0,
                metrics_report.pitch is not None,
                metrics_report.timing is not None,
            )
        except Exception as exc:
            logger.error("[Pipeline] Metric computation failed: %s", exc)
        return result

    def _run_reference_alignment(
        self,
        result: UnifiedInferenceResult,
        musicxml_path: Optional[Union[str, Path]],
        textgrid_path: Optional[Union[str, Path]],
    ) -> UnifiedInferenceResult:
        """
        Parse reference files and align them against the fused inference output.

        Runs only when at least one of musicxml_path / textgrid_path is provided.
        Alignment requires a fused representation (fusion.enabled=true); if
        fusion is disabled, the reference is parsed but alignment is skipped.
        """
        t0 = time.perf_counter()
        try:
            from reference.reference_builder import build_reference_representation
            from reference.validation import validate_reference_representation

            phoneme_tier = self._ref_cfg.get("phoneme_tier", "phonemes")
            word_tier = self._ref_cfg.get("word_tier", "words")
            default_tempo = float(self._ref_cfg.get("default_tempo_bpm", 120.0))

            reference = build_reference_representation(
                musicxml_path=musicxml_path,
                textgrid_path=textgrid_path,
                phoneme_tier=phoneme_tier,
                word_tier=word_tier,
                default_tempo_bpm=default_tempo,
            )
            validate_reference_representation(reference, log_issues=True)
            result.reference = reference

            # Alignment requires a fused representation
            if result.fused is not None:
                from alignment.reference_alignment import align_performance

                tol_ms = float(self._ref_cfg.get("alignment_tolerance_ms", 50))
                alignment_cfg: Dict[str, Any] = {
                    "min_overlap_s": float(self._ref_cfg.get("min_overlap_s", 0.01)),
                    "max_onset_deviation_s": tol_ms / 1000.0,
                    "phoneme_min_overlap_s": 0.005,
                }
                result.alignment = align_performance(result.fused, reference, config=alignment_cfg)
            else:
                logger.info(
                    "[Pipeline] Reference parsed but alignment skipped "
                    "(fusion not enabled — set fusion.enabled=true in config)"
                )

            logger.debug(
                "[Pipeline] Reference alignment done (%.2fs)",
                time.perf_counter() - t0,
            )
        except Exception as exc:
            logger.error("[Pipeline] Reference alignment failed: %s", exc)

        return result

    def _align(self, result: UnifiedInferenceResult) -> UnifiedInferenceResult:
        """Merge all available streams onto the canonical frame grid."""
        if result.pitch_timestamps is None:
            return result

        n_canonical = len(result.pitch_timestamps)
        try:
            result.aligned = merge_model_outputs(
                n_canonical=n_canonical,
                pitch_times=result.pitch_timestamps.astype(np.float64),
                f0=result.f0,
                voiced=result.voiced,
                onset_times=(
                    result.onset_timestamps.astype(np.float64)
                    if result.onset_timestamps is not None else None
                ),
                onset_probs=result.onset_probs,
                offset_times=(
                    result.onset_timestamps.astype(np.float64)
                    if result.onset_timestamps is not None else None
                ),
                offset_probs=result.offset_probs,
                phoneme_segments=result.phoneme_segments,
            )
            logger.debug(
                f"[Pipeline] Aligned {n_canonical} canonical frames "
                f"(hop={HOP_LENGTH}, {SAMPLE_RATE/HOP_LENGTH:.0f}fps)"
            )
        except Exception as exc:
            logger.error(f"[Pipeline] Alignment failed: {exc}")
        return result

    # ------------------------------------------------------------------
    # Model accessors (lazy-load via registry)
    # ------------------------------------------------------------------

    def _get_pitch_model(self):
        if not self._registry.is_loaded("pitch"):
            from models.pitch.pipeline import PipelineConfig
            try:
                from configs.loader import load_model_config
                yaml_cfg = load_model_config("pitch")
                # Propagate device preference
                yaml_cfg.setdefault("device", {})["preference"] = self._device_manager.device_str
                pcfg = PipelineConfig.from_yaml(yaml_cfg)
                pcfg.export_json = bool(self._pl.get("export_pitch_json", False))
            except Exception:
                pcfg = None
            self._registry.load("pitch", pipeline_config=pcfg)
        return self._registry.get("pitch")

    def _get_phoneme_model(self):
        if not self._registry.is_loaded("phoneme"):
            try:
                from configs.loader import load_model_config
                cfg = load_model_config("phoneme")
            except Exception:
                cfg = {}
            self._registry.load("phoneme", config=cfg)
        return self._registry.get("phoneme")

    def _get_onset_model(self):
        if not self._registry.is_loaded("onset_offset"):
            ckpt = (
                self._pl.get("checkpoints", {}).get("onset_offset")
                or None
            )
            self._registry.load("onset_offset", checkpoint_path=ckpt)
        return self._registry.get("onset_offset")
