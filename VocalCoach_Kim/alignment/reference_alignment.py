"""
alignment/reference_alignment.py - Prediction ↔ reference alignment engine.

Provides deterministic, greedy alignment between predicted VocalCoach events
(NoteEvent, LyricEvent, WordEvent from FusedPerformanceRepresentation) and
ground-truth reference annotations (ReferenceNote, ReferencePhoneme,
ReferenceWord from ReferencePerformanceRepresentation).

All alignments are:
  - Deterministic: output is fully determined by input and thresholds.
  - One-to-one: each predicted event is matched to at most one reference event
    and vice versa (greedy, best-overlap-first).
  - Configurable: thresholds exposed as parameters with sensible defaults.

Alignment strategy:
  1. Compute an overlap matrix between all predicted and reference intervals.
  2. Greedily assign matches in descending overlap order (Hungarian-free).
  3. Events below min_overlap_s or outside max_onset_deviation_s are unmatched.

Usage:
    from alignment.reference_alignment import align_performance
    result = align_performance(fused, reference)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from alignment.alignment_utils import (
    offset_deviation,
    onset_deviation,
    overlap_duration,
    overlap_fraction_of_a,
    pitch_deviation_cents,
)
from utils.types import (
    AlignmentResult,
    FusedPerformanceRepresentation,
    LyricEvent,
    NoteAlignmentMatch,
    NoteEvent,
    PhonemeAlignmentMatch,
    ReferencePerformanceRepresentation,
    ReferencePhoneme,
    WordAlignmentMatch,
    WordEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: greedy overlap matching
# ---------------------------------------------------------------------------

def _greedy_match_intervals(
    pred_intervals: List[Tuple[float, float]],
    ref_intervals: List[Tuple[float, float]],
    min_overlap_s: float,
    max_onset_deviation_s: Optional[float],
) -> Tuple[List[Tuple[int, int, float, float]], List[int], List[int]]:
    """
    Greedy one-to-one interval matching by descending overlap.

    Args:
        pred_intervals:          List of (start, end) for predicted events.
        ref_intervals:           List of (start, end) for reference events.
        min_overlap_s:           Minimum temporal overlap to form a match.
        max_onset_deviation_s:   Maximum |onset_time_pred - onset_time_ref|.
                                 Pass None to disable onset-deviation filtering.

    Returns:
        (matches, unmatched_pred, unmatched_ref)
        matches: list of (pred_idx, ref_idx, overlap_s, overlap_fraction)
    """
    # Build all candidate (pred_idx, ref_idx, overlap_s, overlap_frac) pairs
    candidates: List[Tuple[float, int, int, float]] = []
    for pi, (ps, pe) in enumerate(pred_intervals):
        for ri, (rs, re) in enumerate(ref_intervals):
            ov = overlap_duration(ps, pe, rs, re)
            if ov < min_overlap_s:
                continue
            if max_onset_deviation_s is not None and abs(ps - rs) > max_onset_deviation_s:
                continue
            # Fraction: shorter interval as denominator (for stability)
            dur_pred = pe - ps
            dur_ref = re - rs
            ref_dur = min(dur_pred, dur_ref) if min(dur_pred, dur_ref) > 0 else max(dur_pred, dur_ref)
            frac = ov / ref_dur if ref_dur > 0 else 0.0
            candidates.append((-ov, pi, ri, frac))  # negative for ascending sort = descending ov

    # Sort by descending overlap (ascending negative overlap)
    candidates.sort(key=lambda x: x[0])

    matched_pred: set = set()
    matched_ref: set = set()
    matches: List[Tuple[int, int, float, float]] = []

    for neg_ov, pi, ri, frac in candidates:
        if pi in matched_pred or ri in matched_ref:
            continue
        matches.append((pi, ri, -neg_ov, frac))
        matched_pred.add(pi)
        matched_ref.add(ri)

    unmatched_pred = [i for i in range(len(pred_intervals)) if i not in matched_pred]
    unmatched_ref = [i for i in range(len(ref_intervals)) if i not in matched_ref]
    return matches, unmatched_pred, unmatched_ref


# ---------------------------------------------------------------------------
# Public alignment functions
# ---------------------------------------------------------------------------

def align_notes(
    predicted: List[NoteEvent],
    reference,  # List[ReferenceNote]
    min_overlap_s: float = 0.01,
    max_onset_deviation_s: float = 0.5,
) -> Tuple[List[NoteAlignmentMatch], List[int], List[int]]:
    """
    Align predicted NoteEvents against reference ReferenceNotes.

    Filters reference rests before matching — rests are never matched to
    predicted notes.

    Args:
        predicted:               Predicted NoteEvent list.
        reference:               ReferenceNote list.
        min_overlap_s:           Minimum temporal overlap to count as a match.
        max_onset_deviation_s:   Maximum |pred_onset - ref_onset|.

    Returns:
        (matches, unmatched_pred_indices, unmatched_ref_indices)
    """
    ref_pitched = [(n, i) for i, n in enumerate(reference) if not n.is_rest]

    pred_intervals = [
        (n.onset_time, n.offset_time or n.onset_time + (n.duration or 0.0))
        for n in predicted
    ]
    ref_intervals = [
        (n.onset_time, n.offset_time)
        for n, _ in ref_pitched
    ]

    raw_matches, unmatched_pred, unmatched_ref_local = _greedy_match_intervals(
        pred_intervals, ref_intervals, min_overlap_s, max_onset_deviation_s
    )

    # Map local ref indices back to original reference list indices
    local_to_orig = {li: orig_i for li, (_, orig_i) in enumerate(ref_pitched)}
    unmatched_ref_orig = [local_to_orig[li] for li in unmatched_ref_local]

    # Also mark all original rest indices as unmatched
    rest_indices = [i for i, n in enumerate(reference) if n.is_rest]

    matches: List[NoteAlignmentMatch] = []
    for pi, ri_local, ov, frac in raw_matches:
        ri_orig = local_to_orig[ri_local]
        pred_n = predicted[pi]
        ref_n = ref_pitched[ri_local][0]
        pred_off = pred_n.offset_time or (pred_n.onset_time + (pred_n.duration or 0.0))
        pitch_cents = pitch_deviation_cents(pred_n.pitch_hz, ref_n.pitch_hz)
        matches.append(NoteAlignmentMatch(
            pred_idx=pi,
            ref_idx=ri_orig,
            overlap_s=round(ov, 6),
            overlap_fraction=round(frac, 4),
            onset_deviation_s=round(onset_deviation(pred_n.onset_time, ref_n.onset_time), 6),
            offset_deviation_s=round(offset_deviation(pred_off, ref_n.offset_time), 6),
            pitch_deviation_cents=round(pitch_cents, 2) if pitch_cents is not None else None,
        ))

    return matches, unmatched_pred, unmatched_ref_orig + rest_indices


def align_phonemes(
    predicted: List[LyricEvent],
    reference: List[ReferencePhoneme],
    min_overlap_s: float = 0.005,
    match_phoneme_label: bool = False,
) -> Tuple[List[PhonemeAlignmentMatch], List[int], List[int]]:
    """
    Align predicted LyricEvents against reference ReferencePhonemes.

    Args:
        predicted:            Predicted LyricEvent list.
        reference:            ReferencePhoneme list.
        min_overlap_s:        Minimum temporal overlap.
        match_phoneme_label:  If True, candidates with mismatched phoneme
                              strings are penalised (overlap halved for sorting)
                              but not excluded.

    Returns:
        (matches, unmatched_pred_indices, unmatched_ref_indices)
    """
    pred_intervals = [(e.start_time, e.end_time) for e in predicted]
    ref_intervals = [(p.start_time, p.end_time) for p in reference]

    # Build candidate pairs manually to support label-aware scoring
    candidates: List[Tuple[float, int, int, float]] = []
    for pi, (ps, pe) in enumerate(pred_intervals):
        for ri, (rs, re) in enumerate(ref_intervals):
            ov = overlap_duration(ps, pe, rs, re)
            if ov < min_overlap_s:
                continue
            ref_dur = min(pe - ps, re - rs)
            frac = ov / ref_dur if ref_dur > 0 else 0.0
            sort_ov = ov
            if match_phoneme_label and predicted[pi].phoneme != reference[ri].phoneme:
                sort_ov *= 0.5   # penalise label mismatch in sort order
            candidates.append((-sort_ov, pi, ri, frac, ov))

    candidates.sort(key=lambda x: x[0])

    matched_pred: set = set()
    matched_ref: set = set()
    matches: List[PhonemeAlignmentMatch] = []

    for neg_sort_ov, pi, ri, frac, raw_ov in candidates:
        if pi in matched_pred or ri in matched_ref:
            continue
        label_match = (predicted[pi].phoneme == reference[ri].phoneme)
        matches.append(PhonemeAlignmentMatch(
            pred_idx=pi,
            ref_idx=ri,
            overlap_s=round(raw_ov, 6),
            overlap_fraction=round(frac, 4),
            onset_deviation_s=round(onset_deviation(predicted[pi].start_time, reference[ri].start_time), 6),
            label_match=label_match,
        ))
        matched_pred.add(pi)
        matched_ref.add(ri)

    unmatched_pred = [i for i in range(len(predicted)) if i not in matched_pred]
    unmatched_ref = [i for i in range(len(reference)) if i not in matched_ref]
    return matches, unmatched_pred, unmatched_ref


def align_words(
    predicted: List[WordEvent],
    reference,  # List[ReferenceWord]
    min_overlap_s: float = 0.01,
    max_onset_deviation_s: float = 0.5,
) -> Tuple[List[WordAlignmentMatch], List[int], List[int]]:
    """
    Align predicted WordEvents against reference ReferenceWords.

    Returns:
        (matches, unmatched_pred_indices, unmatched_ref_indices)
    """
    pred_intervals = [(w.start_time, w.end_time) for w in predicted]
    ref_intervals = [(w.start_time, w.end_time) for w in reference]

    raw_matches, unmatched_pred, unmatched_ref = _greedy_match_intervals(
        pred_intervals, ref_intervals, min_overlap_s, max_onset_deviation_s
    )

    matches = [
        WordAlignmentMatch(
            pred_idx=pi,
            ref_idx=ri,
            overlap_s=round(ov, 6),
            overlap_fraction=round(frac, 4),
            onset_deviation_s=round(onset_deviation(predicted[pi].start_time, reference[ri].start_time), 6),
        )
        for pi, ri, ov, frac in raw_matches
    ]
    return matches, unmatched_pred, unmatched_ref


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def align_performance(
    fused: FusedPerformanceRepresentation,
    reference: ReferencePerformanceRepresentation,
    config: Optional[Dict[str, Any]] = None,
) -> AlignmentResult:
    """
    Align all event streams in a FusedPerformanceRepresentation against a
    ReferencePerformanceRepresentation.

    Args:
        fused:     Predicted performance from the inference pipeline.
        reference: Ground-truth reference from reference_builder.
        config:    Optional dict with alignment thresholds:
                     min_overlap_s          (default 0.01)
                     max_onset_deviation_s  (default 0.5)
                     phoneme_min_overlap_s  (default 0.005)
                     match_phoneme_label    (default False)
                     word_min_overlap_s     (default 0.01)

    Returns:
        AlignmentResult with all match lists and unmatched indices.
    """
    cfg = config or {}
    min_overlap = float(cfg.get("min_overlap_s", 0.01))
    max_dev = float(cfg.get("max_onset_deviation_s", 0.5))
    ph_min = float(cfg.get("phoneme_min_overlap_s", 0.005))
    match_label = bool(cfg.get("match_phoneme_label", False))
    w_min = float(cfg.get("word_min_overlap_s", 0.01))

    logger.info(
        "[reference_alignment] Aligning %d predicted notes vs %d reference notes",
        len(fused.note_events), len(reference.notes),
    )

    # ── Note alignment ──────────────────────────────────────────────────────
    note_matches, unmatched_pred_notes, unmatched_ref_notes = align_notes(
        fused.note_events, reference.notes,
        min_overlap_s=min_overlap,
        max_onset_deviation_s=max_dev,
    )

    # ── Phoneme alignment ───────────────────────────────────────────────────
    phoneme_matches, unmatched_pred_ph, unmatched_ref_ph = align_phonemes(
        fused.lyric_events, reference.phonemes,
        min_overlap_s=ph_min,
        match_phoneme_label=match_label,
    )

    # ── Word alignment ──────────────────────────────────────────────────────
    word_matches, unmatched_pred_words, unmatched_ref_words = align_words(
        fused.word_events, reference.words,
        min_overlap_s=w_min,
        max_onset_deviation_s=max_dev,
    )

    # ── Alignment quality metadata ──────────────────────────────────────────
    n_ref_pitched = sum(1 for n in reference.notes if not n.is_rest)
    alignment_meta: Dict[str, Any] = {
        "config": {
            "min_overlap_s": min_overlap,
            "max_onset_deviation_s": max_dev,
            "phoneme_min_overlap_s": ph_min,
        },
        "note": {
            "n_matched": len(note_matches),
            "n_unmatched_pred": len(unmatched_pred_notes),
            "n_unmatched_ref": len(unmatched_ref_notes),
            "n_ref_pitched": n_ref_pitched,
        },
        "phoneme": {
            "n_matched": len(phoneme_matches),
            "n_unmatched_pred": len(unmatched_pred_ph),
            "n_unmatched_ref": len(unmatched_ref_ph),
            "label_match_rate": (
                sum(1 for m in phoneme_matches if m.label_match) / len(phoneme_matches)
                if phoneme_matches else None
            ),
        },
        "word": {
            "n_matched": len(word_matches),
            "n_unmatched_pred": len(unmatched_pred_words),
            "n_unmatched_ref": len(unmatched_ref_words),
        },
    }

    if note_matches:
        devs = [abs(m.onset_deviation_s) for m in note_matches]
        alignment_meta["note"]["mean_abs_onset_deviation_s"] = round(
            sum(devs) / len(devs), 4
        )
        pitch_devs = [abs(m.pitch_deviation_cents) for m in note_matches
                      if m.pitch_deviation_cents is not None]
        if pitch_devs:
            alignment_meta["note"]["mean_abs_pitch_deviation_cents"] = round(
                sum(pitch_devs) / len(pitch_devs), 2
            )

    logger.info(
        "[reference_alignment] Matched: %d notes, %d phonemes, %d words",
        len(note_matches), len(phoneme_matches), len(word_matches),
    )

    return AlignmentResult(
        predicted_audio_path=fused.audio_path,
        reference_source_path=reference.source_path,
        note_matches=note_matches,
        phoneme_matches=phoneme_matches,
        word_matches=word_matches,
        unmatched_pred_notes=unmatched_pred_notes,
        unmatched_ref_notes=unmatched_ref_notes,
        unmatched_pred_phonemes=unmatched_pred_ph,
        unmatched_ref_phonemes=unmatched_ref_ph,
        unmatched_pred_words=unmatched_pred_words,
        unmatched_ref_words=unmatched_ref_words,
        alignment_metadata=alignment_meta,
    )
