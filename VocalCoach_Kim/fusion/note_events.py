"""
fusion/note_events.py - Note event construction from onset/offset probability curves.

Transforms the raw onset/offset probability arrays produced by the CNN+BiLSTM
model into structured NoteEvent objects enriched with pitch statistics.

Pipeline position:
    aligned.onset_probs + aligned.offset_probs + aligned.f0 + aligned.voiced
                                      ↓
                        build_note_events()
                                      ↓
                          List[NoteEvent]  (with pitch_hz, pitch_midi,
                                            pitch_stability, voiced_fraction,
                                            onset/offset confidence)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from preprocessing.timestamps import HOP_LENGTH, SAMPLE_RATE
from utils.types import NoteEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _peak_pick(
    probs: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    min_distance_s: float,
    hop_length: int,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (peak_times, peak_probs) for local maxima above threshold.

    Uses scipy.signal.find_peaks with a minimum distance constraint converted
    from seconds to frames at the given hop_length/sample_rate.
    """
    try:
        from scipy.signal import find_peaks
        min_dist_frames = max(1, int(min_distance_s * sample_rate / hop_length))
        peaks, props = find_peaks(
            probs.astype(float),
            height=threshold,
            distance=min_dist_frames,
        )
    except ImportError:
        # Fallback: simple threshold crossing without distance constraint
        peaks = np.where(probs >= threshold)[0]

    if len(peaks) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float32)
    return timestamps[peaks].astype(np.float64), probs[peaks].astype(np.float32)


def _pair_onsets_offsets(
    onset_times: np.ndarray,
    onset_confs: np.ndarray,
    offset_times: np.ndarray,
    offset_confs: np.ndarray,
    max_duration_s: float,
    audio_duration_s: float,
) -> List[Tuple[float, float, float, float]]:
    """
    Greedy left-to-right pairing of onset peaks with offset peaks.

    Each onset is matched with the first unused offset that occurs strictly
    after it and within max_duration_s. If no valid offset is found, a
    pseudo-offset is synthesised (10 ms before the next onset, or onset+300 ms
    for the final note).

    Returns:
        List of (onset_time, offset_time, onset_conf, offset_conf).
    """
    pairs: List[Tuple[float, float, float, float]] = []
    used_offsets: set = set()

    for i, (ot, oc) in enumerate(zip(onset_times, onset_confs)):
        matched = False
        for j, (ft, fc) in enumerate(zip(offset_times, offset_confs)):
            if j in used_offsets:
                continue
            if ft > ot and (ft - ot) <= max_duration_s:
                pairs.append((float(ot), float(ft), float(oc), float(fc)))
                used_offsets.add(j)
                matched = True
                break

        if not matched:
            if i + 1 < len(onset_times):
                pseudo = float(onset_times[i + 1]) - 0.01
            else:
                pseudo = min(float(ot) + 0.3, audio_duration_s)
            if pseudo > ot:
                pairs.append((float(ot), pseudo, float(oc), 0.0))

    return pairs


def _compute_pitch_stats(
    onset_time: float,
    offset_time: float,
    timestamps: np.ndarray,
    f0: Optional[np.ndarray],
    voiced: Optional[np.ndarray],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute pitch statistics for a note spanning [onset_time, offset_time).

    Returns:
        (mean_f0_hz, pitch_midi, pitch_stability, voiced_fraction)
        Any element may be None if insufficient voiced frames exist.
    """
    if f0 is None:
        return None, None, None, None

    mask = (timestamps >= onset_time) & (timestamps < offset_time)
    n_frames = int(np.sum(mask))
    if n_frames == 0:
        return None, None, None, None

    f0_seg = f0[mask]

    if voiced is not None:
        voiced_seg = voiced[mask]
        voiced_fraction = float(np.mean(voiced_seg.astype(float)))
        voiced_f0 = f0_seg[voiced_seg & (f0_seg > 0)]
    else:
        voiced_fraction = float(np.mean((f0_seg > 0).astype(float)))
        voiced_f0 = f0_seg[f0_seg > 0]

    if len(voiced_f0) == 0:
        return None, None, None, voiced_fraction

    mean_f0 = float(np.mean(voiced_f0))
    pitch_stability = float(np.std(voiced_f0))

    if mean_f0 > 0:
        pitch_midi = float(69.0 + 12.0 * np.log2(mean_f0 / 440.0))
    else:
        pitch_midi = None

    return mean_f0, pitch_midi, pitch_stability, voiced_fraction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_note_events(
    timestamps: np.ndarray,
    onset_probs: np.ndarray,
    offset_probs: np.ndarray,
    f0: Optional[np.ndarray] = None,
    voiced: Optional[np.ndarray] = None,
    onset_threshold: float = 0.5,
    offset_threshold: float = 0.5,
    min_duration_s: float = 0.05,
    min_voiced_fraction: float = 0.0,
    max_duration_s: float = 10.0,
    min_distance_s: float = 0.05,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> List[NoteEvent]:
    """
    Build structured NoteEvent objects from onset/offset probability curves.

    All inputs must be aligned to the same canonical timestamp grid (i.e. they
    should come from FrameAlignedFeatures after merge_model_outputs()).

    Args:
        timestamps:          Canonical frame timestamps (N,) in seconds.
        onset_probs:         Onset probability curve (N,) ∈ [0, 1].
        offset_probs:        Offset probability curve (N,) ∈ [0, 1].
        f0:                  F0 values (N,) Hz; 0 = unvoiced. Optional.
        voiced:              Boolean voiced mask (N,). Optional.
        onset_threshold:     Minimum onset peak height.
        offset_threshold:    Minimum offset peak height.
        min_duration_s:      Discard notes shorter than this (seconds).
        min_voiced_fraction: Discard notes whose voiced fraction is below this.
        max_duration_s:      Discard paired notes longer than this (seconds).
        min_distance_s:      Minimum distance between peaks of the same type.
        hop_length:          Canonical hop size (for peak distance conversion).
        sample_rate:         Canonical sample rate.

    Returns:
        List of NoteEvent sorted by onset_time, with pitch metadata populated.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    onset_probs = np.asarray(onset_probs, dtype=np.float32)
    offset_probs = np.asarray(offset_probs, dtype=np.float32)

    if len(timestamps) == 0:
        return []

    audio_duration_s = float(timestamps[-1]) + (hop_length / sample_rate)

    # ── 1. Peak picking ─────────────────────────────────────────────────────
    onset_times, onset_confs = _peak_pick(
        onset_probs, timestamps, onset_threshold, min_distance_s, hop_length, sample_rate
    )
    offset_times, offset_confs = _peak_pick(
        offset_probs, timestamps, offset_threshold, min_distance_s, hop_length, sample_rate
    )

    if len(onset_times) == 0:
        logger.debug("[note_events] No onset peaks found above threshold %.2f", onset_threshold)
        return []

    # ── 2. Onset-offset pairing ─────────────────────────────────────────────
    pairs = _pair_onsets_offsets(
        onset_times, onset_confs,
        offset_times, offset_confs,
        max_duration_s=max_duration_s,
        audio_duration_s=audio_duration_s,
    )

    # ── 3. Build NoteEvent objects with pitch stats ─────────────────────────
    note_events: List[NoteEvent] = []
    for idx, (ot, ft, oc, fc) in enumerate(pairs):
        dur = ft - ot
        if dur < min_duration_s:
            continue

        mean_f0, pitch_midi, stability, voiced_frac = _compute_pitch_stats(
            ot, ft, timestamps, f0, voiced
        )

        if voiced_frac is not None and voiced_frac < min_voiced_fraction:
            continue

        confidence = float(np.sqrt(oc * fc)) if fc > 0 else float(oc) * 0.5

        note = NoteEvent(
            onset_time=round(ot, 6),
            offset_time=round(ft, 6),
            duration=round(dur, 6),
            pitch_hz=round(mean_f0, 4) if mean_f0 is not None else None,
            pitch_midi=round(pitch_midi, 4) if pitch_midi is not None else None,
            pitch_stability=round(stability, 4) if stability is not None else None,
            voiced_fraction=round(voiced_frac, 4) if voiced_frac is not None else None,
            confidence=round(confidence, 4),
            onset_confidence=round(oc, 4),
            offset_confidence=round(fc, 4),
            note_idx=len(note_events),
        )
        note_events.append(note)

    logger.debug(
        "[note_events] Built %d note events from %d onset / %d offset peaks",
        len(note_events), len(onset_times), len(offset_times),
    )
    return note_events


def estimate_tempo(
    note_events: List[NoteEvent],
    min_ioi_s: float = 0.1,
    max_ioi_s: float = 2.0,
) -> Optional[float]:
    """
    Estimate tempo in BPM from inter-onset intervals.

    Returns None if fewer than 2 notes are available or no IOIs fall within
    [min_ioi_s, max_ioi_s].
    """
    if len(note_events) < 2:
        return None

    onsets = sorted(n.onset_time for n in note_events)
    iois = [b - a for a, b in zip(onsets, onsets[1:])
            if min_ioi_s <= (b - a) <= max_ioi_s]
    if not iois:
        return None

    median_ioi = float(np.median(iois))
    return round(60.0 / median_ioi, 2)
