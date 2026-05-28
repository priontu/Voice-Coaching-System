"""
fusion/alignment.py - Frame-rate alignment and feature merging.

Extends models/pitch/alignment.py (which handles boolean VAD masks only) to
support continuous float sequences and multi-model output merging.

The canonical timeline is HOP_LENGTH=160 at 16 kHz (10 ms, 100 fps).
All per-model outputs at other frame rates are resampled here before being
combined into a FrameAlignedFeatures object.

Frame-rate reference:
  Model             native hop    fps
  ─────────────────────────────────────
  torchcrepe pitch     160       100      ← canonical
  Wav2Vec2 phoneme     320        50
  WebRTC VAD           320        50
  CNN+BiLSTM onset     256       ~62.5
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np

from preprocessing.timestamps import (
    HOP_LENGTH,
    SAMPLE_RATE,
    align_to_grid,
    canonical_timestamps,
    frame_overlap_ratio,
    times_to_frames,
)
from utils.types import FrameAlignedFeatures, InferenceResult, PhonemeSegment, TimestampedFeatureSequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boolean mask resampling
# ---------------------------------------------------------------------------

def resample_mask(
    src_times: np.ndarray,
    src_mask: np.ndarray,
    tgt_times: np.ndarray,
) -> np.ndarray:
    """
    Resample a boolean voiced/activity mask onto a new timestamp grid.

    Uses nearest-neighbour assignment to avoid introducing fractional values
    in a boolean signal.

    Args:
        src_times: Source frame timestamps, shape (N,).
        src_mask:  Source boolean array, shape (N,).
        tgt_times: Target frame timestamps, shape (M,).

    Returns:
        Boolean array, shape (M,).
    """
    result = align_to_grid(src_times, src_mask.astype(np.float32), tgt_times, kind="nearest")
    return result > 0.5


# ---------------------------------------------------------------------------
# Float sequence resampling
# ---------------------------------------------------------------------------

def resample_sequence(
    src_times: np.ndarray,
    src_values: np.ndarray,
    tgt_times: np.ndarray,
    kind: str = "linear",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Linearly interpolate a continuous feature sequence onto a new grid.

    Use this for F0, onset/offset probabilities, and any other real-valued
    per-frame signal. For boolean masks, use resample_mask() instead.

    Args:
        src_times:  Source timestamps, shape (N,).
        src_values: Source values, shape (N,) or (N, D).
        tgt_times:  Target timestamps, shape (M,).
        kind:       Interpolation kind: 'linear', 'nearest', 'zero'.
        fill_value: Value outside the source range.

    Returns:
        Interpolated values, shape (M,) or (M, D), same dtype as src_values.
    """
    return align_to_grid(src_times, src_values, tgt_times, kind=kind, fill_value=fill_value)


# ---------------------------------------------------------------------------
# Canonical-grid alignment
# ---------------------------------------------------------------------------

def align_to_canonical(
    src_times: np.ndarray,
    src_values: np.ndarray,
    n_canonical: int,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    kind: str = "linear",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Resample a sequence onto the canonical n_canonical-frame grid.

    Convenience wrapper around resample_sequence that builds the canonical
    timestamp array for you.

    Returns:
        Values resampled to shape (n_canonical,) or (n_canonical, D).
    """
    tgt_times = canonical_timestamps(n_canonical, hop_length=hop_length, sample_rate=sample_rate)
    return resample_sequence(src_times, src_values, tgt_times, kind=kind, fill_value=fill_value)


def align_mask_to_canonical(
    src_times: np.ndarray,
    src_mask: np.ndarray,
    n_canonical: int,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Resample a boolean mask onto the canonical n_canonical-frame grid.
    """
    tgt_times = canonical_timestamps(n_canonical, hop_length=hop_length, sample_rate=sample_rate)
    return resample_mask(src_times, src_mask, tgt_times)


# ---------------------------------------------------------------------------
# Phoneme segment → frame-label mapping
# ---------------------------------------------------------------------------

def segments_to_frame_labels(
    segments: List[PhonemeSegment],
    n_frames: int,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> List[str]:
    """
    Assign one phoneme label to each canonical frame.

    For each frame, the phoneme whose time segment has the largest overlap
    with the frame interval is selected. Frames with no overlapping segment
    receive the empty string.

    Args:
        segments:    List of PhonemeSegment objects sorted by start_time.
        n_frames:    Number of canonical frames.
        hop_length:  Canonical hop size in samples.
        sample_rate: Audio sample rate.

    Returns:
        List of str of length n_frames.
    """
    labels = [""] * n_frames

    for frame_idx in range(n_frames):
        best_phoneme = ""
        best_overlap = 0.0
        for seg in segments:
            overlap = frame_overlap_ratio(
                seg.start_time, seg.end_time,
                frame_idx,
                hop_length=hop_length,
                sample_rate=sample_rate,
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_phoneme = seg.phoneme
        labels[frame_idx] = best_phoneme

    return labels


# ---------------------------------------------------------------------------
# Boundary snapping
# ---------------------------------------------------------------------------

def snap_boundary(
    time_s: float,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    center: bool = True,
) -> float:
    """
    Snap a continuous event time to the nearest canonical frame boundary.

    Useful when converting note onset/offset times (detected by the
    onset/offset model) to exact frame indices on the canonical grid.
    """
    hop_sec = hop_length / sample_rate
    if center:
        frame = int(round((time_s - hop_sec / 2.0) / hop_sec))
    else:
        frame = int(round(time_s / hop_sec))
    snapped_frame = max(0, frame)
    t = snapped_frame * hop_sec
    if center:
        t += hop_sec / 2.0
    return t


# ---------------------------------------------------------------------------
# Multi-model output merger
# ---------------------------------------------------------------------------

def merge_model_outputs(
    n_canonical: int,
    pitch_times: Optional[np.ndarray] = None,
    f0: Optional[np.ndarray] = None,
    voiced: Optional[np.ndarray] = None,
    onset_times: Optional[np.ndarray] = None,
    onset_probs: Optional[np.ndarray] = None,
    offset_times: Optional[np.ndarray] = None,
    offset_probs: Optional[np.ndarray] = None,
    phoneme_segments: Optional[List[PhonemeSegment]] = None,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> FrameAlignedFeatures:
    """
    Merge outputs from all three models onto the canonical frame grid.

    Each model output is resampled to n_canonical frames at the given
    hop_length/sample_rate. Missing streams are left as None.

    Args:
        n_canonical:       Number of canonical frames (from pitch output).
        pitch_times:       Pitch model frame timestamps, shape (N_p,).
        f0:                F0 values in Hz, shape (N_p,); 0.0 = unvoiced.
        voiced:            Boolean voiced mask, shape (N_p,).
        onset_times:       Onset model frame timestamps, shape (N_o,).
        onset_probs:       Onset probability curve, shape (N_o,).
        offset_times:      Onset model frame timestamps, shape (N_o,).
        offset_probs:      Offset probability curve, shape (N_o,).
        phoneme_segments:  List of PhonemeSegment objects.
        hop_length:        Canonical hop size.
        sample_rate:       Canonical sample rate.

    Returns:
        FrameAlignedFeatures with all available streams on the same grid.
    """
    tgt = canonical_timestamps(n_canonical, hop_length=hop_length, sample_rate=sample_rate)

    aligned_f0: Optional[np.ndarray] = None
    aligned_voiced: Optional[np.ndarray] = None
    aligned_onset: Optional[np.ndarray] = None
    aligned_offset: Optional[np.ndarray] = None
    aligned_phonemes: Optional[List[str]] = None

    if pitch_times is not None and f0 is not None:
        aligned_f0 = resample_sequence(pitch_times, f0, tgt, kind="linear", fill_value=0.0)

    if pitch_times is not None and voiced is not None:
        aligned_voiced = resample_mask(pitch_times, voiced, tgt)

    if onset_times is not None and onset_probs is not None:
        aligned_onset = resample_sequence(
            onset_times, onset_probs, tgt, kind="linear", fill_value=0.0
        )

    if offset_times is not None and offset_probs is not None:
        aligned_offset = resample_sequence(
            offset_times, offset_probs, tgt, kind="linear", fill_value=0.0
        )

    if phoneme_segments is not None:
        aligned_phonemes = segments_to_frame_labels(
            phoneme_segments, n_canonical, hop_length=hop_length, sample_rate=sample_rate
        )

    logger.debug(
        f"[fusion] merged {n_canonical} canonical frames "
        f"(f0={'yes' if aligned_f0 is not None else 'no'}, "
        f"voiced={'yes' if aligned_voiced is not None else 'no'}, "
        f"onset={'yes' if aligned_onset is not None else 'no'}, "
        f"phonemes={'yes' if aligned_phonemes is not None else 'no'})"
    )

    return FrameAlignedFeatures(
        timestamps=tgt,
        f0=aligned_f0,
        voiced=aligned_voiced,
        onset_probs=aligned_onset,
        offset_probs=aligned_offset,
        phoneme_labels=aligned_phonemes,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )
