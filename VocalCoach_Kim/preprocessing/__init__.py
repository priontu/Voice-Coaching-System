"""preprocessing/ — Centralized audio preprocessing for VocalCoach."""

from preprocessing.timestamps import (
    SAMPLE_RATE,
    HOP_LENGTH,
    FRAME_DURATION,
    frames_to_times,
    times_to_frames,
    canonical_timestamps,
    duration_to_frames,
    frames_to_duration,
    samples_to_frames,
    align_to_grid,
    snap_to_frame,
    frame_overlap_ratio,
)
from preprocessing.framing import (
    n_frames_from_samples,
    n_frames_from_duration,
    frame_signal,
    vad_frame_boundaries,
    canonical_frame_count,
)
from preprocessing.normalization import (
    peak_normalize,
    rms_normalize,
    normalize_log_mel,
    clip_log_mel,
)
from preprocessing.spectrograms import (
    compute_log_mel,
    compute_log_mel_torch,
)
from preprocessing.audio_pipeline import (
    AudioPreprocessor,
    get_preprocessor,
)

__all__ = [
    # constants
    "SAMPLE_RATE",
    "HOP_LENGTH",
    "FRAME_DURATION",
    # timestamps
    "frames_to_times",
    "times_to_frames",
    "canonical_timestamps",
    "duration_to_frames",
    "frames_to_duration",
    "samples_to_frames",
    "align_to_grid",
    "snap_to_frame",
    "frame_overlap_ratio",
    # framing
    "n_frames_from_samples",
    "n_frames_from_duration",
    "frame_signal",
    "vad_frame_boundaries",
    "canonical_frame_count",
    # normalization
    "peak_normalize",
    "rms_normalize",
    "normalize_log_mel",
    "clip_log_mel",
    # spectrograms
    "compute_log_mel",
    "compute_log_mel_torch",
    # pipeline
    "AudioPreprocessor",
    "get_preprocessor",
]
