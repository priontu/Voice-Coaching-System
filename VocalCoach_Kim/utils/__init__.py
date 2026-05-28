"""
utils/ - Shared infrastructure for VocalCoach.

Submodules:
  audio         WAV loading, resampling, normalization, framing
  types         Shared dataclasses (PhonemeSegment, PitchFrame, NoteEvent, …)
  device        PyTorch device detection and assignment
  logging_utils Structured logging setup
  checkpoints   Checkpoint save/load helpers
"""

from utils.audio import (
    TARGET_SAMPLE_RATE,
    audio_to_pcm16,
    frame_audio,
    generate_timestamps,
    load_audio,
    load_audio_torch,
)
from utils.device import get_device, get_torch_device
from utils.logging_utils import get_logger, setup_logging
from utils.types import (
    AudioFeatures,
    InferenceResult,
    NoteEvent,
    PhonemeSegment,
    PitchFrame,
    PitchResult,
)

__all__ = [
    # audio
    "TARGET_SAMPLE_RATE",
    "load_audio",
    "load_audio_torch",
    "frame_audio",
    "audio_to_pcm16",
    "generate_timestamps",
    # device
    "get_device",
    "get_torch_device",
    # logging
    "setup_logging",
    "get_logger",
    # types
    "AudioFeatures",
    "InferenceResult",
    "PhonemeSegment",
    "PitchFrame",
    "PitchResult",
    "NoteEvent",
]
