"""
models/phoneme/ - Phoneme boundary detection via Wav2Vec2 + CTC alignment.

Primary entry point: PhonemeInferenceModel (implements BaseInferenceModel).
For direct pipeline use: extract_phoneme_boundaries_from_audio().
"""

from models.phoneme.phoneme_model import (
    PhonemeInferenceModel,
    PhonemeBoundaryConfig,
    extract_phoneme_boundaries_from_audio,
)

__all__ = [
    "PhonemeInferenceModel",
    "PhonemeBoundaryConfig",
    "extract_phoneme_boundaries_from_audio",
]
