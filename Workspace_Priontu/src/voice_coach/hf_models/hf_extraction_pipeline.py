from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hf_pitch_extractor import HFPitchExtractor
from .hf_phoneme_extractor import HFPhonemeExtractor
from .hf_vocalparse_extractor import HFVocalParseExtractor


@dataclass
class HFExtractionPipeline:
    pitch_extractor: HFPitchExtractor | None = None
    phoneme_extractor: HFPhonemeExtractor | None = None
    vocalparse_extractor: HFVocalParseExtractor | None = None

    def __post_init__(self):
        if self.pitch_extractor is None:
            self.pitch_extractor = HFPitchExtractor()
        if self.phoneme_extractor is None:
            self.phoneme_extractor = HFPhonemeExtractor()
        if self.vocalparse_extractor is None:
            self.vocalparse_extractor = HFVocalParseExtractor()

    def run(self, wav_path: str) -> dict[str, Any]:
        pitch = self.pitch_extractor.predict(wav_path)
        phonemes = self.phoneme_extractor.predict(wav_path)
        vocal = self.vocalparse_extractor.predict(wav_path)

        return {
            "pitch": pitch,
            "phonemes": phonemes,
            "vocalparse": vocal,
        }
