from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class HFPhonemeExtractor:
    """Lightweight phoneme extractor shim.

    This is a stub that will return empty outputs when heavy transformer
    dependencies are not available. Implement a proper Hugging Face
    Wav2Vec2-based extractor here when integrating models.
    """

    model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"

    def predict(self, wav_path: str) -> dict[str, List | np.ndarray]:
        # Placeholder implementation: return empty arrays to allow pipelines
        # to run without heavy dependencies. Integrate a real model as needed.
        return {
            "phonemes": [],
            "start_times": np.zeros((0,), dtype=np.float32),
            "end_times": np.zeros((0,), dtype=np.float32),
            "confidences": np.zeros((0,), dtype=np.float32),
        }
