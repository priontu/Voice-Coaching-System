from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HFVocalParseExtractor:
    """Stub wrapper for VocalParse-style transcription.

    Returns simple empty/placeholder outputs when the optional model is
    not installed. Replace with a real integration when ready.
    """

    model_name: str = "pymaster/VocalParse"

    def predict(self, wav_path: str) -> dict[str, Any]:
        return {
            "lyrics": "",
            "note_tokens": [],
            "pitch_tokens": [],
            "bpm": None,
        }
