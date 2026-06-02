"""
models/onset_offset/ - Note onset/offset detection via CNN + BiLSTM.

Primary entry point: OnsetOffsetInferenceModel (implements BaseInferenceModel).
For direct use: NoteDetector (full inference class with checkpoint loading).
"""

from models.onset_offset.detector import NoteDetector, OnsetOffsetInferenceModel

__all__ = ["NoteDetector", "OnsetOffsetInferenceModel"]
