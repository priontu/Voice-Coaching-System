"""High-level Hugging Face style extractor wrappers.

These are lightweight shims that try to use optional heavy-weight
dependencies when available and otherwise raise clear errors or return
best-effort empty outputs. They intentionally avoid importing large
libraries at package import time.
"""

__all__ = [
    "HFPitchExtractor",
    "HFPhonemeExtractor",
    "HFVocalParseExtractor",
    "HFOnsetOffsetModel",
    "HFExtractionPipeline",
]
