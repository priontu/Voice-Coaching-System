"""
reference/ - Ground-truth parsing and reference representation construction.

Phase 5 additions:
  musicxml_parser  - parse MusicXML scores into ReferenceNote lists
  textgrid_parser  - parse Praat TextGrid annotations into phoneme/word lists
  reference_builder - combine sources into ReferencePerformanceRepresentation
  validation       - structural validation of reference representations
"""

from reference.musicxml_parser import parse_musicxml
from reference.textgrid_parser import parse_textgrid
from reference.reference_builder import build_reference_representation
from reference.validation import validate_reference_representation

__all__ = [
    "parse_musicxml",
    "parse_textgrid",
    "build_reference_representation",
    "validate_reference_representation",
]
