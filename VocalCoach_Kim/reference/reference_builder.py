"""
reference/reference_builder.py - Combine MusicXML and TextGrid into a unified reference.

build_reference_representation() is the single public entry point for creating
a ReferencePerformanceRepresentation from any combination of:
  - a MusicXML score (notes, tempo, time/key signature, optional lyrics)
  - a Praat TextGrid file (phoneme and word annotations)

When both sources are provided, phoneme/word information from the TextGrid is
merged into the representation already populated from MusicXML.

When only a TextGrid is provided, a bare ReferencePerformanceRepresentation is
returned containing phonemes and words but no notes.

When only a MusicXML is provided, the phoneme/word lists remain empty.

Phrase segmentation mirrors the prediction-side logic in
fusion/event_alignment.build_phrase_events(): inter-note gaps above
phrase_gap_s start a new phrase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from utils.types import (
    ReferencePerformanceRepresentation,
    ReferencePhoneme,
    ReferencePhrase,
    ReferenceWord,
)

logger = logging.getLogger(__name__)

_DEFAULT_PHRASE_GAP_S: float = 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_phrases(
    notes,
    words: List[ReferenceWord],
    phrase_gap_s: float,
) -> List[ReferencePhrase]:
    """
    Segment notes into phrases by inter-note gap.

    A new phrase starts whenever the gap between consecutive non-rest notes
    exceeds phrase_gap_s, or after each rest.
    """
    if not notes:
        return []

    phrases: List[ReferencePhrase] = []
    current_indices: List[int] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    prev_offset: Optional[float] = None

    # Build a simple lookup: word_idx → list of note indices with overlap
    word_by_time: List[ReferenceWord] = sorted(words, key=lambda w: w.start_time)

    def _flush(note_indices, start, end):
        if not note_indices:
            return
        # Which word indices fall within this phrase?
        w_indices = [
            w.word_idx for w in word_by_time
            if w.word_idx is not None and w.start_time < end and w.end_time > start
        ]
        phrases.append(ReferencePhrase(
            start_time=round(start, 6),
            end_time=round(end, 6),
            note_indices=list(note_indices),
            word_indices=w_indices,
            phrase_idx=len(phrases),
        ))

    for note in notes:
        if note.is_rest:
            _flush(current_indices, current_start, current_end)
            current_indices = []
            current_start = None
            current_end = None
            prev_offset = note.offset_time
            continue

        gap = (note.onset_time - prev_offset) if prev_offset is not None else 0.0
        if current_indices and gap > phrase_gap_s:
            _flush(current_indices, current_start, current_end)
            current_indices = []
            current_start = None
            current_end = None

        if current_start is None:
            current_start = note.onset_time
        current_end = note.offset_time
        current_indices.append(note.note_idx)
        prev_offset = note.offset_time

    _flush(current_indices, current_start, current_end)
    return phrases


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_reference_representation(
    musicxml_path: Optional[Union[str, Path]] = None,
    textgrid_path: Optional[Union[str, Path]] = None,
    phoneme_tier: str = "phonemes",
    word_tier: str = "words",
    default_tempo_bpm: float = 120.0,
    merge_ties: bool = True,
    include_rests: bool = True,
    phrase_gap_s: float = _DEFAULT_PHRASE_GAP_S,
    skip_silence: bool = True,
) -> ReferencePerformanceRepresentation:
    """
    Build a ReferencePerformanceRepresentation from optional MusicXML and TextGrid inputs.

    At least one of musicxml_path or textgrid_path must be provided.

    Args:
        musicxml_path:     Path to a MusicXML file.
        textgrid_path:     Path to a Praat TextGrid file.
        phoneme_tier:      Name of the phoneme tier in the TextGrid.
        word_tier:         Name of the word tier in the TextGrid.
        default_tempo_bpm: Fallback tempo if no MetronomeMark in the score.
        merge_ties:        Merge tied notes in MusicXML parsing.
        include_rests:     Include rest notes from MusicXML.
        phrase_gap_s:      Inter-note gap threshold for phrase segmentation.
        skip_silence:      Skip silence intervals in TextGrid parsing.

    Returns:
        ReferencePerformanceRepresentation

    Raises:
        ValueError: If neither musicxml_path nor textgrid_path is provided.
    """
    if musicxml_path is None and textgrid_path is None:
        raise ValueError(
            "build_reference_representation() requires at least one of "
            "musicxml_path or textgrid_path."
        )

    phonemes: List[ReferencePhoneme] = []
    words: List[ReferenceWord] = []

    # ── MusicXML ────────────────────────────────────────────────────────────
    if musicxml_path is not None:
        from reference.musicxml_parser import parse_musicxml
        ref = parse_musicxml(
            musicxml_path,
            default_tempo_bpm=default_tempo_bpm,
            merge_ties=merge_ties,
            include_rests=include_rests,
        )
    else:
        ref = ReferencePerformanceRepresentation(
            source_path=str(textgrid_path),
            duration_s=0.0,
        )

    # ── TextGrid ────────────────────────────────────────────────────────────
    if textgrid_path is not None:
        from reference.textgrid_parser import parse_textgrid
        phonemes, words = parse_textgrid(
            textgrid_path,
            phoneme_tier=phoneme_tier,
            word_tier=word_tier,
            skip_silence=skip_silence,
        )

        # Update duration_s if TextGrid extends beyond score
        if phonemes:
            tg_end = max(p.end_time for p in phonemes)
            ref = ReferencePerformanceRepresentation(
                source_path=ref.source_path,
                duration_s=max(ref.duration_s, tg_end),
                tempo_bpm=ref.tempo_bpm,
                time_signature=ref.time_signature,
                key_signature=ref.key_signature,
                notes=ref.notes,
                phonemes=phonemes,
                words=words,
                metadata=ref.metadata,
            )
        else:
            ref = ReferencePerformanceRepresentation(
                source_path=ref.source_path,
                duration_s=ref.duration_s,
                tempo_bpm=ref.tempo_bpm,
                time_signature=ref.time_signature,
                key_signature=ref.key_signature,
                notes=ref.notes,
                phonemes=phonemes,
                words=words,
                metadata=ref.metadata,
            )

    # ── Phrase segmentation ─────────────────────────────────────────────────
    non_rest_notes = [n for n in ref.notes if not n.is_rest]
    phrases = _build_phrases(non_rest_notes, ref.words, phrase_gap_s)

    final = ReferencePerformanceRepresentation(
        source_path=ref.source_path,
        duration_s=ref.duration_s,
        tempo_bpm=ref.tempo_bpm,
        time_signature=ref.time_signature,
        key_signature=ref.key_signature,
        notes=ref.notes,
        phonemes=ref.phonemes,
        words=ref.words,
        phrases=phrases,
        metadata={
            **ref.metadata,
            "phrase_gap_s": phrase_gap_s,
            "n_phrases": len(phrases),
            "sources": {
                "musicxml": str(musicxml_path) if musicxml_path else None,
                "textgrid": str(textgrid_path) if textgrid_path else None,
            },
        },
    )

    logger.info(
        "[reference_builder] Built reference: %d notes, %d phonemes, "
        "%d words, %d phrases (%.2fs)",
        len(final.notes), len(final.phonemes),
        len(final.words), len(final.phrases), final.duration_s,
    )
    return final
