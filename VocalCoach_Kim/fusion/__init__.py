"""
fusion/ — Cross-modal feature alignment and fusion utilities.

Phase 2:  alignment.py  — frame-rate resampling and multi-model merging
Phase 4:  note_events.py   — onset/offset → NoteEvent construction
          lyric_events.py  — PhonemeSegment → LyricEvent / WordEvent
          event_alignment.py — note↔phoneme mapping, phrase segmentation
          validation.py    — structural consistency checks
"""

from fusion.alignment import (
    align_mask_to_canonical,
    align_to_canonical,
    merge_model_outputs,
    resample_mask,
    resample_sequence,
    segments_to_frame_labels,
    snap_boundary,
)
from fusion.event_alignment import (
    align_phonemes_to_notes,
    align_words_to_notes,
    annotate_lyrics_with_notes,
    annotate_notes_with_phonemes,
    annotate_words_with_notes,
    build_phrase_events,
    build_voiced_regions,
    overlap_duration,
    overlap_fraction,
    score_note_phoneme_alignment,
    snap_event_boundaries,
)
from fusion.lyric_events import build_lyric_events, merge_word_events
from fusion.note_events import build_note_events, estimate_tempo
from fusion.validation import (
    ValidationIssue,
    ValidationReport,
    validate_fused_representation,
    validate_lyric_events,
    validate_note_events,
    validate_phrase_events,
    validate_temporal_regions,
    validate_word_events,
)

__all__ = [
    # alignment
    "align_mask_to_canonical",
    "align_to_canonical",
    "merge_model_outputs",
    "resample_mask",
    "resample_sequence",
    "segments_to_frame_labels",
    "snap_boundary",
    # event_alignment
    "align_phonemes_to_notes",
    "align_words_to_notes",
    "annotate_lyrics_with_notes",
    "annotate_notes_with_phonemes",
    "annotate_words_with_notes",
    "build_phrase_events",
    "build_voiced_regions",
    "overlap_duration",
    "overlap_fraction",
    "score_note_phoneme_alignment",
    "snap_event_boundaries",
    # lyric_events
    "build_lyric_events",
    "merge_word_events",
    # note_events
    "build_note_events",
    "estimate_tempo",
    # validation
    "ValidationIssue",
    "ValidationReport",
    "validate_fused_representation",
    "validate_lyric_events",
    "validate_note_events",
    "validate_phrase_events",
    "validate_temporal_regions",
    "validate_word_events",
]
