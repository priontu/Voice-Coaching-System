"""
reference/musicxml_parser.py - MusicXML score parser.

Converts a MusicXML file into a ReferencePerformanceRepresentation using the
music21 library. All timestamps are returned in seconds (converted from
beats/measures via the score's MetronomeMark or a configurable default tempo).

Key behaviours:
  - Tempo: first MetronomeMark in the score is used; falls back to default_tempo_bpm.
  - Tied notes: consecutive notes connected by a tie are merged into one
    ReferenceNote with the combined duration.
  - Rests: included as ReferenceNote objects with is_rest=True and pitch_midi=None.
  - Lyrics: the first lyric syllable attached to a note is stored in lyric field.
  - Time signature and key signature: the first occurrence is captured.
  - Multiple parts: all parts are flattened and sorted by onset_time.

Usage:
    ref = parse_musicxml("score.xml")
    ref = parse_musicxml("score.xml", default_tempo_bpm=90.0)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

from utils.types import ReferenceNote, ReferencePerformanceRepresentation

logger = logging.getLogger(__name__)

# Quarter-note beats per minute used when the score has no MetronomeMark.
_DEFAULT_TEMPO: float = 120.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _beats_to_seconds(beats: float, tempo_bpm: float) -> float:
    """Convert quarter-note beat position to seconds."""
    return beats * (60.0 / tempo_bpm)


def _midi_to_name(midi: int) -> str:
    """Convert integer MIDI note number to a human-readable pitch name."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi // 12) - 1
    name = names[midi % 12]
    return f"{name}{octave}"


def _extract_tempo(score) -> float:
    """Return the BPM of the first MetronomeMark in the score, or _DEFAULT_TEMPO."""
    try:
        import music21
        marks = score.flatten().getElementsByClass(music21.tempo.MetronomeMark)
        for m in marks:
            if m.number is not None:
                return float(m.number)
    except Exception:
        pass
    return _DEFAULT_TEMPO


def _extract_time_signature(score) -> Optional[Tuple[int, int]]:
    """Return (numerator, denominator) of the first time signature found."""
    try:
        import music21
        sigs = score.flatten().getElementsByClass(music21.meter.TimeSignature)
        for ts in sigs:
            return (int(ts.numerator), int(ts.denominator))
    except Exception:
        pass
    return None


def _extract_key_signature(score) -> Optional[str]:
    """Return a human-readable key signature string, e.g. 'C major'."""
    try:
        import music21
        keys = score.flatten().getElementsByClass(music21.key.Key)
        for k in keys:
            return str(k)
        keys = score.flatten().getElementsByClass(music21.key.KeySignature)
        for k in keys:
            asKey = k.asKey()
            return str(asKey)
    except Exception:
        pass
    return None


def _merge_tied_notes(raw_notes: List[dict]) -> List[dict]:
    """
    Merge consecutive tied notes into single notes.

    A note is a tie continuation when 'tie_type' in ('continue', 'stop').
    Merging: extend the previous non-start note's offset_time and duration.
    """
    if not raw_notes:
        return []

    merged: List[dict] = []
    pending: Optional[dict] = None

    for note in raw_notes:
        tie_type = note.get("tie_type")  # 'start', 'continue', 'stop', or None

        if tie_type in ("continue", "stop") and pending is not None:
            # Extend the pending note's duration to cover this note's end
            pending["offset_time"] = note["offset_time"]
            pending["duration"] = pending["offset_time"] - pending["onset_time"]
            pending["is_tied"] = True
            if tie_type == "stop":
                merged.append(pending)
                pending = None
        else:
            if pending is not None:
                merged.append(pending)
            if tie_type == "start":
                pending = note.copy()
            else:
                merged.append(note)
                pending = None

    if pending is not None:
        merged.append(pending)

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_musicxml(
    path: Union[str, Path],
    default_tempo_bpm: float = _DEFAULT_TEMPO,
    merge_ties: bool = True,
    include_rests: bool = True,
) -> ReferencePerformanceRepresentation:
    """
    Parse a MusicXML file into a ReferencePerformanceRepresentation.

    Args:
        path:              Path to a .xml / .mxl / .musicxml file.
        default_tempo_bpm: Fallback tempo when the score has no MetronomeMark.
        merge_ties:        If True, consecutive tied notes are merged.
        include_rests:     If True, rests are included as ReferenceNote
                           objects with is_rest=True.

    Returns:
        ReferencePerformanceRepresentation with notes list populated.

    Raises:
        ImportError:  If music21 is not installed.
        FileNotFoundError: If the path does not exist.
        ValueError:   If the file cannot be parsed as a MusicXML score.
    """
    try:
        import music21
    except ImportError as exc:
        raise ImportError(
            "music21 is required for MusicXML parsing. "
            "Install it with: pip install music21"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MusicXML file not found: {path}")

    logger.info("[musicxml_parser] Parsing: %s", path.name)

    try:
        score = music21.converter.parse(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to parse MusicXML file {path}: {exc}") from exc

    # ── Score-level metadata ────────────────────────────────────────────────
    tempo_bpm = _extract_tempo(score) or default_tempo_bpm
    time_sig = _extract_time_signature(score)
    key_sig = _extract_key_signature(score)

    logger.debug(
        "[musicxml_parser] tempo=%.1f BPM, time_sig=%s, key=%s",
        tempo_bpm, time_sig, key_sig,
    )

    # ── Extract all notes and rests (flat, across all parts) ───────────────
    raw_notes: List[dict] = []
    flat = score.flatten()

    for element in flat.notesAndRests:
        onset_beats = float(element.offset)
        dur_beats = float(element.duration.quarterLength)
        onset_s = round(_beats_to_seconds(onset_beats, tempo_bpm), 6)
        offset_s = round(onset_s + _beats_to_seconds(dur_beats, tempo_bpm), 6)

        # Measure / beat
        measure_num: Optional[int] = None
        beat_in_measure: Optional[float] = None
        try:
            site = element.getContextByClass(music21.stream.Measure)
            if site is not None:
                measure_num = int(site.number)
                beat_in_measure = float(element.beat)
        except Exception:
            pass

        is_rest = element.isRest

        # Pitch
        pitch_midi: Optional[float] = None
        pitch_hz: Optional[float] = None
        pitch_name: Optional[str] = None
        if not is_rest and hasattr(element, "pitch"):
            try:
                midi_int = int(element.pitch.midi)
                pitch_midi = float(midi_int)
                pitch_hz = round(440.0 * (2.0 ** ((pitch_midi - 69.0) / 12.0)), 4)
                pitch_name = _midi_to_name(midi_int)
            except Exception:
                pass

        # Lyric
        lyric: Optional[str] = None
        if not is_rest and hasattr(element, "lyrics") and element.lyrics:
            try:
                lyric = element.lyrics[0].text
            except Exception:
                pass

        # Tie type
        tie_type: Optional[str] = None
        if hasattr(element, "tie") and element.tie is not None:
            tie_type = element.tie.type  # 'start', 'continue', 'stop'

        if is_rest and not include_rests:
            continue

        raw_notes.append({
            "onset_time": onset_s,
            "offset_time": offset_s,
            "duration": round(offset_s - onset_s, 6),
            "pitch_midi": pitch_midi,
            "pitch_hz": pitch_hz,
            "pitch_name": pitch_name,
            "lyric": lyric,
            "measure": measure_num,
            "beat": beat_in_measure,
            "duration_beats": round(dur_beats, 6),
            "is_rest": is_rest,
            "is_tied": False,
            "tie_type": tie_type,
        })

    # ── Sort by onset time (multi-part scores can be interleaved) ──────────
    raw_notes.sort(key=lambda n: (n["onset_time"], n.get("pitch_midi") or 0))

    # ── Merge tied notes ────────────────────────────────────────────────────
    if merge_ties:
        raw_notes = _merge_tied_notes(raw_notes)

    # ── Build ReferenceNote objects ─────────────────────────────────────────
    notes: List[ReferenceNote] = []
    for idx, n in enumerate(raw_notes):
        notes.append(ReferenceNote(
            onset_time=n["onset_time"],
            offset_time=n["offset_time"],
            duration=n["duration"],
            pitch_midi=n["pitch_midi"],
            pitch_hz=n["pitch_hz"],
            pitch_name=n["pitch_name"],
            lyric=n["lyric"],
            measure=n["measure"],
            beat=n["beat"],
            duration_beats=n["duration_beats"],
            is_rest=n["is_rest"],
            is_tied=n["is_tied"],
            note_idx=idx,
        ))

    # ── Infer total duration ────────────────────────────────────────────────
    duration_s = max((n.offset_time for n in notes), default=0.0)

    logger.info(
        "[musicxml_parser] Parsed %d notes (%.2fs, tempo=%.1f BPM)",
        len(notes), duration_s, tempo_bpm,
    )

    return ReferencePerformanceRepresentation(
        source_path=str(path),
        duration_s=round(duration_s, 4),
        tempo_bpm=round(tempo_bpm, 2),
        time_signature=time_sig,
        key_signature=key_sig,
        notes=notes,
        metadata={
            "parser": "musicxml_parser",
            "n_raw_elements": len(raw_notes),
            "merge_ties": merge_ties,
            "include_rests": include_rests,
        },
    )
