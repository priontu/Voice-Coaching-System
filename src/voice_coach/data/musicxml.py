from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


STEP_TO_SEMITONE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


@dataclass(frozen=True)
class NoteEvent:
    start_sec: float
    end_sec: float
    midi: int
    frequency_hz: float


def parse_musicxml_notes(path: str | Path, default_tempo_bpm: float = 120.0) -> list[NoteEvent]:
    """Parse a practical monophonic note timeline from a MusicXML score."""

    root = ET.parse(path).getroot()
    divisions = 1.0
    tempo_bpm = _find_initial_tempo(root) or default_tempo_bpm
    current_time_sec = 0.0
    notes: list[NoteEvent] = []

    part = root.find(_qname(root, "part"))
    if part is None:
        return notes

    for measure in part.findall(_qname(root, "measure")):
        measure_start_sec = current_time_sec
        measure_position_sec = 0.0
        measure_duration_sec = 0.0

        attributes = measure.find(_qname(root, "attributes"))
        if attributes is not None:
            div_el = attributes.find(_qname(root, "divisions"))
            if div_el is not None and div_el.text:
                divisions = float(div_el.text)

        seconds_per_division = 60.0 / (tempo_bpm * divisions)

        for child in list(measure):
            tag = _strip_namespace(child.tag)
            if tag == "attributes":
                div_el = child.find(_qname(root, "divisions"))
                if div_el is not None and div_el.text:
                    divisions = float(div_el.text)
                    seconds_per_division = 60.0 / (tempo_bpm * divisions)
            elif tag == "direction":
                tempo_bpm = _tempo_from_direction(root, child) or tempo_bpm
                seconds_per_division = 60.0 / (tempo_bpm * divisions)
            if tag == "backup":
                measure_position_sec -= _duration_seconds(root, child, seconds_per_division)
            elif tag == "forward":
                measure_position_sec += _duration_seconds(root, child, seconds_per_division)
                measure_duration_sec = max(measure_duration_sec, measure_position_sec)
            elif tag == "note":
                duration_sec = _duration_seconds(root, child, seconds_per_division)
                if child.find(_qname(root, "chord")) is None:
                    note_start = measure_start_sec + measure_position_sec
                    measure_position_sec += duration_sec
                    measure_duration_sec = max(measure_duration_sec, measure_position_sec)
                else:
                    note_start = measure_start_sec + measure_position_sec - duration_sec

                if child.find(_qname(root, "rest")) is not None:
                    continue

                midi = _midi_from_note(root, child)
                if midi is None:
                    continue
                notes.append(
                    NoteEvent(
                        start_sec=note_start,
                        end_sec=note_start + duration_sec,
                        midi=midi,
                        frequency_hz=midi_to_hz(midi),
                    )
                )
        current_time_sec = measure_start_sec + measure_duration_sec

    return sorted(notes, key=lambda note: note.start_sec)


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _duration_seconds(root: ET.Element, element: ET.Element, seconds_per_division: float) -> float:
    duration = element.find(_qname(root, "duration"))
    if duration is None or not duration.text:
        return 0.0
    return float(duration.text) * seconds_per_division


def _midi_from_note(root: ET.Element, note: ET.Element) -> int | None:
    pitch = note.find(_qname(root, "pitch"))
    if pitch is None:
        return None
    step_el = pitch.find(_qname(root, "step"))
    octave_el = pitch.find(_qname(root, "octave"))
    alter_el = pitch.find(_qname(root, "alter"))
    if step_el is None or octave_el is None or not step_el.text or not octave_el.text:
        return None

    step = STEP_TO_SEMITONE[step_el.text]
    alter = int(float(alter_el.text)) if alter_el is not None and alter_el.text else 0
    octave = int(octave_el.text)
    return 12 * (octave + 1) + step + alter


def _find_initial_tempo(root: ET.Element) -> float | None:
    for direction in root.iter(_qname(root, "direction")):
        tempo = _tempo_from_direction(root, direction)
        if tempo is not None:
            return tempo
    return None


def _tempo_from_direction(root: ET.Element, direction: ET.Element) -> float | None:
    sound = direction.find(_qname(root, "sound"))
    if sound is not None and "tempo" in sound.attrib:
        return float(sound.attrib["tempo"])

    per_minute = direction.find(f".//{_qname(root, 'per-minute')}")
    if per_minute is not None and per_minute.text:
        return float(per_minute.text)
    return None


def _qname(root: ET.Element, tag: str) -> str:
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0][1:]
        return f"{{{namespace}}}{tag}"
    return tag


def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1]
