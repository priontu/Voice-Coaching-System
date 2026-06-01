from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voice_coach.data.musicxml import NoteEvent, midi_to_hz


def parse_gtsinger_json_notes(path: str | Path) -> list[NoteEvent]:
    """Read audio-aligned GTSinger note events from the sidecar JSON file."""

    rows: list[dict[str, Any]] = json.loads(Path(path).read_text())
    notes: list[NoteEvent] = []
    for row in rows:
        midi_values = row.get("note", [])
        starts = row.get("note_start", [])
        ends = row.get("note_end", [])
        durations = row.get("note_dur", [])

        for index, midi_value in enumerate(midi_values):
            midi = int(midi_value)
            if midi <= 0:
                continue
            if index >= len(starts):
                continue
            start_sec = float(starts[index])
            if index < len(ends):
                end_sec = float(ends[index])
            elif index < len(durations):
                end_sec = start_sec + float(durations[index])
            else:
                continue
            if end_sec <= start_sec:
                continue
            notes.append(
                NoteEvent(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    midi=midi,
                    frequency_hz=midi_to_hz(midi),
                )
            )

    return sorted(notes, key=lambda note: note.start_sec)
