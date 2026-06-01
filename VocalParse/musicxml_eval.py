from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from voice_coach.data.gtsinger import parse_gtsinger_json_notes
from voice_coach.data.musicxml import NoteEvent, parse_musicxml_notes


@dataclass(frozen=True)
class MusicXMLSummary:
    midi_notes: list[int]
    bpm: float | None


def load_musicxml_summary(path: str | Path) -> MusicXMLSummary:
    notes = parse_musicxml_notes(path)
    return MusicXMLSummary(
        midi_notes=[note.midi for note in notes],
        bpm=parse_initial_bpm(path),
    )


def load_reference_summary(musicxml_path: str | Path, wav_path: str | Path | None = None) -> MusicXMLSummary:
    notes = load_reference_notes(musicxml_path, wav_path)
    return MusicXMLSummary(
        midi_notes=[note.midi for note in notes],
        bpm=parse_initial_bpm(musicxml_path),
    )


def load_reference_notes(musicxml_path: str | Path, wav_path: str | Path | None = None) -> list[NoteEvent]:
    if wav_path is not None:
        json_path = Path(wav_path).with_suffix(".json")
        if json_path.exists():
            notes = parse_gtsinger_json_notes(json_path)
            if notes:
                return notes
    return parse_musicxml_notes(musicxml_path)


def parse_initial_bpm(path: str | Path) -> float | None:
    root = ET.parse(path).getroot()
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag == "sound" and "tempo" in elem.attrib:
            return float(elem.attrib["tempo"])
        if tag == "per-minute" and elem.text:
            return float(elem.text)
    return None


def sequence_pitch_metrics(predicted: list[int], reference: list[int]) -> dict:
    alignment = needleman_wunsch(predicted, reference)
    matched_errors = [abs(p - r) for p, r in alignment if p is not None and r is not None]
    exact = [err == 0 for err in matched_errors]
    within_one = [err <= 1 for err in matched_errors]
    insertions = sum(1 for p, r in alignment if p is not None and r is None)
    deletions = sum(1 for p, r in alignment if p is None and r is not None)

    return {
        "predicted_pitch_count": len(predicted),
        "reference_note_count": len(reference),
        "aligned_pairs": len(matched_errors),
        "insertions": insertions,
        "deletions": deletions,
        "pitch_exact_accuracy": mean_bool(exact),
        "pitch_within_1_semitone": mean_bool(within_one),
        "pitch_mae_semitones": float(sum(matched_errors) / len(matched_errors)) if matched_errors else None,
    }


def bpm_metric(predicted_bpm: int | None, reference_bpm: float | None) -> dict:
    if predicted_bpm is None or reference_bpm is None:
        return {
            "predicted_bpm": predicted_bpm,
            "reference_bpm": reference_bpm,
            "bpm_abs_error": None,
        }
    return {
        "predicted_bpm": predicted_bpm,
        "reference_bpm": reference_bpm,
        "bpm_abs_error": abs(float(predicted_bpm) - float(reference_bpm)),
    }


def note_token_summary(note_tokens: list[str]) -> dict:
    return {
        "predicted_note_token_count": len(note_tokens),
        "unique_note_tokens": sorted(set(note_tokens)),
    }


def needleman_wunsch(predicted: list[int], reference: list[int]) -> list[tuple[int | None, int | None]]:
    match_score = 2
    mismatch_penalty = -1
    gap_penalty = -2
    n = len(predicted)
    m = len(reference)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        score[i][0] = i * gap_penalty
        back[i][0] = "up"
    for j in range(1, m + 1):
        score[0][j] = j * gap_penalty
        back[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag_value = match_score if predicted[i - 1] == reference[j - 1] else mismatch_penalty
            choices = [
                (score[i - 1][j - 1] + diag_value, "diag"),
                (score[i - 1][j] + gap_penalty, "up"),
                (score[i][j - 1] + gap_penalty, "left"),
            ]
            score[i][j], back[i][j] = max(choices, key=lambda item: item[0])

    alignment: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            alignment.append((predicted[i - 1], reference[j - 1]))
            i -= 1
            j -= 1
        elif move == "up":
            alignment.append((predicted[i - 1], None))
            i -= 1
        else:
            alignment.append((None, reference[j - 1]))
            j -= 1
    alignment.reverse()
    return alignment


def mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
