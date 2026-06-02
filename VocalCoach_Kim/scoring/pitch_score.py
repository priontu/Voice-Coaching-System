"""
scoring/pitch_score.py - Score extracted pitch data against a MusicXML reference.

Coordinates the full scoring workflow:
  1. Load pitch_data.json (from inference/run_pitch.py)
  2. Load MusicXML reference
  3. Compute frame-level and note-level metrics
  4. Optionally save visualizations

Usage:
    python scoring/pitch_score.py \\
        --pitch-json pitch_data.json \\
        --musicxml reference.musicxml

The scoring logic (metric computation) lives in metrics/pitch_metrics.py.
This file handles I/O, MusicXML parsing, and orchestration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from metrics.pitch_metrics import (
    compute_frame_pitch_metrics,
    compute_note_level_pitch_correctness,
    _hz_to_midi,
    _midi_to_hz,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pitch_data(json_path: str):
    """Load pitch_data.json and return (times, f0) arrays."""
    with open(json_path, "r") as f:
        data = json.load(f)

    times = np.array([frame["time"] for frame in data["frames"]])
    f0 = np.array([frame["f0"] for frame in data["frames"]])
    return times, f0


def get_tempo_seconds_per_beat(xml_path: str):
    """Return (bpm, seconds_per_beat) from MusicXML. Defaults to 84 BPM."""
    import music21

    score = music21.converter.parse(xml_path)
    tempo_marks = list(score.flatten().getElementsByClass(music21.tempo.MetronomeMark))

    bpm = float(tempo_marks[0].number) if tempo_marks and tempo_marks[0].number else 84.0
    return bpm, 60.0 / bpm


def load_musicxml_notes(xml_path: str):
    """Parse MusicXML notes into a list of dicts with start_beat, end_beat, hz, midi, lyric."""
    import music21

    score = music21.converter.parse(xml_path)
    try:
        score = score.stripTies(inPlace=False)
    except Exception:
        pass

    notes = []
    for element in score.flatten().notesAndRests:
        if isinstance(element, music21.note.Rest):
            continue
        if isinstance(element, music21.note.Note):
            start_beat = float(element.offset)
            end_beat = start_beat + float(element.quarterLength)
            midi = element.pitch.midi
            notes.append({
                "midi": midi,
                "hz": _midi_to_hz(midi),
                "start_beat": start_beat,
                "end_beat": end_beat,
                "duration_beats": end_beat - start_beat,
                "lyric": element.lyric if element.lyric else None,
            })

    return notes


def build_reference_f0(times: np.ndarray, notes: list, seconds_per_beat: float) -> np.ndarray:
    """Build frame-level reference F0 timeline from MusicXML notes."""
    reference_f0 = np.zeros_like(times)
    for note in notes:
        start_sec = note["start_beat"] * seconds_per_beat
        end_sec = note["end_beat"] * seconds_per_beat
        mask = (times >= start_sec) & (times < end_sec)
        reference_f0[mask] = note["hz"]
    return reference_f0


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_note_results(note_metrics: dict, output_path: str = "note_pitch_results.json") -> None:
    with open(output_path, "w") as f:
        json.dump(note_metrics, f, indent=2)
    print(f"Saved note-level results → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score extracted pitch data against a MusicXML reference."
    )
    parser.add_argument(
        "--pitch-json", "-p", default="pitch_data.json",
        help="Path to pitch_data.json produced by inference/run_pitch.py",
    )
    parser.add_argument(
        "--musicxml", "-m", default="test.musicxml",
        help="Path to the MusicXML reference file",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save note-level results to JSON (default: note_pitch_results.json)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate and save visualization plots",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("Loading extracted pitch data...")
    times, predicted_f0 = load_pitch_data(args.pitch_json)

    print("Loading MusicXML reference...")
    notes = load_musicxml_notes(args.musicxml)
    print(f"Loaded {len(notes)} reference notes")

    bpm, seconds_per_beat = get_tempo_seconds_per_beat(args.musicxml)
    print(f"Detected BPM: {bpm}  |  Seconds per beat: {seconds_per_beat:.4f}")

    print("Building reference pitch timeline...")
    reference_f0 = build_reference_f0(times, notes, seconds_per_beat)

    print("Computing frame-level pitch metrics...")
    frame_metrics = compute_frame_pitch_metrics(times, predicted_f0, reference_f0)

    if frame_metrics is None:
        print("No overlapping voiced frames between extracted pitch and MusicXML reference.")
        return

    print("\nFrame-Level Pitch Scoring Results")
    print("-" * 35)
    print(f"Pitch Accuracy ±50 cents: {frame_metrics['PitchAcc50']:.2f}%")
    print(f"Mean Absolute Cent Error: {frame_metrics['MACE']:.2f} cents")
    print(f"Pitch RMSE:               {frame_metrics['PitchRMSE']:.2f} cents")
    print(f"Compared voiced frames:   {frame_metrics['VoicedComparedFrames']}")

    for i, mm in enumerate(frame_metrics["WorstMismatches"], 1):
        print(f"\nWorst Mismatch #{i} @ {mm['time']:.2f}s | "
              f"Expected {mm['reference_hz']:.1f} Hz | "
              f"Got {mm['predicted_hz']:.1f} Hz | "
              f"Error {mm['cent_error']:.1f}¢ ({mm['direction']})")

    print("\nComputing note-level pitch correctness...")
    note_metrics = compute_note_level_pitch_correctness(
        times, predicted_f0, notes, seconds_per_beat
    )

    if note_metrics is None:
        print("No notes had enough voiced frames to score.")
        return

    print("\nNote-Level Pitch Correctness")
    print("-" * 30)
    print(f"NotePitchAcc50:     {note_metrics['NotePitchAcc50']:.2f}%")
    print(f"Correct notes:      {note_metrics['CorrectNotes']} / {note_metrics['ScoredNotes']}")
    print(f"Total ref notes:    {note_metrics['TotalReferenceNotes']}")

    out_path = args.output or "note_pitch_results.json"
    save_note_results(note_metrics, out_path)

    if args.visualize:
        try:
            from visualization.pitch_viz import (
                plot_pitch_contour,
                plot_cent_error,
                plot_note_level_errors,
            )
            plot_pitch_contour(
                times, predicted_f0, predicted_f0 > 0,
                reference_f0=reference_f0,
                save_path="pitch_vs_reference.png",
            )
            plot_cent_error(
                times, predicted_f0, reference_f0,
                save_path="cent_error_over_time.png",
            )
            plot_note_level_errors(
                note_metrics["NoteResults"],
                save_path="note_level_pitch_errors.png",
            )
        except Exception as exc:
            print(f"[Warning] Visualization failed: {exc}")


if __name__ == "__main__":
    main()
