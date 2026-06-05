#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from VocalParse import VocalParseTranscriber
from VocalParse.musicxml_eval import load_reference_notes, parse_initial_bpm
from VocalParse.runtime import VocalParseConfig
from hf_models.data.manifest import Recording, save_manifest


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English/EN-Alto-1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONTROL_GROUP = "Control_Group"


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive VocalParse pseudo onset/offsets and compare to MusicXML.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default="pymaster/VocalParse")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--selection", choices=["round_robin", "first"], default="round_robin")
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--start-offset-sec", type=float, default=0.0)
    args = parser.parse_args()

    raw_dir = args.output_dir / "raw_outputs"
    event_dir = args.output_dir / "note_events"
    raw_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_control_group_triplets(args.dataset_root, args.limit, args.selection)
    if not recordings:
        raise RuntimeError(f"No Control_Group WAV/MusicXML triplets found under {args.dataset_root}")
    manifest_path = args.output_dir / "manifest.json"
    save_manifest(recordings, manifest_path)

    transcriber = VocalParseTranscriber(
        VocalParseConfig(
            checkpoint=args.checkpoint,
            max_duration_sec=args.max_duration_sec,
        )
    )

    rows = []
    for recording in recordings:
        result = transcriber.transcribe(recording.wav_path)
        raw_path = raw_dir / f"{recording.recording_id}.txt"
        raw_path.write_text(result.raw_text)

        reference_notes = load_reference_notes(recording.musicxml_path, recording.wav_path)
        reference_bpm = parse_initial_bpm(recording.musicxml_path)
        bpm = result.bpm or int(reference_bpm or 120)
        predicted_events = pseudo_events_from_vocalparse(
            pitch_tokens=result.pitch_tokens,
            note_tokens=result.note_tokens,
            bpm=bpm,
            start_offset_sec=args.start_offset_sec,
        )
        aligned = align_events(predicted_events, reference_notes)
        metrics = boundary_metrics(aligned)
        event_csv = event_dir / f"{recording.recording_id}_pseudo_events.csv"
        write_event_csv(event_csv, aligned)

        rows.append(
            {
                "recording_id": recording.recording_id,
                "wav_path": str(recording.wav_path),
                "musicxml_path": str(recording.musicxml_path),
                "raw_output_path": str(raw_path),
                "event_csv": str(event_csv),
                "predicted_bpm": result.bpm,
                "reference_bpm": reference_bpm,
                "bpm_used_for_pseudo_timing": bpm,
                "predicted_pitch_count": len(result.pitch_tokens),
                "predicted_note_value_count": len(result.note_tokens),
                "reference_note_count": len(reference_notes),
                "metrics": metrics,
                "note_tokens_first_40": result.note_tokens[:40],
                "pitch_tokens_first_40": result.pitch_tokens[:40],
            }
        )

    summary = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": args.checkpoint,
        "group": CONTROL_GROUP,
        "num_recordings": len(recordings),
        "manifest_path": str(manifest_path),
        "aggregate": aggregate(rows),
        "recordings": rows,
        "notes": [
            "These are pseudo onset/offsets derived from VocalParse symbolic note values.",
            "For GTSinger, reference notes prefer the audio-aligned JSON note events when present.",
            "VocalParse does not output physical note start/end timestamps.",
            "Metrics should be interpreted as a rough symbolic-timing baseline only.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary["aggregate"], indent=2))


def pseudo_events_from_vocalparse(
    pitch_tokens: list[int],
    note_tokens: list[str],
    bpm: int | float,
    start_offset_sec: float = 0.0,
) -> list[dict[str, float | int | str]]:
    events = []
    time_sec = float(start_offset_sec)
    count = min(len(pitch_tokens), len(note_tokens))
    for index in range(count):
        duration_sec = note_token_to_seconds(note_tokens[index], bpm)
        start = time_sec
        end = start + duration_sec
        events.append(
            {
                "index": index,
                "pitch_midi": pitch_tokens[index],
                "note_token": note_tokens[index],
                "start_sec": start,
                "end_sec": end,
                "duration_sec": duration_sec,
            }
        )
        time_sec = end
    return events


def note_token_to_seconds(note_token: str, bpm: int | float) -> float:
    match = re.fullmatch(r"NOTE(_DOT)?_(\d+)", note_token)
    if not match:
        return 60.0 / float(bpm)
    dotted = match.group(1) is not None
    denominator = int(match.group(2))
    quarter_note_sec = 60.0 / float(bpm)
    duration = quarter_note_sec * (4.0 / denominator)
    if dotted:
        duration *= 1.5
    return duration


def align_events(predicted_events: list[dict], reference_notes: list) -> list[dict[str, Any]]:
    count = min(len(predicted_events), len(reference_notes))
    rows = []
    for index in range(count):
        pred = predicted_events[index]
        ref = reference_notes[index]
        rows.append(
            {
                "index": index,
                "pred_pitch_midi": pred["pitch_midi"],
                "ref_pitch_midi": ref.midi,
                "pred_note_token": pred["note_token"],
                "pred_start_sec": pred["start_sec"],
                "pred_end_sec": pred["end_sec"],
                "pred_duration_sec": pred["duration_sec"],
                "ref_start_sec": ref.start_sec,
                "ref_end_sec": ref.end_sec,
                "ref_duration_sec": ref.end_sec - ref.start_sec,
                "onset_error_sec": abs(float(pred["start_sec"]) - ref.start_sec),
                "offset_error_sec": abs(float(pred["end_sec"]) - ref.end_sec),
                "duration_error_sec": abs(float(pred["duration_sec"]) - (ref.end_sec - ref.start_sec)),
            }
        )
    return rows


def boundary_metrics(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    if not aligned:
        return {
            "aligned_note_count": 0,
            "onset_mae_sec": None,
            "offset_mae_sec": None,
            "duration_mae_sec": None,
            "pitch_exact_accuracy": None,
        }
    return {
        "aligned_note_count": len(aligned),
        "onset_mae_sec": mean([row["onset_error_sec"] for row in aligned]),
        "offset_mae_sec": mean([row["offset_error_sec"] for row in aligned]),
        "duration_mae_sec": mean([row["duration_error_sec"] for row in aligned]),
        "pitch_exact_accuracy": mean([1.0 if row["pred_pitch_midi"] == row["ref_pitch_midi"] else 0.0 for row in aligned]),
    }


def write_event_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        "pred_pitch_midi",
        "ref_pitch_midi",
        "pred_note_token",
        "pred_start_sec",
        "pred_end_sec",
        "pred_duration_sec",
        "ref_start_sec",
        "ref_end_sec",
        "ref_duration_sec",
        "onset_error_sec",
        "offset_error_sec",
        "duration_error_sec",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_recordings": len(rows),
        "mean_onset_mae_sec": mean_nested(rows, "onset_mae_sec"),
        "mean_offset_mae_sec": mean_nested(rows, "offset_mae_sec"),
        "mean_duration_mae_sec": mean_nested(rows, "duration_mae_sec"),
        "mean_pitch_exact_accuracy": mean_nested(rows, "pitch_exact_accuracy"),
    }


def mean_nested(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["metrics"][key] for row in rows if row["metrics"][key] is not None]
    return mean(values)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def discover_control_group_triplets(dataset_root: Path, limit: int | None, selection: str) -> list[Recording]:
    candidates: list[Recording] = []
    for wav_path in sorted(dataset_root.rglob("*.wav")):
        if wav_path.parent.name != CONTROL_GROUP:
            continue
        musicxml_path = wav_path.with_suffix(".musicxml")
        if not musicxml_path.exists():
            musicxml_path = wav_path.with_suffix(".xml")
        if not musicxml_path.exists():
            continue
        textgrid_path = wav_path.with_suffix(".TextGrid")
        relative = wav_path.relative_to(dataset_root).with_suffix("")
        candidates.append(
            Recording(
                recording_id=sanitize_recording_id(str(relative)),
                song_id=sanitize_recording_id(str(relative.parent)),
                wav_path=wav_path,
                musicxml_path=musicxml_path,
                textgrid_path=textgrid_path if textgrid_path.exists() else None,
            )
        )
    if selection == "first":
        return candidates[:limit] if limit is not None else candidates
    if selection == "round_robin":
        return round_robin_by_technique(candidates, dataset_root, limit)
    raise ValueError(f"Unsupported selection mode: {selection}")


def round_robin_by_technique(recordings: list[Recording], dataset_root: Path, limit: int | None) -> list[Recording]:
    grouped: dict[str, list[Recording]] = defaultdict(list)
    for recording in recordings:
        relative = recording.wav_path.relative_to(dataset_root)
        technique = relative.parts[0] if relative.parts else "unknown"
        grouped[technique].append(recording)
    selected: list[Recording] = []
    techniques = sorted(grouped)
    index = 0
    while True:
        added = False
        for technique in techniques:
            items = grouped[technique]
            if index < len(items):
                selected.append(items[index])
                added = True
                if limit is not None and len(selected) >= limit:
                    return selected
        if not added:
            return selected
        index += 1


def sanitize_recording_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    main()
