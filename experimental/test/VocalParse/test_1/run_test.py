#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from VocalParse.musicxml_eval import bpm_metric, load_reference_summary, note_token_summary, sequence_pitch_metrics
from VocalParse.runtime import VocalParseConfig
from hf_models.data.manifest import Recording, save_manifest


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English/EN-Alto-1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONTROL_GROUP = "Control_Group"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VocalParse on GTSinger Control_Group recordings.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default="pymaster/VocalParse")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--selection", choices=["round_robin", "first"], default="round_robin")
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)

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
        reference = load_reference_summary(recording.musicxml_path, recording.wav_path)
        raw_path = raw_dir / f"{recording.recording_id}.txt"
        raw_path.write_text(result.raw_text)

        rows.append(
            {
                "recording_id": recording.recording_id,
                "wav_path": str(recording.wav_path),
                "musicxml_path": str(recording.musicxml_path),
                "raw_output_path": str(raw_path),
                "lyrics_text": result.lyrics_text,
                "pitch_metrics": sequence_pitch_metrics(result.pitch_tokens, reference.midi_notes),
                "bpm_metric": bpm_metric(result.bpm, reference.bpm),
                "note_token_summary": note_token_summary(result.note_tokens),
                "predicted_bpm": result.bpm,
                "predicted_pitch_tokens_first_40": result.pitch_tokens[:40],
                "reference_midi_first_40": reference.midi_notes[:40],
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
            "VocalParse outputs symbolic lyrics, MIDI pitch tokens, note-value tokens, and BPM.",
            "For GTSinger, reference MIDI tokens prefer the audio-aligned JSON notes when present.",
            "It does not output onset/offset timestamps, so this test does not compute onset/offset metrics.",
            "The model card says it is primarily trained on Mandarin Chinese singing; English GTSinger may be out of domain.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary["aggregate"], indent=2))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_recordings": len(rows),
        "mean_pitch_mae_semitones": mean_nested(rows, "pitch_metrics", "pitch_mae_semitones"),
        "mean_pitch_exact_accuracy": mean_nested(rows, "pitch_metrics", "pitch_exact_accuracy"),
        "mean_pitch_within_1_semitone": mean_nested(rows, "pitch_metrics", "pitch_within_1_semitone"),
        "mean_bpm_abs_error": mean_nested(rows, "bpm_metric", "bpm_abs_error"),
    }


def mean_nested(rows: list[dict[str, Any]], section: str, key: str) -> float | None:
    values = [row[section][key] for row in rows if row[section][key] is not None]
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
