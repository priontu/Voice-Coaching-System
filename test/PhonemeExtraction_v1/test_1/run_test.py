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

from PhonemeExtraction_v1 import HFPhonemeExtractor
from PhonemeExtraction_v1.eval import compare_phoneme_boundaries
from voice_coach.data.manifest import Recording, save_manifest
from voice_coach.data.textgrid import phoneme_intervals


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English/EN-Alto-1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONTROL_GROUP = "Control_Group"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HF phoneme extraction on GTSinger Control_Group.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--selection", choices=["round_robin", "first"], default="round_robin")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pred_dir = args.output_dir / "predictions"
    match_dir = args.output_dir / "matches"
    pred_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_control_group_triplets(args.dataset_root, args.limit, args.selection)
    if not recordings:
        raise RuntimeError(f"No Control_Group WAV/TextGrid files found under {args.dataset_root}")
    manifest_path = args.output_dir / "manifest.json"
    save_manifest(recordings, manifest_path)

    extractor = HFPhonemeExtractor(model_name=args.model_name, device=args.device)
    rows = []
    for recording in recordings:
        prediction = extractor.predict(recording.wav_path)
        reference = phoneme_intervals(recording.textgrid_path) if recording.textgrid_path else []
        metrics = compare_phoneme_boundaries(prediction["phonemes"], reference)

        pred_path = pred_dir / f"{recording.recording_id}.json"
        pred_path.write_text(json.dumps(prediction, indent=2))
        match_path = match_dir / f"{recording.recording_id}_matches.csv"
        write_matches_csv(match_path, metrics["matches"])

        rows.append(
            {
                "recording_id": recording.recording_id,
                "wav_path": str(recording.wav_path),
                "textgrid_path": str(recording.textgrid_path),
                "prediction_path": str(pred_path),
                "matches_path": str(match_path),
                "transcript": prediction["transcript"],
                "metrics": {key: value for key, value in metrics.items() if key != "matches"},
            }
        )

    summary = {
        "dataset_root": str(args.dataset_root),
        "model_name": args.model_name,
        "group": CONTROL_GROUP,
        "num_recordings": len(recordings),
        "manifest_path": str(manifest_path),
        "aggregate": aggregate(rows),
        "recordings": rows,
        "notes": [
            "Boundaries are rough CTC frame-collapse estimates.",
            "This is not forced alignment and should be treated as a baseline only.",
            "Sequence comparison is by index, so insertions/deletions can inflate boundary error.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary["aggregate"], indent=2))


def write_matches_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "predicted_phoneme",
        "reference_phoneme",
        "predicted_start_sec",
        "predicted_end_sec",
        "reference_start_sec",
        "reference_end_sec",
        "start_error_sec",
        "end_error_sec",
        "boundary_error_sec",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_recordings": len(rows),
        "mean_boundary_mae_sec": mean_nested(rows, "boundary_mae_sec"),
        "mean_start_mae_sec": mean_nested(rows, "start_mae_sec"),
        "mean_end_mae_sec": mean_nested(rows, "end_mae_sec"),
        "mean_phoneme_sequence_exact_rate": mean_nested(rows, "phoneme_sequence_exact_rate"),
    }


def mean_nested(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["metrics"][key] for row in rows if row["metrics"][key] is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def discover_control_group_triplets(dataset_root: Path, limit: int | None, selection: str) -> list[Recording]:
    candidates: list[Recording] = []
    for wav_path in sorted(dataset_root.rglob("*.wav")):
        if wav_path.parent.name != CONTROL_GROUP:
            continue
        textgrid_path = wav_path.with_suffix(".TextGrid")
        if not textgrid_path.exists():
            textgrid_path = wav_path.with_suffix(".textgrid")
        if not textgrid_path.exists():
            continue
        musicxml_path = wav_path.with_suffix(".musicxml")
        if not musicxml_path.exists():
            musicxml_path = wav_path.with_suffix(".xml")
        relative = wav_path.relative_to(dataset_root).with_suffix("")
        candidates.append(
            Recording(
                recording_id=sanitize_recording_id(str(relative)),
                song_id=sanitize_recording_id(str(relative.parent)),
                wav_path=wav_path,
                musicxml_path=musicxml_path,
                textgrid_path=textgrid_path,
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

