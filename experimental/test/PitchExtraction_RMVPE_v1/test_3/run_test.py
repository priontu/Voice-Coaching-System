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

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from PitchExtraction_RMVPE_v1 import RMVPEBatchConverter, RMVPECoachingScoreConfig, RMVPERuntimeConfig
from PitchExtraction_RMVPE_v1.scoring import (
    apply_confidence,
    frame_cents,
    metrics_from_cents,
    note_events_to_cents,
    note_median_pitch_events,
    stable_note_mask,
)
from hf_models.data.collator import VoiceCoachRawBatchCollator
from hf_models.data.manifest import Recording, save_manifest
from hf_models.data.runtime_dataset import VoiceCoachRuntimeDataset


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English/EN-Alto-1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONTROL_GROUP = "Control_Group"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep higher RMVPE confidence thresholds for recommended coaching scoring.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--providers", nargs="*", default=None)
    parser.add_argument("--selection", choices=["round_robin", "first"], default="round_robin")
    parser.add_argument("--start-threshold", type=float, default=0.40)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--num-thresholds", type=int, default=4)
    parser.add_argument("--stable-trim-sec", type=float, default=0.08)
    parser.add_argument("--min-note-voiced-ratio", type=float, default=0.5)
    parser.add_argument("--save-note-events", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_dir = args.output_dir / "note_events"
    if args.save_note_events:
        event_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [round(args.start_threshold + args.threshold_step * index, 10) for index in range(args.num_thresholds)]
    recordings = discover_control_group_triplets(args.dataset_root, args.limit, args.selection)
    if not recordings:
        raise RuntimeError(f"No Control_Group WAV/MusicXML triplets found under {args.dataset_root}")

    manifest_path = args.output_dir / "manifest.json"
    save_manifest(recordings, manifest_path)

    dataset = VoiceCoachRuntimeDataset(recordings=recordings, cache_references=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=VoiceCoachRawBatchCollator(),
    )
    converter = RMVPEBatchConverter(
        rmvpe_config=RMVPERuntimeConfig(
            model_path=args.model_path,
            confidence_threshold=0.0,
            providers=tuple(args.providers) if args.providers else None,
        ),
        coaching_score_config=RMVPECoachingScoreConfig(
            confidence_threshold=0.35,
            stable_trim_sec=args.stable_trim_sec,
            min_note_voiced_ratio=args.min_note_voiced_ratio,
        ),
        device=args.device,
        use_amp_features=args.device.startswith("cuda"),
    )

    frame_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    stable_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    note_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for raw_batch in loader:
        batch = converter(raw_batch)
        batch_rows, batch_frame, batch_stable, batch_note = process_batch(
            batch=batch,
            raw_batch=raw_batch,
            thresholds=thresholds,
            stable_trim_sec=args.stable_trim_sec,
            min_note_voiced_ratio=args.min_note_voiced_ratio,
            event_dir=event_dir if args.save_note_events else None,
        )
        rows.extend(batch_rows)
        append_parts(frame_parts, batch_frame)
        append_parts(stable_parts, batch_stable)
        append_parts(note_parts, batch_note)

    summary = {
        "dataset_root": str(args.dataset_root),
        "model_path": str(args.model_path) if args.model_path else None,
        "device": args.device,
        "providers": args.providers,
        "group": CONTROL_GROUP,
        "selection": args.selection,
        "num_recordings": len(recordings),
        "manifest_path": str(manifest_path),
        "settings": {
            "thresholds": thresholds,
            "stable_trim_sec": args.stable_trim_sec,
            "min_note_voiced_ratio": args.min_note_voiced_ratio,
        },
        "aggregate_frame_metrics": aggregate_parts(frame_parts),
        "aggregate_stable_frame_metrics": aggregate_parts(stable_parts),
        "aggregate_note_median_metrics": aggregate_parts(note_parts),
        "recordings": rows,
        "notes": [
            "This test sweeps four confidence thresholds above the previous best 0.35.",
            "Recommended coaching score is note-level median F0 on stable note interiors.",
            "Frame metrics are included only for comparison.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary["aggregate_note_median_metrics"], indent=2))


def process_batch(
    *,
    batch: dict[str, Any],
    raw_batch: dict[str, Any],
    thresholds: list[float],
    stable_trim_sec: float,
    min_note_voiced_ratio: float,
    event_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rows = []
    frame_parts: dict[str, torch.Tensor] = {}
    stable_parts: dict[str, torch.Tensor] = {}
    note_parts: dict[str, torch.Tensor] = {}

    rmvpe = batch["rmvpe"]
    labels = batch["labels"]

    for index, metadata in enumerate(batch["metadata"]):
        length = int(rmvpe["frame_lengths"][index].item())
        times = rmvpe["times"][index, :length].detach().float().cpu()
        f0_raw = rmvpe["f0_hz"][index, :length].detach().float().cpu()
        confidence = rmvpe["confidence"][index, :length].detach().float().cpu()
        ref_f0 = labels["rmvpe_ref_f0"][index, :length].detach().float().cpu()
        ref_voiced = labels["rmvpe_ref_voicing"][index, :length].detach().float().cpu() > 0.5
        notes = raw_batch["examples"][index].notes
        stable_mask = stable_note_mask(times, notes, stable_trim_sec)

        row = {
            "recording_id": metadata["recording_id"],
            "wav_path": metadata["wav_path"],
            "rmvpe_frames": length,
            "reference_voiced_frames": int(ref_voiced.sum().item()),
            "stable_reference_voiced_frames": int((ref_voiced & stable_mask).sum().item()),
            "num_reference_notes": len(notes),
            "thresholds": {},
        }

        for threshold in thresholds:
            key = f"conf_{format_threshold(threshold)}"
            f0, voiced = apply_confidence(f0_raw, confidence, threshold)
            frame_cents_value = frame_cents(f0, voiced, ref_f0, ref_voiced).cpu()
            stable_cents_value = frame_cents(f0, voiced, ref_f0, ref_voiced & stable_mask).cpu()
            events = note_median_pitch_events(
                times_sec=times,
                f0_hz=f0_raw,
                confidence=confidence,
                notes=notes,
                config=RMVPECoachingScoreConfig(
                    confidence_threshold=threshold,
                    stable_trim_sec=stable_trim_sec,
                    min_note_voiced_ratio=min_note_voiced_ratio,
                ),
            )
            note_cents_value = note_events_to_cents(events).cpu()

            frame_parts[key] = append_tensor(frame_parts.get(key), frame_cents_value)
            stable_parts[key] = append_tensor(stable_parts.get(key), stable_cents_value)
            note_parts[key] = append_tensor(note_parts.get(key), note_cents_value)
            row["thresholds"][key] = {
                "frame": metrics_from_cents(frame_cents_value),
                "stable_frame": metrics_from_cents(stable_cents_value),
                "note_median": metrics_from_cents(note_cents_value),
                "note_events": len(events),
            }

            if event_dir is not None:
                event_path = event_dir / f"{metadata['recording_id']}_{key}_note_events.csv"
                write_note_events(event_path, events)
                row["thresholds"][key]["note_event_csv"] = str(event_path)

        rows.append(row)
    return rows, frame_parts, stable_parts, note_parts


def append_parts(target: dict[str, list[torch.Tensor]], source: dict[str, torch.Tensor]) -> None:
    for key, value in source.items():
        if value.numel():
            target[key].append(value)


def append_tensor(existing: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    if existing is None or existing.numel() == 0:
        return value
    if value.numel() == 0:
        return existing
    return torch.cat([existing, value])


def aggregate_parts(parts: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    output = {}
    for key, values in sorted(parts.items()):
        output[key] = metrics_from_cents(torch.cat(values) if values else torch.empty(0))
    return output


def write_note_events(path: Path, events: list[dict[str, float | int]]) -> None:
    fields = [
        "note_index",
        "note_start_sec",
        "note_end_sec",
        "stable_start_sec",
        "stable_end_sec",
        "target_midi",
        "target_f0_hz",
        "median_f0_hz",
        "error_cents",
        "abs_error_cents",
        "voiced_ratio",
        "voiced_frames",
        "stable_frames",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def discover_control_group_triplets(
    dataset_root: Path,
    limit: int | None,
    selection: str,
) -> list[Recording]:
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
        if not textgrid_path.exists():
            textgrid_path = wav_path.with_suffix(".textgrid")
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


def format_threshold(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p")


if __name__ == "__main__":
    main()
