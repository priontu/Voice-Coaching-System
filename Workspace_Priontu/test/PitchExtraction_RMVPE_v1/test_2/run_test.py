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

from PitchExtraction_RMVPE_v1 import RMVPEBatchConverter, RMVPERuntimeConfig
from PitchExtraction_v1.alignment import estimate_global_lag, local_lag_alignment, pitch_metrics
from voice_coach.data.collator import VoiceCoachRawBatchCollator
from voice_coach.data.manifest import Recording, save_manifest
from voice_coach.data.runtime_dataset import VoiceCoachRuntimeDataset


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English/EN-Alto-1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONTROL_GROUP = "Control_Group"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RMVPE post-processing/evaluation strategies on GTSinger.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--providers", nargs="*", default=None)
    parser.add_argument("--selection", choices=["round_robin", "first"], default="round_robin")
    parser.add_argument("--confidence-thresholds", type=float, nargs="+", default=[0.03, 0.10, 0.20, 0.35])
    parser.add_argument("--stable-trim-sec", type=float, default=0.08)
    parser.add_argument("--median-window-sec", type=float, default=0.05)
    parser.add_argument("--max-gap-sec", type=float, default=0.06)
    parser.add_argument("--max-jump-cents", type=float, default=700.0)
    parser.add_argument("--min-note-voiced-ratio", type=float, default=0.5)
    parser.add_argument("--max-lag-sec", type=float, default=0.4)
    parser.add_argument("--local-window-sec", type=float, default=3.0)
    parser.add_argument("--local-hop-sec", type=float, default=1.5)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

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
        device=args.device,
        use_amp_features=args.device.startswith("cuda"),
    )

    variant_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    variant_note_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    recording_rows: list[dict[str, Any]] = []

    for raw_batch in loader:
        batch = converter(raw_batch)
        rows, frame_parts, note_parts = process_batch(batch, raw_batch, args)
        recording_rows.extend(rows)
        for key, value in frame_parts.items():
            if value.numel():
                variant_parts[key].append(value.cpu())
        for key, value in note_parts.items():
            if value.numel():
                variant_note_parts[key].append(value.cpu())
        if args.save_predictions:
            write_prediction_csvs(batch, prediction_dir)

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
            "confidence_thresholds": args.confidence_thresholds,
            "stable_trim_sec": args.stable_trim_sec,
            "median_window_sec": args.median_window_sec,
            "max_gap_sec": args.max_gap_sec,
            "max_jump_cents": args.max_jump_cents,
            "min_note_voiced_ratio": args.min_note_voiced_ratio,
            "max_lag_sec": args.max_lag_sec,
            "local_window_sec": args.local_window_sec,
            "local_hop_sec": args.local_hop_sec,
        },
        "aggregate_frame_metrics": aggregate_variants(variant_parts),
        "aggregate_note_metrics": aggregate_variants(variant_note_parts),
        "recordings": recording_rows,
        "notes": [
            "Frame metrics compare estimated F0 to note-level JSON/MusicXML targets.",
            "Stable variants ignore note edges to avoid scoops, offsets, consonants, and transitions.",
            "Note median variants compare one median F0 per note interior to the note target.",
            "Lag variants are diagnostics for annotation timing mismatch, not final coaching scores.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary["aggregate_frame_metrics"], indent=2))
    print(json.dumps(summary["aggregate_note_metrics"], indent=2))


def process_batch(
    batch: dict[str, Any],
    raw_batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rows = []
    frame_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    note_parts: dict[str, list[torch.Tensor]] = defaultdict(list)

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
        frame_seconds = estimate_frame_seconds(times)

        stable_mask = stable_note_mask(times, notes, args.stable_trim_sec)
        variants = build_f0_variants(
            f0_raw=f0_raw,
            confidence=confidence,
            frame_seconds=frame_seconds,
            thresholds=args.confidence_thresholds,
            median_window_sec=args.median_window_sec,
            max_gap_sec=args.max_gap_sec,
            max_jump_cents=args.max_jump_cents,
        )

        row_metrics: dict[str, Any] = {}
        for name, (f0, voiced) in variants.items():
            cents = frame_cents(f0, voiced, ref_f0, ref_voiced)
            stable_cents = frame_cents(f0, voiced, ref_f0, ref_voiced & stable_mask)
            frame_parts[name].append(cents)
            frame_parts[f"{name}_stable"].append(stable_cents)
            row_metrics[name] = metrics_from_cents(cents)
            row_metrics[f"{name}_stable"] = metrics_from_cents(stable_cents)

            note_cents = note_median_cents(
                times=times,
                f0=f0,
                voiced=voiced,
                notes=notes,
                trim_sec=args.stable_trim_sec,
                min_voiced_ratio=args.min_note_voiced_ratio,
            )
            note_parts[f"{name}_note_median"].append(note_cents)
            row_metrics[f"{name}_note_median"] = metrics_from_cents(note_cents)

        base_name = f"conf_{format_threshold(args.confidence_thresholds[0])}"
        base_f0, base_voiced = variants[base_name]
        max_lag_frames = max(1, round(args.max_lag_sec / frame_seconds))
        global_result = estimate_global_lag(
            base_f0,
            base_voiced,
            ref_f0,
            ref_voiced,
            max_lag_frames=max_lag_frames,
            frame_seconds=frame_seconds,
        )
        frame_parts[f"{base_name}_global_lag"].append(
            frame_cents(base_f0, base_voiced, global_result.shifted_ref_f0, global_result.shifted_ref_voiced)
        )
        row_metrics[f"{base_name}_global_lag"] = global_result.to_dict()

        local_result = local_lag_alignment(
            base_f0,
            base_voiced,
            ref_f0,
            ref_voiced,
            window_frames=max(1, round(args.local_window_sec / frame_seconds)),
            hop_frames=max(1, round(args.local_hop_sec / frame_seconds)),
            max_lag_frames=max_lag_frames,
            frame_seconds=frame_seconds,
        )
        row_metrics[f"{base_name}_local_lag"] = local_result.to_dict()

        rows.append(
            {
                "recording_id": metadata["recording_id"],
                "wav_path": metadata["wav_path"],
                "rmvpe_frames": length,
                "reference_voiced_frames": int(ref_voiced.sum().item()),
                "stable_reference_voiced_frames": int((ref_voiced & stable_mask).sum().item()),
                "num_reference_notes": len(notes),
                "metrics": row_metrics,
            }
        )

    return rows, {key: torch.cat(value) if value else torch.empty(0) for key, value in frame_parts.items()}, {
        key: torch.cat(value) if value else torch.empty(0) for key, value in note_parts.items()
    }


def build_f0_variants(
    *,
    f0_raw: torch.Tensor,
    confidence: torch.Tensor,
    frame_seconds: float,
    thresholds: list[float],
    median_window_sec: float,
    max_gap_sec: float,
    max_jump_cents: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    variants = {}
    for threshold in thresholds:
        name = f"conf_{format_threshold(threshold)}"
        f0, voiced = apply_confidence(f0_raw, confidence, threshold)
        variants[name] = (f0, voiced)

        smoothed = median_filter_f0(f0, voiced, frame_seconds, median_window_sec)
        smoothed, smoothed_voiced = interpolate_short_gaps(smoothed, smoothed > 0, frame_seconds, max_gap_sec)
        smoothed, smoothed_voiced = remove_large_jumps(smoothed, smoothed_voiced, max_jump_cents)
        variants[f"{name}_smooth"] = (smoothed, smoothed_voiced)
    return variants


def apply_confidence(f0: torch.Tensor, confidence: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    voiced = (confidence >= threshold) & (f0 > 0)
    return torch.where(voiced, f0, torch.zeros_like(f0)), voiced


def median_filter_f0(f0: torch.Tensor, voiced: torch.Tensor, frame_seconds: float, window_sec: float) -> torch.Tensor:
    window = max(1, round(window_sec / frame_seconds))
    if window % 2 == 0:
        window += 1
    if window <= 1:
        return f0.clone()
    half = window // 2
    output = f0.clone()
    for index in range(f0.numel()):
        start = max(0, index - half)
        end = min(f0.numel(), index + half + 1)
        values = f0[start:end][voiced[start:end] & (f0[start:end] > 0)]
        if values.numel():
            output[index] = values.median()
    return torch.where(voiced, output, torch.zeros_like(output))


def interpolate_short_gaps(
    f0: torch.Tensor,
    voiced: torch.Tensor,
    frame_seconds: float,
    max_gap_sec: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_gap = max(1, round(max_gap_sec / frame_seconds))
    output = f0.clone()
    output_voiced = voiced.clone()
    indices = torch.nonzero(voiced & (f0 > 0), as_tuple=False).flatten()
    if indices.numel() < 2:
        return output, output_voiced
    for left, right in zip(indices[:-1].tolist(), indices[1:].tolist(), strict=True):
        gap = right - left - 1
        if 0 < gap <= max_gap:
            interp = torch.linspace(float(f0[left]), float(f0[right]), gap + 2)[1:-1]
            output[left + 1 : right] = interp
            output_voiced[left + 1 : right] = True
    return output, output_voiced


def remove_large_jumps(
    f0: torch.Tensor,
    voiced: torch.Tensor,
    max_jump_cents: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = f0.clone()
    output_voiced = voiced.clone()
    prev_value: float | None = None
    for index in range(f0.numel()):
        if not bool(output_voiced[index]) or float(output[index]) <= 0:
            continue
        current = float(output[index])
        if prev_value is not None:
            jump = abs(1200.0 * torch.log2(torch.tensor(current / max(prev_value, 1e-6))).item())
            if jump > max_jump_cents:
                output[index] = 0.0
                output_voiced[index] = False
                continue
        prev_value = current
    return output, output_voiced


def stable_note_mask(times: torch.Tensor, notes: list[Any], trim_sec: float) -> torch.Tensor:
    mask = torch.zeros_like(times, dtype=torch.bool)
    for note in notes:
        start = float(note.start_sec) + trim_sec
        end = float(note.end_sec) - trim_sec
        if end <= start:
            midpoint = (float(note.start_sec) + float(note.end_sec)) / 2.0
            start = midpoint
            end = midpoint
        mask |= (times >= start) & (times < end)
    return mask


def frame_cents(
    f0: torch.Tensor,
    voiced: torch.Tensor,
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
) -> torch.Tensor:
    both = voiced & ref_voiced & (f0 > 0) & (ref_f0 > 0)
    if not bool(both.any()):
        return torch.empty(0)
    return cents_error(f0[both], ref_f0[both])


def note_median_cents(
    *,
    times: torch.Tensor,
    f0: torch.Tensor,
    voiced: torch.Tensor,
    notes: list[Any],
    trim_sec: float,
    min_voiced_ratio: float,
) -> torch.Tensor:
    cents = []
    for note in notes:
        start = float(note.start_sec) + trim_sec
        end = float(note.end_sec) - trim_sec
        if end <= start:
            start = float(note.start_sec)
            end = float(note.end_sec)
        mask = (times >= start) & (times < end)
        if not bool(mask.any()):
            continue
        voiced_values = f0[mask & voiced & (f0 > 0)]
        if voiced_values.numel() / max(int(mask.sum().item()), 1) < min_voiced_ratio:
            continue
        pred = voiced_values.median()
        ref = torch.tensor(float(note.frequency_hz), dtype=torch.float32)
        cents.append(cents_error(pred.view(1), ref.view(1)))
    if not cents:
        return torch.empty(0)
    return torch.cat(cents)


def estimate_frame_seconds(times: torch.Tensor) -> float:
    if times.numel() >= 2:
        diffs = times[1:] - times[:-1]
        positive = diffs[diffs > 0]
        if positive.numel():
            return float(positive.median().item())
    return 0.01


def metrics_from_cents(cents: torch.Tensor) -> dict[str, Any]:
    if cents.numel() == 0:
        return {
            "comparison_frames": 0,
            "pitch_acc_50": None,
            "pitch_acc_100": None,
            "mace_cents": None,
            "median_abs_cents": None,
            "pitch_rmse_cents": None,
        }
    abs_cents = cents.abs()
    return {
        "comparison_frames": int(cents.numel()),
        "pitch_acc_50": float((abs_cents <= 50.0).float().mean().item()),
        "pitch_acc_100": float((abs_cents <= 100.0).float().mean().item()),
        "mace_cents": float(abs_cents.mean().item()),
        "median_abs_cents": float(abs_cents.median().item()),
        "pitch_rmse_cents": float(torch.sqrt(cents.square().mean()).item()),
    }


def aggregate_variants(parts: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    return {
        key: metrics_from_cents(torch.cat(value) if value else torch.empty(0))
        for key, value in sorted(parts.items())
    }


def cents_error(pred_f0: torch.Tensor, ref_f0: torch.Tensor) -> torch.Tensor:
    return 1200.0 * torch.log2(torch.clamp(pred_f0, min=1e-6) / torch.clamp(ref_f0, min=1e-6))


def format_threshold(value: float) -> str:
    return str(value).replace(".", "p")


def write_prediction_csvs(batch: dict[str, Any], prediction_dir: Path) -> None:
    rmvpe = batch["rmvpe"]
    labels = batch["labels"]
    for index, metadata in enumerate(batch["metadata"]):
        output_path = prediction_dir / f"{metadata['recording_id']}_rmvpe.csv"
        length = int(rmvpe["frame_lengths"][index].item())
        rows = zip(
            rmvpe["times"][index, :length].detach().cpu().tolist(),
            rmvpe["f0_hz"][index, :length].detach().cpu().tolist(),
            rmvpe["confidence"][index, :length].detach().cpu().tolist(),
            labels["rmvpe_ref_f0"][index, :length].detach().cpu().tolist(),
            labels["rmvpe_ref_voicing"][index, :length].detach().cpu().tolist(),
            strict=True,
        )
        with output_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_sec", "rmvpe_f0_hz", "confidence", "reference_f0_hz", "reference_voiced"])
            writer.writerows(rows)


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


if __name__ == "__main__":
    main()
