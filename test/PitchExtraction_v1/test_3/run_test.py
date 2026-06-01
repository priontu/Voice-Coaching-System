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
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from PitchExtraction_v1 import PitchExtractionBatchConverter
from PitchExtraction_v1.nanopitch_runtime import NanoPitchRuntimeConfig
from voice_coach.data.collator import VoiceCoachRawBatchCollator
from voice_coach.data.manifest import Recording, save_manifest
from voice_coach.data.runtime_dataset import VoiceCoachRuntimeDataset


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English/EN-Alto-1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONTROL_GROUP = "Control_Group"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NanoPitch vs MusicXML reference F0.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "PitchExtraction_v1" / "weights.pth")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decoder", choices=["argmax", "offline", "realtime"], default="argmax")
    parser.add_argument("--selection", choices=["round_robin", "first"], default="round_robin")
    parser.add_argument("--max-lag-sec", type=float, default=2.0)
    args = parser.parse_args()

    contour_dir = args.output_dir / "contours"
    plot_dir = args.output_dir / "plots"
    contour_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_control_group_triplets(args.dataset_root, args.limit, args.selection)
    if not recordings:
        raise RuntimeError(f"No Control_Group WAV/MusicXML triplets found under {args.dataset_root}")

    manifest_path = args.output_dir / "manifest.json"
    save_manifest(recordings, manifest_path)
    nanopitch_config = NanoPitchRuntimeConfig(weights_path=args.weights, decoder=args.decoder)

    dataset = VoiceCoachRuntimeDataset(recordings=recordings, cache_references=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=VoiceCoachRawBatchCollator(),
    )
    converter = PitchExtractionBatchConverter(
        nanopitch_config=nanopitch_config,
        device=args.device,
        use_amp_features=args.device.startswith("cuda"),
    )

    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = converter(raw_batch)
        rows.extend(process_batch(batch, contour_dir, plot_dir, args.max_lag_sec, nanopitch_config.hop_length / nanopitch_config.sample_rate))

    summary = {
        "dataset_root": str(args.dataset_root),
        "weights": str(args.weights),
        "device": args.device,
        "decoder": args.decoder,
        "group": CONTROL_GROUP,
        "num_recordings": len(recordings),
        "manifest_path": str(manifest_path),
        "recordings": rows,
        "aggregate": aggregate_rows(rows),
        "notes": [
            "Unshifted metrics compare NanoPitch to the runtime reference labels at the same timestamp.",
            "Pitch metrics compare NanoPitch on its native frame grid against reference labels sampled at NanoPitch times.",
            "For GTSinger, runtime reference labels prefer the audio-aligned JSON notes when present.",
            "Lag-corrected metrics shift the reference by the estimated best global lag.",
            "If lag-corrected metrics improve a lot, timing alignment is a major issue.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary["aggregate"], indent=2))


def process_batch(
    batch: dict[str, Any],
    contour_dir: Path,
    plot_dir: Path,
    max_lag_sec: float,
    frame_seconds: float,
) -> list[dict[str, Any]]:
    rows = []
    labels = batch["labels"]
    nanopitch = batch["nanopitch"]

    for index, metadata in enumerate(batch["metadata"]):
        recording_id = metadata["recording_id"]
        pred_f0 = nanopitch["f0_hz"][index].detach().float().cpu()
        pred_voiced = (nanopitch["voiced"][index].detach().float().cpu() > 0.5)
        ref_f0 = labels["nanopitch_ref_f0"][index].detach().float().cpu()
        ref_voiced = (labels["nanopitch_ref_voicing"][index].detach().float().cpu() > 0.5)
        num_frames = int(min(pred_f0.numel(), ref_f0.numel()))
        pred_f0 = pred_f0[:num_frames]
        pred_voiced = pred_voiced[:num_frames]
        ref_f0 = ref_f0[:num_frames]
        ref_voiced = ref_voiced[:num_frames]

        unshifted = pitch_metrics(pred_f0, pred_voiced, ref_f0, ref_voiced)
        lag_frames, shifted_ref_f0, shifted_ref_voiced = estimate_best_lag(
            pred_f0,
            pred_voiced,
            ref_f0,
            ref_voiced,
            max_lag_frames=max(1, int(max_lag_sec / frame_seconds)),
        )
        shifted = pitch_metrics(pred_f0, pred_voiced, shifted_ref_f0, shifted_ref_voiced)

        times_sec = torch.arange(num_frames, dtype=torch.float32) * frame_seconds
        csv_path = contour_dir / f"{recording_id}_contours.csv"
        plot_path = plot_dir / f"{recording_id}_contours.png"
        write_contour_csv(
            csv_path,
            times_sec,
            pred_f0,
            pred_voiced,
            ref_f0,
            ref_voiced,
            shifted_ref_f0,
            shifted_ref_voiced,
        )
        write_plot(
            plot_path,
            times_sec,
            pred_f0,
            pred_voiced,
            ref_f0,
            ref_voiced,
            shifted_ref_f0,
            shifted_ref_voiced,
            recording_id,
            lag_frames,
        )

        lag_sec = lag_frames * frame_seconds
        rows.append(
            {
                "recording_id": recording_id,
                "wav_path": metadata["wav_path"],
                "frames": num_frames,
                "nanopitch_native_frames": int(nanopitch["f0_hz"][index].shape[0]),
                "best_lag_frames": int(lag_frames),
                "best_lag_sec": float(lag_sec),
                "unshifted": unshifted,
                "lag_corrected": shifted,
                "contour_csv": str(csv_path),
                "plot_png": str(plot_path),
            }
        )

    return rows


def pitch_metrics(
    pred_f0: torch.Tensor,
    pred_voiced: torch.Tensor,
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
) -> dict[str, Any]:
    both = pred_voiced & ref_voiced & (pred_f0 > 0) & (ref_f0 > 0)
    if not both.any():
        return {
            "comparison_frames": 0,
            "pitch_acc_50": None,
            "mace_cents": None,
            "pitch_rmse_cents": None,
        }
    cents = cents_error(pred_f0[both], ref_f0[both])
    return {
        "comparison_frames": int(cents.numel()),
        "pitch_acc_50": float((cents.abs() <= 50).float().mean().item()),
        "mace_cents": float(cents.abs().mean().item()),
        "pitch_rmse_cents": float(torch.sqrt(cents.square().mean()).item()),
    }


def estimate_best_lag(
    pred_f0: torch.Tensor,
    pred_voiced: torch.Tensor,
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
    max_lag_frames: int,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    best_lag = 0
    best_error = float("inf")

    for lag in range(-max_lag_frames, max_lag_frames + 1):
        shifted_f0, shifted_voiced = shift_reference(ref_f0, ref_voiced, lag)
        both = pred_voiced & shifted_voiced & (pred_f0 > 0) & (shifted_f0 > 0)
        if int(both.sum().item()) < 20:
            continue
        error = float(cents_error(pred_f0[both], shifted_f0[both]).abs().median().item())
        if error < best_error:
            best_error = error
            best_lag = lag

    shifted_f0, shifted_voiced = shift_reference(ref_f0, ref_voiced, best_lag)
    return best_lag, shifted_f0, shifted_voiced


def shift_reference(
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
    lag_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_f0 = torch.zeros_like(ref_f0)
    shifted_voiced = torch.zeros_like(ref_voiced)
    if lag_frames == 0:
        return ref_f0.clone(), ref_voiced.clone()
    if lag_frames > 0:
        shifted_f0[lag_frames:] = ref_f0[:-lag_frames]
        shifted_voiced[lag_frames:] = ref_voiced[:-lag_frames]
    else:
        shift = -lag_frames
        shifted_f0[:-shift] = ref_f0[shift:]
        shifted_voiced[:-shift] = ref_voiced[shift:]
    return shifted_f0, shifted_voiced


def cents_error(pred_f0: torch.Tensor, ref_f0: torch.Tensor) -> torch.Tensor:
    return 1200.0 * torch.log2(torch.clamp(pred_f0, min=1e-6) / torch.clamp(ref_f0, min=1e-6))


def write_contour_csv(
    path: Path,
    times_sec: torch.Tensor,
    pred_f0: torch.Tensor,
    pred_voiced: torch.Tensor,
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
    shifted_ref_f0: torch.Tensor,
    shifted_ref_voiced: torch.Tensor,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_sec",
                "nanopitch_f0_hz",
                "nanopitch_voiced",
                "reference_f0_hz",
                "reference_voiced",
                "shifted_reference_f0_hz",
                "shifted_reference_voiced",
            ]
        )
        for row in zip(
            times_sec.tolist(),
            pred_f0.tolist(),
            pred_voiced.int().tolist(),
            ref_f0.tolist(),
            ref_voiced.int().tolist(),
            shifted_ref_f0.tolist(),
            shifted_ref_voiced.int().tolist(),
            strict=True,
        ):
            writer.writerow(row)


def write_plot(
    path: Path,
    times_sec: torch.Tensor,
    pred_f0: torch.Tensor,
    pred_voiced: torch.Tensor,
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
    shifted_ref_f0: torch.Tensor,
    shifted_ref_voiced: torch.Tensor,
    recording_id: str,
    lag_frames: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = times_sec.numpy()
    pred = pred_f0.numpy()
    ref = ref_f0.numpy()
    shifted = shifted_ref_f0.numpy()

    pred[~pred_voiced.numpy()] = float("nan")
    ref[~ref_voiced.numpy()] = float("nan")
    shifted[~shifted_ref_voiced.numpy()] = float("nan")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, ref, label="Reference", linewidth=1.2, alpha=0.75)
    ax.plot(t, shifted, label=f"Reference shifted ({lag_frames} frames)", linewidth=1.2, alpha=0.75)
    ax.plot(t, pred, label="NanoPitch", linewidth=1.0)
    ax.set_title(recording_id)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_ylim(50, 900)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_recordings": len(rows),
        "mean_unshifted_pitch_acc_50": mean_metric(rows, "unshifted", "pitch_acc_50"),
        "mean_lag_corrected_pitch_acc_50": mean_metric(rows, "lag_corrected", "pitch_acc_50"),
        "mean_unshifted_mace_cents": mean_metric(rows, "unshifted", "mace_cents"),
        "mean_lag_corrected_mace_cents": mean_metric(rows, "lag_corrected", "mace_cents"),
        "best_lag_sec_values": [row["best_lag_sec"] for row in rows],
    }


def mean_metric(rows: list[dict[str, Any]], section: str, key: str) -> float | None:
    values = [row[section][key] for row in rows if row[section][key] is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def discover_control_group_triplets(
    dataset_root: Path,
    limit: int | None,
    selection: str,
) -> list[Recording]:
    wavs = sorted(dataset_root.rglob("*.wav"))
    candidates: list[Recording] = []
    for wav_path in wavs:
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


def round_robin_by_technique(
    recordings: list[Recording],
    dataset_root: Path,
    limit: int | None,
) -> list[Recording]:
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
