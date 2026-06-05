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

from PitchExtraction_v1 import PitchExtractionBatchConverter
from PitchExtraction_v1.alignment import (
    dtw_pitch_alignment,
    estimate_global_lag,
    local_lag_alignment,
    pitch_metrics,
)
from PitchExtraction_v1.nanopitch_runtime import NanoPitchRuntimeConfig
from hf_models.data.collator import VoiceCoachRawBatchCollator
from hf_models.data.manifest import Recording, save_manifest
from hf_models.data.runtime_dataset import VoiceCoachRuntimeDataset
from hf_models.preprocessing.torch_features import TorchAudioFeatureConfig


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English/EN-Alto-1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONTROL_GROUP = "Control_Group"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NanoPitch with raw, lag, local-lag, and DTW alignment diagnostics.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "PitchExtraction_v1" / "weights.pth")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decoder", choices=["argmax", "offline", "realtime"], default="argmax")
    parser.add_argument("--selection", choices=["round_robin", "first"], default="round_robin")
    parser.add_argument("--align-mode", choices=["raw", "global_lag", "local_lag", "dtw", "all"], default="all")
    parser.add_argument("--max-lag-sec", type=float, default=2.0)
    parser.add_argument("--local-window-sec", type=float, default=8.0)
    parser.add_argument("--local-hop-sec", type=float, default=4.0)
    parser.add_argument("--max-warp-sec", type=float, default=2.0)
    parser.add_argument("--max-dtw-frames", type=int, default=2500)
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contour_dir = args.output_dir / "contours"
    plot_dir = args.output_dir / "plots"
    contour_dir.mkdir(parents=True, exist_ok=True)
    if args.save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_control_group_triplets(args.dataset_root, args.limit, args.selection)
    if not recordings:
        raise RuntimeError(f"No Control_Group WAV/MusicXML triplets found under {args.dataset_root}")

    manifest_path = args.output_dir / "manifest.json"
    save_manifest(recordings, manifest_path)

    feature_config = TorchAudioFeatureConfig()
    nanopitch_config = NanoPitchRuntimeConfig(weights_path=args.weights, decoder=args.decoder)
    frame_seconds = nanopitch_config.hop_length / nanopitch_config.sample_rate
    max_lag_frames = max(1, round(args.max_lag_sec / frame_seconds))
    local_window_frames = max(1, round(args.local_window_sec / frame_seconds))
    local_hop_frames = max(1, round(args.local_hop_sec / frame_seconds))
    max_warp_frames = max(1, round(args.max_warp_sec / frame_seconds))

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
        feature_config=feature_config,
        nanopitch_config=nanopitch_config,
        device=args.device,
        use_amp_features=args.device.startswith("cuda"),
    )

    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = converter(raw_batch)
        rows.extend(
            process_batch(
                batch=batch,
                contour_dir=contour_dir,
                plot_dir=plot_dir,
                align_mode=args.align_mode,
                frame_seconds=frame_seconds,
                max_lag_frames=max_lag_frames,
                local_window_frames=local_window_frames,
                local_hop_frames=local_hop_frames,
                max_warp_frames=max_warp_frames,
                max_dtw_frames=args.max_dtw_frames,
                save_plots=args.save_plots,
            )
        )

    summary = {
        "dataset_root": str(args.dataset_root),
        "weights": str(args.weights),
        "device": args.device,
        "decoder": args.decoder,
        "group": CONTROL_GROUP,
        "align_mode": args.align_mode,
        "num_recordings": len(recordings),
        "manifest_path": str(manifest_path),
        "frame_seconds": frame_seconds,
        "settings": {
            "max_lag_sec": args.max_lag_sec,
            "local_window_sec": args.local_window_sec,
            "local_hop_sec": args.local_hop_sec,
            "max_warp_sec": args.max_warp_sec,
            "max_dtw_frames": args.max_dtw_frames,
        },
        "recordings": rows,
        "aggregate": aggregate_rows(rows),
        "notes": [
            "raw compares NanoPitch on its native frame grid against reference labels sampled at NanoPitch times.",
            "For GTSinger, runtime reference labels prefer the audio-aligned JSON notes when present.",
            "global_lag shifts the reference contour by one estimated lag.",
            "local_lag estimates a lag per overlapping window; use it as a timing diagnostic, not a final score.",
            "dtw allows local warping; it can reveal pitch agreement hidden by alignment errors, but it can overstate real onset timing quality.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary["aggregate"], indent=2))


def process_batch(
    *,
    batch: dict[str, Any],
    contour_dir: Path,
    plot_dir: Path,
    align_mode: str,
    frame_seconds: float,
    max_lag_frames: int,
    local_window_frames: int,
    local_hop_frames: int,
    max_warp_frames: int,
    max_dtw_frames: int,
    save_plots: bool,
) -> list[dict[str, Any]]:
    rows = []
    labels = batch["labels"]
    nanopitch = batch["nanopitch"]

    for index, metadata in enumerate(batch["metadata"]):
        recording_id = metadata["recording_id"]
        pred_f0 = nanopitch["f0_hz"][index].detach().float().cpu()
        pred_voiced = nanopitch["voiced"][index].detach().float().cpu() > 0.5
        ref_f0 = labels["nanopitch_ref_f0"][index].detach().float().cpu()
        ref_voiced = labels["nanopitch_ref_voicing"][index].detach().float().cpu() > 0.5
        num_frames = int(min(pred_f0.numel(), ref_f0.numel()))
        times_sec = torch.arange(num_frames, dtype=torch.float32) * frame_seconds

        metrics: dict[str, Any] = {"raw": pitch_metrics(pred_f0, pred_voiced, ref_f0, ref_voiced).to_dict()}
        shifted_ref_f0 = ref_f0[:num_frames]
        shifted_ref_voiced = ref_voiced[:num_frames]

        if align_mode in {"global_lag", "all"}:
            global_result = estimate_global_lag(
                pred_f0,
                pred_voiced,
                ref_f0,
                ref_voiced,
                max_lag_frames=max_lag_frames,
                frame_seconds=frame_seconds,
            )
            metrics["global_lag"] = global_result.to_dict()
            shifted_ref_f0 = global_result.shifted_ref_f0
            shifted_ref_voiced = global_result.shifted_ref_voiced

        if align_mode in {"local_lag", "all"}:
            local_result = local_lag_alignment(
                pred_f0,
                pred_voiced,
                ref_f0,
                ref_voiced,
                window_frames=local_window_frames,
                hop_frames=local_hop_frames,
                max_lag_frames=max_lag_frames,
                frame_seconds=frame_seconds,
            )
            metrics["local_lag"] = local_result.to_dict()

        if align_mode in {"dtw", "all"}:
            dtw_result = dtw_pitch_alignment(
                pred_f0,
                pred_voiced,
                ref_f0,
                ref_voiced,
                max_warp_frames=max_warp_frames,
                max_dtw_frames=max_dtw_frames,
            )
            metrics["dtw"] = dtw_result.to_dict()

        csv_path = contour_dir / f"{recording_id}_alignment.csv"
        write_contour_csv(
            csv_path,
            times_sec,
            pred_f0[:num_frames],
            pred_voiced[:num_frames],
            ref_f0[:num_frames],
            ref_voiced[:num_frames],
            shifted_ref_f0[:num_frames],
            shifted_ref_voiced[:num_frames],
        )

        plot_path = None
        if save_plots:
            plot_path = plot_dir / f"{recording_id}_alignment.png"
            write_plot(
                plot_path,
                times_sec,
                pred_f0[:num_frames],
                pred_voiced[:num_frames],
                ref_f0[:num_frames],
                ref_voiced[:num_frames],
                shifted_ref_f0[:num_frames],
                shifted_ref_voiced[:num_frames],
                recording_id,
            )

        rows.append(
            {
                "recording_id": recording_id,
                "wav_path": metadata["wav_path"],
                "frames": num_frames,
                "nanopitch_native_frames": int(nanopitch["f0_hz"][index].shape[0]),
                "metrics": metrics,
                "contour_csv": str(csv_path),
                "plot_png": str(plot_path) if plot_path is not None else None,
            }
        )

    return rows


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
                "global_shifted_reference_f0_hz",
                "global_shifted_reference_voiced",
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
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = times_sec.numpy()
    pred = pred_f0.numpy().copy()
    ref = ref_f0.numpy().copy()
    shifted = shifted_ref_f0.numpy().copy()

    pred[~pred_voiced.numpy()] = float("nan")
    ref[~ref_voiced.numpy()] = float("nan")
    shifted[~shifted_ref_voiced.numpy()] = float("nan")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, ref, label="Reference", linewidth=1.2, alpha=0.75)
    ax.plot(t, shifted, label="Reference global-lag shifted", linewidth=1.2, alpha=0.75)
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
    modes = sorted({mode for row in rows for mode in row["metrics"]})
    aggregate: dict[str, Any] = {"num_recordings": len(rows)}
    for mode in modes:
        aggregate[mode] = {
            "mean_pitch_acc_50": mean_metric(rows, mode, "pitch_acc_50"),
            "mean_mace_cents": mean_metric(rows, mode, "mace_cents"),
            "mean_pitch_rmse_cents": mean_metric(rows, mode, "pitch_rmse_cents"),
            "total_comparison_frames": sum(
                int(row["metrics"][mode].get("comparison_frames") or 0)
                for row in rows
                if mode in row["metrics"]
            ),
        }
        if mode == "global_lag":
            aggregate[mode]["lag_sec_values"] = [
                row["metrics"][mode]["lag_sec"] for row in rows if mode in row["metrics"]
            ]
        if mode == "local_lag":
            aggregate[mode]["mean_num_windows"] = mean_metric(rows, mode, "num_windows")
        if mode == "dtw":
            aggregate[mode]["frame_stride_values"] = [
                row["metrics"][mode]["frame_stride"] for row in rows if mode in row["metrics"]
            ]
    return aggregate


def mean_metric(rows: list[dict[str, Any]], mode: str, key: str) -> float | None:
    values = [row["metrics"][mode][key] for row in rows if mode in row["metrics"] and row["metrics"][mode].get(key) is not None]
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
