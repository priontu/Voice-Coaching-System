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

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from PitchExtraction_RMVPE_v1 import RMVPEBatchConverter, RMVPERuntimeConfig
from hf_models.data.collator import VoiceCoachRawBatchCollator
from hf_models.data.manifest import Recording, save_manifest
from hf_models.data.runtime_dataset import VoiceCoachRuntimeDataset


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English/EN-Alto-1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONTROL_GROUP = "Control_Group"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RMVPE pitch contour plots + metrics on GTSinger Control_Group recordings."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--providers", nargs="*", default=None)
    parser.add_argument("--selection", choices=["round_robin", "first"], default="round_robin")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for contour plots. Install with:\n  pip install matplotlib"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
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
            confidence_threshold=args.confidence_threshold,
            providers=tuple(args.providers) if args.providers else None,
        ),
        device=args.device,
        use_amp_features=args.device.startswith("cuda"),
    )

    rows: list[dict[str, Any]] = []
    cents_parts: list[torch.Tensor] = []
    plot_paths: list[str] = []

    for raw_batch in loader:
        batch = converter(raw_batch)
        rows.extend(build_batch_rows(batch))
        cents = batch_cents(batch)
        if cents.numel() > 0:
            cents_parts.append(cents.cpu())
        if args.save_predictions:
            write_prediction_csvs(batch, prediction_dir)
        new_plots = plot_batch_contours(batch, raw_batch, plot_dir, args.dpi, plt, Line2D)
        plot_paths.extend(new_plots)

    summary = {
        "dataset_root": str(args.dataset_root),
        "model_path": str(args.model_path) if args.model_path else None,
        "device": args.device,
        "providers": args.providers,
        "confidence_threshold": args.confidence_threshold,
        "group": CONTROL_GROUP,
        "selection": args.selection,
        "num_recordings": len(recordings),
        "manifest_path": str(manifest_path),
        "recordings": rows,
        "aggregate_pitch_metrics": aggregate_pitch_metrics(cents_parts),
        "plot_paths": plot_paths,
        "notes": [
            "Each PNG shows RMVPE F0 (voiced frames), reference note segments, and confidence.",
            "Y axis is log-scale Hz; note boundaries are shown as vertical grey dashed lines.",
            "Reference f0 is built from GTSinger JSON/MusicXML note timings.",
            "Pitch metrics are identical to test_1: frame-level comparison on co-voiced frames.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"manifest: {manifest_path}")
    print(f"summary:  {summary_path}")
    print(f"plots:    {plot_dir}  ({len(plot_paths)} files)")
    print(json.dumps(summary["aggregate_pitch_metrics"], indent=2))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_batch_contours(
    batch: dict[str, Any],
    raw_batch: dict[str, Any],
    plot_dir: Path,
    dpi: int,
    plt: Any,
    Line2D: Any,
) -> list[str]:
    rmvpe = batch["rmvpe"]
    labels = batch["labels"]
    plot_paths: list[str] = []

    for index, metadata in enumerate(batch["metadata"]):
        length = int(rmvpe["frame_lengths"][index].item())
        times = rmvpe["times"][index, :length].detach().float().cpu().numpy()
        f0 = rmvpe["f0_hz"][index, :length].detach().float().cpu().numpy()
        voiced = rmvpe["voiced"][index, :length].detach().float().cpu().numpy() > 0.5
        confidence = rmvpe["confidence"][index, :length].detach().float().cpu().numpy()
        ref_f0 = labels["rmvpe_ref_f0"][index, :length].detach().float().cpu().numpy()
        ref_voiced = labels["rmvpe_ref_voicing"][index, :length].detach().float().cpu().numpy() > 0.5
        notes = raw_batch["examples"][index].notes

        plot_path = plot_dir / f"{metadata['recording_id']}_contour.png"
        _save_contour_plot(
            recording_id=metadata["recording_id"],
            times=times,
            f0=f0,
            voiced=voiced,
            confidence=confidence,
            ref_f0=ref_f0,
            ref_voiced=ref_voiced,
            notes=notes,
            plot_path=plot_path,
            dpi=dpi,
            plt=plt,
            Line2D=Line2D,
        )
        plot_paths.append(str(plot_path))
        print(f"  saved: {plot_path.name}")

    return plot_paths


def _save_contour_plot(
    *,
    recording_id: str,
    times: np.ndarray,
    f0: np.ndarray,
    voiced: np.ndarray,
    confidence: np.ndarray,
    ref_f0: np.ndarray,
    ref_voiced: np.ndarray,
    notes: list[Any],
    plot_path: Path,
    dpi: int,
    plt: Any,
    Line2D: Any,
) -> None:
    fig, (ax_pitch, ax_conf) = plt.subplots(
        2, 1,
        figsize=(14, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # --- reference note segments (horizontal bars at target Hz) ---
    for note in notes:
        ax_pitch.hlines(
            y=float(note.frequency_hz),
            xmin=float(note.start_sec),
            xmax=float(note.end_sec),
            colors="tab:orange",
            linewidth=4,
            alpha=0.55,
        )
        ax_pitch.axvline(
            float(note.start_sec),
            color="silver",
            linewidth=0.6,
            linestyle="--",
            alpha=0.6,
            zorder=0,
        )

    # --- RMVPE F0 (NaN gaps for unvoiced frames so the line breaks) ---
    f0_plot = np.where(voiced & (f0 > 0), f0, np.nan)
    ax_pitch.plot(times, f0_plot, color="tab:blue", linewidth=1.2, alpha=0.9)

    # --- legend ---
    legend_handles = [
        Line2D([0], [0], color="tab:blue", linewidth=1.5, label="RMVPE F0 (voiced)"),
        Line2D([0], [0], color="tab:orange", linewidth=4, alpha=0.55, label="Reference notes"),
    ]
    ax_pitch.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax_pitch.set_ylabel("F0 (Hz)")
    ax_pitch.set_yscale("log")
    ax_pitch.set_title(recording_id, fontsize=9)
    ax_pitch.grid(axis="y", linestyle=":", alpha=0.4)

    # y-axis limits: cover voiced RMVPE + reference with a small margin
    voiced_f0_vals = f0[voiced & (f0 > 0)]
    ref_f0_vals = ref_f0[ref_voiced & (ref_f0 > 0)]
    all_vals = np.concatenate([voiced_f0_vals, ref_f0_vals]) if (voiced_f0_vals.size and ref_f0_vals.size) else (
        voiced_f0_vals if voiced_f0_vals.size else ref_f0_vals
    )
    if all_vals.size:
        y_min = max(float(all_vals.min()) * 0.85, 50.0)
        y_max = float(all_vals.max()) * 1.18
        if y_max > y_min:
            ax_pitch.set_ylim(y_min, y_max)

    # --- confidence subplot ---
    ax_conf.fill_between(times, confidence, alpha=0.45, color="tab:green")
    ax_conf.plot(times, confidence, color="tab:green", linewidth=0.9, label="Confidence")
    ax_conf.axhline(
        y=_infer_threshold(confidence),
        color="tab:red",
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
        label=f"threshold≈{_infer_threshold(confidence):.2f}",
    )
    ax_conf.set_ylim(0.0, 1.05)
    ax_conf.set_ylabel("Confidence")
    ax_conf.set_xlabel("Time (s)")
    ax_conf.grid(axis="both", linestyle=":", alpha=0.3)
    ax_conf.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _infer_threshold(confidence: np.ndarray) -> float:
    # Show where voiced frames start — estimated as the lowest confidence among
    # frames that RMVPE considered voiced (f0 > 0 is set by the extractor when
    # confidence >= threshold). We approximate by finding a natural gap.
    if confidence.size == 0:
        return 0.35
    sorted_conf = np.sort(confidence)
    # Use the 30th percentile as a rough visual marker; caller can override dpi etc.
    return float(np.percentile(sorted_conf, 30))


# ---------------------------------------------------------------------------
# Metrics (identical to test_1)
# ---------------------------------------------------------------------------

def build_batch_rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    labels = batch["labels"]
    rmvpe = batch["rmvpe"]

    for index, metadata in enumerate(batch["metadata"]):
        f0 = rmvpe["f0_hz"][index]
        voiced = rmvpe["voiced"][index] > 0.5
        ref_f0 = labels["rmvpe_ref_f0"][index]
        ref_voiced = labels["rmvpe_ref_voicing"][index] > 0.5
        valid = rmvpe["attention_mask"][index]
        both = valid & voiced & ref_voiced & (f0 > 0) & (ref_f0 > 0)
        cents = cents_error(f0[both], ref_f0[both])
        positive = f0[valid & voiced & (f0 > 0)]

        rows.append(
            {
                "recording_id": metadata["recording_id"],
                "song_id": metadata["song_id"],
                "wav_path": metadata["wav_path"],
                "rmvpe_frames": int(rmvpe["frame_lengths"][index].item()),
                "reference_voiced_frames": int((valid & ref_voiced).sum().item()),
                "rmvpe_voiced_frames": int((valid & voiced).sum().item()),
                "rmvpe_f0_min_hz": float(positive.min().item()) if positive.numel() else 0.0,
                "rmvpe_f0_max_hz": float(positive.max().item()) if positive.numel() else 0.0,
                "pitch_acc_50": float((cents.abs() <= 50.0).float().mean().item()) if cents.numel() else None,
                "pitch_acc_100": float((cents.abs() <= 100.0).float().mean().item()) if cents.numel() else None,
                "mace_cents": float(cents.abs().mean().item()) if cents.numel() else None,
                "median_abs_cents": float(cents.abs().median().item()) if cents.numel() else None,
                "pitch_rmse_cents": float(torch.sqrt(cents.square().mean()).item()) if cents.numel() else None,
                "raw_chroma_acc_50": raw_chroma_acc_50(cents),
                "octave_error_rate": octave_error_rate(cents),
                "comparison_frames": int(cents.numel()),
            }
        )
    return rows


def batch_cents(batch: dict[str, Any]) -> torch.Tensor:
    labels = batch["labels"]
    rmvpe = batch["rmvpe"]
    both = (
        rmvpe["attention_mask"]
        & (rmvpe["voiced"] > 0.5)
        & (labels["rmvpe_ref_voicing"] > 0.5)
        & (rmvpe["f0_hz"] > 0)
        & (labels["rmvpe_ref_f0"] > 0)
    )
    if not both.any():
        return torch.empty(0)
    return cents_error(rmvpe["f0_hz"][both], labels["rmvpe_ref_f0"][both])


def aggregate_pitch_metrics(cents_parts: list[torch.Tensor]) -> dict[str, Any]:
    if not cents_parts:
        return {"comparison_frames": 0, "pitch_acc_50": None, "mace_cents": None, "pitch_rmse_cents": None}
    cents = torch.cat(cents_parts)
    return {
        "comparison_frames": int(cents.numel()),
        "pitch_acc_50": float((cents.abs() <= 50.0).float().mean().item()),
        "pitch_acc_100": float((cents.abs() <= 100.0).float().mean().item()),
        "mace_cents": float(cents.abs().mean().item()),
        "median_abs_cents": float(cents.abs().median().item()),
        "pitch_rmse_cents": float(torch.sqrt(cents.square().mean()).item()),
        "raw_chroma_acc_50": raw_chroma_acc_50(cents),
        "octave_error_rate": octave_error_rate(cents),
    }


def cents_error(pred_f0: torch.Tensor, ref_f0: torch.Tensor) -> torch.Tensor:
    return 1200.0 * torch.log2(torch.clamp(pred_f0, min=1e-6) / torch.clamp(ref_f0, min=1e-6))


def raw_chroma_acc_50(cents: torch.Tensor) -> float | None:
    if cents.numel() == 0:
        return None
    chroma = torch.remainder(cents.abs(), 1200.0)
    chroma = torch.minimum(chroma, 1200.0 - chroma)
    return float((chroma <= 50.0).float().mean().item())


def octave_error_rate(cents: torch.Tensor) -> float | None:
    if cents.numel() == 0:
        return None
    abs_cents = cents.abs()
    chroma = torch.remainder(abs_cents, 1200.0)
    chroma = torch.minimum(chroma, 1200.0 - chroma)
    return float(((abs_cents > 600.0) & (chroma <= 50.0)).float().mean().item())


# ---------------------------------------------------------------------------
# Optional CSV predictions (identical to test_1)
# ---------------------------------------------------------------------------

def write_prediction_csvs(batch: dict[str, Any], prediction_dir: Path) -> None:
    rmvpe = batch["rmvpe"]
    labels = batch["labels"]
    for index, metadata in enumerate(batch["metadata"]):
        output_path = prediction_dir / f"{metadata['recording_id']}_rmvpe.csv"
        length = int(rmvpe["frame_lengths"][index].item())
        rows = zip(
            rmvpe["times"][index, :length].detach().cpu().tolist(),
            rmvpe["f0_hz"][index, :length].detach().cpu().tolist(),
            rmvpe["voiced"][index, :length].detach().cpu().tolist(),
            rmvpe["confidence"][index, :length].detach().cpu().tolist(),
            labels["rmvpe_ref_f0"][index, :length].detach().cpu().tolist(),
            labels["rmvpe_ref_voicing"][index, :length].detach().cpu().tolist(),
            strict=True,
        )
        with output_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_sec", "rmvpe_f0_hz", "rmvpe_voiced", "confidence", "reference_f0_hz", "reference_voiced"])
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Recording discovery (identical to test_1)
# ---------------------------------------------------------------------------

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
