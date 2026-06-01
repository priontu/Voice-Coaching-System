#!/usr/bin/env python
"""Plot onset/offset predictions vs. ground truth for a trained checkpoint.

Per-recording figures show the log-mel spectrogram plus three panels:
onset probability, offset probability, and active-note probability, each
with predicted peaks (solid verticals) and ground-truth events (dashed red).

An optional summary figure is produced when --summary-json points to the
output of evaluate.py.

Usage
-----
python OnsetOffsetDetection_v1/plot.py \\
    --checkpoint OnsetOffsetDetection_v1/runs/default/checkpoints/last.ckpt \\
    --dataset-root /mnt/archive/GTSinger/English \\
    --limit 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from OnsetOffsetDetection_v1.data import discover_gtsinger_english_recordings, make_raw_loader
from OnsetOffsetDetection_v1.eval import EventDetectionConfig, extract_events, reference_events
from OnsetOffsetDetection_v1.lightning_module import OnsetOffsetLightningModule


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "OnsetOffsetDetection_v1" / "plots"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot onset/offset predictions vs. ground truth.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=8, help="Number of recordings to plot.")
    parser.add_argument("--group-contains", default=None)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.5)
    parser.add_argument("--tolerance-sec", type=float, default=0.05)
    parser.add_argument("--min-distance-sec", type=float, default=0.05)
    parser.add_argument(
        "--summary-json", type=Path, default=None,
        help="Path to summary.json from evaluate.py — adds aggregate metric plots.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    recordings = discover_gtsinger_english_recordings(
        args.dataset_root, limit=args.limit, group_contains=args.group_contains
    )
    if not recordings:
        raise RuntimeError(f"No recordings found under {args.dataset_root}")

    loader = make_raw_loader(recordings, batch_size=1, num_workers=2, shuffle=False, pin_memory=False)
    module = OnsetOffsetLightningModule.load_from_checkpoint(args.checkpoint, map_location=args.device)
    module.to(args.device)
    module.eval()

    config = EventDetectionConfig(
        onset_threshold=args.onset_threshold,
        offset_threshold=args.offset_threshold,
        min_distance_sec=args.min_distance_sec,
        tolerance_sec=args.tolerance_sec,
    )

    for batch_idx, raw_batch in enumerate(loader):
        with torch.inference_mode():
            batch = module.process_raw_batch(raw_batch)
            outputs = module(batch["input_features"], batch["attention_mask"])

        length = int(batch["frame_lengths"][0].item())
        times = torch.arange(length, dtype=torch.float32) * module.config_obj.features.frame_seconds

        onset_prob = torch.sigmoid(outputs["onset_logits"][0, :length]).detach().cpu()
        offset_prob = torch.sigmoid(outputs["offset_logits"][0, :length]).detach().cpu()
        active_prob = torch.sigmoid(outputs["active_logits"][0, :length]).detach().cpu()
        features = batch["input_features"][0, :length].detach().cpu().float().numpy()

        pred_onsets = extract_events(onset_prob, times, config.onset_threshold, config.min_distance_sec)
        pred_offsets = extract_events(offset_prob, times, config.offset_threshold, config.min_distance_sec)
        ref_onsets, ref_offsets = reference_events(raw_batch["examples"][0].notes)

        recording_id = batch["metadata"][0]["recording_id"]
        out_path = args.output_dir / f"{recording_id}.png"
        _plot_recording(
            plt=plt,
            recording_id=recording_id,
            times=times.numpy(),
            features=features,
            onset_prob=onset_prob.numpy(),
            offset_prob=offset_prob.numpy(),
            active_prob=active_prob.numpy(),
            pred_onsets=pred_onsets,
            pred_offsets=pred_offsets,
            ref_onsets=ref_onsets,
            ref_offsets=ref_offsets,
            notes=raw_batch["examples"][0].notes,
            output_path=out_path,
        )
        print(f"[{batch_idx + 1}/{len(recordings)}] {recording_id} -> {out_path.name}")

    if args.summary_json and args.summary_json.exists():
        summary = json.loads(args.summary_json.read_text())
        out = args.output_dir / "summary_metrics.png"
        _plot_summary(plt, summary, out)
        print(f"summary metrics -> {out.name}")


# ---------------------------------------------------------------------------
# Per-recording plot
# ---------------------------------------------------------------------------

def plot_predictions(
    module,
    recordings: list,
    output_dir: Path,
    plot_limit: int,
    config,
    device: str,
) -> None:
    """Run inference on up to plot_limit recordings and save prediction-vs-GT plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch as _torch

    from OnsetOffsetDetection_v1.data import make_raw_loader
    from OnsetOffsetDetection_v1.eval import extract_events, reference_events

    output_dir.mkdir(parents=True, exist_ok=True)
    subset = recordings[:plot_limit]
    loader = make_raw_loader(subset, batch_size=1, num_workers=2, shuffle=False, pin_memory=False)

    for i, raw_batch in enumerate(loader):
        with _torch.inference_mode():
            batch = module.process_raw_batch(raw_batch)
            outputs = module(batch["input_features"], batch["attention_mask"])

        length = int(batch["frame_lengths"][0].item())
        times = _torch.arange(length, dtype=_torch.float32) * module.config_obj.features.frame_seconds
        onset_prob  = _torch.sigmoid(outputs["onset_logits"][0,  :length]).detach().cpu()
        offset_prob = _torch.sigmoid(outputs["offset_logits"][0, :length]).detach().cpu()
        active_prob = _torch.sigmoid(outputs["active_logits"][0, :length]).detach().cpu()

        # features: spectrogram (T, F) for mel/CQT models, raw waveform (T_samples,) for Wav2Vec2
        inp = batch["input_features"][0]
        features = inp[:length].detach().cpu().float().numpy() if inp.ndim == 2 else None

        pred_onsets  = extract_events(onset_prob,  times, config.onset_threshold,  config.min_distance_sec)
        pred_offsets = extract_events(offset_prob, times, config.offset_threshold, config.min_distance_sec)
        ref_onsets, ref_offsets = reference_events(raw_batch["examples"][0].notes)

        recording_id = batch["metadata"][0]["recording_id"]
        out_path = output_dir / f"{recording_id}.png"
        _plot_recording(
            plt=plt,
            recording_id=recording_id,
            times=times.numpy(),
            features=features,
            onset_prob=onset_prob.numpy(),
            offset_prob=offset_prob.numpy(),
            active_prob=active_prob.numpy(),
            pred_onsets=pred_onsets,
            pred_offsets=pred_offsets,
            ref_onsets=ref_onsets,
            ref_offsets=ref_offsets,
            notes=raw_batch["examples"][0].notes,
            output_path=out_path,
        )
        print(f"  [{i + 1}/{len(subset)}] {recording_id} -> {out_path.name}")


def _plot_recording(
    *,
    plt,
    recording_id: str,
    times: np.ndarray,
    features: np.ndarray | None,
    onset_prob: np.ndarray,
    offset_prob: np.ndarray,
    active_prob: np.ndarray,
    pred_onsets: list[float],
    pred_offsets: list[float],
    ref_onsets: list[float],
    ref_offsets: list[float],
    notes: list[Any],
    output_path: Path,
) -> None:
    import matplotlib.patches as mpatches

    t_min, t_max = float(times[0]), float(times[-1])
    n_panels = 4 if features is not None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 2.5 * n_panels), sharex=True)
    fig.suptitle(recording_id, fontsize=10, fontweight="bold")

    panel = 0
    if features is not None:
        # --- Log-mel / CQT spectrogram ---
        ax = axes[panel]; panel += 1
        ax.imshow(
            features.T,
            aspect="auto",
            origin="lower",
            extent=[t_min, t_max, 0, features.shape[1]],
            cmap="magma",
            interpolation="nearest",
        )
        ax.set_ylabel("Feature bin")
        ax.set_title("Input features (CMVN normalised)")

    # --- Onset ---
    ax = axes[panel]; panel += 1
    ax.fill_between(times, onset_prob, alpha=0.20, color="#2ecc71")
    ax.plot(times, onset_prob, color="#27ae60", linewidth=0.8)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    for t in pred_onsets:
        ax.axvline(t, color="#27ae60", linewidth=0.9, alpha=0.85)
    for t in ref_onsets:
        ax.axvline(t, color="#e74c3c", linewidth=0.9, linestyle="--", alpha=0.85)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("P(onset)")
    ax.legend(
        handles=[
            mpatches.Patch(color="#27ae60", label=f"pred ({len(pred_onsets)})"),
            mpatches.Patch(color="#e74c3c", label=f"ref  ({len(ref_onsets)})"),
        ],
        loc="upper right", fontsize=7,
    )

    # --- Offset ---
    ax = axes[panel]; panel += 1
    ax.fill_between(times, offset_prob, alpha=0.20, color="#e67e22")
    ax.plot(times, offset_prob, color="#d35400", linewidth=0.8)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    for t in pred_offsets:
        ax.axvline(t, color="#d35400", linewidth=0.9, alpha=0.85)
    for t in ref_offsets:
        ax.axvline(t, color="#e74c3c", linewidth=0.9, linestyle="--", alpha=0.85)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("P(offset)")
    ax.legend(
        handles=[
            mpatches.Patch(color="#d35400", label=f"pred ({len(pred_offsets)})"),
            mpatches.Patch(color="#e74c3c", label=f"ref  ({len(ref_offsets)})"),
        ],
        loc="upper right", fontsize=7,
    )

    # --- Active note ---
    ax = axes[panel]
    for note in notes:
        ax.axvspan(note.start_sec, note.end_sec, alpha=0.15, color="#e74c3c", linewidth=0)
    ax.fill_between(times, active_prob, alpha=0.20, color="#3498db")
    ax.plot(times, active_prob, color="#2980b9", linewidth=0.8)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("P(active)")
    ax.set_xlabel("Time (s)")
    ax.legend(
        handles=[
            mpatches.Patch(color="#2980b9", label="pred active"),
            mpatches.Patch(color="#e74c3c", alpha=0.4, label=f"ref notes ({len(notes)})"),
        ],
        loc="upper right", fontsize=7,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate summary plot (from evaluate.py output)
# ---------------------------------------------------------------------------

def _plot_summary(plt, summary: dict[str, Any], output_path: Path) -> None:
    rows = summary.get("recordings", [])
    if not rows:
        print("  summary.json has no recordings — skipping summary plot")
        return

    colors = {"onset": "#27ae60", "offset": "#d35400"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Aggregate evaluation metrics", fontsize=12, fontweight="bold")

    # Top row: per-recording P / R / F1 grouped bars
    for col, section in enumerate(("onset", "offset")):
        ax = axes[0, col]
        x = np.arange(len(rows))
        bar_specs = [
            (-0.25, "precision", 1.00),
            ( 0.00, "recall",    0.70),
            ( 0.25, "f1",        0.45),
        ]
        for offset, metric, alpha in bar_specs:
            vals = [row[section].get(metric) or 0.0 for row in rows]
            ax.bar(x + offset, vals, width=0.22, label=metric,
                   color=colors[section], alpha=alpha, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_short_id(r["recording_id"]) for r in rows],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{section.capitalize()} — P / R / F1 per recording")
        ax.legend(fontsize=8)
        agg = summary.get("aggregate", {}).get(section, {})
        agg_line = "  ".join(
            f"{k}={agg[k]:.3f}" for k in ("precision", "recall", "f1")
            if agg.get(k) is not None
        )
        ax.set_xlabel(f"aggregate:  {agg_line}", fontsize=8)

    # Bottom row: MAE histograms
    for col, section in enumerate(("onset", "offset")):
        ax = axes[1, col]
        mae_vals = [r[section]["mae_sec"] for r in rows if r[section].get("mae_sec") is not None]
        if mae_vals:
            ax.hist(mae_vals, bins=min(20, len(mae_vals)),
                    color=colors[section], alpha=0.75, edgecolor="white")
            mean_mae = float(np.mean(mae_vals))
            ax.axvline(mean_mae, color="black", linestyle="--", linewidth=1.2,
                       label=f"mean = {mean_mae:.4f} s")
            ax.legend(fontsize=8)
        ax.set_xlabel("MAE (s)")
        ax.set_ylabel("Recording count")
        ax.set_title(f"{section.capitalize()} timing error distribution")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _short_id(recording_id: str, max_len: int = 18) -> str:
    parts = recording_id.split("_")
    short = "_".join(parts[-3:]) if len(parts) >= 3 else recording_id
    return short[:max_len]


if __name__ == "__main__":
    main()
