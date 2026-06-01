#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from OnsetOffsetDetection_v1.data import discover_gtsinger_english_recordings, make_raw_loader
from OnsetOffsetDetection_v1.eval import EventDetectionConfig, event_metrics, extract_events, reference_events
from OnsetOffsetDetection_v1.plot import _plot_summary, plot_predictions
from OnsetOffsetDetection_v2.lightning_module import OnsetOffsetLightningModule

DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "OnsetOffsetDetection_v2" / "eval_outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OnsetOffsetDetection_v2.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--group-contains", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.5)
    parser.add_argument("--tolerance-sec", type=float, default=0.05)
    parser.add_argument("--min-distance-sec", type=float, default=0.05)
    parser.add_argument("--plot-limit", type=int, default=8,
                        help="Number of recordings to plot predictions for. 0 skips plotting.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recordings = discover_gtsinger_english_recordings(args.dataset_root, args.limit, args.group_contains)
    if not recordings:
        raise RuntimeError(f"No recordings found under {args.dataset_root}")

    loader = make_raw_loader(recordings, args.batch_size, args.num_workers, False, pin_memory=args.device.startswith("cuda"))
    module = OnsetOffsetLightningModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    module.to(args.device)
    module.eval()

    config = EventDetectionConfig(
        onset_threshold=args.onset_threshold,
        offset_threshold=args.offset_threshold,
        min_distance_sec=args.min_distance_sec,
        tolerance_sec=args.tolerance_sec,
    )
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        with torch.inference_mode():
            batch = module.process_raw_batch(raw_batch)
            outputs = module(batch["input_features"], batch["attention_mask"])
            onset_prob = torch.sigmoid(outputs["onset_logits"]).detach().cpu()
            offset_prob = torch.sigmoid(outputs["offset_logits"]).detach().cpu()
            for index, metadata in enumerate(batch["metadata"]):
                length = int(batch["frame_lengths"][index].item())
                times = torch.arange(length, dtype=torch.float32) * module.config_obj.features.frame_seconds
                pred_onsets = extract_events(onset_prob[index, :length], times, config.onset_threshold, config.min_distance_sec)
                pred_offsets = extract_events(offset_prob[index, :length], times, config.offset_threshold, config.min_distance_sec)
                ref_onsets, ref_offsets = reference_events(raw_batch["examples"][index].notes)
                rows.append({
                    "recording_id": metadata["recording_id"],
                    "wav_path": metadata["wav_path"],
                    "onset": event_metrics(pred_onsets, ref_onsets, config.tolerance_sec),
                    "offset": event_metrics(pred_offsets, ref_offsets, config.tolerance_sec),
                })

    summary = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "num_recordings": len(recordings),
        "settings": vars(args) | {"dataset_root": str(args.dataset_root), "checkpoint": str(args.checkpoint), "output_dir": str(args.output_dir)},
        "aggregate": aggregate(rows),
        "recordings": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")
    print(json.dumps(summary["aggregate"], indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots_dir = args.output_dir / "plots"
    if args.plot_limit > 0:
        print(f"\nPlotting predictions for {min(args.plot_limit, len(recordings))} recordings...")
        plot_predictions(module, recordings, plots_dir, args.plot_limit, config, args.device)
    _plot_summary(plt, summary, args.output_dir / "summary_metrics.png")
    print(f"summary metrics -> {args.output_dir / 'summary_metrics.png'}")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"num_recordings": len(rows), "onset": _section(rows, "onset"), "offset": _section(rows, "offset")}


def _section(rows: list[dict[str, Any]], section: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in ("precision", "recall", "f1", "mae_sec"):
        values = [row[section][key] for row in rows if row[section][key] is not None]
        output[key] = float(sum(values) / len(values)) if values else None
    output["predicted_count"] = sum(row[section]["predicted_count"] for row in rows)
    output["reference_count"] = sum(row[section]["reference_count"] for row in rows)
    output["matched_count"] = sum(row[section]["matched_count"] for row in rows)
    return output


if __name__ == "__main__":
    main()
