#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from OnsetOffsetDetection_v1.data import discover_gtsinger_english_recordings, make_raw_loader
from OnsetOffsetDetection_v1.eval import EventDetectionConfig, event_metrics, extract_events, reference_events
from OnsetOffsetDetection_v1.lightning_module import OnsetOffsetLightningModule


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke/evaluate OnsetOffsetDetection_v1 on GTSinger English.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.5)
    parser.add_argument("--tolerance-sec", type=float, default=0.05)
    parser.add_argument("--min-distance-sec", type=float, default=0.05)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recordings = discover_gtsinger_english_recordings(args.dataset_root, limit=args.limit)
    if not recordings:
        raise RuntimeError(f"No recordings found under {args.dataset_root}")

    loader = make_raw_loader(
        recordings,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.device.startswith("cuda"),
    )
    if args.checkpoint is not None:
        module = OnsetOffsetLightningModule.load_from_checkpoint(args.checkpoint, map_location=args.device)
        mode = "checkpoint"
    else:
        module = OnsetOffsetLightningModule()
        mode = "untrained_smoke"
    module.to(args.device)
    module.eval()

    config = EventDetectionConfig(
        onset_threshold=args.onset_threshold,
        offset_threshold=args.offset_threshold,
        tolerance_sec=args.tolerance_sec,
        min_distance_sec=args.min_distance_sec,
    )
    rows = []
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
                rows.append(
                    {
                        "recording_id": metadata["recording_id"],
                        "wav_path": metadata["wav_path"],
                        "onset": event_metrics(pred_onsets, ref_onsets, config.tolerance_sec),
                        "offset": event_metrics(pred_offsets, ref_offsets, config.tolerance_sec),
                    }
                )

    summary = {
        "mode": mode,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "dataset_root": str(args.dataset_root),
        "num_recordings": len(recordings),
        "aggregate": aggregate(rows),
        "recordings": rows,
        "notes": [
            "Without --checkpoint this only verifies data/model execution; metrics are from an untrained model.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")
    print(json.dumps(summary["aggregate"], indent=2))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_recordings": len(rows),
        "onset": aggregate_section(rows, "onset"),
        "offset": aggregate_section(rows, "offset"),
    }


def aggregate_section(rows: list[dict[str, Any]], section: str) -> dict[str, Any]:
    output = {}
    for key in ("precision", "recall", "f1", "mae_sec"):
        values = [row[section][key] for row in rows if row[section][key] is not None]
        output[key] = float(sum(values) / len(values)) if values else None
    output["predicted_count"] = sum(row[section]["predicted_count"] for row in rows)
    output["reference_count"] = sum(row[section]["reference_count"] for row in rows)
    output["matched_count"] = sum(row[section]["matched_count"] for row in rows)
    return output


if __name__ == "__main__":
    main()
