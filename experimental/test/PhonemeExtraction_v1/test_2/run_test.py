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
from PhonemeExtraction_v1.eval import compare_phoneme_boundaries, is_scored_phone, normalize_phone
from hf_models.data.manifest import Recording, save_manifest
from hf_models.data.textgrid import phoneme_intervals


DEFAULT_DATASET_ROOT = Path("/mnt/archive/GTSinger/English/EN-Alto-1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONTROL_GROUP = "Control_Group"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate HF phoneme extraction on GTSinger Control_Group with timeline plots."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--selection", choices=["round_robin", "first"], default="round_robin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-plot-sec", type=float, default=15.0,
                        help="Only plot this many seconds per recording (avoids illegible dense plots).")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    pred_dir = args.output_dir / "predictions"
    match_dir = args.output_dir / "matches"
    plot_dir = args.output_dir / "plots"
    for directory in (pred_dir, match_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

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
        ref_scored = [item for item in reference if is_scored_phone(item.text)]
        metrics = compare_phoneme_boundaries(prediction["phonemes"], reference)

        pred_path = pred_dir / f"{recording.recording_id}.json"
        pred_path.write_text(json.dumps(prediction, indent=2))
        match_path = match_dir / f"{recording.recording_id}_matches.csv"
        write_matches_csv(match_path, metrics["matches"])

        plot_path = plot_dir / f"{recording.recording_id}_phonemes.png"
        _save_phoneme_plot(
            recording_id=recording.recording_id,
            prediction=prediction,
            ref_scored=ref_scored,
            matches=metrics["matches"],
            plot_path=plot_path,
            max_plot_sec=args.max_plot_sec,
            dpi=args.dpi,
            plt=plt,
            Rectangle=Rectangle,
            Patch=Patch,
        )

        rows.append(
            {
                "recording_id": recording.recording_id,
                "wav_path": str(recording.wav_path),
                "textgrid_path": str(recording.textgrid_path),
                "prediction_path": str(pred_path),
                "matches_path": str(match_path),
                "plot_path": str(plot_path),
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
            "Plots show up to --max-plot-sec seconds.",
            "Green=label match, orange=close boundary/label mismatch, red=large error, gray=unmatched.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"manifest: {manifest_path}")
    print(f"summary:  {summary_path}")
    print(f"plots:    {plot_dir}")
    print(json.dumps(summary["aggregate"], indent=2))


def _save_phoneme_plot(
    *,
    recording_id: str,
    prediction: dict[str, Any],
    ref_scored: list[Any],
    matches: list[dict[str, Any]],
    plot_path: Path,
    max_plot_sec: float,
    dpi: int,
    plt: Any,
    Rectangle: Any,
    Patch: Any,
) -> None:
    phonemes = prediction["phonemes"]

    # Colour for a matched pair (same colour applied to both pred and ref boxes).
    # matches[i] pairs predicted[i] with ref_scored[i].
    def _color(match: dict[str, Any] | None) -> str:
        if match is None:
            return "#9e9e9e"  # gray — no counterpart
        if normalize_phone(match["predicted_phoneme"]) == normalize_phone(match["reference_phoneme"]):
            return "#4caf50"  # green — label matches
        if match["boundary_error_sec"] < 0.15:
            return "#ff9800"  # orange — boundary close, label wrong
        return "#f44336"      # red — boundary error large or both wrong

    fig, (ax_seg, ax_conf) = plt.subplots(
        2, 1,
        figsize=(14, 4),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle(recording_id, fontsize=9)

    PRED_BOT, PRED_H = 1.05, 0.85
    REF_BOT, REF_H = 0.10, 0.85

    # --- Predicted lane (top) ---
    for pred_idx, phoneme in enumerate(phonemes):
        start = float(phoneme["start_sec"])
        if start >= max_plot_sec:
            break
        end = min(float(phoneme["end_sec"]), max_plot_sec)
        width = end - start
        match = matches[pred_idx] if pred_idx < len(matches) else None
        color = _color(match)
        ax_seg.add_patch(Rectangle((start, PRED_BOT), width, PRED_H, color=color, alpha=0.85, linewidth=0))
        ax_seg.axvline(x=start, color="white", linewidth=0.5, alpha=0.7)
        if width > 0.09:
            ax_seg.text(
                start + width / 2, PRED_BOT + PRED_H / 2,
                phoneme["phoneme"],
                ha="center", va="center", fontsize=7, color="white", clip_on=True,
            )

    # --- Reference lane (bottom) ---
    for ref_idx, ref_item in enumerate(ref_scored):
        start = float(ref_item.start_sec)
        if start >= max_plot_sec:
            break
        end = min(float(ref_item.end_sec), max_plot_sec)
        width = end - start
        match = matches[ref_idx] if ref_idx < len(matches) else None
        color = _color(match)
        ax_seg.add_patch(Rectangle((start, REF_BOT), width, REF_H, color=color, alpha=0.5, linewidth=0))
        ax_seg.axvline(x=start, color="white", linewidth=0.5, alpha=0.7)
        if width > 0.09:
            ax_seg.text(
                start + width / 2, REF_BOT + REF_H / 2,
                ref_item.text,
                ha="center", va="center", fontsize=7, color="black", clip_on=True,
            )

    # Lane labels on the left y-axis
    ax_seg.text(
        -0.01, PRED_BOT + PRED_H / 2, "Pred",
        ha="right", va="center", fontsize=8, transform=ax_seg.get_yaxis_transform(),
    )
    ax_seg.text(
        -0.01, REF_BOT + REF_H / 2, "Ref",
        ha="right", va="center", fontsize=8, transform=ax_seg.get_yaxis_transform(),
    )

    legend_elements = [
        Patch(facecolor="#4caf50", alpha=0.85, label="label match"),
        Patch(facecolor="#ff9800", alpha=0.85, label="label mismatch / boundary OK"),
        Patch(facecolor="#f44336", alpha=0.85, label="large boundary error"),
        Patch(facecolor="#9e9e9e", alpha=0.85, label="unmatched"),
    ]
    ax_seg.legend(handles=legend_elements, fontsize=7, loc="upper right")
    ax_seg.set_xlim(0, max_plot_sec)
    ax_seg.set_ylim(0.0, 2.0)
    ax_seg.set_yticks([])
    ax_seg.set_ylabel("Segments", fontsize=8)

    # --- Confidence bars ---
    for phoneme in phonemes:
        start = float(phoneme["start_sec"])
        if start >= max_plot_sec:
            break
        end = min(float(phoneme["end_sec"]), max_plot_sec)
        width = end - start
        ax_conf.bar(
            start + width / 2,
            float(phoneme["confidence"]),
            width=max(width * 0.92, 1e-3),
            color="#43a047",
            alpha=0.75,
            align="center",
        )

    ax_conf.axhline(y=0.35, color="red", linewidth=0.8, linestyle="--", label="threshold 0.35")
    ax_conf.set_xlim(0, max_plot_sec)
    ax_conf.set_ylim(0.0, 1.05)
    ax_conf.set_ylabel("Conf.", fontsize=8)
    ax_conf.set_xlabel("Time (s)", fontsize=8)
    ax_conf.tick_params(labelsize=7)
    ax_conf.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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
    return float(sum(values) / len(values)) if values else None


def discover_control_group_triplets(
    dataset_root: Path,
    limit: int | None,
    selection: str,
) -> list[Recording]:
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
