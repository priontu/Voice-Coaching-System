"""
Visualization script for the note onset/offset detection model.

Generates one or more plots from a WAV file and a trained checkpoint.
All plots are saved as PNG files.

Usage
-----
# Four-panel overview (waveform + spectrogram + onset curve + offset curve):
    python visualize.py \\
        --checkpoint checkpoints/best.pt \\
        --config     configs/default.yaml \\
        --audio      path/to/0000.wav \\
        --plot       overview

# Waveform with boundary markers:
    python visualize.py ... --plot waveform

# Spectrogram with boundary markers:
    python visualize.py ... --plot spectrogram

# Onset / offset probability curves:
    python visualize.py ... --plot probs

# All four plots at once:
    python visualize.py ... --plot all

# Overlay ground truth boundaries (requires GTSinger JSON):
    python visualize.py ... --plot all --label path/to/0000.json

# Save to a specific folder:
    python visualize.py ... --plot all --output_dir plots/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt

from inference import NoteDetector
from utils import (
    compute_log_mel_spectrogram,
    frames_to_time,
    load_audio,
    normalize_spectrogram,
    pair_onsets_offsets,
    parse_gtsinger_json,
    peak_pick_offsets,
    peak_pick_onsets,
)
from visualization import (
    plot_full_overview,
    plot_probability_curves,
    plot_spectrogram_with_boundaries,
    plot_waveform_with_boundaries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_PLOTS = ("overview", "waveform", "spectrogram", "probs", "all")


def run(args: argparse.Namespace) -> None:
    audio_path = args.audio
    stem = Path(audio_path).stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load audio & compute features ─────────────────────────────────────
    logger.info("Loading audio: %s", audio_path)
    waveform, sr = load_audio(audio_path)
    waveform_np = waveform.squeeze(0).numpy()

    log_mel_tensor = normalize_spectrogram(
        compute_log_mel_spectrogram(waveform, sample_rate=sr)
    )
    log_mel = log_mel_tensor.squeeze(0).numpy()   # [n_mels, T]

    # ── Run model ─────────────────────────────────────────────────────────
    detector = NoteDetector.from_checkpoint(args.checkpoint, args.config)
    on_probs, off_probs, frame_times = detector.predict_probs(audio_path)

    onsets  = peak_pick_onsets(on_probs,  frame_times, detector.onset_threshold,  detector.min_distance_frames)
    offsets = peak_pick_offsets(off_probs, frame_times, detector.offset_threshold, detector.min_distance_frames)
    notes   = pair_onsets_offsets(onsets, offsets)
    logger.info("Detected %d notes.", len(notes))

    # ── Load ground truth (optional) ──────────────────────────────────────
    ref_notes: Optional[list] = None
    ref_on:    Optional[List[float]] = None
    ref_off:   Optional[List[float]] = None

    if args.label:
        gt = parse_gtsinger_json(args.label)
        ref_notes = [{"onset_time": n.onset, "offset_time": n.offset} for n in gt]
        ref_on    = [n.onset  for n in gt]
        ref_off   = [n.offset for n in gt]
        logger.info("Ground truth loaded: %d notes.", len(gt))

    plots = {"overview", "waveform", "spectrogram", "probs"} if args.plot == "all" else {args.plot}

    # ── Overview ──────────────────────────────────────────────────────────
    if "overview" in plots:
        path = out_dir / f"{stem}_overview.png"
        plot_full_overview(
            waveform=waveform_np,
            log_mel=log_mel,
            onset_probs=on_probs,
            offset_probs=off_probs,
            frame_times=frame_times,
            sample_rate=sr,
            hop_length=detector.hop_length,
            predicted_notes=notes,
            reference_notes=ref_notes,
            onset_threshold=detector.onset_threshold,
            offset_threshold=detector.offset_threshold,
            save_path=str(path),
        )
        logger.info("Saved → %s", path)
        plt.close("all")

    # ── Waveform ──────────────────────────────────────────────────────────
    if "waveform" in plots:
        path = out_dir / f"{stem}_waveform.png"
        plot_waveform_with_boundaries(
            waveform=waveform_np,
            sample_rate=sr,
            predicted_notes=notes,
            reference_notes=ref_notes,
            save_path=str(path),
        )
        logger.info("Saved → %s", path)
        plt.close("all")

    # ── Spectrogram ───────────────────────────────────────────────────────
    if "spectrogram" in plots:
        path = out_dir / f"{stem}_spectrogram.png"
        plot_spectrogram_with_boundaries(
            log_mel=log_mel,
            hop_length=detector.hop_length,
            sample_rate=sr,
            predicted_notes=notes,
            reference_notes=ref_notes,
            save_path=str(path),
        )
        logger.info("Saved → %s", path)
        plt.close("all")

    # ── Probability curves ────────────────────────────────────────────────
    if "probs" in plots:
        path = out_dir / f"{stem}_probs.png"
        plot_probability_curves(
            onset_probs=on_probs,
            offset_probs=off_probs,
            frame_times=frame_times,
            onset_threshold=detector.onset_threshold,
            offset_threshold=detector.offset_threshold,
            reference_onsets=ref_on,
            reference_offsets=ref_off,
            save_path=str(path),
        )
        logger.info("Saved → %s", path)
        plt.close("all")

    logger.info("Done. All plots saved to: %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize note onset/offset detection results"
    )
    parser.add_argument("--checkpoint",  required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config",      required=True, help="Path to YAML config")
    parser.add_argument("--audio",       required=True, help="Path to input WAV file")
    parser.add_argument("--label",       default=None,
                        help="GTSinger JSON label to overlay ground truth (optional)")
    parser.add_argument("--plot",        default="overview", choices=VALID_PLOTS,
                        help="Which plot(s) to generate (default: overview)")
    parser.add_argument("--output_dir",  default="plots",
                        help="Directory to save PNG files (default: plots/)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
