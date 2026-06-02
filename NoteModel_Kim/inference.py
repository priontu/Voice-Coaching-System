"""
Inference pipeline for note onset/offset detection.

Usage::

    python inference.py \\
        --checkpoint checkpoints/best.pt \\
        --config configs/default.yaml \\
        --audio path/to/singing.wav \\
        --output results.json

Or use NoteDetector programmatically::

    from inference import NoteDetector

    detector = NoteDetector.from_checkpoint("checkpoints/best.pt", "configs/default.yaml")
    notes = detector.detect("singing.wav")
    # [{"onset_time": 1.23, "offset_time": 1.57, "duration": 0.34}, ...]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from model import OnsetOffsetModel
from utils import (
    compute_log_mel_spectrogram,
    frames_to_time,
    load_audio,
    normalize_spectrogram,
    pair_onsets_offsets,
    peak_pick_offsets,
    peak_pick_onsets,
)

logger = logging.getLogger(__name__)


class NoteDetector:
    """
    End-to-end inference pipeline: WAV → note boundaries.

    Attributes:
        model:               Loaded OnsetOffsetModel in eval mode.
        device:              Torch device used for inference.
        onset_threshold:     Minimum peak height for onset detection.
        offset_threshold:    Minimum peak height for offset detection.
        min_distance_frames: Minimum frames between consecutive peaks.
    """

    def __init__(
        self,
        model: OnsetOffsetModel,
        device: torch.device,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        onset_threshold: float = 0.3,
        offset_threshold: float = 0.3,
        min_distance_frames: int = 3,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_distance_frames = min_distance_frames

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str,
    ) -> NoteDetector:
        """
        Instantiate NoteDetector from a saved checkpoint and YAML config.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            config_path:     Path to YAML config used during training.

        Returns:
            Ready-to-use NoteDetector.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mc, ac, pp = cfg["model"], cfg["audio"], cfg["peak_picking"]

        model = OnsetOffsetModel(
            n_mels=ac["n_mels"],
            cnn_channels=mc["cnn_channels"],
            lstm_hidden_size=mc["lstm_hidden_size"],
            lstm_num_layers=mc["lstm_num_layers"],
            lstm_dropout=mc["lstm_dropout"],
            head_hidden_size=mc["head_hidden_size"],
            dropout=mc["dropout"],
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded checkpoint '%s'  (epoch %s)", checkpoint_path, ckpt.get("epoch", "?")
        )

        return cls(
            model=model,
            device=device,
            sample_rate=ac["sample_rate"],
            n_fft=ac["n_fft"],
            hop_length=ac["hop_length"],
            n_mels=ac["n_mels"],
            fmin=ac["fmin"],
            fmax=ac["fmax"],
            onset_threshold=pp["onset_threshold"],
            offset_threshold=pp["offset_threshold"],
            min_distance_frames=pp["min_distance_frames"],
        )

    # ── Core inference ────────────────────────────────────────────────────

    def predict_probs(
        self, audio_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run model inference on a WAV file.

        Args:
            audio_path: Path to WAV file.

        Returns:
            onset_probs:  [T] float32 in [0, 1]
            offset_probs: [T] float32 in [0, 1]
            frame_times:  [T] seconds per frame
        """
        waveform, _ = load_audio(audio_path, target_sr=self.sample_rate)
        log_mel = compute_log_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        log_mel = normalize_spectrogram(log_mel)

        x = log_mel.unsqueeze(0).to(self.device)          # [1, 1, n_mels, T]
        on_probs, off_probs = self.model.predict(x)

        on_probs = on_probs.squeeze(0).cpu().numpy()       # [T]
        off_probs = off_probs.squeeze(0).cpu().numpy()     # [T]
        frame_times = frames_to_time(
            log_mel.shape[-1], self.hop_length, self.sample_rate
        )

        return on_probs, off_probs, frame_times

    def detect(self, audio_path: str) -> List[Dict]:
        """
        Detect note boundaries in a WAV file.

        Args:
            audio_path: Path to WAV file.

        Returns:
            List of dicts, each with:
              onset_time  – seconds (float)
              offset_time – seconds (float or None)
              duration    – seconds (float or None)
        """
        on_probs, off_probs, frame_times = self.predict_probs(audio_path)

        onsets = peak_pick_onsets(
            on_probs, frame_times, self.onset_threshold, self.min_distance_frames
        )
        offsets = peak_pick_offsets(
            off_probs, frame_times, self.offset_threshold, self.min_distance_frames
        )

        notes = pair_onsets_offsets(onsets, offsets)
        logger.info("Detected %d notes in '%s'.", len(notes), audio_path)
        return notes


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    detector = NoteDetector.from_checkpoint(args.checkpoint, args.config)
    notes = detector.detect(args.audio)

    output = json.dumps(notes, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        logger.info("Results written to %s", args.output)
    else:
        print(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Note onset/offset detection inference"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--audio", required=True, help="Path to input WAV file")
    parser.add_argument(
        "--output", default=None, help="Output JSON path (prints to stdout if omitted)"
    )
    run_inference(parser.parse_args())


if __name__ == "__main__":
    main()
