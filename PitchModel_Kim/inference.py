"""
inference.py - Main VAD + Pitch pipeline orchestration and CLI entrypoint.

Data flow:
    WAV file
      ↓ utils.load_audio()
    float32 audio @ 16 kHz
      ↓ vad.WebRTCVAD.run()
    voiced_mask (VAD frame rate)
      ↓ alignment.align_vad_to_pitch()
    voiced_mask (pitch frame rate)
      ↓ pitch_wrapper.PitchModelWrapper.predict()
    (times, f0_raw, confidence)
      ↓ fusion.fuse_vad_and_pitch()
    (f0_clean, voiced_final)
      ↓ utils.save_pitch_json()
    pitch_data.json  ← consumed by pitch_score.py unchanged

CLI:
    python inference.py --audio sample.wav [--output pitch_data.json]
                        [--vad-mode 2] [--device auto] [--backend torchcrepe]
                        [--visualize] [--no-vad]
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils import load_audio, save_pitch_json, TARGET_SAMPLE_RATE
from vad import WebRTCVAD, VADConfig
from pitch_wrapper import PitchModelWrapper, PitchConfig
from alignment import align_vad_to_pitch, synchronize_arrays
from fusion import fuse_vad_and_pitch, FusionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration (aggregates all sub-configs)
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Master configuration that drives the full inference pipeline."""

    # --- Audio ---
    sample_rate: int = TARGET_SAMPLE_RATE
    normalize_audio: bool = True

    # --- VAD ---
    use_vad: bool = True
    """Set to False to skip VAD and use only the pitch model's voiced flags."""
    vad: VADConfig = field(default_factory=VADConfig)

    # --- Pitch ---
    pitch: PitchConfig = field(default_factory=PitchConfig)

    # --- Fusion ---
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # --- Output ---
    output_path: str = "pitch_data.json"
    export_json: bool = True
    """Also write a structured JSON array of {time, f0, voiced} dicts."""


# ---------------------------------------------------------------------------
# Pipeline output container
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    """All outputs produced by one run of the pipeline."""

    timestamps: np.ndarray
    """Frame center timestamps in seconds, shape (T,)."""
    f0: np.ndarray
    """Cleaned F0 in Hz (0.0 = unvoiced), shape (T,)."""
    voiced_mask: np.ndarray
    """Final voiced boolean mask, shape (T,)."""
    vad_mask_raw: Optional[np.ndarray]
    """Raw VAD mask at VAD frame rate, shape (N,). None if VAD skipped."""
    vad_times_raw: Optional[np.ndarray]
    """VAD frame timestamps at VAD frame rate, shape (N,). None if VAD skipped."""
    audio: np.ndarray
    """The loaded mono float32 audio array."""
    sample_rate: int
    """Audio sample rate (always 16000 after load_audio)."""

    def to_list(self) -> List[dict]:
        """Return frame-by-frame data as a list of dicts (JSON-serializable)."""
        out = []
        for t, freq, v in zip(self.timestamps, self.f0, self.voiced_mask):
            out.append({
                "time": float(t),
                "f0": float(freq),
                "voiced": bool(v),
                "midi": float(69 + 12 * np.log2(freq / 440.0)) if freq > 0 else None,
            })
        return out

    def to_numpy(self):
        """Return (timestamps, f0, voiced_mask) as numpy arrays."""
        return self.timestamps, self.f0, self.voiced_mask

    def voiced_duration(self) -> float:
        """Total voiced duration in seconds."""
        frame_step = float(self.timestamps[1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0
        return float(np.sum(self.voiced_mask)) * frame_step

    def voiced_ratio(self) -> float:
        """Fraction of frames classified as voiced (0.0 – 1.0)."""
        return float(np.mean(self.voiced_mask)) if len(self.voiced_mask) > 0 else 0.0


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class PitchVADPipeline:
    """
    Modular VAD + pitch estimation pipeline.

    Can be used programmatically or driven from the CLI.

    Example:
        pipeline = PitchVADPipeline(PipelineConfig())
        result = pipeline.run("my_singing.wav")
        times, f0, voiced = result.to_numpy()
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

        # Instantiate sub-modules (lazy: pitch model may load on first call)
        self._vad = WebRTCVAD(self.config.vad) if self.config.use_vad else None
        self._pitch = PitchModelWrapper(self.config.pitch)

        logger.info("[Pipeline] Initialized.")
        logger.info(f"  VAD:   {'enabled' if self.config.use_vad else 'disabled'}")
        logger.info(f"  Pitch: backend={self.config.pitch.backend}, device={self.config.pitch.device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, audio_path: str) -> PipelineOutput:
        """
        Full pipeline from audio file to cleaned pitch data.

        Args:
            audio_path: Path to a WAV (or any soundfile-readable) audio file.

        Returns:
            PipelineOutput containing timestamps, f0, voiced mask, and metadata.
        """
        path = Path(audio_path)
        logger.info(f"[Pipeline] Processing: {path.name}")

        audio, sr = load_audio(
            path,
            target_sr=self.config.sample_rate,
            normalize=self.config.normalize_audio,
        )

        return self.run_from_array(audio, sr, audio_path=str(path))

    def run_from_array(
        self,
        audio: np.ndarray,
        sr: int,
        audio_path: str = "",
    ) -> PipelineOutput:
        """
        Full pipeline starting from a pre-loaded audio array.

        Args:
            audio: Float32 mono audio, shape (samples,).
            sr: Sample rate (must be 16000).
            audio_path: Optional original file path (used in output metadata).

        Returns:
            PipelineOutput.
        """
        # Step 1 — Voice Activity Detection
        vad_mask_raw, vad_times_raw = self._run_vad(audio, sr)

        # Step 2 — Pitch estimation
        times, f0_raw, confidence = self._pitch.predict(audio, sr)

        # Step 3 — Align VAD mask to pitch frame rate
        vad_mask_aligned = self._align_vad(vad_mask_raw, vad_times_raw, times)

        # Step 4 — Fusion: intersect, clean, smooth
        pitch_voiced = f0_raw > 0
        f0_clean, voiced_final = fuse_vad_and_pitch(
            f0_raw,
            pitch_voiced,
            vad_mask_aligned,
            config=self.config.fusion,
        )

        # Step 5 — Synchronize array lengths (guard against off-by-one)
        times, f0_clean, voiced_final = synchronize_arrays(
            times, f0_clean, voiced_final
        )

        # Step 6 — Save output
        if self.config.export_json:
            save_pitch_json(
                timestamps=times,
                f0=f0_clean,
                voiced_mask=voiced_final,
                output_path=self.config.output_path,
                audio_path=audio_path,
                sample_rate=sr,
                hop_length=self.config.pitch.hop_length,
            )

        result = PipelineOutput(
            timestamps=times,
            f0=f0_clean,
            voiced_mask=voiced_final,
            vad_mask_raw=vad_mask_raw,
            vad_times_raw=vad_times_raw,
            audio=audio,
            sample_rate=sr,
        )

        logger.info(
            f"[Pipeline] Done — {len(times)} frames, "
            f"voiced ratio: {result.voiced_ratio():.1%}, "
            f"voiced duration: {result.voiced_duration():.2f}s"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_vad(
        self,
        audio: np.ndarray,
        sr: int,
    ):
        """Run VAD if enabled, otherwise return an all-voiced mask."""
        if self._vad is not None:
            return self._vad.run(audio, sr)

        # VAD disabled — every frame is "voiced" (let pitch model decide)
        n_frames = len(audio) // int(sr * self.config.vad.frame_duration_ms / 1000)
        frame_sec = self.config.vad.frame_duration_ms / 1000.0
        vad_times = (np.arange(n_frames) + 0.5) * frame_sec
        vad_mask = np.ones(n_frames, dtype=bool)
        logger.info("[Pipeline] VAD disabled — all frames treated as voiced.")
        return vad_mask, vad_times

    def _align_vad(
        self,
        vad_mask: np.ndarray,
        vad_times: np.ndarray,
        pitch_times: np.ndarray,
    ) -> np.ndarray:
        """Align the VAD mask from VAD frame rate to pitch frame rate."""
        if vad_mask is None or len(vad_times) == 0:
            return np.ones(len(pitch_times), dtype=bool)

        return align_vad_to_pitch(vad_mask, vad_times, pitch_times)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VAD + Pitch extraction pipeline for singing analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="Path to the input WAV audio file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="pitch_data.json",
        help="Path for the output pitch JSON (consumed by pitch_score.py).",
    )
    parser.add_argument(
        "--backend",
        choices=["torchcrepe", "pyin"],
        default="torchcrepe",
        help="Pitch estimation backend.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "PyTorch device. 'auto' (default) uses CUDA if available, "
            "then MPS, then CPU. Override with 'cpu', 'cuda', 'cuda:0', or 'mps'."
        ),
    )
    parser.add_argument(
        "--vad-mode",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help="WebRTC VAD aggressiveness (0=least, 3=most).",
    )
    parser.add_argument(
        "--vad-frame-ms",
        type=int,
        choices=[10, 20, 30],
        default=20,
        help="VAD frame duration in ms.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Skip VAD and rely solely on the pitch model's voiced detection.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=160,
        help="Pitch model hop length in samples (160 @ 16kHz = 10ms).",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=50.0,
        help="Minimum detectable F0 in Hz.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=1000.0,
        help="Maximum detectable F0 in Hz.",
    )
    parser.add_argument(
        "--periodicity-threshold",
        type=float,
        default=0.21,
        help="torchcrepe periodicity threshold (frames below this → unvoiced).",
    )
    parser.add_argument(
        "--gap-fill",
        type=int,
        default=10,
        help="Max frames of silence to bridge by interpolation (0 = disabled).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save visualization plots alongside the JSON.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Assemble configuration from CLI args
    config = PipelineConfig(
        use_vad=not args.no_vad,
        vad=VADConfig(
            aggressiveness=args.vad_mode,
            frame_duration_ms=args.vad_frame_ms,
            sample_rate=TARGET_SAMPLE_RATE,
        ),
        pitch=PitchConfig(
            backend=args.backend,
            hop_length=args.hop_length,
            fmin=args.fmin,
            fmax=args.fmax,
            periodicity_threshold=args.periodicity_threshold,
            device=args.device,
        ),
        fusion=FusionConfig(
            max_gap_fill_frames=args.gap_fill,
        ),
        output_path=args.output,
    )

    pipeline = PitchVADPipeline(config)

    try:
        result = pipeline.run(args.audio)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print(f"\n{'─'*50}")
    print(f"  Frames:        {len(result.timestamps)}")
    print(f"  Voiced frames: {int(np.sum(result.voiced_mask))}")
    print(f"  Voiced ratio:  {result.voiced_ratio():.1%}")
    print(f"  Voiced dur:    {result.voiced_duration():.2f}s")
    print(f"  Output JSON:   {args.output}")
    print(f"{'─'*50}\n")

    if np.sum(result.voiced_mask) > 0:
        voiced_f0 = result.f0[result.voiced_mask & (result.f0 > 0)]
        print(f"  F0 min:    {np.min(voiced_f0):.1f} Hz")
        print(f"  F0 max:    {np.max(voiced_f0):.1f} Hz")
        print(f"  F0 median: {np.median(voiced_f0):.1f} Hz")
        print(f"  F0 mean:   {np.mean(voiced_f0):.1f} Hz")
        print()

    if args.visualize:
        try:
            from visualization import plot_pitch_vad_combined
            base = Path(args.output).stem
            plot_pitch_vad_combined(
                audio=result.audio,
                sr=result.sample_rate,
                pitch_times=result.timestamps,
                f0=result.f0,
                voiced_mask=result.voiced_mask,
                vad_times=result.vad_times_raw,
                vad_mask=result.vad_mask_raw,
                save_path=f"{base}_vad_pitch.png",
            )
        except Exception as exc:
            print(f"[Warning] Visualization failed: {exc}")


if __name__ == "__main__":
    main()
