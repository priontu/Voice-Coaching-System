"""
models/pitch/pipeline.py - PitchVADPipeline orchestration and BaseInferenceModel wrapper.

Data flow:
    WAV file
      ↓ utils.audio.load_audio()
    float32 numpy @ 16 kHz
      ↓ WebRTCVAD.run()
    voiced_mask (VAD frame rate)
      ↓ align_vad_to_pitch()
    voiced_mask (pitch frame rate)
      ↓ PitchModelWrapper.predict()
    (times, f0_raw, confidence)
      ↓ fuse_vad_and_pitch()
    (f0_clean, voiced_final)
      ↓ save_pitch_json()
    pitch_data.json

Changes from the original Pitch Model w VAD/inference.py:
  - load_audio() replaced by utils.audio.load_audio (shared)
  - save_pitch_json() moved to scoring/pitch_score_utils.py (import preserved)
  - PitchInferenceModel added (implements BaseInferenceModel)
  - All pipeline logic is UNCHANGED
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from models.base import BaseInferenceModel
from models.pitch.alignment import align_vad_to_pitch, synchronize_arrays
from models.pitch.fusion import FusionConfig, fuse_vad_and_pitch
from models.pitch.pitch_wrapper import PitchConfig, PitchModelWrapper
from models.pitch.vad import VADConfig, WebRTCVAD
from utils.audio import TARGET_SAMPLE_RATE, load_audio
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON I/O (pitch_data.json schema — consumed by scoring/pitch_score.py)
# ---------------------------------------------------------------------------

def save_pitch_json(
    timestamps: np.ndarray,
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    output_path,
    audio_path: str = "",
    sample_rate: int = TARGET_SAMPLE_RATE,
    hop_length: int = 160,
) -> None:
    """Write pitch data to JSON in the format expected by pitch_score.py."""
    frames = []
    for t, freq, v in zip(timestamps, f0, voiced_mask):
        frames.append({
            "time": float(t),
            "f0": float(freq),
            "voiced": bool(v),
            "midi": float(69 + 12 * np.log2(freq / 440.0)) if freq > 0 else None,
        })

    output = {
        "audio_path": audio_path,
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "num_frames": len(frames),
        "frames": frames,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fp:
        json.dump(output, fp, indent=2)

    logger.info(f"[pipeline] Saved {len(frames)} frames → {output_path}")


def load_pitch_json(path) -> tuple:
    """Load pitch data written by save_pitch_json()."""
    with open(path, "r") as fp:
        data = json.load(fp)

    frames = data["frames"]
    timestamps = np.array([f["time"] for f in frames], dtype=np.float32)
    f0 = np.array([f["f0"] for f in frames], dtype=np.float32)
    voiced_mask = np.array(
        [f.get("voiced", f["f0"] > 0) for f in frames], dtype=bool
    )
    return timestamps, f0, voiced_mask


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Master configuration that drives the full inference pipeline."""

    sample_rate: int = TARGET_SAMPLE_RATE
    normalize_audio: bool = True
    use_vad: bool = True
    vad: VADConfig = field(default_factory=VADConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    output_path: str = "pitch_data.json"
    export_json: bool = True

    @classmethod
    def from_yaml(cls, cfg: Dict) -> "PipelineConfig":
        return cls(
            sample_rate=cfg.get("audio", {}).get("sample_rate", TARGET_SAMPLE_RATE),
            normalize_audio=cfg.get("audio", {}).get("normalize", True),
            use_vad=True,
            vad=VADConfig.from_yaml(cfg),
            pitch=PitchConfig.from_yaml(cfg),
            fusion=FusionConfig.from_yaml(cfg),
            output_path=cfg.get("output", {}).get("path", "pitch_data.json"),
            export_json=cfg.get("output", {}).get("export_json", True),
        )


# ---------------------------------------------------------------------------
# Pipeline output container
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    """All outputs produced by one run of the pipeline."""

    timestamps: np.ndarray
    f0: np.ndarray
    voiced_mask: np.ndarray
    vad_mask_raw: Optional[np.ndarray]
    vad_times_raw: Optional[np.ndarray]
    audio: np.ndarray
    sample_rate: int

    def to_list(self) -> List[dict]:
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
        return self.timestamps, self.f0, self.voiced_mask

    def voiced_duration(self) -> float:
        frame_step = float(self.timestamps[1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0
        return float(np.sum(self.voiced_mask)) * frame_step

    def voiced_ratio(self) -> float:
        return float(np.mean(self.voiced_mask)) if len(self.voiced_mask) > 0 else 0.0


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class PitchVADPipeline:
    """
    Modular VAD + pitch estimation pipeline.

    Example:
        pipeline = PitchVADPipeline(PipelineConfig())
        result = pipeline.run("my_singing.wav")
        times, f0, voiced = result.to_numpy()
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._vad = WebRTCVAD(self.config.vad) if self.config.use_vad else None
        self._pitch = PitchModelWrapper(self.config.pitch)

        logger.info("[Pipeline] Initialized.")
        logger.info(f"  VAD:   {'enabled' if self.config.use_vad else 'disabled'}")
        logger.info(f"  Pitch: backend={self.config.pitch.backend}, device={self.config.pitch.device}")

    def run(self, audio_path: str) -> PipelineOutput:
        """Full pipeline from audio file to cleaned pitch data."""
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
        """Full pipeline starting from a pre-loaded audio array."""
        vad_mask_raw, vad_times_raw = self._run_vad(audio, sr)
        times, f0_raw, confidence = self._pitch.predict(audio, sr)
        vad_mask_aligned = self._align_vad(vad_mask_raw, vad_times_raw, times)

        pitch_voiced = f0_raw > 0
        f0_clean, voiced_final = fuse_vad_and_pitch(
            f0_raw, pitch_voiced, vad_mask_aligned, config=self.config.fusion,
        )

        times, f0_clean, voiced_final = synchronize_arrays(times, f0_clean, voiced_final)

        if self.config.export_json:
            save_pitch_json(
                timestamps=times, f0=f0_clean, voiced_mask=voiced_final,
                output_path=self.config.output_path, audio_path=audio_path,
                sample_rate=sr, hop_length=self.config.pitch.hop_length,
            )

        result = PipelineOutput(
            timestamps=times, f0=f0_clean, voiced_mask=voiced_final,
            vad_mask_raw=vad_mask_raw, vad_times_raw=vad_times_raw,
            audio=audio, sample_rate=sr,
        )

        logger.info(
            f"[Pipeline] Done — {len(times)} frames, "
            f"voiced ratio: {result.voiced_ratio():.1%}, "
            f"voiced duration: {result.voiced_duration():.2f}s"
        )
        return result

    def _run_vad(self, audio: np.ndarray, sr: int):
        if self._vad is not None:
            return self._vad.run(audio, sr)

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
        if vad_mask is None or len(vad_times) == 0:
            return np.ones(len(pitch_times), dtype=bool)
        return align_vad_to_pitch(vad_mask, vad_times, pitch_times)


# ---------------------------------------------------------------------------
# BaseInferenceModel wrapper
# ---------------------------------------------------------------------------

class PitchInferenceModel(BaseInferenceModel):
    """
    BaseInferenceModel implementation for the VAD + pitch pipeline.

    Example:
        model = PitchInferenceModel()
        result = model.run("singing.wav")
        times, f0, voiced = result.to_numpy()
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        super().__init__()
        self._pipeline_config = config or PipelineConfig()
        self._pipeline: Optional[PitchVADPipeline] = None

    def load_model(self) -> None:
        self._pipeline = PitchVADPipeline(self._pipeline_config)
        self._is_loaded = True

    def predict(self, audio: np.ndarray) -> PipelineOutput:
        if not self._is_loaded:
            self.load_model()
        return self._pipeline.run_from_array(audio, self._pipeline_config.sample_rate)

    def run(self, audio_path) -> PipelineOutput:
        if not self._is_loaded:
            self.load_model()
        return self._pipeline.run(str(audio_path))
