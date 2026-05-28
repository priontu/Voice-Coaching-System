"""
models/pitch/ - VAD + pitch estimation pipeline.

Primary entry point: PitchInferenceModel (implements BaseInferenceModel).
For direct pipeline use: PitchVADPipeline.

Module layout:
  vad.py           WebRTC voice activity detection
  pitch_wrapper.py torchcrepe / pYIN pitch estimation backends
  alignment.py     VAD-to-pitch frame alignment
  fusion.py        VAD + pitch fusion and contour cleaning
  pipeline.py      PitchVADPipeline (orchestration) + PitchInferenceModel
"""

from models.pitch.pipeline import PitchVADPipeline, PitchInferenceModel, PipelineConfig

__all__ = ["PitchVADPipeline", "PitchInferenceModel", "PipelineConfig"]
