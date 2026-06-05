"""
tests/test_pipeline.py - Unit tests for inference/pipeline.py

All model inference is patched out so tests run without GPU or model weights.
Tests verify orchestration logic, alignment correctness, and result contracts.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List, Optional

import numpy as np
import pytest

from preprocessing.timestamps import HOP_LENGTH, SAMPLE_RATE, canonical_timestamps
from utils.types import FrameAlignedFeatures, PhonemeSegment, UnifiedInferenceResult


# ---------------------------------------------------------------------------
# Fixtures — synthetic audio and model outputs
# ---------------------------------------------------------------------------

DURATION_S = 1.0
N_SAMPLES = int(DURATION_S * SAMPLE_RATE)
N_PITCH = int(DURATION_S * SAMPLE_RATE / HOP_LENGTH)        # 100 frames
N_ONSET = int(DURATION_S * SAMPLE_RATE / 256)               # ~62 frames


def _audio_np():
    """Synthetic 1-second silence audio."""
    return np.zeros(N_SAMPLES, dtype=np.float32)


def _pitch_output():
    ts = canonical_timestamps(N_PITCH).astype(np.float64)
    f0 = np.random.uniform(200, 400, N_PITCH).astype(np.float32)
    voiced = np.ones(N_PITCH, dtype=bool)
    voiced[80:] = False  # last 20 frames unvoiced
    return ts, f0, voiced


def _phoneme_segments():
    return [
        PhonemeSegment("AH", 0.0, 0.3, 1.0),
        PhonemeSegment("EH", 0.3, 0.7, 1.0),
        PhonemeSegment("IH", 0.7, 1.0, 1.0),
    ]


def _onset_output():
    ts = (np.arange(N_ONSET) * 256 / SAMPLE_RATE + 256 / (2 * SAMPLE_RATE)).astype(np.float64)
    on_probs = np.random.rand(N_ONSET).astype(np.float32)
    off_probs = np.random.rand(N_ONSET).astype(np.float32)
    return ts, on_probs, off_probs


# ---------------------------------------------------------------------------
# Pipeline construction (no model loading)
# ---------------------------------------------------------------------------

class TestPipelineConstruction:
    def test_default_construction(self):
        from inference.pipeline import UnifiedInferencePipeline
        p = UnifiedInferencePipeline(config={})
        assert p is not None

    def test_from_dict(self):
        from inference.pipeline import UnifiedInferencePipeline
        p = UnifiedInferencePipeline.from_dict({
            "pipeline": {"enable_pitch": False, "enable_phoneme": False}
        })
        assert not p._enable_pitch
        assert not p._enable_phoneme

    def test_disable_all_modules(self):
        from inference.pipeline import UnifiedInferencePipeline
        p = UnifiedInferencePipeline({"pipeline": {
            "enable_pitch": False,
            "enable_phoneme": False,
            "enable_onset_offset": False,
        }})
        assert not p._enable_pitch
        assert not p._enable_phoneme
        assert not p._enable_onset

    def test_device_manager_created(self):
        from inference.pipeline import UnifiedInferencePipeline
        from inference.device_manager import DeviceManager
        p = UnifiedInferencePipeline({})
        assert isinstance(p._device_manager, DeviceManager)


# ---------------------------------------------------------------------------
# UnifiedInferenceResult contract
# ---------------------------------------------------------------------------

class TestUnifiedInferenceResult:
    def _make_result(self, **kwargs):
        return UnifiedInferenceResult(audio_path="test.wav", **kwargs)

    def test_has_pitch_false_by_default(self):
        r = self._make_result()
        assert not r.has_pitch()

    def test_has_pitch_true_when_f0_set(self):
        r = self._make_result(f0=np.zeros(10))
        assert r.has_pitch()

    def test_has_phonemes_false_by_default(self):
        r = self._make_result()
        assert not r.has_phonemes()

    def test_is_complete_false_partial(self):
        ts, f0, voiced = _pitch_output()
        r = self._make_result(f0=f0, voiced=voiced)
        assert not r.is_complete()

    def test_is_complete_true_all_streams(self):
        ts, f0, voiced = _pitch_output()
        on_ts, on_p, off_p = _onset_output()
        r = self._make_result(
            f0=f0, voiced=voiced,
            phoneme_segments=_phoneme_segments(),
            onset_probs=on_p, offset_probs=off_p,
        )
        assert r.is_complete()

    def test_to_dict_minimal(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["audio_path"] == "test.wav"
        assert "sample_rate" in d
        assert "duration_s" in d

    def test_to_dict_includes_phonemes(self):
        r = self._make_result(phoneme_segments=_phoneme_segments())
        d = r.to_dict()
        assert "phoneme_segments" in d
        assert len(d["phoneme_segments"]) == 3

    def test_to_dict_includes_n_canonical(self):
        ts, f0, voiced = _pitch_output()
        tgt = canonical_timestamps(N_PITCH)
        aligned = FrameAlignedFeatures(timestamps=tgt, f0=f0)
        r = self._make_result(aligned=aligned)
        d = r.to_dict()
        assert d["n_canonical_frames"] == N_PITCH


# ---------------------------------------------------------------------------
# _align() logic (no model weights needed)
# ---------------------------------------------------------------------------

class TestAlignment:
    def _pipeline(self):
        from inference.pipeline import UnifiedInferencePipeline
        return UnifiedInferencePipeline({"pipeline": {
            "enable_pitch": False,
            "enable_phoneme": False,
            "enable_onset_offset": False,
        }})

    def test_align_pitch_only(self):
        p = self._pipeline()
        ts, f0, voiced = _pitch_output()
        result = UnifiedInferenceResult(
            audio_path="x.wav",
            pitch_timestamps=ts, f0=f0, voiced=voiced,
        )
        result = p._align(result)
        assert result.aligned is not None
        assert result.aligned.n_frames == N_PITCH

    def test_align_f0_length(self):
        p = self._pipeline()
        ts, f0, voiced = _pitch_output()
        result = UnifiedInferenceResult(
            audio_path="x.wav",
            pitch_timestamps=ts, f0=f0, voiced=voiced,
        )
        result = p._align(result)
        assert len(result.aligned.f0) == N_PITCH

    def test_align_voiced_dtype(self):
        p = self._pipeline()
        ts, f0, voiced = _pitch_output()
        result = UnifiedInferenceResult(
            audio_path="x.wav",
            pitch_timestamps=ts, f0=f0, voiced=voiced,
        )
        result = p._align(result)
        assert result.aligned.voiced.dtype == bool

    def test_align_onset_resampled_to_canonical(self):
        p = self._pipeline()
        ts, f0, voiced = _pitch_output()
        on_ts, on_p, off_p = _onset_output()
        result = UnifiedInferenceResult(
            audio_path="x.wav",
            pitch_timestamps=ts, f0=f0, voiced=voiced,
            onset_timestamps=on_ts, onset_probs=on_p, offset_probs=off_p,
        )
        result = p._align(result)
        assert result.aligned.onset_probs is not None
        assert len(result.aligned.onset_probs) == N_PITCH

    def test_align_phoneme_labels_length(self):
        p = self._pipeline()
        ts, f0, voiced = _pitch_output()
        result = UnifiedInferenceResult(
            audio_path="x.wav",
            pitch_timestamps=ts, f0=f0, voiced=voiced,
            phoneme_segments=_phoneme_segments(),
        )
        result = p._align(result)
        assert result.aligned.phoneme_labels is not None
        assert len(result.aligned.phoneme_labels) == N_PITCH

    def test_align_no_pitch_skipped(self):
        """If pitch is absent, alignment is skipped gracefully."""
        p = self._pipeline()
        result = UnifiedInferenceResult(audio_path="x.wav")
        result = p._align(result)
        assert result.aligned is None


# ---------------------------------------------------------------------------
# Predict with mocked models
# ---------------------------------------------------------------------------

def _make_mock_pitch_model(ts, f0, voiced):
    """Build a mock PitchInferenceModel whose pipeline returns canned data."""
    from models.pitch.pipeline import PipelineOutput

    mock_pipeline_output = PipelineOutput(
        timestamps=ts,
        f0=f0,
        voiced_mask=voiced,
        vad_mask_raw=None,
        vad_times_raw=None,
        audio=_audio_np(),
        sample_rate=SAMPLE_RATE,
    )
    mock_inner = MagicMock()
    mock_inner.run_from_array.return_value = mock_pipeline_output

    mock_model = MagicMock()
    mock_model._pipeline = mock_inner
    mock_model.is_loaded = True
    mock_model._is_loaded = True
    return mock_model


def _make_mock_phoneme_model(segments):
    mock_model = MagicMock()
    mock_model.predict.return_value = segments
    mock_model.is_loaded = True
    mock_model._is_loaded = True
    return mock_model


class TestPredictWithMocks:
    def _pipeline(self, enable_phoneme=False, enable_onset=False):
        from inference.pipeline import UnifiedInferencePipeline
        return UnifiedInferencePipeline({"pipeline": {
            "enable_pitch": True,
            "enable_phoneme": enable_phoneme,
            "enable_onset_offset": enable_onset,
        }})

    def _mock_audio_load(self):
        audio = _audio_np()
        return audio, SAMPLE_RATE

    def test_predict_returns_result(self, tmp_path):
        p = self._pipeline()
        ts, f0, voiced = _pitch_output()
        mock_pitch = _make_mock_pitch_model(ts, f0, voiced)

        with patch.object(p._preprocessor, "process_for_pitch", return_value=self._mock_audio_load()):
            with patch.object(p, "_get_pitch_model", return_value=mock_pitch):
                wav = tmp_path / "test.wav"
                wav.write_bytes(b"")  # file must exist for Path checks
                import soundfile as sf
                import numpy as np_
                try:
                    sf.write(str(wav), np_.zeros(1600, dtype=np_.float32), 16000)
                except Exception:
                    pass
                result = p.predict(str(wav))

        assert isinstance(result, UnifiedInferenceResult)

    def test_predict_has_pitch_streams(self, tmp_path):
        p = self._pipeline()
        ts, f0, voiced = _pitch_output()
        mock_pitch = _make_mock_pitch_model(ts, f0, voiced)

        with patch.object(p._preprocessor, "process_for_pitch", return_value=self._mock_audio_load()):
            with patch.object(p, "_get_pitch_model", return_value=mock_pitch):
                result = p.predict("fake.wav")

        assert result.has_pitch()
        assert len(result.f0) == N_PITCH
        assert result.aligned is not None

    def test_predict_phoneme_stream_populated(self, tmp_path):
        p = self._pipeline(enable_phoneme=True)
        ts, f0, voiced = _pitch_output()
        segs = _phoneme_segments()

        mock_pitch = _make_mock_pitch_model(ts, f0, voiced)
        mock_phoneme = _make_mock_phoneme_model(segs)

        with patch.object(p._preprocessor, "process_for_pitch", return_value=self._mock_audio_load()):
            with patch.object(p, "_get_pitch_model", return_value=mock_pitch):
                with patch.object(p, "_get_phoneme_model", return_value=mock_phoneme):
                    result = p.predict("fake.wav")

        assert result.has_phonemes()
        assert len(result.phoneme_segments) == 3

    def test_predict_metadata_elapsed(self):
        p = self._pipeline()
        ts, f0, voiced = _pitch_output()
        mock_pitch = _make_mock_pitch_model(ts, f0, voiced)

        with patch.object(p._preprocessor, "process_for_pitch", return_value=self._mock_audio_load()):
            with patch.object(p, "_get_pitch_model", return_value=mock_pitch):
                result = p.predict("fake.wav")

        assert "elapsed_s" in result.metadata
        assert result.metadata["elapsed_s"] >= 0.0

    def test_predict_metadata_device(self):
        p = self._pipeline()
        ts, f0, voiced = _pitch_output()
        mock_pitch = _make_mock_pitch_model(ts, f0, voiced)

        with patch.object(p._preprocessor, "process_for_pitch", return_value=self._mock_audio_load()):
            with patch.object(p, "_get_pitch_model", return_value=mock_pitch):
                result = p.predict("fake.wav")

        assert "device" in result.metadata


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

class TestBatchInference:
    def test_predict_batch_empty_list(self):
        from inference.pipeline import UnifiedInferencePipeline
        p = UnifiedInferencePipeline({"pipeline": {
            "enable_pitch": False, "enable_phoneme": False, "enable_onset_offset": False
        }})
        results = p.predict_batch([])
        assert results == []

    def test_predict_batch_skip_errors(self, tmp_path):
        from inference.pipeline import UnifiedInferencePipeline
        p = UnifiedInferencePipeline({"pipeline": {
            "enable_pitch": False, "enable_phoneme": False, "enable_onset_offset": False
        }})
        # Two files that don't exist — should skip both
        results = p.predict_batch(
            ["nonexistent1.wav", "nonexistent2.wav"],
            skip_errors=True,
        )
        assert results == []

    def test_predict_directory_not_a_dir(self, tmp_path):
        from inference.pipeline import UnifiedInferencePipeline
        p = UnifiedInferencePipeline({})
        with pytest.raises(NotADirectoryError):
            p.predict_directory(tmp_path / "nonexistent_dir")

    def test_predict_directory_empty(self, tmp_path):
        from inference.pipeline import UnifiedInferencePipeline
        p = UnifiedInferencePipeline({"pipeline": {
            "enable_pitch": False, "enable_phoneme": False, "enable_onset_offset": False
        }})
        results = p.predict_directory(tmp_path, pattern="*.wav")
        assert results == []


# ---------------------------------------------------------------------------
# Shape consistency
# ---------------------------------------------------------------------------

class TestShapeConsistency:
    """Verify that all aligned arrays share the same first dimension."""

    def test_all_aligned_arrays_same_length(self):
        from inference.pipeline import UnifiedInferencePipeline
        p = UnifiedInferencePipeline({"pipeline": {
            "enable_pitch": False, "enable_phoneme": False, "enable_onset_offset": False
        }})
        ts, f0, voiced = _pitch_output()
        on_ts, on_p, off_p = _onset_output()
        segs = _phoneme_segments()

        result = UnifiedInferenceResult(
            audio_path="x.wav",
            pitch_timestamps=ts, f0=f0, voiced=voiced,
            onset_timestamps=on_ts, onset_probs=on_p, offset_probs=off_p,
            phoneme_segments=segs,
        )
        result = p._align(result)

        a = result.aligned
        assert a is not None
        n = a.n_frames
        assert len(a.timestamps) == n
        assert len(a.f0) == n
        assert len(a.voiced) == n
        assert len(a.onset_probs) == n
        assert len(a.offset_probs) == n
        assert len(a.phoneme_labels) == n
