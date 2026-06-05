"""
tests/test_alignment.py - Unit tests for fusion/alignment.py

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_alignment.py -v
"""

import numpy as np
import pytest

from fusion.alignment import (
    align_mask_to_canonical,
    align_to_canonical,
    merge_model_outputs,
    resample_mask,
    resample_sequence,
    segments_to_frame_labels,
    snap_boundary,
)
from preprocessing.timestamps import HOP_LENGTH, SAMPLE_RATE, canonical_timestamps
from utils.types import FrameAlignedFeatures, PhonemeSegment


# ---------------------------------------------------------------------------
# resample_mask
# ---------------------------------------------------------------------------

class TestResampleMask:
    def test_identity_same_grid(self):
        t = np.array([0.005, 0.015, 0.025])
        mask = np.array([True, False, True])
        out = resample_mask(t, mask, t)
        np.testing.assert_array_equal(out, mask)

    def test_upsample_2x(self):
        # 50fps → 100fps: each source frame maps to two target frames
        src_t = np.array([0.01, 0.03])   # 50fps frame centers
        src_m = np.array([True, False])
        tgt_t = np.array([0.005, 0.015, 0.025, 0.035])  # 100fps
        out = resample_mask(src_t, src_m, tgt_t)
        # nearest: [0.005→0.01(T), 0.015→0.01(T), 0.025→0.03(F), 0.035→0.03(F)]
        expected = [True, True, False, False]
        np.testing.assert_array_equal(out, expected)

    def test_output_dtype_bool(self):
        t = np.array([0.005])
        m = np.array([True])
        out = resample_mask(t, m, t)
        assert out.dtype == bool

    def test_all_voiced(self):
        src_t = np.linspace(0.005, 0.995, 50)
        src_m = np.ones(50, dtype=bool)
        tgt_t = np.linspace(0.005, 0.995, 100)
        out = resample_mask(src_t, src_m, tgt_t)
        assert out.all()

    def test_all_unvoiced(self):
        src_t = np.linspace(0.005, 0.995, 50)
        src_m = np.zeros(50, dtype=bool)
        tgt_t = np.linspace(0.005, 0.995, 100)
        out = resample_mask(src_t, src_m, tgt_t)
        assert not out.any()


# ---------------------------------------------------------------------------
# resample_sequence
# ---------------------------------------------------------------------------

class TestResampleSequence:
    def test_linear_midpoint(self):
        src_t = np.array([0.0, 1.0])
        src_v = np.array([0.0, 100.0])
        tgt_t = np.array([0.0, 0.5, 1.0])
        out = resample_sequence(src_t, src_v, tgt_t)
        np.testing.assert_allclose(out, [0.0, 50.0, 100.0], atol=1e-4)

    def test_dtype_preserved(self):
        src_t = np.array([0.0, 1.0])
        src_v = np.array([0.0, 1.0], dtype=np.float32)
        tgt_t = np.array([0.0, 0.5, 1.0])
        out = resample_sequence(src_t, src_v, tgt_t)
        assert out.dtype == np.float32

    def test_downsampling(self):
        # 100fps → 50fps: average-ish behaviour
        src_t = canonical_timestamps(10, hop_length=160, sample_rate=16000).astype(np.float64)
        src_v = np.arange(10, dtype=np.float32)
        tgt_t = canonical_timestamps(5, hop_length=320, sample_rate=16000).astype(np.float64)
        out = resample_sequence(src_t, src_v, tgt_t)
        assert len(out) == 5

    def test_f0_zero_fill_outside(self):
        src_t = np.array([0.1, 0.9])
        src_v = np.array([220.0, 440.0])
        tgt_t = np.array([0.0, 0.5, 1.0])
        out = resample_sequence(src_t, src_v, tgt_t, fill_value=0.0)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# align_to_canonical / align_mask_to_canonical
# ---------------------------------------------------------------------------

class TestAlignToCanonical:
    def test_output_length(self):
        src_t = np.linspace(0.01, 0.99, 50)
        src_v = np.ones(50, dtype=np.float32)
        out = align_to_canonical(src_t, src_v, n_canonical=100)
        assert len(out) == 100

    def test_mask_output_bool(self):
        src_t = np.linspace(0.01, 0.99, 50)
        src_m = np.ones(50, dtype=bool)
        out = align_mask_to_canonical(src_t, src_m, n_canonical=100)
        assert out.dtype == bool
        assert len(out) == 100


# ---------------------------------------------------------------------------
# segments_to_frame_labels
# ---------------------------------------------------------------------------

class TestSegmentsToFrameLabels:
    def _seg(self, phoneme, start, end):
        return PhonemeSegment(phoneme=phoneme, start_time=start, end_time=end, confidence=1.0)

    def test_single_segment_covers_all(self):
        segs = [self._seg("AH", 0.0, 0.1)]
        labels = segments_to_frame_labels(segs, n_frames=10)
        assert all(l == "AH" for l in labels)

    def test_two_segments_split(self):
        # Frame 0–4 → "AH" (0.0–0.05), Frame 5–9 → "EH" (0.05–0.1)
        segs = [self._seg("AH", 0.0, 0.05), self._seg("EH", 0.05, 0.1)]
        labels = segments_to_frame_labels(segs, n_frames=10)
        assert labels[0] == "AH"
        assert labels[9] == "EH"

    def test_gap_between_segments_empty(self):
        # Gap from 0.03 to 0.07 → frames 3,4,5,6 are unlabelled
        segs = [self._seg("AH", 0.0, 0.03), self._seg("EH", 0.07, 0.1)]
        labels = segments_to_frame_labels(segs, n_frames=10)
        assert labels[0] == "AH"
        assert labels[9] == "EH"
        # Frames 3–6 should be empty (no overlap)
        for i in (3, 4, 5, 6):
            assert labels[i] == "", f"Expected empty at frame {i}, got '{labels[i]}'"

    def test_empty_segments(self):
        labels = segments_to_frame_labels([], n_frames=5)
        assert labels == [""] * 5


# ---------------------------------------------------------------------------
# snap_boundary
# ---------------------------------------------------------------------------

class TestSnapBoundary:
    def test_snaps_to_nearest_center(self):
        # 0.007s is closest to frame 0 center (0.005)
        snapped = snap_boundary(0.007, hop_length=160, sample_rate=16000, center=True)
        assert abs(snapped - 0.005) < 1e-6

    def test_snaps_to_frame_start(self):
        snapped = snap_boundary(0.007, hop_length=160, sample_rate=16000, center=False)
        # nearest frame start: 0.0 (dist=0.007) or 0.01 (dist=0.003) → 0.01
        assert abs(snapped - 0.01) < 1e-6

    def test_non_negative(self):
        snapped = snap_boundary(-0.1)
        assert snapped >= 0.0


# ---------------------------------------------------------------------------
# merge_model_outputs
# ---------------------------------------------------------------------------

class TestMergeModelOutputs:
    def _pitch_data(self, n=100):
        t = canonical_timestamps(n, hop_length=160, sample_rate=16000).astype(np.float64)
        f0 = np.random.uniform(100, 400, size=n).astype(np.float32)
        voiced = np.random.rand(n) > 0.3
        return t, f0, voiced

    def _onset_data(self, n=62):
        # Simulate onset model at hop=256 (~62.5fps)
        t = canonical_timestamps(n, hop_length=256, sample_rate=16000).astype(np.float64)
        probs = np.random.rand(n).astype(np.float32)
        return t, probs

    def test_returns_frame_aligned_features(self):
        pt, f0, voiced = self._pitch_data(100)
        result = merge_model_outputs(
            n_canonical=100,
            pitch_times=pt,
            f0=f0,
            voiced=voiced,
        )
        assert isinstance(result, FrameAlignedFeatures)

    def test_canonical_length(self):
        pt, f0, voiced = self._pitch_data(100)
        result = merge_model_outputs(100, pitch_times=pt, f0=f0, voiced=voiced)
        assert result.n_frames == 100
        assert len(result.timestamps) == 100

    def test_f0_length(self):
        pt, f0, voiced = self._pitch_data(100)
        result = merge_model_outputs(100, pitch_times=pt, f0=f0, voiced=voiced)
        assert len(result.f0) == 100

    def test_voiced_dtype(self):
        pt, f0, voiced = self._pitch_data(100)
        result = merge_model_outputs(100, pitch_times=pt, f0=f0, voiced=voiced)
        assert result.voiced.dtype == bool

    def test_onset_resampled(self):
        pt, f0, voiced = self._pitch_data(100)
        ot, onset_probs = self._onset_data(62)
        result = merge_model_outputs(
            100,
            pitch_times=pt, f0=f0, voiced=voiced,
            onset_times=ot, onset_probs=onset_probs,
        )
        assert result.onset_probs is not None
        assert len(result.onset_probs) == 100

    def test_missing_streams_are_none(self):
        pt, f0, voiced = self._pitch_data(100)
        result = merge_model_outputs(100, pitch_times=pt, f0=f0, voiced=voiced)
        assert result.onset_probs is None
        assert result.offset_probs is None
        assert result.phoneme_labels is None

    def test_phoneme_labels_length(self):
        pt, f0, voiced = self._pitch_data(100)
        segs = [
            PhonemeSegment("AH", 0.0, 0.5, 1.0),
            PhonemeSegment("EH", 0.5, 1.0, 1.0),
        ]
        result = merge_model_outputs(
            100,
            pitch_times=pt, f0=f0, voiced=voiced,
            phoneme_segments=segs,
        )
        assert result.phoneme_labels is not None
        assert len(result.phoneme_labels) == 100

    def test_all_none_inputs(self):
        result = merge_model_outputs(50)
        assert result.n_frames == 50
        assert result.f0 is None
        assert result.voiced is None
