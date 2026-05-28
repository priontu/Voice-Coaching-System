"""
tests/test_timestamps.py - Unit tests for preprocessing/timestamps.py

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_timestamps.py -v
"""

import numpy as np
import pytest

from preprocessing.timestamps import (
    FRAME_DURATION,
    HOP_LENGTH,
    SAMPLE_RATE,
    align_to_grid,
    canonical_timestamps,
    duration_to_frames,
    frame_overlap_ratio,
    frames_to_duration,
    frames_to_times,
    samples_to_frames,
    snap_to_frame,
    times_to_frames,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_frame_duration_matches_hop(self):
        assert abs(HOP_LENGTH / SAMPLE_RATE - FRAME_DURATION) < 1e-10

    def test_canonical_fps(self):
        assert SAMPLE_RATE / HOP_LENGTH == 100  # 100 fps


# ---------------------------------------------------------------------------
# frames_to_times
# ---------------------------------------------------------------------------

class TestFramesToTimes:
    def test_center_true_first_frame(self):
        t = frames_to_times(0, hop_length=160, sample_rate=16000, center=True)
        assert abs(float(t) - 0.005) < 1e-6  # half of 10ms

    def test_center_false_first_frame(self):
        t = frames_to_times(0, hop_length=160, sample_rate=16000, center=False)
        assert abs(float(t) - 0.0) < 1e-10

    def test_center_true_second_frame(self):
        t = frames_to_times(1, hop_length=160, sample_rate=16000, center=True)
        assert abs(float(t) - 0.015) < 1e-6  # 10ms + 5ms

    def test_array_input(self):
        t = frames_to_times(np.arange(5), hop_length=160, sample_rate=16000, center=False)
        expected = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        np.testing.assert_allclose(t, expected, atol=1e-6)

    def test_phoneme_hop(self):
        # Wav2Vec2 hop = 320 → 20ms per frame
        t = frames_to_times(1, hop_length=320, sample_rate=16000, center=False)
        assert abs(float(t) - 0.02) < 1e-6


# ---------------------------------------------------------------------------
# times_to_frames
# ---------------------------------------------------------------------------

class TestTimesToFrames:
    def test_roundtrip_center_true(self):
        for frame in range(10):
            t = float(frames_to_times(frame, center=True))
            recovered = int(times_to_frames(t, center=True))
            assert recovered == frame, f"frame {frame}: got {recovered}"

    def test_roundtrip_center_false(self):
        for frame in range(10):
            t = float(frames_to_times(frame, center=False))
            recovered = int(times_to_frames(t, center=False))
            assert recovered == frame

    def test_clips_to_zero(self):
        frames = times_to_frames(-0.5)
        assert int(frames) == 0

    def test_array_input(self):
        times = np.array([0.0, 0.01, 0.02])
        frames = times_to_frames(times, center=False)
        np.testing.assert_array_equal(frames, [0, 1, 2])


# ---------------------------------------------------------------------------
# canonical_timestamps
# ---------------------------------------------------------------------------

class TestCanonicalTimestamps:
    def test_length(self):
        ts = canonical_timestamps(50)
        assert len(ts) == 50

    def test_dtype(self):
        ts = canonical_timestamps(10)
        assert ts.dtype == np.float32

    def test_center_true_spacing(self):
        ts = canonical_timestamps(5, hop_length=160, sample_rate=16000, center=True)
        diffs = np.diff(ts)
        np.testing.assert_allclose(diffs, 0.01, atol=1e-6)

    def test_first_value_center(self):
        ts = canonical_timestamps(1, center=True)
        assert abs(float(ts[0]) - 0.005) < 1e-6


# ---------------------------------------------------------------------------
# duration / sample helpers
# ---------------------------------------------------------------------------

class TestDurationHelpers:
    def test_duration_to_frames_exact(self):
        # 1 second at 100fps = 100 frames
        assert duration_to_frames(1.0, hop_length=160, sample_rate=16000) == 100

    def test_duration_to_frames_ceil(self):
        # 1.005 seconds → 101 frames (ceil)
        assert duration_to_frames(1.005, hop_length=160, sample_rate=16000) == 101

    def test_frames_to_duration(self):
        d = frames_to_duration(100, hop_length=160, sample_rate=16000)
        assert abs(d - 1.0) < 1e-10

    def test_samples_to_frames(self):
        # 1600 samples at hop=160 → 10 frames
        assert samples_to_frames(1600, hop_length=160) == 10

    def test_samples_to_frames_partial(self):
        # 1601 samples → still 11 frames (covers the extra sample)
        assert samples_to_frames(1601, hop_length=160) == 11


# ---------------------------------------------------------------------------
# align_to_grid
# ---------------------------------------------------------------------------

class TestAlignToGrid:
    def test_nearest_upsampling(self):
        # Source: [0.0, 0.02, 0.04] (50fps), target: [0.0, 0.01, 0.02, 0.03, 0.04] (100fps)
        src_t = np.array([0.0, 0.02, 0.04])
        src_v = np.array([1.0, 2.0, 3.0])
        tgt_t = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        out = align_to_grid(src_t, src_v, tgt_t, kind="nearest")
        # Each target frame gets its nearest source value
        assert out[0] == pytest.approx(1.0)
        assert out[2] == pytest.approx(2.0)
        assert out[4] == pytest.approx(3.0)

    def test_linear_interpolation(self):
        src_t = np.array([0.0, 1.0])
        src_v = np.array([0.0, 10.0])
        tgt_t = np.array([0.0, 0.5, 1.0])
        out = align_to_grid(src_t, src_v, tgt_t, kind="linear")
        np.testing.assert_allclose(out, [0.0, 5.0, 10.0], atol=1e-5)

    def test_fill_value_outside_range(self):
        src_t = np.array([0.1, 0.9])
        src_v = np.array([1.0, 1.0])
        tgt_t = np.array([0.0, 0.5, 1.0])
        out = align_to_grid(src_t, src_v, tgt_t, kind="linear", fill_value=0.0)
        assert out[0] == pytest.approx(0.0)   # before source range
        assert out[-1] == pytest.approx(0.0)  # after source range

    def test_boolean_mask_nearest(self):
        src_t = np.array([0.0, 0.02, 0.04])
        src_v = np.array([False, True, False])
        tgt_t = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        out = align_to_grid(src_t, src_v.astype(np.float32), tgt_t, kind="nearest")
        assert not bool(out[0] > 0.5)
        assert bool(out[2] > 0.5)

    def test_empty_source(self):
        out = align_to_grid(np.array([]), np.array([]), np.array([0.0, 0.01]), fill_value=-1.0)
        assert out[0] == pytest.approx(-1.0)

    def test_single_source(self):
        out = align_to_grid(np.array([0.5]), np.array([7.0]), np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(out, [7.0, 7.0, 7.0], atol=1e-5)


# ---------------------------------------------------------------------------
# snap_to_frame
# ---------------------------------------------------------------------------

class TestSnapToFrame:
    def test_snaps_to_center(self):
        # 0.007 seconds → nearest frame center at 0.005 or 0.015?
        snapped = snap_to_frame(0.007, hop_length=160, sample_rate=16000, center=True)
        # 0.007 is closer to frame 0 center (0.005) than frame 1 center (0.015)
        assert abs(snapped - 0.005) < 1e-6

    def test_snaps_to_next_frame(self):
        snapped = snap_to_frame(0.013, hop_length=160, sample_rate=16000, center=True)
        assert abs(snapped - 0.015) < 1e-6


# ---------------------------------------------------------------------------
# frame_overlap_ratio
# ---------------------------------------------------------------------------

class TestFrameOverlapRatio:
    def test_full_overlap(self):
        # segment exactly covering frame 0 (0.0 to 0.01 at hop=160, sr=16000)
        ratio = frame_overlap_ratio(0.0, 0.01, frame_idx=0, hop_length=160, sample_rate=16000)
        assert abs(ratio - 1.0) < 1e-6

    def test_half_overlap(self):
        # segment 0.0 to 0.005 covers half of frame 0
        ratio = frame_overlap_ratio(0.0, 0.005, frame_idx=0, hop_length=160, sample_rate=16000)
        assert abs(ratio - 0.5) < 1e-6

    def test_no_overlap(self):
        # segment 0.05 to 0.1 does not overlap frame 0
        ratio = frame_overlap_ratio(0.05, 0.1, frame_idx=0, hop_length=160, sample_rate=16000)
        assert ratio == 0.0
