"""
tests/test_note_events.py - Unit tests for fusion/note_events.py

All tests use synthetic numpy arrays; no model weights or GPU required.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_note_events.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.timestamps import HOP_LENGTH, SAMPLE_RATE, canonical_timestamps
from fusion.note_events import build_note_events, estimate_tempo
from utils.types import NoteEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 200   # 2 seconds at 100fps


def _timestamps():
    return canonical_timestamps(N).astype(np.float64)


def _flat_probs(val=0.0):
    return np.full(N, val, dtype=np.float32)


def _spike_at(frames, val=0.9, base=0.0):
    """Probability array with spikes at given frame indices."""
    arr = np.full(N, base, dtype=np.float32)
    for f in frames:
        arr[f] = val
    return arr


def _f0_and_voiced():
    """Sinusoidal F0 (200–400 Hz) with voiced mask (first 150 frames voiced)."""
    f0 = np.random.uniform(200, 400, N).astype(np.float32)
    voiced = np.zeros(N, dtype=bool)
    voiced[:150] = True
    return f0, voiced


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestBuildNoteEventsBasic:
    def test_empty_on_all_zero_probs(self):
        ts = _timestamps()
        events = build_note_events(ts, _flat_probs(0.0), _flat_probs(0.0))
        assert events == []

    def test_empty_on_below_threshold(self):
        ts = _timestamps()
        onset = _flat_probs(0.3)   # below default 0.5
        offset = _flat_probs(0.3)
        events = build_note_events(ts, onset, offset)
        assert events == []

    def test_single_note_detected(self):
        ts = _timestamps()
        onset = _spike_at([20])
        offset = _spike_at([60])
        events = build_note_events(ts, onset, offset, min_duration_s=0.0)
        assert len(events) >= 1

    def test_returns_note_event_objects(self):
        ts = _timestamps()
        events = build_note_events(ts, _spike_at([10]), _spike_at([50]), min_duration_s=0.0)
        for ev in events:
            assert isinstance(ev, NoteEvent)

    def test_onset_before_offset(self):
        ts = _timestamps()
        events = build_note_events(ts, _spike_at([10]), _spike_at([50]), min_duration_s=0.0)
        for ev in events:
            assert ev.onset_time < ev.offset_time

    def test_positive_duration(self):
        ts = _timestamps()
        events = build_note_events(ts, _spike_at([10]), _spike_at([50]), min_duration_s=0.0)
        for ev in events:
            assert ev.duration is not None
            assert ev.duration > 0

    def test_min_duration_filter(self):
        ts = _timestamps()
        # Onset at frame 10, offset at frame 14 → ~40ms < 50ms default
        onset = _spike_at([10])
        offset = _spike_at([14])
        events = build_note_events(ts, onset, offset, min_duration_s=0.05)
        assert events == []

    def test_multiple_notes(self):
        ts = _timestamps()
        onset = _spike_at([10, 80, 150])
        offset = _spike_at([40, 120, 190])
        events = build_note_events(ts, onset, offset, min_duration_s=0.0)
        assert len(events) >= 2

    def test_sorted_by_onset(self):
        ts = _timestamps()
        onset = _spike_at([10, 80, 150])
        offset = _spike_at([40, 120, 190])
        events = build_note_events(ts, onset, offset, min_duration_s=0.0)
        onsets = [ev.onset_time for ev in events]
        assert onsets == sorted(onsets)

    def test_note_idx_set(self):
        ts = _timestamps()
        onset = _spike_at([10, 80])
        offset = _spike_at([40, 120])
        events = build_note_events(ts, onset, offset, min_duration_s=0.0)
        for i, ev in enumerate(events):
            assert ev.note_idx == i


# ---------------------------------------------------------------------------
# Pitch statistics
# ---------------------------------------------------------------------------

class TestPitchStats:
    def test_pitch_hz_populated(self):
        ts = _timestamps()
        f0, voiced = _f0_and_voiced()
        events = build_note_events(ts, _spike_at([10]), _spike_at([80]), f0=f0, voiced=voiced, min_duration_s=0.0)
        assert len(events) > 0
        assert events[0].pitch_hz is not None
        assert events[0].pitch_hz > 0

    def test_pitch_midi_in_range(self):
        ts = _timestamps()
        f0, voiced = _f0_and_voiced()
        events = build_note_events(ts, _spike_at([10]), _spike_at([80]), f0=f0, voiced=voiced, min_duration_s=0.0)
        if events and events[0].pitch_midi is not None:
            assert 0.0 <= events[0].pitch_midi <= 127.0

    def test_pitch_stability_nonneg(self):
        ts = _timestamps()
        f0, voiced = _f0_and_voiced()
        events = build_note_events(ts, _spike_at([10]), _spike_at([80]), f0=f0, voiced=voiced, min_duration_s=0.0)
        if events and events[0].pitch_stability is not None:
            assert events[0].pitch_stability >= 0.0

    def test_voiced_fraction_in_range(self):
        ts = _timestamps()
        f0, voiced = _f0_and_voiced()
        events = build_note_events(ts, _spike_at([10]), _spike_at([80]), f0=f0, voiced=voiced, min_duration_s=0.0)
        if events and events[0].voiced_fraction is not None:
            assert 0.0 <= events[0].voiced_fraction <= 1.0

    def test_no_f0_gives_none_pitch(self):
        ts = _timestamps()
        events = build_note_events(ts, _spike_at([10]), _spike_at([80]), f0=None, min_duration_s=0.0)
        if events:
            assert events[0].pitch_hz is None
            assert events[0].pitch_midi is None

    def test_unvoiced_region_gives_none_pitch(self):
        ts = _timestamps()
        f0 = np.random.uniform(200, 400, N).astype(np.float32)
        voiced = np.zeros(N, dtype=bool)   # all unvoiced
        events = build_note_events(ts, _spike_at([10]), _spike_at([80]), f0=f0, voiced=voiced, min_duration_s=0.0)
        if events:
            assert events[0].pitch_hz is None


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_nonneg(self):
        ts = _timestamps()
        events = build_note_events(ts, _spike_at([10]), _spike_at([80]), min_duration_s=0.0)
        for ev in events:
            assert ev.confidence is not None
            assert ev.confidence >= 0.0

    def test_onset_confidence_set(self):
        ts = _timestamps()
        events = build_note_events(ts, _spike_at([10], val=0.9), _spike_at([80]), min_duration_s=0.0)
        for ev in events:
            assert ev.onset_confidence is not None

    def test_offset_confidence_set(self):
        ts = _timestamps()
        events = build_note_events(ts, _spike_at([10]), _spike_at([80], val=0.8), min_duration_s=0.0)
        for ev in events:
            assert ev.offset_confidence is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_timestamps(self):
        events = build_note_events(np.array([]), np.array([]), np.array([]))
        assert events == []

    def test_single_frame(self):
        ts = canonical_timestamps(1).astype(np.float64)
        events = build_note_events(ts, np.array([0.9], dtype=np.float32), np.array([0.9], dtype=np.float32))
        assert isinstance(events, list)

    def test_all_above_threshold_no_peaks(self):
        ts = _timestamps()
        # Flat signal at 0.9 — no local peaks
        onset = np.full(N, 0.9, dtype=np.float32)
        offset = np.full(N, 0.9, dtype=np.float32)
        events = build_note_events(ts, onset, offset)
        assert isinstance(events, list)

    def test_max_duration_filter(self):
        ts = _timestamps()
        # Onset at frame 5, offset at frame 195 → nearly 2s > max_duration_s=1s
        onset = _spike_at([5])
        offset = _spike_at([195])
        events = build_note_events(ts, onset, offset, max_duration_s=1.0, min_duration_s=0.0)
        # Pair should be dropped due to max_duration_s
        for ev in events:
            assert ev.duration is not None and ev.duration <= 1.0 + 0.1  # small tolerance for pseudo-offset

    def test_deterministic(self):
        ts = _timestamps()
        onset = _spike_at([10, 80, 150])
        offset = _spike_at([40, 120, 190])
        r1 = build_note_events(ts, onset, offset, min_duration_s=0.0)
        r2 = build_note_events(ts, onset, offset, min_duration_s=0.0)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.onset_time == b.onset_time
            assert a.offset_time == b.offset_time


# ---------------------------------------------------------------------------
# Tempo estimation
# ---------------------------------------------------------------------------

class TestEstimateTempo:
    def test_returns_none_for_single_note(self):
        n = NoteEvent(onset_time=0.0, offset_time=0.5)
        assert estimate_tempo([n]) is None

    def test_returns_none_for_empty(self):
        assert estimate_tempo([]) is None

    def test_regular_tempo(self):
        # Quarter note at 120 BPM → IOI = 0.5s
        notes = [NoteEvent(onset_time=i * 0.5, offset_time=i * 0.5 + 0.4) for i in range(8)]
        tempo = estimate_tempo(notes)
        assert tempo is not None
        assert abs(tempo - 120.0) < 5.0

    def test_returns_float(self):
        notes = [NoteEvent(onset_time=i * 0.5, offset_time=i * 0.5 + 0.4) for i in range(4)]
        assert isinstance(estimate_tempo(notes), float)
