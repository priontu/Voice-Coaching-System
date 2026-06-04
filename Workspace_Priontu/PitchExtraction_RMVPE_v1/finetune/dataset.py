"""GTSinger pitch dataset for RMVPE fine-tuning.

Loads GTSinger/English clips and generates frame-level F0 ground truth
from MIDI note annotations in the accompanying JSON files.

Each sample returns a 3-second audio segment (48 000 samples at 16 kHz)
and the corresponding soft pitch posteriorgram target (T_frames × 360)
using RMVPE's own frequency-bin scheme.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
from torch.utils.data import Dataset

# ── RMVPE pitch-bin constants (must match rmvpe_src/constants.py) ─────────────
CONST = 1997.3794084376191   # cents offset (bin 0 ≈ 31.8 Hz ≈ B0)
N_PITCH_BINS = 360
CENTS_PER_BIN = 20.0

SAMPLE_RATE = 16_000
HOP_LENGTH = 160            # 10 ms per frame at 16 kHz
SEGMENT_SAMPLES = 48_000    # 3 s of audio
SIGMA_BINS = 1.5            # Gaussian width for soft label (≈ 30 cents)

EXCLUDE_GROUPS = {"Paired_Speech_Group"}
_GTSINGER_SR = 48_000       # all GTSinger files are 48 kHz — avoids per-file info() calls


# ── Frequency ↔ pitch-bin conversion ──────────────────────────────────────────

def hz_to_bin(f0_hz: np.ndarray) -> np.ndarray:
    """Convert Hz to RMVPE bin index (float). Unvoiced (f0 ≤ 0) → -1."""
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    bins = np.full_like(f0_hz, -1.0)
    voiced = f0_hz > 0
    cents = 1200.0 * np.log2(f0_hz[voiced] / 10.0)
    bins[voiced] = (cents - CONST) / CENTS_PER_BIN
    return bins


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def soft_pitch_label(bin_float: float, n_bins: int = N_PITCH_BINS,
                     sigma: float = SIGMA_BINS) -> np.ndarray:
    """Gaussian soft label centered at bin_float, shape (360,)."""
    idx = np.arange(n_bins, dtype=np.float64)
    label = np.exp(-0.5 * ((idx - bin_float) / sigma) ** 2).astype(np.float32)
    # Clip bins outside valid range to zero
    label[bin_float < 0] = 0.0
    label[bin_float >= n_bins] = 0.0
    return label


# ── Ground-truth F0 from GTSinger JSON ────────────────────────────────────────

def build_f0_track(notes_json: list[dict], n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Build frame-level (f0_hz, voiced) arrays at 100 fps from GTSinger JSON.

    Notes with MIDI pitch == 0 (<SP> silence tokens) are treated as unvoiced.
    For clips with the glissando flag, we still use the written MIDI pitch as
    an approximate target (Gaussian soft label provides ±30 cent tolerance).
    """
    f0 = np.zeros(n_frames, dtype=np.float32)
    voiced = np.zeros(n_frames, dtype=bool)

    for word in notes_json:
        midi_notes = word.get("note", [])
        note_starts = word.get("note_start", [])
        note_ends = word.get("note_end", [])

        for midi, ns, ne in zip(midi_notes, note_starts, note_ends):
            if not midi:          # 0 or missing → silence
                continue
            start_frame = max(0, int(round(float(ns) * 100)))
            end_frame = min(n_frames, int(round(float(ne) * 100)))
            if start_frame >= end_frame:
                continue
            f0[start_frame:end_frame] = midi_to_hz(midi)
            voiced[start_frame:end_frame] = True

    return f0, voiced


# ── Clip discovery ─────────────────────────────────────────────────────────────

class GTSingerClip(NamedTuple):
    audio_path: Path
    json_path: Path


def _scan_group_dir(group_dir: Path) -> list[GTSingerClip]:
    """Return all (wav, json) pairs in one leaf group directory."""
    json_stems = {p.stem for p in group_dir.glob("*.json")}
    return [
        GTSingerClip(wav, wav.with_suffix(".json"))
        for wav in sorted(group_dir.glob("*.wav"))
        if wav.stem in json_stems
    ]


def discover_clips(root: str | Path,
                   voice_types: list[str] | None = None) -> list[GTSingerClip]:
    """Find (wav, json) pairs under GTSinger/English/<voice>/...

    Skips Paired_Speech_Group. Uses a thread pool to scan leaf directories in
    parallel — cuts NFS latency from ~20 s to ~4 s on a typical cluster FS.
    """
    root = Path(root)
    if voice_types:
        voice_dirs = [root / vt for vt in voice_types if (root / vt).is_dir()]
    else:
        voice_dirs = [d for d in root.iterdir() if d.is_dir()]

    # Collect all leaf group directories first (fast — just directory listings)
    group_dirs: list[Path] = []
    for voice_dir in voice_dirs:
        for technique_dir in voice_dir.iterdir():
            if not technique_dir.is_dir():
                continue
            for song_dir in technique_dir.iterdir():
                if not song_dir.is_dir():
                    continue
                for group_dir in song_dir.iterdir():
                    if group_dir.is_dir() and group_dir.name not in EXCLUDE_GROUPS:
                        group_dirs.append(group_dir)

    # Scan leaf directories in parallel (each does 2 globs: *.wav + *.json)
    clips: list[GTSingerClip] = []
    with ThreadPoolExecutor(max_workers=min(32, len(group_dirs))) as pool:
        for result in pool.map(_scan_group_dir, group_dirs):
            clips.extend(result)

    return sorted(clips)  # deterministic order


# ── Dataset ────────────────────────────────────────────────────────────────────

class GTSingerPitchDataset(Dataset):
    """Returns fixed-length segments ready for RMVPE fine-tuning.

    Each item:
        audio   : (48 000,) float32 — 3 s at 16 kHz
        target  : (T_frames, 360) float32 — Gaussian soft pitch posterior
        voiced  : (T_frames,) float32 — 1 = voiced frame, 0 = unvoiced
    where T_frames = 1 + SEGMENT_SAMPLES // HOP_LENGTH = 301.
    """

    def __init__(self, clips: list[GTSingerClip], stride_frames: int = 150):
        """
        Args:
            clips: list of (audio_path, json_path) pairs
            stride_frames: hop between consecutive segments in frames (default
                           150 = 1.5 s, giving 50 % overlap with a 3 s window)
        """
        self.clips = clips
        self.stride_frames = stride_frames
        self.n_frames_per_seg = 1 + SEGMENT_SAMPLES // HOP_LENGTH  # 301

        # Build flat index: (clip_idx, start_frame)
        self._index: list[tuple[int, int]] = []
        self._build_index()

    # ── Index building ─────────────────────────────────────────────────────────

    @staticmethod
    def _n_frames_from_json(json_path: Path) -> int:
        """Estimate clip length in 100-fps frames from JSON note timings.

        Reads the JSON file (small, already needed for labels) instead of the
        audio header — avoids a second NFS round-trip per clip on startup.
        Falls back to 750 frames (~7.5 s, the dataset median) on any error.
        """
        try:
            with open(json_path) as fh:
                notes = json.load(fh)
            max_end = max(
                (float(ne) for w in notes for ne in w.get("note_end", []) if ne),
                default=0.0,
            )
            return int(max_end * 100) + 20   # +20 frame buffer
        except Exception:
            return 750  # fallback

    def _build_index(self) -> None:
        seg = self.n_frames_per_seg
        stride = self.stride_frames

        def _segments_for(clip_idx: int) -> list[tuple[int, int]]:
            clip = self.clips[clip_idx]
            n_frames = self._n_frames_from_json(clip.json_path)
            segs: list[tuple[int, int]] = []
            start = 0
            while start + seg <= n_frames:
                segs.append((clip_idx, start))
                start += stride
            # Always include at least one segment; tail segment covers clip end
            if not segs:
                segs.append((clip_idx, 0))
            elif n_frames >= seg:
                last = n_frames - seg
                if last > segs[-1][1]:
                    segs.append((clip_idx, last))
            return segs

        # Parallel JSON reads — I/O bound, so threads help on network filesystems
        n_workers = min(32, len(self.clips))
        results: dict[int, list[tuple[int, int]]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_segments_for, i): i for i in range(len(self.clips))}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

        for i in range(len(self.clips)):
            self._index.extend(results[i])

    @classmethod
    def _from_cache(cls, clips: list[GTSingerClip],
                    index: list[tuple[int, int]],
                    stride_frames: int) -> "GTSingerPitchDataset":
        """Reconstruct a dataset from a pre-built index (no NFS scanning)."""
        obj = cls.__new__(cls)
        obj.clips = clips
        obj.stride_frames = stride_frames
        obj.n_frames_per_seg = 1 + SEGMENT_SAMPLES // HOP_LENGTH
        obj._index = index
        return obj

    def __len__(self) -> int:
        return len(self._index)

    # ── Item loading ───────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clip_idx, seg_start_frame = self._index[idx]
        clip = self.clips[clip_idx]
        seg_end_frame = seg_start_frame + self.n_frames_per_seg

        # ── Audio — read only the 3 s slice, not the full file ────────────────
        # Positions in the file's native sample rate (48 kHz for GTSinger)
        native_start = int(seg_start_frame / 100 * _GTSINGER_SR)
        native_end   = native_start + int(SEGMENT_SAMPLES / SAMPLE_RATE * _GTSINGER_SR)
        audio_np, sr = sf.read(
            str(clip.audio_path), start=native_start, stop=native_end,
            dtype="float32", always_2d=True,
        )
        waveform = torch.from_numpy(audio_np.mean(axis=1))   # [native_samples]
        if sr != SAMPLE_RATE:
            waveform = TAF.resample(waveform, sr, SAMPLE_RATE)
        # Exact-length crop / zero-pad to SEGMENT_SAMPLES
        if waveform.shape[0] >= SEGMENT_SAMPLES:
            audio_seg = waveform[:SEGMENT_SAMPLES]
        else:
            audio_seg = F.pad(waveform, (0, SEGMENT_SAMPLES - waveform.shape[0]))

        # ── Ground-truth F0 — build only the segment window ──────────────────
        with open(clip.json_path) as fh:
            notes_json = json.load(fh)

        f0_seg, voiced_seg = build_f0_track(notes_json, seg_end_frame)
        f0_seg     = f0_seg[seg_start_frame:seg_end_frame]
        voiced_seg = voiced_seg[seg_start_frame:seg_end_frame]

        if len(f0_seg) < self.n_frames_per_seg:
            pad = self.n_frames_per_seg - len(f0_seg)
            f0_seg     = np.pad(f0_seg,     (0, pad))
            voiced_seg = np.pad(voiced_seg, (0, pad))

        # ── Soft pitch target ────────────────────────────────────────────────
        bins = hz_to_bin(f0_seg)
        target = np.zeros((self.n_frames_per_seg, N_PITCH_BINS), dtype=np.float32)
        for t in range(self.n_frames_per_seg):
            if voiced_seg[t] and 0.0 <= bins[t] < N_PITCH_BINS:
                target[t] = soft_pitch_label(bins[t])

        return {
            "audio": audio_seg,
            "target": torch.from_numpy(target),           # (T_frames, 360)
            "voiced": torch.from_numpy(voiced_seg.astype(np.float32)),  # (T_frames,)
        }
