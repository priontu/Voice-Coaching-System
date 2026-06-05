"""GTSinger dataset for NanoPitch fine-tuning.

Returns (mel, vad, f0_hz) tuples matching the NanoPitch train loop's data
contract.  Log-mel spectrograms are extracted on-the-fly from GTSinger 48 kHz
WAV files — resampled to 16 kHz and converted to the 40-band format that
NanoPitch expects.

Frame rate: hop_length=160 at 16 kHz → 100 fps — an exact match to GTSinger's
JSON note annotations, so no frame-rate conversion is needed.
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
import torchaudio.transforms as TAT
from torch.utils.data import Dataset

# ── Audio / mel constants — must match the NanoPitch pre-training data ────────
SAMPLE_RATE = 16_000
HOP_LENGTH  = 160       # 10 ms per frame at 16 kHz  (100 fps)
WIN_LENGTH  = 400       # 25 ms STFT window
N_FFT       = 512
N_MELS      = 40
F_MIN       = 27.5      # Hz — A0 (standard librosa default)
F_MAX       = 8_000.0   # Hz — Nyquist for 16 kHz

_GTSINGER_SR   = 48_000              # all GTSinger files are recorded at 48 kHz
EXCLUDE_GROUPS = {"Paired_Speech_Group"}


# ── Clip metadata ─────────────────────────────────────────────────────────────

class GTSingerClip(NamedTuple):
    audio_path: Path
    json_path:  Path


# ── Clip discovery ─────────────────────────────────────────────────────────────

def _scan_group_dir(group_dir: Path) -> list[GTSingerClip]:
    json_stems = {p.stem for p in group_dir.glob("*.json")}
    return [
        GTSingerClip(wav, wav.with_suffix(".json"))
        for wav in sorted(group_dir.glob("*.wav"))
        if wav.stem in json_stems
    ]


def discover_clips(root: str | Path) -> list[GTSingerClip]:
    """Return all (wav, json) pairs under GTSinger/English, skipping speech groups.

    Uses a thread pool to scan leaf directories in parallel — cuts NFS latency
    from ~20 s to ~4 s on a typical cluster filesystem.
    """
    root = Path(root)
    group_dirs: list[Path] = []
    for voice_dir in root.iterdir():
        if not voice_dir.is_dir():
            continue
        for tech_dir in voice_dir.iterdir():
            if not tech_dir.is_dir():
                continue
            for song_dir in tech_dir.iterdir():
                if not song_dir.is_dir():
                    continue
                for group_dir in song_dir.iterdir():
                    if group_dir.is_dir() and group_dir.name not in EXCLUDE_GROUPS:
                        group_dirs.append(group_dir)

    clips: list[GTSingerClip] = []
    with ThreadPoolExecutor(max_workers=min(32, max(1, len(group_dirs)))) as pool:
        for result in pool.map(_scan_group_dir, group_dirs):
            clips.extend(result)
    return sorted(clips)


# ── Ground-truth F0 from GTSinger JSON ────────────────────────────────────────

def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def build_f0_track(notes_json: list[dict],
                   n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Build frame-level (f0_hz, vad) arrays at 100 fps from a GTSinger JSON.

    Notes with MIDI pitch == 0 (silence tokens) are treated as unvoiced.
    Returns:
        f0_hz : (n_frames,) float32 — Hz, 0 for unvoiced frames
        vad   : (n_frames,) float32 — 1=voiced, 0=unvoiced
    """
    f0  = np.zeros(n_frames, dtype=np.float32)
    vad = np.zeros(n_frames, dtype=np.float32)
    for word in notes_json:
        for midi, ns, ne in zip(
            word.get("note", []),
            word.get("note_start", []),
            word.get("note_end", []),
        ):
            if not midi:
                continue
            s = max(0, int(round(float(ns) * 100)))
            e = min(n_frames, int(round(float(ne) * 100)))
            if s >= e:
                continue
            f0[s:e]  = _midi_to_hz(midi)
            vad[s:e] = 1.0
    return f0, vad


def _n_frames_from_json(json_path: Path) -> int:
    """Estimate clip length in 100-fps frames from JSON note timings.

    Reads the JSON file (small, already needed for labels) instead of the audio
    header — avoids a second NFS round-trip per clip on startup.
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
        return 750   # dataset median fallback


# ── Dataset ────────────────────────────────────────────────────────────────────

class GTSingerNanoPitchDataset(Dataset):
    """GTSinger dataset yielding NanoPitch-format tuples.

    Segments GTSinger clips into fixed-length windows with a configurable
    stride.  Two modes are supported via ``return_audio``:

    ``return_audio=False`` (default):
        mel   : (seq_len, 40)  float32 — log-mel spectrogram (computed on CPU)
        vad   : (seq_len,)     float32 — 1=voiced, 0=unvoiced
        f0_hz : (seq_len,)     float32 — F0 in Hz (0=unvoiced)

    ``return_audio=True`` (for GPU mel pipeline):
        audio : (n_samples,)   float32 — 16 kHz mono waveform (no mel yet)
        vad   : (seq_len,)     float32
        f0_hz : (seq_len,)     float32
        The caller is responsible for computing mel on the GPU.

    All JSON files are pre-loaded into RAM at init (≈30 MB for GTSinger English)
    so that ``__getitem__`` only ever touches one NFS file — the WAV slice.
    """

    def __init__(self, clips: list[GTSingerClip], seq_len: int = 200,
                 stride_frames: int = 150, return_audio: bool = False,
                 preload_audio: bool = False):
        """
        Args:
            clips         : list of GTSingerClip (audio_path, json_path) pairs
            seq_len       : output length in frames (200 = 2 s at 100 fps)
            stride_frames : hop between consecutive segments
            return_audio  : if True, return raw 16 kHz waveform instead of mel
                            so the caller can compute mel on the GPU
            preload_audio : if True, read ALL WAV files into RAM at init (~2.8 GB
                            for 4827 GTSinger clips).  Eliminates 150 ms cold NFS
                            reads per clip during training — makes data loading
                            ~200× faster at the cost of a one-time ~20 s startup.
        """
        self.clips         = clips
        self.seq_len       = seq_len
        self.stride_frames = stride_frames
        self.return_audio  = return_audio or preload_audio  # preloaded → raw audio
        self.preload_audio = preload_audio

        # n_samples needed at 16 kHz to get exactly seq_len mel frames (center=True)
        self._audio_samples_16k = (seq_len - 1) * HOP_LENGTH

        # Mel transform — stored on instance so each DataLoader worker process
        # gets its own copy via fork/pickle (only used when return_audio=False).
        self._mel = TAT.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH, n_mels=N_MELS,
            f_min=F_MIN, f_max=F_MAX, power=1.0, center=True,
        )

        # ── Pre-load JSON annotations (F0 + VAD) ─────────────────────────────
        # Eliminates one NFS read per item (≈10 ms). Memory: ~30 MB.
        print(f"  Pre-loading {len(clips)} JSON annotations …", flush=True)

        def _load_json(ci: int) -> tuple[int, np.ndarray, np.ndarray]:
            clip = clips[ci]
            n    = _n_frames_from_json(clip.json_path)
            try:
                with open(clip.json_path) as fh:
                    notes = json.load(fh)
                f0, vad = build_f0_track(notes, n)
            except Exception:
                f0  = np.zeros(n, dtype=np.float32)
                vad = np.zeros(n, dtype=np.float32)
            return ci, f0.astype(np.float32), vad.astype(np.float32)

        n_io = min(32, max(1, len(clips)))
        cache_f0  = [None] * len(clips)
        cache_vad = [None] * len(clips)
        with ThreadPoolExecutor(max_workers=n_io) as pool:
            for ci, f0, vad in pool.map(_load_json, range(len(clips))):
                cache_f0[ci]  = f0
                cache_vad[ci] = vad
        self._f0_cache  = cache_f0
        self._vad_cache = cache_vad
        print(f"  JSON pre-load done.", flush=True)

        # ── Pre-load audio into RAM (optional) ───────────────────────────────
        # Eliminates cold NFS WAV reads (≈150 ms each) during training.
        # Memory: ~2.8 GB for all 4827 GTSinger English clips at 16 kHz.
        # One-time startup cost: ~20 s with 32 parallel workers.
        self._audio_cache: list[np.ndarray] | None = None
        if preload_audio:
            ram_gb = len(clips) * self._audio_samples_16k * 4 / 1e9
            # Actual clips are longer than seq_len; estimate full-clip size
            full_gb = ram_gb * 45   # typical clip is ~45× seq_len
            print(f"  Pre-loading {len(clips)} audio clips into RAM "
                  f"(≈{full_gb:.1f} GB) …", flush=True)

            def _load_audio(ci: int) -> tuple[int, np.ndarray]:
                clip = clips[ci]
                try:
                    audio_np, sr = sf.read(str(clip.audio_path),
                                           dtype="float32", always_2d=True)
                    waveform = torch.from_numpy(audio_np.mean(axis=1))
                    if sr != SAMPLE_RATE:
                        waveform = TAF.resample(waveform, sr, SAMPLE_RATE)
                    return ci, waveform.numpy()
                except Exception:
                    return ci, np.zeros(SAMPLE_RATE, dtype=np.float32)

            audio_cache = [None] * len(clips)
            with ThreadPoolExecutor(max_workers=n_io) as pool:
                for ci, wav in pool.map(_load_audio, range(len(clips))):
                    audio_cache[ci] = wav
            self._audio_cache = audio_cache
            actual_gb = sum(a.nbytes for a in audio_cache) / 1e9
            print(f"  Audio pre-load done ({actual_gb:.1f} GB in RAM).",
                  flush=True)

        self._index: list[tuple[int, int]] = []
        self._build_index()

    # ── Index building ─────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        def _segs(ci: int) -> list[tuple[int, int]]:
            n = len(self._f0_cache[ci])   # already loaded — no extra I/O
            segs, start = [], 0
            while start + self.seq_len <= n:
                segs.append((ci, start))
                start += self.stride_frames
            if not segs:
                segs.append((ci, 0))
            elif n >= self.seq_len and (n - self.seq_len) > segs[-1][1]:
                segs.append((ci, n - self.seq_len))
            return segs

        for i in range(len(self.clips)):
            self._index.extend(_segs(i))

    def __len__(self) -> int:
        return len(self._index)

    # ── Item loading ───────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_idx, seg_start = self._index[idx]
        clip    = self.clips[clip_idx]
        seg_end = seg_start + self.seq_len

        # ── Audio ─────────────────────────────────────────────────────────────
        if self._audio_cache is not None:
            # Fast path: slice from in-memory 16 kHz array — zero NFS I/O
            full_wav = self._audio_cache[clip_idx]
            start_s  = int(seg_start / 100 * SAMPLE_RATE)
            end_s    = start_s + self._audio_samples_16k
            wav_seg  = full_wav[start_s:end_s]
            if len(wav_seg) < self._audio_samples_16k:
                wav_seg = np.pad(wav_seg,
                                 (0, self._audio_samples_16k - len(wav_seg)))
            waveform = torch.from_numpy(wav_seg)
        else:
            # Slow path: NFS read (cold ≈150 ms; warm ≈4 ms from OS cache)
            native_start = int(seg_start / 100 * _GTSINGER_SR)
            native_n     = self._audio_samples_16k * (_GTSINGER_SR // SAMPLE_RATE)
            audio_np, sr = sf.read(
                str(clip.audio_path),
                start=native_start,
                stop=native_start + native_n,
                dtype="float32",
                always_2d=True,
            )
            waveform = torch.from_numpy(audio_np.mean(axis=1))
            if sr != SAMPLE_RATE:
                waveform = TAF.resample(waveform, sr, SAMPLE_RATE)
            if waveform.shape[0] < self._audio_samples_16k:
                waveform = F.pad(waveform,
                                 (0, self._audio_samples_16k - waveform.shape[0]))

        # ── Ground truth: slice the in-memory cache — no NFS read ────────────
        f0_full  = self._f0_cache[clip_idx]
        vad_full = self._vad_cache[clip_idx]
        f0_seg   = f0_full[seg_start:seg_end]
        vad_seg  = vad_full[seg_start:seg_end]
        if len(f0_seg) < self.seq_len:
            pad    = self.seq_len - len(f0_seg)
            f0_seg  = np.pad(f0_seg,  (0, pad))
            vad_seg = np.pad(vad_seg, (0, pad))

        if self.return_audio:
            # Caller computes mel on GPU — return raw 16 kHz waveform
            return (
                waveform.float(),
                torch.from_numpy(vad_seg),
                torch.from_numpy(f0_seg),
            )

        # ── Mel on CPU (default) ──────────────────────────────────────────────
        mel = self._mel(waveform.unsqueeze(0))       # (1, N_MELS, T)
        mel = torch.log(mel.squeeze(0).T + 1e-7)     # (T, N_MELS)
        T   = mel.shape[0]
        if T > self.seq_len:
            mel = mel[:self.seq_len]
        elif T < self.seq_len:
            mel = F.pad(mel, (0, 0, 0, self.seq_len - T))

        return (
            mel.float(),
            torch.from_numpy(vad_seg),
            torch.from_numpy(f0_seg),
        )


# ── Cached dataset (fast path) ────────────────────────────────────────────────

def _clip_cache_path(cache_dir: Path, clip_audio_path: Path) -> Path:
    """Mirror of the same function in cache_dataset.py — must stay in sync."""
    import hashlib
    key = hashlib.md5(str(clip_audio_path.resolve()).encode()).hexdigest()[:12]
    return cache_dir / f"{key}.npz"


class CachedGTSingerDataset(Dataset):
    """Fast dataset that loads pre-computed mel+f0+vad from local SSD cache.

    Use cache_dataset.py to build the cache first.  Loading from local SSD
    is ~70× faster than computing mel on-the-fly from NFS WAV files.

    Interface identical to GTSingerNanoPitchDataset — drop-in replacement.
    """

    def __init__(self, clips: list[GTSingerClip], cache_dir: str | Path,
                 seq_len: int = 200, stride_frames: int = 150):
        self.cache_dir     = Path(cache_dir)
        self.seq_len       = seq_len
        self.stride_frames = stride_frames

        # Only keep clips that have a cache file
        self.clips = [c for c in clips
                      if _clip_cache_path(self.cache_dir, c.audio_path).exists()]
        missing = len(clips) - len(self.clips)
        if missing:
            print(f"  CachedDataset: {missing}/{len(clips)} clips not in cache "
                  f"— run cache_dataset.py to build the full cache")

        self._index: list[tuple[int, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        def _segs(ci: int) -> list[tuple[int, int]]:
            npz = np.load(str(_clip_cache_path(self.cache_dir,
                                               self.clips[ci].audio_path)),
                          mmap_mode="r")
            n = npz["mel"].shape[0]
            segs, start = [], 0
            while start + self.seq_len <= n:
                segs.append((ci, start))
                start += self.stride_frames
            if not segs:
                segs.append((ci, 0))
            elif n >= self.seq_len and (n - self.seq_len) > segs[-1][1]:
                segs.append((ci, n - self.seq_len))
            return segs

        n_workers = min(32, max(1, len(self.clips)))
        results: dict[int, list] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_segs, i): i for i in range(len(self.clips))}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        for i in range(len(self.clips)):
            self._index.extend(results[i])

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_idx, seg_start = self._index[idx]
        path = _clip_cache_path(self.cache_dir, self.clips[clip_idx].audio_path)

        npz = np.load(str(path), mmap_mode="r")
        T_clip = npz["mel"].shape[0]
        seg_end = min(seg_start + self.seq_len, T_clip)

        mel_seg = npz["mel"][seg_start:seg_end].copy().astype(np.float32)
        f0_seg  = npz["f0"] [seg_start:seg_end].copy().astype(np.float32)
        vad_seg = npz["vad"][seg_start:seg_end].copy().astype(np.float32)

        # Pad to seq_len if the clip ends early
        pad = self.seq_len - mel_seg.shape[0]
        if pad > 0:
            mel_seg = np.pad(mel_seg, ((0, pad), (0, 0)))
            f0_seg  = np.pad(f0_seg,  (0, pad))
            vad_seg = np.pad(vad_seg, (0, pad))

        return (
            torch.from_numpy(mel_seg),
            torch.from_numpy(vad_seg),
            torch.from_numpy(f0_seg),
        )
