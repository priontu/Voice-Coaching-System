"""Pre-compute and cache GTSinger mel spectrograms to local SSD.

Reads each GTSinger WAV from NFS (slow, 48 kHz), resamples to 16 kHz,
computes the 40-band log-mel spectrogram, extracts F0 + VAD from the JSON
annotation, and writes a compact .npz file per clip to a local cache directory.

After caching, finetune.py loads from the cache instead of NFS — data loading
drops from ~3500 ms/batch to ~50 ms/batch (a ~70× speedup).

Usage (run once before training):
    python training/cache_dataset.py \\
        --gtsinger-root "/mnt/researchfiles/ECE IMAPLE/.../GTSinger/English" \\
        --cache-dir     "/mnt/active_storage/Priontu/nanopitch_mel_cache"

The cache directory is safe to delete and rebuild at any time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as TAF
import torchaudio.transforms as TAT
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from gtsinger_dataset import (
    discover_clips, build_f0_track,
    SAMPLE_RATE, HOP_LENGTH, N_MELS, F_MIN, F_MAX, N_FFT, WIN_LENGTH,
    _GTSINGER_SR,
)


def _clip_cache_path(cache_dir: Path, clip_audio_path: Path) -> Path:
    """Derive a deterministic cache filename from the clip's absolute path."""
    key = hashlib.md5(str(clip_audio_path.resolve()).encode()).hexdigest()[:12]
    return cache_dir / f"{key}.npz"


def _mel_transform() -> TAT.MelSpectrogram:
    return TAT.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        f_min=F_MIN, f_max=F_MAX, power=1.0, center=True,
    )


def cache_clip(clip, cache_dir: Path, mel_tf: TAT.MelSpectrogram,
               overwrite: bool = False) -> bool:
    """Process one clip and write its .npz to cache_dir. Returns True on success."""
    dst = _clip_cache_path(cache_dir, clip.audio_path)
    if dst.exists() and not overwrite:
        return True

    try:
        # Load full audio from NFS
        audio_np, sr = sf.read(str(clip.audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio_np.mean(axis=1))
        if sr != SAMPLE_RATE:
            waveform = TAF.resample(waveform, sr, SAMPLE_RATE)

        # Mel: (T, N_MELS)
        mel = mel_tf(waveform.unsqueeze(0))         # (1, N_MELS, T)
        mel = torch.log(mel.squeeze(0).T + 1e-7)    # (T, N_MELS)
        T   = mel.shape[0]
        mel_np = mel.numpy().astype(np.float32)

        # F0 + VAD from JSON
        with open(clip.json_path) as fh:
            notes_json = json.load(fh)
        f0_np, vad_np = build_f0_track(notes_json, T + 20)
        f0_np  = f0_np[:T].astype(np.float32)
        vad_np = vad_np[:T].astype(np.float32)

        np.savez_compressed(str(dst), mel=mel_np, f0=f0_np, vad=vad_np)
        return True

    except Exception as e:
        print(f"  WARN skip {clip.audio_path.name}: {e}")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Cache GTSinger mel spectrograms")
    p.add_argument("--gtsinger-root", required=True)
    p.add_argument("--cache-dir",     required=True,
                   help="Local SSD directory to write .npz files")
    p.add_argument("--overwrite",     action="store_true",
                   help="Re-compute even if .npz already exists")
    p.add_argument("--num-workers",   type=int, default=8,
                   help="Parallel workers (I/O bound — more workers = faster)")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering GTSinger clips in: {args.gtsinger_root}")
    clips = discover_clips(args.gtsinger_root)
    print(f"  Found {len(clips)} clips")

    already = sum(1 for c in clips
                  if _clip_cache_path(cache_dir, c.audio_path).exists())
    if already and not args.overwrite:
        print(f"  {already}/{len(clips)} already cached — "
              f"processing remaining {len(clips) - already}")

    # Each worker needs its own MelSpectrogram instance
    def _worker(clip):
        return cache_clip(clip, cache_dir, _mel_transform(), args.overwrite)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_worker, c): c for c in clips}
        pbar = tqdm(as_completed(futures), total=len(clips),
                    desc="Caching", unit="clip")
        for fut in pbar:
            if fut.result():
                ok += 1
            else:
                fail += 1
            pbar.set_postfix(ok=ok, fail=fail)

    print(f"\nDone. {ok} cached, {fail} failed.")
    print(f"Cache directory: {cache_dir}")
    size_gb = sum(f.stat().st_size for f in cache_dir.glob("*.npz")) / 1e9
    print(f"Cache size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
