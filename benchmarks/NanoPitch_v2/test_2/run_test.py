"""NanoPitch GTSinger test: train from scratch → evaluate.

Stages
------
1. Smoke test  — verify imports, dataset scan, and a single forward pass
2. Train       — run train_from_scratch.py on GTSinger/English
3. Evaluate    — evaluate the best checkpoint produced by from-scratch training

Usage (from the NanoPitch_v2 root)
------------------------------------
    python test_2/run_test.py \\
        --gtsinger-root "/mnt/researchfiles/ECE IMAPLE/cluster_data/archive/GTSinger/English"

Quick smoke-test only (no training):
    python test_2/run_test.py --gtsinger-root <path> --smoke-test

Skip training if best.pth already exists:
    python test_2/run_test.py --gtsinger-root <path> --skip-training
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as TAF
import torchaudio.transforms as TAT
import soundfile as sf
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
# File is at: NanoPitch_v2/test_2/run_test.py
# parents: [0]=test_2/ [1]=NanoPitch_v2/
NANOPITCH_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR  = NANOPITCH_DIR / "training"
TRAIN_SCRIPT  = TRAINING_DIR / "train_from_scratch.py"
DEFAULT_OUT   = Path(__file__).resolve().parent / "outputs"
PYTHON        = sys.executable

sys.path.insert(0, str(TRAINING_DIR))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NanoPitch GTSinger from-scratch training test")
    p.add_argument("--gtsinger-root", required=True,
                   help="Path to GTSinger/English directory")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smoke-test", action="store_true",
                   help="Run stage 1 only (fast import + forward-pass check)")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training if best.pth already exists in output-dir")
    p.add_argument("--max-eval-clips", type=int, default=None,
                   help="Limit evaluation to N clips (useful for quick checks)")

    # Training hyper-parameters forwarded to train_from_scratch.py
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--warmup-epochs", type=int,   default=10)
    p.add_argument("--val-split",     type=float, default=0.1)
    p.add_argument("--stride-frames", type=int,   default=150)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--cache-dir",     default=None,
                   help="Pre-computed mel cache directory (from cache_dataset.py). "
                        "Strongly recommended — ~70× faster data loading.")
    p.add_argument("--preload-audio", action="store_true", default=False,
                   help="Pre-load all WAV files into RAM (~2.8 GB, ~20 s startup). "
                        "Eliminates 150 ms cold NFS reads — use with --gpu-mel.")
    p.add_argument("--gpu-mel", action="store_true", default=False,
                   help="Compute mel on GPU. Pairs with --preload-audio for "
                        "maximum speed without disk caching.")
    p.add_argument("--patience",      type=int,   default=15,
                   help="Early-stopping patience in val checks (0 = disabled)")
    p.add_argument("--min-delta",     type=float, default=1e-4,
                   help="Minimum RPA improvement to reset early-stopping counter")
    p.add_argument("--device",        default="auto")

    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}\n")


def _load_checkpoint(weights_path: Path, device: torch.device):
    from model import NanoPitch
    ckpt   = torch.load(str(weights_path), map_location="cpu")
    kwargs = ckpt.get("model_kwargs", {"cond_size": 64, "gru_size": 96})
    model  = NanoPitch(**kwargs)
    state  = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def _make_mel_transform():
    from gtsinger_dataset import SAMPLE_RATE, HOP_LENGTH, N_MELS
    return TAT.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=512, win_length=400,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        f_min=27.5, f_max=8000.0, power=1.0, center=True,
    )


# ── Per-clip metrics ───────────────────────────────────────────────────────────

def _pitch_metrics(f0_pred: np.ndarray, f0_ref: np.ndarray,
                   pred_vad_raw: np.ndarray, vad_ref: np.ndarray) -> dict:
    """Return VAD accuracy, VDR, RPA@50¢, and median cents error."""
    ref_voiced  = vad_ref > 0
    pred_voiced = f0_pred > 0

    vad_acc = float(np.mean((pred_vad_raw > 0.5) == ref_voiced))
    vdr     = (float(np.mean(pred_voiced[ref_voiced]))
               if ref_voiced.sum() > 0 else float("nan"))

    both = ref_voiced & pred_voiced
    if both.sum() > 0:
        err = np.abs(1200.0 * np.log2(
            f0_pred[both] / (f0_ref[both] + 1e-10) + 1e-10))
        rpa = float(np.mean(err < 50))
        med = float(np.median(err))
    else:
        rpa = med = float("nan")

    return {"vad_acc": vad_acc, "vdr": vdr, "rpa": rpa, "median_cents": med}


# ── GTSinger evaluation (in-process, full clips) ──────────────────────────────

@torch.no_grad()
def evaluate_on_gtsinger(model, clips, device: torch.device,
                          max_clips: int | None = None) -> list[dict]:
    """Evaluate on full-length GTSinger clips (no windowing)."""
    from model import viterbi_decode
    from gtsinger_dataset import build_f0_track, SAMPLE_RATE, HOP_LENGTH

    mel_transform = _make_mel_transform()

    if max_clips is not None:
        clips = clips[:max_clips]

    results: list[dict] = []
    for clip in tqdm(clips, desc="Evaluating", unit="clip"):
        try:
            audio_np, sr = sf.read(str(clip.audio_path), dtype="float32",
                                   always_2d=True)
            waveform = torch.from_numpy(audio_np.mean(axis=1))
            if sr != SAMPLE_RATE:
                waveform = TAF.resample(waveform, sr, SAMPLE_RATE)

            mel = mel_transform(waveform.unsqueeze(0))   # (1, N_MELS, T)
            mel = torch.log(mel.squeeze(0).T + 1e-7)     # (T, N_MELS)
            T   = mel.shape[0]

            pred_vad, pred_pitch, _ = model(mel.unsqueeze(0).to(device))
            pred_vad_np   = pred_vad.squeeze().cpu().numpy()     # (T,)
            pred_pitch_np = pred_pitch.squeeze(0).cpu().numpy()  # (T, 360)

            with open(clip.json_path) as fh:
                notes_json = json.load(fh)
            f0_ref, vad_ref = build_f0_track(notes_json, T + 10)
            f0_ref  = f0_ref[:T]
            vad_ref = vad_ref[:T]

            f0_pred = viterbi_decode(pred_pitch_np)

            m = _pitch_metrics(f0_pred, f0_ref, pred_vad_np, vad_ref)
            m["clip_id"]   = clip.audio_path.stem
            m["technique"] = clip.audio_path.parents[2].name
            results.append(m)

        except Exception as e:
            print(f"  skip {clip.audio_path.name}: {e}")
            continue

    return results


def _aggregate(results: list[dict]) -> dict:
    def _smean(key):
        vals = [r[key] for r in results if not np.isnan(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")
    return {k: _smean(k) for k in ("vad_acc", "vdr", "rpa", "median_cents")}


# ── Stage 1: Smoke test ───────────────────────────────────────────────────────

def stage_smoke(gtsinger_root: str, device: torch.device) -> None:
    from model import NanoPitch
    from gtsinger_dataset import discover_clips, GTSingerNanoPitchDataset

    print("  Checking imports … OK")

    clips = discover_clips(gtsinger_root)
    assert len(clips) > 0, "No GTSinger clips found — check --gtsinger-root"
    print(f"  GTSinger clips found: {len(clips)}")

    ds = GTSingerNanoPitchDataset(clips[:5], seq_len=50, stride_frames=50)
    mel, vad, f0 = ds[0]
    assert mel.shape == (50, 40), f"Unexpected mel shape: {mel.shape}"
    print(f"  Dataset item: mel={tuple(mel.shape)}, vad={tuple(vad.shape)}")

    # Verify a randomly initialised model forward pass (no weights loaded)
    model = NanoPitch(cond_size=64, gru_size=96).to(device)
    model.eval()
    with torch.no_grad():
        pred_vad, pred_pitch, _ = model(mel.unsqueeze(0).to(device))
    assert pred_vad.shape   == (1, 50, 1),   f"vad shape:   {pred_vad.shape}"
    assert pred_pitch.shape == (1, 50, 360), f"pitch shape: {pred_pitch.shape}"
    print(f"  Forward pass OK (random init): vad={tuple(pred_vad.shape)}, "
          f"pitch={tuple(pred_pitch.shape)}")
    print("\n  Smoke test passed.")


# ── Stage 2: Train from scratch ───────────────────────────────────────────────

def stage_train_scratch(args: argparse.Namespace, train_out: Path) -> Path:
    """Run train_from_scratch.py as a subprocess; returns path to best.pth."""
    cmd = [
        PYTHON, str(TRAIN_SCRIPT),
        "--gtsinger-root",  args.gtsinger_root,
        "--output-dir",     str(train_out),
        "--epochs",         str(args.epochs),
        "--lr",             str(args.lr),
        "--batch-size",     str(args.batch_size),
        "--warmup-epochs",  str(args.warmup_epochs),
        "--val-split",      str(args.val_split),
        "--stride-frames",  str(args.stride_frames),
        "--num-workers",    str(args.num_workers),
        "--device",         args.device,
        "--patience",       str(args.patience),
        "--min-delta",      str(args.min_delta),
    ]
    if args.cache_dir:
        cmd += ["--cache-dir", args.cache_dir]
    if args.preload_audio:
        cmd.append("--preload-audio")
    if args.gpu_mel:
        cmd.append("--gpu-mel")

    try:
        result = subprocess.run(cmd, text=True)
    except KeyboardInterrupt:
        print("\n  Training interrupted — looking for saved checkpoint …")
        result = None

    best_pth = train_out / "checkpoints" / "best.pth"
    last_pth = train_out / "checkpoints" / "last.pth"

    if best_pth.exists():
        return best_pth
    if last_pth.exists():
        print(f"  best.pth not found — using last.pth")
        return last_pth

    if result is not None and result.returncode != 0:
        print(f"[ERROR] Training failed (exit {result.returncode})",
              file=sys.stderr)
        sys.exit(result.returncode)

    print("[ERROR] No checkpoint produced by training.", file=sys.stderr)
    sys.exit(1)


# ── Stage 3: Trained model evaluation ─────────────────────────────────────────

def stage_trained(gtsinger_root: str, best_pth: Path, device: torch.device,
                  out_dir: Path, max_clips: int | None) -> dict:
    from gtsinger_dataset import discover_clips

    json_path = out_dir / "trained.json"
    clips = discover_clips(gtsinger_root)
    model = _load_checkpoint(best_pth, device)
    results = evaluate_on_gtsinger(model, clips, device, max_clips=max_clips)

    agg = _aggregate(results)
    payload = {"aggregate": agg, "per_clip": results}
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved → {json_path}")
    return payload


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    out_dir   = args.output_dir
    train_out = out_dir / "train_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    _header("STAGE 1  —  SMOKE TEST")
    stage_smoke(args.gtsinger_root, device)
    if args.smoke_test:
        return

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    _header("STAGE 2  —  TRAINING FROM SCRATCH")
    best_pth = train_out / "checkpoints" / "best.pth"
    if args.skip_training and best_pth.exists():
        print(f"  --skip-training: using existing {best_pth}")
    else:
        try:
            best_pth = stage_train_scratch(args, train_out)
        except KeyboardInterrupt:
            print("\n  Interrupted — attempting evaluation with best saved checkpoint …")
            if not best_pth.exists():
                last_pth = train_out / "checkpoints" / "last.pth"
                if last_pth.exists():
                    best_pth = last_pth
                    print(f"  Using last.pth: {best_pth}")
                else:
                    print("  No checkpoint found — exiting.", file=sys.stderr)
                    sys.exit(0)
    print(f"\n  Best checkpoint: {best_pth}")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    _header("STAGE 3  —  TRAINED MODEL EVALUATION")
    trained = stage_trained(args.gtsinger_root, best_pth, device,
                             out_dir, args.max_eval_clips)
    t = trained["aggregate"]
    print(f"\n  From-scratch  RPA={t.get('rpa', float('nan')):.4f}  "
          f"VDR={t.get('vdr', float('nan')):.4f}  "
          f"VAD={t.get('vad_acc', float('nan')):.4f}  "
          f"Med={t.get('median_cents', float('nan')):.1f}¢")

    print(f"\n  Output directory : {out_dir}")
    print(f"  Best checkpoint  : {best_pth}")
    print(f"  Trained JSON     : {out_dir / 'trained.json'}")
    print()


if __name__ == "__main__":
    main()
