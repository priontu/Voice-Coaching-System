"""NanoPitch GTSinger test: baseline eval → fine-tune → fine-tuned eval → comparison.

Stages
------
1. Smoke test  — verify imports, dataset scan, and a single forward pass
2. Baseline    — evaluate Priontu_Chowdhury/weights.pth on GTSinger/English
3. Fine-tune   — run finetune.py on GTSinger/English
4. Fine-tuned  — evaluate the best checkpoint produced by fine-tuning
5. Summary     — side-by-side metric comparison

Usage (from the NanoPitch_v2 root)
------------------------------------
    python test/run_test.py \\
        --gtsinger-root "/mnt/researchfiles/ECE IMAPLE/cluster_data/archive/GTSinger/English"

Quick smoke-test only (no training):
    python test/run_test.py --gtsinger-root <path> --smoke-test

Skip training if best.pth already exists:
    python test/run_test.py --gtsinger-root <path> --skip-training
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
# File is at: NanoPitch_v2/test/run_test.py
# parents: [0]=test/ [1]=NanoPitch_v2/
NANOPITCH_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR  = NANOPITCH_DIR / "training"
FINETUNE_SCRIPT = TRAINING_DIR / "finetune.py"
DEFAULT_WEIGHTS = NANOPITCH_DIR / "submissions" / "Priontu_Chowdhury" / "weights.pth"
DEFAULT_OUT     = Path(__file__).resolve().parent / "outputs"
PYTHON          = sys.executable

sys.path.insert(0, str(TRAINING_DIR))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NanoPitch GTSinger fine-tuning test")
    p.add_argument("--gtsinger-root", required=True,
                   help="Path to GTSinger/English directory")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help="Pretrained weights to use as baseline and FT starting point")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run stage 1 only (fast import + forward-pass check)")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip fine-tuning if best.pth already exists in output-dir")
    p.add_argument("--max-eval-clips", type=int, default=None,
                   help="Limit evaluation to N clips (useful for quick checks)")

    # Fine-tuning hyper-parameters forwarded to finetune.py
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--warmup-epochs", type=int,   default=5)
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
    p.add_argument("--patience",      type=int,   default=10,
                   help="Early-stopping patience in val checks (0 = disabled)")
    p.add_argument("--min-delta",     type=float, default=1e-4,
                   help="Minimum RPA improvement to reset early-stopping counter")
    p.add_argument("--device",        default="auto")
    p.add_argument("--freeze-conv",   action="store_true",  default=True)
    p.add_argument("--no-freeze-conv",dest="freeze_conv",   action="store_false")

    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}\n")


def _load_model(weights_path: Path, device: torch.device):
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
    """Evaluate on full-length GTSinger clips (no windowing).

    Each clip is fed to the model in one shot — the GRU architecture handles
    variable-length sequences natively, so this gives cleaner metrics than
    windowed evaluation.
    """
    from model import viterbi_decode
    from gtsinger_dataset import build_f0_track, SAMPLE_RATE, HOP_LENGTH

    mel_transform = _make_mel_transform()
    _GTSINGER_SR  = 48_000

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

            # Mel spectrogram: (T, N_MELS)
            mel = mel_transform(waveform.unsqueeze(0))   # (1, N_MELS, T)
            mel = torch.log(mel.squeeze(0).T + 1e-7)     # (T, N_MELS)
            T   = mel.shape[0]

            # Inference
            pred_vad, pred_pitch, _ = model(mel.unsqueeze(0).to(device))
            pred_vad_np   = pred_vad.squeeze().cpu().numpy()    # (T,)
            pred_pitch_np = pred_pitch.squeeze(0).cpu().numpy() # (T, 360)

            # Reference from JSON
            with open(clip.json_path) as fh:
                notes_json = json.load(fh)
            f0_ref, vad_ref = build_f0_track(notes_json, T + 10)
            f0_ref  = f0_ref[:T]
            vad_ref = vad_ref[:T]

            # Viterbi decode
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

def stage_smoke(gtsinger_root: str, weights: Path, device: torch.device) -> None:
    from model import NanoPitch
    from gtsinger_dataset import discover_clips, GTSingerNanoPitchDataset

    print("  Checking imports … OK")

    # Dataset scan
    clips = discover_clips(gtsinger_root)
    assert len(clips) > 0, "No GTSinger clips found — check --gtsinger-root"
    print(f"  GTSinger clips found: {len(clips)}")

    # Minimal dataset item
    ds = GTSingerNanoPitchDataset(clips[:5], seq_len=50, stride_frames=50)
    mel, vad, f0 = ds[0]
    assert mel.shape == (50, 40), f"Unexpected mel shape: {mel.shape}"
    print(f"  Dataset item: mel={tuple(mel.shape)}, vad={tuple(vad.shape)}")

    # Forward pass
    model = _load_model(weights, device)
    pred_vad, pred_pitch, _ = model(mel.unsqueeze(0).to(device))
    assert pred_vad.shape   == (1, 50, 1),   f"vad shape:   {pred_vad.shape}"
    assert pred_pitch.shape == (1, 50, 360), f"pitch shape: {pred_pitch.shape}"
    print(f"  Forward pass OK: vad={tuple(pred_vad.shape)}, "
          f"pitch={tuple(pred_pitch.shape)}")
    print("\n  Smoke test passed.")


# ── Stage 2: Baseline evaluation ──────────────────────────────────────────────

def stage_baseline(gtsinger_root: str, weights: Path, device: torch.device,
                   out_dir: Path, max_clips: int | None) -> dict:
    from gtsinger_dataset import discover_clips

    json_path = out_dir / "baseline.json"
    if json_path.exists():
        print(f"  Cached → {json_path}")
        return json.loads(json_path.read_text())

    clips = discover_clips(gtsinger_root)
    model = _load_model(weights, device)
    results = evaluate_on_gtsinger(model, clips, device, max_clips=max_clips)

    agg = _aggregate(results)
    payload = {"aggregate": agg, "per_clip": results}
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved → {json_path}")
    return payload


# ── Stage 3: Fine-tuning ──────────────────────────────────────────────────────

def stage_finetune(args: argparse.Namespace, train_out: Path) -> Path:
    """Run finetune.py as a subprocess; returns path to best.pth."""
    cmd = [
        PYTHON, str(FINETUNE_SCRIPT),
        "--gtsinger-root",  args.gtsinger_root,
        "--weights",        str(args.weights),
        "--output-dir",     str(train_out),
        "--epochs",         str(args.epochs),
        "--lr",             str(args.lr),
        "--batch-size",     str(args.batch_size),
        "--warmup-epochs",  str(args.warmup_epochs),
        "--val-split",      str(args.val_split),
        "--stride-frames",  str(args.stride_frames),
        "--num-workers",    str(args.num_workers),
        "--device",         args.device,
    ]
    if args.cache_dir:
        cmd += ["--cache-dir", args.cache_dir]
    if args.preload_audio:
        cmd.append("--preload-audio")
    if args.gpu_mel:
        cmd.append("--gpu-mel")
    if args.freeze_conv:
        cmd.append("--freeze-conv")
    else:
        cmd.append("--no-freeze-conv")
    cmd += ["--patience",  str(args.patience),
            "--min-delta", str(args.min_delta)]

    try:
        result = subprocess.run(cmd, text=True)
    except KeyboardInterrupt:
        print("\n  Fine-tuning interrupted — looking for saved checkpoint …")
        result = None

    best_pth = train_out / "checkpoints" / "best.pth"
    last_pth = train_out / "checkpoints" / "last.pth"

    if best_pth.exists():
        return best_pth
    if last_pth.exists():
        print(f"  best.pth not found — using last.pth")
        return last_pth

    if result is not None and result.returncode != 0:
        print(f"[ERROR] Fine-tuning failed (exit {result.returncode})",
              file=sys.stderr)
        sys.exit(result.returncode)

    print("[ERROR] No checkpoint produced by fine-tuning.", file=sys.stderr)
    sys.exit(1)


# ── Stage 4: Fine-tuned evaluation ────────────────────────────────────────────

def stage_finetuned(gtsinger_root: str, best_pth: Path, device: torch.device,
                    out_dir: Path, max_clips: int | None) -> dict:
    from gtsinger_dataset import discover_clips

    json_path = out_dir / "finetuned.json"
    clips = discover_clips(gtsinger_root)
    model = _load_model(best_pth, device)
    results = evaluate_on_gtsinger(model, clips, device, max_clips=max_clips)

    agg = _aggregate(results)
    payload = {"aggregate": agg, "per_clip": results}
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved → {json_path}")
    return payload


# ── Stage 5: Comparison table ─────────────────────────────────────────────────

def print_comparison(baseline: dict, finetuned: dict) -> None:
    b = baseline["aggregate"]
    f = finetuned["aggregate"]

    def _fmt(v):
        return f"{v:.4f}" if not np.isnan(v) else "  nan "

    def _delta(bv, fv):
        if np.isnan(bv) or np.isnan(fv):
            return "    —"
        d = fv - bv
        return f"{d:+.4f}"

    metrics = [
        ("RPA@50¢",      "rpa"),
        ("VDR",          "vdr"),
        ("VAD Acc",      "vad_acc"),
        ("Median ¢",     "median_cents"),
    ]

    hdr = f"  {'Metric':<14}  {'Baseline':>10}  {'Fine-tuned':>10}  {'Δ':>9}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for label, key in metrics:
        bv = b.get(key, float("nan"))
        fv = f.get(key, float("nan"))
        print(f"  {label:<14}  {_fmt(bv):>10}  {_fmt(fv):>10}  {_delta(bv, fv):>9}")

    print()
    b_clips = len(baseline.get("per_clip", []))
    f_clips = len(finetuned.get("per_clip", []))
    print(f"  Baseline evaluated on {b_clips} clips.")
    print(f"  Fine-tuned evaluated on {f_clips} clips.")

    # Per-technique breakdown
    by_tech_b: dict[str, list] = {}
    by_tech_f: dict[str, list] = {}
    for r in baseline.get("per_clip", []):
        by_tech_b.setdefault(r["technique"], []).append(r)
    for r in finetuned.get("per_clip", []):
        by_tech_f.setdefault(r["technique"], []).append(r)

    all_techs = sorted(set(by_tech_b) | set(by_tech_f))
    if all_techs:
        print(f"\n  {'Technique':<30}  {'Base RPA':>9}  {'FT RPA':>9}  {'Δ':>7}")
        print("  " + "─" * 62)
        for tech in all_techs:
            def _rpa(clips):
                vals = [r["rpa"] for r in clips if not np.isnan(r.get("rpa", float("nan")))]
                return float(np.mean(vals)) if vals else float("nan")
            br = _rpa(by_tech_b.get(tech, []))
            fr = _rpa(by_tech_f.get(tech, []))
            d  = f"{fr - br:+.4f}" if not (np.isnan(br) or np.isnan(fr)) else "    —"
            print(f"  {tech:<30}  {_fmt(br):>9}  {_fmt(fr):>9}  {d:>7}")


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
    stage_smoke(args.gtsinger_root, args.weights, device)
    if args.smoke_test:
        return

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    _header("STAGE 2  —  BASELINE EVALUATION")
    baseline = stage_baseline(args.gtsinger_root, args.weights, device,
                               out_dir, args.max_eval_clips)
    b = baseline["aggregate"]
    print(f"\n  Baseline  RPA={b.get('rpa', float('nan')):.4f}  "
          f"VDR={b.get('vdr', float('nan')):.4f}  "
          f"VAD={b.get('vad_acc', float('nan')):.4f}  "
          f"Med={b.get('median_cents', float('nan')):.1f}¢")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    _header("STAGE 3  —  FINE-TUNING")
    best_pth = train_out / "checkpoints" / "best.pth"
    if args.skip_training and best_pth.exists():
        print(f"  --skip-training: using existing {best_pth}")
    else:
        try:
            best_pth = stage_finetune(args, train_out)
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

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    _header("STAGE 4  —  FINE-TUNED EVALUATION")
    finetuned = stage_finetuned(args.gtsinger_root, best_pth, device,
                                 out_dir, args.max_eval_clips)
    f = finetuned["aggregate"]
    print(f"\n  Fine-tuned  RPA={f.get('rpa', float('nan')):.4f}  "
          f"VDR={f.get('vdr', float('nan')):.4f}  "
          f"VAD={f.get('vad_acc', float('nan')):.4f}  "
          f"Med={f.get('median_cents', float('nan')):.1f}¢")

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    _header("STAGE 5  —  COMPARISON")
    print_comparison(baseline, finetuned)

    print(f"\n  Output directory : {out_dir}")
    print(f"  Best checkpoint  : {best_pth}")
    print(f"  Baseline JSON    : {out_dir / 'baseline.json'}")
    print(f"  Fine-tuned JSON  : {out_dir / 'finetuned.json'}")
    print()


if __name__ == "__main__":
    main()
