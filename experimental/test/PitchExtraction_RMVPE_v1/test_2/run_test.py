#!/usr/bin/env python
"""End-to-end test: baseline eval → RMVPE fine-tune → fine-tuned eval → plots.

Stages
------
1. Smoke test   — verify imports, dataset, and forward pass before anything heavy
2. Baseline     — evaluate original rmvpe.pt on GTSinger/English
3. Fine-tune    — run train.py with early stopping on the full English dataset
4. Fine-tuned   — evaluate best.pth on the same dataset
5. Plots        — training curves, metric comparison bars, F0 contour overlays

Usage (from repo root)
----------------------
    conda run -n torch_it python \\
        Workspace_Priontu/test/PitchExtraction_RMVPE_v1/test_2/run_test.py \\
        --gtsinger-root "/mnt/researchfiles/ECE IMAPLE/cluster_data/archive/GTSinger/English"

Smoke-test only (fast, no training):
    ... run_test.py --gtsinger-root <path> --smoke-test

Skip training if best.pth already exists:
    ... run_test.py --gtsinger-root <path> --skip-training
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless backend — no display needed
import matplotlib.pyplot as plt
import numpy as np

# ── Repo paths ─────────────────────────────────────────────────────────────────
# File is at: Workspace_Priontu/test/PitchExtraction_RMVPE_v1/test_2/run_test.py
# parents: [0]=test_2/ [1]=PitchExtraction_RMVPE_v1/ [2]=test/ [3]=Workspace_Priontu/
WORKSPACE_DIR = Path(__file__).resolve().parents[3]
FINETUNE_DIR  = WORKSPACE_DIR / "PitchExtraction_RMVPE_v1" / "finetune"
TRAIN_SCRIPT  = FINETUNE_DIR / "train.py"
EVAL_SCRIPT   = FINETUNE_DIR / "evaluate.py"
DEFAULT_OUT   = Path("/mnt/active_storage/Priontu/rmvpe_finetune_outputs")
PYTHON        = sys.executable

COLORS = {"baseline": "#1E88E5", "finetuned": "#E53935"}
TECHNIQUES = [
    "Breathy", "Glissando", "Mixed_Voice_and_Falsetto", "Pharyngeal", "Vibrato",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _header(title: str) -> None:
    bar = "═" * 62
    print(f"\n{bar}\n  {title}\n{bar}")


def _check(condition: bool, msg: str) -> None:
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status} {msg}")
    if not condition:
        sys.exit(1)


def _run(cmd: list, label: str) -> None:
    """Stream a subprocess and exit on non-zero return code."""
    print(f"\n  $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Smoke test (in-process)
# ══════════════════════════════════════════════════════════════════════════════

def stage_smoke_test(gtsinger_root: str, cache_dir: Path, device: str) -> None:
    """Fast in-process check: imports, dataset, shapes, forward pass, script interfaces."""
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(FINETUNE_DIR))

    from dataset import discover_clips, GTSingerPitchDataset, SEGMENT_SAMPLES
    from rmvpe_src.model import E2E0
    from rmvpe_src.spec import MelSpectrogram
    from rmvpe_src.constants import N_MELS, SAMPLE_RATE, WINDOW_LENGTH, MEL_FMIN, MEL_FMAX

    # Dataset
    clips = discover_clips(gtsinger_root)
    _check(len(clips) > 0, f"GTSinger clips discovered: {len(clips)}")

    ds = GTSingerPitchDataset(clips[:10])
    _check(len(ds) > 0, f"Segments from first 10 clips: {len(ds)}")

    sample = ds[0]
    _check(tuple(sample["audio"].shape) == (SEGMENT_SAMPLES,),
           f"Audio shape: {tuple(sample['audio'].shape)}")
    _check(sample["target"].shape[1] == 360,
           f"Target shape: {tuple(sample['target'].shape)}")
    _check(int(sample["voiced"].sum()) > 0,
           f"Voiced frames in sample: {int(sample['voiced'].sum())}")

    # Model instantiation + weights
    dev = torch.device(
        "cuda" if (device == "auto" and torch.cuda.is_available()) else
        device if device != "auto" else "cpu"
    )
    model   = E2E0(4, 1, (2, 2)).to(dev).eval()
    mel_ext = MelSpectrogram(
        N_MELS, SAMPLE_RATE, WINDOW_LENGTH, 160, None, MEL_FMIN, MEL_FMAX
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    _check(n_params > 0, f"E2E0 instantiated: {n_params:,} parameters")

    weights_path = cache_dir / "rmvpe.pt"
    if weights_path.exists():
        ckpt  = torch.load(str(weights_path), map_location="cpu")
        state = ckpt.get("model", ckpt)
        miss, _ = model.load_state_dict(state, strict=False)
        _check(len(miss) < 100, f"Pretrained weights loaded (missing keys: {len(miss)})")
    else:
        _check(True, "rmvpe.pt not yet downloaded — will download during training")

    # Forward pass
    audio = sample["audio"].unsqueeze(0).to(dev)
    with torch.no_grad():
        mel  = mel_ext(audio, center=True)
        n_f  = mel.shape[-1]
        pad  = 32 * ((n_f - 1) // 32 + 1) - n_f
        pred = model(F.pad(mel, (0, pad), mode="reflect"))[:, :n_f]

    _check(pred.shape[-1] == 360,
           f"Forward pass output: {tuple(pred.shape)}, range=[{pred.min():.3f}, {pred.max():.3f}]")
    _check(0.0 <= float(pred.min()) and float(pred.max()) <= 1.0,
           "Output in valid sigmoid range [0, 1]")

    # Script interface checks
    train_src = TRAIN_SCRIPT.read_text()
    eval_src  = EVAL_SCRIPT.read_text()
    _check("--patience"          in train_src, "train.py exposes --patience (early stopping)")
    _check("--gpus"              in train_src, "train.py exposes --gpus (multi-GPU control)")
    _check("DDPStrategy"         in train_src, "train.py uses DDPStrategy for multi-GPU")
    _check("metrics.csv"         in train_src, "train.py writes metrics.csv")
    _check("LightningModule"     in train_src, "train.py uses PyTorch Lightning")
    _check("LightningDataModule" in train_src, "train.py uses LightningDataModule")
    _check("16-mixed"            in train_src, "train.py supports 16-mixed precision (GPU AMP)")
    _check("per_technique"    in eval_src,  "evaluate.py returns per-technique breakdown")
    _check("per_clip"         in eval_src,  "evaluate.py returns per-clip data (for contours)")

    print("\n  All smoke-test checks passed.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Baseline evaluation
# ══════════════════════════════════════════════════════════════════════════════

def stage_baseline(
    gtsinger_root: str, cache_dir: Path, out_dir: Path,
    max_clips: int | None, device: str,
) -> dict:
    out_json = out_dir / "baseline.json"
    cmd = [
        PYTHON, str(EVAL_SCRIPT),
        "--weights",       str(cache_dir / "rmvpe.pt"),
        "--gtsinger-root", gtsinger_root,
        "--tag",           "baseline",
        "--output",        str(out_json),
        "--device",        device,
    ]
    if max_clips:
        cmd += ["--max-clips", str(max_clips)]
    _run(cmd, "Baseline evaluation")
    return json.loads(out_json.read_text())


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Fine-tuning
# ══════════════════════════════════════════════════════════════════════════════

def stage_train(
    gtsinger_root: str, cache_dir: Path, train_out: Path,
    epochs: int, patience: int, lr: float,
    batch_size: int, num_workers: int, device: str,
    gpus: int = 0, warmup_epochs: int = 5,
    weight_decay: float = 1e-2, compile_model: bool = False,
) -> Path:
    cmd = [
        PYTHON, str(TRAIN_SCRIPT),
        "--gtsinger-root",  gtsinger_root,
        "--output-dir",     str(train_out),
        "--cache-dir",      str(cache_dir),
        "--epochs",         str(epochs),
        "--patience",       str(patience),
        "--lr",             str(lr),
        "--weight-decay",   str(weight_decay),
        "--batch-size",     str(batch_size),
        "--num-workers",    str(num_workers),
        "--precision",      "auto",
        "--device",         device,
        "--gpus",           str(gpus),
        "--warmup-epochs",  str(warmup_epochs),
    ]
    if compile_model:
        cmd.append("--compile")
    _run(cmd, "Fine-tuning")

    # best.pth is written by RMVPEFinetuner.on_validation_epoch_end()
    best = train_out / "best.pth"
    if not best.exists():
        # Fall back to the best Lightning checkpoint
        ckpts = sorted((train_out / "checkpoints").glob("*.ckpt"),
                       key=lambda p: p.stat().st_mtime)
        if not ckpts:
            print("[ERROR] No checkpoint produced.", file=sys.stderr)
            sys.exit(1)
        best = ckpts[-1]
        print(f"  Warning: best.pth not found — using last Lightning ckpt: {best}")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Fine-tuned evaluation
# ══════════════════════════════════════════════════════════════════════════════

def stage_finetuned(
    gtsinger_root: str, best_pth: Path, out_dir: Path,
    max_clips: int | None, device: str,
) -> dict:
    out_json = out_dir / "finetuned.json"
    cmd = [
        PYTHON, str(EVAL_SCRIPT),
        "--weights",       str(best_pth),
        "--gtsinger-root", gtsinger_root,
        "--tag",           "finetuned",
        "--output",        str(out_json),
        "--device",        device,
    ]
    if max_clips:
        cmd += ["--max-clips", str(max_clips)]
    _run(cmd, "Fine-tuned evaluation")
    return json.loads(out_json.read_text())


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Plots
# ══════════════════════════════════════════════════════════════════════════════

# ── 5a: Training curves ───────────────────────────────────────────────────────

def plot_training_curves(train_out: Path, plots_dir: Path) -> None:
    csv_path = train_out / "metrics.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("  [SKIP] metrics.csv is empty")
        return

    epochs     = [int(r["epoch"])        for r in rows]
    tr_loss    = [float(r["train_loss"]) for r in rows]
    val_loss   = [float(r["val_loss"])   for r in rows]
    tr_v_loss  = [float(r["train_loss_voiced"]) for r in rows]
    val_v_loss = [float(r["val_loss_voiced"])   for r in rows]
    lr_vals    = [float(r["lr"])         for r in rows]
    rpa_pts    = [(int(r["epoch"]), float(r["val_rpa50"]))
                  for r in rows if r.get("val_rpa50", "").strip()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("RMVPE Fine-tuning — Training Curves", fontsize=13, fontweight="bold")

    # Total loss
    ax = axes[0]
    ax.plot(epochs, tr_loss,  lw=2, color="#1565C0", label="Train total")
    ax.plot(epochs, val_loss, lw=2, color="#C62828", label="Val total",  linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("Total Loss (voiced + 0.3 × unvoiced)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Voiced loss
    ax = axes[1]
    ax.plot(epochs, tr_v_loss,  lw=2, color="#2E7D32", label="Train voiced")
    ax.plot(epochs, val_v_loss, lw=2, color="#FF8F00", label="Val voiced",  linestyle="--")
    if rpa_pts:
        rpa_e, rpa_v = zip(*rpa_pts)
        ax2 = ax.twinx()
        ax2.plot(rpa_e, rpa_v, "o-", color="#6A1B9A", lw=1.5, ms=5, label="Val RPA@50¢")
        ax2.set_ylabel("RPA @ 50 ¢", color="#6A1B9A")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="y", labelcolor="#6A1B9A")
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Voiced BCE Loss")
    ax.set_title("Voiced Loss + Val RPA@50¢")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)

    # Learning rate schedule
    ax = axes[2]
    ax.plot(epochs, lr_vals, lw=2, color="#00838F")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule (cosine annealing)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = plots_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 5b: Metric comparison bars ────────────────────────────────────────────────

def plot_metric_comparison(baseline: dict, finetuned: dict, plots_dir: Path) -> None:
    """2×2 grid: RPA, MACE, MedianCents, GrossErr — aggregate + per technique."""
    PANELS = [
        ("RPA",         "RPA @ 50 ¢",       True,  (0, 1)),
        ("MACE",        "MACE (cents)",      False, None),
        ("MedianCents", "Median |err| (¢)",  False, None),
        ("GrossErr",    "Gross Error Rate",  False, (0, 1)),
    ]
    techniques = sorted(
        set(baseline.get("per_technique", {}).keys()) |
        set(finetuned.get("per_technique", {}).keys())
    )
    labels = ["[All]"] + techniques
    x = np.arange(len(labels))
    w = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("RMVPE Baseline vs Fine-tuned — Metric Comparison", fontsize=13, fontweight="bold")

    for ax, (key, ylabel, higher_better, ylim) in zip(axes.flat, PANELS):
        b_vals = [baseline.get("aggregate", {}).get(key, np.nan)]
        f_vals = [finetuned.get("aggregate", {}).get(key, np.nan)]
        for tech in techniques:
            b_vals.append(baseline.get("per_technique", {}).get(tech, {}).get(key, np.nan))
            f_vals.append(finetuned.get("per_technique", {}).get(tech, {}).get(key, np.nan))

        bars_b = ax.bar(x - w / 2, b_vals, w, label="Baseline",
                        color=COLORS["baseline"],  alpha=0.85, zorder=2)
        bars_f = ax.bar(x + w / 2, f_vals, w, label="Fine-tuned",
                        color=COLORS["finetuned"], alpha=0.85, zorder=2)

        for bar in list(bars_b) + list(bars_f):
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + max(h * 0.01, 0.003),
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        arrow = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{ylabel}  ({arrow})")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    path = plots_dir / "metric_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 5c: F0 contour overlays ───────────────────────────────────────────────────

def plot_f0_contours(baseline: dict, finetuned: dict, plots_dir: Path) -> None:
    """One subplot per technique — GT, baseline, and fine-tuned F0 overlaid."""
    ft_by_id = {c["clip_id"]: c for c in finetuned.get("per_clip", [])}

    techniques_found = []
    chosen_clips: list[dict] = []          # one base clip per technique
    seen_techs: set[str] = set()
    for bc in baseline.get("per_clip", []):
        tech = bc.get("technique", "Unknown")
        if tech not in seen_techs:
            seen_techs.add(tech)
            techniques_found.append(tech)
            chosen_clips.append(bc)
        if len(chosen_clips) >= len(TECHNIQUES):
            break

    if not chosen_clips:
        print("  [SKIP] No per-clip data in results — skipping contour plots")
        return

    n = len(chosen_clips)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8 * n), squeeze=False)
    fig.suptitle("F0 Contours — Ground Truth vs Baseline vs Fine-tuned",
                 fontsize=13, fontweight="bold")

    for ax, bc in zip(axes[:, 0], chosen_clips):
        clip_id = bc["clip_id"]
        tech    = bc.get("technique", "")
        fc      = ft_by_id.get(clip_id, {})

        times  = np.array(bc["times"])
        ref_f0 = np.array(bc["ref_f0"])
        b_f0   = np.array(bc["pred_f0"])
        f_f0   = np.array(fc.get("pred_f0", np.zeros(len(b_f0))))
        T = min(len(times), len(ref_f0), len(b_f0), len(f_f0))
        times, ref_f0, b_f0, f_f0 = times[:T], ref_f0[:T], b_f0[:T], f_f0[:T]

        def _m(arr):  # mask zeros as NaN for clean line breaks
            a = arr.astype(float).copy(); a[a == 0] = np.nan; return a

        ax.fill_between(times, _m(ref_f0), alpha=0.12, color="#2E7D32", zorder=1)
        ax.plot(times, _m(ref_f0), lw=1.8, color="#2E7D32",
                label="Ground truth (MIDI)", zorder=3)
        ax.plot(times, _m(b_f0),   lw=1.4, color=COLORS["baseline"],
                linestyle="--", label="Baseline RMVPE", alpha=0.9, zorder=2)
        ax.plot(times, _m(f_f0),   lw=1.4, color=COLORS["finetuned"],
                linestyle="-.", label="Fine-tuned RMVPE", alpha=0.9, zorder=2)

        b_rpa = bc.get("RPA", float("nan"))
        f_rpa = fc.get("RPA", float("nan"))
        b_mace = bc.get("MACE", float("nan"))
        f_mace = fc.get("MACE", float("nan"))
        ax.set_title(
            f"{tech} — {clip_id}\n"
            f"Baseline: RPA={b_rpa:.3f}, MACE={b_mace:.1f}¢   "
            f"Fine-tuned: RPA={f_rpa:.3f}, MACE={f_mace:.1f}¢",
            fontsize=9,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("F0 (Hz)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    path = plots_dir / "f0_contours_all_techniques.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

    # Also save per-technique individual figures
    for bc in chosen_clips:
        tech   = bc.get("technique", "unknown")
        fc     = ft_by_id.get(bc["clip_id"], {})
        times  = np.array(bc["times"])
        ref_f0 = np.array(bc["ref_f0"])
        b_f0   = np.array(bc["pred_f0"])
        f_f0   = np.array(fc.get("pred_f0", np.zeros(len(b_f0))))
        T = min(len(times), len(ref_f0), len(b_f0), len(f_f0))
        times, ref_f0, b_f0, f_f0 = times[:T], ref_f0[:T], b_f0[:T], f_f0[:T]

        def _m2(arr):
            a = arr.astype(float).copy(); a[a == 0] = np.nan; return a

        fig2, ax2 = plt.subplots(figsize=(13, 3.5))
        ax2.fill_between(times, _m2(ref_f0), alpha=0.12, color="#2E7D32")
        ax2.plot(times, _m2(ref_f0), lw=2,   color="#2E7D32",        label="Ground truth (MIDI)")
        ax2.plot(times, _m2(b_f0),   lw=1.5, color=COLORS["baseline"],  linestyle="--", label="Baseline RMVPE",   alpha=0.9)
        ax2.plot(times, _m2(f_f0),   lw=1.5, color=COLORS["finetuned"], linestyle="-.", label="Fine-tuned RMVPE", alpha=0.9)
        b_rpa = bc.get("RPA", float("nan")); f_rpa = fc.get("RPA", float("nan"))
        ax2.set_title(f"{tech} | baseline RPA={b_rpa:.3f}  fine-tuned RPA={f_rpa:.3f}", fontsize=10)
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("F0 (Hz)")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)
        fig2.tight_layout()
        safe = tech.replace(" ", "_").replace("/", "_")
        p2 = plots_dir / f"f0_contour_{safe}.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {p2.name}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_comparison_table(baseline: dict, finetuned: dict) -> None:
    keys = ["RPA", "RCA", "MACE", "GrossErr", "MedianCents", "VoicingAcc"]
    b = baseline.get("aggregate", {})
    f = finetuned.get("aggregate", {})

    print()
    hdr = f"{'Metric':<16}  {'Baseline':>10}  {'Fine-tuned':>10}  {'Δ':>9}"
    print(hdr); print("─" * len(hdr))
    for key in keys:
        bv = b.get(key, float("nan"))
        fv = f.get(key, float("nan"))
        if np.isnan(bv) or np.isnan(fv):
            print(f"  {key:<14}  {'n/a':>10}  {'n/a':>10}  {'n/a':>9}")
            continue
        delta = fv - bv
        sign  = "+" if delta >= 0 else ""
        print(f"  {key:<14}  {bv:>10.4f}  {fv:>10.4f}  {sign}{delta:>8.4f}")

    print()
    print("  Per technique (RPA  →  fine-tuned):")
    all_techs = sorted(
        set(baseline.get("per_technique", {}).keys()) |
        set(finetuned.get("per_technique", {}).keys())
    )
    for tech in all_techs:
        bv = baseline.get("per_technique",  {}).get(tech, {}).get("RPA", float("nan"))
        fv = finetuned.get("per_technique", {}).get(tech, {}).get("RPA", float("nan"))
        delta = fv - bv if not (np.isnan(bv) or np.isnan(fv)) else float("nan")
        arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
        print(f"    {tech:<32}  {bv:.3f} {arrow} {fv:.3f}  (Δ{delta:+.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RMVPE fine-tune test: baseline → train → eval → plots"
    )
    parser.add_argument("--gtsinger-root", required=True,
                        help="Path to GTSinger/English directory")
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--patience",    type=int,   default=8,
                        help="Early-stopping patience (epochs without val_loss improvement)")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--eval-clips",  type=int,   default=None,
                        help="Limit clip count during evaluation (default: all ~4 800)")
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--gpus",           type=int, default=0,
                        help="GPUs for training (0 = all available, 1 = single, 2 = both)")
    parser.add_argument("--weight-decay",  type=float, default=1e-2,
                        help="AdamW weight decay (default 0.01)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warmup epochs before cosine decay")
    parser.add_argument("--compile",        action="store_true", default=False,
                        help="Apply torch.compile to the model for faster steps")
    parser.add_argument("--smoke-test",  action="store_true",
                        help="Run only the smoke test then exit")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training if best.pth already exists")
    args = parser.parse_args()

    out_dir   = args.output_dir
    plots_dir = out_dir / "plots"
    train_out = out_dir / "train_run"
    cache_dir = FINETUNE_DIR / "_cache"
    for d in [out_dir, plots_dir, train_out, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    _header("STAGE 1  —  SMOKE TEST")
    stage_smoke_test(args.gtsinger_root, cache_dir, args.device)
    if args.smoke_test:
        print("--smoke-test: exiting after smoke test.\n")
        return

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    _header("STAGE 2  —  BASELINE EVALUATION")
    baseline_json = out_dir / "baseline.json"
    if baseline_json.exists():
        print(f"  Cached → {baseline_json}")
        baseline = json.loads(baseline_json.read_text())
    else:
        baseline = stage_baseline(
            args.gtsinger_root, cache_dir, out_dir, args.eval_clips, args.device
        )
    b_agg = baseline.get("aggregate", {})
    print(f"\n  Baseline  RPA@50¢={b_agg.get('RPA', float('nan')):.4f}"
          f"  MACE={b_agg.get('MACE', float('nan')):.2f}¢")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    _header("STAGE 3  —  FINE-TUNING")
    best_pth = train_out / "best.pth"
    if args.skip_training and best_pth.exists():
        print(f"  --skip-training: using {best_pth}")
    else:
        try:
            best_pth = stage_train(
                args.gtsinger_root, cache_dir, train_out,
                epochs=args.epochs, patience=args.patience,
                lr=args.lr, batch_size=args.batch_size,
                num_workers=args.num_workers, device=args.device,
                gpus=args.gpus, warmup_epochs=args.warmup_epochs,
                weight_decay=args.weight_decay, compile_model=args.compile,
            )
        except KeyboardInterrupt:
            # Python's subprocess.run() waits for the child to exit before
            # re-raising, so train.py has already saved best.pth by now.
            print("\n  Training interrupted — attempting evaluation with best saved checkpoint …")
            best_pth = train_out / "best.pth"
            if not best_pth.exists():
                ckpts = sorted((train_out / "checkpoints").glob("*.ckpt"),
                               key=lambda p: p.stat().st_mtime)
                if not ckpts:
                    print("  No checkpoint found — nothing to evaluate.", file=sys.stderr)
                    sys.exit(0)
                best_pth = ckpts[-1]
                print(f"  Falling back to: {best_pth}")

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    _header("STAGE 4  —  FINE-TUNED EVALUATION")
    ft_json = out_dir / "finetuned.json"
    if ft_json.exists() and args.skip_training:
        print(f"  Cached → {ft_json}")
        finetuned = json.loads(ft_json.read_text())
    else:
        finetuned = stage_finetuned(
            args.gtsinger_root, best_pth, out_dir, args.eval_clips, args.device
        )
    f_agg = finetuned.get("aggregate", {})
    print(f"\n  Fine-tuned  RPA@50¢={f_agg.get('RPA', float('nan')):.4f}"
          f"  MACE={f_agg.get('MACE', float('nan')):.2f}¢")

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    _header("STAGE 5  —  PLOTS")

    print("\n  [5a] Training curves …")
    plot_training_curves(train_out, plots_dir)

    print("\n  [5b] Metric comparison bars …")
    plot_metric_comparison(baseline, finetuned, plots_dir)

    print("\n  [5c] F0 contour overlays …")
    plot_f0_contours(baseline, finetuned, plots_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    _header("SUMMARY")
    print_comparison_table(baseline, finetuned)

    print(f"\n  Output directory : {out_dir}")
    print(f"  Plots            : {plots_dir}")
    print(f"  Best checkpoint  : {best_pth}")
    metrics_csv = train_out / "metrics.csv"
    if metrics_csv.exists():
        print(f"  Metrics CSV      : {metrics_csv}")
    print()


if __name__ == "__main__":
    main()
