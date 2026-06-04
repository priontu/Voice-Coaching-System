"""Fine-tune NanoPitch on GTSinger/English.

Loads the Priontu_Chowdhury/weights.pth checkpoint and continues training on
the GTSinger English dataset using the same voiced-weighted BCE loss as the
original NanoPitch training.

By default the Conv1d feature extractor is frozen and only the GRU layers and
output heads are trained (~180 K of 333 K parameters) — analogous to how the
RMVPE fine-tuning freezes the U-Net backbone.

Usage (run from the NanoPitch_v2 root):
    python training/finetune.py \\
        --gtsinger-root "/mnt/.../GTSinger/English" \\
        --output-dir ./runs/finetune_gtsinger

Full fine-tuning (all layers):
    python training/finetune.py ... --no-freeze-conv --lr 3e-5

Resume from a checkpoint:
    python training/finetune.py ... --resume ./runs/finetune_gtsinger/checkpoints/last.pth
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as TAF
import torchaudio.transforms as TAT
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import NanoPitch, PITCH_BINS, PITCH_FMIN, PITCH_CENTS_PER_BIN, viterbi_decode
from gtsinger_dataset import (GTSingerNanoPitchDataset, CachedGTSingerDataset,
                               discover_clips, SAMPLE_RATE, HOP_LENGTH,
                               N_MELS, N_FFT, WIN_LENGTH, F_MIN, F_MAX,
                               _GTSINGER_SR)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune NanoPitch on GTSinger")

    # Data
    p.add_argument("--gtsinger-root", required=True,
                   help="Path to GTSinger/English directory")
    p.add_argument("--weights", default=None,
                   help="Pretrained weights.pth to initialise from. "
                        "Defaults to submissions/Priontu_Chowdhury/weights.pth "
                        "relative to the NanoPitch_v2 root.")
    p.add_argument("--output-dir", default="./runs/finetune_gtsinger")
    p.add_argument("--resume", default=None,
                   help="Resume fine-tuning from this checkpoint")
    p.add_argument("--cache-dir", default=None,
                   help="Path to pre-computed mel cache from cache_dataset.py. "
                        "Loads from local SSD instead of NFS — ~70× faster. "
                        "Falls back to on-the-fly computation if not set.")
    p.add_argument("--gpu-mel", action="store_true", default=False,
                   help="Compute mel spectrogram on GPU instead of in DataLoader "
                        "workers. Eliminates CPU mel bottleneck. JSON annotations "
                        "are pre-loaded into RAM automatically.")
    p.add_argument("--preload-audio", action="store_true", default=False,
                   help="Pre-load all WAV files into RAM at startup (~2.8 GB, "
                        "~20 s one-time cost). Eliminates 150 ms cold NFS reads "
                        "per clip — makes data loading ~200× faster. "
                        "Combine with --gpu-mel for maximum speed.")

    # Architecture — must match the pretrained checkpoint
    p.add_argument("--cond-size", type=int, default=64)
    p.add_argument("--gru-size",  type=int, default=96)

    # Fine-tuning strategy
    p.add_argument("--freeze-conv", dest="freeze_conv",
                   action="store_true", default=True,
                   help="Freeze Conv1d layers; train GRU + output heads only (default)")
    p.add_argument("--no-freeze-conv", dest="freeze_conv",
                   action="store_false",
                   help="Train all parameters (full fine-tuning, use a lower --lr)")

    # Training schedule
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="Peak learning rate (lower than from-scratch 1e-3)")
    p.add_argument("--warmup-epochs", type=int,   default=5,
                   help="Linear LR warmup before cosine decay")
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--seq-len",       type=int,   default=200,
                   help="Training window in frames (200 = 2 s at 100 fps)")
    p.add_argument("--stride-frames", type=int,   default=150,
                   help="Hop between training segments (150 → 50%% overlap)")
    p.add_argument("--val-split",     type=float, default=0.1,
                   help="Fraction of clips held out for validation")
    p.add_argument("--num-workers",   type=int,   default=4)

    # Loss
    p.add_argument("--w-vad",   type=float, default=0.1,
                   help="Weight for VAD BCE loss")
    p.add_argument("--w-pitch", type=float, default=1.0,
                   help="Weight for pitch BCE loss")

    # Evaluation / checkpointing
    p.add_argument("--eval-every", type=int, default=5,
                   help="Run validation every N epochs")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save an epoch checkpoint every N epochs")
    p.add_argument("--patience",   type=int, default=10,
                   help="Early-stopping patience in validation checks "
                        "(0 = disabled). Stops if val RPA does not improve "
                        "by --min-delta for this many consecutive checks.")
    p.add_argument("--min-delta",  type=float, default=1e-4,
                   help="Minimum RPA improvement to reset the early-stopping counter")
    p.add_argument("--device", default="auto")

    return p.parse_args()


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_pretrained(model: NanoPitch, path: Path) -> int:
    """Load weights into model. Returns the epoch stored in the checkpoint."""
    ckpt  = torch.load(str(path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): "
              f"{missing[:3]}{'…' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): "
              f"{unexpected[:3]}{'…' if len(unexpected) > 3 else ''}")
    epoch = ckpt.get("epoch", 0)
    loss  = ckpt.get("loss",  float("nan"))
    rpa   = ckpt.get("rpa",   float("nan"))
    print(f"  Loaded: epoch={epoch}  loss={loss:.5f}  rpa={rpa:.4f}")
    return epoch


def save_checkpoint(model: NanoPitch, args: argparse.Namespace,
                    epoch: int, loss: float, rpa: float, path: Path) -> None:
    torch.save({
        "epoch":        epoch,
        "state_dict":   model.state_dict(),
        "model_kwargs": {"cond_size": args.cond_size, "gru_size": args.gru_size},
        "loss":         loss,
        "rpa":          rpa,
    }, str(path))


# ── Loss ───────────────────────────────────────────────────────────────────────

def build_pitch_targets(f0_batch: torch.Tensor, T: int,
                        device: torch.device,
                        sigma_bins: float = 1.2) -> torch.Tensor:
    """Build (B, T, 360) pitch posteriorgrams — vectorized on GPU.

    Replaces the original per-sample CPU numpy loop with a single batched
    GPU operation using broadcasting, reducing per-batch time from ~50 ms
    to <1 ms.
    """
    f0 = f0_batch[:, :T].to(device).float()          # (B, T)
    voiced = f0 > 0                                    # (B, T) bool

    # Convert Hz → bin index; clamp to avoid log(0) on unvoiced frames
    safe_f0 = f0.clamp(min=1e-10)
    bins = (1200.0 * torch.log2(safe_f0 / PITCH_FMIN)
            / PITCH_CENTS_PER_BIN)                     # (B, T)

    # Gaussian: (B, T, 1) broadcast against (1, 1, PITCH_BINS)
    bin_idx = torch.arange(PITCH_BINS, device=device, dtype=torch.float32)
    dist    = bins.unsqueeze(-1) - bin_idx             # (B, T, PITCH_BINS)
    target  = torch.exp(-0.5 * (dist / sigma_bins) ** 2)
    target  = target * voiced.unsqueeze(-1).float()    # zero out unvoiced frames
    return target


def compute_loss(pred_vad: torch.Tensor, pred_pitch: torch.Tensor,
                 vad_target: torch.Tensor, pitch_target: torch.Tensor,
                 w_vad: float, w_pitch: float
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voiced-weighted BCE — same formulation as the original NanoPitch train.py."""
    bce = nn.functional.binary_cross_entropy
    vad_loss   = bce(pred_vad.squeeze(-1), vad_target)
    voiced_w   = vad_target.unsqueeze(-1)               # (B, T, 1)
    pitch_loss = (voiced_w * bce(pred_pitch, pitch_target,
                                 reduction="none")).mean()
    total      = w_vad * vad_loss + w_pitch * pitch_loss
    return total, vad_loss, pitch_loss


# ── Training / validation ──────────────────────────────────────────────────────

def _make_gpu_mel(device: torch.device) -> TAT.MelSpectrogram:
    return TAT.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        f_min=F_MIN, f_max=F_MAX, power=1.0, center=True,
    ).to(device)


def _audio_to_mel_gpu(audio: torch.Tensor, mel_transform: TAT.MelSpectrogram,
                      seq_len: int) -> torch.Tensor:
    """Convert (B, n_samples) 16 kHz audio to (B, seq_len, N_MELS) log-mel on GPU."""
    mel = mel_transform(audio)              # (B, N_MELS, T)
    mel = torch.log(mel.transpose(1, 2) + 1e-7)  # (B, T, N_MELS)
    T   = mel.shape[1]
    if T > seq_len:
        mel = mel[:, :seq_len, :]
    elif T < seq_len:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, seq_len - T))
    return mel


def train_epoch(model: NanoPitch, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler,
                device: torch.device, args: argparse.Namespace,
                writer: SummaryWriter, epoch: int,
                gpu_mel_transform: TAT.MelSpectrogram | None = None) -> float:
    model.train()
    totals = {"loss": 0.0, "vad": 0.0, "pitch": 0.0}
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}", unit="batch", leave=False)
    for audio_or_mel, vad_t, f0_t in pbar:
        vad_t = vad_t.to(device)     # (B, T)

        if gpu_mel_transform is not None:
            # GPU mel path: audio_or_mel is (B, n_samples) raw 16 kHz waveform
            audio = audio_or_mel.to(device)
            mel   = _audio_to_mel_gpu(audio, gpu_mel_transform, args.seq_len)
        else:
            mel = audio_or_mel.to(device)   # (B, T, 40) pre-computed
        T     = mel.shape[1]

        pitch_t = build_pitch_targets(f0_t, T, device)  # (B, T, 360)

        pred_vad, pred_pitch, _ = model(mel)
        loss, vl, pl = compute_loss(pred_vad, pred_pitch, vad_t, pitch_t,
                                    args.w_vad, args.w_pitch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        totals["loss"]  += loss.item()
        totals["vad"]   += vl.item()
        totals["pitch"] += pl.item()
        n += 1
        pbar.set_postfix(loss=f"{totals['loss']/n:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")

    for k, v in totals.items():
        writer.add_scalar(f"train/{k}", v / max(n, 1), epoch)
    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
    return totals["loss"] / max(n, 1)


@torch.no_grad()
def validate(model: NanoPitch, loader: DataLoader, device: torch.device,
             args: argparse.Namespace, writer: SummaryWriter,
             epoch: int,
             gpu_mel_transform: TAT.MelSpectrogram | None = None,
             ) -> tuple[float, float]:
    """Compute val loss and RPA@50¢ via offline Viterbi decoding."""
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    rpa_sum   = 0.0
    n_clips   = 0

    for audio_or_mel, vad_t, f0_t in tqdm(loader, desc="  Val", unit="batch", leave=False):
        vad_t = vad_t.to(device)
        if gpu_mel_transform is not None:
            mel = _audio_to_mel_gpu(audio_or_mel.to(device),
                                    gpu_mel_transform, args.seq_len)
        else:
            mel = audio_or_mel.to(device)
        T     = mel.shape[1]

        pitch_t = build_pitch_targets(f0_t, T, device)
        pred_vad, pred_pitch, _ = model(mel)
        loss, _, _ = compute_loss(pred_vad, pred_pitch, vad_t, pitch_t,
                                  args.w_vad, args.w_pitch)
        loss_sum  += loss.item()
        n_batches += 1

        # RPA@50¢: decode each clip in the batch independently
        pp_np = pred_pitch.cpu().numpy()   # (B, T, 360)
        f0_np = f0_t.numpy()               # (B, T)
        for b in range(mel.shape[0]):
            f0_ref  = f0_np[b]
            f0_pred = viterbi_decode(pp_np[b])
            ref_v   = f0_ref  > 0
            pred_v  = f0_pred > 0
            both    = ref_v & pred_v
            if both.sum() > 0:
                err = np.abs(1200.0 * np.log2(
                    f0_pred[both] / (f0_ref[both] + 1e-10) + 1e-10))
                rpa_sum += float(np.mean(err < 50))
                n_clips += 1

    val_loss = loss_sum / max(n_batches, 1)
    val_rpa  = rpa_sum  / max(n_clips,  1)
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/rpa",  val_rpa,  epoch)
    return val_loss, val_rpa


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Output dirs
    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NanoPitch(cond_size=args.cond_size, gru_size=args.gru_size)
    start_epoch = 1
    best_rpa    = 0.0

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        resumed_epoch = load_pretrained(model, Path(args.resume))
        start_epoch   = resumed_epoch + 1
        # Recover best_rpa from the checkpoint if stored
        ckpt_meta = torch.load(args.resume, map_location="cpu")
        best_rpa  = ckpt_meta.get("rpa", 0.0)
    else:
        # Default weights path: submissions/Priontu_Chowdhury/weights.pth
        weights = args.weights or str(
            Path(__file__).resolve().parents[1] /
            "submissions" / "Priontu_Chowdhury" / "weights.pth"
        )
        print(f"\nLoading pretrained weights: {weights}")
        load_pretrained(model, Path(weights))

    if args.freeze_conv:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.conv2.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nFrozen Conv: {n_frozen:,} params frozen, {n_train:,} trainable "
              f"(GRU + output heads)")
    else:
        n_train = sum(p.numel() for p in model.parameters())
        print(f"\nFull fine-tuning: {n_train:,} trainable parameters")

    model.to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\nDiscovering GTSinger clips in: {args.gtsinger_root}")
    all_clips = discover_clips(args.gtsinger_root)
    print(f"  Found {len(all_clips)} clips")

    # Deterministic train/val split by clip
    rng = np.random.default_rng(42)
    indices = np.arange(len(all_clips))
    rng.shuffle(indices)
    n_val   = max(1, int(len(all_clips) * args.val_split))
    n_train = len(all_clips) - n_val
    train_clips = [all_clips[i] for i in indices[n_val:]]
    val_clips   = [all_clips[i] for i in indices[:n_val]]
    print(f"  Train clips: {n_train}   Val clips: {n_val}")

    gpu_mel_transform = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        print(f"  Using mel cache: {cache_dir}")
        train_ds = CachedGTSingerDataset(train_clips, cache_dir,
                                          seq_len=args.seq_len,
                                          stride_frames=args.stride_frames)
        val_ds   = CachedGTSingerDataset(val_clips, cache_dir,
                                          seq_len=args.seq_len,
                                          stride_frames=args.seq_len)
    elif args.preload_audio or args.gpu_mel:
        if args.preload_audio:
            print("  Audio + JSON pre-loaded into RAM; mel computed on GPU.")
        else:
            print("  GPU mel pipeline: JSON pre-loaded into RAM, "
                  "mel computed on GPU each batch.")
        gpu_mel_transform = _make_gpu_mel(device)
        train_ds = GTSingerNanoPitchDataset(
            train_clips, seq_len=args.seq_len, stride_frames=args.stride_frames,
            return_audio=True, preload_audio=args.preload_audio)
        val_ds   = GTSingerNanoPitchDataset(
            val_clips, seq_len=args.seq_len, stride_frames=args.seq_len,
            return_audio=True, preload_audio=args.preload_audio)
    else:
        print("  CPU mel (default). Use --preload-audio --gpu-mel for fastest loading.")
        train_ds = GTSingerNanoPitchDataset(
            train_clips, seq_len=args.seq_len, stride_frames=args.stride_frames)
        val_ds   = GTSingerNanoPitchDataset(
            val_clips,   seq_len=args.seq_len, stride_frames=args.seq_len)
    print(f"  Train segments: {len(train_ds)}   Val segments: {len(val_ds)}")

    pin  = device.type == "cuda"
    nw   = args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, drop_last=True, pin_memory=pin,
        persistent_workers=(nw > 0))
    val_loader   = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin,
        persistent_workers=(nw > 0))

    # ── Optimiser + LR schedule ───────────────────────────────────────────────
    trainable  = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(trainable, lr=args.lr,
                                   betas=(0.8, 0.98), eps=1e-8,
                                   weight_decay=args.weight_decay)

    # Linear warmup → cosine decay to 1% of peak LR
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ── Training loop ─────────────────────────────────────────────────────────
    es_info = (f"patience={args.patience}, min_delta={args.min_delta}"
               if args.patience > 0 else "disabled")
    print(f"\nFine-tuning for {args.epochs} epochs "
          f"(lr={args.lr}, warmup={args.warmup_epochs}, "
          f"{'conv frozen' if args.freeze_conv else 'all layers'}, "
          f"early stopping: {es_info})\n")

    epoch       = start_epoch - 1
    train_loss  = float("nan")
    interrupted = False
    es_counter  = 0          # counts consecutive val checks without improvement

    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                     device, args, writer, epoch,
                                     gpu_mel_transform)
            dt = time.time() - t0

            if epoch % args.eval_every == 0 or epoch == start_epoch:
                val_loss, val_rpa = validate(model, val_loader, device,
                                             args, writer, epoch,
                                             gpu_mel_transform)

                improved = val_rpa > best_rpa + args.min_delta
                if improved:
                    best_rpa   = val_rpa
                    es_counter = 0
                    save_checkpoint(model, args, epoch, val_loss, val_rpa,
                                    ckpt_dir / "best.pth")
                else:
                    es_counter += 1

                es_tag = (f"  [ES {es_counter}/{args.patience}]"
                          if args.patience > 0 else "")
                flag   = " ← best" if improved else ""
                print(f"Epoch {epoch:3d}/{start_epoch + args.epochs - 1} | "
                      f"train={train_loss:.4f}  val={val_loss:.4f}  "
                      f"rpa={val_rpa:.4f}  dt={dt:.0f}s{flag}{es_tag}")

                if args.patience > 0 and es_counter >= args.patience:
                    print(f"\n  Early stopping: val RPA has not improved by "
                          f"{args.min_delta} for {args.patience} consecutive "
                          f"validation checks.")
                    break
            else:
                print(f"Epoch {epoch:3d}/{start_epoch + args.epochs - 1} | "
                      f"train={train_loss:.4f}  dt={dt:.0f}s")

            if epoch % args.save_every == 0:
                save_checkpoint(model, args, epoch, train_loss, best_rpa,
                                ckpt_dir / f"epoch_{epoch:03d}.pth")

    except KeyboardInterrupt:
        interrupted = True
        print("\n  Interrupted — saving last checkpoint …")

    # Always save the most recent state so training can resume
    save_checkpoint(model, args, epoch, train_loss, best_rpa,
                    ckpt_dir / "last.pth")

    writer.close()
    status = "Interrupted" if interrupted else "Done"
    print(f"\n{status}. Best val RPA@50¢: {best_rpa:.4f}")
    print(f"Best checkpoint : {ckpt_dir / 'best.pth'}")
    print(f"Last checkpoint : {ckpt_dir / 'last.pth'}")


if __name__ == "__main__":
    main()
