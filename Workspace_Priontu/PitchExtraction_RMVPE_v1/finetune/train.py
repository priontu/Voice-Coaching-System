"""Fine-tune RMVPE on GTSinger/English — PyTorch Lightning.

Strategy
--------
Freeze the U-Net backbone (model.unet).  Fine-tune model.cnn + model.fc
(Conv2d projection head + BiGRU + Linear output).

Setup (once, in torch_it conda env)
-------------------------------------
    pip install "lightning>=2.2" torchmetrics huggingface_hub

Usage
-----
    python train.py \\
        --gtsinger-root "/mnt/.../GTSinger/English" \\
        --output-dir    ./runs/rmvpe_ft_01

Resume from a Lightning checkpoint
    python train.py ... --resume ./runs/rmvpe_ft_01/checkpoints/last.ckpt
"""
from __future__ import annotations

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*isinstance\(treespec, LeafSpec\).*",
    category=UserWarning,
)

import argparse
import csv
import hashlib
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchmetrics
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, random_split

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from rmvpe_src.model import E2E0
from rmvpe_src.spec import MelSpectrogram
from rmvpe_src.constants import N_MELS, SAMPLE_RATE, WINDOW_LENGTH, MEL_FMIN, MEL_FMAX
from dataset import GTSingerPitchDataset, discover_clips, CENTS_PER_BIN

HF_REPO      = "lj1995/VoiceConversionWebUI"
WEIGHTS_FILE = "rmvpe.pt"
N_PITCH_BINS = 360


# ── Weights download ───────────────────────────────────────────────────────────

def download_weights(cache_dir: Path) -> Path:
    dest = cache_dir / WEIGHTS_FILE
    if dest.exists():
        return dest
    print(f"Downloading {WEIGHTS_FILE} from HuggingFace …")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HF_REPO, filename=WEIGHTS_FILE,
                                local_dir=str(cache_dir)))


# ── Custom torchmetrics metric: RPA ───────────────────────────────────────────

class RawPitchAccuracy(torchmetrics.Metric):
    """Raw Pitch Accuracy at ±tol_cents on voiced frames.

    Accumulates correct/total counts across batches so the per-epoch value
    is clip-weighted (not batch-weighted), which matters when clips differ in
    the number of voiced frames.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(self, tol_cents: float = 50.0) -> None:
        super().__init__()
        self.tol = tol_cents
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("total",   default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor,
               voiced: torch.Tensor) -> None:
        # pred:   [B, T, 360]   target: [B, T, 360]   voiced: [B, T]
        bin_idx   = torch.arange(N_PITCH_BINS, device=pred.device, dtype=torch.float32)
        pred_bin  = pred.argmax(-1).float()                    # [B, T]
        wsum      = target.sum(-1).clamp(min=1e-6)
        true_bin  = (target * bin_idx).sum(-1) / wsum          # [B, T]
        cent_err  = (pred_bin - true_bin).abs() * CENTS_PER_BIN
        vmask     = voiced.bool()
        self.correct += (cent_err[vmask] <= self.tol).sum()
        self.total   += vmask.sum()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total.clamp(min=1)


# ── Callback: per-epoch terminal summary ──────────────────────────────────────

class EpochSummaryCallback(L.Callback):
    """Prints one persistent line per epoch so the full training history is
    visible in the terminal (Lightning's progress bar updates in-place and
    leaves no permanent record).

    Example output:
        Epoch   1/300 | train=0.2553  val=0.0305  rpa50=0.756  lr=1.00e-04
        Epoch   2/300 | train=0.2490  val=0.0295  rpa50=0.762  lr=9.87e-05  [ES 0/20]
    """

    def on_train_epoch_end(self, trainer: L.Trainer,
                           pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        m    = trainer.callback_metrics
        ep   = trainer.current_epoch + 1
        maxe = trainer.max_epochs

        def _v(key: str, fmt: str = ".4f") -> str:
            val = m.get(key)
            return format(float(val), fmt) if val is not None else "n/a"

        lr   = trainer.optimizers[0].param_groups[0]["lr"]
        line = (
            f"Epoch {ep:4d}/{maxe} | "
            f"train={_v('train/loss')}  "
            f"val={_v('val/loss')}  "
            f"rpa50={_v('val/rpa50', '.4f')}  "
            f"lr={lr:.2e}"
        )

        # Append early-stopping counter if the callback is present
        for cb in trainer.callbacks:
            if hasattr(cb, "wait_count") and hasattr(cb, "patience"):
                if cb.wait_count > 0:
                    line += f"  [ES {cb.wait_count}/{cb.patience}]"
                break

        print(line)


# ── Callback: write metrics.csv in the format expected by run_test.py ─────────

class MetricsCSVCallback(L.Callback):
    """Appends one row per epoch to <output_dir>/metrics.csv."""

    HEADER = [
        "epoch", "train_loss", "train_loss_voiced", "train_loss_unvoiced",
        "val_loss", "val_loss_voiced", "val_loss_unvoiced", "val_rpa50", "lr",
    ]

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / "metrics.csv"
        with open(self._path, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADER)

    def on_train_epoch_end(self, trainer: L.Trainer,
                           pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:   # only rank 0 writes the CSV in DDP
            return
        # Lightning fires on_validation_epoch_end BEFORE on_train_epoch_end.
        # By the time on_train_epoch_end runs, callback_metrics contains all
        # epoch-averaged metrics from both training and validation phases,
        # including val/rpa50 (logged in the module's on_validation_epoch_end).
        m  = trainer.callback_metrics
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        row = [
            trainer.current_epoch + 1,
            _r(m.get("train/loss")),
            _r(m.get("train/loss_voiced")),
            _r(m.get("train/loss_unvoiced")),
            _r(m.get("val/loss")),
            _r(m.get("val/loss_voiced")),
            _r(m.get("val/loss_unvoiced")),
            _r(m.get("val/rpa50")),
            round(float(lr), 9),
        ]
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow(row)


def _r(v) -> str | float:
    """Round a tensor or float to 6 dp; return '' if None."""
    if v is None:
        return ""
    return round(float(v), 6)



# ── LightningDataModule ────────────────────────────────────────────────────────

class GTSingerDataModule(L.LightningDataModule):
    """Handles discovery, splitting, and DataLoader creation for GTSinger."""

    def __init__(
        self,
        gtsinger_root: str,
        cache_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def _index_cache_path(self) -> Path:
        """Unique cache path per (gtsinger_root, val_split) combination."""
        key = hashlib.md5(
            f"{self.hparams.gtsinger_root}:{self.hparams.val_split}".encode()
        ).hexdigest()[:10]
        return Path(self.hparams.cache_dir) / f"dataset_index_{key}.pkl"

    def prepare_data(self) -> None:
        """Runs on rank 0 only — download weights and build the segment index.

        The segment index is the expensive part of startup (~25 s on NFS):
        it calls discover_clips() and reads 4 800+ JSON files.  We build it
        once here and persist it so that setup() on every DDP rank can load
        from cache in < 1 s instead of re-doing the NFS scan.
        """
        download_weights(Path(self.hparams.cache_dir))

        cache_path = self._index_cache_path()
        if cache_path.exists():
            return  # already built on a previous run

        print("Building dataset index (first run — saved to cache for future runs) …")
        clips = discover_clips(self.hparams.gtsinger_root)
        print(f"  Found {len(clips)} clips")

        n_val   = max(1, int(len(clips) * self.hparams.val_split))
        n_train = len(clips) - n_val
        gen     = torch.Generator().manual_seed(42)
        train_raw, val_raw = random_split(clips, [n_train, n_val], generator=gen)
        train_clips = list(train_raw)
        val_clips   = list(val_raw)

        train_ds = GTSingerPitchDataset(train_clips, stride_frames=150)
        val_ds   = GTSingerPitchDataset(val_clips,   stride_frames=300)
        print(f"  Train: {len(train_ds):,} segments   Val: {len(val_ds):,} segments")

        with open(cache_path, "wb") as fh:
            pickle.dump({
                "train_clips": train_clips,
                "val_clips":   val_clips,
                "train_index": train_ds._index,
                "val_index":   val_ds._index,
            }, fh)
        print(f"  Index cached → {cache_path}")

    def setup(self, stage: str | None = None) -> None:
        """Load the pre-built index from cache (all DDP ranks, < 1 s)."""
        cache_path = self._index_cache_path()
        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                payload = pickle.load(fh)
            self.train_ds = GTSingerPitchDataset._from_cache(
                payload["train_clips"], payload["train_index"], stride_frames=150
            )
            self.val_ds = GTSingerPitchDataset._from_cache(
                payload["val_clips"], payload["val_index"], stride_frames=300
            )
        else:
            # Cache missing (e.g. prepare_data skipped) — build from scratch
            print("Warning: index cache not found, building from scratch …")
            clips   = discover_clips(self.hparams.gtsinger_root)
            n_val   = max(1, int(len(clips) * self.hparams.val_split))
            n_train = len(clips) - n_val
            gen     = torch.Generator().manual_seed(42)
            train_raw, val_raw = random_split(clips, [n_train, n_val], generator=gen)
            self.train_ds = GTSingerPitchDataset(list(train_raw), stride_frames=150)
            self.val_ds   = GTSingerPitchDataset(list(val_raw),   stride_frames=300)

        print(f"Train: {len(self.train_ds):,} segments   Val: {len(self.val_ds):,} segments")

    def _loader(self, ds: GTSingerPitchDataset, shuffle: bool) -> DataLoader:
        nw = self.hparams.num_workers
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=4 if nw > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)


# ── LightningModule ────────────────────────────────────────────────────────────

class RMVPEFinetuner(L.LightningModule):
    """Fine-tunes the RMVPE E2E0 model on GTSinger pitch annotations.

    The U-Net backbone is frozen.  Only the Conv2d projection head (model.cnn)
    and the BiGRU + Linear output layers (model.fc) are trained.
    """

    def __init__(
        self,
        weights_path: str,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 30,
        warmup_epochs: int = 5,
        compile_model: bool = False,
        output_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["weights_path", "output_dir"])
        self._output_dir = Path(output_dir) if output_dir else None

        # ── Backbone ──────────────────────────────────────────────────────────
        self.model = E2E0(4, 1, (2, 2))
        ckpt  = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        miss, unexp = self.model.load_state_dict(state, strict=False)
        if miss:
            print(f"  Missing keys ({len(miss)}): {miss[:3]} …")
        if unexp:
            print(f"  Unexpected keys ({len(unexp)}): {unexp[:3]} …")

        # Freeze the entire U-Net — all parameters get requires_grad=False.
        # The U-Net runs inside torch.no_grad() in forward(), so any parameter
        # with requires_grad=True inside the U-Net would confuse DDP's bucket
        # tracker (it would see a parameter used in forward but never receiving
        # a gradient), causing a RuntimeError with find_unused_parameters=False.
        # Keeping the full U-Net frozen avoids this conflict while preserving
        # the torch.no_grad() VRAM savings that allow batch_size=128.
        for p in self.model.unet.parameters():
            p.requires_grad = False
        for p in self.model.cnn.parameters():
            p.requires_grad = True
        for p in self.model.fc.parameters():
            p.requires_grad = True

        n_frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Frozen: {n_frozen:,}  Trainable: {n_train:,}")

        # ── Preprocessing (moves to device with the module) ───────────────────
        self.mel = MelSpectrogram(
            n_mel_channels=N_MELS,
            sampling_rate=SAMPLE_RATE,
            win_length=WINDOW_LENGTH,
            hop_length=160,
            n_fft=None,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        self.val_rpa = RawPitchAccuracy(tol_cents=50.0)
        self._best_rpa: float = 0.0

    # ── torch.compile (called once per rank after device placement) ───────────

    def setup(self, stage: str) -> None:
        if self.hparams.compile_model and stage == "fit":
            # max-autotune-no-cudagraphs: same Triton kernel autotuning as
            # max-autotune, but without CUDA graph capture.
            # CUDA graphs cannot be used here because the U-Net's skip
            # connections (encoder concat_tensors → decoder) hold references
            # to intermediate tensors that get overwritten on graph replay,
            # causing the "CUDAGraphs tensor overwritten" runtime error.
            mode = "max-autotune-no-cudagraphs"
            self.model.unet = torch.compile(self.model.unet, mode=mode)
            self.model.cnn  = torch.compile(self.model.cnn,  mode=mode)
            self.model.fc   = torch.compile(self.model.fc,   mode=mode)
            self.print(f"  torch.compile ({mode}) applied per submodule")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Mel → pad-to-32 → U-Net (no grad) → cnn+fc (grad) → trim.

        The U-Net is 89% of forward compute but is frozen — it never receives
        gradients.  Wrapping it in torch.no_grad() stops PyTorch from storing
        its intermediate activations (~200 MB/batch) for a backward pass that
        will never happen, freeing VRAM and reducing bookkeeping overhead.
        """
        mel      = self.mel(audio, center=True)          # [B, 128, T]
        n_frames = mel.shape[-1]
        pad      = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if pad > 0:
            mel = F.pad(mel, (0, pad), mode="reflect")

        # Reshape to U-Net's expected layout: [B, 1, T, N_MELS]
        mel_in = mel.transpose(-1, -2).unsqueeze(1)

        # U-Net: frozen — no autograd graph, no activation storage
        with torch.no_grad():
            unet_out = self.model.unet(mel_in)           # [B, 16, T, N_MELS]

        # Projection head + BiGRU: trainable — autograd graph built here only
        x    = self.model.cnn(unet_out).transpose(1, 2).flatten(-2)  # [B, T, 384]
        pred = self.model.fc(x)                          # [B, T, 360]
        return pred[:, :n_frames], n_frames

    # ── Shared loss computation ────────────────────────────────────────────────

    def _loss(self, pred: torch.Tensor, target: torch.Tensor,
              voiced: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Voiced-weighted BCE.  Returns (total, voiced_term, unvoiced_term).

        F.binary_cross_entropy requires float32 inputs and must run outside the
        AMP autocast region (AMP blocks BCE entirely because it is numerically
        unsafe in float16).  We disable autocast explicitly here and upcast both
        tensors, then the returned losses are float32 scalars that Lightning can
        scale and backprop through normally.
        """
        vmask = voiced.bool()
        umask = ~vmask
        device_type = pred.device.type  # "cuda" or "cpu"

        with torch.amp.autocast(device_type=device_type, enabled=False):
            p32 = pred.float()
            t32 = target.float()
            loss_v = (F.binary_cross_entropy(p32[vmask], t32[vmask])
                      if vmask.any() else pred.new_zeros(()))
            loss_u = (F.binary_cross_entropy(p32[umask], torch.zeros_like(p32[umask]))
                      if umask.any() else pred.new_zeros(()))

        return loss_v + 0.3 * loss_u, loss_v, loss_u

    def _shared_step(self, batch: dict) -> tuple:
        pred, n_frames = self(batch["audio"])
        target = batch["target"]
        voiced = batch["voiced"]
        T      = min(pred.shape[1], target.shape[1], voiced.shape[1])
        pred   = pred[:, :T]
        target = target[:, :T]
        voiced = voiced[:, :T]
        loss, loss_v, loss_u = self._loss(pred, target, voiced)
        return loss, loss_v, loss_u, pred, target, voiced

    # ── Training / validation steps ───────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_v, loss_u, *_ = self._shared_step(batch)
        # Step-level metric for the progress bar; sync_dist=False is fine here
        # because it's just for live monitoring, not aggregation.
        self.log("train/loss_step",     loss,   on_step=True,  on_epoch=False, prog_bar=True)
        # Epoch averages — sync_dist=True reduces across GPUs in DDP so the
        # logged value is the global mean, matching single-GPU behaviour.
        self.log("train/loss",          loss,   on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_voiced",   loss_v, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_unvoiced", loss_u, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_v, loss_u, pred, target, voiced = self._shared_step(batch)
        self.log("val/loss",          loss,   on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss_voiced",   loss_v, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_unvoiced", loss_u, on_step=False, on_epoch=True, sync_dist=True)
        # RawPitchAccuracy uses dist_reduce_fx="sum" — handles DDP automatically
        self.val_rpa.update(pred.detach(), target.detach(), voiced.detach())
        return loss

    def on_validation_epoch_end(self) -> None:
        rpa = self.val_rpa.compute()   # already reduced across GPUs by torchmetrics
        # sync_dist=True: torchmetrics has already all-reduced, so all ranks hold
        # the same value; taking the mean of N identical values is a no-op, but
        # it satisfies Lightning's distributed-logging contract and silences the warning.
        self.log("val/rpa50", rpa, prog_bar=True, sync_dist=True)
        self.val_rpa.reset()

        # Guard to rank 0: all DDP ranks call this hook; only rank 0 should write
        rpa_f = float(rpa)
        if rpa_f > self._best_rpa and self._output_dir is not None \
                and self.trainer.is_global_zero:
            self._best_rpa = rpa_f
            # torch.compile wraps submodules in OptimizedModule, which inserts
            # '._orig_mod.' into every state-dict key.  Strip it so the saved
            # weights load cleanly into an uncompiled E2E0 in evaluate.py.
            raw = self.model.state_dict()
            raw = {k.replace("._orig_mod.", "."): v for k, v in raw.items()}
            torch.save({"model": raw}, self._output_dir / "best.pth")
            self.print(f"  → New best RPA@50¢: {self._best_rpa:.4f}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable, lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup = self.hparams.warmup_epochs
        decay  = max(1, self.hparams.max_epochs - warmup)

        if warmup > 0:
            # Linear ramp from lr*0.1 → lr over `warmup` epochs, then cosine
            # decay from lr → lr*0.01 over the remaining epochs.
            # SequentialLR chains two schedulers at a milestone.
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, end_factor=1.0, total_iters=warmup
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=decay, eta_min=self.hparams.lr * 0.01
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
            )
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr * 0.01
            )

        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RMVPE on GTSinger (Lightning)")
    parser.add_argument("--gtsinger-root", required=True)
    parser.add_argument("--output-dir",    default="./runs/rmvpe_ft")
    parser.add_argument("--cache-dir",     default="./_cache")
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--batch-size",    type=int,   default=8)
    parser.add_argument("--num-workers",   type=int,   default=4)
    parser.add_argument("--val-split",     type=float, default=0.1)
    parser.add_argument("--weight-decay",  type=float, default=1e-2,
                        help="AdamW weight decay (L2 regularisation, default 0.01)")
    parser.add_argument("--warmup-epochs", type=int,   default=5,
                        help="Linear LR warmup epochs before cosine decay (0 = disabled)")
    parser.add_argument("--compile",       action="store_true", default=False,
                        help="Apply torch.compile to the model for faster iteration")
    parser.add_argument("--patience",      type=int,   default=8,
                        help="EarlyStopping patience in epochs (0 = disabled)")
    parser.add_argument("--precision",     default="auto",
                        choices=["auto", "16-mixed", "bf16-mixed", "32"],
                        help="Training precision (auto picks 16-mixed when CUDA available)")
    parser.add_argument("--resume",        default=None,
                        help="Path to a Lightning .ckpt to resume from")
    parser.add_argument("--device",        default="auto",
                        help="auto | cpu | cuda (passed to Lightning accelerator)")
    parser.add_argument("--gpus",          type=int, default=0,
                        help="Number of GPUs to use (0 = all available, 1 = single GPU)")
    args = parser.parse_args()

    import os
    # Suppress max-autotune's verbose per-kernel benchmarking output.
    # The autotuning itself still runs and picks optimal kernels — only the
    # printed candidate tables are hidden.
    os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_VERBOSE", "0")
    try:
        import torch._inductor.config as _inductor_cfg
        _inductor_cfg.autotune_verbose = False
    except Exception:
        pass
    # Disable NCCL's HeartbeatMonitor — it loops indefinitely trying to reach
    # the TCPStore after rank 0 exits, flooding the terminal with "Broken pipe".
    # Only useful for multi-node setups; irrelevant for single-node multi-GPU.
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    # RTX 3090s do not support PCIe peer-to-peer access. Without this flag NCCL
    # probes P2P on startup, the probe fails and leaves a dangling async CUDA
    # error that surfaces as "out of memory" on the next synchronising call
    # (torch.cuda.set_device). Disabling P2P makes NCCL use the CPU/PCIe path.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    # Let PyTorch use Tensor Cores for float32 matmuls on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")
    # Fixed input shapes (same mel size every step) — let cuDNN benchmark
    # multiple convolution algorithms and cache the fastest one
    torch.backends.cudnn.benchmark = True
    # DDP + torch.compile causes benign CUDA stream ordering warnings on
    # AccumulateGrad nodes; the underlying sync still happens correctly.
    # (API added in PyTorch 2.1 — silently skip on older builds.)
    _suppress = getattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch", None)
    if _suppress:
        _suppress(False)


    out_dir   = Path(args.output_dir)
    ckpt_dir  = out_dir / "checkpoints"
    cache_dir = Path(args.cache_dir)
    for d in [out_dir, ckpt_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Precision ─────────────────────────────────────────────────────────────
    if args.precision == "auto":
        precision = "16-mixed" if torch.cuda.is_available() else "32"
    else:
        precision = args.precision

    # ── Accelerator ───────────────────────────────────────────────────────────
    if args.device == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    elif args.device.startswith("cuda"):
        accelerator = "gpu"
    else:
        accelerator = args.device

    # ── DataModule ────────────────────────────────────────────────────────────
    dm = GTSingerDataModule(
        gtsinger_root=args.gtsinger_root,
        cache_dir=str(cache_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    weights_path = download_weights(cache_dir)
    litmodel = RMVPEFinetuner(
        weights_path=str(weights_path),
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        compile_model=args.compile,
        output_dir=str(out_dir),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        MetricsCSVCallback(out_dir),
        EpochSummaryCallback(),
    ]
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=args.patience,
                mode="min",
                min_delta=1e-5,
                verbose=True,
            )
        )

    # ── Loggers ───────────────────────────────────────────────────────────────
    loggers = [
        TensorBoardLogger(save_dir=str(out_dir), name="tb",  version=""),
        CSVLogger(        save_dir=str(out_dir), name="csv", version=""),
    ]

    # ── Devices + strategy ────────────────────────────────────────────────────
    devices = args.gpus if args.gpus > 0 else "auto"
    n_gpus  = torch.cuda.device_count() if args.gpus == 0 else args.gpus

    # DDPStrategy with find_unused_parameters=False:
    #   - The frozen U-Net has no gradients, so skipping the unused-parameter
    #     check avoids a per-step overhead that DDP would otherwise pay.
    #   - static_graph=True gives further speedup since the computation graph
    #     is identical every step (no dynamic control flow in the model).
    strategy = (
        DDPStrategy(find_unused_parameters=False)
        if accelerator == "gpu" and n_gpus > 1
        else "auto"
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        gradient_clip_val=5.0,
        log_every_n_steps=20,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    interrupted = False
    try:
        trainer.fit(litmodel, datamodule=dm, ckpt_path=args.resume or None)
    except KeyboardInterrupt:
        interrupted = True
        print("\n  Training interrupted — saving best checkpoint …", flush=True)

    # Ensure best.pth exists even when training is interrupted before the first
    # validation epoch completes (on_validation_epoch_end never ran).
    best_pth = out_dir / "best.pth"
    if not best_pth.exists():
        raw = litmodel.model.state_dict()
        raw = {k.replace("._orig_mod.", "."): v for k, v in raw.items()}
        torch.save({"model": raw}, best_pth)
        print(f"  Saved current model state → {best_pth}", flush=True)

    status = "Interrupted" if interrupted else "Done"
    print(f"\n{status}.  Best val RPA@50¢: {litmodel._best_rpa:.4f}")
    print(f"Best checkpoint (evaluate.py format): {out_dir}/best.pth")
    print(f"Metrics CSV:                          {out_dir}/metrics.csv")


if __name__ == "__main__":
    main()
