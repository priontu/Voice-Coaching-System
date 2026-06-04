"""Evaluate a (fine-tuned) RMVPE checkpoint against GTSinger/English.

Computes RPA, RCA, MACE, gross-error rate, and median cent error for each
SNR condition.  Run this after training to compare baseline vs. fine-tuned.

Usage
-----
    # Baseline (original rmvpe.pt)
    python evaluate.py \\
        --weights ./_cache/rmvpe.pt \\
        --gtsinger-root "/mnt/researchfiles/ECE IMAPLE/cluster_data/archive/GTSinger/English" \\
        --tag baseline

    # Fine-tuned checkpoint
    python evaluate.py \\
        --weights ./runs/rmvpe_ft_01/best.pth \\
        --gtsinger-root "/mnt/researchfiles/ECE IMAPLE/cluster_data/archive/GTSinger/English" \\
        --tag finetuned
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
from tqdm import tqdm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from rmvpe_src.model import E2E0
from rmvpe_src.spec import MelSpectrogram
from rmvpe_src.constants import N_MELS, N_CLASS, SAMPLE_RATE, WINDOW_LENGTH, MEL_FMIN, MEL_FMAX
from rmvpe_src.utils import to_local_average_f0

from dataset import (
    discover_clips, build_f0_track, hz_to_bin,
    CONST, CENTS_PER_BIN, N_PITCH_BINS, HOP_LENGTH,
)

THRED = 0.03      # voicing threshold (max of pitch posterior)
RPA_TOL = 50.0    # cents
GROSS_TOL = 50.0  # cents (same as RPA for this dataset)


# ── Model utilities ────────────────────────────────────────────────────────────

def load_model(weights_path: Path, device: torch.device) -> E2E0:
    model = E2E0(4, 1, (2, 2))
    ckpt = torch.load(str(weights_path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    # torch.compile inserts '._orig_mod.' into keys — strip it if present
    state = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def build_mel_extractor(device: torch.device) -> MelSpectrogram:
    return MelSpectrogram(
        n_mel_channels=N_MELS,
        sampling_rate=SAMPLE_RATE,
        win_length=WINDOW_LENGTH,
        hop_length=160,
        n_fft=None,
        mel_fmin=MEL_FMIN,
        mel_fmax=MEL_FMAX,
    ).to(device)


@torch.no_grad()
def infer_clip(audio_np: np.ndarray, model: E2E0,
               mel_extractor: MelSpectrogram,
               device: torch.device) -> np.ndarray:
    """Run RMVPE inference on a single audio clip.

    Returns f0 array in Hz (0 = unvoiced), shape (n_frames,).
    """
    audio = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)
    mel = mel_extractor(audio, center=True)           # [1, 128, T]
    n_frames = mel.shape[-1]
    pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
    if pad > 0:
        mel = F.pad(mel, (0, pad), mode="reflect")
    hidden = model(mel)                                # [1, T_padded, 360]
    hidden = hidden[:, :n_frames]                      # [1, T, 360]
    f0 = to_local_average_f0(hidden, thred=THRED)     # (T,) Hz
    return f0


# ── Metrics ────────────────────────────────────────────────────────────────────

def pitch_metrics(pred_f0: np.ndarray, ref_f0: np.ndarray) -> dict[str, float]:
    """Compute frame-level pitch metrics on voiced frames.

    Args:
        pred_f0: (T,) predicted F0 in Hz (0 = unvoiced)
        ref_f0:  (T,) reference F0 in Hz from MIDI (0 = unvoiced)

    Returns dict with RPA, RCA, MACE, GrossErr, MedianCents, VoicingAcc.
    """
    T = min(len(pred_f0), len(ref_f0))
    pred_f0 = pred_f0[:T]
    ref_f0  = ref_f0[:T]

    pred_voiced = pred_f0 > 0
    ref_voiced  = ref_f0 > 0
    both_voiced = pred_voiced & ref_voiced

    # Voicing accuracy (correct voiced + correct unvoiced) / T
    voicing_acc = np.mean((pred_voiced == ref_voiced).astype(float))

    if not both_voiced.any():
        # No jointly-voiced frames: RPA/RCA/MACE/GrossErr are all undefined.
        # If ref has no voiced frames at all, GrossErr=1.0 would be wrong.
        return {
            "RPA": np.nan, "RCA": np.nan, "MACE": np.nan,
            "GrossErr": np.nan, "MedianCents": np.nan, "VoicingAcc": float(voicing_acc),
        }

    pred_cents = 1200.0 * np.log2(pred_f0[both_voiced] / 10.0)
    ref_cents  = 1200.0 * np.log2(ref_f0[both_voiced]  / 10.0)
    err_cents  = pred_cents - ref_cents

    rpa        = float(np.mean(np.abs(err_cents) <= RPA_TOL))
    rca        = float(np.mean(np.abs((err_cents + 600) % 1200 - 600) <= RPA_TOL))
    mace       = float(np.mean(np.abs(err_cents)))
    # GrossErr: fraction of ref-voiced frames where prediction is either unvoiced
    # OR pitch error exceeds threshold.  Denominator = all ref-voiced frames.
    ref_voiced_count = int(ref_voiced.sum())
    gross_err  = float((np.sum(np.abs(err_cents) > GROSS_TOL) + np.sum(ref_voiced & ~pred_voiced)) / ref_voiced_count)
    median_c   = float(np.median(np.abs(err_cents)))

    return {
        "RPA": rpa, "RCA": rca, "MACE": mace,
        "GrossErr": gross_err, "MedianCents": median_c,
        "VoicingAcc": float(voicing_acc),
    }


# ── Evaluation loop ────────────────────────────────────────────────────────────

def _technique_from_path(audio_path: "Path") -> str:
    """Extract GTSinger technique name from clip path.

    Path structure: .../English/<voice>/<technique>/<song>/<group>/<clip>.wav
    The technique directory is 3 levels above the wav file.
    """
    return audio_path.parents[2].name


def evaluate(
    gtsinger_root: str,
    model: E2E0,
    mel_extractor: MelSpectrogram,
    device: torch.device,
    max_clips: int | None = None,
) -> dict:
    """Evaluate over all GTSinger/English clips.

    Returns a dict with:
        'aggregate'     : metric_name → mean float (across all clips)
        'per_technique' : technique → metric_name → mean float
        'per_clip'      : list of {clip_id, technique, pred_f0, ref_f0, times, **metrics}
                          (pred/ref arrays stored for contour plotting)
    """
    clips = discover_clips(gtsinger_root)
    if max_clips:
        clips = clips[:max_clips]

    METRIC_KEYS = ["RPA", "RCA", "MACE", "GrossErr", "MedianCents", "VoicingAcc"]
    all_metrics: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    by_technique: dict[str, dict[str, list[float]]] = {}
    per_clip: list[dict] = []

    for clip in tqdm(clips, desc="Evaluating"):
        try:
            audio_np_2d, sr = sf.read(str(clip.audio_path), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio_np_2d.mean(axis=1))
            if sr != SAMPLE_RATE:
                waveform = TAF.resample(waveform, sr, SAMPLE_RATE)
            audio_np = waveform.numpy()

            pred_f0 = infer_clip(audio_np, model, mel_extractor, device)

            with open(clip.json_path) as fh:
                notes_json = json.load(fh)
            n_frames_ref = len(pred_f0) + 10
            ref_f0, _ = build_f0_track(notes_json, n_frames_ref)
            ref_f0 = ref_f0[:len(pred_f0)]

            m = pitch_metrics(pred_f0, ref_f0)
            technique = _technique_from_path(clip.audio_path)
        except Exception as e:
            print(f"  skip {clip.audio_path}: {e}")
            continue

        for k, v in m.items():
            if not np.isnan(v):
                all_metrics[k].append(v)
                by_technique.setdefault(technique, {k: [] for k in METRIC_KEYS})
                by_technique[technique].setdefault(k, []).append(v)

        hop_s = 160 / SAMPLE_RATE
        times = np.arange(len(pred_f0)) * hop_s
        per_clip.append({
            "clip_id":   clip.audio_path.stem,
            "technique": technique,
            "audio_path": str(clip.audio_path),
            "pred_f0":   pred_f0.tolist(),
            "ref_f0":    ref_f0.tolist(),
            "times":     times.tolist(),
            **{k: v for k, v in m.items() if not np.isnan(v)},
        })

    # Clip-weighted aggregate (each clip contributes equally regardless of length).
    # NOTE: train.py's val/rpa50 is frame-weighted — these numbers are not directly
    # comparable for variable-length clips. Use n_voiced_frames as a guide.
    aggregate = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
    aggregate["n_clips"] = len(per_clip)

    per_tech_summary = {
        tech: {k: float(np.mean(vals)) if vals else float("nan")
               for k, vals in metrics.items()}
        for tech, metrics in by_technique.items()
    }

    return {
        "aggregate":     aggregate,
        "per_technique": per_tech_summary,
        "per_clip":      per_clip,
    }


def print_summary(results: dict, tag: str = "") -> None:
    header = f"=== {tag} ===" if tag else "=== Results ==="
    print(header)
    agg = results.get("aggregate", results)  # backwards-compat with old dict format
    print(f"  Clips: {agg.get('n_clips', '?')}")
    for metric, value in agg.items():
        if metric == "n_clips":
            continue
        print(f"  {metric:14s}: {value:.4f}")
    per_tech = results.get("per_technique", {})
    if per_tech:
        print(f"  --- Per technique ---")
        for tech, metrics in sorted(per_tech.items()):
            rpa = metrics.get("RPA", float("nan"))
            mace = metrics.get("MACE", float("nan"))
            print(f"  {tech:30s}  RPA={rpa:.3f}  MACE={mace:.1f}¢")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RMVPE on GTSinger")
    parser.add_argument("--weights", required=True, help="Path to rmvpe.pt or fine-tuned .pth")
    parser.add_argument("--gtsinger-root", required=True)
    parser.add_argument("--tag",      default="", help="Label for this run in output")
    parser.add_argument("--output",   default=None, help="Save results JSON to this path")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Limit number of clips (for quick testing)")
    parser.add_argument("--device",  default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    model         = load_model(Path(args.weights), device)
    mel_extractor = build_mel_extractor(device)

    results = evaluate(args.gtsinger_root, model, mel_extractor, device,
                       max_clips=args.max_clips)
    print_summary(results, tag=args.tag or args.weights)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        results["tag"] = args.tag
        # per_clip arrays can be large — write full JSON
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
