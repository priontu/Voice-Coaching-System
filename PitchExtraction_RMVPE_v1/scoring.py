from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RMVPECoachingScoreConfig:
    confidence_threshold: float = 0.35
    stable_trim_sec: float = 0.08
    min_note_voiced_ratio: float = 0.5


def apply_confidence(
    f0_hz: torch.Tensor,
    confidence: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    voiced = (confidence >= threshold) & (f0_hz > 0)
    return torch.where(voiced, f0_hz, torch.zeros_like(f0_hz)), voiced


def stable_note_mask(times_sec: torch.Tensor, notes: list[Any], trim_sec: float) -> torch.Tensor:
    mask = torch.zeros_like(times_sec, dtype=torch.bool)
    for note in notes:
        start = float(note.start_sec) + trim_sec
        end = float(note.end_sec) - trim_sec
        if end <= start:
            midpoint = (float(note.start_sec) + float(note.end_sec)) / 2.0
            start = midpoint
            end = midpoint
        mask |= (times_sec >= start) & (times_sec < end)
    return mask


def note_median_pitch_events(
    *,
    times_sec: torch.Tensor,
    f0_hz: torch.Tensor,
    confidence: torch.Tensor,
    notes: list[Any],
    config: RMVPECoachingScoreConfig = RMVPECoachingScoreConfig(),
) -> list[dict[str, float | int]]:
    thresholded_f0, voiced = apply_confidence(f0_hz, confidence, config.confidence_threshold)
    events: list[dict[str, float | int]] = []
    for index, note in enumerate(notes):
        start = float(note.start_sec) + config.stable_trim_sec
        end = float(note.end_sec) - config.stable_trim_sec
        if end <= start:
            start = float(note.start_sec)
            end = float(note.end_sec)
        mask = (times_sec >= start) & (times_sec < end)
        if not bool(mask.any()):
            continue

        voiced_values = thresholded_f0[mask & voiced & (thresholded_f0 > 0)]
        voiced_ratio = voiced_values.numel() / max(int(mask.sum().item()), 1)
        if voiced_ratio < config.min_note_voiced_ratio:
            continue

        pred_f0 = voiced_values.median()
        ref_f0 = torch.tensor(float(note.frequency_hz), dtype=torch.float32, device=pred_f0.device)
        cents = cents_error(pred_f0.view(1), ref_f0.view(1))[0]
        events.append(
            {
                "note_index": index,
                "note_start_sec": float(note.start_sec),
                "note_end_sec": float(note.end_sec),
                "stable_start_sec": float(start),
                "stable_end_sec": float(end),
                "target_midi": int(note.midi),
                "target_f0_hz": float(note.frequency_hz),
                "median_f0_hz": float(pred_f0.item()),
                "error_cents": float(cents.item()),
                "abs_error_cents": float(cents.abs().item()),
                "voiced_ratio": float(voiced_ratio),
                "voiced_frames": int(voiced_values.numel()),
                "stable_frames": int(mask.sum().item()),
            }
        )
    return events


def frame_cents(
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    ref_f0_hz: torch.Tensor,
    ref_voiced: torch.Tensor,
) -> torch.Tensor:
    both = voiced & ref_voiced & (f0_hz > 0) & (ref_f0_hz > 0)
    if not bool(both.any()):
        return torch.empty(0, dtype=torch.float32, device=f0_hz.device)
    return cents_error(f0_hz[both], ref_f0_hz[both])


def cents_error(pred_f0: torch.Tensor, ref_f0: torch.Tensor) -> torch.Tensor:
    return 1200.0 * torch.log2(torch.clamp(pred_f0, min=1e-6) / torch.clamp(ref_f0, min=1e-6))


def metrics_from_cents(cents: torch.Tensor) -> dict[str, float | int | None]:
    if cents.numel() == 0:
        return {
            "comparison_frames": 0,
            "pitch_acc_50": None,
            "pitch_acc_100": None,
            "mace_cents": None,
            "median_abs_cents": None,
            "pitch_rmse_cents": None,
        }
    abs_cents = cents.abs()
    return {
        "comparison_frames": int(cents.numel()),
        "pitch_acc_50": float((abs_cents <= 50.0).float().mean().item()),
        "pitch_acc_100": float((abs_cents <= 100.0).float().mean().item()),
        "mace_cents": float(abs_cents.mean().item()),
        "median_abs_cents": float(abs_cents.median().item()),
        "pitch_rmse_cents": float(torch.sqrt(cents.square().mean()).item()),
    }


def note_events_to_cents(events: list[dict[str, float | int]]) -> torch.Tensor:
    if not events:
        return torch.empty(0, dtype=torch.float32)
    return torch.tensor([float(event["error_cents"]) for event in events], dtype=torch.float32)
