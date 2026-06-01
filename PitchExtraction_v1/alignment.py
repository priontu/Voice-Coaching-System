from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PitchMetricResult:
    comparison_frames: int
    pitch_acc_50: float | None
    mace_cents: float | None
    pitch_rmse_cents: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_frames": self.comparison_frames,
            "pitch_acc_50": self.pitch_acc_50,
            "mace_cents": self.mace_cents,
            "pitch_rmse_cents": self.pitch_rmse_cents,
        }


@dataclass(frozen=True)
class LagAlignmentResult:
    lag_frames: int
    lag_sec: float
    shifted_ref_f0: torch.Tensor
    shifted_ref_voiced: torch.Tensor
    metrics: PitchMetricResult

    def to_dict(self) -> dict[str, Any]:
        data = self.metrics.to_dict()
        data.update(
            {
                "lag_frames": self.lag_frames,
                "lag_sec": self.lag_sec,
            }
        )
        return data


@dataclass(frozen=True)
class LocalLagWindow:
    start_frame: int
    end_frame: int
    lag_frames: int
    lag_sec: float
    metrics: PitchMetricResult

    def to_dict(self) -> dict[str, Any]:
        data = self.metrics.to_dict()
        data.update(
            {
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "lag_frames": self.lag_frames,
                "lag_sec": self.lag_sec,
            }
        )
        return data


@dataclass(frozen=True)
class LocalLagAlignmentResult:
    metrics: PitchMetricResult
    windows: list[LocalLagWindow]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metrics.to_dict(),
            "num_windows": len(self.windows),
            "lag_sec_values": [window.lag_sec for window in self.windows],
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass(frozen=True)
class DTWAlignmentResult:
    metrics: PitchMetricResult
    path_length: int
    frame_stride: int
    band_frames: int
    effective_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metrics.to_dict(),
            "path_length": self.path_length,
            "frame_stride": self.frame_stride,
            "band_frames": self.band_frames,
            "effective_frames": self.effective_frames,
        }


def cents_error(pred_f0: torch.Tensor, ref_f0: torch.Tensor) -> torch.Tensor:
    return 1200.0 * torch.log2(torch.clamp(pred_f0, min=1e-6) / torch.clamp(ref_f0, min=1e-6))


def pitch_metrics(
    pred_f0: torch.Tensor,
    pred_voiced: torch.Tensor,
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
) -> PitchMetricResult:
    pred_f0, pred_voiced, ref_f0, ref_voiced = _trim_to_common_length(
        pred_f0.float().cpu(),
        pred_voiced.bool().cpu(),
        ref_f0.float().cpu(),
        ref_voiced.bool().cpu(),
    )
    both = pred_voiced & ref_voiced & (pred_f0 > 0) & (ref_f0 > 0)
    if not bool(both.any()):
        return PitchMetricResult(
            comparison_frames=0,
            pitch_acc_50=None,
            mace_cents=None,
            pitch_rmse_cents=None,
        )

    cents = cents_error(pred_f0[both], ref_f0[both])
    return _metrics_from_cents(cents)


def estimate_global_lag(
    pred_f0: torch.Tensor,
    pred_voiced: torch.Tensor,
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
    *,
    max_lag_frames: int,
    frame_seconds: float,
    min_overlap_frames: int = 20,
) -> LagAlignmentResult:
    pred_f0, pred_voiced, ref_f0, ref_voiced = _trim_to_common_length(
        pred_f0.float().cpu(),
        pred_voiced.bool().cpu(),
        ref_f0.float().cpu(),
        ref_voiced.bool().cpu(),
    )
    best_lag = 0
    best_error = float("inf")

    for lag in range(-max_lag_frames, max_lag_frames + 1):
        shifted_f0, shifted_voiced = shift_reference(ref_f0, ref_voiced, lag)
        both = pred_voiced & shifted_voiced & (pred_f0 > 0) & (shifted_f0 > 0)
        if int(both.sum().item()) < min_overlap_frames:
            continue
        error = float(cents_error(pred_f0[both], shifted_f0[both]).abs().median().item())
        if error < best_error:
            best_error = error
            best_lag = lag

    shifted_f0, shifted_voiced = shift_reference(ref_f0, ref_voiced, best_lag)
    metrics = pitch_metrics(pred_f0, pred_voiced, shifted_f0, shifted_voiced)
    return LagAlignmentResult(
        lag_frames=best_lag,
        lag_sec=best_lag * frame_seconds,
        shifted_ref_f0=shifted_f0,
        shifted_ref_voiced=shifted_voiced,
        metrics=metrics,
    )


def shift_reference(
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
    lag_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_f0 = ref_f0.float().cpu()
    ref_voiced = ref_voiced.bool().cpu()
    shifted_f0 = torch.zeros_like(ref_f0)
    shifted_voiced = torch.zeros_like(ref_voiced)
    if lag_frames == 0:
        return ref_f0.clone(), ref_voiced.clone()
    if abs(lag_frames) >= ref_f0.numel():
        return shifted_f0, shifted_voiced
    if lag_frames > 0:
        shifted_f0[lag_frames:] = ref_f0[:-lag_frames]
        shifted_voiced[lag_frames:] = ref_voiced[:-lag_frames]
    else:
        shift = -lag_frames
        shifted_f0[:-shift] = ref_f0[shift:]
        shifted_voiced[:-shift] = ref_voiced[shift:]
    return shifted_f0, shifted_voiced


def local_lag_alignment(
    pred_f0: torch.Tensor,
    pred_voiced: torch.Tensor,
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
    *,
    window_frames: int,
    hop_frames: int,
    max_lag_frames: int,
    frame_seconds: float,
    min_overlap_frames: int = 20,
) -> LocalLagAlignmentResult:
    pred_f0, pred_voiced, ref_f0, ref_voiced = _trim_to_common_length(
        pred_f0.float().cpu(),
        pred_voiced.bool().cpu(),
        ref_f0.float().cpu(),
        ref_voiced.bool().cpu(),
    )
    num_frames = pred_f0.numel()
    windows: list[LocalLagWindow] = []
    cents_parts: list[torch.Tensor] = []

    for start, end in iter_windows(num_frames, window_frames, hop_frames):
        local = estimate_global_lag(
            pred_f0[start:end],
            pred_voiced[start:end],
            ref_f0[start:end],
            ref_voiced[start:end],
            max_lag_frames=max_lag_frames,
            frame_seconds=frame_seconds,
            min_overlap_frames=min_overlap_frames,
        )
        both = (
            pred_voiced[start:end]
            & local.shifted_ref_voiced
            & (pred_f0[start:end] > 0)
            & (local.shifted_ref_f0 > 0)
        )
        if bool(both.any()):
            cents_parts.append(cents_error(pred_f0[start:end][both], local.shifted_ref_f0[both]))
        windows.append(
            LocalLagWindow(
                start_frame=start,
                end_frame=end,
                lag_frames=local.lag_frames,
                lag_sec=local.lag_sec,
                metrics=local.metrics,
            )
        )

    return LocalLagAlignmentResult(metrics=_metrics_from_parts(cents_parts), windows=windows)


def iter_windows(num_frames: int, window_frames: int, hop_frames: int) -> list[tuple[int, int]]:
    if num_frames <= 0:
        return []
    window_frames = max(1, min(window_frames, num_frames))
    hop_frames = max(1, hop_frames)
    windows = []
    start = 0
    while start < num_frames:
        end = min(start + window_frames, num_frames)
        windows.append((start, end))
        if end == num_frames:
            break
        start += hop_frames
    return windows


def dtw_pitch_alignment(
    pred_f0: torch.Tensor,
    pred_voiced: torch.Tensor,
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
    *,
    max_warp_frames: int,
    max_dtw_frames: int = 2500,
    gap_penalty_cents: float = 100.0,
) -> DTWAlignmentResult:
    pred_f0, pred_voiced, ref_f0, ref_voiced = _trim_to_common_length(
        pred_f0.float().cpu(),
        pred_voiced.bool().cpu(),
        ref_f0.float().cpu(),
        ref_voiced.bool().cpu(),
    )
    original_frames = pred_f0.numel()
    if original_frames == 0:
        return DTWAlignmentResult(_metrics_from_parts([]), 0, 1, max_warp_frames, 0)

    stride = max(1, int(torch.ceil(torch.tensor(original_frames / max_dtw_frames)).item()))
    pred_f0 = pred_f0[::stride]
    pred_voiced = pred_voiced[::stride]
    ref_f0 = ref_f0[::stride]
    ref_voiced = ref_voiced[::stride]
    n = int(pred_f0.numel())
    m = int(ref_f0.numel())
    band = max(1, int(torch.ceil(torch.tensor(max_warp_frames / stride)).item()))

    inf = torch.tensor(float("inf"), dtype=torch.float32)
    dp = torch.full((n + 1, m + 1), inf, dtype=torch.float32)
    move = torch.zeros((n + 1, m + 1), dtype=torch.uint8)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_min = max(1, i - band)
        j_max = min(m, i + band)
        for j in range(j_min, j_max + 1):
            candidates = torch.stack((dp[i - 1, j - 1], dp[i - 1, j] + gap_penalty_cents, dp[i, j - 1] + gap_penalty_cents))
            best_cost, best_move = candidates.min(dim=0)
            dp[i, j] = best_cost + _frame_cost(pred_f0[i - 1], pred_voiced[i - 1], ref_f0[j - 1], ref_voiced[j - 1])
            move[i, j] = best_move.to(torch.uint8) + 1

    path_pairs = _backtrace_dtw(move, n, m)
    cents_parts = []
    for i, j in path_pairs:
        if pred_voiced[i] and ref_voiced[j] and pred_f0[i] > 0 and ref_f0[j] > 0:
            cents_parts.append(cents_error(pred_f0[i].view(1), ref_f0[j].view(1)))

    return DTWAlignmentResult(
        metrics=_metrics_from_parts(cents_parts),
        path_length=len(path_pairs),
        frame_stride=stride,
        band_frames=band * stride,
        effective_frames=n,
    )


def _frame_cost(pred_f0: torch.Tensor, pred_voiced: torch.Tensor, ref_f0: torch.Tensor, ref_voiced: torch.Tensor) -> torch.Tensor:
    if bool(pred_voiced) and bool(ref_voiced) and float(pred_f0) > 0.0 and float(ref_f0) > 0.0:
        return cents_error(pred_f0.view(1), ref_f0.view(1)).abs()[0]
    if bool(pred_voiced) == bool(ref_voiced):
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.tensor(600.0, dtype=torch.float32)


def _backtrace_dtw(move: torch.Tensor, n: int, m: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        step = int(move[i, j].item())
        if step == 1:
            i -= 1
            j -= 1
        elif step == 2:
            i -= 1
        elif step == 3:
            j -= 1
        else:
            break
    pairs.reverse()
    return pairs


def _metrics_from_parts(cents_parts: list[torch.Tensor]) -> PitchMetricResult:
    if not cents_parts:
        return PitchMetricResult(0, None, None, None)
    cents = torch.cat([part.flatten().float().cpu() for part in cents_parts if part.numel()])
    if cents.numel() == 0:
        return PitchMetricResult(0, None, None, None)
    return _metrics_from_cents(cents)


def _metrics_from_cents(cents: torch.Tensor) -> PitchMetricResult:
    return PitchMetricResult(
        comparison_frames=int(cents.numel()),
        pitch_acc_50=float((cents.abs() <= 50.0).float().mean().item()),
        mace_cents=float(cents.abs().mean().item()),
        pitch_rmse_cents=float(torch.sqrt(cents.square().mean()).item()),
    )


def _trim_to_common_length(
    pred_f0: torch.Tensor,
    pred_voiced: torch.Tensor,
    ref_f0: torch.Tensor,
    ref_voiced: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = min(pred_f0.numel(), pred_voiced.numel(), ref_f0.numel(), ref_voiced.numel())
    return pred_f0[:length], pred_voiced[:length], ref_f0[:length], ref_voiced[:length]
