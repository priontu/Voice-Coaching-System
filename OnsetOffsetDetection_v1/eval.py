from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class EventDetectionConfig:
    onset_threshold: float = 0.5
    offset_threshold: float = 0.5
    min_distance_sec: float = 0.05
    tolerance_sec: float = 0.05


def extract_events(
    probs: torch.Tensor,
    times_sec: torch.Tensor,
    threshold: float,
    min_distance_sec: float,
) -> list[float]:
    if probs.numel() == 0:
        return []
    frame_seconds = estimate_frame_seconds(times_sec)
    min_distance_frames = max(1, round(min_distance_sec / frame_seconds))
    peaks = []
    for index in range(probs.numel()):
        value = float(probs[index].item())
        if value < threshold:
            continue
        left = max(0, index - min_distance_frames)
        right = min(probs.numel(), index + min_distance_frames + 1)
        if value >= float(probs[left:right].max().item()):
            if peaks and index - peaks[-1] < min_distance_frames:
                if value > float(probs[peaks[-1]].item()):
                    peaks[-1] = index
            else:
                peaks.append(index)
    return [float(times_sec[index].item()) for index in peaks]


def event_metrics(
    predicted: list[float],
    reference: list[float],
    tolerance_sec: float,
) -> dict[str, Any]:
    matches = greedy_match_events(predicted, reference, tolerance_sec)
    precision = len(matches) / len(predicted) if predicted else None
    recall = len(matches) / len(reference) if reference else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    errors = [abs(predicted[pred_i] - reference[ref_i]) for pred_i, ref_i in matches]
    return {
        "predicted_count": len(predicted),
        "reference_count": len(reference),
        "matched_count": len(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae_sec": float(sum(errors) / len(errors)) if errors else None,
    }


def greedy_match_events(predicted: list[float], reference: list[float], tolerance_sec: float) -> list[tuple[int, int]]:
    candidates = []
    for pred_i, pred_time in enumerate(predicted):
        for ref_i, ref_time in enumerate(reference):
            error = abs(pred_time - ref_time)
            if error <= tolerance_sec:
                candidates.append((error, pred_i, ref_i))
    candidates.sort()
    used_pred = set()
    used_ref = set()
    matches = []
    for _, pred_i, ref_i in candidates:
        if pred_i in used_pred or ref_i in used_ref:
            continue
        used_pred.add(pred_i)
        used_ref.add(ref_i)
        matches.append((pred_i, ref_i))
    return matches


def reference_events(notes: list[Any]) -> tuple[list[float], list[float]]:
    return [float(note.start_sec) for note in notes], [float(note.end_sec) for note in notes]


def estimate_frame_seconds(times_sec: torch.Tensor) -> float:
    if times_sec.numel() >= 2:
        diffs = times_sec[1:] - times_sec[:-1]
        positive = diffs[diffs > 0]
        if positive.numel():
            return float(positive.median().item())
    return 0.01
