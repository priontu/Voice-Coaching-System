from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchLabelConfig:
    onset_sigma_sec: float = 0.03
    offset_sigma_sec: float = 0.03
    phoneme_sigma_sec: float = 0.02


def build_reference_f0_and_voicing(
    frame_times_sec: torch.Tensor,
    note_starts_sec: torch.Tensor,
    note_ends_sec: torch.Tensor,
    note_frequencies_hz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    f0 = torch.zeros_like(frame_times_sec)
    voicing = torch.zeros_like(frame_times_sec)
    if note_starts_sec.numel() == 0:
        return f0, voicing

    active = (frame_times_sec[:, None] >= note_starts_sec[None, :]) & (
        frame_times_sec[:, None] < note_ends_sec[None, :]
    )
    has_note = active.any(dim=1)
    first_note = active.float().argmax(dim=1)
    f0[has_note] = note_frequencies_hz[first_note[has_note]]
    voicing[has_note] = 1.0
    return f0, voicing


def build_event_target(
    frame_times_sec: torch.Tensor,
    event_times_sec: torch.Tensor,
    sigma_sec: float,
) -> torch.Tensor:
    if event_times_sec.numel() == 0:
        return torch.zeros_like(frame_times_sec)
    distances = frame_times_sec[:, None] - event_times_sec[None, :]
    gaussian = torch.exp(-(distances.square()) / (2.0 * sigma_sec**2))
    return gaussian.amax(dim=1)


def build_training_labels(
    frame_times_sec: torch.Tensor,
    note_starts_sec: torch.Tensor,
    note_ends_sec: torch.Tensor,
    note_frequencies_hz: torch.Tensor,
    phoneme_boundaries_sec: torch.Tensor,
    config: TorchLabelConfig,
) -> dict[str, torch.Tensor]:
    f0, voicing = build_reference_f0_and_voicing(
        frame_times_sec,
        note_starts_sec,
        note_ends_sec,
        note_frequencies_hz,
    )
    return {
        "f0": f0,
        "voicing": voicing,
        "onset": build_event_target(frame_times_sec, note_starts_sec, config.onset_sigma_sec),
        "offset": build_event_target(frame_times_sec, note_ends_sec, config.offset_sigma_sec),
        "phoneme_boundary": build_event_target(
            frame_times_sec,
            phoneme_boundaries_sec,
            config.phoneme_sigma_sec,
        ),
    }


def pad_label_dicts(label_dicts: list[dict[str, torch.Tensor]], max_frames: int) -> dict[str, torch.Tensor]:
    keys = label_dicts[0].keys() if label_dicts else []
    padded: dict[str, torch.Tensor] = {}
    for key in keys:
        values = []
        for labels in label_dicts:
            value = labels[key]
            values.append(torch.nn.functional.pad(value, (0, max_frames - value.numel())))
        padded[key] = torch.stack(values, dim=0)
    return padded
