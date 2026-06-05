from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ReferenceInterval(Protocol):
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class BoundaryMatch:
    predicted_phoneme: str
    reference_phoneme: str
    predicted_start_sec: float
    predicted_end_sec: float
    reference_start_sec: float
    reference_end_sec: float
    start_error_sec: float
    end_error_sec: float
    boundary_error_sec: float


def compare_phoneme_boundaries(predicted: list[dict], reference: list[ReferenceInterval]) -> dict:
    ref = [item for item in reference if is_scored_phone(item.text)]
    count = min(len(predicted), len(ref))
    matches: list[BoundaryMatch] = []
    for index in range(count):
        pred = predicted[index]
        ref_item = ref[index]
        start_error = abs(float(pred["start_sec"]) - ref_item.start_sec)
        end_error = abs(float(pred["end_sec"]) - ref_item.end_sec)
        matches.append(
            BoundaryMatch(
                predicted_phoneme=str(pred["phoneme"]),
                reference_phoneme=ref_item.text,
                predicted_start_sec=float(pred["start_sec"]),
                predicted_end_sec=float(pred["end_sec"]),
                reference_start_sec=ref_item.start_sec,
                reference_end_sec=ref_item.end_sec,
                start_error_sec=start_error,
                end_error_sec=end_error,
                boundary_error_sec=(start_error + end_error) / 2.0,
            )
        )

    boundary_errors = [item.boundary_error_sec for item in matches]
    start_errors = [item.start_error_sec for item in matches]
    end_errors = [item.end_error_sec for item in matches]
    phone_matches = [
        normalize_phone(item.predicted_phoneme) == normalize_phone(item.reference_phoneme)
        for item in matches
    ]
    return {
        "predicted_count": len(predicted),
        "reference_count": len(ref),
        "aligned_count": len(matches),
        "phoneme_sequence_exact_rate": mean_bool(phone_matches),
        "start_mae_sec": mean(start_errors),
        "end_mae_sec": mean(end_errors),
        "boundary_mae_sec": mean(boundary_errors),
        "matches": [item.__dict__ for item in matches],
    }


def normalize_phone(value: str) -> str:
    value = value.lower().strip().replace("ː", "").replace(":", "")
    return "".join(ch for ch in value if not ch.isdigit())


def is_scored_phone(value: str) -> bool:
    normalized = normalize_phone(value)
    if not normalized:
        return False
    return normalized not in {"<sp>", "<ap>", "sp", "ap", "sil", "silence", "pau", "pause"}


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
