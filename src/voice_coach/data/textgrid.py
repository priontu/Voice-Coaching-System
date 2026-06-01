from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextGridInterval:
    tier: str
    start_sec: float
    end_sec: float
    text: str


def parse_textgrid(path: str | Path) -> list[TextGridInterval]:
    """Parse common Praat TextGrid interval tiers."""

    text = Path(path).read_text(errors="ignore")
    intervals: list[TextGridInterval] = []
    tier_blocks = re.split(r"(?=item \[\d+\]:)", text)

    for block in tier_blocks:
        name_match = re.search(r'name = "([^"]*)"', block)
        if not name_match:
            continue
        tier_name = name_match.group(1)
        for interval_block in re.split(r"(?=intervals \[\d+\]:)", block):
            xmin = re.search(r"xmin = ([0-9.eE+-]+)", interval_block)
            xmax = re.search(r"xmax = ([0-9.eE+-]+)", interval_block)
            label = re.search(r'text = "([^"]*)"', interval_block)
            if xmin and xmax and label:
                intervals.append(
                    TextGridInterval(
                        tier=tier_name,
                        start_sec=float(xmin.group(1)),
                        end_sec=float(xmax.group(1)),
                        text=label.group(1),
                    )
                )

    return intervals


def phoneme_intervals(
    path: str | Path,
    tier_names: tuple[str, ...] = ("phones", "phone", "phoneme", "phonemes"),
) -> list[TextGridInterval]:
    intervals = parse_textgrid(path)
    names = {name.lower() for name in tier_names}
    selected = [item for item in intervals if item.tier.lower() in names]
    if selected:
        return selected
    return intervals
