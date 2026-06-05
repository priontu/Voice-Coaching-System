"""
reference/textgrid_parser.py - Praat TextGrid annotation parser.

Converts a Praat TextGrid file into lists of ReferencePhoneme and
ReferenceWord objects. Timestamps are already in seconds in the TextGrid
format, so no tempo conversion is required.

Supported backends (in order of preference):
  1. praatio  (pip install praatio)
  2. textgrid (pip install textgrid)
  3. Built-in plain-text parser (no dependencies — handles standard TextGrid
     files produced by MFA, Praat, and most forced-alignment tools).

Key behaviours:
  - Tier names are configurable; missing tiers are handled gracefully.
  - Empty intervals (label == "" or "<SIL>" or "SIL") are skipped by default.
  - Word tier back-annotates phoneme_indices on each ReferenceWord.
  - Silence labels can be preserved with skip_silence=False.

Usage:
    phonemes, words = parse_textgrid("annotation.TextGrid")
    phonemes, words = parse_textgrid(
        "annotation.TextGrid",
        phoneme_tier="phones",
        word_tier="words",
    )
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from utils.types import ReferencePhoneme, ReferenceWord
from reference.phoneme_normalization import normalize_to_ipa

logger = logging.getLogger(__name__)

# Labels treated as silence / empty and skipped by default
_SILENCE_LABELS = {"", "SIL", "<SIL>", "sp", "spn", "sil", "<eps>"}


# ---------------------------------------------------------------------------
# Built-in plain-text TextGrid parser (no external deps)
# ---------------------------------------------------------------------------

def _parse_textgrid_plain(text: str) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Minimal TextGrid parser that covers the common interval-tier format.

    Returns:
        Dict mapping tier name → list of (start_time, end_time, label) tuples.
    """
    tiers: Dict[str, List[Tuple[float, float, str]]] = {}

    # Split into tier blocks.  Each block starts with "item [N]:" or "item [N] :"
    blocks = re.split(r'\bitem\s*\[\d+\]\s*:', text)

    for block in blocks[1:]:  # skip preamble before first item
        # Tier name
        name_match = re.search(r'name\s*=\s*"([^"]*)"', block)
        if not name_match:
            continue
        tier_name = name_match.group(1)

        # Only handle IntervalTier
        class_match = re.search(r'class\s*=\s*"([^"]*)"', block)
        if not class_match or class_match.group(1) != "IntervalTier":
            continue

        intervals: List[Tuple[float, float, str]] = []

        # Extract intervals
        for interval_match in re.finditer(
            r'intervals\s*\[\d+\]\s*:.*?'
            r'xmin\s*=\s*([\d.eE+\-]+).*?'
            r'xmax\s*=\s*([\d.eE+\-]+).*?'
            r'text\s*=\s*"([^"]*)"',
            block,
            re.DOTALL,
        ):
            start = float(interval_match.group(1))
            end = float(interval_match.group(2))
            label = interval_match.group(3).strip()
            intervals.append((start, end, label))

        tiers[tier_name] = intervals

    return tiers


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

def _load_tiers(path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    """Load all interval tiers from a TextGrid file using the best available backend."""

    # Try praatio first
    try:
        from praatio import textgrid as praatio_tg  # type: ignore
        tg = praatio_tg.openTextgrid(str(path), includeEmptyIntervals=True)
        tiers: Dict[str, List[Tuple[float, float, str]]] = {}
        for tier_name in tg.tierNames:
            tier = tg.getTier(tier_name)
            # praatio returns entries as (start, end, label)
            tiers[tier_name] = [(float(e.start), float(e.end), e.label) for e in tier.entries]
        logger.debug("[textgrid_parser] Loaded with praatio: %s", path.name)
        return tiers
    except (ImportError, Exception):
        pass

    # Try textgrid package
    try:
        import textgrid as tg_pkg  # type: ignore
        tg = tg_pkg.TextGrid.fromFile(str(path))
        tiers = {}
        for tier in tg.tiers:
            if hasattr(tier, "intervals"):
                tiers[tier.name] = [
                    (float(iv.minTime), float(iv.maxTime), iv.mark or "")
                    for iv in tier.intervals
                ]
        logger.debug("[textgrid_parser] Loaded with textgrid package: %s", path.name)
        return tiers
    except (ImportError, Exception):
        pass

    # Fallback: plain-text parser
    logger.debug("[textgrid_parser] Using built-in plain-text parser: %s", path.name)
    text = path.read_text(encoding="utf-8", errors="replace")
    return _parse_textgrid_plain(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_textgrid(
    path: Union[str, Path],
    phoneme_tier: str = "phonemes",
    word_tier: str = "words",
    skip_silence: bool = True,
    silence_labels: Optional[set] = None,
) -> Tuple[List[ReferencePhoneme], List[ReferenceWord]]:
    """
    Parse a Praat TextGrid file into phoneme and word annotation lists.

    Args:
        path:           Path to a .TextGrid file.
        phoneme_tier:   Name of the phoneme/phone interval tier.
        word_tier:      Name of the word interval tier.
        skip_silence:   If True, intervals whose label is in silence_labels
                        are excluded from the output.
        silence_labels: Set of label strings treated as silence.
                        Defaults to {_SILENCE_LABELS} if None.

    Returns:
        (phonemes, words) — lists of ReferencePhoneme and ReferenceWord.
        Either list may be empty if the corresponding tier is absent.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TextGrid file not found: {path}")

    sil = silence_labels if silence_labels is not None else _SILENCE_LABELS
    logger.info("[textgrid_parser] Parsing: %s", path.name)

    tiers = _load_tiers(path)

    # ── Available tier names for debugging ──────────────────────────────────
    available = list(tiers.keys())
    logger.debug("[textgrid_parser] Available tiers: %s", available)

    # ── Phoneme tier ────────────────────────────────────────────────────────
    phonemes: List[ReferencePhoneme] = []
    phoneme_intervals = tiers.get(phoneme_tier, [])
    if not phoneme_intervals:
        # Try common alternative names
        for alt in ("phones", "phone", "phoneme", "Phoneme", "Phone"):
            if alt in tiers:
                phoneme_intervals = tiers[alt]
                logger.debug("[textgrid_parser] Using tier '%s' for phonemes", alt)
                break

    for idx, (start, end, label) in enumerate(phoneme_intervals):
        if skip_silence and label in sil:
            continue
        if end <= start:
            continue
        phonemes.append(ReferencePhoneme(
            phoneme=normalize_to_ipa(label),
            start_time=round(start, 6),
            end_time=round(end, 6),
            phoneme_idx=len(phonemes),
        ))

    # ── Word tier ───────────────────────────────────────────────────────────
    words: List[ReferenceWord] = []
    word_intervals = tiers.get(word_tier, [])
    if not word_intervals:
        for alt in ("word", "Word", "Words", "orthography"):
            if alt in tiers:
                word_intervals = tiers[alt]
                logger.debug("[textgrid_parser] Using tier '%s' for words", alt)
                break

    for w_idx, (start, end, label) in enumerate(word_intervals):
        if skip_silence and label in sil:
            continue
        if end <= start:
            continue

        # Back-annotate phoneme indices that fall within this word's span
        p_indices: List[int] = []
        for p in phonemes:
            if p.start_time >= start and p.end_time <= end + 1e-6:
                p_indices.append(p.phoneme_idx)  # type: ignore[arg-type]
                p.word_idx = len(words)

        words.append(ReferenceWord(
            text=label,
            start_time=round(start, 6),
            end_time=round(end, 6),
            phoneme_indices=p_indices,
            word_idx=len(words),
        ))

    logger.info(
        "[textgrid_parser] Parsed %d phonemes, %d words",
        len(phonemes), len(words),
    )
    return phonemes, words
