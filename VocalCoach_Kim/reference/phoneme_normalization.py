"""
reference/phoneme_normalization.py
Normalize ARPABET phoneme labels (with optional stress digits) to eSpeak-NG IPA.

Stress digits (0 = unstressed, 1 = primary, 2 = secondary) are stripped before
lookup. Two phonemes are stress-sensitive and get distinct IPA targets:
  AH0 → ə  (schwa)     AH1/AH2 → ʌ
  ER0 → ɚ  (r-colored)  ER1/ER2 → ɝ

Input that is not recognized as ARPABET (i.e. already IPA or unknown) is
returned lowercased and unchanged, so the function is safe to call on any label.
"""

from __future__ import annotations

from typing import Optional

# Standard ARPABET → eSpeak-NG IPA.
# Keys are bare ARPABET phonemes (no stress digit).
# Stress-sensitive exceptions (AH, ER) are handled in normalize_to_ipa().
_ARPABET_TO_IPA: dict[str, str] = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",    # stressed default; AH0 → ə handled separately
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B":  "b",
    "CH": "tʃ",
    "D":  "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",    # stressed default; ER0 → ɚ handled separately
    "EY": "eɪ",
    "F":  "f",
    "G":  "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "iː",
    "JH": "dʒ",
    "K":  "k",
    "L":  "l",
    "M":  "m",
    "N":  "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P":  "p",
    "R":  "ɹ",
    "S":  "s",
    "SH": "ʃ",
    "T":  "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "uː",
    "V":  "v",
    "W":  "w",
    "Y":  "j",
    "Z":  "z",
    "ZH": "ʒ",
}

_ARPABET_KEYS: frozenset[str] = frozenset(_ARPABET_TO_IPA)


def is_arpabet(label: str) -> bool:
    """Return True if label looks like an ARPABET phoneme (with or without stress digit)."""
    bare = label.rstrip("012") if label and label[-1].isdigit() else label
    return bare.upper() in _ARPABET_KEYS


def normalize_to_ipa(label: str) -> str:
    """
    Convert an ARPABET label to its eSpeak-NG IPA equivalent.

    - Stress digits are stripped before lookup (AY1 → AY, IH0 → IH).
    - AH0 → ə, AH1/AH2 → ʌ.
    - ER0 → ɚ, ER1/ER2 → ɝ.
    - Labels not in the ARPABET table are returned lowercased (IPA pass-through).
    - Silence / empty labels are returned unchanged.
    """
    if not label:
        return label

    # Extract trailing stress digit if present
    stress: Optional[int] = None
    if label[-1].isdigit():
        stress = int(label[-1])
        bare = label[:-1].upper()
    else:
        bare = label.upper()

    # Stress-sensitive exceptions
    if bare == "AH":
        return "ə" if stress == 0 else "ʌ"
    if bare == "ER":
        return "ɚ" if stress == 0 else "ɝ"

    ipa = _ARPABET_TO_IPA.get(bare)
    if ipa is not None:
        return ipa

    # Not ARPABET — assume already IPA or other notation; return as-is (lowercased)
    return label.lower()
