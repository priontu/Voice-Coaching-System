from __future__ import annotations

import re
from dataclasses import asdict, dataclass


AST_TOKEN_RE = re.compile(r"<(?:P_\d+|NOTE(?:_DOT)?_\d+|BPM_\d+|SIL)>")
PITCH_RE = re.compile(r"<P_(\d+)>")
NOTE_RE = re.compile(r"<(NOTE(?:_DOT)?_\d+)>")
BPM_RE = re.compile(r"<BPM_(\d+)>")


@dataclass(frozen=True)
class VocalParseOutput:
    raw_text: str
    lyrics_text: str
    score_text: str
    pitch_tokens: list[int]
    note_tokens: list[str]
    bpm: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_vocalparse_text(text: str) -> VocalParseOutput:
    """Parse VocalParse's autoregressive AST token string."""

    parts = text.split("<|file_sep|>", maxsplit=1)
    if len(parts) == 2:
        lyrics_text = parts[0].strip()
        score_text = parts[1].strip()
    else:
        score_text = text.strip()
        lyrics_text = strip_ast_tokens(score_text)

    pitch_tokens = [int(value) for value in PITCH_RE.findall(score_text)]
    note_tokens = NOTE_RE.findall(score_text)
    bpm_matches = BPM_RE.findall(score_text)
    bpm = int(bpm_matches[-1]) if bpm_matches else None

    return VocalParseOutput(
        raw_text=text,
        lyrics_text=lyrics_text,
        score_text=score_text,
        pitch_tokens=pitch_tokens,
        note_tokens=note_tokens,
        bpm=bpm,
    )


def strip_ast_tokens(text: str) -> str:
    return AST_TOKEN_RE.sub("", text).replace("<|file_sep|>", "").strip()

