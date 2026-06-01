from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Recording:
    recording_id: str
    wav_path: Path
    musicxml_path: Path
    textgrid_path: Path | None = None
    song_id: str | None = None

    def to_json_dict(self) -> dict:
        data = asdict(self)
        for key in ("wav_path", "musicxml_path", "textgrid_path"):
            if data[key] is not None:
                data[key] = str(data[key])
        return data


def load_manifest(path: str | Path) -> list[Recording]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text())
    elif path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported manifest type: {path.suffix}")

    base_dir = path.parent
    return [_recording_from_row(row, base_dir) for row in rows]


def save_manifest(recordings: Iterable[Recording], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [recording.to_json_dict() for recording in recordings]
    path.write_text(json.dumps(rows, indent=2))


def discover_recordings(dataset_root: str | Path) -> list[Recording]:
    """Discover triplets by matching WAV, MusicXML, and TextGrid stems.

    Matching uses each file's relative path without suffix. This matters for
    datasets like GTSinger where many directories contain ``0000.wav``.
    """

    root = Path(dataset_root)
    wavs = {_relative_stem(p, root): p for p in root.rglob("*.wav")}
    musicxmls = {
        _relative_stem(p, root): p
        for pattern in ("*.musicxml", "*.xml", "*.mxl")
        for p in root.rglob(pattern)
    }
    textgrids = {
        _relative_stem(p, root): p
        for pattern in ("*.TextGrid", "*.textgrid")
        for p in root.rglob(pattern)
    }

    recordings: list[Recording] = []
    for relative_stem, wav_path in sorted(wavs.items()):
        musicxml_path = musicxmls.get(relative_stem)
        if musicxml_path is None:
            continue
        recording_id = _sanitize_id(relative_stem)
        recordings.append(
            Recording(
                recording_id=recording_id,
                song_id=_sanitize_id(str(Path(relative_stem).parent)),
                wav_path=wav_path,
                musicxml_path=musicxml_path,
                textgrid_path=textgrids.get(relative_stem),
            )
        )
    return recordings


def _relative_stem(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix(""))


def _sanitize_id(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _recording_from_row(row: dict, base_dir: Path) -> Recording:
    def resolve(value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else (base_dir / path)

    wav_path = resolve(row.get("wav_path"))
    musicxml_path = resolve(row.get("musicxml_path"))
    if wav_path is None or musicxml_path is None:
        raise ValueError("Each manifest row must include wav_path and musicxml_path")

    return Recording(
        recording_id=row.get("recording_id") or wav_path.stem,
        song_id=row.get("song_id") or musicxml_path.stem,
        wav_path=wav_path,
        musicxml_path=musicxml_path,
        textgrid_path=resolve(row.get("textgrid_path")),
    )
