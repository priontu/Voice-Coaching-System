"""
GTSinger Dataset Preparation Script
=====================================
Walks a GTSinger directory, pairs every .wav file with its .json annotation,
and produces train / val / test manifest files ready for NoteDetectionDataset.

The GTSinger JSON is used DIRECTLY — no conversion step is needed.
dataset.py auto-detects the GTSinger format by inspecting the "note_start" key.

Usage
-----
    python prepare_gtsinger.py \\
        --gtsinger_root "path/to/gtsinger" \\
        --output_dir    "data/manifests" \\
        --language      English \\
        --technique     Breathy \\
        --group         Breathy_Group \\
        --train_ratio   0.8 \\
        --val_ratio     0.1 \\
        --seed          42

Examples
--------
# Use all English Breathy_Group samples:
    python prepare_gtsinger.py \\
        --gtsinger_root "C:/Users/kimhu/Documents/VSCode/MusicAI/VocalCoach/Phoneme Model/gtsinger" \\
        --output_dir data/manifests

# Use all techniques and all groups (full dataset):
    python prepare_gtsinger.py \\
        --gtsinger_root "..." \\
        --output_dir data/manifests \\
        --technique ALL \\
        --group ALL

Output
------
    data/manifests/
        train.json   — list of {"audio": "...", "label": "..."}
        val.json
        test.json
        dataset_stats.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GTSinger directory layout
# ─────────────────────────────────────────────────────────────────────────────
#
#   gtsinger/
#   └── {Language}/              e.g. English
#       └── {VoiceType}/         e.g. EN-Alto-1
#           └── {Technique}/     e.g. Breathy | Glissando | Mixed_Voice_and_Falsetto
#               └── {Song}/      e.g. "let it go"
#                   └── {Group}/ e.g. Breathy_Group | Control_Group | Paired_Speech_Group
#                       ├── 0000.wav
#                       ├── 0000.json    ← note timings (note_start / note_end)
#                       ├── 0000.musicxml
#                       └── 0000.TextGrid

KNOWN_TECHNIQUES = ["Breathy", "Glissando", "Mixed_Voice_and_Falsetto"]
KNOWN_GROUPS = ["Breathy_Group", "Control_Group", "Paired_Speech_Group"]


def discover_samples(
    gtsinger_root: Path,
    language: str = "English",
    techniques: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Walk a GTSinger directory and collect all (wav, json) pairs.

    Args:
        gtsinger_root: Root path of the GTSinger dataset.
        language:      Language subdirectory (e.g. "English").
        techniques:    Singing techniques to include, or None for all.
        groups:        Recording groups to include, or None for all.

    Returns:
        List of {"audio": str, "label": str, "meta": {...}} dicts.
    """
    lang_dir = gtsinger_root / language
    if not lang_dir.exists():
        raise FileNotFoundError(
            f"Language directory not found: {lang_dir}\n"
            f"Available: {[p.name for p in gtsinger_root.iterdir() if p.is_dir()]}"
        )

    samples: List[Dict] = []

    for voice_dir in sorted(lang_dir.iterdir()):
        if not voice_dir.is_dir():
            continue
        for technique_dir in sorted(voice_dir.iterdir()):
            if not technique_dir.is_dir():
                continue
            if techniques and technique_dir.name not in techniques:
                continue
            for song_dir in sorted(technique_dir.iterdir()):
                if not song_dir.is_dir():
                    continue
                for group_dir in sorted(song_dir.iterdir()):
                    if not group_dir.is_dir():
                        continue
                    if groups and group_dir.name not in groups:
                        continue
                    for wav_file in sorted(group_dir.glob("*.wav")):
                        json_file = wav_file.with_suffix(".json")
                        if not json_file.exists():
                            logger.warning("Missing JSON for %s — skipping.", wav_file)
                            continue
                        samples.append({
                            "audio": str(wav_file),
                            "label": str(json_file),
                            "meta": {
                                "language": language,
                                "voice": voice_dir.name,
                                "technique": technique_dir.name,
                                "song": song_dir.name,
                                "group": group_dir.name,
                                "stem": wav_file.stem,
                            },
                        })

    logger.info("Discovered %d samples total.", len(samples))
    return samples


def split_samples(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Shuffle and split samples into train / val / test sets.

    Splits are done at the song level to prevent data leakage between
    segments of the same song appearing in different splits.

    Args:
        samples:     Full list of sample dicts.
        train_ratio: Fraction for training.
        val_ratio:   Fraction for validation (remainder goes to test).
        seed:        Random seed for reproducibility.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    # Group by (voice, technique, song) to split at song level
    song_buckets: Dict[str, List[Dict]] = {}
    for s in samples:
        key = f"{s['meta']['voice']}/{s['meta']['technique']}/{s['meta']['song']}"
        song_buckets.setdefault(key, []).append(s)

    songs = sorted(song_buckets.keys())
    rng = random.Random(seed)
    rng.shuffle(songs)

    n_train = max(1, int(len(songs) * train_ratio))
    n_val = max(1, int(len(songs) * val_ratio))

    train_songs = songs[:n_train]
    val_songs = songs[n_train : n_train + n_val]
    test_songs = songs[n_train + n_val :]

    train = [s for k in train_songs for s in song_buckets[k]]
    val = [s for k in val_songs for s in song_buckets[k]]
    test = [s for k in test_songs for s in song_buckets[k]]

    logger.info(
        "Split: train=%d  val=%d  test=%d  (songs: %d/%d/%d)",
        len(train), len(val), len(test),
        len(train_songs), len(val_songs), len(test_songs),
    )
    return train, val, test


def write_manifest(samples: List[Dict], path: Path) -> None:
    """Write a manifest JSON — each entry has 'audio' and 'label' keys."""
    manifest = [{"audio": s["audio"], "label": s["label"]} for s in samples]
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %d entries → %s", len(manifest), path)


def write_stats(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
    path: Path,
) -> None:
    """Write a human-readable summary of the dataset split."""
    all_samples = train + val + test

    technique_counts: Dict[str, int] = {}
    group_counts: Dict[str, int] = {}
    song_counts: Dict[str, int] = {}

    for s in all_samples:
        m = s["meta"]
        technique_counts[m["technique"]] = technique_counts.get(m["technique"], 0) + 1
        group_counts[m["group"]] = group_counts.get(m["group"], 0) + 1
        song_counts[m["song"]] = song_counts.get(m["song"], 0) + 1

    lines = [
        "GTSinger Dataset Statistics",
        "=" * 40,
        f"Total samples : {len(all_samples)}",
        f"  Train       : {len(train)}",
        f"  Val         : {len(val)}",
        f"  Test        : {len(test)}",
        "",
        "Technique breakdown:",
    ]
    for k, v in sorted(technique_counts.items()):
        lines.append(f"  {k:<30} {v}")
    lines += ["", "Recording group breakdown:"]
    for k, v in sorted(group_counts.items()):
        lines.append(f"  {k:<30} {v}")
    lines += ["", "Song breakdown:"]
    for k, v in sorted(song_counts.items()):
        lines.append(f"  {k:<30} {v}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Stats written → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare GTSinger manifests for note onset/offset training"
    )
    parser.add_argument(
        "--gtsinger_root",
        required=True,
        help='Root path of the GTSinger dataset (the folder containing "English/", etc.)',
    )
    parser.add_argument(
        "--output_dir",
        default="data/manifests",
        help="Directory to write train.json / val.json / test.json",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Language subdirectory to use (default: English)",
    )
    parser.add_argument(
        "--technique",
        default="ALL",
        help=(
            "Singing technique to filter on, or ALL for every technique. "
            f"Known values: {', '.join(KNOWN_TECHNIQUES)}"
        ),
    )
    parser.add_argument(
        "--group",
        default="Breathy_Group",
        help=(
            "Recording group to include, or ALL for every group. "
            f"Known values: {', '.join(KNOWN_GROUPS)}"
        ),
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible splits"
    )
    args = parser.parse_args()

    gtsinger_root = Path(args.gtsinger_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    techniques = None if args.technique == "ALL" else [args.technique]
    groups = None if args.group == "ALL" else [args.group]

    samples = discover_samples(gtsinger_root, args.language, techniques, groups)

    if not samples:
        logger.error(
            "No samples found. Check --gtsinger_root, --language, --technique, --group."
        )
        return

    train, val, test = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)

    write_manifest(train, output_dir / "train.json")
    write_manifest(val, output_dir / "val.json")
    write_manifest(test, output_dir / "test.json")
    write_stats(train, val, test, output_dir / "dataset_stats.txt")

    logger.info("Done. Manifests written to: %s", output_dir)


if __name__ == "__main__":
    main()
