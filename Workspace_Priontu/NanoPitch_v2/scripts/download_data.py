#!/usr/bin/env python3
"""
Download NanoPitch pre-extracted data from Hugging Face.

Default source:
  https://huggingface.co/datasets/smulelabs/NanoPitch-PreExtract

This script downloads the `.npz` feature files into a local output directory
(`data/` by default) and verifies that the expected files are present.
"""

import argparse
import sys
from pathlib import Path


DEFAULT_REPO_ID = "smulelabs/NanoPitch-PreExtract"
REQUIRED_FILES = ("clean.npz", "noise.npz", "test.npz")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download NanoPitch pre-extracted dataset from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision to download (default: main)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Local directory to store downloaded files (default: data)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Missing dependency: huggingface_hub\n"
            "Install it with:\n"
            "  pip install huggingface_hub"
        )
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{args.repo_id}' (revision: {args.revision})")
    print(f"Destination: {output_dir}")

    # Download only the pre-extracted feature files needed by training/eval.
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(output_dir),
        allow_patterns=["*.npz"],
    )

    missing = [name for name in REQUIRED_FILES if not (output_dir / name).exists()]
    if missing:
        print("Download finished, but some required files are missing:")
        for name in missing:
            print(f"  - {name}")
        return 2

    print("Download complete. Found required files:")
    for name in REQUIRED_FILES:
        size_mb = (output_dir / name).stat().st_size / (1024 * 1024)
        print(f"  - {name} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
