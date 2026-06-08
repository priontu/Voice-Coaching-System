#!/usr/bin/env python3
"""
main.py — Voice Coaching System launcher.

Downloads model weights from HuggingFace (if not already present),
then starts the web frontend at http://localhost:8080.

Usage:
    python main.py                 # download weights (if needed) then launch UI
    python main.py --no-download   # skip download, launch UI immediately
    python main.py --download-only # download weights and exit without launching
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all resolved relative to this file so the script works from any CWD)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "pipeline" / "checkpoints"
FRONTEND_APP = ROOT / "frontend" / "app.py"

HF_REPO_ID = "WestDoor/voice-coaching-system-weights"
CHECKPOINT_FILES = ["best.ckpt", "best.pth"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _check_missing() -> list[str]:
    return [f for f in CHECKPOINT_FILES if not (CHECKPOINT_DIR / f).exists()]


def download_weights() -> None:
    missing = _check_missing()
    if not missing:
        print("[weights] All checkpoints already present — skipping download.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "[weights] ERROR: huggingface_hub is not installed.\n"
            "  Install it with:  pip install huggingface_hub\n"
            "  Then re-run:      python main.py\n"
            "  Or skip download: python main.py --no-download"
        )
        sys.exit(1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[weights] Downloading {len(missing)} file(s) from {HF_REPO_ID} ...")

    for filename in missing:
        print(f"  {filename} ...", end=" ", flush=True)
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=str(CHECKPOINT_DIR),
                local_dir_use_symlinks=False,
            )
            print("done.")
        except Exception as exc:
            print(f"FAILED\n  {exc}")
            print(
                "  Download failed. You can download manually from:\n"
                f"  https://huggingface.co/{HF_REPO_ID}\n"
                f"  and place the files in: {CHECKPOINT_DIR}"
            )
            sys.exit(1)

    print("[weights] All checkpoints ready.\n")


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_frontend() -> None:
    if not FRONTEND_APP.exists():
        print(f"[launch] ERROR: frontend not found at {FRONTEND_APP}")
        sys.exit(1)

    print("[launch] Starting Voice Coaching System ...")
    print("[launch] Open your browser at http://localhost:8080")
    print("[launch] Press Ctrl+C to stop.\n")

    try:
        subprocess.run([sys.executable, str(FRONTEND_APP)], check=False)
    except KeyboardInterrupt:
        print("\n[launch] Stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice Coaching System — download weights and launch the web UI."
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip weight download and launch the frontend immediately.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download weights and exit without launching the frontend.",
    )
    args = parser.parse_args()

    if not args.no_download:
        download_weights()

    if not args.download_only:
        launch_frontend()


if __name__ == "__main__":
    main()
