"""
scripts/validate_environment.py - Pre-flight environment check for VocalCoach.

Run from the VocalCoach root:
    py scripts/validate_environment.py

Exit code 0 = all required packages present.
Exit code 1 = one or more required packages missing.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Package matrix
# ---------------------------------------------------------------------------

# (import_name, pip_name, required)
REQUIRED: List[Tuple[str, str]] = [
    ("torch",          "torch"),
    ("torchaudio",     "torchaudio"),
    ("numpy",          "numpy"),
    ("librosa",        "librosa"),
    ("yaml",           "pyyaml"),
    ("matplotlib",     "matplotlib"),
]

OPTIONAL: List[Tuple[str, str]] = [
    ("music21",        "music21"),
    ("praatio",        "praatio"),
    ("torchcrepe",     "torchcrepe"),
    ("transformers",   "transformers"),
    ("scipy",          "scipy"),
    ("soundfile",      "soundfile"),
    ("pytest",         "pytest"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def _ok(msg: str)   -> str: return f"  {GREEN}[PASS]{RESET} {msg}"
def _fail(msg: str) -> str: return f"  {RED}[FAIL]{RESET} {msg}"
def _warn(msg: str) -> str: return f"  {YELLOW}[WARN]{RESET} {msg}"
def _info(msg: str) -> str: return f"  {'':6} {msg}"


def _check_import(import_name: str, pip_name: str, required: bool) -> bool:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        print(_ok(f"{pip_name} {ver}"))
        return True
    except ImportError:
        if required:
            print(_fail(f"{pip_name} not found  →  pip install {pip_name}"))
        else:
            print(_warn(f"{pip_name} not found (optional)  →  pip install {pip_name}"))
        return not required   # optional missing = non-fatal


def _check_python() -> bool:
    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}.{sys.version_info[2]}"
    if major == 3 and minor >= 9:
        print(_ok(f"Python {ver}"))
        return True
    else:
        print(_fail(f"Python {ver} — requires 3.9+"))
        return False


def _check_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(_ok(f"CUDA available — {name} ({mem} GB)"))
        else:
            print(_warn("CUDA not available — will use CPU (inference will be slower)"))
    except ImportError:
        pass   # torch already reported as missing


def _check_ffmpeg() -> None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            first_line = result.stdout.splitlines()[0]
            print(_ok(f"ffmpeg — {first_line[:60]}"))
        else:
            print(_warn("ffmpeg found but returned non-zero — check your ffmpeg install"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(_warn(
            "ffmpeg not found in PATH — audio loading may fail for MP3/FLAC files\n"
            "  Install: https://ffmpeg.org/download.html  (add to PATH)"
        ))


def _check_configs() -> bool:
    configs_dir = Path(__file__).parent.parent / "configs"
    required_configs = ["pipeline.yaml", "system.yaml", "pitch.yaml"]
    all_ok = True
    for fname in required_configs:
        cfg_path = configs_dir / fname
        if cfg_path.exists():
            print(_ok(f"configs/{fname}"))
        else:
            print(_fail(f"configs/{fname} missing"))
            all_ok = False
    return all_ok


def _check_directories() -> None:
    root = Path(__file__).parent.parent
    for d in ("checkpoints", "outputs", "samples", "references"):
        p = root / d
        if p.exists():
            print(_ok(f"{d}/  (exists)"))
        else:
            print(_warn(f"{d}/  not found — will be created on first use"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n" + "=" * 56)
    print("  VocalCoach Environment Validation")
    print("=" * 56)

    failures = 0

    # Python version
    print("\n--- Python ---")
    if not _check_python():
        failures += 1

    # Required packages
    print("\n--- Required packages ---")
    for import_name, pip_name in REQUIRED:
        if not _check_import(import_name, pip_name, required=True):
            failures += 1

    # Optional packages
    print("\n--- Optional packages ---")
    for import_name, pip_name in OPTIONAL:
        _check_import(import_name, pip_name, required=False)

    # GPU / CUDA
    print("\n--- GPU ---")
    _check_cuda()

    # System tools
    print("\n--- System tools ---")
    _check_ffmpeg()

    # Config files
    print("\n--- Config files ---")
    if not _check_configs():
        failures += 1

    # Directory structure
    print("\n--- Directories ---")
    _check_directories()

    # Summary
    print("\n" + "=" * 56)
    if failures == 0:
        print(f"  {GREEN}All checks passed. Environment is ready.{RESET}")
    else:
        print(f"  {RED}{failures} check(s) failed. Fix the issues above before running.{RESET}")
        print("\n  Quick fix:")
        print("    pip install torch torchaudio numpy librosa pyyaml matplotlib")
    print("=" * 56 + "\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
