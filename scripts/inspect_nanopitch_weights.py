#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_coach.pitch.nanopitch_extractor import NanoPitchExtractor, resolve_nanopitch_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or smoke-test a NanoPitch checkpoint.")
    parser.add_argument("--checkpoint", default="priontu_chowdhury")
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decoder", choices=["offline", "realtime", "argmax"], default="offline")
    args = parser.parse_args()

    checkpoint = resolve_nanopitch_checkpoint(args.checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {checkpoint}")
    print(f"keys: {sorted(ckpt.keys()) if isinstance(ckpt, dict) else 'state_dict'}")
    print(f"model_kwargs: {ckpt.get('model_kwargs') if isinstance(ckpt, dict) else {}}")

    if args.wav is None:
        return

    extractor = NanoPitchExtractor(
        checkpoint_path=checkpoint,
        decoder=args.decoder,
        device=args.device,
    )
    pred = extractor.predict(args.wav)
    print(f"frames: {len(pred['f0_hz'])}")
    print(f"voiced_frames: {int(pred['voiced'].sum())}")
    print(f"f0_minmax: {float(pred['f0_hz'].min()):.2f}, {float(pred['f0_hz'].max()):.2f}")


if __name__ == "__main__":
    main()
