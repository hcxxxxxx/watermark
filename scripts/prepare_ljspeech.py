#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the expected LJSpeech layout.")
    parser.add_argument("--root", default="data/LJSpeech-1.1")
    args = parser.parse_args()

    root = Path(args.root)
    metadata = root / "metadata.csv"
    wav_dir = root / "wavs"
    if metadata.exists() and wav_dir.exists():
        wav_count = len(list(wav_dir.glob("*.wav")))
        print(f"OK: found {wav_count} wav files under {root}")
        return

    print("LJSpeech is not in the expected layout.")
    print(f"Place it at: {root}")
    print("Expected files:")
    print(f"  {metadata}")
    print(f"  {wav_dir}/LJ001-0001.wav")


if __name__ == "__main__":
    main()
