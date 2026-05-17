#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from melshield.datasets import load_audio
from melshield.mel import MelFrontend
from melshield.watermark import MelShield, ReferenceRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one suspect wav against a saved MelShield reference.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--key", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    record = ReferenceRecord.load(Path(args.reference))
    frontend = MelFrontend(record.mel_config, device=args.device)
    shield = MelShield(record.wm_config)
    waveform, sample_rate = load_audio(args.audio)
    detected = frontend.waveform_to_normalized_logmel(
        waveform, sample_rate, stats=record.norm_stats
    )
    result = shield.extract(detected.normalized, record, key=args.key)
    decoded = "".join(str(int(bit)) for bit in result.decoded.tolist())
    print(f"bit_acc={result.bit_accuracy:.4f}")
    print(f"verified={result.verified}")
    print(f"decoded={decoded}")


if __name__ == "__main__":
    main()
