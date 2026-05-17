#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from melshield.attacks import build_attacks
from melshield.datasets import iter_ljspeech, load_audio, save_audio
from melshield.mel import MelConfig, MelFrontend
from melshield.metrics import compare_audio
from melshield.vocoders import build_vocoder
from melshield.watermark import MelShield, MelShieldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MelShield LJSpeech reproduction.")
    parser.add_argument("--config", default="configs/melshield_ljspeech.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--vocoder",
        default=None,
        choices=["mel", "griffinlim", "hifigan", "command"],
        help="Use 'mel' for a no-waveform watermark sanity check.",
    )
    parser.add_argument("--vocoder-checkpoint", default=None)
    parser.add_argument("--vocoder-config", default=None)
    parser.add_argument("--vocoder-command", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--payload-bits", type=int, default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--attacks", nargs="+", default=None)
    parser.add_argument("--save-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    data_root = Path(args.data_root or cfg["paths"]["data_root"])
    output_dir = Path(args.output_dir or cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "refs").mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)

    mel_config = MelConfig(**cfg["mel"])
    wm_cfg_data = dict(cfg["watermark"])
    wm_cfg_data["band"] = tuple(wm_cfg_data["band"])
    if args.alpha is not None:
        wm_cfg_data["alpha"] = args.alpha
    if args.payload_bits is not None:
        wm_cfg_data["payload_bits"] = args.payload_bits
    if args.key is not None:
        wm_cfg_data["key"] = args.key
    if args.threshold is not None:
        wm_cfg_data["threshold"] = args.threshold
    wm_config = MelShieldConfig(**wm_cfg_data)

    vocoder_name = args.vocoder or cfg["vocoder"]["name"]
    attacks = args.attacks or cfg["evaluation"]["attacks"]
    if vocoder_name == "mel":
        attacks = ["none"]

    frontend = MelFrontend(mel_config, device=args.device)
    shield = MelShield(wm_config)
    vocoder = None
    if vocoder_name != "mel":
        vocoder = build_vocoder(
            vocoder_name,
            mel_config=mel_config,
            device=args.device,
            checkpoint=args.vocoder_checkpoint or cfg["vocoder"].get("checkpoint"),
            config=args.vocoder_config or cfg["vocoder"].get("config"),
            command_template=args.vocoder_command or cfg["vocoder"].get("command"),
            sample_rate=mel_config.sample_rate,
        )

    attack_fns = build_attacks(attacks)
    rows: list[dict[str, Any]] = []

    for item in iter_ljspeech(data_root, limit=args.limit or cfg["evaluation"].get("limit")):
        waveform, sample_rate = load_audio(item.wav_path)
        bundle = frontend.waveform_to_normalized_logmel(waveform, sample_rate)
        message = shield.message_from_id(item.utterance_id)
        watermarked_norm, reference = shield.embed(
            clean_mel=bundle.normalized,
            message=message,
            utterance_id=item.utterance_id,
            norm_stats=bundle.stats,
            mel_config=mel_config,
        )
        reference.save(output_dir / "refs" / f"{item.utterance_id}.npz")

        if vocoder_name == "mel":
            result = shield.extract(watermarked_norm, reference)
            rows.append(
                {
                    "utterance_id": item.utterance_id,
                    "attack": "none",
                    "bit_acc": result.bit_accuracy,
                    "verified": result.verified,
                    "pesq_gt": "",
                    "stoi_gt": "",
                    "pesq_bm": "",
                    "stoi_bm": "",
                }
            )
            continue

        assert vocoder is not None
        clean_log_mel = torch.from_numpy(bundle.log_mel)
        wm_log_mel = torch.from_numpy(frontend.denormalize(watermarked_norm, bundle.stats))
        benchmark_audio = vocoder.synthesize(clean_log_mel)
        watermarked_audio = vocoder.synthesize(wm_log_mel)
        vocoder_sr = int(vocoder.sample_rate)

        if args.save_audio:
            save_audio(output_dir / "audio" / f"{item.utterance_id}_bm.wav", benchmark_audio, vocoder_sr)
            save_audio(output_dir / "audio" / f"{item.utterance_id}_wm.wav", watermarked_audio, vocoder_sr)

        for attack_name, attack_fn in attack_fns.items():
            attacked = attack_fn(watermarked_audio, vocoder_sr)
            detected = frontend.waveform_to_normalized_logmel(
                attacked.waveform,
                attacked.sample_rate,
                stats=reference.norm_stats,
            )
            result = shield.extract(detected.normalized, reference)

            gt_metrics = compare_audio(
                waveform,
                sample_rate,
                attacked.waveform,
                attacked.sample_rate,
            )
            bm_metrics = compare_audio(
                benchmark_audio,
                vocoder_sr,
                attacked.waveform,
                attacked.sample_rate,
            )
            rows.append(
                {
                    "utterance_id": item.utterance_id,
                    "attack": attack_name,
                    "bit_acc": result.bit_accuracy,
                    "verified": result.verified,
                    "pesq_gt": _fmt(gt_metrics.pesq),
                    "stoi_gt": _fmt(gt_metrics.stoi),
                    "pesq_bm": _fmt(bm_metrics.pesq),
                    "stoi_bm": _fmt(bm_metrics.stoi),
                }
            )
        print(f"processed {item.utterance_id}")

    write_results(output_dir, rows, config=cfg, wm_config=wm_config, mel_config=mel_config)
    print(f"wrote results to {output_dir}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    wm_config: MelShieldConfig,
    mel_config: MelConfig,
) -> None:
    csv_path = output_dir / "results.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[str, list[float]] = defaultdict(list)
    verified: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["bit_acc"] is not None:
            grouped[row["attack"]].append(float(row["bit_acc"]))
        if row["verified"] is not None:
            verified[row["attack"]].append(float(bool(row["verified"])))

    summary = {
        "mean_bit_acc": {k: float(np.mean(v)) for k, v in grouped.items()},
        "verification_rate": {k: float(np.mean(v)) for k, v in verified.items()},
        "num_rows": len(rows),
        "mel_config": mel_config.to_dict(),
        "watermark_config": wm_config.to_dict(),
        "run_config": config,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
