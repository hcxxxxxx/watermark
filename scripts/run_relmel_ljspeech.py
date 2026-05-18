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
from melshield.relmel import RelMelConfig, RelMelMark
from melshield.vocoders import build_vocoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run block-wise relative-energy Mel watermarking.")
    parser.add_argument("--config", default="configs/relmel_hifigan.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vocoder", default=None, choices=["mel", "griffinlim", "hifigan", "command"])
    parser.add_argument("--vocoder-checkpoint", default=None)
    parser.add_argument("--vocoder-config", default=None)
    parser.add_argument("--vocoder-command", default=None)
    parser.add_argument("--attacks", nargs="+", default=None)
    parser.add_argument("--save-audio", action="store_true")

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--payload-bits", type=int, default=None)
    parser.add_argument("--block-frames", type=int, default=None)
    parser.add_argument("--block-stride", type=int, default=None)
    parser.add_argument("--bits-per-block", type=int, default=None)
    parser.add_argument("--pair-bins", type=int, default=None)
    parser.add_argument("--mask-floor", type=float, default=None)
    parser.add_argument("--energy-gamma", type=float, default=None)
    parser.add_argument("--boundary-margin", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_root = Path(args.data_root or cfg["paths"]["data_root"])
    output_dir = Path(args.output_dir or cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "refs").mkdir(exist_ok=True)
    if args.save_audio:
        (output_dir / "audio").mkdir(exist_ok=True)

    mel_config = MelConfig(**cfg["mel"])
    relmel_config = build_relmel_config(cfg, args)
    vocoder_name = args.vocoder or cfg["vocoder"]["name"]
    attacks = args.attacks or cfg["evaluation"]["attacks"]
    if vocoder_name == "mel":
        attacks = ["none"]

    frontend = MelFrontend(mel_config, device=args.device)
    relmel = RelMelMark(relmel_config)
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

    for item in iter_ljspeech(
        data_root,
        limit=args.limit or cfg["evaluation"].get("limit"),
        sample_mode=args.sample_mode,
        seed=args.seed,
    ):
        waveform, sample_rate = load_audio(item.wav_path)
        bundle = frontend.waveform_to_normalized_logmel(waveform, sample_rate)
        message = relmel.message_from_id(item.utterance_id)
        watermarked_norm, reference = relmel.embed(
            clean_mel=bundle.normalized,
            message=message,
            utterance_id=item.utterance_id,
            norm_stats=bundle.stats,
            mel_config=mel_config,
        )
        reference.save(output_dir / "refs" / f"{item.utterance_id}.npz")

        if vocoder_name == "mel":
            result = relmel.extract(watermarked_norm, reference)
            rows.append(result_row(item.utterance_id, "none", result) | empty_metrics())
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
            result = relmel.extract(detected.normalized, reference)
            gt_metrics = compare_audio(waveform, sample_rate, attacked.waveform, attacked.sample_rate)
            bm_metrics = compare_audio(
                benchmark_audio,
                vocoder_sr,
                attacked.waveform,
                attacked.sample_rate,
            )
            row = result_row(item.utterance_id, attack_name, result)
            row.update(
                {
                    "pesq_gt": _fmt(gt_metrics.pesq),
                    "stoi_gt": _fmt(gt_metrics.stoi),
                    "pesq_bm": _fmt(bm_metrics.pesq),
                    "stoi_bm": _fmt(bm_metrics.stoi),
                }
            )
            rows.append(row)
        print(f"processed {item.utterance_id}")

    write_outputs(output_dir, rows, cfg, mel_config, relmel_config)
    print(f"wrote results to {output_dir}")


def build_relmel_config(cfg: dict[str, Any], args: argparse.Namespace) -> RelMelConfig:
    data = dict(cfg["relmel"])
    data["band"] = tuple(data["band"])
    overrides = {
        "alpha": args.alpha,
        "threshold": args.threshold,
        "payload_bits": args.payload_bits,
        "block_frames": args.block_frames,
        "block_stride": args.block_stride,
        "bits_per_block": args.bits_per_block,
        "pair_bins": args.pair_bins,
        "mask_floor": args.mask_floor,
        "energy_gamma": args.energy_gamma,
        "boundary_margin": args.boundary_margin,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    return RelMelConfig(**data)


def result_row(utterance_id: str, attack: str, result: Any) -> dict[str, Any]:
    votes = result.votes[result.votes > 0]
    return {
        "utterance_id": utterance_id,
        "attack": attack,
        "bit_acc": result.bit_accuracy,
        "verified": result.verified,
        "confidence": float(np.mean(np.abs(result.scores))),
        "min_votes": int(votes.min()) if votes.size else 0,
        "mean_votes": float(votes.mean()) if votes.size else 0.0,
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    mel_config: MelConfig,
    relmel_config: RelMelConfig,
) -> None:
    if rows:
        with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["attack"]].append(row)

    def mean_key(attack: str, key: str) -> float | None:
        values = [float(row[key]) for row in grouped[attack] if row.get(key) not in ("", None)]
        if not values:
            return None
        return float(np.mean(values))

    summary = {
        "mean_bit_acc": {
            attack: mean_key(attack, "bit_acc") for attack in sorted(grouped)
        },
        "verification_rate": {
            attack: float(np.mean([bool(row["verified"]) for row in attack_rows]))
            for attack, attack_rows in sorted(grouped.items())
        },
        "mean_confidence": {
            attack: mean_key(attack, "confidence") for attack in sorted(grouped)
        },
        "mean_min_votes": {
            attack: mean_key(attack, "min_votes") for attack in sorted(grouped)
        },
        "mean_votes": {
            attack: mean_key(attack, "mean_votes") for attack in sorted(grouped)
        },
        "mean_pesq_bm": {
            attack: mean_key(attack, "pesq_bm") for attack in sorted(grouped)
        },
        "mean_stoi_bm": {
            attack: mean_key(attack, "stoi_bm") for attack in sorted(grouped)
        },
        "num_rows": len(rows),
        "mel_config": mel_config.to_dict(),
        "relmel_config": relmel_config.to_dict(),
        "run_config": cfg,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def empty_metrics() -> dict[str, str]:
    return {"pesq_gt": "", "stoi_gt": "", "pesq_bm": "", "stoi_bm": ""}


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
