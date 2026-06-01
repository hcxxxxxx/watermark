#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_relmel_ljspeech import (
    build_codec,
    build_relmel_config,
    info_payload_bits,
    load_yaml,
    payload_from_id,
    result_row,
)
from melshield.attacks import build_attacks
from melshield.datasets import iter_ljspeech, load_audio
from melshield.mel import MelConfig, MelFrontend
from melshield.metrics import compare_audio
from melshield.relmel import RelMelMark
from melshield.vocoders import build_vocoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RelMel clean-reference quantization and storage variants."
    )
    parser.add_argument("--config", default="configs/relmel_hifigan.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attacks", nargs="+", default=["none", "noise20", "noise10", "noise5"])
    parser.add_argument(
        "--reference-variants",
        nargs="+",
        default=["float32", "float16", "uint8", "uint6", "uint4", "uint2", "band_uint8", "band_uint4", "band_uint2"],
    )
    parser.add_argument("--vocoder", default=None, choices=["mel", "griffinlim", "hifigan", "command"])
    parser.add_argument("--vocoder-checkpoint", default=None)
    parser.add_argument("--vocoder-config", default=None)
    parser.add_argument("--vocoder-command", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--band", default=None)
    parser.add_argument("--payload-bits", type=int, default=None)
    parser.add_argument("--info-bits", type=int, default=None)
    parser.add_argument("--ecc-repeat", type=int, default=1)
    parser.add_argument("--no-ecc-interleave", action="store_true")
    parser.add_argument("--block-frames", type=int, default=None)
    parser.add_argument("--block-stride", type=int, default=None)
    parser.add_argument("--min-block-frames", type=int, default=None)
    parser.add_argument("--bits-per-block", type=int, default=None)
    parser.add_argument("--pair-bins", type=int, default=None)
    parser.add_argument("--pair-candidates", type=int, default=None)
    parser.add_argument("--detector-mode", choices=["plain", "boundary"], default=None)
    parser.add_argument("--mask-floor", type=float, default=None)
    parser.add_argument("--energy-gamma", type=float, default=None)
    parser.add_argument("--boundary-margin", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root or cfg["paths"]["data_root"])
    mel_config = MelConfig(**cfg["mel"])
    relmel_config = build_relmel_config(cfg, args)
    codec = build_codec(args)
    info_bits = info_payload_bits(relmel_config, codec)
    frontend = MelFrontend(mel_config, device=args.device)
    relmel = RelMelMark(relmel_config)

    vocoder_name = args.vocoder or cfg["vocoder"]["name"]
    if vocoder_name == "mel":
        raise ValueError("Reference compression evaluation expects a waveform vocoder.")
    vocoder = build_vocoder(
        vocoder_name,
        mel_config=mel_config,
        device=args.device,
        checkpoint=args.vocoder_checkpoint or cfg["vocoder"].get("checkpoint"),
        config=args.vocoder_config or cfg["vocoder"].get("config"),
        command_template=args.vocoder_command or cfg["vocoder"].get("command"),
        sample_rate=mel_config.sample_rate,
    )
    attack_fns = build_attacks(args.attacks)

    rows: list[dict[str, Any]] = []
    for item in iter_ljspeech(data_root, limit=args.limit, sample_mode=args.sample_mode, seed=args.seed):
        waveform, sample_rate = load_audio(item.wav_path)
        bundle = frontend.waveform_to_normalized_logmel(waveform, sample_rate)
        payload = payload_from_id(relmel_config, item.utterance_id, info_bits)
        message = codec.encode(payload)
        watermarked_norm, reference = relmel.embed(
            clean_mel=bundle.normalized,
            message=message,
            utterance_id=item.utterance_id,
            norm_stats=bundle.stats,
            mel_config=mel_config,
        )
        clean_log_mel = torch.from_numpy(bundle.log_mel)
        wm_log_mel = torch.from_numpy(frontend.denormalize(watermarked_norm, bundle.stats))
        benchmark_audio = vocoder.synthesize(clean_log_mel)
        watermarked_audio = vocoder.synthesize(wm_log_mel)
        vocoder_sr = int(vocoder.sample_rate)

        compressed_refs = {
            name: compress_reference(reference, name)
            for name in args.reference_variants
        }
        for attack_name, attack_fn in attack_fns.items():
            attacked = attack_fn(watermarked_audio, vocoder_sr)
            bm_metrics = compare_audio(
                benchmark_audio,
                vocoder_sr,
                attacked.waveform,
                attacked.sample_rate,
            )
            for variant_name, (variant_ref, storage_bytes) in compressed_refs.items():
                detected = frontend.waveform_to_normalized_logmel(
                    attacked.waveform,
                    attacked.sample_rate,
                    stats=variant_ref.norm_stats,
                )
                extracted = relmel.extract(detected.normalized, variant_ref)
                row = result_row(
                    item.utterance_id,
                    attack_name,
                    extracted,
                    relmel_config,
                    codec,
                    payload,
                )
                row.update(
                    {
                        "reference_variant": variant_name,
                        "reference_bytes": storage_bytes,
                        "reference_kb": storage_bytes / 1024.0,
                        "pesq_bm": _fmt(bm_metrics.pesq),
                        "stoi_bm": _fmt(bm_metrics.stoi),
                    }
                )
                rows.append(row)
        print(f"processed {item.utterance_id}")

    write_outputs(output_dir, rows, cfg, mel_config, relmel_config, args)
    print(f"wrote reference-compression results to {output_dir}")


def compress_reference(reference: Any, name: str) -> tuple[Any, int]:
    clean = np.asarray(reference.clean_mel, dtype=np.float32)
    c_min, c_max = reference.relmel_config.band
    if name == "float32":
        return reference, int(clean.size * 4)
    if name == "float16":
        return replace(reference, clean_mel=clean.astype(np.float16).astype(np.float32)), int(clean.size * 2)
    if name.startswith("uint") and name[4:].isdigit():
        bits = int(name[4:])
        quantized = quantize_unit(clean, bits)
        return replace(reference, clean_mel=quantized), int(math.ceil(clean.size * bits / 8.0))
    if name.startswith("band_uint") and name[len("band_uint") :].isdigit():
        bits = int(name[len("band_uint") :])
        band = quantize_unit(clean[c_min:c_max], bits)
        proxy = np.zeros_like(clean, dtype=np.float32)
        proxy[c_min:c_max] = band
        return replace(reference, clean_mel=proxy), int(math.ceil(band.size * bits / 8.0))
    raise ValueError(f"Unknown reference variant: {name}")


def quantize_unit(value: np.ndarray, bits: int) -> np.ndarray:
    if bits < 1:
        raise ValueError(f"Quantization bits must be >= 1, got {bits}")
    levels = float((1 << bits) - 1)
    return (np.round(np.clip(value, 0.0, 1.0) * levels) / levels).astype(np.float32)


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    mel_config: MelConfig,
    relmel_config: Any,
    args: argparse.Namespace,
) -> None:
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['reference_variant']}|{row['attack']}", []).append(row)

    def mean(rows_: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows_ if row.get(key) not in ("", None)]
        return None if not values else float(np.mean(values))

    summary_rows = {}
    for name, group_rows in sorted(grouped.items()):
        variant, attack = name.split("|", 1)
        summary_rows.setdefault(variant, {})[attack] = {
            "mean_bit_acc": mean(group_rows, "bit_acc"),
            "verification_rate": float(np.mean([bool(row["verified"]) for row in group_rows])),
            "mean_confidence": mean(group_rows, "confidence"),
            "mean_reference_kb": mean(group_rows, "reference_kb"),
            "mean_pesq_bm": mean(group_rows, "pesq_bm"),
            "mean_stoi_bm": mean(group_rows, "stoi_bm"),
        }

    payload = {
        "args": vars(args),
        "num_rows": len(rows),
        "reference_variants": summary_rows,
        "mel_config": mel_config.to_dict(),
        "relmel_config": relmel_config.to_dict(),
        "run_config": cfg,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
