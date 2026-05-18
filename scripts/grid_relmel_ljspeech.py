#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from scripts.run_relmel_ljspeech import (
    build_relmel_config,
    load_yaml,
    write_outputs,
)
from melshield.attacks import build_attacks
from melshield.datasets import iter_ljspeech, load_audio
from melshield.mel import MelConfig, MelFrontend
from melshield.metrics import compare_audio
from melshield.relmel import RelMelConfig, RelMelMark
from melshield.vocoders import build_vocoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search RelMelMark and aggregate results.")
    parser.add_argument("--config", default="configs/relmel_hifigan.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attacks", nargs="+", default=["none", "noise20"])
    parser.add_argument("--vocoder", default=None, choices=["mel", "griffinlim", "hifigan", "command"])
    parser.add_argument("--vocoder-checkpoint", default=None)
    parser.add_argument("--vocoder-config", default=None)
    parser.add_argument("--vocoder-command", default=None)

    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.44])
    parser.add_argument("--mask-floor-grid", nargs="+", type=float, default=[0.15])
    parser.add_argument("--boundary-margin-grid", nargs="+", type=float, default=[0.03])
    parser.add_argument("--block-frames-grid", nargs="+", type=int, default=[16])
    parser.add_argument("--bits-per-block-grid", nargs="+", type=int, default=[4])
    parser.add_argument("--pair-bins-grid", nargs="+", type=int, default=[4])
    parser.add_argument("--energy-gamma-grid", nargs="+", type=float, default=[0.5])

    parser.add_argument("--quality-floor", type=float, default=3.5)
    parser.add_argument("--noise20-weight", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=0.25)
    parser.add_argument("--keep-candidate-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root or cfg["paths"]["data_root"])
    mel_config = MelConfig(**cfg["mel"])
    vocoder_name = args.vocoder or cfg["vocoder"]["name"]
    attacks = args.attacks
    if vocoder_name == "mel":
        attacks = ["none"]

    frontend = MelFrontend(mel_config, device=args.device)
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

    candidates = list(
        itertools.product(
            args.alpha_grid,
            args.mask_floor_grid,
            args.boundary_margin_grid,
            args.block_frames_grid,
            args.bits_per_block_grid,
            args.pair_bins_grid,
            args.energy_gamma_grid,
        )
    )
    aggregate_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for idx, values in enumerate(candidates, start=1):
        alpha, mask_floor, boundary_margin, block_frames, bits_per_block, pair_bins, energy_gamma = values
        relmel_config = relmel_config_from_values(
            cfg=cfg,
            alpha=alpha,
            mask_floor=mask_floor,
            boundary_margin=boundary_margin,
            block_frames=block_frames,
            bits_per_block=bits_per_block,
            pair_bins=pair_bins,
            energy_gamma=energy_gamma,
        )
        candidate_dir = output_dir / f"candidate_{idx:03d}"
        rows, summary = run_candidate(
            data_root=data_root,
            output_dir=candidate_dir,
            cfg=cfg,
            mel_config=mel_config,
            frontend=frontend,
            vocoder=vocoder,
            vocoder_name=vocoder_name,
            relmel_config=relmel_config,
            attacks=attacks,
            limit=args.limit,
            keep_outputs=args.keep_candidate_results,
        )
        objective = objective_from_summary(summary, args)
        flat = flatten_summary(
            candidate=idx,
            objective=objective,
            relmel_config=relmel_config,
            summary=summary,
            attacks=attacks,
        )
        aggregate_rows.append(flat)
        if best is None or objective > best["objective"]:
            best = flat
        print(f"candidate {idx}/{len(candidates)} objective={objective:.4f} {flat}")

    write_grid_outputs(output_dir, aggregate_rows, best, args, cfg)
    print(f"wrote grid results to {output_dir}")


def run_candidate(
    data_root: Path,
    output_dir: Path,
    cfg: dict[str, Any],
    mel_config: MelConfig,
    frontend: MelFrontend,
    vocoder: Any,
    vocoder_name: str,
    relmel_config: RelMelConfig,
    attacks: list[str],
    limit: int,
    keep_outputs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    relmel = RelMelMark(relmel_config)
    attack_fns = build_attacks(attacks)
    rows: list[dict[str, Any]] = []

    for item in iter_ljspeech(data_root, limit=limit):
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

        if vocoder_name == "mel":
            result = relmel.extract(watermarked_norm, reference)
            rows.append(result_row(item.utterance_id, "none", result, "", "", "", ""))
            continue

        clean_log_mel = torch.from_numpy(bundle.log_mel)
        wm_log_mel = torch.from_numpy(frontend.denormalize(watermarked_norm, bundle.stats))
        benchmark_audio = vocoder.synthesize(clean_log_mel)
        watermarked_audio = vocoder.synthesize(wm_log_mel)
        vocoder_sr = int(vocoder.sample_rate)

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
            rows.append(
                result_row(
                    item.utterance_id,
                    attack_name,
                    result,
                    _fmt(gt_metrics.pesq),
                    _fmt(gt_metrics.stoi),
                    _fmt(bm_metrics.pesq),
                    _fmt(bm_metrics.stoi),
                )
            )

    summary = summarize_rows(rows)
    if keep_outputs:
        write_outputs(output_dir, rows, cfg, mel_config, relmel_config)
    return rows, summary


def result_row(
    utterance_id: str,
    attack: str,
    result: Any,
    pesq_gt: str,
    stoi_gt: str,
    pesq_bm: str,
    stoi_bm: str,
) -> dict[str, Any]:
    votes = result.votes[result.votes > 0]
    return {
        "utterance_id": utterance_id,
        "attack": attack,
        "bit_acc": result.bit_accuracy,
        "verified": result.verified,
        "confidence": float(np.mean(np.abs(result.scores))),
        "min_votes": int(votes.min()) if votes.size else 0,
        "mean_votes": float(votes.mean()) if votes.size else 0.0,
        "pesq_gt": pesq_gt,
        "stoi_gt": stoi_gt,
        "pesq_bm": pesq_bm,
        "stoi_bm": stoi_bm,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["attack"], []).append(row)

    def mean_key(attack: str, key: str) -> float | None:
        values = [float(row[key]) for row in grouped[attack] if row.get(key) not in ("", None)]
        if not values:
            return None
        return float(np.mean(values))

    return {
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
        "mean_pesq_bm": {
            attack: mean_key(attack, "pesq_bm") for attack in sorted(grouped)
        },
        "mean_stoi_bm": {
            attack: mean_key(attack, "stoi_bm") for attack in sorted(grouped)
        },
        "num_rows": len(rows),
    }


def relmel_config_from_values(
    cfg: dict[str, Any],
    alpha: float,
    mask_floor: float,
    boundary_margin: float,
    block_frames: int,
    bits_per_block: int,
    pair_bins: int,
    energy_gamma: float,
) -> RelMelConfig:
    class Args:
        pass

    args = Args()
    args.alpha = alpha
    args.threshold = None
    args.payload_bits = None
    args.block_frames = block_frames
    args.bits_per_block = bits_per_block
    args.pair_bins = pair_bins
    args.mask_floor = mask_floor
    args.energy_gamma = energy_gamma
    args.boundary_margin = boundary_margin
    return build_relmel_config(cfg, args)  # type: ignore[arg-type]


def objective_from_summary(summary: dict[str, Any], args: argparse.Namespace) -> float:
    acc = summary["mean_bit_acc"]
    pesq = summary["mean_pesq_bm"]
    none_acc = float(acc.get("none") or 0.0)
    noise20_acc = float(acc.get("noise20") or none_acc)
    none_pesq = pesq.get("none")
    penalty = 0.0 if none_pesq is None else max(0.0, args.quality_floor - float(none_pesq))
    return none_acc + args.noise20_weight * noise20_acc - args.quality_weight * penalty


def flatten_summary(
    candidate: int,
    objective: float,
    relmel_config: RelMelConfig,
    summary: dict[str, Any],
    attacks: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate": candidate,
        "objective": objective,
        "alpha": relmel_config.alpha,
        "mask_floor": relmel_config.mask_floor,
        "boundary_margin": relmel_config.boundary_margin,
        "block_frames": relmel_config.block_frames,
        "bits_per_block": relmel_config.bits_per_block,
        "pair_bins": relmel_config.pair_bins,
        "energy_gamma": relmel_config.energy_gamma,
        "num_rows": summary["num_rows"],
    }
    for attack in attacks:
        row[f"{attack}_bit_acc"] = summary["mean_bit_acc"].get(attack)
        row[f"{attack}_verified"] = summary["verification_rate"].get(attack)
        row[f"{attack}_pesq_bm"] = summary["mean_pesq_bm"].get(attack)
        row[f"{attack}_stoi_bm"] = summary["mean_stoi_bm"].get(attack)
        row[f"{attack}_confidence"] = summary["mean_confidence"].get(attack)
    return row


def write_grid_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (output_dir / "grid_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "best": best,
        "rows": rows,
        "args": vars(args),
        "run_config": cfg,
    }
    with (output_dir / "grid_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with (output_dir / "best_candidate.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2, sort_keys=True)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
