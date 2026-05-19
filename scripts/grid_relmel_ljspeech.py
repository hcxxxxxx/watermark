#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from scripts.run_relmel_ljspeech import (
    build_codec,
    build_relmel_config,
    info_payload_bits,
    load_yaml,
    payload_from_id,
    result_row as relmel_result_row,
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
    parser.add_argument(
        "--suite",
        choices=["custom", "relmel-audit"],
        default="custom",
        help="Use relmel-audit for a targeted multi-group RelMel search.",
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attacks", nargs="+", default=["none", "noise20"])
    parser.add_argument("--vocoder", default=None, choices=["mel", "griffinlim", "hifigan", "command"])
    parser.add_argument("--vocoder-checkpoint", default=None)
    parser.add_argument("--vocoder-config", default=None)
    parser.add_argument("--vocoder-command", default=None)
    parser.add_argument("--payload-bits", type=int, default=None)
    parser.add_argument("--info-bits", type=int, default=None)
    parser.add_argument("--ecc-repeat", type=int, default=1)
    parser.add_argument("--no-ecc-interleave", action="store_true")

    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.44])
    parser.add_argument("--band-grid", nargs="+", default=None)
    parser.add_argument("--mask-floor-grid", nargs="+", type=float, default=[0.15])
    parser.add_argument("--boundary-margin-grid", nargs="+", type=float, default=[0.03])
    parser.add_argument("--block-frames-grid", nargs="+", type=int, default=[16])
    parser.add_argument("--block-stride-grid", nargs="+", type=int, default=None)
    parser.add_argument("--min-block-frames-grid", nargs="+", type=int, default=None)
    parser.add_argument("--bits-per-block-grid", nargs="+", type=int, default=[4])
    parser.add_argument("--pair-bins-grid", nargs="+", type=int, default=[4])
    parser.add_argument("--energy-gamma-grid", nargs="+", type=float, default=[0.5])
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=None)
    parser.add_argument("--align-max-shift-grid", nargs="+", type=int, default=None)

    parser.add_argument("--quality-floor", type=float, default=3.5)
    parser.add_argument("--noise20-weight", type=float, default=1.0)
    parser.add_argument("--noise10-weight", type=float, default=0.5)
    parser.add_argument("--noise5-weight", type=float, default=0.25)
    parser.add_argument("--echo-weight", type=float, default=0.25)
    parser.add_argument("--quality-weight", type=float, default=0.25)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
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

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    base_relmel = build_relmel_config(
        cfg,
        argparse.Namespace(
            alpha=None,
            threshold=None,
            payload_bits=args.payload_bits,
            info_bits=args.info_bits,
            ecc_repeat=args.ecc_repeat,
            no_ecc_interleave=args.no_ecc_interleave,
            block_frames=None,
            block_stride=None,
            bits_per_block=None,
            pair_bins=None,
            mask_floor=None,
            energy_gamma=None,
            boundary_margin=None,
        ),
    )
    if args.suite == "relmel-audit":
        candidates = make_relmel_audit_suite(base_relmel)
    else:
        candidates = make_custom_grid(base_relmel, args)
    indexed_candidates = list(enumerate(candidates, start=1))
    total_candidates = len(indexed_candidates)
    if args.num_shards > 1:
        indexed_candidates = [
            (idx, candidate)
            for idx, candidate in indexed_candidates
            if (idx - 1) % args.num_shards == args.shard_index
        ]
        print(
            f"running shard {args.shard_index}/{args.num_shards}: "
            f"{len(indexed_candidates)} of {total_candidates} candidates"
        )
    aggregate_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    codec = build_codec(args)

    for local_idx, (idx, (group, relmel_config)) in enumerate(indexed_candidates, start=1):
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
            sample_mode=args.sample_mode,
            seed=args.seed,
            codec=codec,
            keep_outputs=args.keep_candidate_results,
        )
        objective = objective_from_summary(summary, args)
        flat = flatten_summary(
            candidate=idx,
            group=group,
            objective=objective,
            relmel_config=relmel_config,
            summary=summary,
            attacks=attacks,
        )
        aggregate_rows.append(flat)
        if best is None or objective > best["objective"]:
            best = flat
        none_acc = flat.get("none_bit_acc")
        noise20_acc = flat.get("noise20_bit_acc")
        noise10_acc = flat.get("noise10_bit_acc")
        noise5_acc = flat.get("noise5_bit_acc")
        none_pesq = flat.get("none_pesq_bm")
        print(
            f"candidate {idx}/{total_candidates} local={local_idx}/{len(indexed_candidates)} "
            f"group={group} objective={objective:.4f} alpha={relmel_config.alpha} "
            f"band={relmel_config.band[0]}:{relmel_config.band[1]} "
            f"none={none_acc} n20={noise20_acc} n10={noise10_acc} "
            f"n5={noise5_acc} pesq={none_pesq}"
        )

    write_grid_outputs(output_dir, aggregate_rows, best, args, cfg, total_candidates)
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
    sample_mode: str,
    seed: int,
    codec: Any,
    keep_outputs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if keep_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
    relmel = RelMelMark(relmel_config)
    info_bits = info_payload_bits(relmel_config, codec)
    attack_fns = build_attacks(attacks)
    rows: list[dict[str, Any]] = []

    for item in iter_ljspeech(data_root, limit=limit, sample_mode=sample_mode, seed=seed):
        waveform, sample_rate = load_audio(item.wav_path)
        bundle = frontend.waveform_to_normalized_logmel(waveform, sample_rate)
        payload_message = payload_from_id(relmel_config, item.utterance_id, info_bits)
        message = codec.encode(payload_message)
        watermarked_norm, reference = relmel.embed(
            clean_mel=bundle.normalized,
            message=message,
            utterance_id=item.utterance_id,
            norm_stats=bundle.stats,
            mel_config=mel_config,
        )

        if vocoder_name == "mel":
            result = relmel.extract(watermarked_norm, reference)
            rows.append(
                relmel_result_row(
                    item.utterance_id,
                    "none",
                    result,
                    relmel_config,
                    codec,
                    payload_message,
                )
                | empty_metrics()
            )
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
                    relmel_config,
                    codec,
                    payload_message,
                    _fmt(gt_metrics.pesq),
                    _fmt(gt_metrics.stoi),
                    _fmt(bm_metrics.pesq),
                    _fmt(bm_metrics.stoi),
                )
            )

    summary = summarize_rows(rows)
    summary["coding_config"] = {
        "payload_bits": info_bits,
        "code_bits": relmel_config.payload_bits,
        "ecc_repeat": codec.repeat,
        "interleave": codec.interleave,
    }
    if keep_outputs:
        write_outputs(output_dir, rows, cfg, mel_config, relmel_config)
    return rows, summary


def result_row(
    utterance_id: str,
    attack: str,
    result: Any,
    relmel_config: RelMelConfig,
    codec: Any,
    payload_message: np.ndarray,
    pesq_gt: str,
    stoi_gt: str,
    pesq_bm: str,
    stoi_bm: str,
) -> dict[str, Any]:
    row = relmel_result_row(
        utterance_id,
        attack,
        result,
        relmel_config,
        codec,
        payload_message,
    )
    row.update(
        {
        "pesq_gt": pesq_gt,
        "stoi_gt": stoi_gt,
        "pesq_bm": pesq_bm,
        "stoi_bm": stoi_bm,
        }
    )
    return row


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
        "mean_payload_bit_acc": {
            attack: mean_key(attack, "payload_bit_acc") for attack in sorted(grouped)
        },
        "payload_verification_rate": {
            attack: float(np.mean([bool(row["payload_verified"]) for row in attack_rows]))
            for attack, attack_rows in sorted(grouped.items())
        },
        "mean_payload_confidence": {
            attack: mean_key(attack, "payload_confidence") for attack in sorted(grouped)
        },
        "mean_code_bit_acc": {
            attack: mean_key(attack, "code_bit_acc") for attack in sorted(grouped)
        },
        "code_verification_rate": {
            attack: float(np.mean([bool(row["code_verified"]) for row in attack_rows]))
            for attack, attack_rows in sorted(grouped.items())
        },
        "mean_code_confidence": {
            attack: mean_key(attack, "code_confidence") for attack in sorted(grouped)
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
    }


def make_custom_grid(
    base: RelMelConfig, args: argparse.Namespace
) -> list[tuple[str, RelMelConfig]]:
    bands = [_parse_band(v) for v in (args.band_grid or [_band_to_cli(base.band)])]
    min_block_frames_grid = args.min_block_frames_grid or [base.min_block_frames]
    block_stride_grid = args.block_stride_grid or [base.block_stride]
    threshold_grid = args.threshold_grid or [base.threshold]
    align_max_shift_grid = args.align_max_shift_grid or [base.align_max_shift]
    configs: list[tuple[str, RelMelConfig]] = []
    for (
        alpha,
        band,
        mask_floor,
        boundary_margin,
        block_frames,
        block_stride,
        min_block_frames,
        bits_per_block,
        pair_bins,
        energy_gamma,
        threshold,
        align_max_shift,
    ) in itertools.product(
        args.alpha_grid,
        bands,
        args.mask_floor_grid,
        args.boundary_margin_grid,
        args.block_frames_grid,
        block_stride_grid,
        min_block_frames_grid,
        args.bits_per_block_grid,
        args.pair_bins_grid,
        args.energy_gamma_grid,
        threshold_grid,
        align_max_shift_grid,
    ):
        configs.append(
            (
                "custom",
                replace(
                    base,
                    alpha=float(alpha),
                    band=band,
                    mask_floor=float(mask_floor),
                    boundary_margin=float(boundary_margin),
                    block_frames=int(block_frames),
                    block_stride=None if block_stride is None else int(block_stride),
                    min_block_frames=int(min_block_frames),
                    bits_per_block=int(bits_per_block),
                    pair_bins=int(pair_bins),
                    energy_gamma=float(energy_gamma),
                    threshold=float(threshold),
                    align_max_shift=int(align_max_shift),
                ),
            )
        )
    return configs


def make_relmel_audit_suite(base: RelMelConfig) -> list[tuple[str, RelMelConfig]]:
    candidates: list[tuple[str, RelMelConfig]] = []
    seen: set[str] = set()

    def add(group: str, cfg: RelMelConfig) -> None:
        key = json.dumps(cfg.to_dict(), sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        candidates.append((group, cfg))

    anchor = replace(
        base,
        alpha=0.43,
        band=(20, 60),
        block_frames=16,
        min_block_frames=16,
        bits_per_block=4,
        pair_bins=4,
        mask_floor=0.25,
        boundary_margin=0.01,
        energy_gamma=0.5,
    )

    for alpha in [0.38, 0.40, 0.42, 0.43, 0.44, 0.45, 0.46, 0.48]:
        add("alpha_anchor", replace(anchor, alpha=alpha))

    for alpha, band in itertools.product(
        [0.40, 0.43, 0.45],
        [(16, 56), (16, 60), (16, 64), (20, 56), (20, 60), (20, 64), (24, 60)],
    ):
        add("band", replace(anchor, alpha=alpha, band=band))

    for alpha, mask_floor, boundary_margin in itertools.product(
        [0.40, 0.43, 0.45],
        [0.15, 0.20, 0.25, 0.30, 0.35],
        [0.0, 0.005, 0.01, 0.02, 0.03],
    ):
        add(
            "mask_boundary",
            replace(
                anchor,
                alpha=alpha,
                mask_floor=mask_floor,
                boundary_margin=boundary_margin,
            ),
        )

    for alpha, block_frames, bits_per_block, pair_bins in itertools.product(
        [0.40, 0.43],
        [8, 12, 16, 20, 24],
        [2, 4, 6, 8],
        [3, 4, 5, 6],
    ):
        add(
            "structure",
            replace(
                anchor,
                alpha=alpha,
                block_frames=block_frames,
                min_block_frames=min(16, block_frames),
                bits_per_block=bits_per_block,
                pair_bins=pair_bins,
            ),
        )

    for alpha, energy_gamma in itertools.product(
        [0.40, 0.43, 0.45],
        [0.0, 0.25, 0.5, 0.75, 1.0],
    ):
        add("energy_gamma", replace(anchor, alpha=alpha, energy_gamma=energy_gamma))

    for alpha, threshold, align_max_shift in itertools.product(
        [0.40, 0.43],
        [0.65, 0.70, 0.75],
        [4, 12, 24],
    ):
        add(
            "verifier",
            replace(anchor, alpha=alpha, threshold=threshold, align_max_shift=align_max_shift),
        )

    return candidates


def objective_from_summary(summary: dict[str, Any], args: argparse.Namespace) -> float:
    acc = summary["mean_bit_acc"]
    pesq = summary["mean_pesq_bm"]
    none_acc = float(acc.get("none") if acc.get("none") is not None else 0.0)
    noise20_acc = float(acc.get("noise20") if acc.get("noise20") is not None else none_acc)
    noise10_acc = float(acc.get("noise10") if acc.get("noise10") is not None else 0.0)
    noise5_acc = float(acc.get("noise5") if acc.get("noise5") is not None else 0.0)
    echo_acc = float(acc.get("echo") if acc.get("echo") is not None else 0.0)
    none_pesq = pesq.get("none")
    penalty = 0.0 if none_pesq is None else max(0.0, args.quality_floor - float(none_pesq))
    return (
        none_acc
        + args.noise20_weight * noise20_acc
        + args.noise10_weight * noise10_acc
        + args.noise5_weight * noise5_acc
        + args.echo_weight * echo_acc
        - args.quality_weight * penalty
    )


def flatten_summary(
    candidate: int,
    group: str,
    objective: float,
    relmel_config: RelMelConfig,
    summary: dict[str, Any],
    attacks: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate": candidate,
        "group": group,
        "objective": objective,
        "alpha": relmel_config.alpha,
        "band": _band_to_cli(relmel_config.band),
        "mask_floor": relmel_config.mask_floor,
        "boundary_margin": relmel_config.boundary_margin,
        "block_frames": relmel_config.block_frames,
        "block_stride": relmel_config.block_stride,
        "min_block_frames": relmel_config.min_block_frames,
        "bits_per_block": relmel_config.bits_per_block,
        "pair_bins": relmel_config.pair_bins,
        "energy_gamma": relmel_config.energy_gamma,
        "threshold": relmel_config.threshold,
        "align_max_shift": relmel_config.align_max_shift,
        "payload_bits": summary["coding_config"]["payload_bits"],
        "code_bits": summary["coding_config"]["code_bits"],
        "ecc_repeat": summary["coding_config"]["ecc_repeat"],
        "ecc_interleave": summary["coding_config"]["interleave"],
        "num_rows": summary["num_rows"],
    }
    for attack in attacks:
        row[f"{attack}_bit_acc"] = summary["mean_bit_acc"].get(attack)
        row[f"{attack}_verified"] = summary["verification_rate"].get(attack)
        row[f"{attack}_payload_bit_acc"] = summary["mean_payload_bit_acc"].get(attack)
        row[f"{attack}_payload_verified"] = summary["payload_verification_rate"].get(attack)
        row[f"{attack}_code_bit_acc"] = summary["mean_code_bit_acc"].get(attack)
        row[f"{attack}_code_verified"] = summary["code_verification_rate"].get(attack)
        row[f"{attack}_pesq_bm"] = summary["mean_pesq_bm"].get(attack)
        row[f"{attack}_stoi_bm"] = summary["mean_stoi_bm"].get(attack)
        row[f"{attack}_confidence"] = summary["mean_confidence"].get(attack)
        row[f"{attack}_payload_confidence"] = summary["mean_payload_confidence"].get(attack)
        row[f"{attack}_code_confidence"] = summary["mean_code_confidence"].get(attack)
        row[f"{attack}_min_votes"] = summary["mean_min_votes"].get(attack)
        row[f"{attack}_mean_votes"] = summary["mean_votes"].get(attack)
    return row


def write_grid_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    total_candidates: int,
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
        "total_candidates": total_candidates,
        "run_config": cfg,
    }
    with (output_dir / "grid_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with (output_dir / "best_candidate.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2, sort_keys=True)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _parse_band(value: str) -> tuple[int, int]:
    if ":" not in value:
        raise ValueError(f"Band must be formatted as start:end, got {value!r}")
    start, end = value.split(":", maxsplit=1)
    return int(start), int(end)


def _band_to_cli(value: tuple[int, int]) -> str:
    return f"{value[0]}:{value[1]}"


if __name__ == "__main__":
    main()
