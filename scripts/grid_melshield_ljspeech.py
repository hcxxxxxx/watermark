#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from melshield.attacks import build_attacks
from melshield.datasets import iter_ljspeech, load_audio
from melshield.mel import MelConfig, MelFrontend
from melshield.metrics import compare_audio
from melshield.vocoders import build_vocoder
from melshield.watermark import MelShield, MelShieldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search MelShield reproduction variants.")
    parser.add_argument("--config", default="configs/melshield_hifigan_official.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--suite",
        choices=["custom", "repro-audit"],
        default="custom",
        help="Use repro-audit for a targeted multi-group MelShield reproduction sweep.",
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attacks", nargs="+", default=["none", "noise20"])
    parser.add_argument("--vocoder", default=None, choices=["mel", "griffinlim", "hifigan", "command"])
    parser.add_argument("--vocoder-checkpoint", default=None)
    parser.add_argument("--vocoder-config", default=None)
    parser.add_argument("--vocoder-command", default=None)

    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.04, 0.05, 0.06])
    parser.add_argument("--band-grid", nargs="+", default=["20:56"])
    parser.add_argument("--mask-floor-grid", nargs="+", type=float, default=[0.05])
    parser.add_argument("--energy-gamma-grid", nargs="+", type=float, default=[0.75])
    parser.add_argument("--boundary-margin-grid", nargs="+", type=float, default=[0.02])
    parser.add_argument("--headroom-grid", nargs="+", type=float, default=[0.0])
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=None)
    parser.add_argument("--align-max-shift-grid", nargs="+", type=int, default=None)
    parser.add_argument("--mask-mode-grid", nargs="+", default=None)
    parser.add_argument("--freq-gamma-grid", nargs="+", type=float, default=None)
    parser.add_argument("--texture-gamma-grid", nargs="+", type=float, default=None)
    parser.add_argument("--smooth-frames-grid", nargs="+", type=int, default=None)

    parser.add_argument("--f-min-grid", nargs="+", type=float, default=None)
    parser.add_argument("--f-max-grid", nargs="+", default=None)
    parser.add_argument("--center-grid", nargs="+", default=None)
    parser.add_argument("--norm-grid", nargs="+", default=None)
    parser.add_argument("--mel-scale-grid", nargs="+", default=None)

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

    base_mel = MelConfig(**cfg["mel"])
    base_wm = build_wm_config(cfg)
    data_root = Path(args.data_root or cfg["paths"]["data_root"])
    vocoder_name = args.vocoder or cfg["vocoder"]["name"]
    attacks = args.attacks
    if vocoder_name == "mel":
        attacks = ["none"]

    if args.suite == "repro-audit":
        candidates = make_repro_suite(base_mel, base_wm)
    else:
        mel_grid = make_mel_grid(base_mel, args)
        wm_grid = make_wm_grid(base_wm, args)
        candidates = [
            ("custom", mel_config, wm_config)
            for mel_config, wm_config in itertools.product(mel_grid, wm_grid)
        ]

    aggregate_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    vocoder = None
    if vocoder_name != "mel":
        vocoder = build_vocoder(
            vocoder_name,
            mel_config=base_mel,
            device=args.device,
            checkpoint=args.vocoder_checkpoint or cfg["vocoder"].get("checkpoint"),
            config=args.vocoder_config or cfg["vocoder"].get("config"),
            command_template=args.vocoder_command or cfg["vocoder"].get("command"),
            sample_rate=base_mel.sample_rate,
        )

    for idx, (group, mel_config, wm_config) in enumerate(candidates, start=1):
        candidate_dir = output_dir / f"candidate_{idx:03d}"
        rows, summary = run_candidate(
            data_root=data_root,
            output_dir=candidate_dir,
            cfg=cfg,
            mel_config=mel_config,
            wm_config=wm_config,
            vocoder=vocoder,
            vocoder_name=vocoder_name,
            attacks=attacks,
            limit=args.limit,
            device=args.device,
            keep_outputs=args.keep_candidate_results,
        )
        objective = objective_from_summary(summary, args)
        flat = flatten_summary(idx, group, objective, mel_config, wm_config, summary, attacks)
        aggregate_rows.append(flat)
        if best is None or objective > best["objective"]:
            best = flat
        none_acc = flat.get("none_bit_acc")
        noise20_acc = flat.get("noise20_bit_acc")
        none_pesq = flat.get("none_pesq_bm")
        print(
            f"candidate {idx}/{len(candidates)} "
            f"group={group} objective={objective:.4f} "
            f"alpha={wm_config.alpha} none={none_acc} noise20={noise20_acc} "
            f"pesq={none_pesq}"
        )

    write_grid_outputs(output_dir, aggregate_rows, best, args, cfg)
    print(f"wrote grid results to {output_dir}")


def run_candidate(
    data_root: Path,
    output_dir: Path,
    cfg: dict[str, Any],
    mel_config: MelConfig,
    wm_config: MelShieldConfig,
    vocoder: Any,
    vocoder_name: str,
    attacks: list[str],
    limit: int,
    device: str,
    keep_outputs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if keep_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
    frontend = MelFrontend(mel_config, device=device)
    shield = MelShield(wm_config)
    attack_fns = build_attacks(attacks)
    rows: list[dict[str, Any]] = []

    for item in iter_ljspeech(data_root, limit=limit):
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

        if vocoder_name == "mel":
            result = shield.extract(watermarked_norm, reference)
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
            result = shield.extract(detected.normalized, reference)
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
        write_candidate_outputs(output_dir, rows, cfg, mel_config, wm_config, summary)
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
    return {
        "utterance_id": utterance_id,
        "attack": attack,
        "bit_acc": result.bit_accuracy,
        "verified": result.verified,
        "confidence": float(np.mean(np.abs(result.scores))),
        "pesq_gt": pesq_gt,
        "stoi_gt": stoi_gt,
        "pesq_bm": pesq_bm,
        "stoi_bm": stoi_bm,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["attack"]].append(row)

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


def make_mel_grid(base: MelConfig, args: argparse.Namespace) -> list[MelConfig]:
    f_min_grid = args.f_min_grid if args.f_min_grid is not None else [base.f_min]
    f_max_grid = [_parse_fmax(v, base.sample_rate) for v in (args.f_max_grid or [base.resolved_f_max()])]
    center_grid = [_parse_bool(v) for v in (args.center_grid or [str(base.center)])]
    norm_grid = [_parse_optional_str(v) for v in (args.norm_grid or [_optional_to_cli(base.norm)])]
    mel_scale_grid = args.mel_scale_grid or [base.mel_scale]
    configs = []
    for f_min, f_max, center, norm, mel_scale in itertools.product(
        f_min_grid, f_max_grid, center_grid, norm_grid, mel_scale_grid
    ):
        configs.append(
            replace(
                base,
                f_min=float(f_min),
                f_max=f_max,
                center=center,
                norm=norm,
                mel_scale=mel_scale,
            )
        )
    return configs


def make_wm_grid(base: MelShieldConfig, args: argparse.Namespace) -> list[MelShieldConfig]:
    configs = []
    bands = [_parse_band(v) for v in args.band_grid]
    threshold_grid = args.threshold_grid or [base.threshold]
    align_max_shift_grid = args.align_max_shift_grid or [base.align_max_shift]
    mask_mode_grid = args.mask_mode_grid or [base.mask_mode]
    freq_gamma_grid = args.freq_gamma_grid or [base.freq_gamma]
    texture_gamma_grid = args.texture_gamma_grid or [base.texture_gamma]
    smooth_frames_grid = args.smooth_frames_grid or [base.smooth_frames]
    for (
        alpha,
        band,
        mask_floor,
        energy_gamma,
        boundary_margin,
        headroom,
        threshold,
        align_max_shift,
        mask_mode,
        freq_gamma,
        texture_gamma,
        smooth_frames,
    ) in itertools.product(
        args.alpha_grid,
        bands,
        args.mask_floor_grid,
        args.energy_gamma_grid,
        args.boundary_margin_grid,
        args.headroom_grid,
        threshold_grid,
        align_max_shift_grid,
        mask_mode_grid,
        freq_gamma_grid,
        texture_gamma_grid,
        smooth_frames_grid,
    ):
        configs.append(
            replace(
                base,
                alpha=float(alpha),
                band=band,
                mask_floor=float(mask_floor),
                energy_gamma=float(energy_gamma),
                boundary_margin=float(boundary_margin),
                headroom=float(headroom),
                threshold=float(threshold),
                align_max_shift=int(align_max_shift),
                mask_mode=str(mask_mode),
                freq_gamma=float(freq_gamma),
                texture_gamma=float(texture_gamma),
                smooth_frames=int(smooth_frames),
            )
        )
    return configs


def make_repro_suite(
    base_mel: MelConfig,
    base_wm: MelShieldConfig,
) -> list[tuple[str, MelConfig, MelShieldConfig]]:
    """Build a targeted suite without exploding into a full Cartesian product."""

    candidates: list[tuple[str, MelConfig, MelShieldConfig]] = []
    seen: set[str] = set()

    def add(group: str, mel_config: MelConfig, wm_config: MelShieldConfig) -> None:
        key = json.dumps(
            {
                "mel": mel_config.to_dict(),
                "wm": wm_config.to_dict(),
            },
            sort_keys=True,
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append((group, mel_config, wm_config))

    for alpha in [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08]:
        add("alpha_baseline", base_mel, replace(base_wm, alpha=alpha))

    for alpha, f_min, f_max, center, norm, mel_scale in itertools.product(
        [0.045, 0.05, 0.055],
        [0.0, 20.0],
        [8000.0, float(base_mel.sample_rate / 2)],
        [True, False],
        ["slaney", None],
        ["slaney", "htk"],
    ):
        add(
            "mel_frontend",
            replace(
                base_mel,
                f_min=f_min,
                f_max=f_max,
                center=center,
                norm=norm,
                mel_scale=mel_scale,
            ),
            replace(base_wm, alpha=alpha),
        )

    for alpha, band in itertools.product(
        [0.045, 0.05, 0.055],
        [(16, 52), (16, 56), (16, 60), (20, 56), (20, 60), (24, 56)],
    ):
        add("band", base_mel, replace(base_wm, alpha=alpha, band=band))

    for alpha, mask_floor in itertools.product(
        [0.045, 0.05, 0.055],
        [0.0, 0.03, 0.05, 0.1, 0.15, 0.25],
    ):
        add("mask_floor", base_mel, replace(base_wm, alpha=alpha, mask_floor=mask_floor))

    for alpha, energy_gamma in itertools.product(
        [0.045, 0.05, 0.055],
        [0.25, 0.5, 0.75, 1.0, 1.25],
    ):
        add(
            "energy_gamma",
            base_mel,
            replace(base_wm, alpha=alpha, energy_gamma=energy_gamma),
        )

    for alpha, boundary_margin in itertools.product(
        [0.045, 0.05, 0.055],
        [0.0, 0.01, 0.02, 0.05],
    ):
        add(
            "boundary_margin",
            base_mel,
            replace(base_wm, alpha=alpha, boundary_margin=boundary_margin),
        )

    for alpha, mask_floor, boundary_margin in itertools.product(
        [0.045, 0.05, 0.055],
        [0.05, 0.15, 0.25],
        [0.0, 0.01, 0.02],
    ):
        add(
            "mask_floor_boundary",
            base_mel,
            replace(
                base_wm,
                alpha=alpha,
                mask_floor=mask_floor,
                boundary_margin=boundary_margin,
            ),
        )

    for alpha, headroom in itertools.product([0.045, 0.05, 0.055], [0.0, 0.005, 0.01]):
        add("headroom", base_mel, replace(base_wm, alpha=alpha, headroom=headroom))

    for alpha, mask_floor, freq_gamma, texture_gamma, smooth_frames in itertools.product(
        [0.045, 0.05],
        [0.05, 0.15],
        [0.25, 0.5],
        [0.0, 0.25],
        [1, 5],
    ):
        add(
            "reliability_mask",
            base_mel,
            replace(
                base_wm,
                alpha=alpha,
                mask_floor=mask_floor,
                mask_mode="reliability",
                freq_gamma=freq_gamma,
                texture_gamma=texture_gamma,
                smooth_frames=smooth_frames,
            ),
        )

    for alpha, threshold, align_max_shift in itertools.product(
        [0.045, 0.05, 0.055],
        [0.55, 0.61, 0.7],
        [4, 12, 24],
    ):
        add(
            "verifier",
            base_mel,
            replace(
                base_wm,
                alpha=alpha,
                threshold=threshold,
                align_max_shift=align_max_shift,
            ),
        )

    return candidates


def build_wm_config(cfg: dict[str, Any]) -> MelShieldConfig:
    data = dict(cfg["watermark"])
    data["band"] = tuple(data["band"])
    return MelShieldConfig(**data)


def objective_from_summary(summary: dict[str, Any], args: argparse.Namespace) -> float:
    acc = summary["mean_bit_acc"]
    pesq = summary["mean_pesq_bm"]
    none_acc = float(acc.get("none") if acc.get("none") is not None else 0.0)
    noise20_acc = float(acc.get("noise20") if acc.get("noise20") is not None else none_acc)
    none_pesq = pesq.get("none")
    penalty = 0.0 if none_pesq is None else max(0.0, args.quality_floor - float(none_pesq))
    return none_acc + args.noise20_weight * noise20_acc - args.quality_weight * penalty


def flatten_summary(
    candidate: int,
    group: str,
    objective: float,
    mel_config: MelConfig,
    wm_config: MelShieldConfig,
    summary: dict[str, Any],
    attacks: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate": candidate,
        "group": group,
        "objective": objective,
        "alpha": wm_config.alpha,
        "band": f"{wm_config.band[0]}:{wm_config.band[1]}",
        "mask_floor": wm_config.mask_floor,
        "energy_gamma": wm_config.energy_gamma,
        "boundary_margin": wm_config.boundary_margin,
        "headroom": wm_config.headroom,
        "threshold": wm_config.threshold,
        "align_max_shift": wm_config.align_max_shift,
        "mask_mode": wm_config.mask_mode,
        "freq_gamma": wm_config.freq_gamma,
        "texture_gamma": wm_config.texture_gamma,
        "smooth_frames": wm_config.smooth_frames,
        "f_min": mel_config.f_min,
        "f_max": mel_config.resolved_f_max(),
        "center": mel_config.center,
        "norm": _optional_to_cli(mel_config.norm),
        "mel_scale": mel_config.mel_scale,
        "num_rows": summary["num_rows"],
    }
    for attack in attacks:
        row[f"{attack}_bit_acc"] = summary["mean_bit_acc"].get(attack)
        row[f"{attack}_verified"] = summary["verification_rate"].get(attack)
        row[f"{attack}_pesq_bm"] = summary["mean_pesq_bm"].get(attack)
        row[f"{attack}_stoi_bm"] = summary["mean_stoi_bm"].get(attack)
        row[f"{attack}_confidence"] = summary["mean_confidence"].get(attack)
    return row


def write_candidate_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    mel_config: MelConfig,
    wm_config: MelShieldConfig,
    summary: dict[str, Any],
) -> None:
    if rows:
        with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    payload = dict(summary)
    payload.update(
        {
            "mel_config": mel_config.to_dict(),
            "watermark_config": wm_config.to_dict(),
            "run_config": cfg,
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_grid_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> None:
    if not rows:
        return
    with (output_dir / "grid_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parse_band(value: str) -> tuple[int, int]:
    if ":" not in value:
        raise ValueError(f"Band must be formatted as start:end, got {value!r}")
    start, end = value.split(":", maxsplit=1)
    return int(start), int(end)


def _parse_fmax(value: Any, sample_rate: int) -> Optional[float]:
    if isinstance(value, str) and value.lower() in {"none", "nyquist", "sr/2"}:
        return float(sample_rate / 2)
    return float(value)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool: {value!r}")


def _parse_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value)
    if value.lower() in {"none", "null"}:
        return None
    return value


def _optional_to_cli(value: Optional[str]) -> str:
    return "none" if value is None else str(value)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
