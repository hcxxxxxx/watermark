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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from melshield.attacks import build_attacks
from melshield.coding import RepetitionCode, bit_accuracy, mean_abs_confidence
from melshield.datasets import iter_ljspeech, load_audio, save_audio
from melshield.mel import MelConfig, MelFrontend
from melshield.metrics import compare_audio
from melshield.vocoders import build_vocoder
from melshield.watermark import MelShield, MelShieldConfig, deterministic_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run research MelShield with adaptive masking, soft ECC, and attack-aware search."
    )
    parser.add_argument("--config", default="configs/research_melshield_hifigan.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vocoder", default=None, choices=["mel", "griffinlim", "hifigan", "command"])
    parser.add_argument("--vocoder-checkpoint", default=None)
    parser.add_argument("--vocoder-config", default=None)
    parser.add_argument("--vocoder-command", default=None)
    parser.add_argument("--attacks", nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--payload-bits", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--mask-mode", choices=["energy", "reliability"], default=None)
    parser.add_argument("--freq-gamma", type=float, default=None)
    parser.add_argument("--texture-gamma", type=float, default=None)
    parser.add_argument("--smooth-frames", type=int, default=None)
    parser.add_argument("--save-audio", action="store_true")

    parser.add_argument("--optimize", action="store_true", help="Run attack-aware grid search.")
    parser.add_argument("--search-limit", type=int, default=20)
    parser.add_argument("--train-attacks", nargs="+", default=None)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.03, 0.04, 0.05])
    parser.add_argument("--repeat-grid", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--freq-gamma-grid", nargs="+", type=float, default=[0.0, 0.5])
    parser.add_argument("--texture-gamma-grid", nargs="+", type=float, default=[0.0, 0.25])
    parser.add_argument("--smooth-grid", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--quality-floor", type=float, default=3.5)
    parser.add_argument("--quality-weight", type=float, default=0.20)
    parser.add_argument("--noise20-weight", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    output_dir = Path(args.output_dir or cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    mel_config = MelConfig(**cfg["mel"])
    base_wm_config = build_wm_config(cfg, args)
    base_codec = build_codec(cfg, args)
    data_root = Path(args.data_root or cfg["paths"]["data_root"])
    vocoder_name = args.vocoder or cfg["vocoder"]["name"]

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

    if args.optimize:
        best = run_search(
            args=args,
            cfg=cfg,
            output_dir=output_dir,
            data_root=data_root,
            mel_config=mel_config,
            frontend=frontend,
            vocoder=vocoder,
            vocoder_name=vocoder_name,
            base_wm_config=base_wm_config,
            base_codec=base_codec,
        )
        print("best candidate:")
        print(json.dumps(best, indent=2, sort_keys=True))
        return

    attacks = args.attacks or cfg["evaluation"]["attacks"]
    if vocoder_name == "mel":
        attacks = ["none"]
    rows, summary = run_experiment(
        data_root=data_root,
        output_dir=output_dir,
        mel_config=mel_config,
        frontend=frontend,
        vocoder=vocoder,
        vocoder_name=vocoder_name,
        wm_config=base_wm_config,
        codec=base_codec,
        attacks=attacks,
        limit=args.limit or cfg["evaluation"].get("limit"),
        save_audio_files=args.save_audio,
        write_refs=True,
    )
    write_outputs(output_dir, rows, summary, cfg, base_wm_config, base_codec, mel_config)
    print(f"wrote results to {output_dir}")


def run_search(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    output_dir: Path,
    data_root: Path,
    mel_config: MelConfig,
    frontend: MelFrontend,
    vocoder: Any,
    vocoder_name: str,
    base_wm_config: MelShieldConfig,
    base_codec: RepetitionCode,
) -> dict[str, Any]:
    train_attacks = args.train_attacks or cfg["evaluation"].get("train_attacks") or ["none", "noise20"]
    if vocoder_name == "mel":
        train_attacks = ["none"]
    candidates = list(
        itertools.product(
            args.alpha_grid,
            args.repeat_grid,
            args.freq_gamma_grid,
            args.texture_gamma_grid,
            args.smooth_grid,
        )
    )
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for idx, (alpha, repeat, freq_gamma, texture_gamma, smooth_frames) in enumerate(candidates, start=1):
        codec = replace(base_codec, repeat=repeat)
        wm_config = replace(
            base_wm_config,
            alpha=alpha,
            payload_bits=base_payload_bits(base_wm_config, base_codec) * repeat,
            mask_mode="reliability",
            freq_gamma=freq_gamma,
            texture_gamma=texture_gamma,
            smooth_frames=smooth_frames,
        )
        candidate_dir = output_dir / "search_tmp" / f"candidate_{idx:03d}"
        candidate_rows, summary = run_experiment(
            data_root=data_root,
            output_dir=candidate_dir,
            mel_config=mel_config,
            frontend=frontend,
            vocoder=vocoder,
            vocoder_name=vocoder_name,
            wm_config=wm_config,
            codec=codec,
            attacks=train_attacks,
            limit=args.search_limit,
            save_audio_files=False,
            write_refs=False,
        )
        objective = search_objective(summary, args)
        row = {
            "candidate": idx,
            "objective": objective,
            "alpha": alpha,
            "repeat": repeat,
            "freq_gamma": freq_gamma,
            "texture_gamma": texture_gamma,
            "smooth_frames": smooth_frames,
            "mean_payload_bit_acc": summary["mean_payload_bit_acc"],
            "verification_rate": summary["verification_rate"],
            "mean_pesq_bm": summary["mean_pesq_bm"],
            "mean_stoi_bm": summary["mean_stoi_bm"],
        }
        rows.append(row)
        print(f"candidate {idx}/{len(candidates)} objective={objective:.4f} {row}")
        if best is None or objective > best["objective"]:
            best = row

    search_path = output_dir / "search_results.csv"
    with search_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "best_search_candidate.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2, sort_keys=True)
    assert best is not None
    return best


def run_experiment(
    data_root: Path,
    output_dir: Path,
    mel_config: MelConfig,
    frontend: MelFrontend,
    vocoder: Any,
    vocoder_name: str,
    wm_config: MelShieldConfig,
    codec: RepetitionCode,
    attacks: list[str],
    limit: int | None,
    save_audio_files: bool,
    write_refs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_refs:
        (output_dir / "refs").mkdir(exist_ok=True)
    if save_audio_files:
        (output_dir / "audio").mkdir(exist_ok=True)

    shield = MelShield(wm_config)
    attack_fns = build_attacks(attacks)
    rows: list[dict[str, Any]] = []
    payload_bits = wm_config.payload_bits // codec.repeat

    for item in iter_ljspeech(data_root, limit=limit):
        waveform, sample_rate = load_audio(item.wav_path)
        bundle = frontend.waveform_to_normalized_logmel(waveform, sample_rate)
        payload = deterministic_bits(
            key=f"{wm_config.key}|payload",
            identifier=item.utterance_id,
            length=payload_bits,
        )
        codeword = codec.encode(payload)
        watermarked_norm, reference = shield.embed(
            clean_mel=bundle.normalized,
            message=codeword,
            utterance_id=item.utterance_id,
            norm_stats=bundle.stats,
            mel_config=mel_config,
        )
        if write_refs:
            reference.save(output_dir / "refs" / f"{item.utterance_id}.npz")

        if vocoder_name == "mel":
            extracted = shield.extract(watermarked_norm, reference)
            row = decode_row(item.utterance_id, "none", payload, codeword, extracted, codec, wm_config)
            row.update(empty_metrics())
            rows.append(row)
            continue

        assert vocoder is not None
        clean_log_mel = torch.from_numpy(bundle.log_mel)
        wm_log_mel = torch.from_numpy(frontend.denormalize(watermarked_norm, bundle.stats))
        benchmark_audio = vocoder.synthesize(clean_log_mel)
        watermarked_audio = vocoder.synthesize(wm_log_mel)
        vocoder_sr = int(vocoder.sample_rate)

        if save_audio_files:
            save_audio_path = output_dir / "audio" / f"{item.utterance_id}_wm.wav"
            save_audio(save_audio_path, watermarked_audio, vocoder_sr)
            save_audio(output_dir / "audio" / f"{item.utterance_id}_bm.wav", benchmark_audio, vocoder_sr)

        for attack_name, attack_fn in attack_fns.items():
            attacked = attack_fn(watermarked_audio, vocoder_sr)
            detected = frontend.waveform_to_normalized_logmel(
                attacked.waveform,
                attacked.sample_rate,
                stats=reference.norm_stats,
            )
            extracted = shield.extract(detected.normalized, reference)
            row = decode_row(item.utterance_id, attack_name, payload, codeword, extracted, codec, wm_config)
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

    return rows, summarize_rows(rows)


def decode_row(
    utterance_id: str,
    attack_name: str,
    payload: np.ndarray,
    codeword: np.ndarray,
    extracted: Any,
    codec: RepetitionCode,
    wm_config: MelShieldConfig,
) -> dict[str, Any]:
    payload_soft, payload_scores = codec.decode_scores(extracted.scores)
    payload_hard = codec.decode_hard(extracted.decoded)
    payload_soft_acc = bit_accuracy(payload, payload_soft)
    payload_hard_acc = bit_accuracy(payload, payload_hard)
    code_bit_acc = bit_accuracy(codeword, extracted.decoded)
    return {
        "utterance_id": utterance_id,
        "attack": attack_name,
        "payload_bit_acc": payload_soft_acc,
        "payload_hard_bit_acc": payload_hard_acc,
        "code_bit_acc": code_bit_acc,
        "verified": bool(payload_soft_acc >= wm_config.threshold),
        "payload_confidence": mean_abs_confidence(payload_scores),
        "code_confidence": mean_abs_confidence(extracted.scores),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["attack"]].append(row)

    def mean_float(attack: str, key: str) -> float | None:
        values = [float(row[key]) for row in grouped[attack] if row.get(key) not in ("", None)]
        if not values:
            return None
        return float(np.mean(values))

    return {
        "mean_payload_bit_acc": {
            attack: mean_float(attack, "payload_bit_acc") for attack in sorted(grouped)
        },
        "mean_payload_hard_bit_acc": {
            attack: mean_float(attack, "payload_hard_bit_acc") for attack in sorted(grouped)
        },
        "mean_code_bit_acc": {
            attack: mean_float(attack, "code_bit_acc") for attack in sorted(grouped)
        },
        "verification_rate": {
            attack: float(np.mean([bool(row["verified"]) for row in rows_for_attack]))
            for attack, rows_for_attack in sorted(grouped.items())
        },
        "mean_pesq_bm": {
            attack: mean_float(attack, "pesq_bm") for attack in sorted(grouped)
        },
        "mean_stoi_bm": {
            attack: mean_float(attack, "stoi_bm") for attack in sorted(grouped)
        },
        "num_rows": len(rows),
    }


def search_objective(summary: dict[str, Any], args: argparse.Namespace) -> float:
    acc = summary["mean_payload_bit_acc"]
    attacks = list(acc.keys())
    robust_mean = float(np.mean([acc[a] for a in attacks if acc[a] is not None]))
    if "noise20" in acc and acc["noise20"] is not None:
        robust_mean += args.noise20_weight * float(acc["noise20"])
    pesq_none = summary["mean_pesq_bm"].get("none")
    quality_penalty = 0.0
    if pesq_none is not None:
        quality_penalty = max(0.0, args.quality_floor - float(pesq_none))
    return robust_mean - args.quality_weight * quality_penalty


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    cfg: dict[str, Any],
    wm_config: MelShieldConfig,
    codec: RepetitionCode,
    mel_config: MelConfig,
) -> None:
    if rows:
        with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(summary)
    summary.update(
        {
            "mel_config": mel_config.to_dict(),
            "watermark_config": wm_config.to_dict(),
            "coding": {"repeat": codec.repeat, "interleave": codec.interleave},
            "run_config": cfg,
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_wm_config(cfg: dict[str, Any], args: argparse.Namespace) -> MelShieldConfig:
    wm_data = dict(cfg["watermark"])
    wm_data["band"] = tuple(wm_data["band"])
    if args.alpha is not None:
        wm_data["alpha"] = args.alpha
    if args.payload_bits is not None:
        wm_data["payload_bits"] = args.payload_bits
    if args.threshold is not None:
        wm_data["threshold"] = args.threshold
    if args.mask_mode is not None:
        wm_data["mask_mode"] = args.mask_mode
    if args.freq_gamma is not None:
        wm_data["freq_gamma"] = args.freq_gamma
    if args.texture_gamma is not None:
        wm_data["texture_gamma"] = args.texture_gamma
    if args.smooth_frames is not None:
        wm_data["smooth_frames"] = args.smooth_frames

    payload_bits = int(wm_data["payload_bits"])
    repeat = int(args.repeat if args.repeat is not None else cfg.get("coding", {}).get("repeat", 1))
    wm_data["payload_bits"] = payload_bits * repeat
    return MelShieldConfig(**wm_data)


def build_codec(cfg: dict[str, Any], args: argparse.Namespace) -> RepetitionCode:
    coding_cfg = cfg.get("coding", {})
    repeat = int(args.repeat if args.repeat is not None else coding_cfg.get("repeat", 1))
    interleave = bool(coding_cfg.get("interleave", True))
    return RepetitionCode(repeat=repeat, interleave=interleave)


def base_payload_bits(wm_config: MelShieldConfig, codec: RepetitionCode) -> int:
    if wm_config.payload_bits % codec.repeat != 0:
        raise ValueError("wm_config.payload_bits must be divisible by codec.repeat")
    return wm_config.payload_bits // codec.repeat


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def empty_metrics() -> dict[str, str]:
    return {"pesq_gt": "", "stoi_gt": "", "pesq_bm": "", "stoi_bm": ""}


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
