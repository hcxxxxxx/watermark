#!/usr/bin/env python
"""实验 P0-1: ACC-vs-PESQ 曲线（α sweep）—— 审稿人要求的最重要补充实验

回应的审稿意见（R1-Method/EIC MAJOR）：
  "a single alpha point can sit anywhere on the quality-robustness trade-off —
   report ACC-vs-PESQ curves across an alpha sweep for both methods so the
   comparison holds at all matched qualities, not one."

同时用于解释 HiFi-GAN 复现异常（我们 PESQ 3.51/ACC 0.713 vs 原文 4.13/0.782）。

重要：RAWMER 用 RelMelMark（差分编码），MelShield 用 MelShield 类（绝对残差扩频）。
两者是不同实现，不是同一个类改参数。

运行：
    python scripts/exp_p0_1_acc_vs_pesq_sweep.py \
        --config configs/relmel_hifigan.yaml \
        --vocoder hifigan \
        --vocoder-checkpoint checkpoints/hifigan/generator_v1 \
        --vocoder-config checkpoints/hifigan/config.json \
        --limit 500 --output-dir runs/exp_p0_1_hifigan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from melshield.attacks import build_attacks
from melshield.coding import RepetitionCode
from melshield.datasets import iter_ljspeech, load_audio
from melshield.mel import MelFrontend
from melshield.vocoders import build_vocoder
from scripts.exp_common import (
    ALPHA_MAIN,
    aggregate,
    build_marker,
    load_mel_config,
    run_trial,
    write_results,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACC-vs-PESQ sweep (RAWMER vs MelShield)")
    p.add_argument("--config", default="configs/relmel_hifigan.yaml")
    p.add_argument("--data-root", default="data/LJSpeech-1.1")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vocoder", required=True, choices=["hifigan", "command"])
    p.add_argument("--vocoder-checkpoint", default=None)
    p.add_argument("--vocoder-config", default=None)
    p.add_argument("--vocoder-command", default=None)
    p.add_argument(
        "--alphas-rawmer",
        nargs="+",
        type=float,
        default=[0.25, 0.30, 0.355, 0.40, 0.435, 0.50, 0.58],
    )
    p.add_argument(
        "--alphas-melshield",
        nargs="+",
        type=float,
        default=[0.020, 0.035, 0.050, 0.061, 0.080, 0.110, 0.150],
    )
    p.add_argument(
        "--attacks", nargs="+", default=["none", "noise20", "noise10", "noise5"]
    )
    return p.parse_args()


def sweep_method(
    method: str,
    alphas: list[float],
    args: argparse.Namespace,
    frontend: MelFrontend,
    mel_config,
    vocoder,
    attack_fns: dict,
) -> list[dict]:
    codec = RepetitionCode(repeat=1, interleave=True)
    rows: list[dict] = []

    for idx, alpha in enumerate(alphas, 1):
        marker, threshold, key = build_marker(method, alpha)
        print(f"  [{idx}/{len(alphas)}] {method} alpha={alpha:.3f}", flush=True)

        n_done = 0
        for item in iter_ljspeech(
            Path(args.data_root),
            limit=args.limit,
            sample_mode="random",
            seed=args.seed,
        ):
            waveform, sr = load_audio(item.wav_path)
            for attack_name, attack_fn in attack_fns.items():
                out = run_trial(
                    marker, threshold, key, frontend, vocoder, mel_config,
                    item.utterance_id, waveform, sr, attack_fn, codec,
                )
                rows.append({
                    "method": method,
                    "alpha": f"{alpha:.3f}",
                    "utterance_id": item.utterance_id,
                    "attack": attack_name,
                    "bit_acc": out.bit_acc,
                    "verified": out.verified,
                    "pesq_bm": out.pesq_bm,
                    "stoi_bm": out.stoi_bm,
                })
            n_done += 1
            if n_done % 100 == 0:
                print(f"      {n_done} utterances", flush=True)
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    mel_config, cfg = load_mel_config(args.config)
    frontend = MelFrontend(mel_config, device=args.device)
    vocoder = build_vocoder(
        args.vocoder,
        mel_config=mel_config,
        device=args.device,
        checkpoint=args.vocoder_checkpoint or cfg["vocoder"].get("checkpoint"),
        config=args.vocoder_config or cfg["vocoder"].get("config"),
        command_template=args.vocoder_command or cfg["vocoder"].get("command"),
        sample_rate=mel_config.sample_rate,
    )
    attack_fns = build_attacks(args.attacks)

    print("[P0-1] ACC-vs-PESQ sweep")
    print(f"  vocoder={args.vocoder}  limit={args.limit}  seed={args.seed}")
    print(f"  attacks={args.attacks}")
    print(f"  RAWMER alphas    = {args.alphas_rawmer}")
    print(f"  MelShield alphas = {args.alphas_melshield}")
    print()

    rows: list[dict] = []
    print("=== RAWMER (RelMelMark, differential) ===")
    rows += sweep_method("rawmer", args.alphas_rawmer, args, frontend, mel_config, vocoder, attack_fns)
    print("=== MelShield (absolute-residual spread spectrum) ===")
    rows += sweep_method("melshield", args.alphas_melshield, args, frontend, mel_config, vocoder, attack_fns)

    summary = {
        "experiment": "p0_1_acc_vs_pesq_sweep",
        "vocoder": args.vocoder,
        "limit": args.limit,
        "seed": args.seed,
        "note": (
            "RAWMER=RelMelMark(differential+selection); "
            "MelShield=MelShield(absolute-residual spread spectrum). "
            "alpha is NOT comparable across the two formulations."
        ),
        "by_method_alpha_attack": aggregate(rows, ["method", "alpha", "attack"]),
    }

    # 便于人读的权衡表：每个 α 的 clean PESQ 对应各噪声 ACC
    tradeoff: dict[str, list[dict]] = {}
    agg = summary["by_method_alpha_attack"]
    for method in ("rawmer", "melshield"):
        pts = []
        alphas = sorted({k.split("|")[1] for k in agg if k.startswith(f"{method}|")})
        for a in alphas:
            clean = agg.get(f"{method}|{a}|none", {})
            row = {
                "alpha": a,
                "clean_pesq": clean.get("mean_pesq_bm"),
                "clean_stoi": clean.get("mean_stoi_bm"),
                "is_paper_main_point": abs(
                    float(a) - ALPHA_MAIN.get((method, args.vocoder), -1)
                ) < 1e-6,
            }
            for atk in args.attacks:
                if atk == "none":
                    continue
                cell = agg.get(f"{method}|{a}|{atk}", {})
                row[f"acc_{atk}"] = cell.get("mean_bit_acc")
                row[f"vr_{atk}"] = cell.get("verification_rate")
            pts.append(row)
        tradeoff[method] = pts
    summary["tradeoff_curve"] = tradeoff

    write_results(out_dir, rows, summary)

    print(f"\n✓ 写入 {out_dir}/results.csv 和 summary.json")
    print("\n=== 权衡曲线（clean PESQ vs ACC@5dB） ===")
    for method in ("rawmer", "melshield"):
        print(f"{method}:")
        for pt in tradeoff[method]:
            mark = "  <-- 论文主工作点" if pt["is_paper_main_point"] else ""
            pesq = pt["clean_pesq"]
            acc5 = pt.get("acc_noise5")
            pesq_s = f"{pesq:.3f}" if pesq is not None else "n/a"
            acc_s = f"{acc5:.3f}" if acc5 is not None else "n/a"
            print(f"    alpha={pt['alpha']}  PESQ={pesq_s}  ACC@5dB={acc_s}{mark}")


if __name__ == "__main__":
    main()
