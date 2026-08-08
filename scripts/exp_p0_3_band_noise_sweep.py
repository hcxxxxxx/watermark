#!/usr/bin/env python
"""实验 P0-3: Band-targeted noise 强度扫描

回应审稿意见 R1-Method CRITICAL (c)：
  "an adaptive attacker who knows the scheme (band 20-60, block structure) but
   not the key - e.g., noise concentrated in the watermark band."

对每个 SNR 同时跑 band 定向噪声和全频段噪声，判断"定向"是否真的更高效。

注意 SNR 定义差异：band 噪声的 SNR 相对**带内**信号功率，全频段噪声相对
**总**信号功率。两者不是等总功率对比，报告时必须说明。

运行：
    python scripts/exp_p0_3_band_noise_sweep.py \
        --config configs/relmel_hifigan.yaml \
        --vocoder-checkpoint checkpoints/hifigan/generator_v1 \
        --vocoder-config checkpoints/hifigan/config.json \
        --limit 500 --output-dir runs/exp_p0_3
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
from scripts.new_attacks import band_targeted_noise, describe_band


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Band-targeted noise sweep")
    p.add_argument("--config", default="configs/relmel_hifigan.yaml")
    p.add_argument("--data-root", default="data/LJSpeech-1.1")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vocoder", default="hifigan", choices=["hifigan", "command"])
    p.add_argument("--vocoder-checkpoint", default=None)
    p.add_argument("--vocoder-config", default=None)
    p.add_argument("--vocoder-command", default=None)
    p.add_argument("--snrs", nargs="+", type=float, default=[20.0, 10.0, 5.0, 0.0])
    return p.parse_args()


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

    band = (20, 60)
    attack_fns: dict = {"none": build_attacks(["none"])["none"]}
    for snr in args.snrs:
        tag = f"{int(snr)}"
        attack_fns[f"band_noise{tag}"] = band_targeted_noise(
            snr, band=band, mel_config=mel_config
        )
        full_name = f"noise{tag}"
        try:
            attack_fns[full_name] = build_attacks([full_name])[full_name]
        except ValueError:
            print(f"  提示：全频段对照 {full_name} 未在注册表中，跳过")

    codec = RepetitionCode(repeat=1, interleave=True)
    rows: list[dict] = []

    print("[P0-3] Band-targeted noise sweep")
    print(f"  {describe_band(band, mel_config)}")
    print(f"  limit={args.limit}  SNRs={args.snrs}")
    print(f"  attacks={list(attack_fns)}")
    print()

    for method in ("rawmer", "melshield"):
        alpha = ALPHA_MAIN[(method, "hifigan")]
        marker, threshold, key = build_marker(method, alpha)
        print(f"=== {method} (alpha={alpha}) ===", flush=True)

        n = 0
        for item in iter_ljspeech(
            Path(args.data_root), limit=args.limit, sample_mode="random", seed=args.seed
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
            n += 1
            if n % 100 == 0:
                print(f"    {n} utterances", flush=True)

    summary = {
        "experiment": "p0_3_band_noise_sweep",
        "band_mel_bins": list(band),
        "band_hz": describe_band(band, mel_config),
        "limit": args.limit,
        "seed": args.seed,
        "snr_definition": (
            "band_noiseX 的 SNR 相对带内信号功率；noiseX 相对总信号功率。"
            "两者非等总功率对比。"
        ),
        "by_method_attack": aggregate(rows, ["method", "attack"]),
    }
    write_results(out_dir, rows, summary)

    print(f"\n✓ 写入 {out_dir}")
    print("\n=== band 定向 vs 全频段 (ACC) ===")
    agg = summary["by_method_attack"]
    for method in ("rawmer", "melshield"):
        print(f"{method}:")
        for snr in args.snrs:
            tag = int(snr)
            b = agg.get(f"{method}|band_noise{tag}")
            f = agg.get(f"{method}|noise{tag}")
            if b and f:
                d = f["mean_bit_acc"] - b["mean_bit_acc"]
                print(
                    f"    {tag:>2} dB   band={b['mean_bit_acc']:.3f}  "
                    f"full={f['mean_bit_acc']:.3f}  定向增益={d:+.3f}"
                )
            elif b:
                print(f"    {tag:>2} dB   band={b['mean_bit_acc']:.3f}  (无全频段对照)")
    print("\n（定向增益 > 0 表示 band 定向噪声更有效地破坏了水印）")


if __name__ == "__main__":
    main()
