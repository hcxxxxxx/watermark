#!/usr/bin/env python
"""实验 P0-2: Re-vocoding 攻击（含 re-vocoding + 噪声组合攻击）

回应的审稿意见（R1-Method CRITICAL）：
  "re-vocoding - re-extract the mel from the watermarked waveform and re-synthesize
   with a public HiFi-GAN; this is the single most natural attack against a
   mel-domain watermark and is absent."

band-targeted noise 见 exp_p0_3_band_noise_sweep.py。

运行：
    python scripts/exp_p0_2_revocoding.py \
        --config configs/relmel_hifigan.yaml \
        --vocoder-checkpoint checkpoints/hifigan/generator_v1 \
        --vocoder-config checkpoints/hifigan/config.json \
        --limit 500 --output-dir runs/exp_p0_2
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
from scripts.new_attacks import revocoding_attack


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-vocoding attack on mel-domain marks")
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
    p.add_argument(
        "--attack-vocoder-checkpoint",
        default=None,
        help="攻击者用的 vocoder checkpoint（默认与嵌入用的同一个，即最强攻击假设）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    mel_config, cfg = load_mel_config(args.config)
    frontend = MelFrontend(mel_config, device=args.device)

    embed_ckpt = args.vocoder_checkpoint or cfg["vocoder"].get("checkpoint")
    embed_cfg = args.vocoder_config or cfg["vocoder"].get("config")

    vocoder = build_vocoder(
        args.vocoder,
        mel_config=mel_config,
        device=args.device,
        checkpoint=embed_ckpt,
        config=embed_cfg,
        command_template=args.vocoder_command or cfg["vocoder"].get("command"),
        sample_rate=mel_config.sample_rate,
    )

    # 攻击集合：re-vocoding，以及 re-vocoding 叠加噪声（组合攻击）
    revocode = revocoding_attack(
        vocoder_name=args.vocoder,
        checkpoint=args.attack_vocoder_checkpoint or embed_ckpt,
        config=embed_cfg,
        mel_config=mel_config,
        device=args.device,
    )
    base = build_attacks(["none", "noise10"])

    def revocode_then_noise(waveform, sample_rate):
        r = revocode(waveform, sample_rate)
        return base["noise10"](r.waveform, r.sample_rate)

    attack_fns = {
        "none": base["none"],
        "revocode": revocode,
        "revocode_noise10": revocode_then_noise,
    }

    codec = RepetitionCode(repeat=1, interleave=True)
    rows: list[dict] = []

    print("[P0-2] Re-vocoding attack")
    print(f"  limit={args.limit}  seed={args.seed}")
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
        "experiment": "p0_2_revocoding",
        "limit": args.limit,
        "seed": args.seed,
        "attack_vocoder_note": (
            "攻击者与提供方使用同一 vocoder checkpoint（最强攻击假设）"
            if not args.attack_vocoder_checkpoint
            else f"攻击者使用独立 checkpoint: {args.attack_vocoder_checkpoint}"
        ),
        "by_method_attack": aggregate(rows, ["method", "attack"]),
    }
    write_results(out_dir, rows, summary)

    print(f"\n✓ 写入 {out_dir}")
    print("\n=== 结果 ===")
    agg = summary["by_method_attack"]
    for method in ("rawmer", "melshield"):
        print(f"{method}:")
        for atk in ("none", "revocode", "revocode_noise10"):
            cell = agg.get(f"{method}|{atk}")
            if cell:
                print(f"    {atk:14s} ACC={cell['mean_bit_acc']:.3f}  VR={cell['verification_rate']:.3f}")


if __name__ == "__main__":
    main()
