"""exp_common.py — 审稿补充实验的共享工具

关键设计决定：
  - RAWMER 用 melshield.relmel.RelMelMark（差分编码 + 可靠对选择）
  - MelShield 用 melshield.watermark.MelShield（绝对残差扩频），**不是**
    RelMel 改参数。用 RelMel + pair_candidates=1 得到的是论文的 C=1 消融，
    不是 MelShield 基线，两者不可混淆。
  - mel 参数从 YAML 读，不硬编码，保证与 vocoder 一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from melshield.coding import RepetitionCode
from melshield.mel import MelConfig, MelFrontend
from melshield.relmel import RelMelConfig, RelMelMark
from melshield.watermark import MelShield, MelShieldConfig, deterministic_bits


# ============ 论文最终主配置（来自 HANDOVER.MD §5，权威值） ============
# 注意：configs/*.yaml 里的默认值早于论文最终参数（band 20:56 vs 20:60），
# 因此这里显式写出，不依赖 YAML 的 relmel/melshield 段。

RAWMER_BASE = dict(
    key="relmel-review-2026",
    band=(20, 60),
    threshold=0.75,
    payload_bits=32,
    block_frames=8,
    block_stride=8,
    min_block_frames=8,
    bits_per_block=6,
    pair_bins=6,
    pair_candidates=16,
    detector_mode="plain",
    mask_floor=0.20,
    energy_gamma=0.5,
    boundary_margin=0.01,
    align_max_shift=12,
)

MELSHIELD_BASE = dict(
    key="melshield-review-2026",
    band=(20, 60),
    threshold=0.61,
    payload_bits=32,
    headroom=0.0,
    mask_floor=0.05,
    energy_gamma=0.75,
    boundary_margin=0.02,
    align_max_shift=12,
)

# 论文的质量匹配工作点（HANDOVER.MD §5）
ALPHA_MAIN = {
    ("rawmer", "hifigan"): 0.435,
    ("rawmer", "diffwave"): 0.355,
    ("melshield", "hifigan"): 0.050,
    ("melshield", "diffwave"): 0.061,
}


def load_mel_config(config_path: str | Path) -> tuple[MelConfig, dict[str, Any]]:
    """从 YAML 读 mel 配置，保证与 vocoder 一致。"""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return MelConfig(**cfg["mel"]), cfg


def build_marker(method: str, alpha: float) -> tuple[Any, float, str]:
    """构造水印器。

    Returns:
        (marker, threshold, key) — marker 有 .embed()/.extract() 接口
    """
    if method == "rawmer":
        config = RelMelConfig(alpha=alpha, **RAWMER_BASE)
        return RelMelMark(config), config.threshold, config.key
    if method == "melshield":
        config = MelShieldConfig(alpha=alpha, **MELSHIELD_BASE)
        return MelShield(config), config.threshold, config.key
    raise ValueError(f"未知方法: {method!r}，应为 'rawmer' 或 'melshield'")


@dataclass
class TrialOutcome:
    bit_acc: float
    verified: bool
    pesq_bm: float | None
    stoi_bm: float | None


def run_trial(
    marker: Any,
    threshold: float,
    key: str,
    frontend: MelFrontend,
    vocoder: Any,
    mel_config: MelConfig,
    utterance_id: str,
    waveform: torch.Tensor,
    sample_rate: int,
    attack_fn: Any,
    codec: RepetitionCode,
) -> TrialOutcome:
    """跑单条音频的完整 embed → vocode → attack → extract 流程。"""
    from melshield.metrics import compare_audio

    bundle = frontend.waveform_to_normalized_logmel(waveform, sample_rate)

    payload = deterministic_bits(
        key=f"{key}|payload", identifier=utterance_id, length=32
    )
    message = codec.encode(payload)

    watermarked_norm, reference = marker.embed(
        clean_mel=bundle.normalized,
        message=message,
        utterance_id=utterance_id,
        norm_stats=bundle.stats,
        mel_config=mel_config,
    )

    clean_log_mel = torch.from_numpy(bundle.log_mel)
    wm_log_mel = torch.from_numpy(frontend.denormalize(watermarked_norm, bundle.stats))
    benchmark_audio = vocoder.synthesize(clean_log_mel)
    watermarked_audio = vocoder.synthesize(wm_log_mel)
    vocoder_sr = int(vocoder.sample_rate)

    attacked = attack_fn(watermarked_audio, vocoder_sr)
    detected = frontend.waveform_to_normalized_logmel(
        attacked.waveform, attacked.sample_rate, stats=reference.norm_stats
    )
    result = marker.extract(detected.normalized, reference)

    payload_decoded, _ = codec.decode_scores(result.scores)
    bit_acc = float(np.mean(payload == payload_decoded))

    bm = compare_audio(
        benchmark_audio, vocoder_sr, attacked.waveform, attacked.sample_rate
    )
    return TrialOutcome(
        bit_acc=bit_acc,
        verified=bool(bit_acc >= threshold),
        pesq_bm=bm.pesq,
        stoi_bm=bm.stoi,
    )


def aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> dict[str, Any]:
    """按指定键聚合 ACC / VR / PESQ / STOI。"""
    from collections import defaultdict

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)

    out: dict[str, Any] = {}
    for gkey, grows in sorted(grouped.items(), key=lambda kv: [str(x) for x in kv[0]]):
        label = "|".join(str(x) for x in gkey)

        def mean_of(field: str) -> float | None:
            vals = [
                float(r[field])
                for r in grows
                if r.get(field) not in (None, "", float("nan"))
            ]
            return float(np.mean(vals)) if vals else None

        out[label] = {
            "n": len(grows),
            "mean_bit_acc": mean_of("bit_acc"),
            "verification_rate": float(np.mean([bool(r["verified"]) for r in grows])),
            "mean_pesq_bm": mean_of("pesq_bm"),
            "mean_stoi_bm": mean_of("stoi_bm"),
        }
    return out


def write_results(output_dir: Path, rows: list[dict], summary: dict) -> None:
    """写 results.csv + summary.json。"""
    import csv
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
