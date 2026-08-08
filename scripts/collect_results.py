#!/usr/bin/env python
"""collect_results.py — 把所有补充实验结果汇总成单一 Markdown 文件

这个文件就是"验收件"：跑完实验后把它发给我，我据此把数字写进论文。
所有需要的元信息（协议、样本量、SNR 定义、方法实现差异）都内嵌在文件里，
不需要额外解释。

运行：
    python scripts/collect_results.py \
        --results-dir runs/review_p0_20260807_0130 \
        --output runs/review_p0_20260807_0130/SUPPLEMENT_RESULTS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": str(e)}


def fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def section_sweep(out: list[str], summ: dict, vocoder_label: str) -> None:
    out.append(f"\n## P0-1: ACC-vs-PESQ 权衡曲线（{vocoder_label}）\n")
    out.append(f"- 样本量：{summ.get('limit')} 条，seed={summ.get('seed')}\n")
    out.append(f"- 说明：{summ.get('note', '')}\n")

    tc = summ.get("tradeoff_curve", {})
    attacks = ["noise20", "noise10", "noise5"]

    for method, label in (("rawmer", "RAWMER"), ("melshield", "MelShield")):
        pts = tc.get(method, [])
        if not pts:
            continue
        out.append(f"\n### {label}\n")
        head = "| α | clean PESQ | clean STOI | " + " | ".join(
            f"ACC@{a.replace('noise', '')}dB" for a in attacks
        ) + " | " + " | ".join(f"VR@{a.replace('noise','')}dB" for a in attacks) + " | |\n"
        out.append(head)
        out.append("|" + "---|" * (3 + 2 * len(attacks) + 1) + "\n")
        for p in pts:
            cells = [p["alpha"], fmt(p.get("clean_pesq")), fmt(p.get("clean_stoi"))]
            cells += [fmt(p.get(f"acc_{a}")) for a in attacks]
            cells += [fmt(p.get(f"vr_{a}")) for a in attacks]
            cells.append("**论文主工作点**" if p.get("is_paper_main_point") else "")
            out.append("| " + " | ".join(cells) + " |\n")

    # 找出 clean PESQ 相近的配对，这是审稿人真正要看的
    r_pts = tc.get("rawmer", [])
    m_pts = tc.get("melshield", [])
    if r_pts and m_pts:
        out.append("\n### 等质量配对（clean PESQ 最接近的 α 组合）\n")
        out.append("| RAWMER α | MelShield α | PESQ 差 | ACC@5dB (R) | ACC@5dB (M) | Δ |\n")
        out.append("|---|---|---|---|---|---|\n")
        for rp in r_pts:
            rq = rp.get("clean_pesq")
            if rq is None:
                continue
            best, bd = None, 1e9
            for mp in m_pts:
                mq = mp.get("clean_pesq")
                if mq is None:
                    continue
                if abs(rq - mq) < bd:
                    best, bd = mp, abs(rq - mq)
            if best is None or bd > 0.15:
                continue
            ra, ma = rp.get("acc_noise5"), best.get("acc_noise5")
            delta = f"{ra - ma:+.3f}" if (ra is not None and ma is not None) else "n/a"
            out.append(
                f"| {rp['alpha']} | {best['alpha']} | {bd:.3f} | "
                f"{fmt(ra)} | {fmt(ma)} | {delta} |\n"
            )
        out.append(
            "\n> 这张表直接回应「单一 α 工作点不可比」：如果 Δ 在所有等质量配对上"
            "都为正，则 RAWMER 的优势不依赖某个特定工作点。\n"
        )


def section_revocode(out: list[str], summ: dict) -> None:
    out.append("\n## P0-2: Re-vocoding 攻击\n")
    out.append(f"- 样本量：{summ.get('limit')} 条，seed={summ.get('seed')}\n")
    out.append(f"- 攻击者假设：{summ.get('attack_vocoder_note', '')}\n\n")
    agg = summ.get("by_method_attack", {})
    out.append("| Method | Attack | ACC | VR | PESQ | STOI |\n")
    out.append("|---|---|---|---|---|---|\n")
    for method, label in (("rawmer", "RAWMER"), ("melshield", "MelShield")):
        for atk in ("none", "revocode", "revocode_noise10"):
            c = agg.get(f"{method}|{atk}")
            if c:
                out.append(
                    f"| {label} | {atk} | {fmt(c['mean_bit_acc'])} | "
                    f"{fmt(c['verification_rate'])} | {fmt(c.get('mean_pesq_bm'), 2)} | "
                    f"{fmt(c.get('mean_stoi_bm'))} |\n"
                )
    out.append(
        "\n> `revocode` = 从水印波形重提 mel 后用公开 vocoder 重合成；\n"
        "> `revocode_noise10` = re-vocoding 后再叠加 10 dB 噪声（组合攻击）。\n"
    )


def section_band(out: list[str], summ: dict) -> None:
    out.append("\n## P0-3: Band-targeted noise\n")
    out.append(f"- 样本量：{summ.get('limit')} 条，seed={summ.get('seed')}\n")
    out.append(f"- 水印频段：{summ.get('band_hz', '')}\n")
    out.append(f"- **SNR 定义**：{summ.get('snr_definition', '')}\n\n")
    agg = summ.get("by_method_attack", {})
    out.append("| Method | SNR | band ACC | full-band ACC | 定向增益 | band VR | full VR |\n")
    out.append("|---|---|---|---|---|---|---|\n")
    for method, label in (("rawmer", "RAWMER"), ("melshield", "MelShield")):
        for tag in (20, 10, 5, 0):
            b = agg.get(f"{method}|band_noise{tag}")
            f = agg.get(f"{method}|noise{tag}")
            if not b:
                continue
            if f:
                gain = f["mean_bit_acc"] - b["mean_bit_acc"]
                out.append(
                    f"| {label} | {tag} dB | {fmt(b['mean_bit_acc'])} | "
                    f"{fmt(f['mean_bit_acc'])} | {gain:+.3f} | "
                    f"{fmt(b['verification_rate'])} | {fmt(f['verification_rate'])} |\n"
                )
            else:
                out.append(
                    f"| {label} | {tag} dB | {fmt(b['mean_bit_acc'])} | n/a | n/a | "
                    f"{fmt(b['verification_rate'])} | n/a |\n"
                )
    out.append("\n> 定向增益 > 0 表示把噪声集中在水印频段更有效地破坏了水印。\n")


def section_ci(out: list[str], results_dir: Path) -> None:
    files = sorted(results_dir.glob("*/bootstrap_ci.json"))
    if not files:
        return
    out.append("\n## P0-4: Bootstrap 95% 置信区间\n")
    for f in files:
        data = load(f)
        if not data or "_error" in data:
            continue
        out.append(f"\n### {f.parent.name}\n")
        widths = []
        out.append("| 条件 | ACC | 95% CI | VR | 95% CI | n |\n")
        out.append("|---|---|---|---|---|---|\n")
        for cond, st in sorted(data.items()):
            a, v = st.get("bit_acc", {}), st.get("verification_rate", {})
            if not a:
                continue
            half = (a["ci_upper"] - a["ci_lower"]) / 2
            widths.append(half)
            out.append(
                f"| {cond} | {fmt(a['mean'])} | "
                f"[{fmt(a['ci_lower'])}, {fmt(a['ci_upper'])}] | "
                f"{fmt(v.get('mean'))} | "
                f"[{fmt(v.get('ci_lower'))}, {fmt(v.get('ci_upper'))}] | "
                f"{st.get('n_samples')} |\n"
            )
        if widths:
            out.append(
                f"\n> 最大 CI 半宽 = {max(widths):.4f}"
                f"（论文可写「95% bootstrap CI 半宽 ≤ {max(widths):.3f}」）\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总补充实验结果为单一文件")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    out: list[str] = []
    out.append("# RAWMER 审稿补充实验结果\n\n")
    out.append(f"结果目录：`{rd}`\n\n")
    out.append(
        "> **给 Claude 的说明**：这个文件包含全部补充实验结果与协议元信息。\n"
        "> 把整个文件发给我，我会据此更新论文的 Table 1、Table 2 和 experiments.tex。\n"
        "> 重要：RAWMER 用 `RelMelMark`（差分编码 + 可靠对选择），MelShield 用\n"
        "> `MelShield` 类（绝对残差扩频），两者是不同实现，α 不可跨方法比较。\n\n"
    )
    out.append("---\n")

    found = 0

    for name, label in (("p0_1_hifigan", "HiFi-GAN"), ("p0_1_diffwave", "DiffWave")):
        s = load(rd / name / "summary.json")
        if s and "_error" not in s:
            section_sweep(out, s, label)
            found += 1

    s = load(rd / "p0_2_revocoding" / "summary.json")
    if s and "_error" not in s:
        section_revocode(out, s)
        found += 1

    s = load(rd / "p0_3_band_noise" / "summary.json")
    if s and "_error" not in s:
        section_band(out, s)
        found += 1

    section_ci(out, rd)

    # 缺失项与失败日志
    out.append("\n---\n\n## 运行状态\n\n")
    for name in ("p0_1_hifigan", "p0_1_diffwave", "p0_2_revocoding", "p0_3_band_noise"):
        d = rd / name
        if (d / "summary.json").is_file():
            n = len((d / "results.csv").read_text().splitlines()) - 1 if (d / "results.csv").is_file() else 0
            out.append(f"- ✓ `{name}`：完成，{n} 行原始数据\n")
        else:
            out.append(f"- ✗ `{name}`：未完成或已跳过\n")

    out.append("\n## 仍未跑的实验（P1/P2）\n\n")
    out.append("- 主观听测（MUSHRA/CMOS）—— 回应 PESQ 3.51 vs 4.0-4.5 的可听差距\n")
    out.append("- 反向消融：绝对残差编码 + 可靠对选择 —— 彻底分离两个机制\n")
    out.append("- 波形基线对齐到 2000 条 / 同 payload 长度\n")
    out.append("- float32 band-only 压缩行（很便宜，可坐实 1/4 比值）\n")
    out.append("- 端到端 TTS mel、多说话人（VCTK/LibriTTS）、重录/RIR 信道\n")

    Path(args.output).write_text("".join(out), encoding="utf-8")
    print(f"✓ 汇总写入 {args.output}（{found} 组实验有结果）")
    print(f"\n把这个文件发给我即可：{args.output}")


if __name__ == "__main__":
    main()
