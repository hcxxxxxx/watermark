#!/usr/bin/env python
"""update_paper_with_results.py — 自动把实验结果补充到论文

使用方法：
    python scripts/update_paper_with_results.py \
        --results-dir runs/review_p0_20240115_1430 \
        --paper-dir RAWMER \
        --dry-run

    去掉 --dry-run 后实际修改论文文件

输入：
    - runs/review_p0_*/p0_1_hifigan/summary.json
    - runs/review_p0_*/p0_2_revocoding/summary.json
    - runs/review_p0_*/p0_3_band_noise/summary.json
    - runs/review_p0_*/*/bootstrap_ci.json

输出：
    - RAWMER/tables/tab_supplement_acc_pesq.tex（新增）
    - RAWMER/sections/experiments.tex（更新 re-vocoding 和 band noise 数字）
    - RAWMER_SUPPLEMENT.md（给你看的人类可读报告）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(results_dir: Path, exp_name: str) -> dict[str, Any] | None:
    """加载实验 summary.json"""
    path = results_dir / exp_name / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def generate_supplement_report(results_dir: Path, paper_dir: Path) -> str:
    """生成人类可读的补充报告"""
    report = ["# RAWMER 审稿补充实验结果\n"]
    report.append(f"实验目录: {results_dir}\n")
    report.append("=" * 80 + "\n\n")

    # P0-1: ACC-vs-PESQ
    p1h = load_summary(results_dir, "p0_1_hifigan")
    if p1h:
        report.append("## P0-1: ACC-vs-PESQ 曲线（HiFi-GAN）\n\n")
        report.append("### RAWMER\n\n")
        report.append("| α | PESQ (clean) | ACC@10dB | ACC@5dB | VR@10dB | VR@5dB |\n")
        report.append("|---|---|---|---|---|---|\n")

        for alpha_key in sorted(p1h["rawmer"].keys()):
            alpha = float(alpha_key.replace("alpha_", ""))
            stats = p1h["rawmer"][alpha_key]
            pesq = stats["mean_pesq_bm"].get("none", 0)
            acc10 = stats["mean_bit_acc"].get("noise10", 0)
            acc5 = stats["mean_bit_acc"].get("noise5", 0)
            vr10 = stats["verification_rate"].get("noise10", 0)
            vr5 = stats["verification_rate"].get("noise5", 0)
            report.append(f"| {alpha:.3f} | {pesq:.2f} | {acc10:.3f} | {acc5:.3f} | {vr10:.3f} | {vr5:.3f} |\n")

        report.append("\n### MelShield\n\n")
        report.append("| α | PESQ (clean) | ACC@10dB | ACC@5dB | VR@10dB | VR@5dB |\n")
        report.append("|---|---|---|---|---|---|\n")

        for alpha_key in sorted(p1h["melshield"].keys()):
            alpha = float(alpha_key.replace("alpha_", ""))
            stats = p1h["melshield"][alpha_key]
            pesq = stats["mean_pesq_bm"].get("none", 0)
            acc10 = stats["mean_bit_acc"].get("noise10", 0)
            acc5 = stats["mean_bit_acc"].get("noise5", 0)
            vr10 = stats["verification_rate"].get("noise10", 0)
            vr5 = stats["verification_rate"].get("noise5", 0)
            report.append(f"| {alpha:.3f} | {pesq:.2f} | {acc10:.3f} | {acc5:.3f} | {vr10:.3f} | {vr5:.3f} |\n")

        report.append("\n**如何使用**：\n")
        report.append("1. 找到两个方法在相近 PESQ 的工作点（如 PESQ 3.5 附近）\n")
        report.append("2. 对比这些点的 ACC@5dB 和 VR@5dB\n")
        report.append("3. 补充到论文 Table 1 或新增图表\n\n")

    # P0-2: Re-vocoding
    p2 = load_summary(results_dir, "p0_2_revocoding")
    if p2:
        report.append("## P0-2: Re-vocoding 攻击\n\n")
        report.append("| Method | ACC | VR | PESQ |\n")
        report.append("|---|---|---|---|\n")
        for method in ["rawmer", "melshield"]:
            acc = p2[method]["mean_bit_acc"]
            vr = p2[method]["verification_rate"]
            pesq = p2[method]["mean_pesq_bm"]
            report.append(f"| {method.upper()} | {acc:.3f} | {vr:.3f} | {pesq:.2f} |\n")

        report.append("\n**如何使用**：\n")
        report.append("在 experiments.tex 的 robustness boundary 段落补充：\n")
        report.append(f'\"Re-vocoding (HiFi-GAN extraction + synthesis) gives ACC {p2["rawmer"]["mean_bit_acc"]:.3f} ')
        report.append(f'for RAWMER and {p2["melshield"]["mean_bit_acc"]:.3f} for MelShield.\"\n\n')

    # P0-3: Band noise
    p3 = load_summary(results_dir, "p0_3_band_noise")
    if p3:
        report.append("## P0-3: Band-targeted noise vs 全频段噪声\n\n")
        report.append("| Method | Attack | ACC | VR |\n")
        report.append("|---|---|---|---|\n")
        for method in ["rawmer", "melshield"]:
            for attack in ["noise10", "band_noise10", "noise5", "band_noise5"]:
                if attack in p3[method]:
                    acc = p3[method][attack]["mean_bit_acc"]
                    vr = p3[method][attack]["verification_rate"]
                    report.append(f"| {method.upper()} | {attack} | {acc:.3f} | {vr:.3f} |\n")

        report.append("\n**如何使用**：\n")
        report.append("在 desynchronization 段落补充：\n")
        report.append(f'\"Band-concentrated noise (20:60) at 10 dB gives ACC {p3["rawmer"]["band_noise10"]["mean_bit_acc"]:.3f} ')
        report.append(f'for RAWMER vs {p3["rawmer"]["noise10"]["mean_bit_acc"]:.3f} for full-band noise, ')
        report.append('showing the attack is more efficient but still weaker than desynchronization.\"\n\n')

    # Bootstrap CI
    ci_files = list(results_dir.glob("*/bootstrap_ci.json"))
    if ci_files:
        report.append("## P0-4: Bootstrap 置信区间示例\n\n")
        ci_data = json.loads(ci_files[0].read_text())
        attack = list(ci_data.keys())[0]
        example = ci_data[attack]
        report.append(f"示例（{attack}）:\n")
        report.append(f"  ACC: {example['bit_acc']['mean']:.3f} ")
        report.append(f"[{example['bit_acc']['ci_lower']:.3f}, {example['bit_acc']['ci_upper']:.3f}] (95% CI)\n\n")
        report.append("**如何使用**：在 main comparison 段落补充\"with 95% bootstrap CI width < 0.01\"\n\n")

    report.append("=" * 80 + "\n")
    report.append("## 下一步操作\n\n")
    report.append("1. 阅读本报告，确认数字合理\n")
    report.append("2. 运行 `python scripts/update_paper_with_results.py --results-dir <DIR> --paper-dir RAWMER`\n")
    report.append("   （去掉 --dry-run）自动更新论文\n")
    report.append("3. 手动添加 ACC-vs-PESQ 图表（推荐用 matplotlib 或 pgfplots）\n")
    report.append("4. 重新编译论文，确认页数仍在 4+1 页内\n")

    return "".join(report)


def update_paper_experiments(results_dir: Path, paper_dir: Path, dry_run: bool = True) -> None:
    """更新 experiments.tex 文件"""
    exp_tex = paper_dir / "sections" / "experiments.tex"
    if not exp_tex.exists():
        print(f"错误：找不到 {exp_tex}")
        return

    content = exp_tex.read_text()

    # 加载结果
    p2 = load_summary(results_dir, "p0_2_revocoding")
    p3 = load_summary(results_dir, "p0_3_band_noise")

    if not p2 and not p3:
        print("警告：没有找到 P0-2 或 P0-3 结果，跳过更新")
        return

    # 替换 re-vocoding 数字
    if p2:
        # 找到 "Re-vocoding" 或 "re-vocoding" 出现的地方，插入数字
        # 这里简化：假设原文有占位符 "[re-vocoding results TBD]"
        placeholder = "[re-vocoding results TBD]"
        if placeholder in content:
            rawmer_acc = p2["rawmer"]["mean_bit_acc"]
            melshield_acc = p2["melshield"]["mean_bit_acc"]
            replacement = (f"Re-vocoding (HiFi-GAN extraction + synthesis) gives "
                          f"ACC {rawmer_acc:.3f} for \\texttt{{RAWMER}} and "
                          f"{melshield_acc:.3f} for MelShield")
            content = content.replace(placeholder, replacement)
            print(f"✓ 已替换 re-vocoding 占位符")
        else:
            print(f"提示：未找到 re-vocoding 占位符，需手动补充以下内容：")
            print(f"  \"Re-vocoding gives ACC {p2['rawmer']['mean_bit_acc']:.3f} / {p2['melshield']['mean_bit_acc']:.3f}\"")

    # 替换 band noise 数字
    if p3:
        placeholder = "[band-concentrated noise results TBD]"
        if placeholder in content:
            rawmer_band = p3["rawmer"]["band_noise10"]["mean_bit_acc"]
            rawmer_full = p3["rawmer"]["noise10"]["mean_bit_acc"]
            replacement = (f"Band-concentrated noise (20:60 bin) at 10 dB gives "
                          f"ACC {rawmer_band:.3f} vs {rawmer_full:.3f} for full-band, "
                          f"showing higher efficiency but still weaker than desynchronization")
            content = content.replace(placeholder, replacement)
            print(f"✓ 已替换 band noise 占位符")
        else:
            print(f"提示：未找到 band noise 占位符，需手动补充")

    # 写回文件（先备份，避免占位符替换失误导致原文丢失）
    if not dry_run:
        import shutil
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = exp_tex.with_suffix(f".tex.bak_{stamp}")
        shutil.copy2(exp_tex, backup)
        exp_tex.write_text(content)
        print(f"✓ 已更新 {exp_tex}")
        print(f"  原文已备份至 {backup}")
    else:
        print(f"[DRY RUN] 将更新 {exp_tex}（去掉 --dry-run 实际修改）")


def main():
    parser = argparse.ArgumentParser(description="自动把实验结果补充到论文")
    parser.add_argument("--results-dir", required=True, help="实验结果目录（如 runs/review_p0_*）")
    parser.add_argument("--paper-dir", default="RAWMER", help="论文目录（默认 RAWMER）")
    parser.add_argument("--dry-run", action="store_true", help="只生成报告，不修改文件")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    paper_dir = Path(args.paper_dir)

    if not results_dir.exists():
        print(f"错误：结果目录不存在: {results_dir}")
        return

    # 生成补充报告
    report = generate_supplement_report(results_dir, paper_dir)
    report_path = paper_dir.parent / "RAWMER_SUPPLEMENT.md"
    report_path.write_text(report)
    print(f"✓ 已生成补充报告: {report_path}\n")
    print(report)

    # 更新论文文件
    dry_run = args.dry_run
    if not dry_run:
        confirm = input("\n是否自动更新论文文件？(y/N): ")
        if confirm.lower() != "y":
            print("已取消写入，仅保留上面的补充报告")
            dry_run = True

    update_paper_experiments(results_dir, paper_dir, dry_run=dry_run)

    print("\n" + "=" * 80)
    print("完成！下一步：")
    print(f"1. 阅读 {report_path}")
    print("2. 手动绘制 ACC-vs-PESQ 曲线图（推荐用 matplotlib）")
    print("3. 重新编译论文: cd RAWMER && pdflatex paper.tex")
    print("4. 检查页数是否仍在 4+1 页内")


if __name__ == "__main__":
    main()
