#!/usr/bin/env python
"""实验 P0-4: Bootstrap 95% 置信区间

从已有 results.csv 计算 utterance 级 bootstrap 置信区间。

运行方法：
    python scripts/exp_p0_4_bootstrap_ci.py \
        --input runs/relmel_hifigan_main/results.csv \
        --output runs/relmel_hifigan_main/bootstrap_ci.json \
        --n-bootstrap 10000

预计时间：<2 分钟
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95) -> dict:
    """计算 bootstrap 置信区间"""
    n = len(data)
    means = np.array([np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)])
    alpha = 1 - confidence
    lower = np.percentile(means, alpha / 2 * 100)
    upper = np.percentile(means, (1 - alpha / 2) * 100)
    return {"mean": float(np.mean(data)), "ci_lower": float(lower), "ci_upper": float(upper), "n": int(n)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="results.csv 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"加载 {len(df)} 行数据")

    # 按攻击分组计算
    attacks = df["attack"].unique()
    result = {}

    for attack in attacks:
        attack_df = df[df["attack"] == attack]
        bit_acc = attack_df["bit_acc"].values
        verified = attack_df["verified"].astype(float).values

        result[attack] = {
            "bit_acc": bootstrap_ci(bit_acc, args.n_bootstrap, args.confidence),
            "verification_rate": bootstrap_ci(verified, args.n_bootstrap, args.confidence),
            "n_samples": int(len(attack_df)),
        }

        print(f"{attack:15s} ACC: {result[attack]['bit_acc']['mean']:.3f} "
              f"[{result[attack]['bit_acc']['ci_lower']:.3f}, {result[attack]['bit_acc']['ci_upper']:.3f}]")

    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\n✓ 写入 {args.output}")


if __name__ == "__main__":
    main()
