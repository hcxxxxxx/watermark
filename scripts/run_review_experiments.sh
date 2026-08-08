#!/usr/bin/env bash
# run_review_experiments.sh — 审稿补充实验一键运行
#
# 用法：
#   bash scripts/run_review_experiments.sh                 # 完整 P0
#   bash scripts/run_review_experiments.sh --quick         # 20 条冒烟测试
#   bash scripts/run_review_experiments.sh --limit 200     # 自定义样本量
#   bash scripts/run_review_experiments.sh --skip-diffwave # 跳过 DiffWave
#
# 单张 A100 完整 P0 约 2.5 小时；--quick 约 5 分钟。

set -uo pipefail   # 故意不用 -e：单个实验失败不应中断其余实验

LIMIT=500
SKIP_DIFFWAVE=0
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)         QUICK=1; LIMIT=20; shift ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --skip-diffwave) SKIP_DIFFWAVE=1; shift ;;
        *) echo "未知参数：$1"; exit 2 ;;
    esac
done

BASE_DIR="runs/review_p0_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_DIR"

HG_CKPT="checkpoints/hifigan/generator_v1"
HG_CFG="checkpoints/hifigan/config.json"

echo "=============================================================="
echo "  RAWMER 审稿补充实验（P0 优先级）"
echo "  样本量: $LIMIT   输出: $BASE_DIR"
[[ $QUICK -eq 1 ]] && echo "  模式: QUICK 冒烟测试"
echo "=============================================================="

# ---------- 飞行前检查 ----------
echo
echo "[0/5] 飞行前检查"
if ! python scripts/preflight_check.py 2>&1 | tee "$BASE_DIR/preflight.log"; then
    echo
    echo "✗ 飞行前检查失败，已中止。修复上面的 FAIL 项后重跑。"
    exit 1
fi

run_step() {
    local name="$1"; shift
    echo
    echo "--------------------------------------------------------------"
    echo "$name"
    echo "--------------------------------------------------------------"
    if "$@" 2>&1 | tee "$BASE_DIR/${name%% *}.log"; then
        echo "✓ 完成：$name"
    else
        echo "✗ 失败：$name（继续后续实验，日志见 $BASE_DIR）"
    fi
}

# ---------- P0-1a: ACC-vs-PESQ (HiFi-GAN) ----------
run_step "[1/5] P0-1a ACC-vs-PESQ sweep (HiFi-GAN)" \
    python scripts/exp_p0_1_acc_vs_pesq_sweep.py \
        --config configs/relmel_hifigan.yaml \
        --vocoder hifigan \
        --vocoder-checkpoint "$HG_CKPT" \
        --vocoder-config "$HG_CFG" \
        --limit "$LIMIT" \
        --output-dir "$BASE_DIR/p0_1_hifigan"

# ---------- P0-1b: ACC-vs-PESQ (DiffWave) ----------
DW_CKPT=""
for cand in checkpoints/diffwave/*.pt external/diffwave/checkpoints/*.pt; do
    [[ -f "$cand" ]] && DW_CKPT="$cand" && break
done

if [[ $SKIP_DIFFWAVE -eq 1 ]]; then
    echo; echo "[2/5] P0-1b DiffWave —— 按 --skip-diffwave 跳过"
elif [[ -z "$DW_CKPT" ]]; then
    echo; echo "[2/5] P0-1b DiffWave —— 未找到 checkpoint，跳过"
elif [[ ! -f "configs/relmel_diffwave.yaml" ]]; then
    echo; echo "[2/5] P0-1b DiffWave —— 缺 configs/relmel_diffwave.yaml，跳过"
else
    run_step "[2/5] P0-1b ACC-vs-PESQ sweep (DiffWave)" \
        python scripts/exp_p0_1_acc_vs_pesq_sweep.py \
            --config configs/relmel_diffwave.yaml \
            --vocoder command \
            --limit "$LIMIT" \
            --output-dir "$BASE_DIR/p0_1_diffwave"
fi

# ---------- P0-2: Re-vocoding ----------
run_step "[3/5] P0-2 Re-vocoding 攻击" \
    python scripts/exp_p0_2_revocoding.py \
        --config configs/relmel_hifigan.yaml \
        --vocoder hifigan \
        --vocoder-checkpoint "$HG_CKPT" \
        --vocoder-config "$HG_CFG" \
        --limit "$LIMIT" \
        --output-dir "$BASE_DIR/p0_2_revocoding"

# ---------- P0-3: Band-targeted noise ----------
run_step "[4/5] P0-3 Band-targeted noise sweep" \
    python scripts/exp_p0_3_band_noise_sweep.py \
        --config configs/relmel_hifigan.yaml \
        --vocoder hifigan \
        --vocoder-checkpoint "$HG_CKPT" \
        --vocoder-config "$HG_CFG" \
        --limit "$LIMIT" \
        --output-dir "$BASE_DIR/p0_3_band_noise"

# ---------- P0-4: Bootstrap CI ----------
echo
echo "--------------------------------------------------------------"
echo "[5/5] P0-4 Bootstrap 95% 置信区间"
echo "--------------------------------------------------------------"
for exp in p0_1_hifigan p0_1_diffwave p0_2_revocoding p0_3_band_noise; do
    csv="$BASE_DIR/$exp/results.csv"
    if [[ -f "$csv" ]]; then
        python scripts/exp_p0_4_bootstrap_ci.py \
            --input "$csv" \
            --output "$BASE_DIR/$exp/bootstrap_ci.json" \
            --n-bootstrap 10000 \
            2>&1 | tail -6
    fi
done

# ---------- 汇总成单一文件 ----------
echo
echo "=============================================================="
echo "  汇总"
echo "=============================================================="
python scripts/collect_results.py \
    --results-dir "$BASE_DIR" \
    --output "$BASE_DIR/SUPPLEMENT_RESULTS.md"

echo
echo "全部完成。"
echo
echo "  ▸ 把这一个文件发给我，我就能把结果写进论文："
echo "      $BASE_DIR/SUPPLEMENT_RESULTS.md"
echo
