#!/usr/bin/env bash
# setup_server.sh — 在全新租用服务器上配置 RAWMER 实验环境
#
# 使用方法：
#   chmod +x setup_server.sh
#   ./setup_server.sh
#
# 前提：
#   - 已安装 Miniconda 或 Anaconda
#   - CUDA 12.8 驱动（脚本安装 cu128 wheel）
#   - 足够磁盘空间（数据集 ~2.6GB + checkpoint ~200MB + conda env ~8GB）

set -e  # 遇错即停

echo "=== RAWMER 实验环境配置 ==="
echo

# ========== 1. 克隆仓库 ==========
# 若脚本已在仓库内运行（存在 melshield/ 目录），就地使用；否则克隆
if [ -d "melshield" ] && [ -f "environment.yml" ]; then
    echo "[1/7] 已在仓库内，跳过克隆"
elif [ -d "watermark/melshield" ]; then
    echo "[1/7] 仓库已存在，进入目录"
    cd watermark
else
    echo "[1/7] 克隆仓库..."
    # TODO: 替换为你的实际仓库地址
    git clone "${RAWMER_REPO_URL:?请先设置 RAWMER_REPO_URL 环境变量，或手动改这一行}" watermark
    cd watermark
fi

# ========== 2. 创建 conda 环境 ==========
echo "[2/7] 创建 conda 环境 melshield（Python 3.10）..."
if conda env list | grep -q "^melshield "; then
    echo "环境已存在，跳过创建"
else
    conda env create -f environment.yml
fi

echo "[3/7] 激活环境并安装依赖..."
eval "$(conda shell.bash hook)"
conda activate melshield

# 确认 torch 安装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ========== 3. 下载 LJSpeech 数据集 ==========
echo "[4/7] 下载 LJSpeech 数据集（~2.6GB，需时 3-8 分钟）..."
mkdir -p data
if [ ! -d "data/LJSpeech-1.1" ]; then
    cd data
    wget -q --show-progress https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    echo "解压中..."
    tar -xjf LJSpeech-1.1.tar.bz2
    rm LJSpeech-1.1.tar.bz2
    cd ..
    echo "数据集就绪：$(find data/LJSpeech-1.1/wavs -name '*.wav' | wc -l) 条音频"
else
    echo "数据集已存在"
fi

# ========== 4. 下载 HiFi-GAN checkpoint ==========
echo "[5/7] 下载 HiFi-GAN V1 checkpoint（LJSpeech 22.05kHz）..."
mkdir -p checkpoints/hifigan
cd checkpoints/hifigan

if [ ! -f "generator_v1" ]; then
    wget -q --show-progress https://drive.google.com/uc?export=download&id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW -O generator_v1
fi

if [ ! -f "config.json" ]; then
    cat > config.json <<'EOJ'
{
  "resblock": "1",
  "num_gpus": 1,
  "batch_size": 16,
  "learning_rate": 0.0002,
  "adam_b1": 0.8,
  "adam_b2": 0.99,
  "lr_decay": 0.999,
  "seed": 1234,

  "upsample_rates": [8,8,2,2],
  "upsample_kernel_sizes": [16,16,4,4],
  "upsample_initial_channel": 512,
  "resblock_kernel_sizes": [3,7,11],
  "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],

  "segment_size": 8192,
  "num_mels": 80,
  "num_freq": 1025,
  "n_fft": 1024,
  "hop_size": 256,
  "win_size": 1024,

  "sampling_rate": 22050,

  "fmin": 0,
  "fmax": 8000,
  "fmax_for_loss": null,

  "num_workers": 4,

  "dist_config": {
    "dist_backend": "nccl",
    "dist_url": "tcp://localhost:54321",
    "world_size": 1
  }
}
EOJ
fi
cd ../..

# ========== 5. 下载 DiffWave checkpoint ==========
echo "[6/7] 下载 DiffWave checkpoint（LJSpeech 22.05kHz）..."
mkdir -p checkpoints/diffwave
cd checkpoints/diffwave

if [ ! -f "diffwave-ljspeech-22k.pt" ]; then
    # 示例链接，替换为实际地址（如 https://github.com/lmnt-com/diffwave）
    echo "警告：DiffWave checkpoint 需手动下载并放置于 checkpoints/diffwave/diffwave-ljspeech-22k.pt"
    echo "  参考：https://github.com/lmnt-com/diffwave"
fi
cd ../..

# ========== 6. 烟雾测试 ==========
echo "[7/7] 烟雾测试：运行 10 条数据..."
SMOKE_DIR="runs/smoke_test_$(date +%Y%m%d_%H%M%S)"
python scripts/run_relmel_ljspeech.py \
  --config configs/relmel_hifigan.yaml \
  --vocoder mel \
  --limit 10 \
  --output-dir "$SMOKE_DIR"

if [ -f "$SMOKE_DIR/summary.json" ]; then
    echo
    echo "✓ 环境配置完成！"
    echo
    echo "快速测试命令："
    echo "  conda activate melshield"
    echo "  python scripts/run_relmel_ljspeech.py --config configs/relmel_hifigan.yaml --vocoder hifigan --vocoder-checkpoint checkpoints/hifigan/generator_v1 --vocoder-config checkpoints/hifigan/config.json --limit 100 --attacks none"
    echo
    echo "运行完整实验套件："
    echo "  bash scripts/run_review_experiments.sh"
else
    echo
    echo "✗ 烟雾测试失败，请检查日志"
    exit 1
fi
