# RAWMER 审稿补充实验 - GPU 需求与时间估算

## 推荐 GPU 配置

### 最低要求
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) 或同等性能
- **VRAM**: 至少 12GB（HiFi-GAN 推理 ~4GB，DiffWave ~8GB）
- **存储**: 50GB 可用空间（数据集 2.6GB + checkpoint 0.5GB + 结果 5GB + conda env 8GB）
- **内存**: 32GB RAM

### 推荐配置
- **GPU**: A100 (40GB/80GB) 或 V100 (32GB)
- **原因**: 
  - A100 比 3090 快 1.5-2×（Tensor Core 优化）
  - V100/A100 支持多任务并行（可同时跑 HiFi-GAN + DiffWave）
  - 更大 VRAM 允许更大 batch（虽然当前脚本是逐条推理）

### 云服务商参考价格（2024 年）
- **AWS**: `p3.2xlarge` (V100 16GB) ~$3.06/小时
- **GCP**: `a2-highgpu-1g` (A100 40GB) ~$3.67/小时
- **AutoDL**: V100 (32GB) ~¥3.5/小时，A100 (40GB) ~¥6/小时

---

## 实验时间估算

### P0 优先级实验（4 个）

| 实验 | 数据量 | GPU 时间（V100） | GPU 时间（A100） | 备注 |
|---|---|---|---|---|
| **P0-1a: ACC-vs-PESQ (HiFi-GAN)** | 500 × 7α × 4攻击 = 14k | 45 分钟 | 28 分钟 | 最重要 |
| **P0-1b: ACC-vs-PESQ (DiffWave)** | 500 × 7α × 4攻击 = 14k | 120 分钟 | 75 分钟 | DiffWave 慢 3× |
| **P0-2: Re-vocoding** | 500 × 2方法 = 1k | 35 分钟 | 22 分钟 | 单攻击 |
| **P0-3: Band noise** | 500 × 2方法 × 6攻击 = 6k | 40 分钟 | 25 分钟 | 含全频段对比 |
| **P0-4: Bootstrap CI** | 后处理 | <2 分钟 | <2 分钟 | CPU 计算 |
| **总计** | ~35k 推理 | **~4.2 小时** | **~2.5 小时** | 单 GPU 串行 |

### P1 优先级实验（估算，未写脚本）

| 实验 | 数据量 | GPU 时间（V100） | 备注 |
|---|---|---|---|
| 主观测试样本生成 | 20 clip × 2方法 × 10条件 | 10 分钟 | 仅生成音频 |
| 反向消融（abs+select） | 500 × 1方法 × 4攻击 | 20 分钟 | 需修改 relmel.py |
| 波形基线对齐 | 2000 × 2基线 × 4攻击 | 60 分钟 | AudioSeal/WavMark |
| float32 band-only | 500 × 1配置 × 4攻击 | 15 分钟 | 很便宜 |
| **P1 小计** | | **~1.8 小时** | |

### P2 优先级实验（估算）

| 实验 | 数据量 | GPU 时间 | 备注 |
|---|---|---|---|
| 端到端 TTS (Tacotron2+HiFi-GAN) | 500 × 2方法 × 4攻击 | 90 分钟 | 需 TTS checkpoint |
| 多说话人 (VCTK 前 10 人) | 1000 × 2方法 × 4攻击 | 80 分钟 | 跨说话人泛化 |
| RIR / 重录信道 | 500 × 2方法 × 5 RIR | 50 分钟 | 需 RIR 数据集 |
| Regeneration (扩散去噪) | 500 × 2方法 × 3强度 | 180 分钟 | 扩散模型慢 |
| **P2 小计** | | **~6.5 小时** | |

---

## 推荐执行策略

### 方案 A：单 GPU 串行（省钱）
- **配置**: V100 (32GB) 或 A100 (40GB)
- **时间**: P0 实验 4.2小时，P1+P2 共 8.3小时 = **总计 12.5 小时**
- **成本**: 
  - V100: 12.5h × $3/h = **~$38**
  - A100: 7.5h × $3.7/h = **~$28**（A100 更快更省）

### 方案 B：双 GPU 并行（推荐）
- **配置**: 2× V100 或 2× A100
- **时间**: P0 并行（HiFi-GAN 和 DiffWave 分开跑）= **~2.5 小时**，P1+P2 分配 = **总计 6 小时**
- **成本**: 
  - 2× V100: 6h × $6/h = **~$36**
  - 2× A100: 6h × $7.4/h = **~$44**

### 方案 C：最快交付（土豪）
- **配置**: 4× A100
- **时间**: 全部并行 = **~2 小时**
- **成本**: 2h × $14.8/h = **~$30**（时间优先）

---

## 检查点（Checkpoint）文件

### HiFi-GAN V1 (LJSpeech 22.05kHz)
- **来源**: https://github.com/jik876/hifi-gan
- **文件**: 
  - `generator_v1` (55MB)
  - `config.json` (1KB)
- **下载**:
  ```bash
  wget https://drive.google.com/uc?export=download&id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW \
       -O checkpoints/hifigan/generator_v1
  ```

### DiffWave (LJSpeech 22.05kHz)
- **来源**: https://github.com/lmnt-com/diffwave
- **文件**: `diffwave-ljspeech-22k.pt` (~180MB)
- **说明**: 官方 README 提供预训练权重下载链接

### AudioSeal / WavMark（P1 需要）
- AudioSeal: `pip install audioseal` 自动下载
- WavMark: https://github.com/wavmark/wavmark

---

## 最终建议

**如果预算有限，只跑 P0**：
- 租 1× A100 (40GB)
- 运行 `bash scripts/run_review_experiments.sh`
- 预计 **2.5 小时，成本 $10**
- 足以回应审稿人的 CRITICAL 要求

**如果想补全 P1**：
- 再加 1.8 小时 = **总计 4.3 小时，$16**

**如果要做完整评估（P0+P1+P2）**：
- 租 2× A100，并行跑 6 小时
- 或租 1× A100，串行跑 12 小时
- **成本 $20-45**

我的建议是**先跑 P0（$10），根据审稿人反馈决定是否补 P1/P2**。
