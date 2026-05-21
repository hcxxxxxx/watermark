# RelMel 论文实验结果汇总

本文档只保留当前可以用于论文主文或补充材料的结果。默认数据集为 LJSpeech，随机抽样种子为 2026。表中的 `none` 表示无攻击条件下的比特准确率，`PESQ` 和 `STOI` 均为水印音频相对原始音频的客观音质指标。

## 结论速览

- HiFi-GAN 条件下，RelMel 可靠候选对版本在干净 PESQ 约 3.51 时，显著优于本地复现的 MelShield。
- DiffWave 条件下，RelMel 当前主配置在干净 PESQ 约 3.54 时，明显优于本地复现的 MelShield，但 MelShield 本地复现的干净 PESQ 更高，因此还需要做一个 MelShield + DiffWave 的质量边缘扫描。
- MelShield 论文 reported 数据只能作为外部参考；最公平的主结论应优先基于同一代码、同一声码器、同一攻击、同一指标下的本地复现实验。

## 主对比表

| 声码器 | 方法 | 来源 | 样本数 | 干净 PESQ | none | noise20 | noise10 | noise5 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| HiFi-GAN | MelShield | 原文 reported | 论文 | 不直接匹配 | 1.0000 | 1.0000 | 0.7815 | 0.7052 | 外部参考 |
| HiFi-GAN | MelShield | 本地复现，alpha 0.05，band 20:60 | 100 | 3.5069 | 1.0000 | 0.9072 | 0.7097 | 0.6391 | 质量匹配基线 |
| HiFi-GAN | MelShield | 本地复现，alpha 0.055，band 20:60 | 100 | 3.4324 | 1.0000 | 0.9269 | 0.7325 | 0.6475 | 强度更高，但低于 3.5 PESQ |
| HiFi-GAN | RelMel | 可靠候选对，pair_candidates 16 | 500 | 3.5102 | 0.9999 | 0.9946 | 0.9455 | 0.8648 | 当前 HiFi-GAN 主结果 |
| DiffWave | MelShield | 原文 reported | 论文 | 不直接匹配 | 1.0000 | 0.9834 | 0.7788 | 0.7006 | 外部参考 |
| DiffWave | MelShield | 本地复现，alpha 0.05，band 20:56 | 500 | 3.8779 | 0.9996 | 0.9383 | 0.7725 | 0.6824 | 当前本地基线，音质偏高 |
| DiffWave | RelMel | alpha 0.35，mask_floor 0.20，boundary_margin 0.01 | 500 | 3.5443 | 0.9998 | 0.9914 | 0.9225 | 0.8308 | 当前 DiffWave 主结果 |

## RelMel 主配置

### HiFi-GAN 主结果

配置：`alpha=0.435`，`band=20:60`，`mask_floor=0.20`，`boundary_margin=0.01`，`block_frames=8`，`block_stride=8`，`bits_per_block=6`，`pair_bins=6`，`pair_candidates=16`，`detector_mode=plain`。

| 攻击 | 比特准确率 | 验证率 | PESQ | STOI |
|---|---:|---:|---:|---:|
| none | 0.9999 | 1.000 | 3.5102 | 0.9685 |
| mp3 | 0.9999 | 1.000 | 3.5093 | 0.9685 |
| aac | 0.9999 | 1.000 | 3.4918 | 0.9683 |
| scale | 0.9999 | 1.000 | 3.5102 | 0.9685 |
| rs16 | 0.9999 | 1.000 | 3.5102 | 0.9685 |
| bandpass | 0.9999 | 1.000 | 3.5299 | 0.9632 |
| lowpass | 0.9998 | 1.000 | 3.5185 | 0.9678 |
| noise20 | 0.9946 | 1.000 | 1.5574 | 0.9527 |
| noise10 | 0.9455 | 0.990 | 1.0994 | 0.8991 |
| noise5 | 0.8648 | 0.932 | 1.0416 | 0.8430 |
| echo | 0.9983 | 1.000 | 1.7726 | 0.9177 |

### DiffWave 主结果

配置：`alpha=0.35`，`band=20:60`，`mask_floor=0.20`，`boundary_margin=0.01`，`block_frames=8`，`block_stride=8`，`bits_per_block=6`，`pair_bins=6`，`pair_candidates=16`，`detector_mode=plain`。DiffWave 固定采样种子为 0。

| 攻击 | 比特准确率 | 验证率 | PESQ | STOI |
|---|---:|---:|---:|---:|
| none | 0.9998 | 1.000 | 3.5443 | 0.9612 |
| mp3 | 0.9998 | 1.000 | 3.5428 | 0.9612 |
| aac | 0.9998 | 1.000 | 3.5191 | 0.9609 |
| scale | 0.9998 | 1.000 | 3.5443 | 0.9612 |
| rs16 | 0.9998 | 1.000 | 3.5443 | 0.9612 |
| bandpass | 0.9998 | 1.000 | 3.5227 | 0.9559 |
| lowpass | 0.9989 | 1.000 | 3.6133 | 0.9612 |
| noise20 | 0.9914 | 1.000 | 1.4930 | 0.9438 |
| noise10 | 0.9225 | 0.984 | 1.0787 | 0.8932 |
| noise5 | 0.8308 | 0.852 | 1.0351 | 0.8437 |
| echo | 0.9971 | 1.000 | 1.6116 | 0.9128 |

## MelShield 对照结果

### DiffWave 原文 reported

该表来自 MelShield 论文的 DiffWave 鲁棒性表格。它不是本地同协议复现，只能作为外部参考。

| 攻击 | 比特准确率 | PESQ | STOI | MOS |
|---|---:|---:|---:|---:|
| mp3 | 1.0000 | 3.4077 | 0.9602 | 3.6170 |
| aac | 1.0000 | 3.3945 | 0.9601 | 3.6097 |
| scale | 1.0000 | 3.4079 | 0.9602 | 3.6163 |
| rs16 | 1.0000 | 3.4747 | 0.9553 | 3.5952 |
| bandpass | 1.0000 | 3.4194 | 0.9600 | 3.5546 |
| lowpass | 1.0000 | 3.4089 | 0.9602 | 3.6499 |
| noise20 | 0.9834 | 1.1136 | 0.8961 | 2.9502 |
| noise10 | 0.7788 | 1.0723 | 0.8913 | 2.5550 |
| noise5 | 0.7006 | 1.0334 | 0.8414 | 2.3595 |
| echo | 1.0000 | 1.3824 | 0.9133 | 3.0473 |

### DiffWave 本地复现

配置：`alpha=0.05`，`band=20:56`，`mask_floor=0.05`，`energy_gamma=0.75`，`boundary_margin=0.02`，`threshold=0.61`，`align_max_shift=12`，`headroom=0.0`。

| 攻击 | 比特准确率 | 验证率 | PESQ | STOI |
|---|---:|---:|---:|---:|
| none | 0.9996 | 1.000 | 3.8779 | 0.9662 |
| mp3 | 0.9994 | 1.000 | 3.8770 | 0.9661 |
| aac | 0.9995 | 1.000 | 3.8545 | 0.9659 |
| scale | 0.9996 | 1.000 | 3.8779 | 0.9662 |
| rs16 | 0.9982 | 1.000 | 3.8779 | 0.9662 |
| bandpass | 0.9976 | 1.000 | 3.8782 | 0.9610 |
| lowpass | 0.9869 | 1.000 | 3.9093 | 0.9659 |
| noise20 | 0.9383 | 0.998 | 1.4972 | 0.9488 |
| noise10 | 0.7725 | 0.940 | 1.0779 | 0.8983 |
| noise5 | 0.6824 | 0.800 | 1.0349 | 0.8478 |
| echo | 0.9945 | 1.000 | 1.6347 | 0.9167 |

相同 random500、全攻击协议下，RelMel + DiffWave 相比本地复现 MelShield + DiffWave：`noise20` 提升 5.3 个百分点，`noise10` 提升 15.0 个百分点，`noise5` 提升 14.8 个百分点。不过 MelShield 这组配置干净 PESQ 更高，因此还应补跑质量边缘版本。

## 关键消融实验

### 可靠候选对数量消融

固定配置：HiFi-GAN，`alpha=0.435`，`band=20:60`，`mask_floor=0.20`，`boundary_margin=0.01`，random500，攻击为 `none noise20 noise10 noise5`。

| pair_candidates | 干净 PESQ | none | noise20 | noise10 | noise5 | 说明 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 3.5485 | 0.9999 | 0.9459 | 0.7674 | 0.6684 | 随机候选对基线 |
| 4 | 3.5299 | 0.9999 | 0.9873 | 0.8894 | 0.7845 | 鲁棒性明显提升 |
| 8 | 3.5184 | 0.9999 | 0.9918 | 0.9232 | 0.8315 | 强配置 |
| 16 | 3.5102 | 0.9999 | 0.9952 | 0.9439 | 0.8607 | 当前主配置 |

结论：`pair_candidates` 从 1 增加到 16 后，`noise20` 提升 4.9 个百分点，`noise10` 提升 17.7 个百分点，`noise5` 提升 19.2 个百分点；PESQ 只从 3.5485 降到 3.5102。

### 多随机种子稳定性

固定主配置，random500，攻击为 `none noise20 noise10 noise5`。

| 种子 | 干净 PESQ | none | noise20 | noise10 | noise5 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 3.5102 | 0.9999 | 0.9952 | 0.9439 | 0.8607 |
| 2027 | 3.5108 | 0.9999 | 0.9950 | 0.9411 | 0.8588 |
| 2028 | 3.5164 | 0.9998 | 0.9938 | 0.9391 | 0.8574 |
| 平均 | 3.5125 | 0.9999 | 0.9946 | 0.9413 | 0.8590 |

## DiffWave 调参摘要

RelMel + DiffWave 的有用强度区间集中在 `alpha=0.30` 到 `0.35`。当前选择 `alpha=0.35, mask_floor=0.20, boundary_margin=0.01`，因为 random500 全攻击下干净 PESQ 为 3.5443，同时 `noise10=0.9225`、`noise5=0.8308`。

| alpha | mask_floor | boundary_margin | 干净 PESQ | none | noise20 | noise10 | noise5 | 说明 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.30 | 0.25 | 0.005 | 3.7717 | 1.0000 | 0.9820 | 0.8922 | 0.8000 | 音质余量较大 |
| 0.34 | 0.25 | 0.005 | 3.5557 | 1.0000 | 0.9867 | 0.9109 | 0.8125 | 接近质量边缘 |
| 0.35 | 0.20 | 0.010 | 3.5363 | 1.0000 | 0.9883 | 0.9094 | 0.8289 | pilot 推荐点 |
| 0.38 | 0.25 | 0.005 | 3.3374 | 1.0000 | 0.9938 | 0.9297 | 0.8281 | 低于 3.5 PESQ |

## 下一步实验命令

### MelShield + DiffWave 质量边缘扫描

目标：寻找干净 PESQ 刚好高于 3.5 的 MelShield + DiffWave 配置，再用该配置的 `noise20`、`noise10`、`noise5` 与 RelMel + DiffWave 对比。

后台运行版本：

```bash
mkdir -p logs

nohup python scripts/grid_melshield_ljspeech.py \
  --config configs/melshield_diffwave.yaml \
  --suite custom \
  --output-dir runs/melshield_diffwave_quality_edge_random40_seed2026 \
  --limit 40 \
  --sample-mode random \
  --seed 2026 \
  --attacks none noise20 noise10 noise5 \
  --device cuda \
  --vocoder command \
  --vocoder-command "python scripts/diffwave_vocoder.py external/diffwave/checkpoints/diffwave-ljspeech-22kHz-1000578.pt {mel_npy} --output {audio_wav} --device cuda --fast --seed 0" \
  --alpha-grid 0.055 0.060 0.065 0.070 0.075 0.080 \
  --band-grid 20:56 20:60 \
  --mask-floor-grid 0.05 0.075 \
  --energy-gamma-grid 0.75 \
  --boundary-margin-grid 0.02 \
  --threshold-grid 0.61 \
  --align-max-shift-grid 12 \
  --headroom-grid 0.0 \
  --quality-floor 3.5 \
  --noise20-weight 1.0 \
  --quality-weight 2.5 \
  --keep-candidate-results \
  > logs/melshield_diffwave_quality_edge_random40_seed2026.log 2>&1 &
```

查看日志：

```bash
tail -f logs/melshield_diffwave_quality_edge_random40_seed2026.log
```

选择规则：

- 先保留 `none_pesq_bm >= 3.5` 的候选。
- 在这些候选中，优先选 `none_pesq_bm` 最接近 3.5 且仍高于 3.5 的配置。
- 选定后再用相同参数跑 random500 全攻击。

## 写论文时的口径

- reported 数据和本地复现数据要分开表述，不能混成同一个公平协议。
- HiFi-GAN 的主对比已经基本完整：RelMel 在相近干净 PESQ 下明显强于本地复现 MelShield。
- DiffWave 的 RelMel 结果已经很强，但为了更严谨，还需要补一个 MelShield + DiffWave 的质量边缘复现。
- 方法贡献建议表述为：基于块级相对 Mel 关系的可靠候选对选择机制，提高了水印在加性噪声下的稳定性。
