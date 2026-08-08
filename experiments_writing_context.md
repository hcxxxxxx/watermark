# RAWMER Experiments Writing Context

本文档汇总仓库中对论文 experiments 章节写作最有用的信息。它面向“把实验章节交给另一个模型或合作者撰写”的场景，重点整理实验设置、数据集、攻击、指标、主结果、baseline、消融、负控、reference 压缩、片段验证、跨声码器设置和写作口径。

相关源文件：

- `core_idea.md`
- `paper_experiment_results.md`
- `README.md`
- `configs/relmel_hifigan.yaml`
- `configs/relmel_diffwave.yaml`
- `configs/melshield_hifigan_official.yaml`
- `configs/melshield_diffwave.yaml`
- `melshield/datasets.py`
- `melshield/metrics.py`
- `melshield/attacks.py`
- `scripts/run_relmel_ljspeech.py`
- `scripts/run_melshield_ljspeech.py`
- `scripts/grid_relmel_ljspeech.py`
- `scripts/grid_melshield_ljspeech.py`
- `scripts/grid_waveform_baselines_ljspeech.py`
- `scripts/eval_relmel_reference_controls_ljspeech.py`
- `scripts/eval_relmel_reference_compression_ljspeech.py`
- `scripts/eval_relmel_fragments_ljspeech.py`
- `scripts/eval_relmel_blind_fragments_ljspeech.py`
- `scripts/run_relmel_core_random2000_experiments.sh`
- `scripts/run_relmel_reference_random2000_experiments.sh`
- `scripts/run_reference_compression_uint_sweep.sh`
- `scripts/run_diffwave_paper_suite.sh`

## 1. 论文实验主线

RAWMER 是 reference-assisted mel-domain speech watermarking 方法。实验章节应围绕以下问题组织：

1. 在相近无攻击音质下，RAWMER 是否比同范式 MelShield baseline 更抗攻击，尤其是加性噪声？
2. 该结论是否在 HiFi-GAN 和 DiffWave 两种声码器下成立？
3. RAWMER 与 waveform-domain baselines AudioSeal/WavMark 的关系是什么？
4. Reference-assisted verification 是否容易产生假阳性？
5. Reference 保存成本是否可接受，是否可以压缩？
6. 若只获得局部音频片段，是否还能验证？
7. 方法的边界在哪里，尤其是 speed/pitch 等同步破坏？

推荐 experiments 章节结构：

1. Experimental setup
2. Main comparison with MelShield
3. Comparison with waveform baselines
4. Reference-assisted verification controls
5. Reference compression
6. Partial-clip and blind-fragment verification
7. Ablations and robustness boundary

## 2. Dataset and Sampling Protocol

数据集默认使用 LJSpeech，路径约定来自 `melshield/datasets.py`：

```text
data/LJSpeech-1.1/
  metadata.csv
  wavs/
    LJ001-0001.wav
    ...
```

采样方式：

- `sample_mode=first`：按 metadata 顺序取前 `limit` 条。
- `sample_mode=random`：使用 Python `random.Random(seed).shuffle` 打乱后取前 `limit` 条。
- 论文主结果默认使用 `sample_mode=random`，`seed=2026`。
- DiffWave 的 vocoder sampling seed 固定为 `0`。

常见样本规模：

- 主对比：random2000。
- Waveform baselines：random500。
- Reference compression 主表：random2000。
- Reference compression uint sweep：random500。
- Blind fragments：random500 或 random2000，主文可优先使用 blind search 的结果。
- Pair candidate ablation：random500。
- 多随机种子稳定性：random500，seed 2026/2027/2028。

## 3. Mel Frontend and Vocoder Settings

### HiFi-GAN Mel Configuration

来自 `configs/relmel_hifigan.yaml` 和 `configs/melshield_hifigan_official.yaml`：

```yaml
sample_rate: 22050
n_fft: 1024
hop_length: 256
win_length: 1024
n_mels: 80
f_min: 0.0
f_max: 8000.0
power: 1.0
eps: 1.0e-5
```

HiFi-GAN checkpoint 默认：

```text
checkpoints/hifigan/config.json
checkpoints/hifigan/generator_v1
```

### DiffWave Mel Configuration

来自 `configs/relmel_diffwave.yaml` 和 `configs/melshield_diffwave.yaml`：

```yaml
sample_rate: 22050
n_fft: 1024
hop_length: 256
win_length: 1024
n_mels: 80
f_min: 20.0
f_max: null
power: 1.0
eps: 1.0e-5
center: true
normalized: true
norm: null
mel_scale: htk
```

DiffWave 使用 command vocoder wrapper：

```text
python scripts/diffwave_vocoder.py external/diffwave/checkpoints/diffwave-ljspeech-22kHz-1000578.pt {mel_npy} --output {audio_wav} --device cuda --fast --seed 0
```

写作时可说明：HiFi-GAN 和 DiffWave 使用各自兼容的 Mel frontend，以匹配对应声码器预处理。

## 4. RAWMER Main Configuration

主实验配置来自 `paper_experiment_results.md` 和运行脚本。

### HiFi-GAN RAWMER 主配置

```text
alpha = 0.435
band = 20:60
threshold = 0.75
block_frames = 8
block_stride = 8
min_block_frames = 8
bits_per_block = 6
pair_bins = 6
pair_candidates = 16
detector_mode = plain
mask_floor = 0.20
energy_gamma = 0.5
boundary_margin = 0.01
align_max_shift = 12
payload_bits = 32
ecc_repeat = 1
```

### DiffWave RAWMER 主配置

```text
alpha = 0.355
band = 20:60
threshold = 0.75
block_frames = 8
block_stride = 8
min_block_frames = 8
bits_per_block = 6
pair_bins = 6
pair_candidates = 16
detector_mode = plain
mask_floor = 0.20
energy_gamma = 0.5
boundary_margin = 0.01
align_max_shift = 12
payload_bits = 32
ecc_repeat = 1
```

注意：`configs/relmel_hifigan.yaml` 中的默认值不是最终论文主配置，论文主结果由命令行 grid/run 脚本覆盖。

## 5. MelShield Baseline Configuration

MelShield 是同范式 reference-assisted mel-domain baseline，代码在 `melshield/watermark.py` 和 `scripts/run_melshield_ljspeech.py`。

### HiFi-GAN 质量匹配 MelShield

主对照配置：

```text
alpha = 0.05
band = 20:60
mask_floor = 0.05
energy_gamma = 0.75
boundary_margin = 0.02
threshold = 0.61
align_max_shift = 12
headroom = 0.0
payload_bits = 32
```

### DiffWave 质量匹配 MelShield

主对照配置：

```text
alpha = 0.06
band = 20:60
mask_floor = 0.05
energy_gamma = 0.75
boundary_margin = 0.02
threshold = 0.61
align_max_shift = 12
headroom = 0.0
payload_bits = 32
```

写作注意：

- MelShield 原文 reported 数字只能作为外部参考。
- 公平主结论应优先基于本仓库同一数据、同一声码器、同一攻击、同一指标下的本地复现 MelShield。
- HiFi-GAN 和 DiffWave 的 quality-matched baseline 应分别和对应 RAWMER 主结果比较。

## 6. Attacks

攻击实现集中在 `melshield/attacks.py`。

常规攻击：

| attack | 含义 |
|---|---|
| `none` | 无攻击 |
| `mp3` | MP3 128 kbps |
| `aac` | AAC 96 kbps |
| `scale` | 幅度缩放，gain=0.8 |
| `rs16` | 重采样到 16 kHz |
| `bandpass` | 300 Hz 到 8 kHz 带通 |
| `lowpass` | 3 kHz 低通 |
| `noise20` | 20 dB SNR 加性高斯噪声 |
| `noise10` | 10 dB SNR 加性高斯噪声 |
| `noise5` | 5 dB SNR 加性高斯噪声 |
| `echo` | 80 ms delay, decay 0.35 |

扩展攻击：

| attack | 含义 |
|---|---|
| `mp3_64` | MP3 64 kbps |
| `aac_48` | AAC 48 kbps |
| `rs8` | 重采样到 8 kHz |
| `rs24` | 重采样到 24 kHz |
| `lowpass2k` | 2 kHz 低通 |
| `bandpass_wide` | 100 Hz 到 7 kHz 带通 |
| `clip05` | 幅度裁剪到 ±0.5 |
| `quant8` | 8-bit 均匀量化 |
| `reverb` | 120 ms delay, decay 0.55 |
| `noise0` | 0 dB SNR 极强噪声 |
| `speed090/speed110` | 0.90/1.10 倍速 |
| `speed095/speed105` | 0.95/1.05 倍速 |
| `speed097/speed103` | 0.97/1.03 倍速 |
| `pitch_up/down` | 约 ±100 cents |
| `pitch_up50/down50` | ±50 cents |
| `pitch_up25/down25` | ±25 cents |

写作口径：

- 常规攻击是主比较协议。
- 扩展攻击用于说明现代攻击和边界。
- speed/pitch 攻击改变时间或频率同步关系，是当前方法主要弱点。

## 7. Metrics

### Bit Accuracy / ACC

恢复 payload 与 expected payload 的逐比特平均正确率。

### Verification Rate / VR

样本级通过比例。若某条样本的 bit accuracy 达到阈值，则验证通过。

RAWMER 主实验：

```text
payload_bits = 32
threshold = 0.75
```

即约至少 24/32 bits 正确即可通过。

MelShield 本地复现：

```text
threshold = 0.61
```

### Confidence

RAWMER 中 confidence 是 score 绝对值的平均强度，表示检测到的水印方向信号强弱。不能单独作为归属判断依据。尤其 wrong_payload 中 confidence 可接近正例，但 bit accuracy/VR 回到随机水平。

### PESQ and STOI

来自 `melshield/metrics.py`：

- `compare_audio` 会把 reference 和 degraded audio 重采样到 16 kHz。
- PESQ 使用 `pesq(..., "wb")`。
- STOI 使用 `pystoi.stoi(..., extended=False)`。
- 如果依赖不可用或计算失败，返回 `None`。

实验表中通常报告相对 benchmark audio 的 `pesq_bm` 和 `stoi_bm`，其中 benchmark audio 是同一声码器从 clean Mel 合成的无水印音频。

写作注意：

- `none` 条件下的 PESQ 用于衡量水印本身音质损伤。
- `noise20/noise10/noise5` 下 PESQ 低主要来自攻击本身，不应作为水印质量失败来解释。

## 8. Main Comparison: RAWMER vs MelShield

### Main Summary Table

| Vocoder | Method | Source | N | Clean PESQ | none | noise20 | noise10 | noise5 | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| HiFi-GAN | MelShield | paper reported | paper | not matched | 1.0000 | 1.0000 | 0.7815 | 0.7052 | external reference |
| HiFi-GAN | MelShield | local, alpha 0.05, band 20:60 | 2000 | 3.5100 | 0.9993 | 0.9028 | 0.7129 | 0.6328 | quality-matched |
| HiFi-GAN | RAWMER | reliable pairs, C=16 | 2000 | 3.5141 | 0.9999 | 0.9942 | 0.9408 | 0.8587 | main result |
| DiffWave | MelShield | paper reported | paper | not matched | 1.0000 | 0.9834 | 0.7788 | 0.7006 | external reference |
| DiffWave | MelShield | local high-quality | 500 | 3.8779 | 0.9996 | 0.9383 | 0.7725 | 0.6824 | high PESQ, not strict match |
| DiffWave | MelShield | local, alpha 0.061, band 20:60 | 2000 | 3.5149 | 0.9998 | 0.9685 | 0.8225 | 0.7219 | quality-matched |
| DiffWave | RAWMER | alpha 0.355 | 2000 | 3.5140 | 0.9998 | 0.9904 | 0.9218 | 0.8353 | main result |

Main conclusion:

- HiFi-GAN quality-matched comparison: RAWMER improves `noise20/noise10/noise5` by about 9.1, 22.8, and 22.6 percentage points over local MelShield.
- DiffWave quality-matched comparison: RAWMER improves `noise20/noise10/noise5` by about 2.2, 9.9, and 11.3 percentage points over local MelShield.
- Non-noise attacks are near saturated for both methods, so the strongest differentiator is additive noise.

## 9. RAWMER Full Attack Results

### HiFi-GAN RAWMER random2000

Configuration:

```text
alpha=0.435, band=20:60, mask_floor=0.20, boundary_margin=0.01,
block_frames=8, block_stride=8, bits_per_block=6, pair_bins=6,
pair_candidates=16, detector_mode=plain
```

| attack | ACC | VR | PESQ | STOI |
|---|---:|---:|---:|---:|
| none | 0.9999 | 1.000 | 3.5141 | 0.9685 |
| mp3 | 0.9999 | 1.000 | 3.5133 | 0.9685 |
| aac | 0.9999 | 1.000 | 3.4954 | 0.9683 |
| scale | 0.9999 | 1.000 | 3.5141 | 0.9685 |
| rs16 | 0.9999 | 1.000 | 3.5141 | 0.9685 |
| bandpass | 0.9999 | 1.000 | 3.5337 | 0.9632 |
| lowpass | 0.9999 | 1.000 | 3.5222 | 0.9678 |
| noise20 | 0.9942 | 1.000 | 1.5561 | 0.9525 |
| noise10 | 0.9408 | 0.991 | 1.0994 | 0.8988 |
| noise5 | 0.8587 | 0.918 | 1.0415 | 0.8428 |
| echo | 0.9986 | 1.000 | 1.7707 | 0.9178 |

### DiffWave RAWMER random2000

Configuration:

```text
alpha=0.355, band=20:60, mask_floor=0.20, boundary_margin=0.01,
block_frames=8, block_stride=8, bits_per_block=6, pair_bins=6,
pair_candidates=16, detector_mode=plain, DiffWave seed=0
```

| attack | ACC | VR | PESQ | STOI |
|---|---:|---:|---:|---:|
| none | 0.9998 | 1.000 | 3.5140 | 0.9604 |
| mp3 | 0.9998 | 1.000 | 3.5124 | 0.9603 |
| aac | 0.9998 | 1.000 | 3.4971 | 0.9602 |
| scale | 0.9998 | 1.000 | 3.5140 | 0.9604 |
| rs16 | 0.9998 | 1.000 | 3.5140 | 0.9604 |
| bandpass | 0.9998 | 1.000 | 3.4927 | 0.9551 |
| lowpass | 0.9988 | 1.000 | 3.5801 | 0.9603 |
| noise20 | 0.9904 | 0.9995 | 1.4905 | 0.9428 |
| noise10 | 0.9218 | 0.9825 | 1.0782 | 0.8920 |
| noise5 | 0.8353 | 0.871 | 1.0349 | 0.8425 |
| echo | 0.9967 | 1.000 | 1.6058 | 0.9120 |

## 10. MelShield Full Attack Baselines

### HiFi-GAN MelShield Quality-Matched random2000

| attack | ACC | VR | PESQ | STOI |
|---|---:|---:|---:|---:|
| none | 0.9993 | 1.000 | 3.5100 | 0.9687 |
| mp3 | 0.9993 | 1.000 | 3.5093 | 0.9687 |
| aac | 0.9992 | 1.000 | 3.4940 | 0.9685 |
| scale | 0.9993 | 1.000 | 3.5100 | 0.9687 |
| rs16 | 0.9993 | 1.000 | 3.5100 | 0.9687 |
| bandpass | 0.9992 | 1.000 | 3.5612 | 0.9639 |
| lowpass | 0.9980 | 1.000 | 3.5087 | 0.9681 |
| noise20 | 0.9028 | 0.999 | 1.5483 | 0.9523 |
| noise10 | 0.7129 | 0.874 | 1.0972 | 0.8974 |
| noise5 | 0.6328 | 0.612 | 1.0410 | 0.8400 |
| echo | 0.9829 | 1.000 | 1.7740 | 0.9181 |

### DiffWave MelShield Quality-Matched random2000

| attack | ACC | VR | PESQ | STOI |
|---|---:|---:|---:|---:|
| none | 0.9998 | 1.000 | 3.5149 | 0.9543 |
| mp3 | 0.9998 | 1.000 | 3.5136 | 0.9543 |
| aac | 0.9998 | 1.000 | 3.4994 | 0.9541 |
| scale | 0.9998 | 1.000 | 3.5149 | 0.9543 |
| rs16 | 0.9995 | 1.000 | 3.5149 | 0.9543 |
| bandpass | 0.9992 | 1.000 | 3.5121 | 0.9492 |
| lowpass | 0.9958 | 1.000 | 3.5779 | 0.9541 |
| noise20 | 0.9685 | 0.9995 | 1.4967 | 0.9378 |
| noise10 | 0.8225 | 0.9725 | 1.0784 | 0.8907 |
| noise5 | 0.7219 | 0.872 | 1.0348 | 0.8426 |
| echo | 0.9972 | 1.000 | 1.5998 | 0.9057 |

## 11. Waveform Baselines: AudioSeal and WavMark

Waveform baselines are evaluated in `scripts/grid_waveform_baselines_ljspeech.py`.

Protocol:

1. Generate benchmark audio from clean Mel using the selected vocoder.
2. Embed a waveform-domain watermark into benchmark audio.
3. Apply the same attack suite.
4. Decode/detect using AudioSeal or WavMark.
5. Report payload bit accuracy, verification rate, decode/detection rate, PESQ/STOI/SNR relative to benchmark audio.

Important differences from RAWMER/MelShield:

- AudioSeal/WavMark are waveform-domain methods.
- Main waveform baseline payload length is 16 bits.
- RAWMER/MelShield main experiments use 32-bit payloads.
- Do not present AudioSeal/WavMark as perfectly fair same-setting baselines; present them as strong cross-paradigm waveform baselines.

Waveform baseline summary:

| Vocoder | Method | payload bits | Clean PESQ | none ACC/VR | noise20 ACC/VR | noise10 ACC/VR | noise5 ACC/VR | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HiFi-GAN | AudioSeal | 16 | 4.4574 | 1.0000 / 1.000 | 0.9736 / 0.304 | 0.7900 / 0.000 | 0.6465 / 0.000 | detector rejects under noise |
| HiFi-GAN | WavMark forced | 16 | 4.2975 | 0.9998 / 1.000 | 0.7523 / 0.630 | 0.5114 / 0.046 | 0.4926 / 0.034 | forced ACC; low decode rate |
| DiffWave | AudioSeal | 16 | 4.4225 | 1.0000 / 1.000 | 0.9659 / 0.138 | 0.7544 / 0.000 | 0.6175 / 0.000 | detector rejects under noise |
| DiffWave | WavMark forced | 16 | 4.0211 | 1.0000 / 1.000 | 0.6700 / 0.376 | 0.5120 / 0.034 | 0.4978 / 0.046 | forced ACC; low decode rate |

Writing conclusion:

- AudioSeal and WavMark are strong under many non-noise attacks and preserve high clean audio quality.
- Under strong additive noise, their verification rates or forced bit accuracies degrade sharply.
- RAWMER should be described as stronger in this paper's reference-assisted TTS verification setting under additive noise, not as universally superior.

## 12. Reference-Assisted Verification Controls

Script: `scripts/eval_relmel_reference_controls_ljspeech.py`.

Purpose: show that verification does not pass with arbitrary reference/key/payload.

Controls:

| condition | meaning |
|---|---|
| `correct` | correct watermarked audio, correct reference, correct key/payload |
| `clean_unmarked` | unwatermarked benchmark audio against claimed reference/payload |
| `wrong_key` | watermarked audio, correct reference/payload, wrong key |
| `wrong_payload` | watermarked audio/reference, but wrong claimed payload |
| `wrong_reference` | watermarked audio checked against another utterance's full reference |
| `wrong_reference_mel` | only clean mel replaced by another utterance; semi-wrong control |

Random500, HiFi-GAN, attacks `none noise20`:

| condition | none ACC | none VR | noise20 ACC | noise20 VR | note |
|---|---:|---:|---:|---:|---|
| correct | 0.9999 | 1.000 | 0.9948 | 1.000 | positive |
| clean_unmarked | 0.4964 | 0.002 | 0.4966 | 0.004 | near random |
| wrong_key | 0.5046 | 0.000 | 0.5011 | 0.002 | near random |
| wrong_payload | 0.5017 | 0.000 | 0.5020 | 0.000 | payload mismatch |
| wrong_reference | 0.4976 | 0.002 | 0.5026 | 0.004 | wrong utterance |
| wrong_reference_mel | 0.5150 | 0.020 | 0.5041 | 0.004 | semi-wrong |

Random2000, HiFi-GAN, no attack:

| condition | ACC | VR | confidence | PESQ | STOI |
|---|---:|---:|---:|---:|---:|
| correct | 0.9999 | 1.0000 | 0.001373 | 3.5141 | 0.9685 |
| clean_unmarked | 0.5008 | 0.0040 | 0.000106 | 4.6439 | 1.0000 |
| wrong_key | 0.5030 | 0.0015 | 0.000210 | 3.5141 | 0.9685 |
| wrong_payload | 0.5000 | 0.0040 | 0.001373 | 3.5141 | 0.9685 |
| wrong_reference | 0.4991 | 0.0035 | 0.000460 | 3.5141 | 0.9685 |

Writing conclusion:

- Correct reference/key/payload passes reliably.
- Wrong key, wrong payload, wrong reference return to random-level ACC and very low VR.
- Confidence alone is not sufficient for provenance; final decision is based on payload match/bit accuracy.

## 13. Reference Compression

Script: `scripts/eval_relmel_reference_compression_ljspeech.py`.

Purpose: evaluate storage cost and robustness under compressed clean Mel references.

Reference variants:

| variant | meaning |
|---|---|
| `float32` | full 80-bin clean Mel, float32 |
| `float16` | full clean Mel quantized through float16 |
| `uint8/uint6/uint4/uint2` | full 80-bin clean Mel quantized to N bits |
| `band_uint8/band_uint4/band_uint2` | only watermark band quantized to N bits |

Important implementation detail:

- `band_uint*` stores only bins in `band=20:60`; outside-band clean Mel is set to zeros in the proxy reference.
- This works for RAWMER because verification only needs the watermark band.

Random2000 HiFi-GAN reference compression:

| reference | avg KB | none ACC/VR | noise20 ACC/VR | noise10 ACC/VR | noise5 ACC/VR |
|---|---:|---:|---:|---:|---:|
| float32 | 174.96 | 0.9999 / 1.000 | 0.9943 / 1.000 | 0.9407 / 0.994 | 0.8595 / 0.931 |
| band_uint8 | 21.87 | 0.9999 / 1.000 | 0.9939 / 1.000 | 0.9394 / 0.993 | 0.8584 / 0.925 |
| uint4 | 21.87 | 0.9986 / 1.000 | 0.9820 / 1.000 | 0.9048 / 0.975 | 0.8183 / 0.842 |

Conclusion:

- `band_uint8` reduces average storage to about 21.87 KB/reference, around 1/8 of full float32.
- Robustness is almost unchanged relative to float32.
- At equal size, preserving precision in the watermark band is better than coarse full-Mel quantization.

### Extreme Compression and MelShield Comparison

Random500 HiFi-GAN uint sweep:

RAWMER:

| reference | avg KB | none ACC/VR | noise20 ACC/VR | noise10 ACC/VR | noise5 ACC/VR |
|---|---:|---:|---:|---:|---:|
| float32 | 176.72 | 0.9999 / 1.000 | 0.9948 / 1.000 | 0.9451 / 0.994 | 0.8619 / 0.924 |
| band_uint8 | 22.09 | 1.0000 / 1.000 | 0.9941 / 1.000 | 0.9424 / 0.994 | 0.8598 / 0.924 |
| uint4 | 22.09 | 0.9986 / 1.000 | 0.9828 / 1.000 | 0.9134 / 0.984 | 0.8236 / 0.854 |
| band_uint4 | 11.05 | 0.9984 / 1.000 | 0.9826 / 1.000 | 0.9128 / 0.984 | 0.8245 / 0.856 |
| uint2 | 11.05 | 0.8558 / 0.904 | 0.7554 / 0.618 | 0.6466 / 0.168 | 0.5926 / 0.064 |
| band_uint2 | 5.52 | 0.8538 / 0.896 | 0.7559 / 0.616 | 0.6479 / 0.172 | 0.5923 / 0.068 |

MelShield:

| reference | avg KB | none ACC/VR | noise20 ACC/VR | noise10 ACC/VR | noise5 ACC/VR |
|---|---:|---:|---:|---:|---:|
| float32 | 176.72 | 0.9993 / 1.000 | 0.9015 / 0.998 | 0.7129 / 0.876 | 0.6313 / 0.622 |
| band_uint8 | 22.09 | 0.7589 / 0.944 | 0.6390 / 0.652 | 0.5767 / 0.368 | 0.5498 / 0.236 |
| uint4 | 22.09 | 0.9983 / 1.000 | 0.8968 / 1.000 | 0.7103 / 0.862 | 0.6326 / 0.616 |
| band_uint4 | 11.05 | 0.7583 / 0.936 | 0.6378 / 0.634 | 0.5748 / 0.360 | 0.5504 / 0.254 |
| uint2 | 11.05 | 0.9418 / 1.000 | 0.7859 / 0.982 | 0.6554 / 0.710 | 0.6007 / 0.442 |
| band_uint2 | 5.52 | 0.7323 / 0.904 | 0.6289 / 0.604 | 0.5714 / 0.348 | 0.5481 / 0.250 |

Writing conclusion:

- RAWMER naturally supports band-only reference storage because its verification signal is localized to selected relative relations in the watermark band.
- MelShield is much less stable under band-only reference in the current implementation.
- 2-bit quantization is too aggressive for RAWMER.

## 14. Partial Clips and Blind Fragment Verification

Known-position fragment script: `scripts/eval_relmel_fragments_ljspeech.py`.

Blind-position fragment script: `scripts/eval_relmel_blind_fragments_ljspeech.py`.

### Known-Position Fragments

Protocol:

- Crop watermarked and benchmark audio based on Mel frame boundaries.
- Verifier knows the fragment's start/end frame in the reference.
- Evaluate full/start25/middle25/end25/start50/middle50/end50/middle75.

Random500 HiFi-GAN:

| fragment | mean ratio | mean votes | ACC | VR | PESQ | STOI |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 | 13.18 | 0.9999 | 1.000 | 3.5101 | 0.9685 |
| start25 | 0.250 | 3.31 | 0.9911 | 1.000 | 3.5095 | 0.9546 |
| middle25 | 0.250 | 3.30 | 0.9908 | 1.000 | 3.5632 | 0.9541 |
| end25 | 0.250 | 3.24 | 0.9890 | 1.000 | 3.4943 | 0.9547 |
| start50 | 0.500 | 6.63 | 0.9984 | 1.000 | 3.5127 | 0.9679 |
| middle50 | 0.500 | 6.62 | 0.9986 | 1.000 | 3.5481 | 0.9698 |
| end50 | 0.500 | 6.54 | 0.9987 | 1.000 | 3.5161 | 0.9692 |
| middle75 | 0.750 | 9.93 | 0.9998 | 1.000 | 3.5256 | 0.9694 |

### Blind Fragment Verification

Protocol:

- The verifier does not know the crop start.
- It slides the fragment over the clean reference using `search_step_frames=4`.
- It selects the window with the highest signed payload score.

Random500 HiFi-GAN:

| fragment | mean ratio | searched windows | start error mean/median frame | mean votes | ACC | VR | PESQ | STOI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start25 | 0.250 | 107.2 | 0.00 / 0.0 | 3.33 | 0.9908 | 1.000 | 3.5095 | 0.9546 |
| middle25 | 0.250 | 107.2 | 1.15 / 1.0 | 3.35 | 0.9854 | 0.998 | 3.5632 | 0.9541 |
| end25 | 0.250 | 107.2 | 1.14 / 1.0 | 3.26 | 0.9884 | 0.998 | 3.4943 | 0.9547 |
| start50 | 0.500 | 71.7 | 0.00 / 0.0 | 6.64 | 0.9984 | 1.000 | 3.5127 | 0.9679 |
| middle50 | 0.500 | 71.7 | 1.04 / 1.0 | 6.66 | 0.9973 | 1.000 | 3.5481 | 0.9698 |
| end50 | 0.500 | 71.7 | 1.00 / 1.0 | 6.57 | 0.9989 | 1.000 | 3.5161 | 0.9692 |

Writing conclusion:

- Even 25% clips remain verifiable.
- Blind search is more compelling than known-position verification for the main paper.
- The small median start error suggests the reference search can localize the fragment accurately.

## 15. Modern Attacks and Robustness Boundary

Random500, band 20:60, main RAWMER configuration.

| attack | HiFi-GAN ACC/VR | HiFi-GAN PESQ | DiffWave ACC/VR | DiffWave PESQ | note |
|---|---:|---:|---:|---:|---|
| none | 0.9999 / 1.000 | 3.5101 | 0.9998 / 1.000 | 3.5459 | clean |
| mp3_64 | 0.9999 / 1.000 | 3.4705 | 0.9998 / 1.000 | 3.4883 | low-bitrate MP3 |
| aac_48 | 0.9998 / 1.000 | 2.9301 | 0.9997 / 1.000 | 2.9286 | low-bitrate AAC |
| rs8 | 0.9999 / 1.000 | 2.3599 | 0.9483 / 0.994 | 2.5121 | 8 kHz resampling |
| rs24 | 0.9999 / 1.000 | 3.5101 | 0.9998 / 1.000 | 3.5460 | 24 kHz resampling |
| lowpass2k | 0.9991 / 1.000 | 3.4663 | 0.9930 / 1.000 | 3.5735 | 2 kHz lowpass |
| bandpass_wide | 0.9999 / 1.000 | 3.5045 | 0.9998 / 1.000 | 3.5454 | 100 Hz-7 kHz |
| clip05 | 0.9999 / 1.000 | 3.4840 | 0.9998 / 1.000 | 3.5289 | clipping |
| quant8 | 0.9993 / 1.000 | 2.4019 | 0.9991 / 1.000 | 2.3432 | 8-bit quantization |
| reverb | 0.9885 / 1.000 | 1.3004 | 0.9872 / 1.000 | 1.2564 | stronger echo |
| noise0 | 0.7516 / 0.598 | 1.0262 | 0.7294 / 0.486 | 1.0237 | extreme 0 dB noise |
| pitch_down | 0.7854 / 0.732 | 1.2358 | 0.6475 / 0.162 | 1.2216 | -100 cents |
| pitch_up | 0.7824 / 0.724 | 1.2266 | 0.6481 / 0.160 | 1.2140 | +100 cents |
| speed090 | 0.6632 / 0.260 | 1.2531 | 0.6510 / 0.186 | 1.2529 | 0.9x speed |
| speed110 | 0.6867 / 0.320 | 1.8922 | 0.6678 / 0.250 | 1.8895 | 1.1x speed |

Writing conclusion:

- RAWMER is robust to compression, filtering, clipping, quantization, reverb, and moderate noise.
- 0 dB noise is an additive-noise boundary.
- Strong speed and pitch transformations are the main failure modes because they break frame-level time/frequency alignment with the reference.

## 16. Speed and Pitch Strength Curve

HiFi-GAN RAWMER random500:

| perturbation | ACC | VR | PESQ | STOI | note |
|---|---:|---:|---:|---:|---|
| none | 0.9999 | 1.000 | 3.5101 | 0.9685 | clean |
| pitch_down25 | 0.9997 | 1.000 | 2.7081 | 0.9380 | -25 cents |
| pitch_up25 | 0.9997 | 1.000 | 2.8172 | 0.9091 | +25 cents |
| pitch_down50 | 0.9961 | 1.000 | 1.8741 | 0.9126 | -50 cents |
| pitch_up50 | 0.9953 | 1.000 | 1.9169 | 0.8881 | +50 cents |
| pitch_down | 0.7854 | 0.732 | 1.2358 | 0.8620 | -100 cents |
| pitch_up | 0.7824 | 0.724 | 1.2266 | 0.8429 | +100 cents |
| speed097 | 0.9566 | 0.996 | 2.1991 | 0.2472 | 0.97x |
| speed103 | 0.9612 | 0.998 | 2.5370 | 0.3159 | 1.03x |
| speed095 | 0.8332 | 0.850 | 1.6658 | 0.1730 | 0.95x |
| speed105 | 0.8524 | 0.906 | 2.2263 | 0.2283 | 1.05x |
| speed090 | 0.6632 | 0.260 | 1.2531 | 0.1391 | 0.90x |
| speed110 | 0.6867 | 0.320 | 1.8922 | 0.1643 | 1.10x |

Writing conclusion:

- Pitch shift within ±50 cents remains robust.
- Speed changes of ±3% remain mostly usable.
- ±5% speed begins to degrade.
- ±10% speed and ±100 cents pitch are current boundaries.

## 17. Key Ablations

### Pair Candidates Ablation

HiFi-GAN, random500, `alpha=0.435`, `band=20:60`, `mask_floor=0.20`, `boundary_margin=0.01`, attacks `none noise20 noise10 noise5`.

| pair_candidates | Clean PESQ | none | noise20 | noise10 | noise5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.5485 | 0.9999 | 0.9459 | 0.7674 | 0.6684 |
| 4 | 3.5299 | 0.9999 | 0.9873 | 0.8894 | 0.7845 |
| 8 | 3.5184 | 0.9999 | 0.9918 | 0.9232 | 0.8315 |
| 16 | 3.5102 | 0.9999 | 0.9952 | 0.9439 | 0.8607 |

Conclusion:

- Increasing candidate count from 1 to 16 substantially improves additive-noise robustness.
- PESQ decreases only modestly.
- This supports reliability-aware pair selection as a key design.

### Multi-Seed Stability

HiFi-GAN, random500:

| seed | Clean PESQ | none | noise20 | noise10 | noise5 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 3.5102 | 0.9999 | 0.9952 | 0.9439 | 0.8607 |
| 2027 | 3.5108 | 0.9999 | 0.9950 | 0.9411 | 0.8588 |
| 2028 | 3.5164 | 0.9998 | 0.9938 | 0.9391 | 0.8574 |
| average | 3.5125 | 0.9999 | 0.9946 | 0.9413 | 0.8590 |

Conclusion:

- Main trend is stable across random sampling seeds.

### DiffWave Tuning Summary

Useful RAWMER + DiffWave alpha range is approximately 0.30 to 0.358.

| alpha | mask_floor | boundary_margin | Clean PESQ | none | noise20 | noise10 | noise5 | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.30 | 0.25 | 0.005 | 3.7717 | 1.0000 | 0.9820 | 0.8922 | 0.8000 | high quality |
| 0.34 | 0.25 | 0.005 | 3.5557 | 1.0000 | 0.9867 | 0.9109 | 0.8125 | near quality edge |
| 0.35 | 0.20 | 0.010 | 3.5363 | 1.0000 | 0.9883 | 0.9094 | 0.8289 | pilot recommendation |
| 0.355 | 0.20 | 0.010 | 3.5140 | 0.9998 | 0.9904 | 0.9218 | 0.8353 | random2000 main result |
| 0.358 | 0.20 | 0.010 | 3.4971 | 0.9998 | 0.9903 | 0.9231 | 0.8362 | slightly below 3.5 PESQ |
| 0.38 | 0.25 | 0.005 | 3.3374 | 1.0000 | 0.9938 | 0.9297 | 0.8281 | below 3.5 PESQ |

Final DiffWave main result uses `alpha=0.355`, `mask_floor=0.20`, `boundary_margin=0.01`. The `alpha=0.358` setting is slightly more robust but falls below the 3.5 clean-PESQ tuning target, so it should be treated as a quality-boundary reference rather than the main paper configuration.

## 18. Grid Search and Objective

RAWMER grid search: `scripts/grid_relmel_ljspeech.py`.

MelShield grid search: `scripts/grid_melshield_ljspeech.py`.

RAWMER objective roughly combines clean/noise robustness with PESQ penalty:

```text
objective =
  none_acc
  + noise20_weight * noise20_acc
  + noise10_weight * noise10_acc
  + noise5_weight * noise5_acc
  + echo_weight * echo_acc
  - quality_weight * max(0, quality_floor - none_pesq)
```

Default grid script quality floor is 3.5, but some pilot scripts use different quality weights/floors to explore candidates. Do not overstate PESQ 3.5 as a universal standard; it is a practical tuning target.

Grid outputs:

- `grid_results.csv`
- `grid_results.json`
- `best_candidate.json`

Note: `scripts/grid_relmel_ljspeech.py` uses `empty_metrics()` in the `vocoder=mel` branch but does not import it. This does not affect paper waveform-vocoder experiments, but the `mel` grid branch may need a small fix if used.

## 19. Reproducibility Commands

Core random2000 experiments are represented by `scripts/run_relmel_core_random2000_experiments.sh`.

Main common attack list:

```text
none mp3 aac scale rs16 bandpass lowpass noise20 noise10 noise5 echo
```

Reference experiments are represented by `scripts/run_relmel_reference_random2000_experiments.sh`:

- reference compression: `none noise20 noise10 noise5`
- blind fragments: start25/middle25/end25/start50/middle50/end50, `search_step_frames=4`
- controls: correct/clean_unmarked/wrong_key/wrong_payload/wrong_reference

Reference compression uint sweep is represented by `scripts/run_reference_compression_uint_sweep.sh`:

- variants: `float32 band_uint8 uint4 uint2 band_uint4 band_uint2`
- attacks: `none noise20 noise10 noise5`
- default limit: 500

DiffWave paper suite is represented by `scripts/run_diffwave_paper_suite.sh`:

- `smoke`
- `relmel-stage1`
- `relmel-full`
- `relmel-pair-ablation`
- `melshield-stage1`
- `melshield-full`

## 20. Writing Do's and Don'ts

Do write:

- RAWMER is reference-assisted, not blind.
- The primary setting is platform-side provenance verification for neural speech generation.
- Main advantage is robustness to additive noise at comparable clean PESQ.
- MelShield reported results and local reproductions should be separated.
- AudioSeal/WavMark are cross-paradigm waveform baselines with 16-bit payloads.
- Verification is based on payload match/bit accuracy, not confidence alone.
- Reference compression and blind fragment verification support practical deployment.
- Strong speed/pitch changes are current limitations due to synchronization disruption.

Do not write:

- RAWMER is a blind watermarking method.
- RAWMER does not need a reference.
- RAWMER universally outperforms AudioSeal/WavMark under all conditions.
- Confidence alone proves attribution.
- PESQ 3.5 is a strict universal standard.
- Reported MelShield paper numbers and local quality-matched numbers are one fair protocol.

## 21. Concise Experiments Narrative

Suggested paragraph skeleton:

RAWMER is evaluated on LJSpeech using HiFi-GAN and DiffWave vocoders. We compare against a local reproduction of MelShield under quality-matched settings and also report MelShield paper numbers as external reference. All main mel-domain experiments use 32-bit payloads and reference-assisted verification; RAWMER uses a threshold of 0.75. We evaluate common signal-processing attacks including compression, scaling, resampling, filtering, additive noise, and echo. We report bit accuracy, verification rate, PESQ, and STOI, with PESQ/STOI computed against the unwatermarked vocoder benchmark audio.

At comparable clean PESQ, RAWMER substantially improves additive-noise robustness over MelShield on both vocoders. Under HiFi-GAN, RAWMER reaches 0.9942/0.9408/0.8587 ACC on noise20/noise10/noise5, compared with 0.9028/0.7129/0.6328 for MelShield. Under DiffWave, RAWMER reaches 0.9904/0.9218/0.8353, compared with 0.9685/0.8225/0.7219. Both methods remain near saturated under non-noise attacks.

Additional experiments validate the reference-assisted setting. Wrong-key, wrong-payload, and wrong-reference controls return to random-level bit accuracy with low verification rates. Band-limited 8-bit clean Mel references reduce storage to about 21.87 KB per utterance while preserving robustness close to full float32 references. Blind sliding-window fragment verification shows that 25% clips remain verifiable with about 0.985-0.991 bit accuracy. Extended attacks show robustness to compression, filtering, clipping, quantization, and moderate noise, while strong speed changes and pitch shifts remain the primary boundary.
