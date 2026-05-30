# Introduction Generation Brief for RelMel

本文件用于指导 AI 生成论文 introduction 初稿。目标是提供完整故事线、必要背景、段落安排、关键实验事实和写作边界，使生成的 introduction 能自然引出 RelMel，而不是简单堆砌实验结果。

## 1. 论文核心命题

本文的核心命题是：

在 TTS（text-to-speech，文本转语音）和神经语音生成平台中，服务方通常可以保存生成过程中的 clean mel spectrogram（干净梅尔频谱）或其压缩版本。因此，除了完全 blind watermark detection（盲水印检测）之外，reference-assisted verification（参考辅助验证）也是一个现实且重要的语音水印设定。在这个设定下，将 payload（载荷信息）编码为 mel 域的 relative energy relations（相对能量关系），可以显著提升对声码器重建和加性噪声的鲁棒性。

## 2. 建议 Introduction 总体结构

建议写成 7 到 9 个自然段。

### 第 1 段：背景和动机

讲 neural speech generation（神经语音生成）、TTS、voice cloning（语音克隆）带来的 provenance（来源追踪）和 misuse detection（滥用检测）需求。

可表达为：

- 高质量 TTS 和 voice conversion 系统降低了合成语音门槛。
- 平台和内容分发方需要在音频被传播、压缩、加噪或二次处理后仍能确认来源。
- Audio watermarking（音频水印）是比元数据更稳的选择，因为元数据容易被移除。

### 第 2 段：现有音频水印范式

介绍两类主流路线：

- waveform-domain watermarking（波形域水印），如 AudioSeal、WavMark，直接在音频波形中嵌入和检测。
- generation-aware 或 mel-domain watermarking（生成链路感知或梅尔域水印），在声码器前的中间表示中嵌入水印，如 MelShield。

强调波形域方法通常更通用，干净音质和非噪声攻击下表现强；但在强加性噪声、声码器/中间表示相关场景下，mel-domain 方法可能有优势。

### 第 3 段：Reference-assisted verification 的合理性

这是本文需要讲清楚的关键设定。

要点：

- 许多 TTS 服务方本来就保存请求日志、utterance id（语音编号）、speaker id（说话人编号）、生成配置或中间特征。
- 对平台侧 forensic verification（取证验证）而言，目标不是任意第三方在没有上下文时盲检，而是服务方确认“这段疑似音频是否来自我生成的某条记录”。
- 因此，保存 clean mel reference 或其压缩版本是合理代价。
- 本文实验显示 reference 可压缩到约 22KB/utterance，仍保持接近 float32 reference 的鲁棒性。

必须避免：

- 不要把 reference 设定写成“不需要解释的默认设定”。
- 不要把它说成比 blind detection 绝对更好。应说是 complementary setting（互补设定）。

### 第 4 段：MelShield 的启发与局限

介绍 baseline MelShield：

- MelShield 在 mel 域加入 key-dependent perturbation（密钥相关扰动）。
- 验证时用 suspect mel 与 clean reference mel 的 residual（残差）恢复 payload。
- 这证明了 mel-domain reference-assisted watermarking 的可行性。

然后指出局限：

- 它更依赖 absolute perturbation patterns（绝对扰动模式）。
- 加性噪声会掩盖残差信号，尤其在 noise10/noise5 条件下鲁棒性下降。
- 声码器和后处理会带来幅度、频带和局部能量变化，使绝对模式不够稳定。

### 第 5 段：RelMel 的关键观察

引出本文核心观察：

相比检测某个 mel bin 的绝对残差，检测同一时间块中两组 mel bins 的 relative energy difference（相对能量差）更稳。

解释方式：

- 对每个 payload bit 和 block，RelMel 选择 positive group（正组）和 negative group（负组）。
- bit=1 时增强正组、削弱负组；bit=0 时反向。
- 检测分数可概括为：

```text
score = mean(residual on positive group) - mean(residual on negative group)
```

- 这种差分关系可以抵消部分共同扰动，例如整体响度变化、局部重建误差和部分声码器失真。

### 第 6 段：Reliable pair selection

讲本文的设计不是随机选频带，而是进行 reliable pair selection（可靠候选对选择）。

要点：

- 并非所有 mel bins 都适合嵌入。
- 低能量区域、靠近边界的区域、不平衡的正负组都容易不稳定。
- RelMel 使用 clean mel reference 评估 candidate pairs（候选对）的可靠性，优先选择能量稳定、边界安全、正负组平衡的 pairs。
- 这使 reference 不仅用于验证，也用于选择更稳的嵌入位置。

### 第 7 段：实验概览

概括实验设计，不要变成结果表。

应提到：

- 数据集：LJSpeech，random500 主实验。
- 声码器：HiFi-GAN 和 DiffWave。
- Baselines：MelShield、本地复现 MelShield、AudioSeal、WavMark。
- 攻击：mp3、aac、scale、resampling、bandpass、lowpass、noise20、noise10、noise5、echo，以及现代扩展攻击如 clipping、quantization、reverb、speed、pitch。
- 指标：bit accuracy、verification rate、PESQ、STOI；对 reference 负控报告 false positive behavior。

### 第 8 段：主要结果

可以概括为：

- 在 HiFi-GAN 上，RelMel 在干净 PESQ 约 3.51 下达到 noise20/noise10/noise5 bit accuracy 0.9946/0.9455/0.8648。
- 在 DiffWave 上，RelMel 在干净 PESQ 约 3.54 下达到 0.9914/0.9225/0.8308。
- 相比质量匹配的 MelShield + DiffWave，本地复现 MelShield 在相近 PESQ 约 3.55 下为 0.9658/0.8212/0.7210。
- Reference 负控实验显示 wrong key、wrong payload、wrong reference 下 bit accuracy 接近随机水平。
- 只保存 `20:60` 频带的 8-bit reference，约 22KB/utterance，仍接近 float32 reference。
- 25% 已知位置片段也能稳定验证。

### 第 9 段：贡献列表

Introduction 末尾可列贡献。建议三点：

1. 提出 RelMel，一种 reference-assisted mel-domain speech watermarking 方法，将 payload 编码为 block-wise relative mel energy relations。
2. 提出 reliability-aware pair selection，利用 clean mel reference 选择稳定、平衡、可感知控制的嵌入频带对。
3. 系统评估 RelMel 在跨声码器、强噪声、波形 baseline、reference 负控、reference 压缩、局部片段和现代攻击下的表现，证明其适合平台侧语音来源验证。

## 3. 当前实验事实

生成 introduction 时可以引用这些事实：

### RelMel 主结果

HiFi-GAN 主配置：

- `alpha=0.435`
- `band=20:60`
- `mask_floor=0.20`
- `boundary_margin=0.01`
- `pair_candidates=16`
- random500
- clean PESQ 3.5102
- none bit accuracy 0.9999
- noise20 0.9946
- noise10 0.9455
- noise5 0.8648

DiffWave 主配置：

- `alpha=0.35`
- `band=20:60`
- `mask_floor=0.20`
- `boundary_margin=0.01`
- `pair_candidates=16`
- random500
- clean PESQ 3.5443
- none 0.9998
- noise20 0.9914
- noise10 0.9225
- noise5 0.8308

### MelShield 质量匹配基线

DiffWave MelShield 本地复现，质量匹配配置：

- clean PESQ 3.5492
- none 0.9998
- noise20 0.9658
- noise10 0.8212
- noise5 0.7210

### Waveform baselines

AudioSeal / WavMark 为 16-bit waveform watermark，RelMel/MelShield 为 32-bit mel-domain watermark。对比时必须说明 payload 长度和范式不同。

AudioSeal 在非噪声攻击下很强，但噪声条件下 detector-based verification 下降明显：

- HiFi-GAN AudioSeal noise20 verification 0.304，noise10 0.000，noise5 0.000。
- DiffWave AudioSeal noise20 verification 0.138，noise10 0.000，noise5 0.000。

WavMark forced ACC 显示强噪声下接近随机：

- HiFi-GAN WavMark forced noise20 0.7523，noise10 0.5114，noise5 0.4926。
- DiffWave WavMark forced noise20 0.6700，noise10 0.5120，noise5 0.4978。

### Reference 负控

HiFi-GAN random500，band 20:60：

- correct reference：none/noise20 verification 均为 1.000。
- clean_unmarked、wrong_key、wrong_payload、wrong_reference 的 bit accuracy 基本约 0.5。
- 完整 wrong_reference 的 verification rate 约 0.2% 到 0.4%。
- wrong_reference_mel 的 none verification rate 为 2.0%，但它是半错误 reference 条件，应谨慎解释。

### Reference 压缩

- float32 reference 平均约 176.72KB。
- band_uint8 reference 平均约 22.09KB。
- band_uint8 下 noise20/noise10/noise5 分别为 0.9941/0.9399/0.8574，接近 float32。

### 局部片段

已知位置裁剪：

- 25% 片段 bit accuracy 约 0.989 到 0.991，verification 1.000。
- 50% 片段接近完整音频结果。

### 现代攻击边界

RelMel 对压缩、重采样、滤波、裁剪、量化、混响总体稳健。

同步破坏是主要弱点：

- HiFi-GAN 下 speed090 verification 0.260，speed110 0.320。
- 但轻微 speed 变化较稳：speed097 verification 0.996，speed103 0.998。
- ±50 cents pitch shift verification 1.000；±100 cents pitch shift 约 0.724 到 0.732。

这说明本文应把 speed/pitch 作为 limitation（局限）和 future work（未来工作），不要声称全攻击无弱点。

## 4. 相关工作写作方向

Introduction 不需要展开完整 related work，但可以自然提及这些类别：

- Classical audio watermarking：传统扩频、回声隐藏、变换域水印。
- Neural waveform watermarking：AudioSeal、WavMark、Timbre 等。
- Generation-aware speech watermarking：利用 TTS 生成链路、中间表示或声码器前特征。
- Robustness benchmark：AudioMarkBench、RAW-Bench 等可以在 related work 或 experiment 中再详细讨论。

不要在 introduction 里过多评价某篇 baseline 论文质量。语气应聚焦技术差异。

## 5. 推荐标题方向

如果 introduction 中需要方法名解释，可使用：

- RelMel: Relative Mel-Energy Watermarking
- WaterMelon: Watermarking with Mel-domain Relative Energy Relations
- ROMEO: Relative-Energy Optimization for Mel-Domain Speech Watermarking
- MERIT: Mel Energy Relations for Robust Speech Identity Tracing

当前最直观的名字仍是 RelMel，因为它直接表达 relative mel relation。

## 6. 生成时必须避免的错误

- 不要把 PESQ 3.5 写成原论文公认硬阈值。它只是当前实验中用于质量匹配和筛选的经验基准。
- 不要说 verification rate 等于 bit accuracy。二者不同。
- 不要说 32 bit 全部正确才算验证通过。当前阈值为 0.75，即约 24/32 bit 正确即可通过。
- 不要声称 RelMel 不需要 key。它需要 key、utterance id、payload/reference 配合。
- 不要把 `confidence` 写成最终验证依据。最终验证依据是 payload bit accuracy 是否超过 threshold。
- 不要把 AudioSeal/WavMark 说成弱方法。它们在非噪声攻击和干净音质上很强，本文优势主要在 reference-assisted TTS setting 下的强噪声鲁棒性。
- 不要隐瞒 reference-based 设定，也不要把它包装成 blind。
- 不要把已知位置片段验证写成盲裁剪定位，除非使用 blind fragment 实验结果。

## 7. 推荐 Introduction 语气

- 克制、清楚、面向 reviewer。
- 先承认现有 waveform watermarking 的价值，再说明本文场景不同。
- 把 reference-assisted verification 讲成现实部署设定，而不是权宜之计。
- 强调方法可解释、无需训练新检测器、跨 vocoder 稳定。
- 主张要有边界：对强时间/频率同步破坏仍需后续改进。

## 8. 可直接使用的贡献段模板

The main contributions of this work are:

1. We formulate speech watermarking for platform-side reference-assisted verification and introduce RelMel, a mel-domain watermarking method that encodes payload bits as block-wise relative energy relations.
2. We design a reliability-aware pair selection mechanism that uses the clean mel reference to select stable and balanced mel-bin groups, improving robustness under vocoder reconstruction and additive noise.
3. We provide a systematic evaluation across HiFi-GAN and DiffWave, MelShield, AudioSeal, and WavMark baselines, negative reference controls, compressed references, partial clips, and modern audio perturbations.

## 9. 最终写作目标

Introduction 读完后，审稿人应该理解：

- 为什么语音水印值得做。
- 为什么 mel 域是合理嵌入位置。
- 为什么 reference-assisted verification 是一个合理应用场景。
- 为什么相对能量关系比绝对扰动模式更稳。
- RelMel 解决了什么、没有解决什么。
- 主要实验已经覆盖了公平 baseline、假阳性、存储成本和鲁棒性边界。
