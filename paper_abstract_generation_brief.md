# Abstract Generation Brief for RelMel

本文件用于指导 AI 生成论文 abstract 初稿。目标是让生成结果准确概括本文贡献，避免把本文误写成通用 blind watermarking（盲水印检测）方法，也避免夸大实验结论。

## 1. 论文暂定主题

本文提出 RelMel，一种用于神经语音生成场景的 reference-assisted mel-domain speech watermarking（参考辅助的梅尔域语音水印）方法。

核心思想：在 mel spectrogram（梅尔频谱）中，不直接依赖绝对扰动模式，而是将 payload（载荷信息）编码为 block-wise relative energy relations（块级相对能量关系）。验证时，平台方使用保存的 clean mel reference（干净梅尔参考）或其压缩版本，与疑似音频重建出的 mel 进行对比，从而恢复 payload 并判断归属。

## 2. 必须体现的定位

Abstract 中必须明确：

- 本文关注的是 reference-assisted verification（参考辅助验证），适用于平台侧 provenance verification（来源验证）、forensic attribution（取证归属）、TTS 服务审计等场景。
- 本文不是要替代所有 blind detector（盲检测器），也不是声称不需要任何 reference。
- Reference 的需求是本文的设计选择和应用设定，而不是缺陷掩盖；实验中已经证明 reference 可以压缩到约 22KB/utterance。
- 方法不需要训练新的神经检测器，主要依赖可解释的 mel-domain 相对关系。

## 3. 推荐 Abstract 结构

建议 abstract 分成 5 个逻辑句群：

1. 背景和问题：神经语音生成需要可靠水印；现有波形域或绝对扰动式 mel 水印在强噪声和跨声码器条件下容易退化。
2. 关键观察：TTS 平台通常可以保存生成时的 mel reference，因此 reference-assisted verification 是合理设定；在该设定下，相对能量关系比绝对扰动更稳。
3. 方法：提出 RelMel，将 payload 编码为多个时间块内的正负 mel bin groups（梅尔频带组）之间的相对能量偏移，并通过 reliable pair selection（可靠候选对选择）选择稳定频带对。
4. 实验结果：在 LJSpeech 上，使用 HiFi-GAN 和 DiffWave 两种 vocoder（声码器）评测；RelMel 在相近 PESQ 下显著优于 MelShield，尤其是 noise20/noise10/noise5；同时与 AudioSeal 和 WavMark 进行 waveform baseline（波形基线）对比。
5. 附加验证：负控实验显示低假阳性；reference 压缩和片段验证说明实际可用；最后点出边界，如 speed/pitch 大幅同步破坏仍是未来方向。

## 4. 可以使用的关键数据

Abstract 里只放 2 到 4 个最强数字，不要堆表格。

推荐数字：

- HiFi-GAN，random500，RelMel 在干净 PESQ 约 3.51 下达到：
  - noise20 bit accuracy 0.9946
  - noise10 bit accuracy 0.9455
  - noise5 bit accuracy 0.8648
- DiffWave，random500，RelMel 在干净 PESQ 约 3.54 下达到：
  - noise20 bit accuracy 0.9914
  - noise10 bit accuracy 0.9225
  - noise5 bit accuracy 0.8308
- DiffWave 质量匹配对比中，MelShield 在相近 PESQ 约 3.55 下为：
  - noise20 0.9658
  - noise10 0.8212
  - noise5 0.7210
- Reference 压缩：只保存水印频带的 8-bit clean mel，约 22KB/utterance，仍保持接近 float32 reference 的鲁棒性。
- 负控：wrong reference / wrong key / wrong payload 的 bit accuracy 基本回到 0.5 附近，完整错误 reference 验证率约 0.2% 到 0.4%。
- 片段验证：已知位置的 25% 音频片段仍达到约 0.989 到 0.991 bit accuracy，验证率为 1.000。

## 5. 建议强调的贡献

Abstract 中可以概括为三点，但不要写得像 bullet list，除非目标会议允许结构化 abstract。

- A relative mel-energy watermarking formulation：将水印编码为正负 mel 频带组之间的相对能量关系。
- A reliability-aware pair selection strategy：利用 clean mel reference 选择更稳定、更平衡的候选频带对。
- A practical reference-assisted evaluation：包括跨 vocoder、强噪声、负控假阳性、reference 压缩、局部片段、AudioSeal/WavMark 对比等实验。

## 6. 不要误写或夸大的内容

生成 abstract 时避免以下说法：

- 不要说 RelMel 是 blind watermarking。正确说法是 reference-assisted verification。
- 不要说 RelMel 在所有攻击下全面鲁棒。speed change（变速）和 pitch shift（变调）大幅攻击仍是弱点。
- 不要说完全碾压 AudioSeal/WavMark。更准确的说法是：在本文的 reference-assisted TTS verification setting 下，RelMel 在 additive noise（加性噪声）下比这些 waveform baselines 更稳定；但 AudioSeal/WavMark 在非噪声攻击和干净音质上很强。
- 不要混淆 bit accuracy（比特准确率）和 verification rate（验证率）。bit accuracy 是恢复 payload 的平均比特正确率；verification rate 是超过阈值后判定通过的比例。
- 不要声称 reference 存储为零成本。正确说法是通过 band-limited 8-bit compression 降到约 22KB/utterance。
- 不要把 MelShield 原文 reported 数据当作完全公平主对照。公平主对照应优先使用本地复现、同声码器、同攻击、同样本数、相近 PESQ 的结果。

## 7. 推荐英文关键词

可在 abstract 中使用这些表达：

- neural speech watermarking
- mel-domain watermarking
- reference-assisted verification
- relative energy relations
- robust provenance verification
- text-to-speech watermarking
- reliable pair selection
- additive noise robustness
- vocoder transferability
- compressed reference

## 8. 一句话中心论断

可以围绕下面这句话展开 abstract：

RelMel shows that, when a clean mel reference is available to the service provider, encoding payload bits as relative energy relations in the mel domain yields a practical and noise-robust speech watermarking scheme for neural TTS provenance verification.

## 9. 期望生成风格

- 英文学术风格，面向 speech / audio / machine learning 会议。
- 长度建议 150 到 220 words。
- 语气克制，不要宣传腔。
- 不要在 abstract 中解释过多实现细节。
- 最后一句应落在 contribution 或 result 上，而不是泛泛的 future work。
