# RelMel 工作核心思路说明

## 1. 为什么选择 Mel 域水印

当前神经语音生成系统中，mel spectrogram（梅尔频谱）通常是声学模型和声码器之间的中间表示。文本到语音模型先生成 mel，再由 HiFi-GAN、DiffWave 等 vocoder（声码器）把 mel 转成 waveform（波形音频）。因此，在 mel 域加入水印有几个天然优势：

- mel 表示与听觉感知更接近，频带能量变化比直接波形扰动更容易控制音质影响。
- mel 是生成链路中的关键中间层，水印可以在声码器前注入，并通过声码器传播到最终音频。
- mel 结构相对稳定，适合设计可解释的频带关系，而不是只依赖黑盒神经检测器。
- 对 TTS 平台或模型服务方来说，保存生成时的 reference mel（参考梅尔谱）是可行的，可以用于后续私有验证和归属判断。

这项工作的定位不是公开盲检测水印，而是 reference-assisted verification（参考辅助验证）：平台方保留生成时的 clean mel（原始干净梅尔谱）或其压缩版本，后续拿疑似音频与 reference 对比，判断是否包含对应 payload（载荷信息）。

## 2. Baseline MelShield 的基本思路

MelShield 同样是在 mel 域中加入水印。它的基本做法是：

1. 根据 secret key（密钥）和 utterance id（语音编号）生成一组伪随机模式。
2. 对每个 payload bit（载荷比特），在 mel 上加入正向或负向扰动。
3. 验证时，把疑似音频重新转成 mel，与保存的 reference mel 相减。
4. 检测 residual（残差）与密钥生成的模式是否相关，从而恢复 payload。

这个思路的优点是简单直接，但它更偏向“绝对扰动模式”。在噪声攻击下，尤其是 additive noise（加性噪声）较强时，绝对扰动模式容易被掩盖，导致 `noise10/noise5` 条件下 bit accuracy（比特准确率）下降明显。

## 3. RelMel 的核心想法

RelMel 的核心改进是：不再只依赖某个 mel bin（梅尔频带格）上的绝对扰动，而是把水印编码为 **relative energy relation（相对能量关系）**。

具体来说，对每个 block（时间块）和 payload bit：

1. 在指定 mel band（梅尔频带区间，例如 20:60）中选择两组 mel bins。
2. 一组作为 positive group（正组），另一组作为 negative group（负组）。
3. 如果 bit 是 1，就提高正组、降低负组。
4. 如果 bit 是 0，就降低正组、提高负组。
5. 检测时不看单个频带的绝对值，而看正负两组之间的相对能量漂移。

也就是说，RelMel 读出的不是“某个位置有没有变大”，而是：

```text
score = mean(residual on positive group) - mean(residual on negative group)
```

如果 score 为正，倾向于 bit=1；如果 score 为负，倾向于 bit=0。

这种相对关系比绝对扰动更稳，因为很多攻击会对整体能量、局部频带或声码器输出产生共同影响，而正负组之间的差分关系可以抵消一部分共同扰动。

## 4. 块级重复嵌入

RelMel 不是只在一个位置写入每个 bit，而是在多个时间 block 中重复写入。每个 block 中只激活一部分 payload bits，多个 block 的检测结果再进行投票或累积。

这种 blockwise repetition（块级重复）带来两个好处：

- 提高鲁棒性：某些 block 被噪声、滤波或声码器失真破坏时，其他 block 仍然提供证据。
- 支持局部片段验证：即使只保留部分音频片段，仍可能包含足够数量的 block 来恢复 payload。

实验中，25% 的已知位置片段仍能保持约 0.989 到 0.991 的 bit accuracy，并且验证率达到 1.000，说明水印证据在时间上分布较均匀。

## 5. 可靠候选对选择

RelMel 的关键机制之一是 reliable pair selection（可靠候选对选择）。

如果只随机选择正负 mel bins，有些候选对可能落在不稳定位置：

- 能量太低，声码器重建后不稳定。
- 接近 mel normalized range（归一化范围）的边界，扰动容易被 clipping（裁剪）。
- 正负组可靠性不平衡，导致检测分数偏置。

因此，RelMel 对每个 bit/block 生成多个候选 pair，并根据 clean mel 的局部结构选择最可靠的一组。可靠性主要由两部分组成：

- boundary reliability（边界可靠性）：距离 0 或 1 越远，说明还有可扰动空间。
- energy reliability（能量可靠性）：语音能量更明显的位置通常更稳定。

候选对的选择目标是让正组和负组都可靠，并且两边尽量平衡。实验显示，`pair_candidates` 从 1 增加到 16 后，`noise10` 和 `noise5` 的鲁棒性大幅提升，而 PESQ（客观音质）只小幅下降。

## 6. Verification 标准

RelMel 的验证对象是 payload，而不是单纯判断“有没有水印”。流程是：

1. 根据 utterance id 和 key 生成 expected payload（期望载荷）。
2. 从疑似音频中恢复 decoded payload（解码载荷）。
3. 计算 bit accuracy。
4. 如果 bit accuracy 超过 threshold（阈值，当前主实验为 0.75），则认为验证通过。

因此，验证通过不是要求 32 bit 全部正确，而是要求至少达到设定比例。例如 32 bit payload、threshold 0.75 意味着至少约 24 bit 正确即可通过。

负控实验显示：

- 正确 reference 下，`none/noise20` 验证率均为 1.000。
- 未加水印音频、错误密钥、错误 payload、错误 reference 的 bit accuracy 基本回到 0.5 附近。
- 完整错误 reference 的验证率只有 0.2% 到 0.4%。

这说明 RelMel 的验证强依赖正确 reference、key 和 payload，不是简单靠阈值宽松产生过度自信。

## 7. Reference 存储是否现实

reference-based verification 的一个自然问题是：是否需要保存完整 float32 mel？

实验表明并不需要。只保存水印频带 `20:60` 的 8-bit clean mel，平均约 22KB/reference，就能保持接近 float32 reference 的鲁棒性：

- `noise20` 约 0.9941
- `noise10` 约 0.9399
- `noise5` 约 0.8574

这说明 reference 可以被压缩保存，实际部署成本并不高。对于平台方归属验证、内部审计、模型服务追踪等场景，这种 reference-assisted 方法是合理的。

## 8. 当前实验结论

在 HiFi-GAN 和 DiffWave 两种声码器下，RelMel 都显示出比 MelShield 更强的噪声鲁棒性，尤其在 `noise10/noise5` 条件下优势明显。

与 AudioSeal、WavMark 等 waveform-domain（波形域）baseline 相比，RelMel 在非噪声攻击下同样稳定，而在加性噪声下验证率明显更高。需要注意的是，AudioSeal/WavMark 是 16-bit waveform watermark，而 RelMel 主实验是 32-bit mel-domain watermark，两者属于不同范式，应在论文中分组对比。

现代攻击实验显示：

- 对低码率压缩、重采样、滤波、幅度裁剪、量化和混响，RelMel 基本保持高鲁棒性。
- 对 0 dB 极强噪声，RelMel 接近当前鲁棒边界。
- 对 speed change（变速）和 pitch shift（变调），RelMel 会明显退化，原因主要是 reference 和 suspect mel 之间的时间/频率同步关系被破坏。

进一步的强度曲线显示：

- ±50 cents 以内的 pitch shift 仍然很稳。
- ±3% speed change 基本可用。
- ±5% speed change 开始明显下降。
- ±10% speed change 是当前方法的明显边界。

## 9. 论文中的故事线

这项工作的论文叙事可以概括为：

1. 语音生成场景中，mel 是自然的水印嵌入位置。
2. 现有 mel-domain watermark 主要依赖绝对扰动模式，在强噪声下不够稳。
3. RelMel 将 payload 编码为 block-wise relative mel energy relations。
4. 可靠候选对选择让嵌入位置避开不稳定 mel bins。
5. reference-assisted verification 适合平台侧私有归属验证。
6. 实验显示 RelMel 在跨声码器、加性噪声、局部片段和 reference 压缩场景下都具有优势。
7. 当前边界主要来自时间/频率同步破坏，这也为后续加入 multi-scale alignment（多尺度对齐）或 DTW（动态时间规整）提供了自然拓展方向。

一句话总结：

> RelMel uses reference-assisted relative energy relations in mel spectrograms to make speech watermarks more robust to vocoding and additive noise while keeping storage and verification practical.

中文表述为：

> RelMel 通过在梅尔谱中嵌入可验证的相对能量关系，使语音水印在声码器重建和加性噪声攻击下更加稳定，同时保持参考存储和平台侧验证的实际可行性。
