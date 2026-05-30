# RelMel 工作核心思路说明

本文档用于让没有任何上下文背景的读者或 AI 模型理解本工作的研究问题、方法设计、实验结果、论文叙事和边界条件。它不是论文正文，而是一份写作和沟通用的技术说明。

## 0. 一句话总结

RelMel 是一种用于神经语音生成场景的 reference-assisted mel-domain speech watermarking（参考辅助的梅尔域语音水印）方法。它不是把 payload（载荷信息）写成某个 mel bin（梅尔频带格）的绝对扰动，而是写成多个时间块内两组 mel bins 之间的 relative energy relation（相对能量关系）。平台方保存生成时的 clean mel reference（干净梅尔参考）或其压缩版本，后续用疑似音频与 reference 对比，恢复 payload 并验证归属。

英文核心表述：

> RelMel uses reference-assisted relative energy relations in mel spectrograms to make speech watermarks more robust to vocoding and additive noise while keeping storage and verification practical.

中文核心表述：

> RelMel 通过在梅尔谱中嵌入可验证的相对能量关系，使语音水印在声码器重建和加性噪声攻击下更加稳定，同时保持参考存储和平台侧验证的实际可行性。

## 1. 研究背景：为什么语音水印重要

当前 TTS（text-to-speech，文本转语音）、voice cloning（语音克隆）、voice conversion（语音转换）和 neural speech generation（神经语音生成）模型已经可以生成高自然度语音。随之出现几个现实问题：

- 生成语音可能被用于诈骗、冒充、虚假内容传播。
- 平台方希望证明某段疑似音频是否由自己的 TTS 服务生成。
- 内容分发或审核方希望在音频被压缩、加噪、重采样、裁剪、回放后仍能追踪来源。
- 普通 metadata（元数据）容易被删除，不能作为可靠来源证明。

Audio watermarking（音频水印）因此成为一种重要方案。水印方法通常把某种 payload（载荷信息，如 bit string、身份码、生成记录 ID）嵌入音频中，使音频在听感上尽量不变，但检测器可以在之后恢复或验证这个 payload。

## 2. 本工作的应用设定：不是通用盲检测，而是参考辅助验证

很多音频水印工作追求 blind detection（盲检测）：检测器只拿到疑似音频，不需要原始音频或生成记录，就判断是否含水印并恢复 payload。这种设定很通用，但也很难，尤其在强噪声、强压缩、复杂后处理下容易失败。

本文的设定不同：reference-assisted verification（参考辅助验证）。

在这个设定中，平台方在生成语音时保存一些 reference information（参考信息），例如：

- utterance id（语音编号）
- secret key（密钥）
- payload（载荷）
- clean mel spectrogram（生成时未加水印或加水印前的干净梅尔谱）
- 或 clean mel 的压缩版本

后续如果出现一段疑似音频，平台方可以问：

> 这段音频是否来自我保存的某条生成记录，并且是否携带对应 payload？

这和 blind detection 的目标不同。RelMel 的目标不是让任何第三方无上下文地检测所有水印，而是帮助服务方做 private provenance verification（私有来源验证）、forensic attribution（取证归属）和 audit（审计）。

这个设定有实际合理性：

- TTS 服务方通常本来就保存请求日志、speaker id、模型版本、生成配置等记录。
- 在许多平台场景中，验证者不是公众，而是模型或服务提供方。
- 保存完整 waveform（波形音频）成本较大，但保存压缩后的 mel reference 成本可控。
- 实验显示，只保存水印频带 `20:60` 的 8-bit clean mel，平均约 22KB/reference，已经接近 float32 reference 的验证效果。

因此，reference 不是本文偷偷依赖的“作弊信息”，而是本文主动定义的应用场景和系统假设。

## 3. 为什么选择 Mel 域水印

Mel spectrogram（梅尔频谱）是许多神经语音生成系统中的关键中间表示。典型 TTS 流程是：

```text
text -> acoustic model -> mel spectrogram -> vocoder -> waveform audio
```

其中 vocoder（声码器）可以是 HiFi-GAN、DiffWave、WaveGlow、WaveRNN 等。声学模型生成 mel，声码器把 mel 转换成最终 waveform。

在 mel 域加入水印有几个优势：

1. **贴近生成链路**  
   对 TTS 系统来说，mel 是自然的中间层。平台方在声码器之前就可以嵌入水印，不需要等到 waveform 后处理。

2. **更接近听觉结构**  
   mel 频带是按照人耳感知尺度设计的。相比直接修改 waveform，控制 mel 频带能量更容易和感知质量联系起来。

3. **跨声码器传播**  
   如果水印在 mel 中足够稳定，它可以通过不同 vocoder 传播到 waveform。本文已经在 HiFi-GAN 和 DiffWave 上测试。

4. **可解释性更强**  
   waveform-domain neural watermark 通常依赖神经检测器；mel-domain 方法可以设计更可解释的频带关系、能量差和投票机制。

5. **适合 reference-assisted verification**  
   如果平台方保存 clean mel，那么疑似音频重新提取 mel 后，可以直接和 reference 比较，形成 residual（残差），从中恢复水印。

## 4. Baseline：MelShield 的基本思路

MelShield 是本文最重要的同范式 baseline。它同样是在 mel 域嵌入水印，并使用 clean mel reference 验证。

MelShield 的基本流程可以概括为：

1. 根据 secret key（密钥）和 utterance id（语音编号）生成伪随机 pattern（模式）。
2. 对每个 payload bit，在 mel 上加入正向或负向扰动。
3. 将加水印后的 mel 输入 vocoder，得到 watermarked waveform（加水印波形音频）。
4. 验证时，将疑似音频重新转换为 mel。
5. 用 suspect mel 减去保存的 clean reference mel，得到 residual（残差）。
6. 检测 residual 是否和密钥生成的 pattern 相关，从而恢复 payload。

MelShield 证明了 mel-domain reference-assisted watermarking 是可行的。但它更依赖 absolute perturbation pattern（绝对扰动模式）。也就是说，它更关心某些位置是否出现了预期方向和强度的残差。

这种设计在强噪声下会遇到问题：

- 加性噪声会把 residual 中的水印信号掩盖。
- 声码器会引入非线性重建误差。
- 某些 mel bins 本身不稳定，水印扰动传播到 waveform 后不易恢复。
- 绝对幅度对整体音量、频带能量、归一化范围等因素比较敏感。

实验上，MelShield 在 `noise10/noise5` 这类较强加性噪声条件下 bit accuracy 明显下降。本文的核心动机就是：能否在同样 reference-assisted 的 mel 域设定下，让水印不依赖脆弱的绝对模式，而依赖更稳定的相对关系？

## 5. RelMel 的核心观察

RelMel 的关键观察是：

> 在同一个局部时间块内，两组 mel 频带之间的相对能量关系，往往比单个 mel bin 的绝对残差更稳定。

换句话说，与其问：

```text
某个 mel bin 有没有按预期变大？
```

不如问：

```text
positive group 相对 negative group 是否按 payload 指定方向发生了能量偏移？
```

这种设计类似差分测量。许多攻击或声码器误差会对一段局部 mel 区域产生共同影响，例如整体增益变化、局部重建偏差、轻微滤波等。相对差分可以抵消一部分共同扰动。

## 6. RelMel 如何编码一个 bit

对每个 payload bit 和每个时间 block，RelMel 在指定 mel band 中选择两组 mel bins：

- positive group（正组）
- negative group（负组）

假设要嵌入的 bit 是 1：

- 提高 positive group 的能量
- 降低 negative group 的能量

假设要嵌入的 bit 是 0：

- 降低 positive group 的能量
- 提高 negative group 的能量

检测时，RelMel 不直接看某个频带的绝对值，而是计算 residual 上两组之间的差：

```text
score = mean(residual on positive group) - mean(residual on negative group)
```

如果 score 为正，说明 positive group 相对 negative group 更强，倾向于 bit=1。  
如果 score 为负，说明 negative group 相对 positive group 更强，倾向于 bit=0。

这里的 residual 来自：

```text
residual = suspect_mel - reference_clean_mel
```

也就是说，RelMel 依赖 clean reference 作为坐标系，观察疑似音频相对于原始 clean mel 的变化。

## 7. RelMel 的完整嵌入流程

一次 RelMel 嵌入可以理解为以下步骤：

1. 输入 clean mel spectrogram。
2. 根据 utterance id 和 secret key 生成 payload 或读取指定 payload。
3. 将时间轴切成多个 block，例如每个 block 8 帧，stride 也是 8 帧。
4. 对每个 block 选择若干 payload bits 写入。
5. 对每个 bit，在指定 mel band 中选择 positive group 和 negative group。
6. 根据 bit 值对两组 mel bins 做相反方向的能量调整。
7. 对扰动进行 mask 和 boundary control，避免过度破坏音质。
8. 将 watermarked mel 输入 vocoder，得到最终 waveform。

当前主配置中，常见参数如下：

- `band=20:60`：只在第 20 到 60 个 mel bins 范围内嵌入。
- `block_frames=8`：每个时间块 8 个 mel frames。
- `block_stride=8`：块之间不重叠或基本按 8 帧移动。
- `bits_per_block=6`：每个 block 写入 6 个 payload bits。
- `pair_bins=6`：每组候选频带对包含若干 mel bins。
- `pair_candidates=16`：为每个 bit/block 生成 16 个候选对，选择最可靠的一组。
- `threshold=0.75`：验证时 bit accuracy 达到 0.75 即通过。

## 8. 块级重复嵌入为什么重要

RelMel 不是只在一个位置写入 payload。它把 payload 分散写入多个时间 block，并在检测时把多个 block 的证据合并。

这样做有几个好处：

1. **抗局部破坏**  
   如果某些时间段被噪声、回声或裁剪破坏，其他 block 仍然保留水印证据。

2. **支持片段验证**  
   只保留一部分音频时，只要片段中还有足够 block，就可以恢复 payload。

3. **投票提高稳定性**  
   单个 block 的判断可能出错，但多个 block 聚合后，平均 bit accuracy 更稳。

4. **适合平台取证**  
   平台方不一定拿到完整音频，可能只拿到社交平台上的短片段。块级分布可以提高实际可用性。

实验结果显示：

- 已知裁剪位置时，25% 音频片段仍有约 0.989 到 0.991 bit accuracy，验证率为 1.000。
- 盲搜索裁剪位置时，25% 片段仍有约 0.985 到 0.991 bit accuracy，验证率为 0.998 到 1.000。
- 盲搜索的起点误差中位数约为 0 到 1 个 mel frame。

这说明水印证据在时间轴上分布较均匀，并且不完全依赖完整音频。

## 9. Reliable pair selection：为什么不能随机选频带

RelMel 的一个关键机制是 reliable pair selection（可靠候选对选择）。

如果只是随机选择 positive group 和 negative group，会遇到几个问题：

- 某些 mel bins 能量太低，声码器重建后不稳定。
- 某些 bins 接近归一化范围边界，继续增加或降低容易被 clipping（裁剪）或 mask 限制。
- positive group 和 negative group 的可靠性不平衡，会导致 score 有偏。
- 某些频带对在当前 utterance 中几乎没有语音结构，扰动传播到 waveform 后容易丢失。

因此，RelMel 对每个 bit/block 不是只生成一个 pair，而是生成多个 candidate pairs（候选对），再用 clean mel 的局部结构选择最可靠的一组。

可靠性主要来自两个方面：

1. **Boundary reliability（边界可靠性）**  
   如果某个 mel bin 已经接近允许范围的上界或下界，那么继续扰动会受限，也更容易产生音质损伤。因此，距离边界越远的位置越安全。

2. **Energy reliability（能量可靠性）**  
   能量太低的位置更容易被声码器忽略或被噪声淹没。具有一定语音能量的位置通常更稳定。

选择目标不是单纯找能量最大的 bin，而是找：

- positive group 可靠
- negative group 可靠
- 两组可靠性尽量平衡
- 嵌入后不会明显破坏音质

一个直观的候选对评分可以理解为：

```text
candidate_score = min(pos_reliability, neg_reliability)
                  - balance_penalty
```

即优先选择“两边都可靠”的候选，而不是一边很强、一边很弱的候选。

消融实验显示，`pair_candidates` 从 1 增加到 16 后，HiFi-GAN random500 下：

- `noise20` 从 0.9459 提升到 0.9952
- `noise10` 从 0.7674 提升到 0.9439
- `noise5` 从 0.6684 提升到 0.8607
- 干净 PESQ 只从 3.5485 降到 3.5102

这说明 reliable pair selection 是 RelMel 鲁棒性提升的核心来源之一。

## 10. Verification 标准：到底怎样算通过

RelMel 的检测目标不是单纯判断“有没有水印”，而是验证一个具体 payload 是否匹配。

验证流程：

1. 根据 utterance id 和 key 得到 expected payload。
2. 从疑似音频中恢复 decoded payload。
3. 计算 bit accuracy：

```text
bit_accuracy = number_of_correct_bits / total_payload_bits
```

4. 如果 bit accuracy 大于等于 threshold，则验证通过。

当前主实验：

- payload_bits = 32
- threshold = 0.75

这意味着验证通过不要求 32 bit 全部正确，而是大约至少 24/32 bit 正确即可通过。

为什么不要求 32 bit 全对？

- 强噪声下少量 bit 错误是正常的。
- 水印验证通常需要容错。
- 只要错误 payload、错误 key、错误 reference 的通过率足够低，阈值式验证就是合理的。

负控实验显示：

- 正确 reference 下，`none/noise20` 验证率为 1.000。
- clean_unmarked、wrong_key、wrong_payload、wrong_reference 的 bit accuracy 基本回到 0.5 附近。
- random2000 无攻击扩展中，wrong_key、wrong_payload、wrong_reference 的验证率分别为 0.15%、0.40%、0.35%。

这说明阈值 0.75 并没有导致明显过度自信。RelMel 的通过依赖正确 reference、key 和 payload 的组合。

## 11. Confidence 不能单独作为归属依据

实验中还有 confidence（置信度）指标。需要注意：

> confidence 不能单独作为最终验证依据。

例如 wrong_payload 条件下，音频中确实存在某种水印方向的强信号，所以 confidence 可能和正例接近；但因为声明的 payload 是错的，bit accuracy 回到随机水平，verification rate 也很低。

因此，论文中应该明确：

- confidence 表示检测到的水印方向信号强度。
- bit accuracy 表示恢复 payload 与 expected payload 的匹配程度。
- verification rate 是最终判定通过比例。
- 归属判断应基于 payload match，而不是单独 confidence。

## 12. Reference 存储是否现实

Reference-assisted verification 最容易被质疑的问题是：是不是需要保存大量 clean mel？

实验显示，不需要保存完整 float32 mel。

Reference compression（参考压缩）实验比较了几种格式：

- float32：完整 80-bin mel，约 176.72KB/reference。
- float16：约 88.36KB/reference。
- uint8：完整 mel 8-bit，约 44.18KB/reference。
- uint6：约 33.14KB/reference。
- uint4：约 22.09KB/reference，但鲁棒性下降明显。
- band_uint8：只保存水印频带 `20:60` 的 8-bit mel，约 22.09KB/reference。

其中最推荐的是 `band_uint8`，因为它只保存真正用于水印验证的频带，存储很小，但鲁棒性接近 float32：

- `none`：1.0000 / 1.000 VR
- `noise20`：0.9941 / 1.000 VR
- `noise10`：0.9399 / 0.992 VR
- `noise5`：0.8574 / 0.920 VR

这说明 reference-based verification 在平台部署中是可以讲得通的。对平台方来说，22KB/utterance 的参考信息并不离谱，尤其相比保存完整 wav 或复杂日志而言。

## 13. 与 MelShield 的核心区别

RelMel 和 MelShield 都是 mel-domain、reference-assisted 方法，但核心区别如下：

| 方面 | MelShield | RelMel |
|---|---|---|
| 水印表示 | 绝对扰动模式 | 相对能量关系 |
| 检测信号 | residual 与伪随机 pattern 的相关性 | positive/negative groups 的 residual 差分 |
| 频带选择 | 更接近固定或随机模式 | 可靠候选对选择 |
| 鲁棒性来源 | pattern 重复和阈值检测 | 相对差分、可靠 pair、块级投票 |
| 强噪声表现 | `noise10/noise5` 下降明显 | 在相近音质下明显更稳 |
| 可解释性 | 能解释为模式相关 | 能解释为局部频带相对关系 |

在 DiffWave 质量匹配主对照中：

- MelShield + DiffWave：干净 PESQ 3.5492
  - `noise20=0.9658`
  - `noise10=0.8212`
  - `noise5=0.7210`
- RelMel + DiffWave：干净 PESQ 3.5443
  - `noise20=0.9914`
  - `noise10=0.9225`
  - `noise5=0.8308`

在相近音质下，RelMel 对 `noise10` 和 `noise5` 的提升尤其明显。

## 14. 与 AudioSeal / WavMark 的关系

AudioSeal 和 WavMark 是 waveform-domain（波形域）baseline。它们通常直接在 waveform 上嵌入 16-bit payload，并用对应检测器恢复。

它们和 RelMel 属于不同范式：

- AudioSeal/WavMark：waveform-domain，通常更接近 blind 或 detector-based。
- RelMel/MelShield：mel-domain，reference-assisted。
- AudioSeal/WavMark 主实验是 16-bit payload。
- RelMel/MelShield 主实验是 32-bit payload。

因此论文中不应把它们和 RelMel 混成完全公平同范式对照，而应作为 strong waveform baselines（强波形基线）单独成组。

实验结论比较清楚：

- AudioSeal/WavMark 在压缩、缩放、重采样、滤波、回声等非噪声攻击下非常强。
- 它们引入的干净音质损失通常也较小，PESQ 很高。
- 但在强加性噪声下，尤其 `noise10/noise5`，detector-based verification 或 forced bit accuracy 明显下降。

例如：

- AudioSeal + HiFi-GAN：`noise20` verification 0.304，`noise10/noise5` verification 0。
- AudioSeal + DiffWave：`noise20` verification 0.138，`noise10/noise5` verification 0。
- WavMark forced + HiFi-GAN：`noise20=0.7523`，`noise10=0.5114`，`noise5=0.4926`。
- WavMark forced + DiffWave：`noise20=0.6700`，`noise10=0.5120`，`noise5=0.4978`。

这些结果支持一种谨慎表述：

> 在本文的 reference-assisted TTS verification setting 下，RelMel 在强加性噪声条件下比这些 waveform baselines 更稳定；但 AudioSeal 和 WavMark 在非噪声攻击和干净音质方面仍是强 baseline。

不要写成“RelMel 全面碾压 AudioSeal/WavMark”。

## 15. 主实验结果摘要

当前主结果默认数据集是 LJSpeech，sample mode 是 random，seed 是 2026。

### HiFi-GAN 主结果

RelMel + HiFi-GAN 主配置：

- `alpha=0.435`
- `band=20:60`
- `mask_floor=0.20`
- `boundary_margin=0.01`
- `block_frames=8`
- `block_stride=8`
- `bits_per_block=6`
- `pair_bins=6`
- `pair_candidates=16`
- `detector_mode=plain`

random500 全攻击结果：

- `none`：ACC 0.9999，VR 1.000，PESQ 3.5102，STOI 0.9685
- `noise20`：ACC 0.9946，VR 1.000，PESQ 1.5574，STOI 0.9527
- `noise10`：ACC 0.9455，VR 0.990，PESQ 1.0994，STOI 0.8991
- `noise5`：ACC 0.8648，VR 0.932，PESQ 1.0416，STOI 0.8430
- mp3、aac、scale、rs16、bandpass、lowpass、echo 等非噪声攻击基本接近 1.000 验证率。

### DiffWave 主结果

RelMel + DiffWave 主配置：

- `alpha=0.35`
- `band=20:60`
- `mask_floor=0.20`
- `boundary_margin=0.01`
- 其他核心参数与 HiFi-GAN 主配置一致
- DiffWave sampling seed 固定为 0

random500 全攻击结果：

- `none`：ACC 0.9998，VR 1.000，PESQ 3.5443，STOI 0.9612
- `noise20`：ACC 0.9914，VR 1.000，PESQ 1.4930，STOI 0.9438
- `noise10`：ACC 0.9225，VR 0.984，PESQ 1.0787，STOI 0.8932
- `noise5`：ACC 0.8308，VR 0.852，PESQ 1.0351，STOI 0.8437
- 非噪声攻击基本稳定。

### 质量匹配 MelShield + DiffWave 对照

MelShield + DiffWave 本地复现质量匹配配置：

- `alpha=0.060`
- `band=20:60`
- `mask_floor=0.05`
- `energy_gamma=0.75`
- `boundary_margin=0.02`
- `threshold=0.61`

random500 全攻击结果：

- `none`：ACC 0.9998，VR 1.000，PESQ 3.5492
- `noise20`：ACC 0.9658，VR 1.000
- `noise10`：ACC 0.8212，VR 0.966
- `noise5`：ACC 0.7210，VR 0.872

与 RelMel + DiffWave 相比，干净 PESQ 基本匹配，但 RelMel 在强噪声下更稳。

## 16. 负控实验：证明不是假阳性

Reference-based 方法必须证明：不是随便拿一个 reference 都能过，也不是只要有水印就会对任意 payload 过。

因此做了多种 negative controls（负控）：

- `correct`：正确 reference、正确 key、正确 payload。
- `clean_unmarked`：未加水印音频。
- `wrong_key`：密钥错误。
- `wrong_payload`：音频有水印，但声明 payload 错误。
- `wrong_reference`：使用另一条语音的完整 reference。
- `wrong_reference_mel`：只替换 clean mel，保留当前 utterance id 和 payload，属于半错误 reference。

random500，HiFi-GAN，`none/noise20`：

- correct 在 `none/noise20` 下 VR 都是 1.000。
- clean_unmarked、wrong_key、wrong_payload、wrong_reference 的 ACC 基本约 0.5。
- 完整 wrong_reference 的 VR 只有 0.2% 到 0.4%。
- wrong_reference_mel 的 `none` VR 为 2.0%，略高，因为它是半错误 reference，论文中应谨慎解释。

random2000，无攻击扩展：

- clean_unmarked：ACC 0.5008，VR 0.0040
- wrong_key：ACC 0.5030，VR 0.0015
- wrong_payload：ACC 0.5000，VR 0.0040
- wrong_reference：ACC 0.4991，VR 0.0035
- correct：ACC 0.9999，VR 1.000

这个实验是论文中证明 reference-assisted verification 合理性的关键证据。

## 17. 局部片段与盲搜索裁剪

局部片段实验回答一个实际问题：

> 如果只拿到原音频的一小段，RelMel 还能验证吗？

已知位置实验中，验证器知道裁剪片段在 reference 中的位置。结果显示：

- 25% 片段：约 0.989 到 0.991 ACC，VR 1.000。
- 50% 片段：约 0.998 以上 ACC，VR 1.000。

更强的盲搜索实验中，验证器不知道裁剪起点，而是在 reference 上滑窗搜索。设置：

- `search_step_frames=4`
- 片段包括 start25、middle25、end25、start50、middle50、end50

结果：

- start25：ACC 0.9908，VR 1.000
- middle25：ACC 0.9854，VR 0.998
- end25：ACC 0.9884，VR 0.998
- start50：ACC 0.9984，VR 1.000
- middle50：ACC 0.9973，VR 1.000
- end50：ACC 0.9989，VR 1.000

搜索定位误差很小：

- start 片段起点误差均值/中位数为 0。
- middle/end 片段起点误差中位数约 1 个 mel frame。

这说明 RelMel 不只是“完整音频可验证”，也可以处理局部片段和未知裁剪位置。

## 18. 现代攻击实验与边界

除了 MelShield 原文中的常规攻击，本文还测试了更多现代音频水印常见攻击：

- `mp3_64`：64 kbps MP3 压缩。
- `aac_48`：48 kbps AAC 压缩。
- `rs8`：重采样到 8 kHz。
- `rs24`：重采样到 24 kHz。
- `lowpass2k`：2 kHz 低通滤波。
- `bandpass_wide`：100 Hz 到 7 kHz 带通滤波。
- `clip05`：幅度裁剪到 ±0.5。
- `quant8`：8-bit 均匀量化。
- `reverb`：更强混响。
- `noise0`：0 dB 极强加性噪声。
- `speed090/speed110`：0.9/1.1 倍速。
- `pitch_up/pitch_down`：约 ±100 cents，也就是约 1 个半音。

结果说明：

- RelMel 对低码率压缩、滤波、重采样、裁剪、量化、混响总体很稳。
- 0 dB 噪声是加性噪声鲁棒性的边界，HiFi-GAN 下 ACC 约 0.7516，VR 0.598；DiffWave 下 ACC 约 0.7294，VR 0.486。
- 强 speed change 和 pitch shift 是当前方法的主要弱点。

为什么 speed/pitch 会弱？

- RelMel 当前检测依赖 reference mel 和 suspect mel 之间的 frame-level alignment（帧级对齐）。
- speed change 改变时间轴长度，导致 block 对不齐。
- pitch shift 改变频率结构，导致原先选择的 mel band/group 关系发生偏移。
- 当前只做小范围 shift alignment，不能完全修复这种时间/频率同步破坏。

进一步的强度曲线显示：

- ±25 cents pitch：几乎不影响，VR 1.000。
- ±50 cents pitch：仍保持约 0.995 ACC 和 1.000 VR。
- ±100 cents pitch：下降到约 0.78 ACC，VR 约 0.72。
- ±3% speed：基本可用，VR 约 0.996 到 0.998。
- ±5% speed：开始下降，VR 约 0.85 到 0.906。
- ±10% speed：明显失败，VR 约 0.26 到 0.32。

因此论文中应主动承认：

> RelMel 的主要边界不是常规压缩或噪声，而是强时间/频率同步破坏。未来可通过 multi-scale alignment（多尺度对齐）、DTW（动态时间规整）或 pitch-aware matching（变调感知匹配）增强。

## 19. 指标解释

论文和实验中常见指标如下：

### Bit accuracy / ACC（比特准确率）

恢复 payload 与 expected payload 的逐比特平均正确率。比如 32 bit 中 30 bit 正确，则 ACC = 30/32 = 0.9375。

### Verification rate / VR（验证率）

样本级通过比例。若某条音频的 bit accuracy 达到 threshold，则该条音频验证通过。VR 是所有样本中通过的比例。

### Payload bit accuracy / Code bit accuracy

当前主实验 `ecc_repeat=1`，payload bits 和 code bits 都是 32，因此 `mean_bit_acc`、`mean_payload_bit_acc`、`mean_code_bit_acc` 三者通常相同。若之后加入 ECC（error-correcting code，纠错码）或 repeat coding，它们才会区分。

### Confidence（置信度）

表示检测到的水印方向信号强度，但不等于归属判断。最终归属应看 payload 是否匹配。

### PESQ

Perceptual Evaluation of Speech Quality，客观语音质量指标。本文通常报告水印或攻击后音频相对无水印 benchmark audio 的 PESQ。

注意：`noise20/noise10/noise5` 的 PESQ 很低是正常的，因为这些攻击本身极大破坏语音质量。论文中用于控制水印音质的主要是 `none_pesq_bm`，也就是无攻击条件下水印音频相对 benchmark 音频的 PESQ。

### STOI

Short-Time Objective Intelligibility，客观可懂度指标。值越高通常表示语音可懂度越好。

### mean_votes / min_votes

表示每个 payload bit 聚合到的 block 投票数量。片段越短，votes 越少。

## 20. 攻击含义

常规攻击：

- `none`：不做攻击。
- `mp3`：MP3 有损压缩。
- `aac`：AAC 有损压缩。
- `scale`：音量缩放。
- `rs16`：重采样到 16 kHz 再恢复。
- `bandpass`：带通滤波。
- `lowpass`：低通滤波。
- `noise20`：20 dB SNR 加性噪声。
- `noise10`：10 dB SNR 加性噪声。
- `noise5`：5 dB SNR 加性噪声。
- `echo`：回声攻击。

扩展攻击：

- `mp3_64`：更低码率 MP3。
- `aac_48`：更低码率 AAC。
- `rs8`：8 kHz 重采样，强带宽限制。
- `rs24`：24 kHz 重采样。
- `lowpass2k`：更强低通。
- `bandpass_wide`：较宽带通。
- `clip05`：幅度裁剪。
- `quant8`：8-bit 量化。
- `reverb`：混响。
- `noise0`：0 dB 极强噪声。
- `speed090/speed110`：时间尺度变化。
- `pitch_up/pitch_down`：变调。

## 21. 论文应该怎样讲故事

建议论文故事线如下：

1. 神经语音生成需要可靠来源验证。
2. 完全 blind watermarking 很重要，但平台侧 verification 是另一个现实设定。
3. 在 TTS 场景中，mel 是自然的嵌入层，reference mel 可以被平台保存。
4. MelShield 证明了 mel-domain reference-assisted watermarking 可行，但绝对扰动模式在强噪声下不够稳。
5. RelMel 将 payload 编码为 block-wise relative mel energy relations。
6. Reliable pair selection 使用 clean mel reference 选择稳定、平衡的正负频带组。
7. 实验显示 RelMel 在 HiFi-GAN 和 DiffWave 上都显著提升加性噪声鲁棒性。
8. 负控实验说明不会对错误 key/payload/reference 过度通过。
9. Reference 压缩说明实际存储可行。
10. 片段和盲搜索实验说明可处理局部片段。
11. 现代攻击实验说明常规信号处理鲁棒，但强 speed/pitch 同步破坏是边界。

## 22. 可以放在论文中的贡献点

建议贡献表述为：

1. **Relative mel-energy watermarking formulation**  
   提出将语音水印 payload 编码为 block-wise relative energy relations，而不是绝对扰动模式。

2. **Reliability-aware pair selection**  
   利用 clean mel reference 选择稳定、边界安全、正负组平衡的 mel-bin pairs，提高强噪声下的鲁棒性。

3. **Practical reference-assisted verification**  
   系统评估负控假阳性、reference 压缩、局部片段、盲搜索裁剪位置和跨声码器表现，说明该设定适合平台侧归属验证。

4. **Comprehensive comparison**  
   与 MelShield、AudioSeal、WavMark 等 baseline 对比，并明确区分 mel-domain reference-assisted 方法与 waveform-domain baseline。

## 23. 论文中必须避免的过度表述

不要写：

- RelMel 是 blind watermarking。
- RelMel 不需要 reference。
- RelMel 全面碾压 AudioSeal/WavMark。
- RelMel 对所有攻击都鲁棒。
- confidence 可以单独判断归属。
- 32 bit 必须全对才算通过。
- PESQ 3.5 是某篇原文严格规定的通用标准。
- reported baseline 和本地复现 baseline 可以混为同一公平协议。

应该写：

- RelMel 是 reference-assisted verification 方法。
- 它适合平台侧 provenance verification 和 forensic attribution。
- 它在强加性噪声下相比质量匹配 MelShield 有明显优势。
- 它和 AudioSeal/WavMark 属于不同范式，优势主要体现在本文的 reference-assisted TTS setting 下。
- 它对强 speed/pitch 同步破坏仍有局限。

## 24. 当前最适合写进主文的实验

主文建议优先放：

1. RelMel vs MelShield 主对比表，覆盖 HiFi-GAN 和 DiffWave。
2. RelMel 全攻击表，至少放 HiFi-GAN/DiffWave 摘要。
3. AudioSeal/WavMark waveform baseline 摘要。
4. Reference false positive random2000 表。
5. Reference compression 表，突出 22KB band_uint8。
6. Blind fragment verification 表。
7. Speed/pitch 边界小表或图。

补充材料可以放：

- pair_candidates 消融。
- 多随机种子稳定性。
- DiffWave 调参过程。
- 已知位置片段验证。
- modern attacks 详细表。
- AudioSeal/WavMark 完整攻击表。

## 25. 如果要继续完善实验

后续最有价值的方向：

1. 将主方法和关键 MelShield 对照扩到 random2000，提高统计说服力。
2. 尝试 AudioMarkBench 或 RAW-Bench 兼容攻击协议，增强 benchmark 认可度。
3. 加入 neural codec 攻击，如 EnCodec、DAC、SoundStream。
4. 在 LibriSpeech 或 Common Voice 上补一个跨数据集验证。
5. 为 speed/pitch 引入更强 alignment，再验证是否能修复同步破坏。

不建议把所有 pilot 或消融都扩到 random2000，因为边际收益较低。

## 26. 给其他 AI 的写作提示

如果让另一个 AI 基于本文档写 abstract 或 introduction，应告诉它：

- 这篇论文的设定是 reference-assisted，不是 blind。
- 方法名是 RelMel，可以解释为 Relative Mel-Energy Relations。
- 核心创新是相对能量关系和可靠候选对选择。
- 主优势是强加性噪声下鲁棒性提升。
- 关键实验证据包括跨 HiFi-GAN/DiffWave、MelShield 对照、AudioSeal/WavMark 对比、负控、reference 压缩、盲片段验证。
- 需要诚实说明 speed/pitch 是边界。
- 语气要克制，不能把不同范式 baseline 写成完全同条件碾压。

## 27. 最短版本

如果只允许用一段话介绍本工作，可以写：

> RelMel is a reference-assisted mel-domain watermarking method for neural speech provenance verification. Instead of encoding payload bits as absolute perturbation patterns, it embeds them as relative energy relations between reliable positive and negative mel-bin groups across repeated time blocks. With a clean mel reference retained by the service provider, RelMel verifies whether a suspect audio clip matches a claimed payload, key, and generation record. Experiments on LJSpeech with HiFi-GAN and DiffWave show that RelMel substantially improves additive-noise robustness over quality-matched MelShield, remains competitive under common signal-processing attacks, produces low false positives under wrong-key/payload/reference controls, supports compressed references of about 22KB per utterance, and can verify partial clips via blind reference search. Its current main limitation is strong time/frequency synchronization distortion such as large speed changes and pitch shifts.

