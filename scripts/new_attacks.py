"""new_attacks.py — 审稿意见要求补充的两个攻击

1. revocoding_attack: 从水印波形重提 mel、用公开 vocoder 重合成。
   这是 mel 域水印最自然的攻击（审稿意见 R1-Method CRITICAL (a)）。

2. band_targeted_noise: 噪声集中在水印频段（审稿意见 R1-Method CRITICAL (c)）。
   实现在 FFT 域做锐截止带通，而不是 InverseMelScale+GriffinLim ——
   后者会对整个信号引入重建损伤，无法区分"攻击造成的损伤"和"逆变换损伤"。

两者都返回 melshield.attacks.AttackResult，可直接放进攻击字典使用。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from melshield.attacks import AttackFn, AttackResult
from melshield.mel import MelConfig, MelFrontend
from melshield.vocoders import build_vocoder


# ---------------------------------------------------------------- re-vocoding

def revocoding_attack(
    vocoder_name: str = "hifigan",
    checkpoint: str | None = None,
    config: str | None = None,
    mel_config: MelConfig | None = None,
    device: str = "cuda",
    command_template: str | None = None,
) -> AttackFn:
    """Re-vocoding: waveform -> mel -> vocoder -> waveform.

    攻击者丢弃原波形的细结构，只保留 mel 包络后重新合成。对 mel 域水印而言，
    水印本身就在 mel 里，所以这个攻击不必然抹除水印 —— 但重合成会引入
    vocoder 自身的重建误差，且丢掉 mel 量化以下的信息。

    注意：默认让攻击者使用与提供方**相同**的 checkpoint，这是最强攻击假设
    （攻击者拿到了同一个公开 vocoder）。传入不同 checkpoint 可模拟较弱攻击。
    """
    if mel_config is None:
        raise ValueError("必须显式传入 mel_config，避免与 vocoder 前端不一致")

    frontend = MelFrontend(mel_config, device=device)
    vocoder = build_vocoder(
        vocoder_name,
        mel_config=mel_config,
        device=device,
        checkpoint=checkpoint,
        config=config,
        command_template=command_template,
        sample_rate=mel_config.sample_rate,
    )

    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        bundle = frontend.waveform_to_normalized_logmel(waveform, sample_rate)
        # vocoder 吃的是 log-mel（未归一化），与 run_relmel_ljspeech.py 一致
        log_mel = torch.from_numpy(bundle.log_mel)
        revocoded = vocoder.synthesize(log_mel)
        return AttackResult(
            waveform=revocoded.clamp(-1.0, 1.0),
            sample_rate=int(vocoder.sample_rate),
        )

    return apply


# ------------------------------------------------------- band-targeted noise

def _slaney_hz_to_mel(f: float) -> float:
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    if f < min_log_hz:
        return f / f_sp
    logstep = math.log(6.4) / 27.0
    return min_log_hz / f_sp + math.log(f / min_log_hz) / logstep


def _slaney_mel_to_hz(m: float) -> float:
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    breakpoint_mel = min_log_hz / f_sp
    if m < breakpoint_mel:
        return m * f_sp
    logstep = math.log(6.4) / 27.0
    return min_log_hz * math.exp(logstep * (m - breakpoint_mel))


def mel_band_to_hz(
    band: tuple[int, int], n_mels: int, f_min: float, f_max: float
) -> tuple[float, float]:
    """把 mel bin 区间换算成 Hz 区间（slaney 刻度，与 torchaudio 一致）。"""
    lo, hi = band
    m_min, m_max = _slaney_hz_to_mel(f_min), _slaney_hz_to_mel(f_max)
    step = (m_max - m_min) / (n_mels + 1)
    return (
        _slaney_mel_to_hz(m_min + lo * step),
        _slaney_mel_to_hz(m_min + (hi + 1) * step),
    )


def band_targeted_noise(
    snr_db: float,
    band: tuple[int, int] = (20, 60),
    mel_config: MelConfig | None = None,
) -> AttackFn:
    """把噪声全部集中在水印频段内。

    这是"知道方案但没有密钥"的攻击者的显然选择：既然水印只写在 mel bin
    20:60，就没必要在带外浪费噪声功率。

    SNR 定义：噪声功率相对于**带内**信号功率。因此 band_noise10 与全频段
    noise10 不是等总功率对比 —— 前者在带内的噪声更强。这正是攻击者的意图，
    但报告时必须说明这个定义差异。

    实现：FFT 域锐截止带通，避免 mel 逆变换的重建损伤污染结果。
    """
    if mel_config is None:
        raise ValueError("必须显式传入 mel_config 以确定频段边界")

    f_lo, f_hi = mel_band_to_hz(
        band, mel_config.n_mels, mel_config.f_min, mel_config.resolved_f_max()
    )

    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        n = waveform.shape[-1]
        spec = torch.fft.rfft(waveform, dim=-1)
        freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate).to(waveform.device)
        mask = ((freqs >= f_lo) & (freqs <= f_hi)).to(waveform.dtype)

        # 带内信号功率（Parseval：用带通后的时域信号算，数值上更稳）
        in_band = torch.fft.irfft(spec * mask, n=n, dim=-1)
        band_power = in_band.pow(2).mean().clamp_min(1e-12)

        noise_power = band_power / (10.0 ** (snr_db / 10.0))
        white = torch.randn_like(waveform)
        noise_band = torch.fft.irfft(
            torch.fft.rfft(white, dim=-1) * mask, n=n, dim=-1
        )
        cur = noise_band.pow(2).mean().clamp_min(1e-12)
        noise_band = noise_band * torch.sqrt(noise_power / cur)

        return AttackResult(
            waveform=(waveform + noise_band).clamp(-1.0, 1.0),
            sample_rate=sample_rate,
        )

    return apply


def describe_band(band: tuple[int, int], mel_config: MelConfig) -> str:
    lo, hi = mel_band_to_hz(
        band, mel_config.n_mels, mel_config.f_min, mel_config.resolved_f_max()
    )
    return f"mel bins {band[0]}:{band[1]} -> [{lo:.0f}, {hi:.0f}] Hz"


if __name__ == "__main__":
    cfg = MelConfig(
        sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024,
        n_mels=80, f_min=0.0, f_max=8000.0, power=1.0, eps=1e-5,
    )
    print(describe_band((20, 60), cfg))

    fn = band_targeted_noise(10.0, band=(20, 60), mel_config=cfg)
    x = torch.randn(1, 22050) * 0.1
    out = fn(x, 22050)
    print(f"输入 {tuple(x.shape)} -> 输出 {tuple(out.waveform.shape)}")

    # 校验带内 SNR 是否真的接近 10 dB
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / 22050)
    f_lo, f_hi = mel_band_to_hz((20, 60), 80, 0.0, 8000.0)
    mask = ((freqs >= f_lo) & (freqs <= f_hi)).float()
    sig_b = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * mask, n=n, dim=-1)
    noi = out.waveform - x
    noi_b = torch.fft.irfft(torch.fft.rfft(noi, dim=-1) * mask, n=n, dim=-1)
    snr = 10 * torch.log10(sig_b.pow(2).mean() / noi_b.pow(2).mean().clamp_min(1e-12))
    print(f"实测带内 SNR = {float(snr):.2f} dB（目标 10.00）")
