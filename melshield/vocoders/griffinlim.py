from __future__ import annotations

import torch
import torchaudio

from ..mel import MelConfig
from .base import MelVocoder


class GriffinLimVocoder(MelVocoder):
    """A checkpoint-free vocoder for smoke tests, not for paper-quality results."""

    def __init__(self, mel_config: MelConfig, device: str = "cpu", n_iter: int = 64) -> None:
        self.mel_config = mel_config
        self.sample_rate = mel_config.sample_rate
        self.device = torch.device(device)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=mel_config.n_fft // 2 + 1,
            n_mels=mel_config.n_mels,
            sample_rate=mel_config.sample_rate,
            f_min=mel_config.f_min,
            f_max=mel_config.resolved_f_max(),
            norm="slaney",
            mel_scale="slaney",
        ).to(self.device)
        self.griffin = torchaudio.transforms.GriffinLim(
            n_fft=mel_config.n_fft,
            win_length=mel_config.win_length,
            hop_length=mel_config.hop_length,
            power=mel_config.power,
            n_iter=n_iter,
        ).to(self.device)

    @torch.no_grad()
    def synthesize(self, log_mel: torch.Tensor) -> torch.Tensor:
        if log_mel.ndim == 2:
            log_mel = log_mel.unsqueeze(0)
        log_mel = log_mel.to(self.device).float()
        mel = torch.exp(log_mel).clamp_min(1e-8)
        spec = self.inverse_mel(mel)
        wav = self.griffin(spec).detach().cpu()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav
