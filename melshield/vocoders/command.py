from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..datasets import load_audio
from .base import MelVocoder


class CommandVocoder(MelVocoder):
    """Adapter for external vocoder scripts.

    The command template must contain ``{mel_npy}`` and ``{audio_wav}``.
    Example:
    ``python external/diffwave/inference.py --spectrogram_path {mel_npy} --output {audio_wav}``
    """

    def __init__(self, command_template: str, sample_rate: int = 22050) -> None:
        if "{mel_npy}" not in command_template or "{audio_wav}" not in command_template:
            raise ValueError("Command template must contain {mel_npy} and {audio_wav}.")
        self.command_template = command_template
        self.sample_rate = int(sample_rate)

    def synthesize(self, log_mel: torch.Tensor) -> torch.Tensor:
        if log_mel.ndim == 3:
            log_mel = log_mel.squeeze(0)
        mel_np = log_mel.detach().cpu().float().numpy().astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            mel_path = Path(tmp) / "mel.npy"
            wav_path = Path(tmp) / "audio.wav"
            np.save(mel_path, mel_np)
            command = self.command_template.format(mel_npy=mel_path, audio_wav=wav_path)
            subprocess.run(shlex.split(command), check=True)
            waveform, sample_rate = load_audio(wav_path)
        self.sample_rate = sample_rate
        return waveform
