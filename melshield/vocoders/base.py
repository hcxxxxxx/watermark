from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch

from ..mel import MelConfig


class MelVocoder(ABC):
    sample_rate: int

    @abstractmethod
    def synthesize(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Convert raw log-Mel [n_mels, frames] to waveform [1, time]."""


def build_vocoder(
    name: str,
    mel_config: MelConfig,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
    command_template: Optional[str] = None,
    **kwargs: Any,
) -> MelVocoder:
    name = name.lower()
    if name == "griffinlim":
        from .griffinlim import GriffinLimVocoder

        return GriffinLimVocoder(mel_config=mel_config, device=device)
    if name == "hifigan":
        if checkpoint is None or config is None:
            raise ValueError("HiFi-GAN requires --vocoder-checkpoint and --vocoder-config.")
        from .hifigan import HiFiGANVocoder

        return HiFiGANVocoder(
            checkpoint_path=Path(checkpoint),
            config_path=Path(config),
            device=device,
        )
    if name == "command":
        if command_template is None:
            raise ValueError("Command vocoder requires --vocoder-command.")
        from .command import CommandVocoder

        return CommandVocoder(
            command_template=command_template,
            sample_rate=int(kwargs.get("sample_rate", mel_config.sample_rate)),
        )
    raise ValueError(f"Unsupported vocoder: {name}")
