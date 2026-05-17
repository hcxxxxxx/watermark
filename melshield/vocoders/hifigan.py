from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from .base import MelVocoder


LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    )
                )
                for dilation in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            y = F.leaky_relu(x, LRELU_SLOPE)
            y = conv1(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            y = conv2(y)
            x = x + y
        return x

    def remove_weight_norm(self) -> None:
        for conv in list(self.convs1) + list(self.convs2):
            remove_weight_norm(conv)


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple[int, ...] = (1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    )
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            y = F.leaky_relu(x, LRELU_SLOPE)
            y = conv(y)
            x = x + y
        return x

    def remove_weight_norm(self) -> None:
        for conv in self.convs:
            remove_weight_norm(conv)


class Generator(nn.Module):
    def __init__(self, h: SimpleNamespace):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )
        resblock_cls = ResBlock1 if str(h.resblock) == "1" else ResBlock2
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            in_ch = h.upsample_initial_channel // (2**i)
            out_ch = h.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel,
                        rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )
            for kernel_size, dilation_sizes in zip(
                h.resblock_kernel_sizes, h.resblock_dilation_sizes
            ):
                self.resblocks.append(resblock_cls(out_ch, kernel_size, tuple(dilation_sizes)))
        self.conv_post = weight_norm(nn.Conv1d(out_ch, 1, 7, 1, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                res = self.resblocks[i * self.num_kernels + j](x)
                xs = res if xs is None else xs + res
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class HiFiGANVocoder(MelVocoder):
    """Native loader for common official HiFi-GAN generator checkpoints."""

    def __init__(self, checkpoint_path: Path, config_path: Path, device: str = "cpu") -> None:
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        self.h = SimpleNamespace(**cfg)
        self.sample_rate = int(self.h.sampling_rate)
        self.device = torch.device(device)
        self.generator = Generator(self.h).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        generator_state = state.get("generator", state)
        self.generator.load_state_dict(generator_state, strict=True)
        self.generator.remove_weight_norm()
        self.generator.eval()

    @torch.no_grad()
    def synthesize(self, log_mel: torch.Tensor) -> torch.Tensor:
        if log_mel.ndim == 2:
            log_mel = log_mel.unsqueeze(0)
        log_mel = log_mel.to(self.device).float()
        audio = self.generator(log_mel).detach().cpu()
        return audio.squeeze(0)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)
