from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torchaudio

from .datasets import load_audio, save_audio


@dataclass(frozen=True)
class AttackResult:
    waveform: torch.Tensor
    sample_rate: int


AttackFn = Callable[[torch.Tensor, int], AttackResult]


def identity(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
    return AttackResult(waveform=waveform.clone(), sample_rate=sample_rate)


def amplitude_scale(gain: float = 0.8) -> AttackFn:
    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        return AttackResult((waveform * gain).clamp(-1.0, 1.0), sample_rate)

    return apply


def additive_noise(snr_db: float) -> AttackFn:
    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        power = waveform.pow(2).mean().clamp_min(1e-12)
        noise_power = power / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return AttackResult((waveform + noise).clamp(-1.0, 1.0), sample_rate)

    return apply


def resample_to(target_sample_rate: int = 16000) -> AttackFn:
    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        return AttackResult(
            torchaudio.functional.resample(waveform, sample_rate, target_sample_rate),
            target_sample_rate,
        )

    return apply


def lowpass(cutoff_hz: float = 3000.0) -> AttackFn:
    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        return AttackResult(
            torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_hz),
            sample_rate,
        )

    return apply


def bandpass(low_hz: float = 300.0, high_hz: float = 8000.0) -> AttackFn:
    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        high = min(high_hz, sample_rate * 0.49)
        y = torchaudio.functional.highpass_biquad(waveform, sample_rate, low_hz)
        y = torchaudio.functional.lowpass_biquad(y, sample_rate, high)
        return AttackResult(y.clamp(-1.0, 1.0), sample_rate)

    return apply


def echo(delay_ms: float = 80.0, decay: float = 0.35) -> AttackFn:
    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        delay = max(1, int(sample_rate * delay_ms / 1000.0))
        padded = torch.nn.functional.pad(waveform, (delay, 0))[:, : waveform.shape[1]]
        y = (waveform + decay * padded).clamp(-1.0, 1.0)
        return AttackResult(y, sample_rate)

    return apply


def ffmpeg_codec(codec: str, bitrate: str) -> AttackFn:
    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for MP3/AAC attacks.")
        suffix = ".mp3" if codec == "mp3" else ".m4a"
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "input.wav"
            encoded_path = Path(tmp) / f"encoded{suffix}"
            out_path = Path(tmp) / "decoded.wav"
            save_audio(in_path, waveform, sample_rate)
            cmd_encode = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(in_path),
                "-b:a",
                bitrate,
                str(encoded_path),
            ]
            cmd_decode = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(encoded_path),
                str(out_path),
            ]
            subprocess.run(cmd_encode, check=True)
            subprocess.run(cmd_decode, check=True)
            decoded, decoded_sr = load_audio(out_path)
        return AttackResult(decoded, decoded_sr)

    return apply


def build_attacks(names: list[str]) -> dict[str, AttackFn]:
    registry: dict[str, AttackFn] = {
        "none": identity,
        "scale": amplitude_scale(0.8),
        "rs16": resample_to(16000),
        "lowpass": lowpass(3000.0),
        "bandpass": bandpass(300.0, 8000.0),
        "echo": echo(),
        "noise20": additive_noise(20.0),
        "noise10": additive_noise(10.0),
        "noise5": additive_noise(5.0),
        "mp3": ffmpeg_codec("mp3", "128k"),
        "aac": ffmpeg_codec("aac", "96k"),
    }
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown attacks: {', '.join(unknown)}")
    return {name: registry[name] for name in names}
