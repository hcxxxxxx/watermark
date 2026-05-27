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


def clip_amplitude(limit: float = 0.5) -> AttackFn:
    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        return AttackResult(waveform.clamp(-limit, limit), sample_rate)

    return apply


def uniform_quantize(bits: int = 8) -> AttackFn:
    levels = float((1 << bits) - 1)

    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        y = torch.round(((waveform.clamp(-1.0, 1.0) + 1.0) * 0.5) * levels) / levels
        y = y * 2.0 - 1.0
        return AttackResult(y.clamp(-1.0, 1.0), sample_rate)

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


def ffmpeg_filter(filter_expr: str, extension: str = ".wav") -> AttackFn:
    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for this attack.")
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "input.wav"
            out_path = Path(tmp) / f"output{extension}"
            save_audio(in_path, waveform, sample_rate)
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(in_path),
                "-af",
                filter_expr,
                str(out_path),
            ]
            subprocess.run(cmd, check=True)
            decoded, decoded_sr = load_audio(out_path)
        return AttackResult(decoded, decoded_sr)

    return apply


def ffmpeg_pitch(factor: float) -> AttackFn:
    def apply(waveform: torch.Tensor, sample_rate: int) -> AttackResult:
        inverse = 1.0 / factor
        return ffmpeg_filter(
            f"asetrate={sample_rate}*{factor:.6f},aresample={sample_rate},atempo={inverse:.6f}"
        )(waveform, sample_rate)

    return apply


def build_attacks(names: list[str]) -> dict[str, AttackFn]:
    registry: dict[str, AttackFn] = {
        "none": identity,
        "scale": amplitude_scale(0.8),
        "clip05": clip_amplitude(0.5),
        "quant8": uniform_quantize(8),
        "rs16": resample_to(16000),
        "rs8": resample_to(8000),
        "rs24": resample_to(24000),
        "lowpass": lowpass(3000.0),
        "lowpass2k": lowpass(2000.0),
        "bandpass": bandpass(300.0, 8000.0),
        "bandpass_wide": bandpass(100.0, 7000.0),
        "echo": echo(),
        "reverb": echo(delay_ms=120.0, decay=0.55),
        "noise20": additive_noise(20.0),
        "noise10": additive_noise(10.0),
        "noise5": additive_noise(5.0),
        "noise0": additive_noise(0.0),
        "mp3": ffmpeg_codec("mp3", "128k"),
        "mp3_64": ffmpeg_codec("mp3", "64k"),
        "aac": ffmpeg_codec("aac", "96k"),
        "aac_48": ffmpeg_codec("aac", "48k"),
        "speed090": ffmpeg_filter("atempo=0.90"),
        "speed110": ffmpeg_filter("atempo=1.10"),
        "pitch_up": ffmpeg_pitch(1.059463),
        "pitch_down": ffmpeg_pitch(0.943874),
    }
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown attacks: {', '.join(unknown)}")
    return {name: registry[name] for name in names}
