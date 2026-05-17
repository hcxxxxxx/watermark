from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 20.0
    f_max: Optional[float] = None
    power: float = 1.0
    eps: float = 1e-5

    def resolved_f_max(self) -> float:
        return float(self.sample_rate / 2 if self.f_max is None else self.f_max)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["f_max"] = self.resolved_f_max()
        return data


@dataclass(frozen=True)
class NormalizationStats:
    minimum: float
    maximum: float

    @property
    def scale(self) -> float:
        return max(float(self.maximum - self.minimum), 1e-8)

    def to_dict(self) -> dict:
        return {"minimum": float(self.minimum), "maximum": float(self.maximum)}


@dataclass(frozen=True)
class MelBundle:
    log_mel: np.ndarray
    normalized: np.ndarray
    stats: NormalizationStats


class MelFrontend:
    """STFT log-Mel frontend used by embedding and verification."""

    def __init__(self, config: MelConfig, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.resolved_f_max(),
            n_mels=config.n_mels,
            power=config.power,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        ).to(self.device)

    def waveform_to_logmel(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        waveform = _to_mono(waveform).to(self.device)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.config.sample_rate
            )
        with torch.no_grad():
            mel = self._mel(waveform)
            log_mel = torch.log(torch.clamp(mel, min=self.config.eps))
        return log_mel.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def normalize(
        self, log_mel: np.ndarray, stats: Optional[NormalizationStats] = None
    ) -> Tuple[np.ndarray, NormalizationStats]:
        log_mel = np.asarray(log_mel, dtype=np.float32)
        if stats is None:
            stats = NormalizationStats(float(log_mel.min()), float(log_mel.max()))
        normalized = (log_mel - stats.minimum) / stats.scale
        return np.clip(normalized, 0.0, 1.0).astype(np.float32), stats

    def denormalize(self, normalized: np.ndarray, stats: NormalizationStats) -> np.ndarray:
        normalized = np.asarray(normalized, dtype=np.float32)
        return (normalized * stats.scale + stats.minimum).astype(np.float32)

    def waveform_to_normalized_logmel(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        stats: Optional[NormalizationStats] = None,
    ) -> MelBundle:
        log_mel = self.waveform_to_logmel(waveform, sample_rate)
        normalized, used_stats = self.normalize(log_mel, stats=stats)
        return MelBundle(log_mel=log_mel, normalized=normalized, stats=used_stats)

    def align_to_reference(
        self,
        detected: np.ndarray,
        reference: np.ndarray,
        max_shift: int = 12,
    ) -> np.ndarray:
        return align_mels(detected, reference, max_shift=max_shift)


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def align_mels(detected: np.ndarray, reference: np.ndarray, max_shift: int = 12) -> np.ndarray:
    """Return detected Mel cropped/padded to the reference time axis.

    Compression and resampling can add a small frame drift. We estimate a bounded
    shift from frame-energy curves, then crop or edge-pad to the reference length.
    """

    detected = np.asarray(detected, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if detected.ndim != 2 or reference.ndim != 2:
        raise ValueError("Mel arrays must be rank-2 [n_mels, frames].")
    if detected.shape[0] != reference.shape[0]:
        raise ValueError(
            f"Mel band mismatch: detected={detected.shape[0]}, reference={reference.shape[0]}"
        )

    ref_frames = reference.shape[1]
    det_frames = detected.shape[1]
    if det_frames == ref_frames:
        return detected

    best_shift = 0
    best_score = -np.inf
    ref_energy = reference.mean(axis=0)
    det_energy = detected.mean(axis=0)
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            det_start = shift
            ref_start = 0
        else:
            det_start = 0
            ref_start = -shift
        overlap = min(det_frames - det_start, ref_frames - ref_start)
        if overlap < max(8, ref_frames // 5):
            continue
        a = det_energy[det_start : det_start + overlap]
        b = ref_energy[ref_start : ref_start + overlap]
        score = _pearson(a, b)
        if score > best_score:
            best_score = score
            best_shift = shift

    if best_shift > 0:
        aligned = detected[:, best_shift:]
    elif best_shift < 0:
        pad = np.repeat(detected[:, :1], -best_shift, axis=1)
        aligned = np.concatenate([pad, detected], axis=1)
    else:
        aligned = detected

    if aligned.shape[1] > ref_frames:
        aligned = aligned[:, :ref_frames]
    elif aligned.shape[1] < ref_frames:
        pad = np.repeat(aligned[:, -1:], ref_frames - aligned.shape[1], axis=1)
        aligned = np.concatenate([aligned, pad], axis=1)
    return aligned.astype(np.float32)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - float(np.mean(a))
    b = b.astype(np.float64) - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)
