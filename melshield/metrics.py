from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchaudio


@dataclass(frozen=True)
class MetricResult:
    pesq: Optional[float]
    stoi: Optional[float]


def compare_audio(
    reference: torch.Tensor,
    reference_sample_rate: int,
    degraded: torch.Tensor,
    degraded_sample_rate: int,
    target_sample_rate: int = 16000,
) -> MetricResult:
    ref = _prep(reference, reference_sample_rate, target_sample_rate)
    deg = _prep(degraded, degraded_sample_rate, target_sample_rate)
    length = min(ref.numel(), deg.numel())
    ref_np = ref[:length].detach().cpu().numpy().astype(np.float32)
    deg_np = deg[:length].detach().cpu().numpy().astype(np.float32)

    pesq_value = None
    stoi_value = None
    try:
        from pesq import pesq

        pesq_value = float(pesq(target_sample_rate, ref_np, deg_np, "wb"))
    except Exception:
        pesq_value = None
    try:
        from pystoi import stoi

        stoi_value = float(stoi(ref_np, deg_np, target_sample_rate, extended=False))
    except Exception:
        stoi_value = None
    return MetricResult(pesq=pesq_value, stoi=stoi_value)


def _prep(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.detach().cpu().float()
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), sample_rate, target_sample_rate
        ).squeeze(0)
    return waveform
