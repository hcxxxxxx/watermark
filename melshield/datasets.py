from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torchaudio


@dataclass(frozen=True)
class AudioItem:
    utterance_id: str
    wav_path: Path
    text: str = ""


def iter_ljspeech(root: str | Path, limit: Optional[int] = None) -> Iterator[AudioItem]:
    root = Path(root)
    metadata = root / "metadata.csv"
    wav_dir = root / "wavs"
    if not metadata.exists():
        raise FileNotFoundError(
            f"Missing {metadata}. Expected LJSpeech at data/LJSpeech-1.1."
        )
    if not wav_dir.exists():
        raise FileNotFoundError(f"Missing wav directory: {wav_dir}")

    count = 0
    with metadata.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            if not row:
                continue
            utterance_id = row[0]
            wav_path = wav_dir / f"{utterance_id}.wav"
            if not wav_path.exists():
                continue
            text = row[1] if len(row) > 1 else ""
            yield AudioItem(utterance_id=utterance_id, wav_path=wav_path, text=text)
            count += 1
            if limit is not None and count >= limit:
                break


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, int(sample_rate)


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.detach().cpu().float().clamp(-1.0, 1.0)
    torchaudio.save(str(path), waveform, sample_rate)
