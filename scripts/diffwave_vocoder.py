#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchaudio


InputFormat = Literal["project-logmel", "diffwave-spec"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert this project's Mel .npy format to the official LMNT DiffWave "
            "conditioning format and synthesize a waveform."
        )
    )
    parser.add_argument("model", help="DiffWave checkpoint file or model directory.")
    parser.add_argument(
        "mel_npy",
        help=(
            "Input .npy. By default this is the raw natural-log Mel produced by "
            "melshield.vocoders.CommandVocoder."
        ),
    )
    parser.add_argument("-o", "--output", required=True, help="Output wav path.")
    parser.add_argument(
        "--input-format",
        choices=["project-logmel", "diffwave-spec"],
        default="project-logmel",
        help="Format of mel_npy. Use diffwave-spec only for files from diffwave.preprocess.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for DiffWave inference.",
    )
    parser.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        default=True,
        help="Use DiffWave's fast inference schedule. This is the default.",
    )
    parser.add_argument(
        "--no-fast",
        dest="fast",
        action="store_false",
        help="Use the full training noise schedule. Much slower.",
    )
    parser.add_argument(
        "--expected-n-mels",
        type=int,
        default=80,
        help="Validate the Mel channel count before inference.",
    )
    parser.add_argument(
        "--debug-save-spec",
        default=None,
        help="Optional path to save the converted DiffWave conditioning .npy.",
    )
    return parser.parse_args()


def load_mel(path: str | Path, expected_n_mels: int) -> np.ndarray:
    mel = np.load(path).astype(np.float32)
    mel = np.squeeze(mel)
    if mel.ndim != 2:
        raise ValueError(f"Expected rank-2 Mel [n_mels, frames], got shape {mel.shape}.")
    if mel.shape[0] != expected_n_mels and mel.shape[1] == expected_n_mels:
        mel = mel.T
    if mel.shape[0] != expected_n_mels:
        raise ValueError(
            f"Expected {expected_n_mels} Mel channels, got shape {mel.shape}. "
            "If this is already a DiffWave spectrogram, pass --input-format diffwave-spec."
        )
    if mel.shape[1] < 1:
        raise ValueError("Mel has no time frames.")
    return np.ascontiguousarray(mel, dtype=np.float32)


def project_logmel_to_diffwave_spec(log_mel: np.ndarray) -> np.ndarray:
    """Match LMNT DiffWave's preprocess.py spectrogram normalization.

    The project stores raw natural-log Mel values before vocoder synthesis. LMNT
    DiffWave preprocesses a Mel magnitude spectrogram as:

        spec = 20 * log10(max(mel, 1e-5)) - 20
        spec = clamp((spec + 100) / 100, 0, 1)

    Since log_mel = ln(max(mel, eps)), conversion is algebraic.
    """

    db_spec = (20.0 / math.log(10.0)) * log_mel - 20.0
    spec = np.clip((db_spec + 100.0) / 100.0, 0.0, 1.0)
    return spec.astype(np.float32)


def as_diffwave_spec(mel: np.ndarray, input_format: InputFormat) -> np.ndarray:
    if input_format == "project-logmel":
        return project_logmel_to_diffwave_spec(mel)
    if input_format == "diffwave-spec":
        return np.clip(mel, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported input format: {input_format}")


def synthesize(
    model: str | Path,
    spectrogram: np.ndarray,
    output: str | Path,
    device: str,
    fast: bool,
) -> None:
    try:
        from diffwave.inference import predict as diffwave_predict
    except ImportError as exc:
        raise SystemExit(
            "Could not import diffwave. Install it in this environment with "
            "`pip install diffwave` or `pip install -e external/diffwave`."
        ) from exc

    tensor = torch.from_numpy(np.ascontiguousarray(spectrogram, dtype=np.float32))
    audio, sample_rate = diffwave_predict(
        tensor,
        model_dir=str(model),
        device=torch.device(device),
        fast_sampling=fast,
    )
    audio = audio.detach().cpu().float().clamp(-1.0, 1.0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), audio, int(sample_rate))


def main() -> None:
    args = parse_args()
    mel = load_mel(args.mel_npy, expected_n_mels=args.expected_n_mels)
    spectrogram = as_diffwave_spec(mel, input_format=args.input_format)
    if args.debug_save_spec:
        debug_path = Path(args.debug_save_spec)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(debug_path, spectrogram)
    synthesize(args.model, spectrogram, args.output, args.device, args.fast)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"diffwave_vocoder failed: {exc}", file=sys.stderr)
        raise
