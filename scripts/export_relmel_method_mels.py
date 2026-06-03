#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from melshield.attacks import build_attacks
from melshield.coding import RepetitionCode
from melshield.datasets import iter_ljspeech, load_audio, save_audio
from melshield.mel import MelConfig, MelFrontend, align_mels
from melshield.relmel import RelMelConfig, RelMelMark
from melshield.vocoders import build_vocoder
from melshield.watermark import deterministic_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export clean, RAWMER-watermarked, and noise-attacked log-Mel "
            "spectrograms for paper method figures."
        )
    )
    parser.add_argument("--config", default="configs/relmel_hifigan.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="figures/relmel_method_mels")
    parser.add_argument("--sample-mode", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--utterance-id", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attack-seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--per-image-scale", action="store_true")

    parser.add_argument("--vocoder", default=None, choices=["griffinlim", "hifigan", "command"])
    parser.add_argument("--vocoder-checkpoint", default=None)
    parser.add_argument("--vocoder-config", default=None)
    parser.add_argument("--vocoder-command", default=None)

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--band", default=None)
    parser.add_argument("--payload-bits", type=int, default=None)
    parser.add_argument("--info-bits", type=int, default=None)
    parser.add_argument("--ecc-repeat", type=int, default=1)
    parser.add_argument("--no-ecc-interleave", action="store_true")
    parser.add_argument("--block-frames", type=int, default=None)
    parser.add_argument("--block-stride", type=int, default=None)
    parser.add_argument("--min-block-frames", type=int, default=None)
    parser.add_argument("--bits-per-block", type=int, default=None)
    parser.add_argument("--pair-bins", type=int, default=None)
    parser.add_argument("--pair-candidates", type=int, default=None)
    parser.add_argument("--detector-mode", choices=["plain", "boundary"], default=None)
    parser.add_argument("--mask-floor", type=float, default=None)
    parser.add_argument("--energy-gamma", type=float, default=None)
    parser.add_argument("--boundary-margin", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.attack_seed)
    np.random.seed(args.attack_seed)

    cfg = load_yaml(args.config)
    data_root = Path(args.data_root or cfg["paths"]["data_root"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_audio:
        (output_dir / "audio").mkdir(exist_ok=True)

    mel_config = MelConfig(**cfg["mel"])
    relmel_config = build_relmel_config(cfg, args)
    codec = build_codec(args)
    info_bits = info_payload_bits(relmel_config, codec)

    frontend = MelFrontend(mel_config, device=args.device)
    relmel = RelMelMark(relmel_config)
    vocoder_name = args.vocoder or cfg["vocoder"]["name"]
    vocoder = build_vocoder(
        vocoder_name,
        mel_config=mel_config,
        device=args.device,
        checkpoint=args.vocoder_checkpoint or cfg["vocoder"].get("checkpoint"),
        config=args.vocoder_config or cfg["vocoder"].get("config"),
        command_template=args.vocoder_command or cfg["vocoder"].get("command"),
        sample_rate=mel_config.sample_rate,
    )
    attack_fns = build_attacks(["noise20", "noise10", "noise5"])

    item = select_ljspeech_item(
        data_root=data_root,
        utterance_id=args.utterance_id,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    waveform, sample_rate = load_audio(item.wav_path)
    bundle = frontend.waveform_to_normalized_logmel(waveform, sample_rate)

    payload = payload_from_id(relmel_config, item.utterance_id, info_bits)
    message = codec.encode(payload)
    watermarked_norm, reference = relmel.embed(
        clean_mel=bundle.normalized,
        message=message,
        utterance_id=item.utterance_id,
        norm_stats=bundle.stats,
        mel_config=mel_config,
    )

    clean_log_mel = bundle.log_mel
    watermarked_log_mel = frontend.denormalize(watermarked_norm, bundle.stats)

    clean_audio = vocoder.synthesize(torch.from_numpy(clean_log_mel))
    watermarked_audio = vocoder.synthesize(torch.from_numpy(watermarked_log_mel))
    vocoder_sr = int(vocoder.sample_rate)

    if args.save_audio:
        save_audio(output_dir / "audio" / "A_clean_vocoder.wav", clean_audio, vocoder_sr)
        save_audio(output_dir / "audio" / "B_rawmer_watermarked.wav", watermarked_audio, vocoder_sr)

    spectra: list[tuple[str, str, np.ndarray]] = [
        ("A", "clean", clean_log_mel),
        ("B", "rawmer_watermarked", watermarked_log_mel),
    ]
    for label, attack_name in [("C", "noise20"), ("D", "noise10"), ("E", "noise5")]:
        attacked = attack_fns[attack_name](watermarked_audio, vocoder_sr)
        if args.save_audio:
            save_audio(output_dir / "audio" / f"{label}_{attack_name}.wav", attacked.waveform, attacked.sample_rate)
        attacked_bundle = frontend.waveform_to_normalized_logmel(
            attacked.waveform,
            attacked.sample_rate,
            stats=reference.norm_stats,
        )
        attacked_log_mel = align_mels(
            attacked_bundle.log_mel,
            clean_log_mel,
            max_shift=relmel_config.align_max_shift,
        )
        spectra.append((label, attack_name, attacked_log_mel))

    arrays = [array for _, _, array in spectra]
    if args.per_image_scale:
        vmin = vmax = None
    else:
        stacked = np.concatenate([array.reshape(-1) for array in arrays])
        vmin = float(np.percentile(stacked, 1.0))
        vmax = float(np.percentile(stacked, 99.0))

    manifest: dict[str, Any] = {
        "utterance_id": item.utterance_id,
        "wav_path": str(item.wav_path),
        "output_dir": str(output_dir),
        "config": args.config,
        "vocoder": vocoder_name,
        "attack_seed": args.attack_seed,
        "color_scale": {
            "mode": "per_image" if args.per_image_scale else "shared_percentile_1_99",
            "vmin": vmin,
            "vmax": vmax,
        },
        "mel_config": mel_config.to_dict(),
        "relmel_config": relmel_config.to_dict(),
        "files": {},
    }

    for label, name, array in spectra:
        stem = f"{label}_{name}_logmel"
        png_path = output_dir / f"{stem}.png"
        npy_path = output_dir / f"{stem}.npy"
        np.save(npy_path, array.astype(np.float32))
        plot_single_logmel(array, png_path, dpi=args.dpi, vmin=vmin, vmax=vmax)
        manifest["files"][label] = {
            "name": name,
            "png": str(png_path),
            "npy": str(npy_path),
            "shape": list(array.shape),
        }

    panel_path = output_dir / "method_mels_A_to_E_panel.png"
    plot_panel(spectra, panel_path, dpi=args.dpi, vmin=vmin, vmax=vmax)
    manifest["panel_png"] = str(panel_path)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"selected {item.utterance_id}")
    print(f"wrote method Mel figures to {output_dir}")


def select_ljspeech_item(data_root: Path, utterance_id: str | None, sample_mode: str, seed: int):
    if utterance_id is None:
        return next(iter_ljspeech(data_root, limit=1, sample_mode=sample_mode, seed=seed))
    for item in iter_ljspeech(data_root, sample_mode="first", seed=seed):
        if item.utterance_id == utterance_id:
            return item
    raise ValueError(f"Could not find utterance_id={utterance_id!r} in {data_root}.")


def plot_single_logmel(
    array: np.ndarray,
    path: Path,
    dpi: int,
    vmin: float | None,
    vmax: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 2.4), dpi=dpi)
    ax.imshow(array, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_panel(
    spectra: list[tuple[str, str, np.ndarray]],
    path: Path,
    dpi: int,
    vmin: float | None,
    vmax: float | None,
) -> None:
    fig, axes = plt.subplots(len(spectra), 1, figsize=(5.6, 9.0), dpi=dpi)
    for ax, (label, name, array) in zip(axes, spectra):
        ax.imshow(array, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.text(
            0.015,
            0.90,
            f"{label}: {name}",
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            weight="bold",
            ha="left",
            va="top",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2},
        )
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.02)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_relmel_config(cfg: dict[str, Any], args: argparse.Namespace) -> RelMelConfig:
    data = dict(cfg["relmel"])
    data["band"] = tuple(data["band"])
    ecc_repeat = int(getattr(args, "ecc_repeat", 1) or 1)
    info_bits = getattr(args, "info_bits", None)
    overrides = {
        "alpha": args.alpha,
        "threshold": args.threshold,
        "band": parse_band(getattr(args, "band", None)),
        "payload_bits": info_bits if info_bits is not None else args.payload_bits,
        "block_frames": args.block_frames,
        "block_stride": args.block_stride,
        "min_block_frames": args.min_block_frames,
        "bits_per_block": args.bits_per_block,
        "pair_bins": args.pair_bins,
        "pair_candidates": args.pair_candidates,
        "detector_mode": args.detector_mode,
        "mask_floor": args.mask_floor,
        "energy_gamma": args.energy_gamma,
        "boundary_margin": args.boundary_margin,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    if ecc_repeat > 1:
        data["payload_bits"] = int(data["payload_bits"]) * ecc_repeat
    return RelMelConfig(**data)


def parse_band(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    left, right = value.split(":", maxsplit=1)
    return (int(left), int(right))


def build_codec(args: argparse.Namespace) -> RepetitionCode:
    repeat = int(getattr(args, "ecc_repeat", 1) or 1)
    interleave = not bool(getattr(args, "no_ecc_interleave", False))
    return RepetitionCode(repeat=repeat, interleave=interleave)


def info_payload_bits(relmel_config: RelMelConfig, codec: RepetitionCode) -> int:
    if relmel_config.payload_bits % codec.repeat != 0:
        raise ValueError(
            f"Codeword length {relmel_config.payload_bits} is not divisible by "
            f"ecc repeat={codec.repeat}."
        )
    return relmel_config.payload_bits // codec.repeat


def payload_from_id(relmel_config: RelMelConfig, utterance_id: str, info_bits: int) -> np.ndarray:
    return deterministic_bits(
        key=f"{relmel_config.key}|payload",
        identifier=utterance_id,
        length=info_bits,
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
