#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchaudio
import yaml

from melshield.attacks import build_attacks
from melshield.datasets import iter_ljspeech, load_audio, save_audio
from melshield.mel import MelConfig, MelFrontend
from melshield.metrics import compare_audio
from melshield.vocoders import build_vocoder
from melshield.watermark import deterministic_bits


class WaveformWatermarker(Protocol):
    payload_bits: int
    sample_rate: int

    def embed(self, audio: torch.Tensor, payload: np.ndarray) -> torch.Tensor:
        ...

    def detect(self, audio: torch.Tensor, payload: np.ndarray) -> dict[str, Any]:
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate waveform-domain audio watermark baselines on LJSpeech vocoder outputs."
    )
    parser.add_argument("--config", default="configs/relmel_hifigan.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", required=True, choices=["audioseal", "wavmark"])
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attacks", nargs="+", default=["none", "noise20", "noise10", "noise5"])
    parser.add_argument("--vocoder", default=None, choices=["hifigan", "command", "griffinlim"])
    parser.add_argument("--vocoder-checkpoint", default=None)
    parser.add_argument("--vocoder-config", default=None)
    parser.add_argument("--vocoder-command", default=None)
    parser.add_argument(
        "--baseline-sample-rate",
        type=int,
        default=16000,
        help="Sample rate used by the waveform baseline. WavMark requires 16 kHz.",
    )
    parser.add_argument("--payload-bits", type=int, default=16)
    parser.add_argument("--bit-acc-threshold", type=float, default=0.75)
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--message-threshold", type=float, default=0.5)
    parser.add_argument("--audioseal-generator", default="audioseal_wm_16bits")
    parser.add_argument("--audioseal-detector", default="audioseal_detector_16bits")
    parser.add_argument("--save-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_audio:
        (output_dir / "audio").mkdir(exist_ok=True)

    data_root = Path(args.data_root or cfg["paths"]["data_root"])
    mel_config = MelConfig(**cfg["mel"])
    frontend = MelFrontend(mel_config, device=args.device)
    vocoder_name = args.vocoder or cfg["vocoder"]["name"]
    if vocoder_name == "mel":
        raise ValueError("Waveform baselines require a waveform vocoder, not --vocoder mel.")
    vocoder = build_vocoder(
        vocoder_name,
        mel_config=mel_config,
        device=args.device,
        checkpoint=args.vocoder_checkpoint or cfg["vocoder"].get("checkpoint"),
        config=args.vocoder_config or cfg["vocoder"].get("config"),
        command_template=args.vocoder_command or cfg["vocoder"].get("command"),
        sample_rate=mel_config.sample_rate,
    )
    watermarker = build_watermarker(args)
    attack_fns = build_attacks(args.attacks)
    rows: list[dict[str, Any]] = []

    for item in iter_ljspeech(data_root, limit=args.limit, sample_mode=args.sample_mode, seed=args.seed):
        waveform, sample_rate = load_audio(item.wav_path)
        bundle = frontend.waveform_to_normalized_logmel(waveform, sample_rate)
        clean_log_mel = torch.from_numpy(bundle.log_mel)
        benchmark_audio = vocoder.synthesize(clean_log_mel).detach().cpu().float()
        vocoder_sr = int(vocoder.sample_rate)

        payload = deterministic_bits(
            key=f"{args.method}|payload",
            identifier=item.utterance_id,
            length=watermarker.payload_bits,
        )
        host_audio = resample_mono(benchmark_audio, vocoder_sr, watermarker.sample_rate)
        watermarked_audio = watermarker.embed(host_audio, payload)

        if args.save_audio:
            save_audio(output_dir / "audio" / f"{item.utterance_id}_bm.wav", benchmark_audio, vocoder_sr)
            save_audio(
                output_dir / "audio" / f"{item.utterance_id}_{args.method}.wav",
                watermarked_audio,
                watermarker.sample_rate,
            )

        for attack_name, attack_fn in attack_fns.items():
            attacked = attack_fn(watermarked_audio, watermarker.sample_rate)
            detect_audio = resample_mono(attacked.waveform, attacked.sample_rate, watermarker.sample_rate)
            detection = watermarker.detect(detect_audio, payload)
            gt_metrics = compare_audio(waveform, sample_rate, attacked.waveform, attacked.sample_rate)
            bm_metrics = compare_audio(
                benchmark_audio,
                vocoder_sr,
                attacked.waveform,
                attacked.sample_rate,
            )
            rows.append(
                {
                    "utterance_id": item.utterance_id,
                    "method": args.method,
                    "attack": attack_name,
                    "payload_bits": watermarker.payload_bits,
                    "bit_acc": detection["bit_acc"],
                    "detected": detection["detected"],
                    "verified": detection["verified"],
                    "detection_score": detection.get("detection_score", ""),
                    "confidence": detection.get("confidence", ""),
                    "pesq_gt": fmt(gt_metrics.pesq),
                    "stoi_gt": fmt(gt_metrics.stoi),
                    "pesq_bm": fmt(bm_metrics.pesq),
                    "stoi_bm": fmt(bm_metrics.stoi),
                    "snr_bm": fmt(snr(benchmark_audio, vocoder_sr, attacked.waveform, attacked.sample_rate)),
                }
            )
        print(f"processed {item.utterance_id}")

    write_outputs(output_dir, rows, args, cfg)
    print(f"wrote waveform baseline results to {output_dir}")


class AudioSealWatermarker:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from audioseal import AudioSeal
        except ImportError as exc:
            raise SystemExit("Missing dependency: install AudioSeal with `pip install audioseal`.") from exc

        self.device = torch.device(args.device)
        self.sample_rate = int(args.baseline_sample_rate)
        self.payload_bits = int(args.payload_bits)
        self.bit_acc_threshold = float(args.bit_acc_threshold)
        self.detection_threshold = float(args.detection_threshold)
        self.message_threshold = float(args.message_threshold)
        self.generator = AudioSeal.load_generator(args.audioseal_generator, device=str(self.device))
        self.detector = AudioSeal.load_detector(args.audioseal_detector, device=str(self.device))
        self.generator.eval()
        self.detector.eval()
        nbits = getattr(getattr(self.generator, "msg_processor", None), "nbits", self.payload_bits)
        if int(nbits) != self.payload_bits:
            raise ValueError(f"AudioSeal model expects {nbits} message bits, got {self.payload_bits}.")

    @torch.no_grad()
    def embed(self, audio: torch.Tensor, payload: np.ndarray) -> torch.Tensor:
        x = ensure_bct(audio).to(self.device)
        message = torch.as_tensor(payload, device=self.device, dtype=torch.int64).view(1, -1)
        watermark = self.generator.get_watermark(x, message=message)
        return (x + watermark).clamp(-1.0, 1.0).detach().cpu().squeeze(0)

    @torch.no_grad()
    def detect(self, audio: torch.Tensor, payload: np.ndarray) -> dict[str, Any]:
        x = ensure_bct(audio).to(self.device)
        prob, decoded = self.detector.detect_watermark(
            x,
            message_threshold=self.message_threshold,
            detection_threshold=self.detection_threshold,
        )
        prob_value = float(prob.detach().cpu().view(-1)[0])
        decoded_np = decoded.detach().cpu().view(-1).numpy().astype(np.uint8)[: payload.size]
        acc = bit_accuracy(payload, decoded_np)
        detected = prob_value >= self.detection_threshold
        return {
            "bit_acc": acc,
            "detected": bool(detected),
            "verified": bool(detected and acc >= self.bit_acc_threshold),
            "detection_score": prob_value,
            "confidence": prob_value,
        }


class WavMarkWatermarker:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import wavmark
        except ImportError as exc:
            raise SystemExit("Missing dependency: install WavMark with `pip install wavmark`.") from exc

        if int(args.baseline_sample_rate) != 16000:
            raise ValueError("WavMark expects 16 kHz audio. Use --baseline-sample-rate 16000.")
        self.wavmark = wavmark
        self.device = torch.device(args.device)
        self.sample_rate = 16000
        self.payload_bits = int(args.payload_bits)
        self.bit_acc_threshold = float(args.bit_acc_threshold)
        self.model = wavmark.load_model().to(self.device)
        self.model.eval()
        if self.payload_bits != 16:
            raise ValueError(
                "This script uses WavMark's public high-level API, which exposes a 16-bit custom payload. "
                "Use --payload-bits 16."
            )

    def embed(self, audio: torch.Tensor, payload: np.ndarray) -> torch.Tensor:
        signal = tensor_to_numpy(audio)
        watermarked, _ = self.wavmark.encode_watermark(
            self.model,
            signal,
            payload.astype(np.int64),
            show_progress=False,
        )
        return numpy_to_tensor(watermarked)

    def detect(self, audio: torch.Tensor, payload: np.ndarray) -> dict[str, Any]:
        signal = tensor_to_numpy(audio)
        decoded, info = self.wavmark.decode_watermark(self.model, signal, show_progress=False)
        decoded_np = np.asarray(decoded, dtype=np.uint8).reshape(-1)[: payload.size]
        if decoded_np.size != payload.size:
            decoded_np = np.zeros_like(payload)
        acc = bit_accuracy(payload, decoded_np)
        confidence = wavmark_confidence(info)
        return {
            "bit_acc": acc,
            "detected": bool(acc >= self.bit_acc_threshold),
            "verified": bool(acc >= self.bit_acc_threshold),
            "detection_score": confidence if confidence is not None else "",
            "confidence": confidence if confidence is not None else "",
        }


def build_watermarker(args: argparse.Namespace) -> WaveformWatermarker:
    if args.method == "audioseal":
        return AudioSealWatermarker(args)
    if args.method == "wavmark":
        return WavMarkWatermarker(args)
    raise ValueError(f"Unsupported method: {args.method}")


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    if rows:
        with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["attack"]].append(row)

    def mean_key(attack: str, key: str) -> float | None:
        values = [row[key] for row in grouped[attack] if row.get(key) not in ("", None)]
        if not values:
            return None
        return float(np.mean([float(v) for v in values]))

    summary = {
        "method": args.method,
        "payload_bits": args.payload_bits,
        "baseline_sample_rate": args.baseline_sample_rate,
        "bit_acc_threshold": args.bit_acc_threshold,
        "detection_threshold": args.detection_threshold,
        "mean_bit_acc": {attack: mean_key(attack, "bit_acc") for attack in sorted(grouped)},
        "detection_rate": {
            attack: float(np.mean([bool(row["detected"]) for row in attack_rows]))
            for attack, attack_rows in sorted(grouped.items())
        },
        "verification_rate": {
            attack: float(np.mean([bool(row["verified"]) for row in attack_rows]))
            for attack, attack_rows in sorted(grouped.items())
        },
        "mean_detection_score": {
            attack: mean_key(attack, "detection_score") for attack in sorted(grouped)
        },
        "mean_pesq_bm": {attack: mean_key(attack, "pesq_bm") for attack in sorted(grouped)},
        "mean_stoi_bm": {attack: mean_key(attack, "stoi_bm") for attack in sorted(grouped)},
        "mean_snr_bm": {attack: mean_key(attack, "snr_bm") for attack in sorted(grouped)},
        "num_rows": len(rows),
        "args": vars(args),
        "run_config": cfg,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def resample_mono(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.detach().cpu().float()
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform.clamp(-1.0, 1.0)


def ensure_bct(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        audio = audio.view(1, 1, -1)
    elif audio.ndim == 2:
        audio = audio.unsqueeze(0)
    if audio.ndim != 3:
        raise ValueError(f"Expected audio shape [batch, channels, time], got {tuple(audio.shape)}")
    if audio.shape[1] > 1:
        audio = audio.mean(dim=1, keepdim=True)
    return audio.float()


def tensor_to_numpy(audio: torch.Tensor) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    return audio.detach().cpu().float().numpy().astype(np.float32).reshape(-1)


def numpy_to_tensor(audio: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(audio, dtype=np.float32)).view(1, -1).clamp(-1.0, 1.0)


def bit_accuracy(expected: np.ndarray, decoded: np.ndarray) -> float:
    expected = np.asarray(expected, dtype=np.uint8).reshape(-1)
    decoded = np.asarray(decoded, dtype=np.uint8).reshape(-1)
    if expected.size == 0 or decoded.size != expected.size:
        return 0.0
    return float(np.mean(expected == decoded))


def wavmark_confidence(info: Any) -> float | None:
    if isinstance(info, dict):
        for key in ("confidence", "score", "sim", "similarity"):
            if key in info:
                try:
                    return float(info[key])
                except Exception:
                    return None
    try:
        value = float(info)
    except Exception:
        return None
    return value


def snr(reference: torch.Tensor, reference_sr: int, degraded: torch.Tensor, degraded_sr: int) -> float | None:
    ref = resample_mono(reference, reference_sr, 16000).view(-1)
    deg = resample_mono(degraded, degraded_sr, 16000).view(-1)
    length = min(ref.numel(), deg.numel())
    if length == 0:
        return None
    ref = ref[:length]
    deg = deg[:length]
    noise = ref - deg
    signal_power = float(ref.pow(2).mean())
    noise_power = float(noise.pow(2).mean())
    if signal_power <= 1e-12 or noise_power <= 1e-12:
        return None
    return float(10.0 * np.log10(signal_power / noise_power))


def fmt(value: float | None) -> str:
    return "" if value is None else f"{float(value):.8f}"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
