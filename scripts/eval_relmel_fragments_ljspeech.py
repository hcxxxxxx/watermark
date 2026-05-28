#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_relmel_ljspeech import (
    build_codec,
    build_relmel_config,
    info_payload_bits,
    load_yaml,
    payload_from_id,
    result_row,
)
from melshield.attacks import AttackResult
from melshield.coding import bit_accuracy
from melshield.datasets import iter_ljspeech, load_audio
from melshield.mel import MelConfig, MelFrontend, align_mels
from melshield.metrics import compare_audio
from melshield.relmel import RelMelReference, RelMelResult, RelMelMark
from melshield.vocoders import build_vocoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RelMel verification on cropped/local audio fragments."
    )
    parser.add_argument("--config", default="configs/relmel_hifigan.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--fragments",
        nargs="+",
        default=["full", "start25", "middle25", "end25", "start50", "middle50", "end50", "middle75"],
    )
    parser.add_argument("--vocoder", default=None, choices=["hifigan", "command"])
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
    cfg = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root or cfg["paths"]["data_root"])
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

    rows: list[dict[str, Any]] = []
    for item in iter_ljspeech(data_root, limit=args.limit, sample_mode=args.sample_mode, seed=args.seed):
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
        clean_log_mel = torch.from_numpy(bundle.log_mel)
        wm_log_mel = torch.from_numpy(frontend.denormalize(watermarked_norm, bundle.stats))
        benchmark_audio = vocoder.synthesize(clean_log_mel)
        watermarked_audio = vocoder.synthesize(wm_log_mel)
        vocoder_sr = int(vocoder.sample_rate)

        for fragment in args.fragments:
            start_frame, end_frame = fragment_frames(fragment, reference.clean_mel.shape[1])
            cropped_wm = crop_audio_by_frames(watermarked_audio, vocoder_sr, mel_config, start_frame, end_frame)
            cropped_bm = crop_audio_by_frames(benchmark_audio, vocoder_sr, mel_config, start_frame, end_frame)
            detected = frontend.waveform_to_normalized_logmel(
                cropped_wm.waveform,
                cropped_wm.sample_rate,
                stats=reference.norm_stats,
            )
            extracted = extract_known_fragment(
                relmel,
                detected.normalized,
                reference,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            row = result_row(
                item.utterance_id,
                fragment,
                extracted,
                relmel_config,
                codec,
                payload,
            )
            bm_metrics = compare_audio(
                cropped_bm.waveform,
                cropped_bm.sample_rate,
                cropped_wm.waveform,
                cropped_wm.sample_rate,
            )
            row.update(
                {
                    "fragment": fragment,
                    "fragment_start_frame": start_frame,
                    "fragment_end_frame": end_frame,
                    "fragment_ratio": (end_frame - start_frame) / float(reference.clean_mel.shape[1]),
                    "pesq_bm": _fmt(bm_metrics.pesq),
                    "stoi_bm": _fmt(bm_metrics.stoi),
                }
            )
            rows.append(row)
        print(f"processed {item.utterance_id}")

    write_outputs(output_dir, rows, cfg, mel_config, relmel_config, args)
    print(f"wrote fragment results to {output_dir}")


def fragment_frames(fragment: str, total_frames: int) -> tuple[int, int]:
    if fragment == "full":
        return 0, total_frames
    if fragment.startswith("start"):
        ratio = int(fragment.replace("start", "")) / 100.0
        return 0, max(1, int(round(total_frames * ratio)))
    if fragment.startswith("end"):
        ratio = int(fragment.replace("end", "")) / 100.0
        length = max(1, int(round(total_frames * ratio)))
        return max(0, total_frames - length), total_frames
    if fragment.startswith("middle"):
        ratio = int(fragment.replace("middle", "")) / 100.0
        length = max(1, int(round(total_frames * ratio)))
        start = max(0, (total_frames - length) // 2)
        return start, min(total_frames, start + length)
    raise ValueError(f"Unknown fragment spec: {fragment}")


def crop_audio_by_frames(
    audio: torch.Tensor,
    sample_rate: int,
    mel_config: MelConfig,
    start_frame: int,
    end_frame: int,
) -> AttackResult:
    start = max(0, int(round(start_frame * mel_config.hop_length * sample_rate / mel_config.sample_rate)))
    end = max(start + 1, int(round(end_frame * mel_config.hop_length * sample_rate / mel_config.sample_rate)))
    end = min(audio.shape[-1], end)
    return AttackResult(audio[:, start:end].clone(), sample_rate)


def extract_known_fragment(
    relmel: RelMelMark,
    detected_fragment: np.ndarray,
    reference: RelMelReference,
    start_frame: int,
    end_frame: int,
) -> RelMelResult:
    cfg = reference.relmel_config
    clean_mel = reference.clean_mel.astype(np.float32)
    ref_window = clean_mel[:, start_frame:end_frame]
    detected = align_mels(detected_fragment, ref_window, max_shift=cfg.align_max_shift)
    c_min, c_max = cfg.band
    clean_band = clean_mel[c_min:c_max, :]
    detected_band = detected[c_min:c_max, :]
    clean_window = ref_window[c_min:c_max, :]
    residual = detected_band - clean_window
    residual = residual - float(residual.mean())

    scores = np.zeros(reference.message.size, dtype=np.float32)
    votes = np.zeros(reference.message.size, dtype=np.float32)
    for block_idx, block_start, block_end in relmel._blocks(clean_band.shape[1], cfg=cfg):
        ov_start = max(block_start, start_frame)
        ov_end = min(block_end, end_frame)
        if ov_end - ov_start < 4:
            continue
        local_start = ov_start - start_frame
        local_end = ov_end - start_frame
        block_residual = residual[:, local_start:local_end]
        clean_full_block = clean_band[:, block_start:block_end]
        clean_overlap = clean_band[:, ov_start:ov_end]
        active_bits = relmel._active_bits(block_idx, reference.message.size, cfg=cfg)
        active_scale = 1.0 / float(np.sqrt(max(len(active_bits), 1)))
        overlap_weight = np.sqrt((ov_end - ov_start) / float(max(block_end - block_start, 1)))
        block_weight = relmel._block_weight(clean_full_block, cfg=cfg)
        for bit_idx in active_bits:
            pair = relmel._pair_vector(
                bit_idx,
                block_idx,
                clean_band.shape[0],
                reference.utterance_id,
                cfg=cfg,
                block=clean_full_block,
            )
            raw_score = relmel._block_score(block_residual, pair, clean_overlap, cfg)
            scores[bit_idx] += raw_score * block_weight * active_scale * overlap_weight
            votes[bit_idx] += overlap_weight

    scores = (scores / np.sqrt(np.maximum(votes, 1.0e-8))).astype(np.float32)
    decoded = (scores >= 0.0).astype(np.uint8)
    accuracy = bit_accuracy(reference.message, decoded)
    return RelMelResult(
        decoded=decoded,
        scores=scores,
        votes=votes,
        bit_accuracy=accuracy,
        verified=bool(accuracy >= cfg.threshold),
    )


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    mel_config: MelConfig,
    relmel_config: Any,
    args: argparse.Namespace,
) -> None:
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["fragment"], []).append(row)

    def mean(rows_: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows_ if row.get(key) not in ("", None)]
        return None if not values else float(np.mean(values))

    summary_rows = {
        fragment: {
            "fragment_ratio": mean(group_rows, "fragment_ratio"),
            "mean_bit_acc": mean(group_rows, "bit_acc"),
            "verification_rate": float(np.mean([bool(row["verified"]) for row in group_rows])),
            "mean_confidence": mean(group_rows, "confidence"),
            "mean_votes": mean(group_rows, "mean_votes"),
            "mean_pesq_bm": mean(group_rows, "pesq_bm"),
            "mean_stoi_bm": mean(group_rows, "stoi_bm"),
        }
        for fragment, group_rows in sorted(grouped.items())
    }

    payload = {
        "args": vars(args),
        "num_rows": len(rows),
        "fragments": summary_rows,
        "mel_config": mel_config.to_dict(),
        "relmel_config": relmel_config.to_dict(),
        "run_config": cfg,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
