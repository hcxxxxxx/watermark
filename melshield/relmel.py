from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .mel import MelConfig, NormalizationStats, align_mels
from .watermark import bit_acc, deterministic_bits


@dataclass(frozen=True)
class RelMelConfig:
    payload_bits: int = 32
    key: str = "relmel-demo-key"
    alpha: float = 0.06
    band: tuple[int, int] = (20, 56)
    threshold: float = 0.75
    block_frames: int = 32
    min_block_frames: int = 16
    bits_per_block: int = 16
    pair_bins: int = 3
    mask_floor: float = 0.15
    energy_gamma: float = 0.5
    boundary_margin: float = 0.03
    align_max_shift: int = 12

    def to_dict(self) -> dict:
        data = asdict(self)
        data["band"] = list(self.band)
        return data


@dataclass(frozen=True)
class RelMelReference:
    utterance_id: str
    clean_mel: np.ndarray
    message: np.ndarray
    norm_stats: NormalizationStats
    mel_config: MelConfig
    relmel_config: RelMelConfig

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "utterance_id": self.utterance_id,
            "norm_stats": self.norm_stats.to_dict(),
            "mel_config": self.mel_config.to_dict(),
            "relmel_config": self.relmel_config.to_dict(),
        }
        np.savez_compressed(
            path,
            clean_mel=self.clean_mel.astype(np.float32),
            message=self.message.astype(np.uint8),
            metadata=json.dumps(metadata, sort_keys=True),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RelMelReference":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            mel_cfg = MelConfig(**metadata["mel_config"])
            cfg_data = metadata["relmel_config"]
            cfg_data["band"] = tuple(cfg_data["band"])
            relmel_cfg = RelMelConfig(**cfg_data)
            return cls(
                utterance_id=metadata["utterance_id"],
                clean_mel=data["clean_mel"].astype(np.float32),
                message=data["message"].astype(np.uint8),
                norm_stats=NormalizationStats(**metadata["norm_stats"]),
                mel_config=mel_cfg,
                relmel_config=relmel_cfg,
            )


@dataclass(frozen=True)
class RelMelResult:
    decoded: np.ndarray
    scores: np.ndarray
    votes: np.ndarray
    bit_accuracy: Optional[float]
    verified: Optional[bool]


class RelMelMark:
    """Block-wise relative-energy watermarking in normalized Mel space.

    Each payload bit is repeated across keyed time blocks. Within a block, the
    bit controls the sign of a relative-energy pair: one keyed Mel-bin group is
    raised and a second group is lowered by the same amount. The detector uses
    the stored reference Mel to measure the corresponding relative-energy drift.
    """

    def __init__(self, config: RelMelConfig) -> None:
        self.config = config

    def message_from_id(self, utterance_id: str, namespace: str = "payload") -> np.ndarray:
        return deterministic_bits(
            key=f"{self.config.key}|{namespace}",
            identifier=utterance_id,
            length=self.config.payload_bits,
        )

    def embed(
        self,
        clean_mel: np.ndarray,
        message: Sequence[int] | np.ndarray,
        utterance_id: str,
        norm_stats: NormalizationStats,
        mel_config: MelConfig,
    ) -> tuple[np.ndarray, RelMelReference]:
        clean_mel = _as_mel(clean_mel)
        message_arr = _as_bits(message)
        if message_arr.size != self.config.payload_bits:
            raise ValueError(
                f"Expected {self.config.payload_bits} bits, got {message_arr.size}"
            )

        c_min, c_max = self._checked_band(clean_mel.shape[0])
        x_wm = clean_mel.copy()
        x_band = clean_mel[c_min:c_max, :]
        blocks = self._blocks(x_band.shape[1])
        if not blocks:
            raise ValueError("Mel is too short for RelMel block embedding.")

        layer = np.zeros_like(x_band, dtype=np.float32)
        counts = np.zeros_like(x_band, dtype=np.float32)
        for block_idx, start, end in blocks:
            active_bits = self._active_bits(block_idx, message_arr.size)
            block_weight = self._block_weight(x_band[:, start:end])
            active_scale = 1.0 / float(np.sqrt(max(len(active_bits), 1)))
            for bit_idx in active_bits:
                polarity = 1.0 if int(message_arr[bit_idx]) == 1 else -1.0
                pair = self._pair_vector(bit_idx, block_idx, x_band.shape[0], utterance_id)
                update = polarity * block_weight * active_scale * pair.reshape(-1, 1)
                layer[:, start:end] += update
                counts[:, start:end] += np.abs(pair).reshape(-1, 1)

        boundary = self._boundary_mask(x_band)
        # Keep heavily reused bins from receiving disproportionate energy.
        reuse = np.maximum(counts, 1.0)
        perturbation = self.config.alpha * boundary * layer / np.sqrt(reuse)
        x_wm[c_min:c_max, :] = np.clip(x_band + perturbation, 0.0, 1.0)

        ref = RelMelReference(
            utterance_id=utterance_id,
            clean_mel=clean_mel.astype(np.float32),
            message=message_arr.astype(np.uint8),
            norm_stats=norm_stats,
            mel_config=mel_config,
            relmel_config=self.config,
        )
        return x_wm.astype(np.float32), ref

    def extract(
        self,
        detected_mel: np.ndarray,
        reference: RelMelReference,
        key: Optional[str] = None,
        expected_message: Optional[Sequence[int] | np.ndarray] = None,
    ) -> RelMelResult:
        detected_mel = _as_mel(detected_mel)
        clean_mel = reference.clean_mel.astype(np.float32)
        cfg = reference.relmel_config
        detected_mel = align_mels(detected_mel, clean_mel, max_shift=cfg.align_max_shift)
        c_min, c_max = self._checked_band(clean_mel.shape[0], band=cfg.band)
        clean_band = clean_mel[c_min:c_max, :]
        detected_band = detected_mel[c_min:c_max, :]
        residual = detected_band - clean_band
        residual = residual - float(residual.mean())

        blocks = self._blocks(clean_band.shape[1], cfg=cfg)
        scores = np.zeros(reference.message.size, dtype=np.float32)
        votes = np.zeros(reference.message.size, dtype=np.float32)
        used_key = self.config.key if key is None else key

        for block_idx, start, end in blocks:
            active_bits = self._active_bits(block_idx, reference.message.size, cfg=cfg)
            block_residual = residual[:, start:end]
            block_weight = self._block_weight(clean_band[:, start:end], cfg=cfg)
            active_scale = 1.0 / float(np.sqrt(max(len(active_bits), 1)))
            for bit_idx in active_bits:
                pair = self._pair_vector(
                    bit_idx,
                    block_idx,
                    clean_band.shape[0],
                    reference.utterance_id,
                    key=used_key,
                    cfg=cfg,
                )
                # Relative energy score: positive means group A gained vs group B.
                raw_score = float(np.mean(block_residual * pair.reshape(-1, 1)))
                scores[bit_idx] += raw_score * block_weight * active_scale
                votes[bit_idx] += 1.0

        normalizer = np.sqrt(np.maximum(votes, 1.0))
        scores = (scores / normalizer).astype(np.float32)
        decoded = (scores >= 0.0).astype(np.uint8)
        target = reference.message if expected_message is None else _as_bits(expected_message)
        accuracy = None
        verified = None
        if target is not None:
            accuracy = bit_acc(target, decoded)
            verified = bool(accuracy >= cfg.threshold)
        return RelMelResult(
            decoded=decoded,
            scores=scores,
            votes=votes,
            bit_accuracy=accuracy,
            verified=verified,
        )

    def _blocks(
        self, frames: int, cfg: Optional[RelMelConfig] = None
    ) -> list[tuple[int, int, int]]:
        cfg = self.config if cfg is None else cfg
        blocks: list[tuple[int, int, int]] = []
        block_idx = 0
        for start in range(0, frames, cfg.block_frames):
            end = min(frames, start + cfg.block_frames)
            if end - start >= cfg.min_block_frames:
                blocks.append((block_idx, start, end))
                block_idx += 1
        return blocks

    def _active_bits(
        self, block_idx: int, payload_bits: int, cfg: Optional[RelMelConfig] = None
    ) -> np.ndarray:
        cfg = self.config if cfg is None else cfg
        count = min(cfg.bits_per_block, payload_bits)
        if count == payload_bits:
            return np.arange(payload_bits, dtype=np.int64)
        seed = _seed_from_parts(cfg.key, "active-order", str(payload_bits), str(count))
        rng = np.random.default_rng(seed)
        order = rng.permutation(payload_bits)
        start = (block_idx * count) % payload_bits
        if start + count <= payload_bits:
            active = order[start : start + count]
        else:
            active = np.concatenate([order[start:], order[: (start + count) % payload_bits]])
        return np.sort(active).astype(np.int64)

    def _pair_vector(
        self,
        bit_idx: int,
        block_idx: int,
        band_bins: int,
        utterance_id: str,
        key: Optional[str] = None,
        cfg: Optional[RelMelConfig] = None,
    ) -> np.ndarray:
        cfg = self.config if cfg is None else cfg
        pair_bins = min(cfg.pair_bins, max(1, band_bins // 2))
        seed = _seed_from_parts(
            key or cfg.key,
            utterance_id,
            "pair",
            str(bit_idx),
            str(block_idx),
            str(band_bins),
            str(pair_bins),
        )
        rng = np.random.default_rng(seed)
        selected = rng.choice(band_bins, size=2 * pair_bins, replace=False)
        pair = np.zeros(band_bins, dtype=np.float32)
        pair[selected[:pair_bins]] = 1.0
        pair[selected[pair_bins:]] = -1.0
        pair /= float(np.sqrt(2 * pair_bins))
        return pair.astype(np.float32)

    def _block_weight(
        self, block: np.ndarray, cfg: Optional[RelMelConfig] = None
    ) -> float:
        cfg = self.config if cfg is None else cfg
        energy = float(np.mean(block))
        # Normalized Mel already lies in [0, 1]. Emphasize energetic speech blocks
        # while keeping quiet blocks active enough for short utterances.
        return float(cfg.mask_floor + (1.0 - cfg.mask_floor) * np.clip(energy, 0.0, 1.0) ** cfg.energy_gamma)

    def _boundary_mask(self, x_band: np.ndarray) -> np.ndarray:
        if self.config.boundary_margin <= 0:
            return np.ones_like(x_band, dtype=np.float32)
        headroom = np.minimum(x_band, 1.0 - x_band)
        return np.clip(headroom / self.config.boundary_margin, 0.0, 1.0).astype(np.float32)

    def _checked_band(
        self, n_mels: int, band: Optional[tuple[int, int]] = None
    ) -> tuple[int, int]:
        c_min, c_max = self.config.band if band is None else band
        if not (0 <= c_min < c_max <= n_mels):
            raise ValueError(f"Invalid Mel band {band or self.config.band} for n_mels={n_mels}")
        return int(c_min), int(c_max)


def _seed_from_parts(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _as_mel(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected Mel shape [n_mels, frames], got {arr.shape}")
    return arr


def _as_bits(value: Sequence[int] | np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.uint8)
    if bits.ndim != 1:
        raise ValueError("Message must be one-dimensional.")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Message can contain only 0/1 bits.")
    return bits
