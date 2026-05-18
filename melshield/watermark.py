from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .mel import MelConfig, NormalizationStats, align_mels


@dataclass(frozen=True)
class MelShieldConfig:
    payload_bits: int = 32
    key: str = "melshield-demo-key"
    alpha: float = 0.25
    band: tuple[int, int] = (20, 56)
    threshold: float = 0.61
    headroom: float = 0.0
    mask_floor: float = 0.05
    energy_gamma: float = 0.75
    boundary_margin: float = 0.02
    align_max_shift: int = 12
    mask_mode: str = "energy"
    freq_gamma: float = 0.0
    texture_gamma: float = 0.0
    smooth_frames: int = 1

    def to_dict(self) -> dict:
        data = asdict(self)
        data["band"] = list(self.band)
        return data


@dataclass(frozen=True)
class ReferenceRecord:
    utterance_id: str
    clean_mel: np.ndarray
    message: np.ndarray
    norm_stats: NormalizationStats
    mel_config: MelConfig
    wm_config: MelShieldConfig

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "utterance_id": self.utterance_id,
            "norm_stats": self.norm_stats.to_dict(),
            "mel_config": self.mel_config.to_dict(),
            "wm_config": self.wm_config.to_dict(),
        }
        np.savez_compressed(
            path,
            clean_mel=self.clean_mel.astype(np.float32),
            message=self.message.astype(np.uint8),
            metadata=json.dumps(metadata, sort_keys=True),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceRecord":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            mel_cfg = MelConfig(**metadata["mel_config"])
            wm_cfg_data = metadata["wm_config"]
            wm_cfg_data["band"] = tuple(wm_cfg_data["band"])
            wm_cfg = MelShieldConfig(**wm_cfg_data)
            stats = NormalizationStats(**metadata["norm_stats"])
            return cls(
                utterance_id=metadata["utterance_id"],
                clean_mel=data["clean_mel"].astype(np.float32),
                message=data["message"].astype(np.uint8),
                norm_stats=stats,
                mel_config=mel_cfg,
                wm_config=wm_cfg,
            )


@dataclass(frozen=True)
class ExtractionResult:
    decoded: np.ndarray
    scores: np.ndarray
    bit_accuracy: Optional[float]
    verified: Optional[bool]


class MelShield:
    """Keyed spread-spectrum watermarking in normalized log-Mel space."""

    def __init__(self, config: MelShieldConfig) -> None:
        self.config = config

    def embed(
        self,
        clean_mel: np.ndarray,
        message: Sequence[int] | np.ndarray,
        utterance_id: str,
        norm_stats: NormalizationStats,
        mel_config: MelConfig,
    ) -> tuple[np.ndarray, ReferenceRecord]:
        clean_mel = _as_mel(clean_mel)
        message_arr = _as_bits(message)
        if message_arr.size != self.config.payload_bits:
            raise ValueError(
                f"Expected {self.config.payload_bits} payload bits, got {message_arr.size}"
            )
        c_min, c_max = self._checked_band(clean_mel.shape[0])
        x_wm = clean_mel.copy()
        x_band = clean_mel[c_min:c_max, :]
        if self.config.headroom > 0:
            x_band = np.clip(x_band, self.config.headroom, 1.0 - self.config.headroom)

        mask = embedding_mask(
            x_band,
            floor=self.config.mask_floor,
            gamma=self.config.energy_gamma,
            boundary_margin=self.config.boundary_margin,
            mode=self.config.mask_mode,
            freq_gamma=self.config.freq_gamma,
            texture_gamma=self.config.texture_gamma,
            smooth_frames=self.config.smooth_frames,
        )
        layer = self._watermark_layer(message_arr, x_band.shape, utterance_id)
        x_wm[c_min:c_max, :] = np.clip(
            x_band + self.config.alpha * mask * layer,
            0.0,
            1.0,
        )

        ref = ReferenceRecord(
            utterance_id=utterance_id,
            clean_mel=clean_mel.astype(np.float32),
            message=message_arr.astype(np.uint8),
            norm_stats=norm_stats,
            mel_config=mel_config,
            wm_config=self.config,
        )
        return x_wm.astype(np.float32), ref

    def extract(
        self,
        detected_mel: np.ndarray,
        reference: ReferenceRecord,
        key: Optional[str] = None,
        expected_message: Optional[Sequence[int] | np.ndarray] = None,
    ) -> ExtractionResult:
        detected_mel = _as_mel(detected_mel)
        clean_mel = reference.clean_mel.astype(np.float32)
        detected_mel = align_mels(
            detected_mel, clean_mel, max_shift=reference.wm_config.align_max_shift
        )
        c_min, c_max = self._checked_band(clean_mel.shape[0], band=reference.wm_config.band)
        delta = detected_mel - clean_mel
        delta = delta - float(delta.mean())
        delta_band = delta[c_min:c_max, :]
        x_band = clean_mel[c_min:c_max, :]
        mask = embedding_mask(
            x_band,
            floor=reference.wm_config.mask_floor,
            gamma=reference.wm_config.energy_gamma,
            boundary_margin=reference.wm_config.boundary_margin,
            mode=reference.wm_config.mask_mode,
            freq_gamma=reference.wm_config.freq_gamma,
            texture_gamma=reference.wm_config.texture_gamma,
            smooth_frames=reference.wm_config.smooth_frames,
        )

        used_key = self.config.key if key is None else key
        scores = np.empty(reference.message.size, dtype=np.float32)
        weighted_delta = delta_band * mask
        normalizer = float(np.sqrt(np.sum(mask * mask)) + 1e-8)
        for bit_idx in range(reference.message.size):
            pattern = self._pattern(
                bit_idx=bit_idx,
                shape=x_band.shape,
                utterance_id=reference.utterance_id,
                key=used_key,
            )
            scores[bit_idx] = float(np.sum(weighted_delta * pattern) / normalizer)

        decoded = (scores >= 0.0).astype(np.uint8)
        target = reference.message if expected_message is None else _as_bits(expected_message)
        bit_accuracy = None
        verified = None
        if target is not None:
            bit_accuracy = bit_acc(target, decoded)
            verified = bool(bit_accuracy >= reference.wm_config.threshold)
        return ExtractionResult(
            decoded=decoded,
            scores=scores,
            bit_accuracy=bit_accuracy,
            verified=verified,
        )

    def message_from_id(self, utterance_id: str, namespace: str = "payload") -> np.ndarray:
        return deterministic_bits(
            key=f"{self.config.key}|{namespace}",
            identifier=utterance_id,
            length=self.config.payload_bits,
        )

    def _watermark_layer(
        self, message: np.ndarray, shape: tuple[int, int], utterance_id: str
    ) -> np.ndarray:
        layer = np.zeros(shape, dtype=np.float32)
        scale = 1.0 / float(np.sqrt(message.size))
        for bit_idx, bit in enumerate(message):
            polarity = 1.0 if int(bit) == 1 else -1.0
            layer += polarity * self._pattern(bit_idx, shape, utterance_id, self.config.key)
        return (scale * layer).astype(np.float32)

    def _pattern(
        self,
        bit_idx: int,
        shape: tuple[int, int],
        utterance_id: str,
        key: Optional[str] = None,
    ) -> np.ndarray:
        seed = _seed_from_parts(
            key or self.config.key,
            utterance_id,
            str(bit_idx),
            str(shape[0]),
            str(shape[1]),
        )
        rng = np.random.default_rng(seed)
        values = rng.integers(0, 2, size=shape, dtype=np.int8)
        return (2.0 * values.astype(np.float32) - 1.0).astype(np.float32)

    def _checked_band(
        self, n_mels: int, band: Optional[tuple[int, int]] = None
    ) -> tuple[int, int]:
        c_min, c_max = self.config.band if band is None else band
        if not (0 <= c_min < c_max <= n_mels):
            raise ValueError(f"Invalid Mel band {self.config.band} for n_mels={n_mels}")
        return int(c_min), int(c_max)


def adaptive_mask(
    x_band: np.ndarray,
    floor: float = 0.05,
    gamma: float = 0.75,
    boundary_margin: float = 0.02,
) -> np.ndarray:
    x_band = np.asarray(x_band, dtype=np.float32)
    frame_energy = x_band.mean(axis=0)
    lo, hi = np.percentile(frame_energy, [10.0, 90.0])
    weights = (frame_energy - lo) / max(float(hi - lo), 1e-8)
    weights = np.clip(weights, 0.0, 1.0) ** gamma
    weights = floor + (1.0 - floor) * weights
    mask = np.broadcast_to(weights.reshape(1, -1), x_band.shape).astype(np.float32).copy()
    if boundary_margin > 0:
        headroom = np.minimum(x_band, 1.0 - x_band)
        boundary = np.clip(headroom / boundary_margin, 0.0, 1.0)
        mask *= boundary.astype(np.float32)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def embedding_mask(
    x_band: np.ndarray,
    floor: float = 0.05,
    gamma: float = 0.75,
    boundary_margin: float = 0.02,
    mode: str = "energy",
    freq_gamma: float = 0.0,
    texture_gamma: float = 0.0,
    smooth_frames: int = 1,
) -> np.ndarray:
    """Build the embedding reliability mask.

    ``mode="energy"`` reproduces the original paper-style adaptive mask.
    ``mode="reliability"`` adds frequency and local-texture reliability terms.
    The added terms are deterministic functions of the reference Mel, so the
    verifier can reconstruct the same mask without storing a dense map.
    """

    mask = adaptive_mask(
        x_band,
        floor=floor,
        gamma=gamma,
        boundary_margin=boundary_margin,
    )
    if mode == "energy":
        return mask
    if mode != "reliability":
        raise ValueError(f"Unknown mask_mode={mode!r}. Use 'energy' or 'reliability'.")

    x_band = np.asarray(x_band, dtype=np.float32)
    if freq_gamma > 0:
        freq_energy = x_band.mean(axis=1)
        freq_weights = _percentile_weights(freq_energy, floor=floor) ** freq_gamma
        mask *= freq_weights.reshape(-1, 1).astype(np.float32)

    if texture_gamma > 0:
        frame_mean = x_band.mean(axis=0, keepdims=True)
        texture = np.abs(x_band - frame_mean)
        lo, hi = np.percentile(texture, [10.0, 90.0])
        texture_weights = (texture - lo) / max(float(hi - lo), 1e-8)
        texture_weights = np.clip(texture_weights, 0.0, 1.0)
        texture_weights = floor + (1.0 - floor) * texture_weights
        mask *= texture_weights.astype(np.float32) ** texture_gamma

    if smooth_frames > 1:
        mask = _smooth_time(mask, smooth_frames)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _percentile_weights(values: np.ndarray, floor: float) -> np.ndarray:
    lo, hi = np.percentile(values, [10.0, 90.0])
    weights = (values - lo) / max(float(hi - lo), 1e-8)
    weights = np.clip(weights, 0.0, 1.0)
    return (floor + (1.0 - floor) * weights).astype(np.float32)


def _smooth_time(mask: np.ndarray, frames: int) -> np.ndarray:
    frames = max(1, int(frames))
    if frames <= 1 or mask.shape[1] <= 1:
        return mask.astype(np.float32)
    kernel = np.ones(frames, dtype=np.float32) / float(frames)
    padded = np.pad(mask, ((0, 0), (frames // 2, frames - 1 - frames // 2)), mode="edge")
    smoothed = np.empty_like(mask, dtype=np.float32)
    for idx in range(mask.shape[0]):
        smoothed[idx] = np.convolve(padded[idx], kernel, mode="valid")
    return smoothed


def deterministic_bits(key: str, identifier: str, length: int) -> np.ndarray:
    bits = []
    counter = 0
    while len(bits) < length:
        digest = hashlib.sha256(f"{key}|{identifier}|{counter}".encode("utf-8")).digest()
        for byte in digest:
            for offset in range(8):
                bits.append((byte >> offset) & 1)
                if len(bits) == length:
                    break
            if len(bits) == length:
                break
        counter += 1
    return np.asarray(bits, dtype=np.uint8)


def bit_acc(expected: Sequence[int] | np.ndarray, decoded: Sequence[int] | np.ndarray) -> float:
    expected = _as_bits(expected)
    decoded = _as_bits(decoded)
    if expected.size != decoded.size:
        raise ValueError(f"Bit length mismatch: {expected.size} != {decoded.size}")
    return float(np.mean(expected == decoded))


def _seed_from_parts(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _as_mel(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected Mel shape [n_mels, frames], got {arr.shape}")
    return arr


def _as_bits(value: Sequence[int] | np.ndarray) -> np.ndarray:
    bits = np.asarray(list(value) if not isinstance(value, np.ndarray) else value, dtype=np.uint8)
    if bits.ndim != 1:
        raise ValueError("Message must be a one-dimensional bit vector.")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Message can contain only 0/1 bits.")
    return bits
