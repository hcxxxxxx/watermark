from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RepetitionCode:
    """Simple soft-decision repetition code for payload attribution IDs."""

    repeat: int = 3
    interleave: bool = True

    def __post_init__(self) -> None:
        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")

    def encode(self, payload: Sequence[int] | np.ndarray) -> np.ndarray:
        payload_arr = _as_bits(payload)
        codeword = np.repeat(payload_arr, self.repeat).astype(np.uint8)
        if self.interleave and self.repeat > 1:
            codeword = codeword.reshape(-1, self.repeat).T.reshape(-1)
        return codeword.astype(np.uint8)

    def decode_hard(self, decoded_codeword: Sequence[int] | np.ndarray) -> np.ndarray:
        decoded = _as_bits(decoded_codeword)
        groups = self._deinterleave(decoded).reshape(-1, self.repeat)
        votes = groups.sum(axis=1)
        return (votes >= (self.repeat / 2.0)).astype(np.uint8)

    def decode_scores(self, scores: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score_arr = np.asarray(scores, dtype=np.float32)
        if score_arr.ndim != 1 or score_arr.size % self.repeat != 0:
            raise ValueError("Score vector length must be divisible by repeat.")
        groups = self._deinterleave(score_arr).reshape(-1, self.repeat)
        soft_scores = groups.sum(axis=1)
        decoded = (soft_scores >= 0.0).astype(np.uint8)
        return decoded, soft_scores.astype(np.float32)

    def _deinterleave(self, values: np.ndarray) -> np.ndarray:
        if not self.interleave or self.repeat == 1:
            return values
        payload_bits = values.size // self.repeat
        return values.reshape(self.repeat, payload_bits).T.reshape(-1)


def bit_accuracy(expected: Sequence[int] | np.ndarray, decoded: Sequence[int] | np.ndarray) -> float:
    expected_arr = _as_bits(expected)
    decoded_arr = _as_bits(decoded)
    if expected_arr.shape != decoded_arr.shape:
        raise ValueError(f"Bit length mismatch: {expected_arr.size} != {decoded_arr.size}")
    return float(np.mean(expected_arr == decoded_arr))


def mean_abs_confidence(scores: Sequence[float] | np.ndarray) -> float:
    score_arr = np.asarray(scores, dtype=np.float32)
    if score_arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(score_arr)))


def _as_bits(value: Sequence[int] | np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.uint8)
    if bits.ndim != 1:
        raise ValueError("Bits must be one-dimensional.")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Bits can contain only 0/1 values.")
    return bits
