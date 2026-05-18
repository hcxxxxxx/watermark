import numpy as np

from melshield.coding import RepetitionCode, bit_accuracy


def test_repetition_code_soft_decoding_corrects_one_error_per_group() -> None:
    payload = np.array([0, 1, 1, 0], dtype=np.uint8)
    codec = RepetitionCode(repeat=3, interleave=True)
    codeword = codec.encode(payload)
    scores = np.where(codeword == 1, 1.0, -1.0).astype(np.float32)
    scores[[0, 1, 2, 3]] *= -1.0

    decoded, soft_scores = codec.decode_scores(scores)

    assert soft_scores.shape == payload.shape
    assert bit_accuracy(payload, decoded) == 1.0
