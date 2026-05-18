import numpy as np

from melshield.mel import MelConfig, NormalizationStats
from melshield.relmel import RelMelConfig, RelMelMark, RelMelReference


def test_relmel_roundtrip_in_mel_domain(tmp_path) -> None:
    rng = np.random.default_rng(13)
    clean = rng.uniform(0.15, 0.85, size=(80, 256)).astype(np.float32)
    cfg = RelMelConfig(
        payload_bits=32,
        key="relmel-test",
        alpha=0.08,
        band=(20, 56),
        block_frames=32,
        bits_per_block=16,
        pair_bins=3,
    )
    marker = RelMelMark(cfg)
    message = marker.message_from_id("utt-rel")
    watermarked, reference = marker.embed(
        clean,
        message,
        "utt-rel",
        NormalizationStats(-8.0, 2.0),
        MelConfig(),
    )
    result = marker.extract(watermarked, reference)
    assert result.bit_accuracy == 1.0

    path = tmp_path / "relmel_ref.npz"
    reference.save(path)
    loaded = RelMelReference.load(path)
    result = marker.extract(watermarked, loaded)
    assert result.bit_accuracy == 1.0
