from pathlib import Path

import numpy as np

from melshield.mel import MelConfig, NormalizationStats
from melshield.watermark import MelShield, MelShieldConfig, ReferenceRecord


def test_embed_extract_in_mel_domain_is_exact() -> None:
    rng = np.random.default_rng(7)
    clean = rng.uniform(0.15, 0.85, size=(80, 192)).astype(np.float32)
    cfg = MelShieldConfig(payload_bits=64, key="unit-key", alpha=0.25, band=(20, 56))
    shield = MelShield(cfg)
    message = shield.message_from_id("utt-001")
    watermarked, record = shield.embed(
        clean,
        message,
        utterance_id="utt-001",
        norm_stats=NormalizationStats(-8.0, 2.0),
        mel_config=MelConfig(),
    )

    result = shield.extract(watermarked, record)
    assert result.bit_accuracy == 1.0
    assert result.verified is True


def test_wrong_key_behaves_like_random_guess() -> None:
    rng = np.random.default_rng(9)
    clean = rng.uniform(0.1, 0.9, size=(80, 240)).astype(np.float32)
    cfg = MelShieldConfig(payload_bits=128, key="right-key", alpha=0.25, band=(20, 56))
    shield = MelShield(cfg)
    message = shield.message_from_id("utt-002")
    watermarked, record = shield.embed(
        clean,
        message,
        utterance_id="utt-002",
        norm_stats=NormalizationStats(-8.0, 2.0),
        mel_config=MelConfig(),
    )

    result = shield.extract(watermarked, record, key="wrong-key")
    assert result.bit_accuracy is not None
    assert result.bit_accuracy < 0.70


def test_reference_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    clean = rng.uniform(0.2, 0.8, size=(80, 96)).astype(np.float32)
    cfg = MelShieldConfig(payload_bits=32, key="roundtrip", alpha=0.25, band=(20, 56))
    shield = MelShield(cfg)
    message = shield.message_from_id("utt-003")
    watermarked, record = shield.embed(
        clean,
        message,
        utterance_id="utt-003",
        norm_stats=NormalizationStats(-7.0, 1.5),
        mel_config=MelConfig(),
    )
    path = tmp_path / "ref.npz"
    record.save(path)
    loaded = ReferenceRecord.load(path)
    result = shield.extract(watermarked, loaded)
    assert result.bit_accuracy == 1.0
