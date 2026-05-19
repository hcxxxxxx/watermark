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


def test_relmel_overlap_blocks_increase_votes() -> None:
    rng = np.random.default_rng(17)
    clean = rng.uniform(0.15, 0.85, size=(80, 96)).astype(np.float32)
    base = RelMelConfig(
        payload_bits=32,
        key="relmel-overlap-test",
        alpha=0.08,
        band=(20, 56),
        block_frames=8,
        bits_per_block=6,
        pair_bins=4,
        min_block_frames=8,
    )
    overlap = RelMelConfig(
        payload_bits=32,
        key="relmel-overlap-test",
        alpha=0.08,
        band=(20, 56),
        block_frames=8,
        block_stride=4,
        bits_per_block=6,
        pair_bins=4,
        min_block_frames=8,
    )

    base_marker = RelMelMark(base)
    overlap_marker = RelMelMark(overlap)
    message = base_marker.message_from_id("utt-overlap")
    stats = NormalizationStats(-8.0, 2.0)
    mel_config = MelConfig()

    base_wm, base_ref = base_marker.embed(clean, message, "utt-overlap", stats, mel_config)
    overlap_wm, overlap_ref = overlap_marker.embed(clean, message, "utt-overlap", stats, mel_config)

    base_result = base_marker.extract(base_wm, base_ref)
    overlap_result = overlap_marker.extract(overlap_wm, overlap_ref)
    assert base_result.bit_accuracy == 1.0
    assert overlap_result.bit_accuracy == 1.0
    assert overlap_result.votes.mean() > base_result.votes.mean()


def test_relmel_reliable_pairs_and_boundary_detector_roundtrip() -> None:
    rng = np.random.default_rng(23)
    clean = rng.uniform(0.05, 0.95, size=(80, 128)).astype(np.float32)
    cfg = RelMelConfig(
        payload_bits=32,
        key="relmel-reliable-test",
        alpha=0.1,
        band=(20, 60),
        block_frames=8,
        block_stride=8,
        bits_per_block=6,
        pair_bins=6,
        pair_candidates=8,
        detector_mode="boundary",
        min_block_frames=8,
        mask_floor=0.2,
        boundary_margin=0.01,
    )
    marker = RelMelMark(cfg)
    message = marker.message_from_id("utt-reliable")
    watermarked, reference = marker.embed(
        clean,
        message,
        "utt-reliable",
        NormalizationStats(-8.0, 2.0),
        MelConfig(),
    )
    result = marker.extract(watermarked, reference)
    assert result.bit_accuracy == 1.0
