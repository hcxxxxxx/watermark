# Paper Experiment Results Summary

This file records the experiment results that are currently useful for the RelMel paper draft.
All RelMel and reproduced MelShield numbers below use LJSpeech with HiFi-GAN unless otherwise noted.

## Main Candidate

Reliable-pair RelMel is the current main method candidate.

Configuration:

| Parameter | Value |
|---|---:|
| alpha | 0.435 |
| band | 20:60 |
| mask_floor | 0.20 |
| boundary_margin | 0.01 |
| block_frames | 8 |
| block_stride | 8 |
| bits_per_block | 6 |
| pair_bins | 6 |
| pair_candidates | 16 |
| detector_mode | plain |
| payload_bits | 32 |
| ecc_repeat | 1 |

Full attacks, random 500 samples, seed 2026:

| Attack | Bit Acc | Verification Rate | PESQ BM | STOI BM |
|---|---:|---:|---:|---:|
| none | 0.9999 | 1.000 | 3.5102 | 0.9685 |
| mp3 | 0.9999 | 1.000 | 3.5093 | 0.9685 |
| aac | 0.9999 | 1.000 | 3.4918 | 0.9683 |
| scale | 0.9999 | 1.000 | 3.5102 | 0.9685 |
| rs16 | 0.9999 | 1.000 | 3.5102 | 0.9685 |
| bandpass | 0.9999 | 1.000 | 3.5299 | 0.9632 |
| lowpass | 0.9998 | 1.000 | 3.5185 | 0.9678 |
| noise20 | 0.9946 | 1.000 | 1.5574 | 0.9527 |
| noise10 | 0.9455 | 0.990 | 1.0994 | 0.8991 |
| noise5 | 0.8648 | 0.932 | 1.0416 | 0.8430 |
| echo | 0.9983 | 1.000 | 1.7726 | 0.9177 |

Short interpretation:

- This is the current strongest paper-ready RelMel result.
- At nearly the same perceptual quality as the reproduced MelShield strong baseline, it substantially improves robustness under additive noise.
- It exceeds MelShield reported HiFi-GAN results on noise10 and noise5, while noise20 is very close to 1.0.

## Main Comparison Table

| Method | Samples | PESQ none | none | noise20 | noise10 | noise5 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| MelShield reported | paper | not directly matched | 1.0000 | 1.0000 | 0.7815 | 0.7052 | HiFi-GAN table, external reported numbers |
| MelShield reproduced, alpha 0.05, band 20:60 | 100 | 3.5069 | 1.0000 | 0.9072 | 0.7097 | 0.6391 | Strong reproduced baseline |
| MelShield reproduced, alpha 0.055, band 20:60 | 100 | 3.4324 | 1.0000 | 0.9269 | 0.7325 | 0.6475 | Stronger but lower quality |
| RelMel raw, alpha 0.43, bf8, bpb6, pb6 | 100 | 3.5206 | 1.0000 | 0.9534 | 0.7841 | 0.6875 | Before reliable-pair selection |
| RelMel raw stronger, alpha 0.455, mf0.20, bm0.005 | 100 | 3.4688 | 1.0000 | 0.9638 | 0.7900 | 0.6950 | Before reliable-pair selection |
| RelMel reliable-pair, pc8 | 100 | 3.5228 | 1.0000 | 0.9941 | 0.9188 | 0.8291 | Higher-quality reliable-pair variant |
| RelMel reliable-pair, pc16 | 100 | 3.4939 | 0.9997 | 0.9941 | 0.9456 | 0.8688 | Main candidate, random 100 |
| RelMel reliable-pair, pc16 | 500 | 3.5102 | 0.9999 | 0.9946 | 0.9455 | 0.8648 | Main candidate, random 500 |

## Reliable Pair Candidate Ablation

Fixed configuration:

- alpha 0.435
- band 20:60
- mask_floor 0.20
- boundary_margin 0.01
- block_frames 8
- block_stride 8
- bits_per_block 6
- pair_bins 6
- detector_mode plain
- random 60 samples, seed 2026
- attacks: none, noise20, noise10, noise5

| pair_candidates | PESQ none | none | noise20 | noise10 | noise5 | Observation |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 3.5186 | 1.0000 | 0.9589 | 0.7870 | 0.6943 | Old random-pair baseline |
| 4 | 3.5576 | 1.0000 | 0.9870 | 0.8740 | 0.7688 | Large improvement |
| 8 | 3.5217 | 1.0000 | 0.9901 | 0.9219 | 0.8214 | Strong reliable-pair setting |
| 16 | 3.4888 | 0.9995 | 0.9943 | 0.9370 | 0.8646 | Best robustness in this sweep |

Short interpretation:

- The largest method-level gain comes from pair_candidates, not from detector_mode.
- The improvement is monotonic or nearly monotonic on noise robustness from 1 to 16 candidates.
- This should become a core ablation table in the paper.

## Detector Mode Observation

The boundary detector was tested together with pair_candidates.
It did not provide stable improvements over the plain detector.

Recommended paper choice:

- Use detector_mode plain in the main method.
- Mention boundary-matched detection as an attempted variant only if useful in supplementary material.

## Earlier RelMel Development Results

These are useful for the ablation narrative but should not be the main final comparison.

| Method / Config | Samples | PESQ none | none | noise20 | noise10 | noise5 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| RelMel bf16 bpb4 pb4, alpha 0.25 | 40 | 4.0909 | 0.9984 | 0.8203 | not tested | not tested | Early high-quality setting |
| RelMel bf16 bpb4 pb4, alpha 0.40 | 40 | 3.6786 | 0.9992 | 0.9063 | not tested | not tested | Stronger early setting |
| RelMel bf16 bpb4 pb4, alpha 0.44 | 100 | 3.4986 | 0.9994 | 0.9194 | 0.7356 | 0.6559 | Early full-attack setting |
| RelMel bf8 bpb6 pb6, alpha 0.43 | 100 | 3.5206 | 1.0000 | 0.9534 | 0.7841 | 0.6875 | Strong raw RelMel before reliable pairs |

## Suggested Next Experiments

These commands are intended to strengthen the paper evidence.

### Pair-Candidate Ablation, Random 500

Run this to turn the random-60 ablation into a stronger paper table.

```bash
python scripts/grid_relmel_ljspeech.py \
  --config configs/relmel_hifigan.yaml \
  --suite custom \
  --output-dir runs/relmel_pair_candidates_ablation_random500 \
  --limit 500 \
  --sample-mode random \
  --seed 2026 \
  --attacks none noise20 noise10 noise5 \
  --alpha-grid 0.435 \
  --band-grid 20:60 \
  --mask-floor-grid 0.20 \
  --boundary-margin-grid 0.01 \
  --block-frames-grid 8 \
  --block-stride-grid 8 \
  --min-block-frames-grid 8 \
  --bits-per-block-grid 6 \
  --pair-bins-grid 6 \
  --pair-candidates-grid 1 4 8 16 \
  --detector-mode-grid plain \
  --energy-gamma-grid 0.5 \
  --keep-candidate-results
```

### Multi-Seed Stability, Key Noise Attacks

Run these to show that the main result is not tied to seed 2026.

```bash
python scripts/grid_relmel_ljspeech.py \
  --config configs/relmel_hifigan.yaml \
  --suite custom \
  --output-dir runs/relmel_reliable_pc16_a0435_mf020_bm001_keyattacks_random500_seed2027 \
  --limit 500 \
  --sample-mode random \
  --seed 2027 \
  --attacks none noise20 noise10 noise5 \
  --alpha-grid 0.435 \
  --band-grid 20:60 \
  --mask-floor-grid 0.20 \
  --boundary-margin-grid 0.01 \
  --block-frames-grid 8 \
  --block-stride-grid 8 \
  --min-block-frames-grid 8 \
  --bits-per-block-grid 6 \
  --pair-bins-grid 6 \
  --pair-candidates-grid 16 \
  --detector-mode-grid plain \
  --energy-gamma-grid 0.5 \
  --keep-candidate-results
```

```bash
python scripts/grid_relmel_ljspeech.py \
  --config configs/relmel_hifigan.yaml \
  --suite custom \
  --output-dir runs/relmel_reliable_pc16_a0435_mf020_bm001_keyattacks_random500_seed2028 \
  --limit 500 \
  --sample-mode random \
  --seed 2028 \
  --attacks none noise20 noise10 noise5 \
  --alpha-grid 0.435 \
  --band-grid 20:60 \
  --mask-floor-grid 0.20 \
  --boundary-margin-grid 0.01 \
  --block-frames-grid 8 \
  --block-stride-grid 8 \
  --min-block-frames-grid 8 \
  --bits-per-block-grid 6 \
  --pair-bins-grid 6 \
  --pair-candidates-grid 16 \
  --detector-mode-grid plain \
  --energy-gamma-grid 0.5 \
  --keep-candidate-results
```

### Quality-First Reliable-Pair Variant, Random 500

Run this if the paper needs a higher-quality variant in addition to the main robust variant.

```bash
python scripts/grid_relmel_ljspeech.py \
  --config configs/relmel_hifigan.yaml \
  --suite custom \
  --output-dir runs/relmel_reliable_pc8_a0435_mf020_bm001_attacks_random500 \
  --limit 500 \
  --sample-mode random \
  --seed 2026 \
  --attacks none mp3 aac scale rs16 bandpass lowpass noise20 noise10 noise5 echo \
  --alpha-grid 0.435 \
  --band-grid 20:60 \
  --mask-floor-grid 0.20 \
  --boundary-margin-grid 0.01 \
  --block-frames-grid 8 \
  --block-stride-grid 8 \
  --min-block-frames-grid 8 \
  --bits-per-block-grid 6 \
  --pair-bins-grid 6 \
  --pair-candidates-grid 8 \
  --detector-mode-grid plain \
  --energy-gamma-grid 0.5 \
  --keep-candidate-results
```

## Notes for Paper Writing

- Use MelShield reported numbers only as external reference.
- The most defensible main comparison is against the reproduced and tuned MelShield baseline under the same codebase, vocoder, Mel frontend, attacks, and metrics.
- Emphasize quality-matched comparison: RelMel reliable-pair pc16 has PESQ 3.5102, close to reproduced MelShield alpha 0.05 with PESQ 3.5069.
- The current core contribution should be phrased as reliable-pair selection for block-wise relative Mel watermarking.
- Avoid over-claiming that RelMel beats MelShield reported on every attack, because MelShield reported noise20 is 1.0000 while RelMel currently has 0.9946.
