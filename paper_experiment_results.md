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

## Current RelMel vs MelShield Comparable Data

This table separates matched reproduced comparisons from external reported numbers.
At this point, the strongest matched RelMel-vs-MelShield comparison is still HiFi-GAN.
DiffWave RelMel results are promising, but the matched MelShield-DiffWave baseline has not been run yet.

| Vocoder | Method | Source / Run | Samples | PESQ none | none | noise20 | noise10 | noise5 | Comparison Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| HiFi-GAN | MelShield | Reported in MelShield paper | paper | n/a | 1.0000 | 1.0000 | 0.7815 | 0.7052 | External reference, not directly quality-matched |
| HiFi-GAN | MelShield | Reproduced, alpha 0.05, band 20:60 | 100 | 3.5069 | 1.0000 | 0.9072 | 0.7097 | 0.6391 | Matched local baseline |
| HiFi-GAN | MelShield | Reproduced, alpha 0.055, band 20:60 | 100 | 3.4324 | 1.0000 | 0.9269 | 0.7325 | 0.6475 | Stronger but below the 3.5 quality target |
| HiFi-GAN | RelMel reliable-pair | pc16, full attacks, random 500 | 500 | 3.5102 | 0.9999 | 0.9946 | 0.9455 | 0.8648 | Main matched RelMel result |
| DiffWave | MelShield | alpha 0.05, band 20:56, full attacks, random 500 | 500 | 3.8779 | 0.9996 | 0.9383 | 0.7725 | 0.6824 | Matched local baseline |
| DiffWave | RelMel reliable-pair | alpha 0.35, mf 0.20, bm 0.01, full attacks, random 500 | 500 | 3.5443 | 0.9998 | 0.9914 | 0.9225 | 0.8308 | Main RelMel-DiffWave result |

## DiffWave MelShield Reproduction Results

These runs use the same fixed DiffWave sampling seed as the RelMel-DiffWave runs:

```text
python scripts/diffwave_vocoder.py ... --device cuda --fast --seed 0
```

### Stage1 Key-Attack Sweep

Random 60 samples, seed 2026, attacks: none, noise20, noise10, noise5.
Fixed parameters: mask_floor 0.05, energy_gamma 0.75, boundary_margin 0.02, threshold 0.61, align_max_shift 12, headroom 0.0.

| alpha | band | PESQ none | none | noise20 | noise10 | noise5 | Observation |
|---:|---|---:|---:|---:|---:|---:|---|
| 0.015 | 20:56 | 4.4415 | 0.9323 | 0.6828 | 0.5833 | 0.5542 | High quality, weak detection |
| 0.015 | 20:60 | 4.4443 | 0.9396 | 0.6854 | 0.5964 | 0.5667 | High quality, weak detection |
| 0.020 | 20:56 | 4.3902 | 0.9698 | 0.7328 | 0.6255 | 0.5703 | Improving robustness |
| 0.020 | 20:60 | 4.3977 | 0.9755 | 0.7406 | 0.6240 | 0.5828 | Improving robustness |
| 0.025 | 20:56 | 4.3279 | 0.9859 | 0.7885 | 0.6344 | 0.5948 | Default-like strength |
| 0.025 | 20:60 | 4.3418 | 0.9901 | 0.7938 | 0.6464 | 0.5990 | Default-like strength |
| 0.030 | 20:56 | 4.2576 | 0.9948 | 0.8266 | 0.6750 | 0.6063 | Better noise20 |
| 0.030 | 20:60 | 4.2709 | 0.9948 | 0.8365 | 0.6802 | 0.6135 | Better noise20 |
| 0.035 | 20:56 | 4.1789 | 0.9979 | 0.8724 | 0.7078 | 0.6271 | Moderate robustness |
| 0.035 | 20:60 | 4.1898 | 0.9969 | 0.8698 | 0.7104 | 0.6417 | Moderate robustness |
| 0.040 | 20:56 | 4.0822 | 0.9990 | 0.9031 | 0.7292 | 0.6495 | Stronger, still high quality |
| 0.040 | 20:60 | 4.0836 | 0.9984 | 0.9026 | 0.7344 | 0.6479 | Stronger, still high quality |
| 0.045 | 20:56 | 3.9699 | 0.9995 | 0.9260 | 0.7552 | 0.6615 | Stronger baseline |
| 0.045 | 20:60 | 3.9761 | 0.9990 | 0.9260 | 0.7573 | 0.6703 | Stronger baseline |
| 0.050 | 20:56 | 3.8563 | 0.9995 | 0.9406 | 0.7714 | 0.6818 | Stage1 best objective |
| 0.050 | 20:60 | 3.8467 | 0.9995 | 0.9396 | 0.7677 | 0.6870 | Slightly higher noise5, lower overall objective |

Current MelShield-DiffWave baseline choice:

- Use alpha 0.05 and band 20:56 for the full random500 run, because it has the best stage1 objective and strong noise20/noise10 while keeping PESQ 3.8563.
- The random500 full-attack result below confirms this choice.
- The matched random500 gap to RelMel-DiffWave is large on additive noise: RelMel-DiffWave has noise20 0.9914, noise10 0.9225, and noise5 0.8308.

### Main Full-Attack Result

Configuration:

| Parameter | Value |
|---|---:|
| alpha | 0.05 |
| band | 20:56 |
| mask_floor | 0.05 |
| energy_gamma | 0.75 |
| boundary_margin | 0.02 |
| threshold | 0.61 |
| align_max_shift | 12 |
| headroom | 0.0 |
| mask_mode | energy |

Full attacks, random 500 samples, seed 2026:

| Attack | Bit Acc | Verification Rate | PESQ BM | STOI BM |
|---|---:|---:|---:|---:|
| none | 0.9996 | 1.000 | 3.8779 | 0.9662 |
| mp3 | 0.9994 | 1.000 | 3.8770 | 0.9661 |
| aac | 0.9995 | 1.000 | 3.8545 | 0.9659 |
| scale | 0.9996 | 1.000 | 3.8779 | 0.9662 |
| rs16 | 0.9982 | 1.000 | 3.8779 | 0.9662 |
| bandpass | 0.9976 | 1.000 | 3.8782 | 0.9610 |
| lowpass | 0.9869 | 1.000 | 3.9093 | 0.9659 |
| noise20 | 0.9383 | 0.998 | 1.4972 | 0.9488 |
| noise10 | 0.7725 | 0.940 | 1.0779 | 0.8983 |
| noise5 | 0.6824 | 0.800 | 1.0349 | 0.8478 |
| echo | 0.9945 | 1.000 | 1.6347 | 0.9167 |

Short interpretation:

- This is the matched MelShield-DiffWave local baseline.
- MelShield-DiffWave has higher none PESQ than RelMel-DiffWave, but substantially lower additive-noise robustness.
- Under the same random500 full-attack protocol, RelMel-DiffWave improves noise20 by 5.3 points, noise10 by 15.0 points, and noise5 by 14.8 points.

## DiffWave RelMel Results

These runs use the DiffWave command vocoder with fixed sampling seed:

```text
python scripts/diffwave_vocoder.py ... --device cuda --fast --seed 0
```

The fixed DiffWave seed is important because otherwise clean and watermarked Mel inputs are synthesized with different diffusion noise, which makes PESQ BM include vocoder sampling randomness.

### Main Full-Attack Result

Configuration:

| Parameter | Value |
|---|---:|
| alpha | 0.35 |
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
| none | 0.9998 | 1.000 | 3.5443 | 0.9612 |
| mp3 | 0.9998 | 1.000 | 3.5428 | 0.9612 |
| aac | 0.9998 | 1.000 | 3.5191 | 0.9609 |
| scale | 0.9998 | 1.000 | 3.5443 | 0.9612 |
| rs16 | 0.9998 | 1.000 | 3.5443 | 0.9612 |
| bandpass | 0.9998 | 1.000 | 3.5227 | 0.9559 |
| lowpass | 0.9989 | 1.000 | 3.6133 | 0.9612 |
| noise20 | 0.9914 | 1.000 | 1.4930 | 0.9438 |
| noise10 | 0.9225 | 0.984 | 1.0787 | 0.8932 |
| noise5 | 0.8308 | 0.852 | 1.0351 | 0.8437 |
| echo | 0.9971 | 1.000 | 1.6116 | 0.9128 |

Short interpretation:

- This is the current strongest RelMel-DiffWave result.
- It satisfies the 3.5 PESQ target while keeping strong noise robustness on random 500.
- Compared with the HiFi-GAN RelMel main result, DiffWave keeps slightly higher none PESQ but lower additive-noise robustness, especially on noise10 and noise5.
- A matched MelShield-DiffWave baseline is still required before making a final DiffWave-specific superiority claim.

### Seeded Pipeline Diagnostic

Alpha-zero probe, random 20 samples, seed 2026, attack: none only.

| alpha | PESQ none | none Bit Acc | Interpretation |
|---:|---:|---:|---|
| 0.00 | 4.6439 | 0.4906 | DiffWave reconstruction is high quality; no-watermark detection is near random as expected |

### Seeded Quality-Robustness Sweep

Random 40 samples, seed 2026, attacks: none, noise20, noise10, noise5.
The table keeps the best objective candidate for each alpha from the broad sweep.

| alpha | mask_floor | boundary_margin | PESQ none | none | noise20 | noise10 | noise5 | Observation |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.12 | 0.25 | 0.005 | 4.3938 | 0.9945 | 0.8563 | 0.7102 | 0.6367 | Very high quality, weak robustness |
| 0.15 | 0.20 | 0.010 | 4.3390 | 0.9984 | 0.8984 | 0.7477 | 0.6680 | Still quality-heavy |
| 0.18 | 0.20 | 0.010 | 4.2731 | 0.9992 | 0.9367 | 0.7891 | 0.7141 | Better noise robustness |
| 0.22 | 0.20 | 0.010 | 4.1421 | 1.0000 | 0.9539 | 0.8352 | 0.7320 | Good quality margin |
| 0.26 | 0.20 | 0.010 | 3.9866 | 1.0000 | 0.9734 | 0.8563 | 0.7672 | Stronger robustness with good quality |
| 0.30 | 0.25 | 0.005 | 3.7717 | 1.0000 | 0.9820 | 0.8922 | 0.8000 | First clearly strong DiffWave point |

Short interpretation:

- DiffWave needs a lower RelMel strength than HiFi-GAN for the same 3.5 PESQ target.
- The useful DiffWave strength range is around alpha 0.30 to 0.35.
- The alpha-zero diagnostic confirms that the low PESQ seen in the unseeded run was a sampling-comparison artifact, not a Mel-format failure.

### Alpha Edge Sweep

Random 40 samples, seed 2026, fixed mask_floor 0.25 and boundary_margin 0.005.

| alpha | PESQ none | none | noise20 | noise10 | noise5 | Interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 0.30 | 3.7717 | 1.0000 | 0.9797 | 0.8859 | 0.7852 | Safe quality margin |
| 0.34 | 3.5557 | 1.0000 | 0.9867 | 0.9109 | 0.8125 | Near the best quality-robustness trade-off |
| 0.38 | 3.3374 | 1.0000 | 0.9938 | 0.9297 | 0.8281 | Below the 3.5 PESQ target |

### Alpha 0.35 / 0.36 Edge Sweep

Random 40 samples, seed 2026, attacks: none, noise20, noise10, noise5.

| alpha | mask_floor | boundary_margin | PESQ none | none | noise20 | noise10 | noise5 | Objective | Selection Note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.35 | 0.20 | 0.005 | 3.5343 | 1.0000 | 0.9891 | 0.9141 | 0.8148 | 3.9426 | Valid quality |
| 0.35 | 0.20 | 0.010 | 3.5363 | 1.0000 | 0.9883 | 0.9094 | 0.8289 | 3.9591 | Recommended DiffWave pilot config |
| 0.35 | 0.25 | 0.005 | 3.5023 | 1.0000 | 0.9914 | 0.9078 | 0.8172 | 3.9434 | Barely above quality floor |
| 0.35 | 0.25 | 0.010 | 3.5041 | 1.0000 | 0.9891 | 0.9102 | 0.8250 | 3.9547 | Barely above quality floor |
| 0.36 | 0.20 | 0.005 | 3.4822 | 1.0000 | 0.9875 | 0.9109 | 0.8352 | 3.9245 | Below quality floor |
| 0.36 | 0.20 | 0.010 | 3.4835 | 1.0000 | 0.9906 | 0.9094 | 0.8242 | 3.9132 | Below quality floor |
| 0.36 | 0.25 | 0.005 | 3.4484 | 1.0000 | 0.9898 | 0.9164 | 0.8164 | 3.8185 | Below quality floor |
| 0.36 | 0.25 | 0.010 | 3.4496 | 1.0000 | 0.9922 | 0.9148 | 0.8281 | 3.8404 | Below quality floor |

Current DiffWave recommendation:

- Use alpha 0.35, mask_floor 0.20, boundary_margin 0.01 as the DiffWave main RelMel configuration.
- It keeps PESQ above 3.5 while reaching noise10 0.9225 and noise5 0.8308 on random 500 full attacks.
- Run the matched MelShield-DiffWave baseline before making a final paper claim for the DiffWave vocoder.

## Reliable Pair Candidate Ablation

### Random 500 Ablation

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
- random 500 samples, seed 2026
- attacks: none, noise20, noise10, noise5

| pair_candidates | PESQ none | none | noise20 | noise10 | noise5 | Observation |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 3.5485 | 0.9999 | 0.9459 | 0.7674 | 0.6684 | Old random-pair baseline |
| 4 | 3.5299 | 0.9999 | 0.9873 | 0.8894 | 0.7845 | Large improvement |
| 8 | 3.5184 | 0.9999 | 0.9918 | 0.9232 | 0.8315 | Strong reliable-pair setting |
| 16 | 3.5102 | 0.9999 | 0.9952 | 0.9439 | 0.8607 | Best robustness in this sweep |

Short interpretation:

- This is now the strongest ablation evidence for the paper.
- Increasing pair_candidates from 1 to 16 improves noise20 by 4.9 points, noise10 by 17.7 points, and noise5 by 19.2 points.
- PESQ changes only from 3.5485 to 3.5102, so the robustness gain is not simply caused by much lower perceptual quality.

### Random 60 Pilot Ablation

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

Pilot interpretation:

- The largest method-level gain comes from pair_candidates, not from detector_mode.
- The improvement is monotonic or nearly monotonic on noise robustness from 1 to 16 candidates.
- This pilot trend is confirmed by the random-500 ablation above.

## Multi-Seed Stability

Main reliable-pair configuration:

- alpha 0.435
- band 20:60
- mask_floor 0.20
- boundary_margin 0.01
- block_frames 8
- block_stride 8
- bits_per_block 6
- pair_bins 6
- pair_candidates 16
- detector_mode plain
- random 500 samples
- attacks: none, noise20, noise10, noise5

| Seed | PESQ none | none | noise20 | noise10 | noise5 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 3.5102 | 0.9999 | 0.9952 | 0.9439 | 0.8607 |
| 2027 | 3.5108 | 0.9999 | 0.9950 | 0.9411 | 0.8588 |
| 2028 | 3.5164 | 0.9998 | 0.9938 | 0.9391 | 0.8574 |
| Mean | 3.5125 | 0.9999 | 0.9946 | 0.9413 | 0.8590 |

Short interpretation:

- The main reliable-pair configuration is stable across random seeds.
- The standard variation is small relative to the improvement over the reproduced MelShield baseline.
- For paper tables, use the full-attack random-500 result as the main result and this table as robustness-to-sampling evidence.
- Do not mix this key-attack-only table with the full-attack table as if they were produced by the exact same command; use each table with its own stated protocol.

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
