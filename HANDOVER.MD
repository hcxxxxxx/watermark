# RAWMER Project Handover

## 1. Purpose and Current State

This file is the starting point for a new AI session or collaborator with no
prior conversation context. The project is an ICASSP-style paper and codebase
for **RAWMER: Reference-Assisted Speech Watermarking via Relative Mel-Energy
Relations**.

The manuscript is an advanced draft, not a finished submission. The core method,
main comparisons, verification controls, compression studies, partial-clip
studies, and EnCodec experiments have been implemented and run. The immediate
work is mainly careful paper assembly: keeping all tables and prose aligned with
the final experiment configurations, making threshold conventions explicit, and
deciding how much of the supplementary evidence fits in the main paper.

Important terminology:

- **RAWMER** is the final method/paper name.
- **RelMel** is the older internal code and experiment-directory name for the
  same method. A `relmel_*` path generally contains a RAWMER experiment.
- **MelShield** is the closest mel-domain, reference-assisted baseline. The
  repository contains a documented local reproduction, not an official release.
- The project is **reference-assisted verification**, not blind watermarking.

Do not describe RAWMER as a blind detector, or imply that it needs no reference.

## 2. Read These Files First

Read these in this order before changing paper text or interpreting experiments:

1. `core_idea.md` -- full technical and narrative explanation, including claims
   that should and should not be made.
2. `paper_experiment_results.md` -- authoritative human-readable result ledger.
3. `RAWMER_paper.txt` -- current ICASSP-style manuscript draft.
4. `experiments_writing_context.md` -- compact experiments-writing reference.
5. `README.md` -- environment, data layout, and baseline reproduction details.

The working tree may be dirty. Preserve all existing user edits. At the time this
handover was written, `RAWMER_paper.txt`, `core_idea.md`, and
`paper_experiment_results.md` already had uncommitted edits, and the repository
also contained a user-provided reviewed PDF. Do not revert or overwrite these
files wholesale.

## 3. Research Problem and System Assumption

RAWMER targets provenance verification for neural speech generation. The
generation provider retains a generation record containing an utterance ID,
payload or metadata to regenerate it, a protected secret key, and a clean Mel
reference (possibly compressed). Given a suspect waveform and a claimed record,
the verifier re-extracts its Mel representation, aligns it with the stored
reference, decodes the expected payload, and decides whether the claimed record
matches.

This setting is practical for a TTS provider that already retains generation
logs. It is deliberately narrower than public, blind watermark detection. The
claim is provider-side provenance verification / forensic attribution, not
universal watermark detection by arbitrary third parties.

At a high level:

```text
clean Mel reference M + key + payload
        -> RAWMER embedder -> watermarked Mel M_hat
        -> HiFi-GAN or DiffWave -> released waveform

suspect waveform + stored M + claimed payload + protected key
        -> Mel extraction/alignment -> residual R = M_suspect - M
        -> relative-energy decoder -> payload ACC -> verification decision
```

## 4. Method: What Is Actually New

MelShield-style methods encode payload bits as absolute, keyed residual
patterns. RAWMER instead encodes each bit as the **sign of a relative Mel-energy
relation**:

- Split the Mel time axis into short blocks.
- For each active payload bit in a block, choose disjoint positive and negative
  Mel-bin groups in the watermark band.
- Bit 1 raises the positive group and lowers the negative group; bit 0 applies
  the opposite polarity.
- At verification, decode from the signed difference between residual energy in
  the two groups, rather than from a single absolute residual pattern.
- Repeat payload bits over multiple blocks and aggregate block votes.

The most important additional mechanism is **reliability-aware pair selection**.
For every bit/block, RAWMER samples multiple keyed candidate positive/negative
pairs and ranks them using the clean reference. The score favors bins with both
boundary headroom and sufficient speech energy, while penalizing positive/
negative reliability imbalance. The main configuration uses 16 candidates and
selects the best one.

Why it matters: a zero-sum differential measurement suppresses local common-mode
effects such as modest gain shifts, smooth vocoder bias, and some spectral
distortion. Selecting pairs that are likely to survive vocoding makes the
relation more stable under additive noise and neural-codec compression.

The implementation center is `melshield/relmel.py`. Supporting components are:

- `melshield/mel.py`: Mel analysis, normalization, alignment.
- `melshield/coding.py`: payload/code handling; current main runs use 32 bits
  with `ecc_repeat=1`, so payload and code accuracy coincide.
- `melshield/attacks.py`: signal-processing and extended attacks.
- `melshield/vocoders/hifigan.py` and `melshield/vocoders/command.py`: vocoder
  adapters.

## 5. Final Main Protocols

All main experiments use LJSpeech, `sample_mode=random`, and `seed=2026`.
Main mel-domain comparisons use 2,000 utterances and 32-bit payloads. DiffWave
uses the command vocoder with sampling seed 0.

### RAWMER, HiFi-GAN (main configuration)

```text
alpha=0.435
band=20:60
mask_floor=0.20
boundary_margin=0.01
energy_gamma=0.5
block_frames=8
block_stride=8
min_block_frames=8
bits_per_block=6
pair_bins=6
pair_candidates=16
detector_mode=plain
align_max_shift=12
payload_bits=32
ecc_repeat=1
verification threshold tau=0.75
```

### RAWMER, DiffWave (main configuration)

Same as HiFi-GAN except `alpha=0.355`. Do **not** replace it with `alpha=0.358`:
the latter has marginally stronger robustness but clean PESQ 3.4971, below the
3.5 target. The final `alpha=0.355` random2000 full-attack run has clean PESQ
3.5140.

### MelShield, quality-matched local reproduction

HiFi-GAN:

```text
alpha=0.05, band=20:60, mask_floor=0.05, energy_gamma=0.75,
boundary_margin=0.02, align_max_shift=12, headroom=0.0,
payload_bits=32, verification threshold tau=0.61
```

DiffWave:

```text
alpha=0.061, band=20:60, mask_floor=0.05, energy_gamma=0.75,
boundary_margin=0.02, align_max_shift=12, headroom=0.0,
payload_bits=32, verification threshold tau=0.61
```

`alpha=0.061` is the final DiffWave MelShield baseline. Earlier runs and some
scripts use 0.060; treat them as pilot/edge-search evidence, not the final main
comparison.

## 6. Metrics and Threshold Semantics

- **ACC (bit accuracy)**: fraction of the expected payload bits recovered
  correctly. It is threshold-free and should be the primary cross-method metric.
- **VR (verification rate)**: fraction of utterances for which ACC reaches a
  method-specific acceptance threshold. It is an operational claimed-record
  verification metric, not strict blind recovery of a user ID.
- **PESQ/STOI**: objective quality/intelligibility of watermarked or attacked
  audio against unwatermarked benchmark audio. Use clean/no-attack PESQ to
  compare embedding quality; low PESQ after severe added noise is expected.

For RAWMER, the main threshold is `tau=0.75` (at least 24 matching bits out of
32). A 32/32 exact match is not required because this work studies robust
claimed-record verification under attacks; exact blind user-ID recovery would
require a different protocol, usually a longer payload plus outer ECC/MAC.

For MelShield, the calibrated threshold in the local reproduction is `tau=0.61`.
The main table must never suggest that both methods use one common VR threshold.
Recommended caption language:

```text
RAWMER uses tau=0.75 and MelShield uses its calibrated tau=0.61.
ACC is threshold-free; VR is reported under the corresponding fixed threshold.
```

Useful threshold-sensitivity result already recomputed from DiffWave per-sample
CSV files:

```text
                 noise20    noise10    noise5
RAWMER tau=0.61   1.0000     1.0000     0.9850
MelShield tau=.75 0.9970     0.8410     0.4850
```

This supports the statement that RAWMER's ACC advantage is not a threshold
artifact. If reporting these values, call them a threshold-sensitivity check,
not another main protocol. A helper script/snippet can recompute HiFi-GAN values
from `results.csv`; no costly experiment rerun is needed as long as the CSV has
been retained.

### Waveform baselines have different VR rules

AudioSeal and WavMark use 16-bit payloads and are cross-paradigm reference
baselines, not same-protocol competitors.

- **AudioSeal VR** requires both detector score >= 0.5 and payload ACC >= 0.75.
  `message_threshold=0.5` only binarizes message probabilities; it is not the
  final verification threshold.
- **WavMark table values use forced decoding**: forced 16-bit payload ACC >=
  0.75. The high-level decoder rejects many noise-corrupted samples; do not
  label this row simply "WavMark" without a `(forced)` qualifier.

## 7. Main Results That Are Safe to Use

### Quality-matched RAWMER vs. MelShield (random2000)

| Vocoder | Method | Clean PESQ | ACC none | ACC noise20 | ACC noise10 | ACC noise5 |
|---|---|---:|---:|---:|---:|---:|
| HiFi-GAN | MelShield | 3.5100 | 0.9993 | 0.9028 | 0.7129 | 0.6328 |
| HiFi-GAN | RAWMER | 3.5141 | 0.9999 | 0.9942 | 0.9408 | 0.8587 |
| DiffWave | MelShield | 3.5149 | 0.9998 | 0.9685 | 0.8225 | 0.7219 |
| DiffWave | RAWMER | 3.5140 | 0.9998 | 0.9904 | 0.9218 | 0.8353 |

The additive-noise gains of RAWMER over quality-matched MelShield are:

- HiFi-GAN: +9.1 / +22.8 / +22.6 percentage points at noise20/noise10/noise5.
- DiffWave: +2.2 / +9.9 / +11.3 percentage points.

The primary full-attack values are in `paper_experiment_results.md`. Key
RAWMER values include:

- HiFi-GAN: noise20 `ACC/VR=0.9942/1.000`, noise10 `0.9408/0.991`, noise5
  `0.8587/0.918`, echo `0.9986/1.000`.
- DiffWave: noise20 `0.9904/0.9995`, noise10 `0.9218/0.9825`, noise5
  `0.8353/0.871`, echo `0.9969/1.000`.

### Neural-codec attack (EnCodec, random500)

EnCodec 6/12/24 kbps is an important modern-attack supplement. RAWMER ACC:

| Vocoder | 6 kbps | 12 kbps | 24 kbps |
|---|---:|---:|---:|
| HiFi-GAN RAWMER | 0.9761 | 0.9901 | 0.9951 |
| HiFi-GAN MelShield | 0.8876 | 0.9417 | 0.9541 |
| DiffWave RAWMER | 0.9768 | 0.9889 | 0.9920 |
| DiffWave MelShield | 0.9431 | 0.9689 | 0.9761 |

RAWMER VR is at least 0.998 in these runs. Keep this as random500 supplementary
evidence; do not mix it into the random2000 main table without noting scale.

### Reference-assistance controls (HiFi-GAN, random2000, no attack)

| Condition | ACC | VR |
|---|---:|---:|
| Correct key/reference/payload | 0.9999 | 1.0000 |
| Clean unmarked audio | 0.5008 | 0.0040 |
| Wrong key | 0.5030 | 0.0015 |
| Wrong payload | 0.5000 | 0.0040 |
| Wrong reference | 0.4991 | 0.0035 |

This is core evidence for the reasonableness of the reference-assisted setting.
The wrong-payload condition can retain high raw confidence but fails payload
matching, so attribution must be written as payload-match verification, not
confidence-only detection.

### Reference compression

HiFi-GAN RAWMER, random2000:

- Full float32 Mel reference: 174.96 KB/utterance.
- `band_uint8` (only bins 20:60, 8-bit): 21.87 KB/utterance, one eighth of
  float32 storage. ACC at noise20/noise10/noise5 is
  `0.9939/0.9394/0.8584`, versus `0.9943/0.9407/0.8595` for float32.
- The corresponding `band_uint8` VR is `1.000/0.9925/0.9245`.

Aggressive random500 comparison:

- RAWMER `band_uint4`, 11.05 KB: noise20/noise10/noise5 ACC/VR
  `0.9826/1.000`, `0.9128/0.984`, `0.8245/0.856`.
- MelShield `uint2`, also 11.05 KB: `0.7859/0.982`, `0.6554/0.710`,
  `0.6007/0.442`.

The precise claim is that RAWMER can retain only its watermark band and quantize
it substantially while preserving verification better in this local
implementation. Do not claim a universal property of all MelShield variants.

### Partial clips and alignment

Blind fragment verification on HiFi-GAN (random500) searches the reference with
a 4-frame step. For 25% fragments, ACC is 0.985--0.991 and VR is 0.998--1.000;
for 50% fragments, ACC is at least 0.997 and VR is 1.000. Recovered crop-start
median error is 0--1 Mel frame. This is stronger evidence than the known-offset
fragment experiment and is the preferred main-paper result.

### Documented limits

RAWMER is robust to common processing, EnCodec, and moderate noise, but depends
on reference/suspect Mel alignment. Pitch and time-scale changes are the main
failure mode:

- Pitch shifts through +/-50 cents: approximately >=0.995 ACC and VR 1.000.
- +/-100 cents: ACC about 0.78, VR about 0.72.
- +/-3% speed: VR about 0.996--0.998.
- +/-5% speed: VR about 0.85--0.906.
- +/-10% speed: VR about 0.26--0.32.

Frame-level alignment is therefore an honest limitation and a clear future-work
direction (multi-scale alignment, DTW, pitch-aware matching).

## 8. Remote Experiment Data Locations

Large experimental artifacts are stored on the remote GPU server, normally at
`/home/csy/watermark`. This local checkout does not necessarily contain a
`runs/` directory. Each final grid run with `--keep-candidate-results` should
have a `candidate_001/summary.json` and `candidate_001/results.csv`.

The following are the most important result locations. Use `summary.json` for
quick reporting and `results.csv` for threshold recomputation or per-example
analysis.

```text
# Main random2000 full attacks
runs/relmel_hifigan_a0435_mf020_bm001_attacks_random2000/candidate_001/
runs/relmel_diffwave_a0355_mf020_bm001_full_attacks_random2000/candidate_001/
runs/melshield_hifigan_a005_mf005_bm002_band2060_attacks_random2000/candidate_001/
runs/melshield_diffwave_a0061_mf005_bm002_full_attacks_random2000/candidate_001/

# Reference controls and compression
runs/relmel_reference_controls_hifigan_band2060_random2000/
runs/relmel_reference_compression_hifigan_band2060_random2000/
runs/relmel_hifigan_reference_uint_sweep_random500/
runs/melshield_hifigan_reference_uint_sweep_random500/

# Partial-clip studies
runs/relmel_fragments_hifigan_band2060_random500/
runs/relmel_blind_fragments_hifigan_band2060_random500/

# Modern attacks
runs/relmel_hifigan_neural_codec_random500/candidate_001/
runs/relmel_diffwave_neural_codec_random500/candidate_001/
runs/melshield_hifigan_neural_codec_random500/candidate_001/
runs/melshield_diffwave_neural_codec_random500/candidate_001/

# Waveform baselines (random500)
runs/wavmark_hifigan_forced_attacks_random500/
runs/wavmark_diffwave_forced_attacks_random500/
```

If a directory name differs slightly, search remotely with:

```bash
cd /home/csy/watermark
find runs -path '*candidate_001/summary.json' | sort
```

## 9. Scripts, Configurations, and Reproducibility

Important runners:

- `scripts/grid_relmel_ljspeech.py`: RAWMER grid/full-attack runner.
- `scripts/grid_melshield_ljspeech.py`: MelShield grid/full-attack runner.
- `scripts/run_relmel_ljspeech.py`, `scripts/run_melshield_ljspeech.py`:
  direct single-configuration runners.
- `scripts/grid_waveform_baselines_ljspeech.py`: AudioSeal/WavMark baselines.
- `scripts/eval_relmel_reference_controls_ljspeech.py`: wrong key/payload/
  reference controls.
- `scripts/eval_relmel_reference_compression_ljspeech.py` and
  `scripts/eval_melshield_reference_compression_ljspeech.py`: compression.
- `scripts/eval_relmel_fragments_ljspeech.py` and
  `scripts/eval_relmel_blind_fragments_ljspeech.py`: known-offset and blind
  partial clips.
- `scripts/run_neural_codec_attacks.sh`: EnCodec study.
- `scripts/run_reference_compression_uint_sweep.sh`: shared RAWMER/MelShield
  aggressive compression sweep.
- `scripts/export_relmel_method_mels.py`: creates clean/watermarked/noise20/
  noise10/noise5 Mel figures and corresponding audio for the method figure.

Relevant configs:

- `configs/relmel_hifigan.yaml`
- `configs/relmel_diffwave.yaml`
- `configs/melshield_hifigan_official.yaml`
- `configs/melshield_diffwave.yaml`

Config defaults often predate the final paper parameters. Treat the explicit
arguments in the final run records and Section 5 above as authoritative.

Remote environment normally uses conda environment `melshield`, Python 3.10,
CUDA, FFmpeg, HiFi-GAN checkpoints in `checkpoints/hifigan/`, and the DiffWave
checkpoint at `external/diffwave/checkpoints/diffwave-ljspeech-22kHz-1000578.pt`.

## 10. Manuscript Writing Rules and Current Paper Plan

The paper should tell this story:

1. Neural speech generation needs provenance verification.
2. Provider-side reference-assisted verification is a realistic, distinct
   deployment setting.
3. Absolute Mel residual patterns are fragile under noise/vocoder distortion.
4. RAWMER embeds relative energy relations and selects reliable, balanced pairs.
5. Quality-matched comparisons show large additive-noise gains across two
   vocoders.
6. Controls establish selectivity; compression and fragments establish practical
   deployment properties.
7. State the synchronization limitation honestly.

Keep the following distinctions explicit:

- Compare RAWMER fairly to the local, quality-matched MelShield reproduction.
  MelShield paper-reported values are external reference only and must remain
  visibly separate.
- AudioSeal/WavMark are 16-bit waveform methods; RAWMER/MelShield main results
  are 32-bit Mel-domain, reference-assisted methods. Call this a
  cross-paradigm reference, not a fully controlled head-to-head comparison.
- Bold main-table **ACC**, not an entire ACC/VR cell, because VR thresholds are
  method-specific.
- Label WavMark as `WavMark (forced)` in any table using forced-decoding values.
- Do not use vague claims such as "near-perfect" when precise numbers fit. For
  example, use `over 99.97% ACC under MP3/AAC` only when the stated row supports
  it.
- Do not say RAWMER is robust to all attacks. It is not robust to sufficiently
  large speed/pitch transformations.

Suggested main-paper tables, in priority order:

1. Quality-matched RAWMER vs. MelShield, both vocoders, random2000.
2. Full-attack ACC/VR table or compact summary, with threshold footnote.
3. Reference controls + compression + blind fragments, possibly grouped into one
   compact practical-validation table.
4. Waveform baseline table, clearly marked cross-paradigm/random500.
5. EnCodec and speed/pitch as an extended-attack table or supplement.

## 11. Remaining Work and Sensible Next Steps

High-value paper tasks:

1. Reconcile `RAWMER_paper.txt` with final results one last time, especially
   final DiffWave alpha 0.355 and MelShield alpha 0.061.
2. Ensure all main-table captions explain the RAWMER/MelShield VR thresholds and
   label the WavMark forced-decoding row.
3. Turn the current plaintext ICASSP tables into final LaTeX tables, keeping
   random2000 and random500 protocols separate.
4. Finalize the method figure using `figures/rawmer_method_mels/` on the remote
   machine if the generated assets are needed.
5. Audit references/citation keys in `RAWMER_paper.txt`; it still contains
   placeholder-style `[REF: ...]` tags and notes about unverified references.

Optional additional experiments, only if time/space justify them:

- Cross-dataset validation on LibriSpeech or Common Voice.
- A second neural codec such as DAC or SoundStream.
- Multi-seed statistics for main claims.
- Alignment-aware RAWMER variant for speed/pitch robustness.

These are improvements, not prerequisites for interpreting the completed core
evidence. Do not rerun completed random2000 studies merely for more data unless
there is a specific inconsistency to resolve.

## 12. One-Paragraph Project Summary for a New Model

RAWMER is a reference-assisted Mel-domain speech watermark for provider-side
provenance verification. It embeds a 32-bit payload as repeated, zero-sum
relative energy relations between reliability-selected positive and negative
Mel-bin groups, then decodes those relations from the residual against a stored
clean Mel reference. On LJSpeech, quality-matched random2000 tests show that it
substantially outperforms a local MelShield reproduction under additive noise on
both HiFi-GAN and DiffWave while preserving clean PESQ around 3.51. The project
also establishes low false-positive rates with wrong key/payload/reference
controls, near-lossless band-only 8-bit reference storage at 21.87 KB per
utterance, blind verification of 25% clips, and strong EnCodec robustness. Its
main current limitation is synchronization-sensitive speed and pitch changes.
