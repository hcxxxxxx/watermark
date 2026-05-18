# MelShield Reproduction

This repository is a from-paper reproduction of **MelShield: Robust Mel-Domain
Audio Watermarking for Provenance Attribution of AI Generated Synthesized
Speech** (`2605.01515v1.pdf`).

The paper does not train a new neural network. MelShield is a keyed,
reference-based spread-spectrum watermark in normalized log-Mel space:

1. Compute an 80-bin normalized log-Mel spectrogram `X in [0, 1]`.
2. Restrict embedding to the mid-frequency band `F = {20, ..., 55}`.
3. Generate one deterministic `{-1, +1}` spreading pattern per payload bit
   from `(secret key, utterance id, bit index)`.
4. Superpose all bit layers with the paper's `1/sqrt(L)` energy normalization.
5. Add `alpha * adaptive_mask * watermark_layer` to the selected Mel band.
6. Store the clean reference Mel and metadata for owner-side verification.
7. Decode from a suspect waveform by recomputing Mel, subtracting the reference,
   and taking the sign of each keyed masked-correlation score.

The adaptive mask is implemented from the paper description: it favors
higher-energy frames and downweights bins close to the `[0, 1]` clipping
boundaries. The exact mask formula is not specified in the PDF, so this part is
a documented faithful approximation rather than a verbatim official release.

## Dataset Path

Download/extract LJSpeech relative to the repository root:

```text
watermark/
  data/
    LJSpeech-1.1/
      metadata.csv
      wavs/
        LJ001-0001.wav
        ...
```

So the default dataset path is:

```text
data/LJSpeech-1.1
```

You can check the layout with:

```bash
python scripts/prepare_ljspeech.py --root data/LJSpeech-1.1
```

## Environment

Your remote server has CUDA 12.8, so the requirements file points pip at the
official PyTorch `cu128` wheel index.

```bash
conda create -n melshield python=3.10 -y
conda activate melshield
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
```

Or:

```bash
conda env create -f environment.yml
conda activate melshield
```

`ffmpeg` is needed only for the MP3-128 and AAC-96 robustness attacks.

## Quick Smoke Test

This runs the watermark pipeline directly in Mel space, without a neural
vocoder. It is useful after pushing to the server to confirm that the repo and
dataset paths are correct.

```bash
python scripts/run_melshield_ljspeech.py \
  --config configs/melshield_ljspeech.yaml \
  --vocoder mel \
  --limit 10
```

Expected output files:

```text
runs/melshield_ljspeech/
  refs/*.npz
  results.csv
  summary.json
```

## HiFi-GAN Reproduction

Place an LJSpeech-compatible 22.05 kHz HiFi-GAN checkpoint here:

```text
checkpoints/hifigan/config.json
checkpoints/hifigan/generator_v1
```

Then run:

```bash
python scripts/run_melshield_ljspeech.py \
  --config configs/melshield_hifigan_official.yaml \
  --vocoder hifigan \
  --vocoder-config checkpoints/hifigan/config.json \
  --vocoder-checkpoint checkpoints/hifigan/generator_v1 \
  --payload-bits 32 \
  --limit 100 \
  --save-audio
```

For common official LJSpeech HiFi-GAN checkpoints, make sure the Mel frontend
matches `checkpoints/hifigan/config.json`; these checkpoints usually use
`f_min = 0` and `f_max = 8000`. The paper reports `alpha = 0.25` for HiFi-GAN
in its normalized Mel setting, but official checkpoints consume raw log-Mel, so
start lower and sweep:

```bash
for a in 0.005 0.01 0.02 0.05 0.1; do
  python scripts/run_melshield_ljspeech.py \
    --config configs/melshield_hifigan_official.yaml \
    --alpha "$a" \
    --attacks none \
    --limit 100 \
    --output-dir "runs/hifigan_alpha_${a}"
done
```

Pick the smallest `alpha` whose no-attack `mean_bit_acc` is high while
`pesq_bm`/`stoi_bm` remain healthy, then run the full attack suite.

## MelShield Reproduction Grid

To audit whether this implementation is underestimating the original MelShield
method, run a consolidated grid over the Mel frontend and embedding-mask
settings. The script writes all candidates into one CSV/JSON pair:

```bash
python scripts/grid_melshield_ljspeech.py \
  --config configs/melshield_hifigan_official.yaml \
  --suite repro-audit \
  --output-dir runs/melshield_repro_grid \
  --limit 40 \
  --attacks none noise20
```

The built-in `repro-audit` suite currently evaluates 227 targeted candidates:
alpha scaling, Mel frontend variants, embedding band, mask floor, mask gamma,
boundary margin, headroom, verifier threshold/shift, and a reliability-mask
variant. It uses only `none` and `noise20` by default so the audit finishes much
faster than the full attack suite. Send back:

```text
runs/melshield_repro_grid/grid_results.csv
runs/melshield_repro_grid/best_candidate.json
```

After the best candidate is selected, rerun only that configuration with the
full attack list in `scripts/run_melshield_ljspeech.py`.

## DiffWave / Other Vocoders

The watermark itself is vocoder-agnostic. For DiffWave, use the command adapter
with the official DiffWave inference script from your server checkout. The
external command must accept a raw log-Mel `.npy` and write a `.wav`:

```bash
python scripts/run_melshield_ljspeech.py \
  --config configs/melshield_ljspeech.yaml \
  --vocoder command \
  --vocoder-command "python external/diffwave/inference.py --spectrogram_path {mel_npy} --output {audio_wav}" \
  --alpha 0.025 \
  --payload-bits 32 \
  --limit 100
```

The paper uses `alpha = 0.025` for DiffWave.

## One-file Verification

After an experiment, verify one suspect waveform against its saved reference:

```bash
python scripts/verify_reference.py \
  --reference runs/melshield_ljspeech/refs/LJ001-0001.npz \
  --audio runs/melshield_ljspeech/audio/LJ001-0001_wm.wav \
  --key change-this-secret-key
```

Verification accepts a claimed identity when `BitAcc >= 0.61`, matching the
threshold region described in the paper.

## Important Files

- `melshield/watermark.py`: embedding, reference saving, extraction, bit accuracy
- `melshield/coding.py`: soft-decision repetition code used by the research prototype
- `melshield/mel.py`: log-Mel frontend, normalization, frame alignment
- `melshield/attacks.py`: MP3, AAC, scaling, resampling, filters, noise, echo
- `melshield/vocoders/hifigan.py`: native HiFi-GAN generator loader
- `scripts/run_melshield_ljspeech.py`: end-to-end LJSpeech experiment runner
- `scripts/grid_melshield_ljspeech.py`: consolidated MelShield reproduction grid
- `scripts/run_research_melshield_ljspeech.py`: adaptive-mask + soft-ECC research runner
- `configs/melshield_ljspeech.yaml`: paper-style default configuration
- `configs/research_melshield_hifigan.yaml`: research prototype configuration

## Research Prototype

The research runner adds three experimental extensions:

- reliability-aware masking with deterministic frequency/texture weights;
- soft-decision repetition coding, e.g. a 32-bit payload becomes a 96-bit codeword
  when `repeat=3`;
- attack-aware grid search over `alpha`, repetition factor, and mask parameters.

Run a small attack-aware search:

```bash
python scripts/run_research_melshield_ljspeech.py \
  --config configs/research_melshield_hifigan.yaml \
  --optimize \
  --search-limit 20 \
  --train-attacks none mp3 aac noise20 echo \
  --alpha-grid 0.025 0.03 0.035 0.04 0.045 0.05 \
  --repeat-grid 1 3 \
  --freq-gamma-grid 0.0 0.5 \
  --texture-gamma-grid 0.0 0.25 \
  --smooth-grid 1 5 \
  --output-dir runs/research_search
```

Then inspect:

```text
runs/research_search/search_results.csv
runs/research_search/best_search_candidate.json
```

Run the best candidate, for example:

```bash
python scripts/run_research_melshield_ljspeech.py \
  --config configs/research_melshield_hifigan.yaml \
  --alpha 0.04 \
  --repeat 3 \
  --freq-gamma 0.5 \
  --texture-gamma 0.25 \
  --smooth-frames 5 \
  --limit 100 \
  --output-dir runs/research_alpha_0.04_repeat3
```

The research CSV reports both `payload_bit_acc` and `code_bit_acc`.
`payload_bit_acc` is the main attribution metric after soft-ECC decoding;
`code_bit_acc` is the raw embedded-codeword accuracy before error correction.

## RelMelMark Prototype

`RelMelMark` is an alternate Mel-domain idea: instead of adding random
spread-spectrum noise, it encodes each bit as a keyed relative-energy difference
between two Mel-bin groups inside repeated time blocks. A bit value controls
which group is raised and which group is lowered, keeping each local update
approximately energy-balanced.

Run a small alpha sweep:

```bash
for a in 0.03 0.04 0.05 0.06 0.07 0.08; do
  python scripts/run_relmel_ljspeech.py \
    --config configs/relmel_hifigan.yaml \
    --alpha "$a" \
    --attacks none noise20 \
    --limit 40 \
    --output-dir "runs/relmel_alpha_${a}_quick"
done
```

Then run the best candidate on the full attack suite:

```bash
python scripts/run_relmel_ljspeech.py \
  --config configs/relmel_hifigan.yaml \
  --alpha 0.06 \
  --limit 100 \
  --output-dir runs/relmel_alpha_0.06_attacks
```

The main CSV fields are `bit_acc`, `verified`, `confidence`, `min_votes`, and
`mean_votes`. `min_votes` should be greater than zero; if it is zero for very
short audio, reduce `--block-frames` or increase `--bits-per-block`.

For a stronger RelMel search, use the consolidated audit suite. It starts from
the current best region around `alpha ~= 0.43`, `band = 20:60`,
`block_frames = 16`, `bits_per_block = 4`, `pair_bins = 4`, then scans alpha,
band, mask/boundary, block structure, energy weighting, and verifier settings:

```bash
python scripts/grid_relmel_ljspeech.py \
  --config configs/relmel_hifigan.yaml \
  --suite relmel-audit \
  --output-dir runs/relmel_audit \
  --limit 40 \
  --attacks none noise20 noise10 \
  --quality-floor 3.5 \
  --noise20-weight 1.0 \
  --noise10-weight 0.5 \
  --quality-weight 0.35
```

If several GPUs are free, shard the same candidate list:

```bash
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/grid_relmel_ljspeech.py \
    --config configs/relmel_hifigan.yaml \
    --suite relmel-audit \
    --output-dir "runs/relmel_audit_shard_${gpu}" \
    --limit 40 \
    --attacks none noise20 noise10 \
    --num-shards 4 \
    --shard-index "$gpu" &
done
wait

python scripts/merge_grid_results.py \
  --output-dir runs/relmel_audit_merged \
  runs/relmel_audit_shard_*/grid_results.csv
```

Send back `grid_results.csv` and `best_candidate.json` from the merged directory.
