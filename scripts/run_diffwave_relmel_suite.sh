#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-external/diffwave/checkpoints/diffwave-ljspeech-22kHz-1000578.pt}"
LIMIT="${2:-60}"
SEED="${3:-2026}"
DIFFWAVE_SEED="${DIFFWAVE_SEED:-0}"

DIFFWAVE_CMD="python scripts/diffwave_vocoder.py ${CHECKPOINT} {mel_npy} --output {audio_wav} --fast --seed ${DIFFWAVE_SEED}"

python scripts/grid_relmel_ljspeech.py \
  --config configs/relmel_diffwave.yaml \
  --suite custom \
  --output-dir "runs/relmel_diffwave_pc16_random${LIMIT}_seed${SEED}" \
  --limit "${LIMIT}" \
  --sample-mode random \
  --seed "${SEED}" \
  --attacks none noise20 noise10 noise5 \
  --vocoder command \
  --vocoder-command "${DIFFWAVE_CMD}" \
  --alpha-grid 0.30 0.35 0.40 0.435 \
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
  --quality-floor 3.0 \
  --noise20-weight 1.0 \
  --noise10-weight 0.8 \
  --noise5-weight 1.5 \
  --quality-weight 2.5 \
  --keep-candidate-results
