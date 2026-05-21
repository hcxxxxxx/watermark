#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-relmel-stage1}"
CHECKPOINT="${DIFFWAVE_CHECKPOINT:-external/diffwave/checkpoints/diffwave-ljspeech-22kHz-1000578.pt}"
LIMIT="${LIMIT:-60}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"
DIFFWAVE_SEED="${DIFFWAVE_SEED:-0}"
MS_ALPHA="${MS_ALPHA:-0.025}"
MS_BAND="${MS_BAND:-20:56}"
MS_MASK_FLOOR="${MS_MASK_FLOOR:-0.05}"
MS_ENERGY_GAMMA="${MS_ENERGY_GAMMA:-0.75}"
MS_BOUNDARY_MARGIN="${MS_BOUNDARY_MARGIN:-0.02}"
MS_THRESHOLD="${MS_THRESHOLD:-0.61}"
MS_ALIGN_MAX_SHIFT="${MS_ALIGN_MAX_SHIFT:-12}"
MS_HEADROOM="${MS_HEADROOM:-0.0}"

DIFFWAVE_CMD="python scripts/diffwave_vocoder.py ${CHECKPOINT} {mel_npy} --output {audio_wav} --device ${DEVICE} --fast --seed ${DIFFWAVE_SEED}"

case "${MODE}" in
  smoke)
    python scripts/run_relmel_ljspeech.py \
      --config configs/relmel_diffwave.yaml \
      --output-dir runs/relmel_diffwave_smoke \
      --limit "${LIMIT}" \
      --sample-mode random \
      --seed "${SEED}" \
      --attacks none noise20 \
      --device "${DEVICE}" \
      --vocoder command \
      --vocoder-command "${DIFFWAVE_CMD}" \
      --save-audio
    ;;

  relmel-stage1)
    python scripts/grid_relmel_ljspeech.py \
      --config configs/relmel_diffwave.yaml \
      --suite custom \
      --output-dir "runs/relmel_diffwave_stage1_random${LIMIT}_seed${SEED}" \
      --limit "${LIMIT}" \
      --sample-mode random \
      --seed "${SEED}" \
      --attacks none noise20 noise10 noise5 \
      --device "${DEVICE}" \
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
    ;;

  relmel-full)
    python scripts/grid_relmel_ljspeech.py \
      --config configs/relmel_diffwave.yaml \
      --suite custom \
      --output-dir "runs/relmel_diffwave_pc16_full_random${LIMIT}_seed${SEED}" \
      --limit "${LIMIT}" \
      --sample-mode random \
      --seed "${SEED}" \
      --attacks none mp3 aac scale rs16 bandpass lowpass noise20 noise10 noise5 echo \
      --device "${DEVICE}" \
      --vocoder command \
      --vocoder-command "${DIFFWAVE_CMD}" \
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
    ;;

  relmel-pair-ablation)
    python scripts/grid_relmel_ljspeech.py \
      --config configs/relmel_diffwave.yaml \
      --suite custom \
      --output-dir "runs/relmel_diffwave_pair_candidates_ablation_random${LIMIT}_seed${SEED}" \
      --limit "${LIMIT}" \
      --sample-mode random \
      --seed "${SEED}" \
      --attacks none noise20 noise10 noise5 \
      --device "${DEVICE}" \
      --vocoder command \
      --vocoder-command "${DIFFWAVE_CMD}" \
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
    ;;

  melshield-stage1|melshield-baseline)
    python scripts/grid_melshield_ljspeech.py \
      --config configs/melshield_diffwave.yaml \
      --suite custom \
      --output-dir "runs/melshield_diffwave_stage1_random${LIMIT}_seed${SEED}" \
      --limit "${LIMIT}" \
      --sample-mode random \
      --seed "${SEED}" \
      --attacks none noise20 noise10 noise5 \
      --device "${DEVICE}" \
      --vocoder command \
      --vocoder-command "${DIFFWAVE_CMD}" \
      --alpha-grid 0.015 0.020 0.025 0.030 0.035 0.040 0.045 0.050 \
      --band-grid 20:56 20:60 \
      --mask-floor-grid 0.05 \
      --energy-gamma-grid 0.75 \
      --boundary-margin-grid 0.02 \
      --threshold-grid 0.61 \
      --headroom-grid 0.0 \
      --quality-floor 3.5 \
      --noise20-weight 1.0 \
      --quality-weight 2.5 \
      --keep-candidate-results
    ;;

  melshield-full)
    python scripts/grid_melshield_ljspeech.py \
      --config configs/melshield_diffwave.yaml \
      --suite custom \
      --output-dir "runs/melshield_diffwave_a${MS_ALPHA}_band${MS_BAND/:/-}_full_random${LIMIT}_seed${SEED}" \
      --limit "${LIMIT}" \
      --sample-mode random \
      --seed "${SEED}" \
      --attacks none mp3 aac scale rs16 bandpass lowpass noise20 noise10 noise5 echo \
      --device "${DEVICE}" \
      --vocoder command \
      --vocoder-command "${DIFFWAVE_CMD}" \
      --alpha-grid "${MS_ALPHA}" \
      --band-grid "${MS_BAND}" \
      --mask-floor-grid "${MS_MASK_FLOOR}" \
      --energy-gamma-grid "${MS_ENERGY_GAMMA}" \
      --boundary-margin-grid "${MS_BOUNDARY_MARGIN}" \
      --threshold-grid "${MS_THRESHOLD}" \
      --align-max-shift-grid "${MS_ALIGN_MAX_SHIFT}" \
      --headroom-grid "${MS_HEADROOM}" \
      --keep-candidate-results
    ;;

  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Modes: smoke, relmel-stage1, relmel-full, relmel-pair-ablation, melshield-stage1, melshield-baseline, melshield-full" >&2
    exit 2
    ;;
esac
