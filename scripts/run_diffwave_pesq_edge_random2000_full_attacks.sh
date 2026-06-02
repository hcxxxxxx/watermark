#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-runs/diffwave_pesq_edge_random2000_full_logs}"
LIMIT="${LIMIT:-2000}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"
DIFFWAVE_SEED="${DIFFWAVE_SEED:-0}"
DIFFWAVE_CKPT="${DIFFWAVE_CKPT:-external/diffwave/checkpoints/diffwave-ljspeech-22kHz-1000578.pt}"

mkdir -p "$LOG_DIR"

if [[ -x /usr/bin/time ]]; then
  TIME_BIN=(/usr/bin/time -v)
else
  TIME_BIN=(time)
fi

run_step() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] START ${name}"
  echo "command: $*" > "$log_file"
  echo >> "$log_file"
  "${TIME_BIN[@]}" "$@" >> "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE  ${name}"
}

ATTACKS=(none mp3 aac scale rs16 bandpass lowpass noise20 noise10 noise5 echo)
DIFFWAVE_COMMAND="python scripts/diffwave_vocoder.py ${DIFFWAVE_CKPT} {mel_npy} --output {audio_wav} --device ${DEVICE} --fast --seed ${DIFFWAVE_SEED}"

COMMON_GRID_ARGS=(
  --suite custom
  --limit "$LIMIT"
  --sample-mode random
  --seed "$SEED"
  --attacks "${ATTACKS[@]}"
  --device "$DEVICE"
  --vocoder command
  --vocoder-command "$DIFFWAVE_COMMAND"
  --quality-floor 3.5
  --quality-weight 0.25
  --keep-candidate-results
)

RELMEL_GRID_ARGS=(
  "${COMMON_GRID_ARGS[@]}"
  --noise20-weight 1.0
  --noise10-weight 0.5
  --noise5-weight 0.25
  --echo-weight 0.25
)

MELSHIELD_GRID_ARGS=(
  "${COMMON_GRID_ARGS[@]}"
  --noise20-weight 1.0
)

run_step "relmel_diffwave_a0358_mf020_bm001_full_attacks_random${LIMIT}" \
  python scripts/grid_relmel_ljspeech.py \
    --config configs/relmel_diffwave.yaml \
    --output-dir "runs/relmel_diffwave_a0358_mf020_bm001_full_attacks_random${LIMIT}" \
    "${RELMEL_GRID_ARGS[@]}" \
    --alpha-grid 0.358 \
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
    --energy-gamma-grid 0.5

run_step "melshield_diffwave_a0061_mf005_bm002_full_attacks_random${LIMIT}" \
  python scripts/grid_melshield_ljspeech.py \
    --config configs/melshield_diffwave.yaml \
    --output-dir "runs/melshield_diffwave_a0061_mf005_bm002_full_attacks_random${LIMIT}" \
    "${MELSHIELD_GRID_ARGS[@]}" \
    --alpha-grid 0.061 \
    --band-grid 20:60 \
    --mask-floor-grid 0.05 \
    --energy-gamma-grid 0.75 \
    --boundary-margin-grid 0.02 \
    --threshold-grid 0.61 \
    --align-max-shift-grid 12 \
    --headroom-grid 0.0

echo "[$(date '+%F %T')] ALL DONE"
