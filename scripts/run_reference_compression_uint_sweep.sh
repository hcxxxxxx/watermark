#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-runs/reference_compression_uint_sweep_logs}"
LIMIT="${LIMIT:-500}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"
RUN_HIFIGAN="${RUN_HIFIGAN:-1}"
RUN_DIFFWAVE="${RUN_DIFFWAVE:-0}"
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

VARIANTS=(float32 band_uint8 uint4 uint2 band_uint4 band_uint2)
ATTACKS=(none noise20 noise10 noise5)

RELMEL_COMMON=(
  --limit "$LIMIT"
  --sample-mode random
  --seed "$SEED"
  --device "$DEVICE"
  --attacks "${ATTACKS[@]}"
  --reference-variants "${VARIANTS[@]}"
  --band 20:60
  --threshold 0.75
  --block-frames 8
  --block-stride 8
  --min-block-frames 8
  --bits-per-block 6
  --pair-bins 6
  --pair-candidates 16
  --detector-mode plain
  --mask-floor 0.20
  --energy-gamma 0.5
  --boundary-margin 0.01
)

MELSHIELD_COMMON=(
  --limit "$LIMIT"
  --sample-mode random
  --seed "$SEED"
  --device "$DEVICE"
  --attacks "${ATTACKS[@]}"
  --reference-variants "${VARIANTS[@]}"
  --band 20:60
  --threshold 0.61
  --mask-floor 0.05
  --energy-gamma 0.75
  --boundary-margin 0.02
  --align-max-shift 12
  --headroom 0.0
)

if [[ "$RUN_HIFIGAN" == "1" ]]; then
  run_step "relmel_hifigan_reference_uint_sweep_random${LIMIT}" \
    python scripts/eval_relmel_reference_compression_ljspeech.py \
      --config configs/relmel_hifigan.yaml \
      --output-dir "runs/relmel_hifigan_reference_uint_sweep_random${LIMIT}" \
      --alpha 0.435 \
      "${RELMEL_COMMON[@]}"

  run_step "melshield_hifigan_reference_uint_sweep_random${LIMIT}" \
    python scripts/eval_melshield_reference_compression_ljspeech.py \
      --config configs/melshield_hifigan_official.yaml \
      --output-dir "runs/melshield_hifigan_reference_uint_sweep_random${LIMIT}" \
      --alpha 0.05 \
      "${MELSHIELD_COMMON[@]}"
fi

if [[ "$RUN_DIFFWAVE" == "1" ]]; then
  DIFFWAVE_COMMAND="python scripts/diffwave_vocoder.py ${DIFFWAVE_CKPT} {mel_npy} --output {audio_wav} --device ${DEVICE} --fast --seed 0"

  run_step "relmel_diffwave_reference_uint_sweep_random${LIMIT}" \
    python scripts/eval_relmel_reference_compression_ljspeech.py \
      --config configs/relmel_diffwave.yaml \
      --output-dir "runs/relmel_diffwave_reference_uint_sweep_random${LIMIT}" \
      --vocoder command \
      --vocoder-command "$DIFFWAVE_COMMAND" \
      --alpha 0.35 \
      "${RELMEL_COMMON[@]}"

  run_step "melshield_diffwave_reference_uint_sweep_random${LIMIT}" \
    python scripts/eval_melshield_reference_compression_ljspeech.py \
      --config configs/melshield_diffwave.yaml \
      --output-dir "runs/melshield_diffwave_reference_uint_sweep_random${LIMIT}" \
      --vocoder command \
      --vocoder-command "$DIFFWAVE_COMMAND" \
      --alpha 0.06 \
      "${MELSHIELD_COMMON[@]}"
fi

echo "[$(date '+%F %T')] ALL DONE"
