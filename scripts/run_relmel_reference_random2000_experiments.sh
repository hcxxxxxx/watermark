#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-runs/paper_reference_random2000_logs}"
LIMIT="${LIMIT:-2000}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"

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

COMMON_SINGLE_ARGS=(
  --config configs/relmel_hifigan.yaml
  --limit "$LIMIT"
  --sample-mode random
  --seed "$SEED"
  --device "$DEVICE"
  --alpha 0.435
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

run_step E28_relmel_reference_compression_hifigan_band2060_random${LIMIT} \
  python scripts/eval_relmel_reference_compression_ljspeech.py \
    --output-dir "runs/relmel_reference_compression_hifigan_band2060_random${LIMIT}" \
    --attacks none noise20 noise10 noise5 \
    --reference-variants float32 band_uint8 uint4 \
    "${COMMON_SINGLE_ARGS[@]}"

run_step E29_relmel_blind_fragments_hifigan_band2060_random${LIMIT} \
  python scripts/eval_relmel_blind_fragments_ljspeech.py \
    --output-dir "runs/relmel_blind_fragments_hifigan_band2060_random${LIMIT}" \
    --fragments start25 middle25 end25 start50 middle50 end50 \
    --search-step-frames 4 \
    "${COMMON_SINGLE_ARGS[@]}"

run_step E30_relmel_reference_controls_hifigan_band2060_noise20_random${LIMIT} \
  python scripts/eval_relmel_reference_controls_ljspeech.py \
    --output-dir "runs/relmel_reference_controls_hifigan_band2060_noise20_random${LIMIT}" \
    --attacks none noise20 \
    --controls correct clean_unmarked wrong_key wrong_payload wrong_reference \
    "${COMMON_SINGLE_ARGS[@]}"

echo "[$(date '+%F %T')] ALL DONE"
