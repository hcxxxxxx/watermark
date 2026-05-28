#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-runs/paper_priority_logs}"
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

COMMON_REL_ARGS=(
  --alpha-grid 0.435
  --band-grid 20:60
  --mask-floor-grid 0.20
  --boundary-margin-grid 0.01
  --block-frames-grid 8
  --block-stride-grid 8
  --min-block-frames-grid 8
  --bits-per-block-grid 6
  --pair-bins-grid 6
  --pair-candidates-grid 16
  --detector-mode-grid plain
  --energy-gamma-grid 0.5
)

COMMON_SINGLE_ARGS=(
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

run_step E21_relmel_hifigan_speed_pitch_curve_random500 \
  python scripts/grid_relmel_ljspeech.py \
    --config configs/relmel_hifigan.yaml \
    --suite custom \
    --output-dir runs/relmel_hifigan_speed_pitch_curve_random500 \
    --limit 500 \
    --sample-mode random \
    --seed 2026 \
    --attacks none speed097 speed103 speed095 speed105 speed090 speed110 pitch_down25 pitch_up25 pitch_down50 pitch_up50 pitch_down pitch_up \
    "${COMMON_REL_ARGS[@]}" \
    --keep-candidate-results

run_step E22_relmel_hifigan_blind_fragments_random500 \
  python scripts/eval_relmel_blind_fragments_ljspeech.py \
    --config configs/relmel_hifigan.yaml \
    --output-dir runs/relmel_blind_fragments_hifigan_band2060_random500 \
    --limit 500 \
    --sample-mode random \
    --seed 2026 \
    --fragments start25 middle25 end25 start50 middle50 end50 \
    --search-step-frames 4 \
    "${COMMON_SINGLE_ARGS[@]}"

run_step E23_relmel_hifigan_false_positive_random2000 \
  python scripts/eval_relmel_reference_controls_ljspeech.py \
    --config configs/relmel_hifigan.yaml \
    --output-dir runs/relmel_reference_controls_hifigan_band2060_random2000 \
    --limit 2000 \
    --sample-mode random \
    --seed 2026 \
    --attacks none \
    --controls correct clean_unmarked wrong_key wrong_payload wrong_reference \
    "${COMMON_SINGLE_ARGS[@]}"

echo "[$(date '+%F %T')] ALL DONE"
