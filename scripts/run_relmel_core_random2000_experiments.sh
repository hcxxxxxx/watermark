#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-runs/paper_core_random2000_logs}"
MAIN_LIMIT="${MAIN_LIMIT:-2000}"
HIFIGAN_MELSHIELD_LIMIT="${HIFIGAN_MELSHIELD_LIMIT:-1000}"
SEED="${SEED:-2026}"
DIFFWAVE_SEED="${DIFFWAVE_SEED:-0}"
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

ATTACKS=(none mp3 aac scale rs16 bandpass lowpass noise20 noise10 noise5 echo)

COMMON_RELMEL_GRID_ARGS=(
  --suite custom
  --sample-mode random
  --seed "$SEED"
  --attacks "${ATTACKS[@]}"
  --device "$DEVICE"
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
  --keep-candidate-results
)

COMMON_MELSHIELD_GRID_ARGS=(
  --suite custom
  --sample-mode random
  --seed "$SEED"
  --attacks "${ATTACKS[@]}"
  --device "$DEVICE"
  --mask-floor-grid 0.05
  --energy-gamma-grid 0.75
  --boundary-margin-grid 0.02
  --threshold-grid 0.61
  --align-max-shift-grid 12
  --headroom-grid 0.0
  --keep-candidate-results
)

run_step E24_relmel_hifigan_attacks_random${MAIN_LIMIT} \
  python scripts/grid_relmel_ljspeech.py \
    --config configs/relmel_hifigan.yaml \
    --output-dir "runs/relmel_hifigan_a0435_mf020_bm001_attacks_random${MAIN_LIMIT}" \
    --limit "$MAIN_LIMIT" \
    --alpha-grid 0.435 \
    "${COMMON_RELMEL_GRID_ARGS[@]}"

run_step E25_relmel_diffwave_attacks_random${MAIN_LIMIT} \
  python scripts/grid_relmel_ljspeech.py \
    --config configs/relmel_diffwave.yaml \
    --output-dir "runs/relmel_diffwave_a035_mf020_bm001_attacks_random${MAIN_LIMIT}" \
    --limit "$MAIN_LIMIT" \
    --alpha-grid 0.35 \
    "${COMMON_RELMEL_GRID_ARGS[@]}" \
    --vocoder command \
    --vocoder-command "python scripts/diffwave_vocoder.py external/diffwave/checkpoints/diffwave-ljspeech-22kHz-1000578.pt {mel_npy} --output {audio_wav} --device ${DEVICE} --fast --seed ${DIFFWAVE_SEED}"

run_step E26_melshield_diffwave_quality_matched_attacks_random${MAIN_LIMIT} \
  python scripts/grid_melshield_ljspeech.py \
    --config configs/melshield_diffwave.yaml \
    --output-dir "runs/melshield_diffwave_a006_mf005_bm002_band2060_attacks_random${MAIN_LIMIT}" \
    --limit "$MAIN_LIMIT" \
    "${COMMON_MELSHIELD_GRID_ARGS[@]}" \
    --alpha-grid 0.060 \
    --band-grid 20:60 \
    --vocoder command \
    --vocoder-command "python scripts/diffwave_vocoder.py external/diffwave/checkpoints/diffwave-ljspeech-22kHz-1000578.pt {mel_npy} --output {audio_wav} --device ${DEVICE} --fast --seed ${DIFFWAVE_SEED}"

run_step E27_melshield_hifigan_quality_matched_attacks_random${HIFIGAN_MELSHIELD_LIMIT} \
  python scripts/grid_melshield_ljspeech.py \
    --config configs/melshield_hifigan_official.yaml \
    --output-dir "runs/melshield_hifigan_a005_mf005_bm002_band2060_attacks_random${HIFIGAN_MELSHIELD_LIMIT}" \
    --limit "$HIFIGAN_MELSHIELD_LIMIT" \
    "${COMMON_MELSHIELD_GRID_ARGS[@]}" \
    --alpha-grid 0.050 \
    --band-grid 20:60 \
    --vocoder hifigan

echo "[$(date '+%F %T')] ALL DONE"
