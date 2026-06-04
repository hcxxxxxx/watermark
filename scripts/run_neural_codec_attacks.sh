#!/usr/bin/env bash
set -euo pipefail

LIMIT="${LIMIT:-500}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"
VOCODERS="${VOCODERS:-hifigan}"
LOG_DIR="${LOG_DIR:-runs/neural_codec_attack_logs}"
ATTACKS="${ATTACKS:-none encodec24 encodec12 encodec6 noise20 noise10 noise5 mp3 aac}"

read -r -a ATTACK_ARRAY <<< "$ATTACKS"
mkdir -p "$LOG_DIR"

python - <<'PY'
try:
    import encodec  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Missing optional package `encodec`. Install it in the melshield env with:\n"
        "  pip install encodec==0.1.1\n"
        "Then rerun this script. The first EnCodec run may also download model weights."
    ) from exc
PY

run_job() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name"
  /usr/bin/time -v "$@" > "$LOG_DIR/${name}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  $name"
}

run_hifigan() {
  run_job "relmel_hifigan_neural_codec_random${LIMIT}" \
    python scripts/grid_relmel_ljspeech.py \
      --config configs/relmel_hifigan.yaml \
      --output-dir "runs/relmel_hifigan_neural_codec_random${LIMIT}" \
      --suite custom \
      --limit "$LIMIT" \
      --sample-mode random \
      --seed "$SEED" \
      --attacks "${ATTACK_ARRAY[@]}" \
      --device "$DEVICE" \
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
      --quality-floor 3.5 \
      --noise20-weight 1.0 \
      --noise10-weight 0.5 \
      --noise5-weight 0.25 \
      --quality-weight 0.25 \
      --keep-candidate-results

  run_job "melshield_hifigan_neural_codec_random${LIMIT}" \
    python scripts/grid_melshield_ljspeech.py \
      --config configs/melshield_hifigan_official.yaml \
      --output-dir "runs/melshield_hifigan_neural_codec_random${LIMIT}" \
      --suite custom \
      --limit "$LIMIT" \
      --sample-mode random \
      --seed "$SEED" \
      --attacks "${ATTACK_ARRAY[@]}" \
      --device "$DEVICE" \
      --alpha-grid 0.05 \
      --band-grid 20:60 \
      --mask-floor-grid 0.05 \
      --energy-gamma-grid 0.75 \
      --boundary-margin-grid 0.02 \
      --threshold-grid 0.61 \
      --align-max-shift-grid 12 \
      --headroom-grid 0.0 \
      --quality-floor 3.5 \
      --noise20-weight 1.0 \
      --quality-weight 0.25 \
      --keep-candidate-results
}

run_diffwave() {
  local diffwave_cmd="python scripts/diffwave_vocoder.py external/diffwave/checkpoints/diffwave-ljspeech-22kHz-1000578.pt {mel_npy} --output {audio_wav} --device ${DEVICE} --fast --seed 0"

  run_job "relmel_diffwave_neural_codec_random${LIMIT}" \
    python scripts/grid_relmel_ljspeech.py \
      --config configs/relmel_diffwave.yaml \
      --output-dir "runs/relmel_diffwave_neural_codec_random${LIMIT}" \
      --suite custom \
      --limit "$LIMIT" \
      --sample-mode random \
      --seed "$SEED" \
      --attacks "${ATTACK_ARRAY[@]}" \
      --device "$DEVICE" \
      --vocoder command \
      --vocoder-command "$diffwave_cmd" \
      --alpha-grid 0.355 \
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
      --quality-floor 3.5 \
      --noise20-weight 1.0 \
      --noise10-weight 0.5 \
      --noise5-weight 0.25 \
      --quality-weight 0.25 \
      --keep-candidate-results

  run_job "melshield_diffwave_neural_codec_random${LIMIT}" \
    python scripts/grid_melshield_ljspeech.py \
      --config configs/melshield_diffwave.yaml \
      --output-dir "runs/melshield_diffwave_neural_codec_random${LIMIT}" \
      --suite custom \
      --limit "$LIMIT" \
      --sample-mode random \
      --seed "$SEED" \
      --attacks "${ATTACK_ARRAY[@]}" \
      --device "$DEVICE" \
      --vocoder command \
      --vocoder-command "$diffwave_cmd" \
      --alpha-grid 0.061 \
      --band-grid 20:60 \
      --mask-floor-grid 0.05 \
      --energy-gamma-grid 0.75 \
      --boundary-margin-grid 0.02 \
      --threshold-grid 0.61 \
      --align-max-shift-grid 12 \
      --headroom-grid 0.0 \
      --quality-floor 3.5 \
      --noise20-weight 1.0 \
      --quality-weight 0.25 \
      --keep-candidate-results
}

case "$VOCODERS" in
  hifigan)
    run_hifigan
    ;;
  diffwave)
    run_diffwave
    ;;
  both)
    run_hifigan
    run_diffwave
    ;;
  *)
    echo "VOCODERS must be one of: hifigan, diffwave, both" >&2
    exit 2
    ;;
esac

echo "All neural codec attack jobs finished."
