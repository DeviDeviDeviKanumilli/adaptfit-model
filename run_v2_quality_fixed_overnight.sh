#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="training/configs/v2_quality_fixed.yaml"
ARTIFACT_ROOT="artifacts/v2-quality-fixed"
TCN_CHECKPOINT="$ARTIFACT_ROOT/checkpoints/tcn_best.pt"
GRU_CHECKPOINT="$ARTIFACT_ROOT/checkpoints/gru_baseline.pt"
RUN_TRAINING="${ADAPTFIT_RUN_TRAINING:-0}"
RUN_SMOKE="${ADAPTFIT_RUN_SMOKE:-1}"
NUM_WORKERS="${ADAPTFIT_NUM_WORKERS:-2}"
FULL_PREFLIGHT="${ADAPTFIT_PREFLIGHT_FULL:-0}"
STRICT_STORAGE="${ADAPTFIT_STRICT_STORAGE:-0}"

mkdir -p "$ARTIFACT_ROOT"
LOG_PATH="$ARTIFACT_ROOT/overnight.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[fixed] legacy identity audit"
python3 -m training.audit \
  --mode legacy \
  --config training/configs/v1_corrected.yaml \
  --output "$ARTIFACT_ROOT/legacy_audit.json"

echo "[fixed] verify UCO source checksums"
./scripts/verify_ucophyrehabpp.sh

echo "[fixed] file-only source preflight"
python3 -m training.preflight --config "$CONFIG" --mode files

if [[ "$FULL_PREFLIGHT" == "1" ]]; then
  echo "[fixed] full source preflight"
  python3 -m training.preflight --config "$CONFIG" --mode full
fi

echo "[fixed] prepared-data audit"
AUDIT_ARGS=(--mode prepared --config "$CONFIG" --output "$ARTIFACT_ROOT/prepared_audit.json")
if [[ "$STRICT_STORAGE" == "1" ]]; then
  echo "[fixed] enabling strict memmap finite-value scan"
  AUDIT_ARGS+=(--strict-storage)
fi
python3 -m training.audit "${AUDIT_ARGS[@]}"

echo "[fixed] stage small metadata artifacts"
for required in \
  artifacts/v2-quality/feature_schema.json \
  artifacts/v2-quality/normalization_stats.npz; do
  if [[ ! -f "$required" ]]; then
    echo "missing required existing artifact: $required" >&2
    exit 2
  fi
done
cp artifacts/v2-quality/feature_schema.json "$ARTIFACT_ROOT/feature_schema.json"
cp artifacts/v2-quality/normalization_stats.npz "$ARTIFACT_ROOT/normalization_stats.npz"
cp "$CONFIG" "$ARTIFACT_ROOT/training_config.yaml"

echo "[fixed] optimized loader/model forward check"
python3 -m training.runtime_check \
  --config "$CONFIG" \
  --model tcn \
  --device auto \
  --num-workers "$NUM_WORKERS"

echo "[fixed] compile and run non-training contract tests"
python3 -m compileall -q training
python3 -m pytest -q \
  -k "not one_epoch_training and not training_explicitly_falls_back_when_legacy_metadata_is_missing"

if [[ "$RUN_TRAINING" != "1" ]]; then
  echo "[fixed] verification complete; training was not run"
  echo "[fixed] launch the full run with: ADAPTFIT_RUN_TRAINING=1 ./run_v2_quality_fixed_overnight.sh"
  exit 0
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "[fixed] two-epoch smoke training"
  python3 -m training.train \
    --config "$CONFIG" \
    --models tcn,gru \
    --device auto \
    --seed 42 \
    --num-workers "$NUM_WORKERS" \
    --max-epochs 2

  echo "[fixed] smoke evaluation: TCN"
  python3 -m training.evaluate \
    --config "$CONFIG" \
    --checkpoint "$TCN_CHECKPOINT" \
    --device auto \
    --num-workers "$NUM_WORKERS"

  echo "[fixed] smoke evaluation: GRU"
  python3 -m training.evaluate \
    --config "$CONFIG" \
    --checkpoint "$GRU_CHECKPOINT" \
    --device auto \
    --num-workers "$NUM_WORKERS"
fi

echo "[fixed] full TCN and GRU training"
python3 -m training.train \
  --config "$CONFIG" \
  --models tcn,gru \
  --device auto \
  --seed 42 \
  --num-workers "$NUM_WORKERS"

echo "[fixed] final evaluation: TCN"
python3 -m training.evaluate \
  --config "$CONFIG" \
  --checkpoint "$TCN_CHECKPOINT" \
  --device auto \
  --num-workers "$NUM_WORKERS"

echo "[fixed] final evaluation: GRU"
python3 -m training.evaluate \
  --config "$CONFIG" \
  --checkpoint "$GRU_CHECKPOINT" \
  --device auto \
  --num-workers "$NUM_WORKERS"

echo "[fixed] completed"
