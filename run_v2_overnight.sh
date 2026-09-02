#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="training/configs/v2_quality.yaml"
ARTIFACT_ROOT="artifacts/v2-quality"
TCN_CHECKPOINT="$ARTIFACT_ROOT/checkpoints/tcn_best.pt"
GRU_CHECKPOINT="$ARTIFACT_ROOT/checkpoints/gru_baseline.pt"
RUN_TRAINING="${ADAPTFIT_RUN_TRAINING:-1}"
RUN_SMOKE="${ADAPTFIT_RUN_SMOKE:-1}"
SKIP_PREPARE="${ADAPTFIT_SKIP_PREPARE:-0}"

echo "[v2] legacy identity audit"
python3 -m training.audit \
  --mode legacy \
  --config training/configs/v1_corrected.yaml \
  --output "$ARTIFACT_ROOT/legacy_audit.json"

echo "[v2] verify UCO source checksums"
./scripts/verify_ucophyrehabpp.sh

echo "[v2] source preflight"
python3 -m training.preflight --config "$CONFIG"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  echo "[v2] prepare isolated quality dataset"
  python3 -m training.prepare_data --config "$CONFIG"
else
  echo "[v2] using existing prepared dataset"
fi

echo "[v2] verify prepared identities, participant splits, and coverage"
python3 -m training.audit \
  --mode prepared \
  --config "$CONFIG" \
  --output "$ARTIFACT_ROOT/prepared_audit.json"

echo "[v2] compile and run non-training contract tests"
python3 -m compileall -q training
python3 -m pytest -q \
  -k "not one_epoch_training and not training_explicitly_falls_back_when_legacy_metadata_is_missing"

if [[ "$RUN_TRAINING" != "1" ]]; then
  echo "[v2] preparation and verification complete; training was intentionally skipped"
  echo "[v2] run overnight training with: ./run_v2_overnight.sh"
  exit 0
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "[v2] two-epoch smoke training"
  python3 -m training.train \
    --config "$CONFIG" \
    --models tcn,gru \
    --device auto \
    --seed 42 \
    --max-epochs 2

  echo "[v2] smoke evaluation: TCN"
  python3 -m training.evaluate \
    --config "$CONFIG" \
    --checkpoint "$TCN_CHECKPOINT" \
    --device auto

  echo "[v2] smoke evaluation: GRU"
  python3 -m training.evaluate \
    --config "$CONFIG" \
    --checkpoint "$GRU_CHECKPOINT" \
    --device auto
fi

echo "[v2] full TCN and GRU training"
python3 -m training.train \
  --config "$CONFIG" \
  --models tcn,gru \
  --device auto \
  --seed 42

echo "[v2] final evaluation: TCN"
python3 -m training.evaluate \
  --config "$CONFIG" \
  --checkpoint "$TCN_CHECKPOINT" \
  --device auto

echo "[v2] final evaluation: GRU"
python3 -m training.evaluate \
  --config "$CONFIG" \
  --checkpoint "$GRU_CHECKPOINT" \
  --device auto

echo "[v2] completed"
