#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="training/configs/v1_corrected.yaml"
TCN_CHECKPOINT="artifacts/corrected-v1/checkpoints/tcn_best.pt"
GRU_CHECKPOINT="artifacts/corrected-v1/checkpoints/gru_baseline.pt"

echo "[corrected] legacy identity audit"
python3 -m training.audit \
  --mode legacy \
  --config "$CONFIG" \
  --output artifacts/corrected-v1/legacy_audit.json

echo "[corrected] source preflight"
python3 -m training.preflight --config "$CONFIG"

echo "[corrected] prepare isolated dataset"
python3 -m training.prepare_data --config "$CONFIG"

echo "[corrected] verify split, identity, and coverage constraints"
python3 -m training.audit \
  --mode prepared \
  --config "$CONFIG" \
  --output artifacts/corrected-v1/prepared_audit.json

echo "[corrected] two-epoch smoke training"
python3 -m training.train \
  --config "$CONFIG" \
  --models tcn,gru \
  --device auto \
  --seed 42 \
  --max-epochs 2

echo "[corrected] smoke evaluation: TCN"
python3 -m training.evaluate \
  --config "$CONFIG" \
  --checkpoint "$TCN_CHECKPOINT" \
  --device auto

echo "[corrected] smoke evaluation: GRU"
python3 -m training.evaluate \
  --config "$CONFIG" \
  --checkpoint "$GRU_CHECKPOINT" \
  --device auto

echo "[corrected] full training"
python3 -m training.train \
  --config "$CONFIG" \
  --models tcn,gru \
  --device auto \
  --seed 42

echo "[corrected] final evaluation: TCN"
python3 -m training.evaluate \
  --config "$CONFIG" \
  --checkpoint "$TCN_CHECKPOINT" \
  --device auto

echo "[corrected] final evaluation: GRU"
python3 -m training.evaluate \
  --config "$CONFIG" \
  --checkpoint "$GRU_CHECKPOINT" \
  --device auto

echo "[corrected] completed"
