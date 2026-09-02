#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

python3 -m training.preflight --config training/configs/v1.yaml
python3 -m training.prepare_data --config training/configs/v1.yaml
python3 -m training.train \
  --config training/configs/v1.yaml \
  --models tcn,gru \
  --device auto \
  --seed 42
python3 -m training.evaluate \
  --config training/configs/v1.yaml \
  --checkpoint artifacts/checkpoints/tcn_best.pt \
  --device auto

