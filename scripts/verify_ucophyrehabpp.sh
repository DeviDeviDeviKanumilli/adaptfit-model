#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/raw/ucophyrehabpp"

verify_file() {
  local expected="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing UCO file: $path" >&2
    exit 2
  fi
  local actual
  if command -v md5 >/dev/null 2>&1; then
    actual="$(md5 -q "$path")"
  elif command -v md5sum >/dev/null 2>&1; then
    actual="$(md5sum "$path" | awk '{print $1}')"
  else
    echo "neither md5 nor md5sum is available" >&2
    exit 2
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "checksum mismatch: $path expected=$expected actual=$actual" >&2
    exit 2
  fi
  echo "verified $(basename "$path")"
}

verify_file \
  cd67039d464660600a564cfef4b25742 \
  "$DATA_DIR/ucophyrehab2_data.jsonl"
verify_file \
  73026a0b44ada0603a6de63dd70f9fd4 \
  "$DATA_DIR/dataset_3d_with_angles.json"
