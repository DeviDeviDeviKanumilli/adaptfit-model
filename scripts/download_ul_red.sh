#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${project_root}/data/raw/ul_red"
mkdir -p "${output_dir}"

for subject_number in 1 2 3 4 5 6 7 8 9 10; do
  subject_id="S$(printf '%02d' "${subject_number}")"
  output_path="${output_dir}/${subject_id}.zip"
  source_url="https://datacat.liverpool.ac.uk/2729/${subject_number}/${subject_id}.zip"

  if [[ -f "${output_path}" ]] && unzip -tq "${output_path}" >/dev/null 2>&1; then
    printf '%s already passes ZIP validation\n' "${output_path}"
    continue
  fi

  curl --fail --location --retry 2 --continue-at - \
    "${source_url}" --output "${output_path}"
done

printf 'Downloaded archives are recorded in %s\n' \
  "${project_root}/data/manifests/ul_red.sha256"
