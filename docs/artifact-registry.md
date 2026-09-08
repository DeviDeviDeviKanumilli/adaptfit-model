# AdaptFit artifact registry

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** artifact paths, lineage, status, provenance, and metric evidence
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `1a46f38` (documentation revision base)
> - **Owner:** AdaptFit training and release engineering
> - **Supersedes or supports:** consolidates artifact status from `current-state.md`, `training-execution-log.md`, and historical benchmark reports
> - **Review trigger:** new run, checkpoint, manifest, evaluation, export, or path change

An artifact is evidence only when its path, source commit, configuration,
manifest, schema, and interpretation limit are recorded together. A filename or
training log alone does not establish a release candidate.

## Current inventory

| Registry ID | Path | Status | Parent/config | Evidence and limit |
|---|---|---|---|---|
| `corrected-v1` | `artifacts/corrected-v1/` | complete benchmark | `training/configs/v1_corrected.yaml` | TCN and GRU trained/evaluated on the corrected participant split; current supervised baseline |
| `v2-quality` | `artifacts/v2-quality/` | prepared only | `training/configs/v2_quality.yaml` | prepared data/manifests; no completed model comparison |
| `v2-quality-fixed` | `artifacts/v2-quality-fixed/` | partial | `training/configs/v2_quality_fixed.yaml` | TCN evaluated; GRU interrupted; quality-4 target coverage remains zero |
| `v2-diagnostics` | `artifacts-v2-diagnostics/` | diagnostic variant | `training/configs/v2_diagnostics.yaml` | 378-feature diagnostic schema; not a replacement for the 283-feature contract |

The current-state and training execution log remain the human-readable snapshot;
this table is the canonical inventory to extend after every run.

## Required manifest fields

Each run/checkpoint must record `ModelArtifactManifestV1` fields:

- registry/model ID and lifecycle status;
- source commit, config hash, dataset manifest IDs, and license/access state;
- participant/source split, subject/session counts, and effective labeled samples;
- feature/model/label/normalization/decoder versions;
- seed, parent checkpoint, trainable layers, optimizer, loss weights, and RNG;
- checkpoint/config/schema/normalization hashes and artifact paths;
- validation and test metrics with split, source, participant scope, and selection
  rule;
- export/runtime/quantization state and parity-fixture results;
- known limitations, unsupported claims, and reviewer/date.

## Lineage rules

Warm-start children point to an immutable parent checkpoint. Exact resume also
records optimizer, scheduler, scaler, sampler, and RNG state. A new data,
feature, label, decoder, or normalization contract creates a new registry ID;
it must not overwrite the parent directory. Missing or unverified fields are
`unavailable`, not guessed from filenames.

## Metric evidence

Published metrics must link to the evaluation artifact and state whether they
are sequence-level primary metrics or window diagnostics. They must name the
participant/source split, commit, config/schema versions, checkpoint, and
label coverage. Zero quality-head coverage remains a valid recorded result;
quality claims are blocked until a manifest proves reviewed targets.
