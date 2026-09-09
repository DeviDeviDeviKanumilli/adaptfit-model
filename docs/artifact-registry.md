# AdaptFit artifact registry

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** artifact paths, lineage, status, provenance, and metric evidence
> - **Last verified:** 2026-09-09
> - **Source commit:** `d7848e2` (R1 full-run evidence; smoke and pre-training gate history)
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
| `r0-baseline` | `artifacts/r0-baseline/` (local, ignored) | evaluation-only audit | `training/configs/experiments/r0_baseline.yaml`; parent `corrected-v1` | Fresh TCN/GRU reports and manifests plus validation-only decoder calibration; no trained child checkpoint |
| `r1-tcn-boundary-finetune` | `artifacts/r1-tcn-boundary/` (local, ignored) | full training complete; validation-only | `training/configs/experiments/r1_tcn_boundary_finetune.yaml`; parent `corrected-v1/tcn_best.pt` | MPS head-only run stopped at epoch 27 with best epoch 12; no test evaluation; decoder calibration and candidate release gates remain pending |

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
- validation-only decoder calibration report and explicit gate status;
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

## Research teacher and recommender artifacts

The methodology reuse work does not make a teacher or recommender artifact current.
When one is run, register it separately and preserve the parent baseline.

A teacher cache or density/saliency artifact must record:

- research method and exact source repository commit;
- code, weights, annotation, and dataset licenses separately;
- source sequence, participant/session identity, and participant/source split;
- canonical joint mapping, frame rate, timestamps, observed/capability masks, and
  augmentation or masking lineage;
- teacher/config/checkpoint hashes and generation commit;
- target type, horizon, density normalization, saliency convention, and valid mask;
- cache path, size, checksum, and expiration/invalidation rule;
- compute device, wall time, memory, and known limitations;
- matched student experiment, metrics, subgroup results, and rejection reason.

A recommendation experiment must additionally record catalog hash, profile/schema versions,
exposure policy, consent state, candidate-set size, shown candidates, feedback reason
codes, user/time split, and fallback behavior. Raw frames and raw pose are prohibited.

External code or weights with missing license, checksum, or provenance remain
research references and cannot be included in a model bundle.
