# AdaptFit artifact registry

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** artifact paths, lineage, status, provenance, and metric evidence
> - **Last verified:** 2026-09-11
> - **Source commit:** `b3926d8` (R1 calibration and phase-weight correction training)
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
| `r1-tcn-boundary-finetune` | `artifacts/r1-tcn-boundary/` (local, ignored) | full training complete; decoder gate blocked; validation-only | `training/configs/experiments/r1_tcn_boundary_finetune.yaml`; parent `corrected-v1/tcn_best.pt` | MPS head-only run stopped at epoch 27 with best epoch 12; calibration report exists at `decoder/decoder_calibration.json` with end F1 `0.0`; no test evaluation |
| `r1-tcn-boundary-mask-correction` | `artifacts/r1-tcn-boundary-mask-correction/` (local, ignored) | training complete; validation-only decoder gate blocked; test locked | `training/configs/experiments/r1_tcn_boundary_mask_correction.yaml`; parent `corrected-v1/tcn_best.pt` | Completed epoch 68 with best epoch 53; corrected unavailable-boundary masking and regenerated isolated prepared root; best checkpoint SHA-256 `317bbbf782f5c69f2888737e8bdcec914c8d812e71001c4f8c718fe861479505`; calibration report records end F1 `0.0`, count MAE `0.2678571`, and 240 missed events; failure analysis records 0/240 qualified apex intervals and 178/219 segmentation starts below tracking floor; no test evaluation |
| `r1-tcn-phase-weight-correction` | `artifacts/r1-tcn-phase-weight-correction/` (local, ignored) | training complete; validation-only decoder gate blocked; test locked | `training/configs/experiments/r1_tcn_phase_weight_correction.yaml`; parent `r1-tcn-boundary-mask-correction/tcn_best.pt` | Absent-class phase-weight normalization corrected; reused the verified corrected-mask prepared root; MPS head-only run stopped at epoch 24 with best epoch 9; best checkpoint SHA-256 `2bac2156ae7ef85b39e65ce5e1d8eff9fe4ea5799ce6b28c2f5ffcf9d30ebf5a`; calibration report SHA-256 `a4df7614d2f2932ba4093352d20c19e83a355ac1cc9e4b4be98e213cfee18b40` records end F1 `0.153846`, count MAE `0.279018`, and 15 false events on empty sequences; no test evaluation |
| `r1-tcn-last-block-event-support` | `artifacts/r1-tcn-last-block-event-support/` (local, ignored) | training complete; validation-only decoder gate blocked; test locked | `training/configs/experiments/r1_tcn_last_block_event_support.yaml`; parent `r1-tcn-phase-weight-correction/tcn_best.pt` | Final-block representation intervention; best epoch 8 of 23; checkpoint SHA-256 `e1ac1ea978b323e128d9e966b067f8c5af073358e86846e8e7c046ec11a156de`; validation sequence score `0.6791891`; calibration report SHA-256 `d495be8ae07d80c7b0b215922b15998d6a44311cd08218148bf10ba2b3ebfaf1` records end F1 `0.094862`, count MAE `0.271205`, and 8 empty-target false events; failure analysis is recorded; no test evaluation |
| `r1-tcn-last-two-blocks-event-support` | `artifacts/r1-tcn-last-two-blocks-event-support/` (local, ignored) | training complete; validation-only decoder gate blocked; test locked | `training/configs/experiments/r1_tcn_last_two_blocks_event_support.yaml`; parent `r1-tcn-last-block-event-support/tcn_best.pt` | Final-two-block representation escalation; best epoch 42 of 57; checkpoint SHA-256 `18c2e3dcb27c64af76b73e8431df8f9eb59939cad9f156b16dc8f3fa24c16599`; validation sequence score `0.7040537`; calibration report SHA-256 `16b1ffb4cd62c9759b02964b26bf744e54bc2e35b56f4a6067d8b3f92712a1cc` records end F1 `0.299674`, count MAE `0.333705`, and 63 empty-target false events; failure analysis is recorded; no test evaluation |
| `r1-tcn-full-finetune-event-support-pilot` | `artifacts/r1-tcn-full-finetune-event-support-pilot/` (local, ignored) | training complete; validation-only decoder gate blocked; test locked | `training/configs/experiments/r1_tcn_full_finetune_event_support_pilot.yaml`; parent `r1-tcn-last-two-blocks-event-support/tcn_best.pt` | Short conservative full-TCN pilot; all 307,410 parameters trainable; best epoch 6 of 8; validation sequence score `0.717668`; checkpoint SHA-256 `1695ab70241e3f058164e82281583b89a899907636e4e43c1f2c94a5761e91b2`; calibration SHA-256 `5a37ff1b1e8c3965b622fba8b4aeb0a8a14fceb47d0aaaf849f8cce7d5fce3cd` records end F1 `0.296053`, count MAE `0.330357`, and 60 empty-target false events; failure-analysis JSON/Markdown SHA-256 `192bb944ff6c29e2af503e542774bae9e34e3521f54826ecf7bac0f8bb17e89b` / `cbe990c531dbb0e8f3912aa29d2a0a41ee3685cc5a91eaa58ade1a29abf70917`; no test evaluation |
| `r1-tcn-uco-boundary-support` | `artifacts/r1-tcn-uco-boundary-support/` (local, ignored) | training complete; validation-only decoder gate blocked; test locked | `training/configs/experiments/r1_tcn_uco_boundary_support.yaml`; parent `corrected-v1` through the explicit normalization-transfer path | UCO exact-boundary support run; best checkpoint SHA-256 `9c69b9aa8173da7f10140779ec70f87c012ae151519e3f2375e61b0dd09327a0`; no test evaluation |
| `r1-tcn-uco-tracking-target-correction` | `artifacts/r1-tcn-uco-tracking-target-correction/` (local, ignored) | training complete; validation-only decoder gate blocked; test locked | `training/configs/experiments/r1_tcn_uco_tracking_target_correction.yaml`; parent `r1-tcn-uco-boundary-support/tcn_best.pt` | Corrected partial-pose tracking denominator; best epoch 2 of 5; sequence score `0.6176060`; checkpoint SHA-256 `bc69117c1a287a27fcf63466238ab97da812eda3ff92e5a4db968f94ef80d20`; calibration SHA-256 `140595d5f986b97960132517fdaeba7f6511f03962f112eec47f061414534045` records end F1 `0.070130`, count MAE `0.584`, and 7 empty-target false events; failure-analysis JSON/Markdown SHA-256 `8f674a3cd4d43a131b4b93885c49dab2c6ca48b76b55fa1d493dbcb78b388b17` / `bb10ce9703ea6751f4e55dd8dc66535d97237ae25e2ccedf663e367cbeaabcac`; no test evaluation |
| `r1-tcn-uco-boundary-positive-weight` | `artifacts/r1-tcn-uco-boundary-positive-weight/` (local, ignored) | training complete; validation-only decoder gate blocked; test locked | `training/configs/experiments/r1_tcn_uco_boundary_positive_weight.yaml`; parent `r1-tcn-uco-tracking-target-correction/tcn_best.pt` | Boundary positive-weight cap `64.0`; best epoch 1 of 4; sequence score `0.5252875`; checkpoint SHA-256 `1e2bdcdcb6870ddaa215ce6d406c2ab0b01f95361fb0ce16a1528df00c0d3c52`; calibration SHA-256 `140a779da9c481bcb62f67350630fb312d52a2fdf364588124af265d95e4d28a` records end F1 `0.142322`, count MAE `0.5864`, and 24 empty-target false events; cap rejected; no test evaluation |
| `r1-tcn-ulred-boundary-support` | `artifacts/r1-tcn-ulred-boundary-support/` (local, ignored) | training complete; validation-only decoder gate blocked; test locked | `training/configs/experiments/r1_tcn_ulred_boundary_support.yaml`; parent `r1-tcn-uco-tracking-target-correction/tcn_best.pt` | Local UL-RED `marker-less/3Rep` boundary-source correction; isolated prepared root `data/processed-r1-ulred-boundary-support/` overlays 219 explicit R3 recordings; best epoch 2 of 5 with sequence score `0.6013545`; checkpoint SHA-256 `7ed351765ad1c943d525721c03fdcec65c1fdd06df2966c59971ca934b5d4ce1`; calibration SHA-256 `c7fb7b28e9d92f496d0b38c1cf3e44614129d5113ee809a06ddd47844fdedac1`; end F1 `0.070718`, count MAE `0.684`, and 8 empty-target false events; failure analysis recorded; no test evaluation |

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
