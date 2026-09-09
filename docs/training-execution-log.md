# AdaptFit Training Execution Log

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** training execution ledger, checkpoint provenance, and audit record
> - **Last verified:** 2026-09-08
> - **Source commit:** `847fd5d` (code and pre-training gate baseline) / `613ff12` (documentation revision base)
> - **Owner:** AdaptFit training engineering
> - **Supersedes or supports:** satisfies the execution log requirement from [documentation-validation.md](documentation-validation.md) and [efficient-training-strategy.md](efficient-training-strategy.md)
> - **Review trigger:** after every experiment run, checkpoint evaluation, data preparation, or task transition

This document is the operational execution log and run inventory for AdaptFit.
It records verified checkouts, candidate checkpoint status, evaluation findings,
known blockers, and task progression following the Luna execution checklist in
[efficient-training-strategy.md](efficient-training-strategy.md).

The normalized checkpoint inventory is also recorded in the [artifact
registry](artifact-registry.md). Add a manifest entry after every run; do not
use this log to promote a partial checkpoint or overwrite a historical result.

## 2A. Pre-training gate verification (2026-09-08 working tree)

This gate was completed without running a new model-training job. The detailed
handoff is [pretraining-readiness.md](pretraining-readiness.md).

| Gate | Result | Evidence |
|---|---|---|
| Immutable corrected-v1 hashes | **PASS** | TCN `e69ff69d…`, GRU `4105bf23…`; no files under `artifacts/corrected-v1/` changed |
| Strict source/config preflight | **PASS** | `r0_baseline.yaml` and `r1_tcn_boundary_finetune.yaml`; enabled sources decode successfully; optional quality sources are disabled |
| R0 test audit | **PASS (audit only)** | `artifacts/r0-baseline/metrics/*_evaluation.json` and model manifests; test was read, never used for calibration |
| Validation-only decoder calibration | **BLOCKED** | `artifacts/r0-baseline/decoder/decoder_calibration.json`; count MAE `0.3649554`, end F1 `0.0`, false empty-sequence events `0` |
| Warm-start/freeze/resume controls | **PASS (code/tests)** | `training/src/runner.py`, `training/train.py`, staged config, and staged-training tests; no R1 training run yet |
| Next eligible action | **R1 smoke training** | `training/configs/experiments/r1_tcn_boundary_finetune.yaml`, max two epochs, `--skip-test` |

The decoder block is a measured baseline limitation. It does not authorize
changing the test split, relaxing label masks, or presenting the decoder as a
release component. Training is the next computational action because R1 is the
declared experiment for improving the boundary signal; product release remains
blocked until the decoder gate is rechecked and all other release gates pass.

---

## 1. Candidate Checkpoint & Run Inventory (Task A1)

| Run / Artifact Root | Config | Checkpoint | Training Status | Labeled Evaluation Status | Model Architecture & Params | Split & Sources |
|---|---|---|---|---|---|---|
| `artifacts/corrected-v1/` | `training/configs/v1_corrected.yaml` | `tcn_best.pt` (epoch 35)<br>`gru_baseline.pt` (epoch 34) | **Training complete** (50 / 49 epochs) | **Evaluated** (Sequence & Window)<br>Family Acc: 91.25% (TCN) / 89.47% (GRU)<br>Start F1: 99.54% (TCN) / 83.50% (GRU)<br>End F1: 18.47% (TCN) / 5.53% (GRU)<br>Rep Count MAE: 0.31 (TCN) / 30.63 (GRU) | Causal TCN (307,410)<br>Causal GRU (68,178)<br>283 inputs, 128 frames | 50 train / 11 val / 11 test participant groups.<br>Zero identity collisions.<br>Sources: REHAB24-6, IntelliRehabDS, MM-Fit, UL-RED, procedural seed. |
| `artifacts/v2-quality/` | `training/configs/v2_quality.yaml` | None | **Prepared data only** | **Not trained / not evaluated** | Target: Causal TCN / GRU with expert quality head | 69 train / 15 val / 15 test groups.<br>Adds UCOPhyRehab++ (exact spans, composite score). |
| `artifacts/v2-quality-fixed/` | `training/configs/v2_quality_fixed.yaml` | `tcn_best.pt` (epoch 42 / 72)<br>`gru_baseline.pt` (interrupted epoch 52) | **TCN training complete** (72 epochs, best 42).<br>**GRU interrupted** (epoch 52). | **TCN Evaluated on Test** (Sequence & Window):<br>Family Acc: 90.47%, Macro-F1: 83.91%<br>Phase Acc: 81.36%, Macro-F1: 54.92%<br>Start F1: 66.96%, End F1: 12.26%<br>Rep Count MAE: 2.03<br>Expert Quality Acc: 56.92% (MAE: 0.43)<br>Quality-4 heads: 0% coverage (masked) | Causal TCN (307,410)<br>Causal GRU (68,178)<br>283 inputs, float16 memmaps | Reuses `data/processed-v2-quality`.<br>1,007 logical test sequences.<br>Tested on UL-RED, UCO, wheelchair-positions. |

---

## 2. Key Findings & Baseline Comparison

### A. Repetition-End Bottleneck
Across both `corrected-v1` and `v2-quality-fixed`:
- **Repetition Start** is reliably detected (99.54% F1 on v1, 66.96% F1 on v2 test).
- **Repetition End** is materially weaker (18.47% F1 on v1, 12.26% F1 on v2 test).
- **Root Cause**: Phase transitions to rest/eccentric boundary have wide source annotation disagreement and loose boundary definitions. As diagnosed in the training strategy, this requires calibration/debouncing in the decoder (Task C1) before additional training compute.

### B. Quality Head Coverage Reality
- All four dimension-specific quality outputs (ROM, tempo, smoothness, trunk compensation) currently have **0.0% labeled coverage** across all existing datasets.
- UCOPhyRehab++ provides an ordinal 1–5 physiotherapist composite execution score mapped to `expert_quality_logits` (5-class). On 65 evaluated test sequences with expert ratings, TCN achieves **56.92% accuracy and 0.43 MAE**. This is a useful auxiliary task, but must never be presented as four independent clinical quality scores.

### C. Target-Population Data Gap
- There are **no real amputee, congenital limb-difference, or wheelchair-user recordings** in the training or test splits.
- Wheelchair-position recordings in IntelliRehabDS are posture proxies by non-wheelchair users.
- Procedural seeds and synthetic limb-occlusion masks are engineering aids for network gradient stability; they are not target-population validation.

---

## 3. Luna Execution Task Progress Tracker

| Task ID | Description | Status | Verification & Evidence |
|---|---|---|---|
| **A1** | Inventory existing work & artifacts | **COMPLETED** | Verified configs, checkpoints, memmap shapes, parameter counts, and histories across `corrected-v1`, `v2-quality`, and `v2-quality-fixed`. |
| **A2** | Define valid comparison cohort | **IN PROGRESS** | Training participant IDs isolated; test splits locked. Union overlap checks enforced by `training.audit`. |
| **A3** | Establish baseline & failure inventory | **COMPLETED** | Sequence evaluation generated for `v2-quality-fixed/tcn_best.pt` on test split (1,007 sequences); compared with `corrected-v1`. |
| **A4** | Prepare focused launch supervision | **PENDING** | Define launch-exercise rep boundary conventions (curls, rows, extensions, marches, reach). |
| **B1** | Safe warm-start weight initialization | **COMPLETED (not run)** | `--init-checkpoint` and `training.init_checkpoint` load weights only, record parent, and enforce model/schema compatibility. |
| **B2** | Trainable-layer selection & resume | **COMPLETED (not run)** | `--freeze-backbone`, `--unfreeze-last-blocks`, separate head/backbone rates, exact optimizer/RNG/sampler/trainable-layer state; covered by staged-training tests. |
| **B3** | Redundant work reduction (stride/sampler) | **PENDING** | Evaluate stride-16/32 training sampling while preserving validation reconstruction. |
| **B4** | Bounded experiment configuration | **COMPLETED** | `training/configs/experiments/r0_baseline.yaml` and `r1_tcn_boundary_finetune.yaml`; both validate and use isolated roots. |
| **C1** | Improve decoding without gradient updates | **COMPLETED (blocked gate)** | Python `decoder.v1` and validation-only calibration are implemented; baseline end-event gate failed and is recorded, not hidden. |

---

## 4. Verification History

| Date | Commit | Role | Action | Result | Notes |
|---|---|---|---|---|---|
| 2026-09-01 | `e75ba65` | Historical baseline | Initial training & corrected-v1 benchmark | Completed | Full TCN/GRU trained; 123 tests passing. |
| 2026-09-07 | `b6ac20c` | Documentation revision | Documentation contracts and runbooks refined | Completed | Unified metadata, schemas, and roadmap across 23 files. |
| 2026-09-07 | `ba8bf1a` | **Code baseline** | Repository-wide audit & TCN v2 evaluation | Passed (124 tests) | Evaluated `v2-quality-fixed` TCN; fixed TCN streaming expert head; added CLI scripts and execution log; last commit modifying model/runtime/test code. |
| 2026-09-08 | `f890101` / `686218a` | Documentation revision | Roadmap alignment & canonical metadata synchronization | Passed (124 tests) | Aligned roadmap statuses, audited dataset mappings, and synchronized canonical documentation metadata. |
| 2026-09-08 | `d5362ee` | Documentation revision | Execution provenance audit & contract grounding | Passed (124 tests) | Updated prepared-manifest counts, labeled planned paths and candidate datasets, grounded speedup hypotheses. |
| 2026-09-08 | `1a46f38` | Documentation revision | Phase 5 planned-metric labeling | Passed (124 tests) | Marked mobile calibration, parity, latency, memory, and thermal values as planned acceptance criteria. |
| 2026-09-08 | `613ff12` | Documentation revision (prior canonical HEAD) | Expand dataset catalog and formalize operational documentation contracts | Passed (124 tests) | Pushed baseline audited before the documentation correctness repair; expanded dataset registry and operational contracts. |
| 2026-09-08 | `847fd5d` | Pre-training gate implementation | Decoder, provenance, staged runner controls, R0/R1 configs, and readiness docs | Passed (132 tests; docs validator) | Corrected-v1 hashes unchanged; R0 manifests/calibration regenerated; R1 training intentionally not launched. |
