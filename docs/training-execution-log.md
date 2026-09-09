# AdaptFit Training Execution Log

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** training execution ledger, checkpoint provenance, and audit record
> - **Last verified:** 2026-09-08
> - **Source commit:** `627283b` (R1 smoke-run evidence; baseline and pre-training gate)
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

This gate was completed before the R1 smoke run. The detailed handoff is
[pretraining-readiness.md](pretraining-readiness.md). The smoke result is
recorded below; the full R1 budget remains gated on this evidence and has not
started.

| Gate | Result | Evidence |
|---|---|---|
| Immutable corrected-v1 hashes | **PASS** | TCN `e69ff69d…`, GRU `4105bf23…`; no files under `artifacts/corrected-v1/` changed |
| Strict source/config preflight | **PASS** | `r0_baseline.yaml` and `r1_tcn_boundary_finetune.yaml`; enabled sources decode successfully; optional quality sources are disabled |
| R0 test audit | **PASS (audit only)** | `artifacts/r0-baseline/metrics/*_evaluation.json` and model manifests; test was read, never used for calibration |
| Validation-only decoder calibration | **BLOCKED** | `artifacts/r0-baseline/decoder/decoder_calibration.json`; count MAE `0.3649554`, end F1 `0.0`, false empty-sequence events `0` |
| Warm-start/freeze/resume controls | **PASS (smoke exercised)** | `training/src/runner.py`, `training/train.py`, staged config, staged-training tests, and the R1 smoke checkpoint; full R1 not run |
| Next eligible action | **Full R1 heads-only training** | Smoke passed with finite losses and complete provenance; keep test evaluation locked |

The decoder block is a measured baseline limitation. It does not authorize
changing the test split, relaxing label masks, or presenting the decoder as a
release component. The full R1 run is now the next computational action because
R1 is the declared experiment for improving the boundary signal; product
release remains blocked until the decoder gate is rechecked and all other
release gates pass.

## 2B. R1 smoke training verification (2026-09-08)

The bounded R1 smoke command completed on CPU for two epochs after strict
preflight passed for the four enabled sources (`rehab24_6`, `intellirehabds`,
`mmfit`, and `ul_red`). It used the corrected-v1 TCN checkpoint as a warm-start,
froze the temporal backbone, trained 1,746 head parameters, and wrote only to
the ignored `artifacts/r1-tcn-boundary/` root. No test evaluation was performed.

| Item | Result |
|---|---|
| Effective source commit | `627283bec6851d95cba988a0235931c158465eab` |
| Config hash | `sha256:0537376cac3337a95277b2b063a6549c13ded9e1b548b2a00266e96827302962` |
| Parent checkpoint | `/Users/devk/AdaptFit/artifacts/corrected-v1/checkpoints/tcn_best.pt` |
| Best epoch | 1 of 2 |
| Epoch 1 | loss `0.199823`; validation sequence score `0.648495`; sequence boundary F1 `0.556144`; sequence repetition-end F1 `0.118085`; count MAE `0.107143` |
| Epoch 2 | loss `0.198885`; validation sequence score `0.647672`; sequence boundary F1 `0.554943`; sequence repetition-end F1 `0.117122`; count MAE `0.131696` |
| Training compute | `848.13` seconds on CPU; validation `23.98` seconds |
| Test evaluation | **Not performed** (`test_evaluation_performed=false`) |
| Best checkpoint | `artifacts/r1-tcn-boundary/checkpoints/tcn_best.pt`; SHA-256 `9c3098b2633292136df99f9c4c52a4623be2c7137bd05bcc661d9412ae6fc772` |
| Latest checkpoint | `artifacts/r1-tcn-boundary/checkpoints/tcn_latest.pt`; SHA-256 `d342b78fbfff55b2d872a70c3f18342ffd19ad4b7175ff4b0d1b8f3b8f5ca017` |
| Run report | `artifacts/r1-tcn-boundary/metrics.json`; SHA-256 `e25849f805c9f2454144b1c160adabb0396e21083e67235c2ba516e089d37d25` |
| Run status | **Smoke complete; full R1 not run** |

Both checkpoints contain model and optimizer state, Python/NumPy/Torch RNG
state, sampler epoch, effective `config`, trainable-layer list, parent
checkpoint, and source commit. The smoke result is a readiness/provenance gate,
not a model-selection result or product claim. The full R1 run is eligible, but
its candidate must still be selected on validation, recalibrated with
`decoder.v1`, and evaluated on the locked test split only after those choices
are frozen.

---

## 1. Candidate Checkpoint & Run Inventory (Task A1)

| Run / Artifact Root | Config | Checkpoint | Training Status | Labeled Evaluation Status | Model Architecture & Params | Split & Sources |
|---|---|---|---|---|---|---|
| `artifacts/corrected-v1/` | `training/configs/v1_corrected.yaml` | `tcn_best.pt` (epoch 35)<br>`gru_baseline.pt` (epoch 34) | **Training complete** (50 / 49 epochs) | **Evaluated** (Sequence & Window)<br>Family Acc: 91.25% (TCN) / 89.47% (GRU)<br>Start F1: 99.54% (TCN) / 83.50% (GRU)<br>End F1: 18.47% (TCN) / 5.53% (GRU)<br>Rep Count MAE: 0.31 (TCN) / 30.63 (GRU) | Causal TCN (307,410)<br>Causal GRU (68,178)<br>283 inputs, 128 frames | 50 train / 11 val / 11 test participant groups.<br>Zero identity collisions.<br>Sources: REHAB24-6, IntelliRehabDS, MM-Fit, UL-RED, procedural seed. |
| `artifacts/v2-quality/` | `training/configs/v2_quality.yaml` | None | **Prepared data only** | **Not trained / not evaluated** | Target: Causal TCN / GRU with expert quality head | 69 train / 15 val / 15 test groups.<br>Adds UCOPhyRehab++ (exact spans, composite score). |
| `artifacts/v2-quality-fixed/` | `training/configs/v2_quality_fixed.yaml` | `tcn_best.pt` (epoch 42 / 72)<br>`gru_baseline.pt` (interrupted epoch 52) | **TCN training complete** (72 epochs, best 42).<br>**GRU interrupted** (epoch 52). | **TCN Evaluated on Test** (Sequence & Window):<br>Family Acc: 90.47%, Macro-F1: 83.91%<br>Phase Acc: 81.36%, Macro-F1: 54.92%<br>Start F1: 66.96%, End F1: 12.26%<br>Rep Count MAE: 2.03<br>Expert Quality Acc: 56.92% (MAE: 0.43)<br>Quality-4 heads: 0% coverage (masked) | Causal TCN (307,410)<br>Causal GRU (68,178)<br>283 inputs, float16 memmaps | Reuses `data/processed-v2-quality`.<br>1,007 logical test sequences.<br>Tested on UL-RED, UCO, wheelchair-positions. |
| `artifacts/r1-tcn-boundary/` | `training/configs/experiments/r1_tcn_boundary_finetune.yaml` | `tcn_best.pt` (epoch 1 / 2)<br>`tcn_latest.pt` (epoch 2) | **Smoke complete; full R1 not run** | Validation-only sequence metrics; no test evaluation. Best sequence boundary F1 `0.5561`, repetition-end F1 `0.1181`, count MAE `0.1071`. | Causal TCN (307,410)<br>1,746 trainable head parameters<br>283 inputs, 128 frames | Corrected-v1 prepared split; parent `corrected-v1/tcn_best.pt`; CPU smoke run. |

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
| **B1** | Safe warm-start weight initialization | **COMPLETED (smoke verified)** | `--init-checkpoint` and `training.init_checkpoint` loaded the corrected-v1 parent and recorded its path and source commit in the R1 checkpoint. |
| **B2** | Trainable-layer selection & resume | **COMPLETED (smoke verified)** | Frozen-backbone head training produced 1,746 trainable parameters and checkpoints with optimizer/RNG/sampler/trainable-layer state; exact resume remains untested on a real interruption. |
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
| 2026-09-08 | `627283b` | R1 smoke verification | Two-epoch CPU heads-only warm-start with test evaluation disabled | Passed | Finite losses; validation-only metrics and provenance complete; full R1 remains unrun. |
