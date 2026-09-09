# AdaptFit current state

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** source code, versioned configuration, filesystem artifacts, and verified reports
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `613ff12` (documentation revision base)
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** supersedes scattered “current status” statements in run reports; supports the contracts, training, evaluation, and deployment documents
> - **Review trigger:** any code/config/schema change, new checkpoint, completed evaluation, adapter change, or deployment conversion

This is the single snapshot of what is implemented and evidenced in the
canonical repository at `/Users/devk/AdaptFit`. A statement in another
document may propose or explain work, but it cannot override this file without
an update here and a source artifact.

Read [system context and data flow](system-context-and-dataflow.md) for the
end-to-end boundary, [repository and implementation map](repository-and-implementation-map.md)
for source/test ownership, and [model registry](model-registry.md) for the
current-versus-planned model roster. The [artifact registry](artifact-registry.md)
is the canonical path inventory. The [pre-training readiness handoff](pretraining-readiness.md)
is the reproducibility gate for the next isolated run; this file remains the
current-state snapshot.

## Repository and verification

| Item | Current value | Evidence or limit |
|---|---|---|
| Active repository | `/Users/devk/AdaptFit` | The empty `/Users/devk/Documents/ChatGPT/AdaptFit` checkout is not the model source of truth. |
| Repository revision at last verification | `8e1db4010257a3ac7211546e3e08d777bba3ef22` | Baseline revision before the current pre-training gate changes; run `git rev-parse HEAD` before reproducing a run. |
| Code baseline | `ba8bf1a` plus current working-tree gate changes | The corrected-v1 model remains unchanged; decoder, provenance, runner controls, and isolated configs are new working-tree changes. |
| Documentation revision baseline | `613ff12` | Latest pushed documentation revision audited before this repair; this is a provenance anchor, not a mutable HEAD claim. |
| Working-tree state | Must be checked | Run `git status --short`; this snapshot never treats an unverified tree as clean. |
| Test suite | 124 tests passed in the last verified run (`python3 -m pytest -q`) | Re-run after code changes. |
| Runtime scope | Python training and streaming runtime | No production mobile bridge is present. |

## Implemented model contract

- Canonical pose input is the 33-joint layout in
  `training/src/data/schema.py`.
- `FeatureSchemaV1` has 283 ordered inputs produced by
  `training/src/features/anatomy.py`. The exact ordering and masks are defined
  in [contracts-and-schemas.md](contracts-and-schemas.md).
- The default temporal window is 128 frames at approximately 30 FPS with
  stride 8 in the current training configuration.
- The TCN uses 96 channels and five causal residual blocks with dilations
  1, 2, 4, 8, and 16. Its causal receptive field is 125 frames. The measured
  corrected-v1 TCN has 307,410 parameters.
- The small GRU is a baseline; the measured corrected-v1 baseline has 68,178
  parameters.
- Current prediction heads are six movement-family logits, five phase logits,
  per-frame repetition start/end logits, four pooled binary quality logits, and
  a per-frame tracking-confidence output. A five-class expert-quality output
  is optional and must remain masked unless its target is present.
- Capability-aware filtering, exercise eligibility, confidence thresholds,
  abstention, decoding, and event emission are deterministic product/runtime
  responsibilities. The Python reference decoder is implemented in
  `training/src/decoder.py` as `decoder.v1`; a native bridge and production
  bundle are still unavailable.

## Data adapters and label state

The integrated preparation path has adapters for REHAB24-6, IntelliRehabDS,
MM-Fit, UL-RED, and UCOPhyRehab++. Adapter presence means data can be converted
to the canonical prepared format; it does not mean every task has valid labels.
The registry in [dataset-catalog.md](dataset-catalog.md) is authoritative for
license/access status, label roles, and intended use.

Current label facts:

- Source-level family, phase, repetition, and correctness labels are retained
  only where the source actually supplies them.
- Weak displacement-derived labels, single-clip repetition assumptions, and
  procedural quality templates are masked by default.
- Coverage for the four dimension-specific quality heads is currently **zero**.
  No quality claim may be made from those heads until reviewed targets exist.
- There is no real amputee, limb-difference, or wheelchair-user validation
  cohort. Public seated metadata and synthetic limb masking are engineering
  aids, not target-population evidence.

## Artifact state

| Artifact directory | State | What it proves |
|---|---|---|
| `artifacts/corrected-v1/` | Complete benchmark | TCN and GRU were trained/evaluated and overlapping windows were merged to sequence metrics. This is the current baseline. |
| `artifacts/v2-quality/` | Prepared data only | Preparation and manifests exist; no completed model comparison. |
| `artifacts/v2-quality-fixed/` | Partial (TCN evaluated; GRU interrupted) | TCN training completed through 72 epochs (best epoch 42) and has been evaluated on the test split with sequence and window metrics; the GRU checkpoint is an interrupted intermediate run at epoch 52. It is not a complete TCN/GRU comparison. |
| `artifacts/r0-baseline/` | Local evaluation-only audit (ignored by Git) | Fresh corrected-v1 test reports, model manifests, and validation-only decoder calibration; regenerate if absent. It contains no trained model. |

Checkpoint-level status at this verification:

- Corrected-v1: `artifacts/corrected-v1/checkpoints/tcn_best.pt` and
  `gru_baseline.pt`, with model-specific histories, evaluations, sequence
  predictions, feature schema, normalization statistics, and model card.
- V2-quality: no checkpoint files; only preparation/config/schema/audit files.
- V2-quality-fixed: `checkpoints/tcn_best.pt` has verified test sequence and
  window evaluation (`metrics/tcn_evaluation.json`, predictions, and sequence
  metadata); `gru_baseline.pt` remains an interrupted intermediate state (epoch 52).
  See [training-execution-log.md](training-execution-log.md) for full metrics.

Corrected-v1 evidence includes 617 logical test sequences and 7,592 windows
with zero identity collisions. The fresh R0 report, using the current masked
phase policy, records family accuracy 91.25%, family macro-F1 87.57%, phase
accuracy 86.10%, phase macro-F1 60.24%, repetition-start F1 99.54%,
repetition-end F1 18.47%, and repetition-count MAE 0.31. These values are
benchmark evidence, not a release claim; use the artifact manifest and
[evaluation-protocol.md](evaluation-protocol.md) for the required split and
metric provenance.

## Product and deployment status

- The model repository does not contain a production exporter, TFLite/Core ML
  conversion, native model bridge, quantized artifact, or mobile golden-fixture
  suite.
- PeddieHacks is a reference application only. Its Android path emits selected
  angles and aggregate confidence, its iOS pose path is stubbed, and its
  deterministic counter is bilateral-first. It is not the active AdaptFit
  implementation.
- Raw camera frames and raw pose should remain local by design, but this has
  not been demonstrated by a production mobile build.
- No mobile latency, memory, thermal, battery, Python/native parity, or
  float/quantized parity gate has passed.
- No neural workout recommender, workout-history model, learned substitution
  ranker, or production exercise-selection service exists. The capability and
  equipment fields are available for contracts, but the complete selection
  system remains planned.

## Known blockers and unsupported claims

1. Quality supervision is absent for all four dimension-specific quality heads.
2. Target-population validation is absent.
3. The runner now exposes warm-start, exact-resume, frozen-backbone,
   partial-block, separate-learning-rate, isolated-root, and test-lock controls;
   these controls are covered by the pre-training handoff but have not been
   used for a completed training experiment in this working tree.
4. Repetition-end performance is materially weaker than repetition-start
   performance and needs boundary/decoder review before product use.
5. A production model bundle and native runtime contract do not exist.
6. No document may claim medical, clinical, force, muscle-activation, joint-load,
   or safety certification from the current evidence.
7. No Motion-JEPA or other world-model teacher is implemented, cached, evaluated,
   or available as a checkpoint. The proposal is training-only and must not be
   treated as current model behavior.
8. Recommendation ranking has no behavioral dataset or validated neural model;
   only reviewed recipes and future deterministic filtering are specified.
9. Demo and marketing language must never outpace empirical evidence: claims of
   “rock-solid repetition counting,” real-time trunk compensation feedback,
   “100% on-device privacy,” or target-population pilot validation are ahead of
   current evidence and strictly prohibited until mobile export, privacy audit,
   and the consented target-population pilot study are executed and evidenced by
   committed artifacts.

## Next validated actions

1. Use the [pre-training readiness handoff](pretraining-readiness.md) to run
   the bounded R1 TCN smoke training in its isolated artifact root.
2. Select on validation, recalibrate the decoder on validation, and keep the
   corrected-v1 test split locked until the candidate is frozen.
3. Use the implemented provenance-aware fine-tuning controls for the isolated
   pilot before any teacher or density experiment.
4. Obtain reviewed quality labels and consented target-population recordings.
5. Define and test the model bundle, Python/native golden fixtures, and mobile
   release gates before describing deployment as available.
6. Treat the [Motion-JEPA plan](motion-jepa-world-model-plan.md) as a gated
   research backlog. Do not start it until decoder calibration and the
   supervised comparison establish a measured need and a valid pretraining
   manifest.
7. Build the recommendation path in order: reviewed recipe metadata, hard
   feasibility mask, content/rules baseline, feedback logging, then a neural
   ranker only after user/time-held-out evaluation data exists.

## Planned work (not current behavior)

The roadmap may propose a density head, Motion-JEPA pretraining, teacher
distillation, a recommendation model, richer quality targets, personalized
calibration, quantization, and native mobile inference. Those are planned
experiments or deliverables.
They become current only when their code/config, artifact, evaluation, and
manifest evidence are recorded here.
