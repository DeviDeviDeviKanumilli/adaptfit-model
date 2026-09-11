# AdaptFit current state

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** source code, versioned configuration, filesystem artifacts, and verified reports
> - **Last verified:** 2026-09-11
> - **Source commit:** `b3926d8` (R1 calibration and phase-weight correction training)
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
| Repository revision at last verification | `b3926d864c1d976504c2d7131db8d4992cf2ef45` | Current checkout used for the R1 decoder calibration; the R1 training artifact records `d7848e2b12322e40e83758abd5e12ae6c7727e80`. |
| Code baseline | `b3926d8` plus the current follow-up working tree | The corrected-v1 model remains unchanged; decoder, provenance, runner controls, label-mask fixes, class-weight fix, regression tests, and isolated configs are present in the current checkout. |
| Documentation revision baseline | `613ff12` | Latest pushed documentation revision audited before this repair; this is a provenance anchor, not a mutable HEAD claim. |
| Working-tree state | Must be checked | Run `git status --short`; this snapshot never treats an unverified tree as clean. |
| Test suite | 138 tests passed (`python3 -m pytest -q`) | Documentation validator and full test suite pass after the UL-RED markerless 3Rep boundary-source correction and extracted-path regression; `git diff --check` is clean. |
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
- UL-RED `marker-less/3Rep_Sxx.csv` supplies explicit zero-based three-
  repetition spans for 219 of 219 available R3 recordings. The adapter maps
  these to one-based AMC frame IDs; the spans are boundary-only and do not
  supply per-frame hold/apex labels.

## Artifact state

| Artifact directory | State | What it proves |
|---|---|---|
| `artifacts/corrected-v1/` | Complete benchmark | TCN and GRU were trained/evaluated and overlapping windows were merged to sequence metrics. This is the current baseline. |
| `artifacts/v2-quality/` | Prepared data only | Preparation and manifests exist; no completed model comparison. |
| `artifacts/v2-quality-fixed/` | Partial (TCN evaluated; GRU interrupted) | TCN training completed through 72 epochs (best epoch 42) and has been evaluated on the test split with sequence and window metrics; the GRU checkpoint is an interrupted intermediate run at epoch 52. It is not a complete TCN/GRU comparison. |
| `artifacts/r0-baseline/` | Local evaluation-only audit (ignored by Git) | Fresh corrected-v1 test reports, model manifests, and validation-only decoder calibration; regenerate if absent. It contains no trained model. |
| `artifacts/r1-tcn-boundary/` | Local full-training artifact (ignored by Git) | 27-epoch MPS heads-only warm-start from corrected-v1; best epoch 12; validation-only decoder report is present but blocked; no test evaluation. |
| `artifacts/r1-tcn-boundary-mask-correction/` | Training complete; validation decoder gate blocked (ignored by Git) | Unavailable-boundary masking corrected; isolated prepared split regenerated; heads-only TCN completed through epoch 68 with best epoch 53; validation-only decoder calibration has end F1 `0.0`; failure analysis is recorded; test evaluation remains locked. |
| `artifacts/r1-tcn-phase-weight-correction/` | Training complete; validation-only decoder gate blocked; test locked (ignored by Git) | Absent-class phase-weight normalization corrected; verified corrected-mask prepared root reused unchanged; heads-only TCN completed through epoch 24 with best epoch 9 and validation sequence score `0.6679357`; calibration end F1 `0.153846`, false events on empty sequences `15`; no test evaluation. |
| `artifacts/r1-tcn-last-block-event-support/` | Training complete; validation decoder gate blocked; test locked (ignored by Git) | Final-block fine-tune from the phase-weight checkpoint; best epoch 8; validation sequence score `0.6791891`; decoder end F1 `0.094862`, count MAE `0.271205`, and 8 empty-target false events; no test evaluation. |
| `artifacts/r1-tcn-last-two-blocks-event-support/` | Training complete; validation decoder gate blocked; test locked (ignored by Git) | Final-two-block fine-tune from the last-block checkpoint; best epoch 42 of 57; validation sequence score `0.7040537`; decoder end F1 `0.299674`, count MAE `0.333705`, and 63 empty-target false events; no test evaluation. |
| `artifacts/r1-tcn-full-finetune-event-support-pilot/` | Training complete; validation decoder gate blocked; test locked (ignored by Git) | Full-TCN pilot from the final-two-block best checkpoint; all 307,410 parameters trainable; best epoch 6 of 8; validation sequence score `0.7176679`; decoder end F1 `0.296053`, count MAE `0.330357`, and 60 empty-target false events; failure analysis recorded; no test evaluation. |
| `artifacts/r1-tcn-uco-boundary-support/` | Training complete; validation decoder gate blocked; test locked (ignored by Git) | UCO exact-boundary support run; best checkpoint SHA-256 `9c69b9aa8173da7f10140779ec70f87c012ae151519e3f2375e61b0dd09327a0`; later used as the parent for the UCO follow-ups; no test evaluation. |
| `artifacts/r1-tcn-uco-tracking-target-correction/` | Training complete; validation decoder gate blocked; test locked (ignored by Git) | Corrected partial-pose tracking denominator; best epoch 2 of 5; validation sequence score `0.6176060`; decoder end F1 `0.070130`, count MAE `0.584`, and 7 empty-target false events; UCO supplied `0/507` nearby starts and `0/507` apex-qualified targets; no test evaluation. |
| `artifacts/r1-tcn-uco-boundary-positive-weight/` | Training complete; validation decoder gate blocked; test locked (ignored by Git) | Boundary positive-weight cap `64.0` intervention; best epoch 1 of 4; validation sequence score `0.5252875` (worse than `0.6176060`); decoder end F1 `0.142322`, count MAE `0.5864`, and 24 empty-target false events; cap rejected; no test evaluation. |
| `artifacts/r1-tcn-ulred-boundary-support/` | Training complete; validation decoder gate blocked; test locked (ignored by Git) | Local UL-RED `marker-less/3Rep` boundary-source correction in an isolated prepared root; best epoch 2 of 5; validation sequence score `0.6013545`; decoder end F1 `0.070718`, count MAE `0.684`, and 8 empty-target false events; failure analysis recorded; no test evaluation. |

Checkpoint-level status at this verification:

- Corrected-v1: `artifacts/corrected-v1/checkpoints/tcn_best.pt` and
  `gru_baseline.pt`, with model-specific histories, evaluations, sequence
  predictions, feature schema, normalization statistics, and model card.
- V2-quality: no checkpoint files; only preparation/config/schema/audit files.
- V2-quality-fixed: `checkpoints/tcn_best.pt` has verified test sequence and
  window evaluation (`metrics/tcn_evaluation.json`, predictions, and sequence
  metadata); `gru_baseline.pt` remains an interrupted intermediate state (epoch 52).
  See [training-execution-log.md](training-execution-log.md) for full metrics.
- R1: `artifacts/r1-tcn-boundary/checkpoints/tcn_best.pt` is the epoch-12
  checkpoint from the completed heads-only full run. It has validation-only
  metrics and a blocked decoder calibration report; it has no test metrics. See
  [training-execution-log.md](training-execution-log.md).
- Corrected mask follow-up: `artifacts/r1-tcn-boundary-mask-correction/checkpoints/tcn_best.pt`
  is the epoch-53 checkpoint from the completed heads-only run. Its
  validation-only decoder report is blocked (`end_f1=0.0`); it has no test
  metrics. See [training-execution-log.md](training-execution-log.md).
- Phase-weight correction: `artifacts/r1-tcn-phase-weight-correction/checkpoints/tcn_best.pt`
  is the epoch-9 checkpoint from the completed heads-only run. It has
  training-time validation metrics and a blocked validation-only decoder
  calibration; it has no test metrics. See
  [training-execution-log.md](training-execution-log.md).
- Last-block event-support intervention: the config is
  `training/configs/experiments/r1_tcn_last_block_event_support.yaml`; its
  best checkpoint is epoch 8 after early stopping at epoch 23. Its
  validation-only decoder gate is blocked (end F1 `0.094862`; 8 empty-target
  false events), and it has no test metrics. Its failure analysis shows
  `21/240` apex-qualified intervals, `219/219` segmentation starts below the
  selected tracking floor, and `92/240` target ends with a nearby end signal.
- Last-two-block event-support intervention: the config is
  `training/configs/experiments/r1_tcn_last_two_blocks_event_support.yaml`; it
  points to the last-block best checkpoint and completed with best epoch 42 of
  57. Its validation-only decoder gate is blocked (end F1 `0.299674`; 63
  empty-target false events), and its failure analysis finds `49/240`
  apex-qualified intervals, `219/219` segmentation starts below the tracking
  floor, and `91/240` target ends with a nearby end signal. Test evaluation
  remains disabled and locked.
- Full-TCN event-support pilot: the config is
  `training/configs/experiments/r1_tcn_full_finetune_event_support_pilot.yaml`;
  it points to the final-two-block best checkpoint and has completed the
  bounded 8-epoch run after passing preflight/runtime checks. Its isolated
  artifact root is `artifacts/r1-tcn-full-finetune-event-support-pilot/`; all
  `307,410` parameters are trainable. Validation-only calibration is blocked
  (end F1 `0.296053`; 60 empty-target false events), and test evaluation remains
  disabled and locked. Failure analysis finds `219/219` segmentation starts
  below the tracking floor and only `101/240` target ends with nearby end
  signal.
- UCO boundary-support run: `artifacts/r1-tcn-uco-boundary-support/` contains
  the exact-boundary parent checkpoint used by the UCO follow-ups; its decoder
  gate is blocked and it has no test metrics.
- UCO tracking-target correction: the config is
  `training/configs/experiments/r1_tcn_uco_tracking_target_correction.yaml`;
  its best checkpoint is epoch 2 at
  `artifacts/r1-tcn-uco-tracking-target-correction/checkpoints/tcn_best.pt`.
  The correction removes partial-pose tracking dilution, but calibration is
  blocked (end F1 `0.070130`; count MAE `0.584`; 7 empty-target false events).
  UCO strong boundaries still have no learned start/end/apex signal and no
  per-frame hold/apex labels.
- UCO boundary-positive-weight intervention: the config is
  `training/configs/experiments/r1_tcn_uco_boundary_positive_weight.yaml` and
  its best checkpoint is at
  `artifacts/r1-tcn-uco-boundary-positive-weight/checkpoints/tcn_best.pt`.
  Raising the cap to `64.0` reduced the validation sequence score to
  `0.5252875` and increased empty-target false events to `24`; the intervention
  is rejected. Do not repeat it.
- UL-RED boundary-source correction: the config is
  `training/configs/experiments/r1_tcn_ulred_boundary_support.yaml`; its
  isolated prepared root is `data/processed-r1-ulred-boundary-support/` and
  its best checkpoint is
  `artifacts/r1-tcn-ulred-boundary-support/checkpoints/tcn_best.pt` (epoch 2,
  SHA-256 `7ed351765ad1c943d525721c03fdcec65c1fdd06df2966c59971ca934b5d4ce1`).
  The overlay records 219 explicit UL-RED R3 boundary sources. Calibration is
  blocked (end F1 `0.070718`; count MAE `0.684`; 8 empty-target false events),
  because the source spans do not provide hold/apex labels. Test evaluation
  remains disabled and locked.

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
   partial-block, separate-learning-rate, isolated-root, and test-lock controls.
   The R1 smoke and full run exercised warm-start and frozen-backbone behavior;
   the corrected mask follow-up verified exact interruption/resume from the
   epoch-10 checkpoint; the phase-weight follow-up exercised the corrected
   absent-class weighting behavior.
4. Repetition-end performance is materially weaker than repetition-start
   performance and needs reviewed boundary/phase supervision before product
   use. The UCO tracking-target correction removed the tracking-floor artifact,
   and the UL-RED 3Rep audit recovered valid boundary spans, but neither source
   supplies the reviewed hold/apex supervision needed by the decoder gate.
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

1. Request reviewed UCO, UL-RED, and/or REHAB24-6 start, hold/apex, and end
   labels, or additional representative seated unilateral recordings carrying
   those labels. The UCO tracking correction, sparse-boundary-weight
   intervention, and UL-RED boundary-source correction are complete; all
   decoder gates remain blocked.
   Use the [reviewed-event supervision intake](reviewed-event-supervision-request.md)
   for the exact row schema and acceptance checks.
2. Keep `data/processed-r1-uco-tracking-target-correction/` and
   `data/processed-r1-ulred-boundary-support/` unchanged until external
   supervision is audited and regenerated into a new isolated prepared root.
3. After new supervision exists, rerun preflight, validation-only calibration,
   and failure analysis in a new artifact root. Only if the decoder gate passes
   should a second-seed confirmation precede one locked test evaluation.
4. Keep corrected-v1 and the locked test split unchanged; do not promote any
   blocked R1 candidate.
5. Obtain reviewed quality labels and consented target-population recordings.
6. Define and test the model bundle, Python/native golden fixtures, and mobile
   release gates before describing deployment as available.
7. Treat the [Motion-JEPA plan](motion-jepa-world-model-plan.md) as a gated
   research backlog. Do not start it until decoder calibration and the
   supervised comparison establish a measured need and a valid pretraining
   manifest.
8. Build the recommendation path in order: reviewed recipe metadata, hard
   feasibility mask, content/rules baseline, feedback logging, then a neural
   ranker only after user/time-held-out evaluation data exists.

## Planned work (not current behavior)

The roadmap may propose a density head, Motion-JEPA pretraining, teacher
distillation, a recommendation model, richer quality targets, personalized
calibration, quantization, and native mobile inference. Those are planned
experiments or deliverables.
They become current only when their code/config, artifact, evaluation, and
manifest evidence are recorded here.
