# AdaptFit evaluation protocol

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** evaluation code/configuration, artifact manifests, and this protocol
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `9fe47fb` (documentation revision base)
> - **Owner:** AdaptFit engineering and evaluation reviewer
> - **Supersedes or supports:** establishes the common protocol for corrected-v1, v2-quality, and future experiments
> - **Review trigger:** metric/code/config changes, new labels, new target population, new runtime, or a release decision

Use the [requirements traceability ledger](requirements-traceability.md) to
connect each metric to a product claim and the [test and fixture matrix](test-and-fixture-matrix.md)
to distinguish Python invariants from missing native, privacy, recommender, and
target-population evidence.

This protocol prevents window-level scores, weak labels, or selective abstention
from being presented as product evidence. Every report must identify the source,
participant split, commit, config hash, checkpoint, artifact paths, and limits.

## Evaluation units and splits

1. **Participant-level split is mandatory.** A participant or household may
   occur in only one of train, validation, or test. Session IDs alone are not
   sufficient when a person has multiple sessions.
2. **Source-held-out evaluation is required for transfer claims.** Hold out a
   complete source or source family when the claim concerns cross-dataset
   generalization.
3. **Validation is for selection; test is locked.** Decoder thresholds,
   calibration, feature changes, and checkpoint choice may use validation only.
4. **Synthetic lineage remains visible.** Synthetic or augmented samples may
   not be counted as independent participants and must not obscure the real
   source/profile breakdown.
5. **Sequence-level metrics are primary.** Merge overlapping windows back to
   sequence coordinates using the stored offsets. Window metrics are
   diagnostics only because stride creates correlated examples.

## Metric definitions

### Movement family and phase

- Report accuracy, macro-F1, per-class support, and a confusion matrix.
- Report phase macro-F1 and per-phase support; do not hide absent phases behind
  a micro average.
- Break down by source, exercise, side, position, capability state, and camera
  visibility where sample size permits.

### Repetition events and counts

- Match predicted and reference start/end events one-to-one within the
  versioned `boundary_tolerance_frames` configured for the evaluation run.
  Convert the tolerance to milliseconds using the recorded frame rate. Never
  silently choose a new tolerance per checkpoint.
- Report start precision/recall/F1, end precision/recall/F1, and matched-event
  timing error (median and p95).
- Report sequence count MAE, median absolute error, one-off accuracy, and the
  distribution of overcount/undercount.
- Report false events during labeled rest per minute and event latency from the
  first eligible causal evidence.
- Include partial reps, pauses, exercise changes, resets, dropped frames, and
  duplicate overlapping-window predictions in replay fixtures.

### Quality, confidence, and abstention

- Report label coverage separately for each quality dimension; zero coverage
  means “unavailable,” not a score of zero.
- For supported labels, report balanced accuracy/macro-F1 or an ordinal/continuous
  metric matching the target type, plus source and reviewer provenance.
- Report confidence calibration (reliability/ECE or a documented alternative),
  precision/recall at the approved operating point, abstention coverage, and
  error conditional on accepting feedback.
- Tracking confidence is an observability signal. Do not call it clinical
  uncertainty or movement quality.

## Robustness and subgroup reporting

Every result should include sample counts and confidence intervals or a stated
uncertainty method for:

- source and held-out source;
- participant/profile and seated versus standing context;
- unilateral, bilateral, assisted, limited, absent, and unknown capability
  declarations;
- camera occlusion, missing joints, dropped frames, and low-confidence pose;
- exercise, side, tempo, ROM, and partial/complete repetitions.

If a subgroup has no real participants, state “not evaluated.” Synthetic limb
masking is an engineering robustness test, not target-population validation.

## Planned Motion-JEPA evaluation

AF-MJEPA is evaluated as a training-only representation/teacher candidate, not
as a deployable model. Pretraining must obey the declared participant/source
split; exposure to held-out test participants invalidates a downstream
generalization claim even when no task labels were used.

Before distillation, report:

- finite-value and target-encoder/collapse diagnostics;
- frozen-probe performance for exercise, phase, repetition, and supported
  robustness tasks, with label coverage and split provenance;
- future-horizon or temporal-mask performance using the exact horizons and mask
  manifest, without interpreting latent loss as physical prediction accuracy;
- matched supervised-only versus JEPA-assisted student sequence metrics,
  abstention, subgroup, and causal-streaming behavior;
- teacher preparation/inference time, model size, memory, and total compute.

Do not accept a JEPA result because its latent loss decreases alone. The release
decision is based on a predeclared held-out product or robustness metric and the
full cost/limitations report. No JEPA score creates quality, clinical, or
target-population evidence.

## Planned recommendation evaluation

Evaluate recommendation in layers. First exhaustively test the deterministic
feasibility mask and candidate explanations; then evaluate ranking only over
eligible candidates. Report candidate-set size, exposure policy, user/time
split, recipe/catalog version, cold-start coverage, and whether an item was
actually shown to the user.

Required measures include:

- zero capability, posture, equipment, avoid-list, or approval-state violations;
- empty-candidate fallback success and no-silent-substitution rate;
- Recall@k, NDCG@k, or an explicitly justified alternative against the
  content/rules baseline;
- completion, pause, skip, swap, and user-reason rates on future user periods;
- approved substitution acceptance and failure reasons;
- routine dose/time/rest constraint satisfaction, variety, and user-edit
  preservation;
- score calibration, recipe coverage, and repetition/novelty concentration;
- breakdowns for capability profile, seated/wheelchair context, equipment,
  new versus returning users, exercise family, and empty candidate cases.

Behavioral ranking metrics do not establish exercise safety, medical benefit, or
clinical improvement. A neural ranker cannot pass release if it worsens hard
constraint behavior or removes the rules/manual fallback path.

## Runtime parity protocol

Before accepting a deployment candidate:

1. Compare Python feature tensors against native tensors on golden unilateral,
   declared-absent, occluded, missing-frame, timestamp-reset, and exercise-change
   fixtures.
2. Compare float Python versus exported runtime outputs within a versioned
   numeric tolerance for every head and decoder input.
3. Compare float versus post-training-quantized output and event traces on
   validation fixtures; calibration data must not come from test.
4. Measure end-to-end latency, p95 latency, memory, thermal behavior, and
   battery cost over complete sessions on the Android-first target device;
   repeat parity expectations on iOS before claiming platform support.
5. Verify that camera denial, permission changes, raw-frame retention, and raw-pose
   retention follow the consent contract.

## Release gates

| Gate | Demo | Research report | Product release | Current state |
|---|---|---|---|---|
| Family recognition | Useful smoke test | Sequence metrics and source splits | Stable operating point and subgroup evidence | Corrected-v1 benchmark exists |
| Repetition counting | Manual review permitted | Event/count metrics with locked split | Latency, rest false-event, reset, and parity gates | End performance and decoder work remain |
| Phase detection | Optional visualization | Macro-F1 with phase coverage | Stable feedback behavior and abstention | Corrected-v1 is historical baseline |
| Quality feedback | Must be disabled or labeled demo-only | Requires reviewed labels and coverage | Requires approved targets, calibration, and safety review | **Unavailable: quality coverage is 0%** |
| Confidence/abstention | Low-confidence UI may be shown | Calibration and coverage reported | Operating threshold, fallback, and audit trail | Tracking signal exists; no release calibration |
| Mobile latency/parity | Recorded prototype timing | Export/parity evidence | Device measurements and rollback bundle | **Unavailable: no native bridge/export** |
| Privacy behavior | Local-only design review | Reproducible fixture checks | Permission, retention, deletion, and audit evidence | Not demonstrated in production app |

Gates can be marked `pass`, `fail`, `blocked`, or `not-applicable` only with an
artifact manifest and reviewer. “Not measured” is not a pass.

## Report template

Every benchmark or release report must include:

- objective and claim boundary;
- repository, commit, config hash, schema/normalization/decoder versions;
- dataset manifests, checksums, licenses, participant/source splits;
- checkpoint parent, seed, trainable layers, effective supervised samples,
  compute/time budget, and stopping reason;
- primary sequence metrics, diagnostic window metrics, subgroup breakdowns,
  calibration/abstention, and runtime parity;
- artifact paths and hashes;
- known label gaps, prohibited interpretations, and next decision.

Corrected-v1 is the complete current benchmark. `v2-quality` is prepared only,
and `v2-quality-fixed` is partial; neither can be cited as a complete comparison
until all required sequence evaluation and manifests are present.
