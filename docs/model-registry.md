# AdaptFit model registry

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** model lifecycle, artifact ownership, deployment status, and release gates
> - **Last verified:** 2026-09-11
> - **Source commit:** `b3926d8` (R1 calibration and phase-weight correction training)
> - **Owner:** AdaptFit ML engineering
> - **Supersedes or supports:** consolidates model status from `current-state.md`, `scalable-ml-architecture.md`, `recommendation-model-plan.md`, and `motion-jepa-world-model-plan.md`
> - **Review trigger:** new model family, checkpoint, head, export, evaluation, or deployment target

This registry distinguishes model families, artifact variants, and deterministic
components. A row is current only when its source code/config and artifact
evidence exist. Planned and research rows are design contracts, not available
models.

## Model and component inventory

| ID | Kind | Inputs | Outputs | Status | Artifact owner | Deployment | Release gate |
|---|---|---|---|---|---|---|---|
| `movement-tcn-v1` | Neural movement model | `FeatureSchemaV1` 283 features, 128-frame causal window | family, phase, boundaries, pooled quality, tracking confidence | Current primary; corrected-v1 complete | ML/training engineering | Python reference; mobile unavailable | sequence metrics, decoder calibration, parity, latency |
| `movement-gru-v1` | Neural movement baseline | same 283-feature contract | same head family | Current comparison baseline; corrected-v1 complete | ML/training engineering | Python reference; mobile unavailable | matched participant split and sequence metrics |
| `movement-tcn-quality-v2` | Neural movement variant | 283 features plus quality-label configuration | expert-quality head plus movement heads | Partial; v2-quality-fixed TCN evaluated, quality-4 coverage zero | ML/training engineering | Not released | valid quality labels, complete comparison, calibration |
| `movement-tcn-boundary-r1` | Warm-start experiment variant | same 283-feature contract, corrected-v1 normalization | existing movement heads; boundary-focused fine-tuning | Original R1, corrected-mask, and phase-weight follow-up decoder gates blocked | ML/training engineering | Isolated Python experiment only | validation-only failure analysis/intervention, then decoder gate and no-leakage test evaluation |
| `movement-tcn-phase-weight-r1` | Warm-start experiment variant | same 283-feature contract and corrected-mask normalization | existing movement heads with corrected phase class weighting | Trained; validation-only decoder gate blocked; test locked | ML/training engineering | Isolated Python experiment only | targeted failure analysis/intervention, then decoder gate and no-leakage test evaluation |
| `movement-tcn-last-block-event-support-r1` | Warm-start experiment variant | same 283-feature contract and corrected-mask normalization | existing movement heads plus the final TCN residual block | Trained; validation-only decoder gate blocked; test locked | ML/training engineering | Isolated Python experiment only | final-two-block validation gate, then no-leakage test evaluation only if the gate passes |
| `movement-tcn-last-two-blocks-event-support-r1` | Warm-start experiment variant | same 283-feature contract and corrected-mask normalization | existing movement heads plus the final two TCN residual blocks | Trained; validation-only decoder gate blocked; test locked | ML/training engineering | Isolated Python experiment only | full-TCN pilot and validation gate, then no-leakage test evaluation only if the gate passes |
| `movement-tcn-full-finetune-event-support-pilot-r1` | Warm-start experiment variant | same 283-feature contract and corrected-mask normalization | full TCN and existing movement heads | Trained; validation-only decoder gate blocked; test locked | ML/training engineering | Isolated Python experiment only | external reviewed boundary supervision or additional representative recordings before another model-only intervention |
| `movement-tcn-uco-tracking-target-correction-r1` | Data-contract follow-up variant | same 283-feature contract and UCO tracking-target-corrected normalization | full TCN and existing movement heads | Trained; validation-only decoder gate blocked; test locked | ML/training engineering | Isolated Python experiment only | reviewed UCO/REHAB24-6 event supervision before another model-only intervention |
| `movement-tcn-uco-boundary-positive-weight-r1` | Loss-weight follow-up variant | same 283-feature contract and tracking-target-corrected normalization | full TCN and existing movement heads | Trained; validation-only decoder gate blocked; cap rejected; test locked | ML/training engineering | Isolated Python experiment only | do not repeat cap increase; reviewed event supervision required |
| `movement-tcn-ulred-boundary-support-r1` | Data-source follow-up variant | same 283-feature contract and tracking-target-corrected normalization | full TCN and existing movement heads | Trained; validation-only decoder gate blocked; test locked | ML/training engineering | Isolated Python experiment only | reviewed hold/apex/end supervision required; boundary-only UL-RED spans are insufficient |
| `recommendation-rules-v1` | Deterministic selector | capability, equipment, reviewed recipes, goals | eligible candidates and fallback | Planned | Product/safety engineering | Product/runtime target | exhaustive zero-rule-violation fixtures |
| `recommendation-ranker-v1` | Planned neural ranker | eligible recipes plus consented profile/history features | ranked candidates and confidence | Research backlog; no checkpoint | Product/ML engineering | Future service or local runtime | user/time-held-out ranking, privacy, hard-mask audit |
| `af-mjepa-v1` | Research-only teacher/world model | masked/contextual canonical pose sequences | latent predictions or distillation targets | Research backlog; no code/checkpoint | ML research | Never directly on mobile | collapse checks and matched student improvement |

The pose estimator is an external runtime dependency, not an AdaptFit model
row. The decoder, feasibility mask, abstention logic, and recipe approval are
deterministic components and cannot be replaced by a model score.

The Python decoder implementation is `training/src/decoder.py` (`decoder.v1`);
its calibration report is a validation artifact, not a model checkpoint. The R1
row is deliberately separate from `movement-tcn-v1` until a trained child has
its own manifest and validation decision.

The corrected mask follow-up has its own checkpoint and validation-only decoder
report under `artifacts/r1-tcn-boundary-mask-correction/`, but remains blocked:
the selected decoder predicts zero accepted events on 896 validation sequences,
with 240 missed target events and end F1 `0.0`. It is not a release candidate
and has no test metrics. Its validation-only failure analysis is recorded in
`artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_failure_analysis.md`;
the dominant issue is missing predicted hold/apex support, not unavailable-label
supervision. A separate phase-weight correction candidate is now trained at
`artifacts/r1-tcn-phase-weight-correction/`: it reuses the corrected-mask
prepared data, has best validation sequence score `0.6679357` at epoch 9, and
has a blocked validation-only decoder calibration: end F1 is `0.153846` and
false events on empty sequences are `15`. It has no test metrics. The final
block candidate then trained at
`artifacts/r1-tcn-last-block-event-support/`, reached validation sequence score
`0.6791891`, and remained blocked after calibration (end F1 `0.094862`, 8
empty-target false events). The final-two-block candidate then trained at
`training/configs/experiments/r1_tcn_last_two_blocks_event_support.yaml` and
writes to `artifacts/r1-tcn-last-two-blocks-event-support/`: it unfreezes only
`blocks.3.*`, `blocks.4.*`, and the existing heads, for `113,106` trainable
parameters, while holding data, labels, loss weights, decoder thresholds, and
test lock constant. It reached validation sequence score `0.7040537`, but
calibration remained blocked with end F1 `0.299674` and 63 empty-target false
events. The full-TCN pilot then trained at
`training/configs/experiments/r1_tcn_full_finetune_event_support_pilot.yaml`
and writes to `artifacts/r1-tcn-full-finetune-event-support-pilot/`: all
`307,410` parameters were trainable, best epoch 6 of 8 reached validation
sequence score `0.717668`, and validation-only calibration remained blocked
with end F1 `0.296053`, count MAE `0.330357`, and 60 empty-target false events.
Failure analysis still finds `219/219` segmentation starts below the tracking
floor and only `101/240` target ends with nearby end signal. Stop model-only
escalation and request reviewed boundary supervision or additional representative
labeled recordings.

The UCO follow-ups are also blocked and are not release candidates. The
tracking-target correction at
`artifacts/r1-tcn-uco-tracking-target-correction/` removes partial-pose
denominator dilution and reaches validation sequence score `0.6176060`, but
validation-only calibration remains blocked with end F1 `0.070130`, count MAE
`0.584`, and 7 empty-target false events. UCO contributes 507 strong boundary
pairs but no per-frame hold/apex labels; failure analysis finds `0/507` nearby
starts, `1/507` nearby ends, and `0/507` apex-qualified targets. The follow-up
at `artifacts/r1-tcn-uco-boundary-positive-weight/` raises the sparse-boundary
positive-weight cap to `64.0`, lowers validation sequence score to `0.5252875`,
and increases empty-target false events to 24; the cap is rejected. Reviewed
event supervision or additional representative labeled recordings are required
before another model-only intervention.

The subsequent local-source audit found that UL-RED archives contain explicit
`marker-less/3Rep_Sxx.csv` spans for 219 of 219 available R3 recordings. The
adapter now converts those zero-based CSV indices to one-based AMC frame IDs,
and `r1-tcn-ulred-boundary-support-r1` records the isolated overlay experiment.
It reached validation sequence score `0.6013545`, but validation-only
calibration remained blocked with end F1 `0.070718`, count MAE `0.684`, and 8
empty-target false events. These spans are boundary-only and do not add
hold/apex labels, so reviewed hold/apex/end supervision is still required.

## Artifact requirements

Every model artifact must carry `ModelArtifactManifestV1` with:

- model ID and variant;
- model/schema/normalization/decoder versions;
- source commit and config hash;
- dataset manifest and participant/source split;
- seed, parent checkpoint, trainable layers, and loss weights;
- checkpoint hash and artifact paths;
- validation/test metrics with split and source scope;
- export/runtime/quantization state;
- known limitations and prohibited interpretations.

The authoritative inventory is [artifact registry](artifact-registry.md).
`best.pt` without this manifest is not a releasable model bundle.

## Non-claims

No row currently establishes clinical validity, medical safety, muscle force,
muscle activation, joint loading, or target-population performance. Quality
head availability requires valid reviewed labels; model architecture alone does
not create quality supervision. The recommendation ranker cannot bypass the
hard feasibility mask, and AF-MJEPA cannot be described as a shipped model.

## Promotion states

`research-backlog → prepared → trained → evaluated → parity-checked →
release-candidate → released`.

Promotion requires the evidence in the applicable release gate. A failed or
missing gate moves the row to `blocked` or `unavailable`; it does not get
silently treated as an earlier successful state.

## External methodology candidates

External methods are research inputs, not additional current model families. The
[cited methodology reuse report](research-method-reuse-report.md) records the
source, license, input mismatch, and experiment gate for each candidate.

| Candidate | Intended role | Current status | Boundary |
|---|---|---|---|
| TransRAC / SSTRAC | Offline density or temporal-correlation teacher | Research candidate; no checkpoint | Cannot replace the causal TCN or decoder |
| RepNet | Pose-preserving synthetic repetition augmentation and TSM diagnostic | Research candidate | Synthetic lineage cannot create participant or clinical evidence |
| PoseRAC | Salient phase/apex annotation aid | Research candidate | Does not replace full start/end, pause, or partial-rep labels |
| MotionBERT / Skeleton2vec | Offline representation pretraining reference | Research candidate | Format, license, and adapter review required |
| Recommendation retrieval/ranking methods | Offline recommender baselines | Planned research | Hard feasibility and manual fallback remain authoritative |
| I-JEPA / V-JEPA / V-JEPA2 | JEPA objective and target-encoder references | Research references | No external JEPA code or teacher ships in the mobile bundle |

A candidate may become a registry row only after its AdaptFit artifact manifest,
split provenance, evaluation evidence, runtime status, and limitations are present.
