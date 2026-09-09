# AdaptFit model registry

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** model lifecycle, artifact ownership, deployment status, and release gates
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `613ff12` (documentation revision base)
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
| `movement-tcn-boundary-r1` | Warm-start experiment variant | same 283-feature contract, corrected-v1 normalization | existing movement heads; boundary-focused fine-tuning | Configured, not trained | ML/training engineering | Isolated Python experiment only | two-epoch smoke, validation improvement, decoder recheck, no test leakage |
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
