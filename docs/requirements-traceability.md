# AdaptFit requirements and evidence traceability

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** mapping from product requirements and claims to contracts, code, tests, artifacts, and release gates
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `9fe47fb` (documentation revision base)
> - **Owner:** AdaptFit engineering, product, and safety review
> - **Supersedes or supports:** supports `product-scope.md`, `evaluation-protocol.md`, `model-registry.md`, and `project-forward-plan.md`
> - **Review trigger:** requirement, public claim, contract, test, artifact, or release-gate change

Every user-facing capability or published metric must be traceable from its
requirement to evidence. A missing link makes the claim unavailable rather than
an invitation to infer.

## Traceability table

| Requirement/claim | Contract | Implementation | Test | Artifact/evidence | Gate and limitation |
|---|---|---|---|---|---|
| Recognize movement family | `MovementPredictionV1` | `training/src/models/heads.py` | model/pipeline tests | `artifacts/corrected-v1` evaluation | participant/source split; not clinical validation |
| Estimate movement phase | `MovementPredictionV1` | `training/src/models/heads.py` | temporal/streaming tests | `v2-quality-fixed` evaluation | 5-class frame logits; requires phase ground truth |
| Count repetitions | `WorkoutEventV1` | planned decoder over boundary heads | planned decoder fixtures | sequence predictions and metrics | end-boundary weakness must be debounced |
| Provide observable quality feedback | `MovementPredictionV1` (`quality_logits`) | 4-head quality outputs in `heads.py` | label-coverage tests | none with valid four-head coverage | unavailable while quality coverage is zero |
| Provide expert composite quality score | `MovementPredictionV1` (`expert_quality_logits`) | expert quality head in `heads.py` | `test_v2_quality_pipeline.py` | `v2-quality-fixed` evaluation | ordinal clinical scale; not interchangeable with 4 observable heads |
| Profile user physical capabilities | `CapabilityProfileV1` | `training/src/data/schema.py` | `test_data_contracts.py` | valid/invalid profile fixtures | self-reported on-device profile; not a medical diagnosis |
| Author exercise recipe safety bounds | `ExerciseRecipeV1` | `docs/recipe-catalog-and-review.md` | documentation validator | catalog fixtures and review rubrics | reviewed safety bounds; requires human kinesiology review |
| Filter compatible exercises | `EligibleRecipeSetV1` | planned deterministic filter | planned feasibility fixtures | reviewed catalog hash | zero rule violations in exhaustive fixtures before ranking |
| Personalize routine recommendations | `RecommendationRequestV1` & `WorkoutRecommendationV1` | planned ranker | planned user/time-held-out tests | no behavioral dataset | unavailable until consented exposure history exists |
| Log workout exposure and feedback | `WorkoutFeedbackEventV1` | planned session logger | planned consent-state fixtures | local session feedback logs | strictly local; consent withdrawal must purge session history |
| Track dataset ingestion provenance | `DatasetManifestV1` | `training/src/data/adapters.py` | `test_data_contracts.py` | `data/manifests/*.sha256` | participant split auditing and license clearance required |
| Export on-device artifact manifest | `ModelArtifactManifestV1` | planned bundle exporter | planned native parity suite | model bundle manifests | on-device local execution; zero telemetry without consent |

## Claim record requirements

For each metric or public statement, record dataset/source, participants, split,
source commit, configuration hash, feature/model/normalization/decoder versions,
checkpoint/artifact path, label coverage, reviewer, approved wording, and
interpretation limits. Claims about safety, clinical benefit, force, muscle
activation, target-population validation, or on-device privacy require their
own evidence and cannot be inferred from movement metrics.

## Gate semantics

`required for demo`, `required for research reporting`, and `required for
product release` are separate labels. `unavailable` means evidence or
implementation is missing; it is not a failing score that can be averaged with
other gates.
