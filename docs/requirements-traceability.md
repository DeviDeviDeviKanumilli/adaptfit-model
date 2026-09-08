# AdaptFit requirements and evidence traceability

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** mapping from product requirements and claims to contracts, code, tests, artifacts, and release gates
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `1a46f38` (documentation revision base)
> - **Owner:** AdaptFit engineering, product, and safety review
> - **Supersedes or supports:** supports `product-scope.md`, `evaluation-protocol.md`, `model-registry.md`, and `project-forward-plan.md`
> - **Review trigger:** requirement, public claim, contract, test, artifact, or release-gate change

Every user-facing capability or published metric must be traceable from its
requirement to evidence. A missing link makes the claim unavailable rather than
an invitation to infer.

## Traceability table

| Requirement/claim | Contract | Implementation | Test | Artifact/evidence | Gate and limitation |
|---|---|---|---|---|---|
| Recognize movement family | `MovementPredictionV1` | `training/src/models/heads.py` | model/pipeline tests | corrected-v1 evaluation | participant/source split; not clinical validation |
| Count repetitions | `WorkoutEventV1` | planned decoder over boundary heads | planned decoder fixtures | sequence predictions and metrics | end-boundary weakness must be resolved |
| Provide quality feedback | quality fields and masks | quality heads | label-coverage tests | none with valid four-head coverage | unavailable while coverage is zero |
| Offer compatible exercise | capability/recipe/recommendation contracts | planned deterministic filter | planned feasibility fixtures | reviewed catalog hash | catalog support is not camera validation |
| Personalize recommendations | request/result/feedback contracts | planned ranker | planned user/time-held-out tests | no behavioral dataset | unavailable until consented history exists |
| Run on device | artifact/deployment contracts | no production bridge | no native parity suite | no export/quantized bundle | unavailable |
| Protect local data | privacy lifecycle contract | no production mobile implementation | planned redaction/reset fixtures | no privacy audit artifact | design requirement, not passed evidence |

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
