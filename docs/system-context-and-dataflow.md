# AdaptFit system context and data flow

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** system boundary and cross-subsystem ownership map; contracts and source code own exact fields and behavior
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `9fe47fb` (documentation revision base)
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** supports `current-state.md`, `contracts-and-schemas.md`, `scalable-ml-architecture.md`, and `on-device-deployment.md`
> - **Review trigger:** new runtime boundary, model family, event contract, storage path, or deployment target

This document is the one-page orientation for an implementation agent. It shows
how the current supervised movement path and the planned recommendation path
fit together. It does not imply that the mobile bridge, recommender, or
Motion-JEPA teacher already exists.

## Product/runtime flow

```text
declared capability + equipment + goals
                    │
                    ▼
          reviewed recipe catalog
                    │
                    ▼
       deterministic feasibility mask
        (capability, posture, equipment,
         avoid list, camera, review status)
                    │
                    ▼
    content/rules recommendation baseline
       └── planned neural ranker later
                    │
                    ▼
          selected exercise variant
                    │
       camera permission / view check
                    │
                    ▼
   native pose estimator → canonical 33-joint pose
                    │
                    ▼
             FeatureSchemaV1 (283)
       torso normalization, velocities,
       angles, confidence and capability masks
                    │
                    ▼
       causal TCN (primary) or GRU baseline
                    │
                    ▼
       prediction heads + tracking confidence
                    │
                    ▼
 deterministic decoder, abstention and safety rules
                    │
                    ▼
                 WorkoutEventV1
                    │
                    ▼
        local session history / feedback event
```

The ranker may order candidates that survive the feasibility mask. It cannot
make an unreviewed recipe eligible, waive a required limb or equipment rule, or
turn camera occlusion into a declared capability.

## Offline training and release flow

```text
source data + licenses
        → adapter + identity policy
        → DatasetManifestV1
        → participant/source split
        → canonical pose/features + normalization
        → supervised training or research teacher
        → validation and sequence-level evaluation
        → ModelArtifactManifestV1
        → Python/native golden fixtures
        → future mobile bundle
```

The offline flow is separate from live inference. Raw frames are not required
by the current training runtime after a source adapter has produced the
canonical prepared representation. AF-MJEPA is an offline, research-only
teacher path; its checkpoint never crosses the product runtime boundary unless
a later student experiment passes the release gates.

## Ownership and trust boundaries

| Boundary | Owner | Contract | Current state |
|---|---|---|---|
| Capability and recipe eligibility | Product and safety review | `CapabilityProfileV1`, `ExerciseRecipeV1`, planned recommendation contracts | Schemas exist; selector service is planned |
| Pose to feature conversion | Training/runtime engineering | `FeatureSchemaV1` | Implemented in Python; native bridge unavailable |
| Temporal prediction | ML engineering | `MovementPredictionV1` | TCN primary and GRU baseline artifacts exist |
| Decoding and event emission | Runtime engineering | `WorkoutEventV1` | Python streaming components exist; production bridge unavailable |
| Recommendation ranking | Product and ML engineering | `WorkoutRecommendationV1` | Deterministic design planned; no learned ranker |
| History and feedback | Product/privacy owner | `WorkoutFeedbackEventV1` | Contract planned; no production store |
| Artifact loading | Release engineering | `ModelArtifactManifestV1` | Manifest requirements documented; production bundle unavailable |

## Latency-sensitive steps

The live path must keep pose estimation, feature construction, temporal forward
passes, decoding, and event emission bounded per frame. Recommendation ranking
is session-level and may run before a workout or between exercises. Catalog
feasibility must be deterministic and must not wait for a neural ranker.

The current Python model uses 128-frame windows at approximately 30 FPS,
stride 8 for training, and a 125-frame causal TCN receptive field. A future
native runtime must prove timestamp, dropped-frame, reset, memory, thermal, and
float/quantized parity before claiming mobile availability.

## Unavailable and failure states

- Camera denied or required joints unavailable: abstain from dependent feedback
  and emit a reason code; do not infer a missing limb.
- Capability unknown: keep recipes requiring that capability unavailable until
  the user declares it or a reviewed fallback exists.
- No eligible candidate: show the empty state and request a profile/equipment
  correction; never rank an infeasible recipe.
- Schema or bundle mismatch: fail closed at load time and use only an explicitly
  compatible fallback bundle.
- Missing labels: mask the task and report zero coverage; never fabricate a
  negative target.

See [failure and recovery matrix](failure-and-recovery-matrix.md) for the
user-visible behavior, reason codes, fixtures, and owners for each case.
