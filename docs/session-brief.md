# Complete Session Brief

> **Documentation metadata**
> - **Status:** historical
> - **Authority:** dated planning/session record
> - **Last verified:** 2026-09-07
> - **Source commit:** `e75ba65`
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** historical context only; current-state and project-forward-plan override its stale status statements
> - **Review trigger:** none for current behavior; update only when preserving a new dated session record

> **Historical interpretation:** This brief preserves planning context from a
> prior session. It may explain why work was proposed, but it cannot establish
> current code, data, artifact, safety, or deployment behavior.

## Original product idea

Many fitness applications assume that people can stand, run, jump, or use both arms. AdaptFit should automatically adapt routines around the movements a person can safely perform.

## Product direction selected

The first population focus is people with bodily limitations, beginning with:

1. Missing or unavailable arm, including one-arm users.
2. Missing or unavailable leg, including one-leg users.
3. Wheelchair and seated users.

The product will expand to other limitations after this foundation works.

Users will self-report their capabilities during onboarding. The camera model should evaluate the selected movement; it should not attempt to diagnose or identify a disability.

## Existing project

The starting repository is [PeddieHacks26](https://github.com/DeviDeviDeviKanumilli/PeddieHacks26), whose product is called AdaptFit. The repository already contains:

- React Native/Expo mobile work.
- A Fastify API and Supabase/Prisma-related data work.
- A Python MediaPipe calibration and exercise-analysis lab.
- A TypeScript intelligence package with deterministic feature, repetition, and orchestration logic.
- Exercise metadata describing body demands, capabilities, equipment, goals, muscles, tracking profiles, and form rules.

## Existing ML state

The current system is primarily pretrained pose estimation plus deterministic movement logic:

- MediaPipe Pose Landmarker Lite extracts one pose.
- The Python analyzer computes angles, range of motion, and repetition state.
- Mobile Android code exposes a small set of angles and confidence values.
- The intelligence package is deterministic and does not currently load learned weights.

This is a strong foundation for data collection and integration, but it is not yet a learned, anatomy-informed personalization model.

## Temporal model decision

The scalable architecture will use a shared temporal encoder rather than one model per disability or exercise.

- Production candidate: causal dilated TCN.
- Baseline: small causal GRU.
- Inputs: anatomy-normalized pose features, velocities, confidence values, and explicit limb-availability masks.
- Outputs: movement phase, repetition boundaries, quality dimensions, and uncertainty.
- Adaptation: deterministic capability and safety rules surrounding the model.

The on-device model should remain small, quantized, and capable of real-time streaming inference.

## Initial exercise set

The recommended first five exercises are:

1. Seated one-arm biceps curl.
2. Seated one-arm band row.
3. Seated single-leg knee extension.
4. Seated single-leg march or hip lift.
5. Seated forward reach.

These exercises are selected because they are seated and can be performed with one available arm or leg. Sit-to-stand and wall push-ups are deferred because they require weight-bearing or bilateral capability that many target users may not have.

## Data decision

There is currently no direct access to target-group participants for consented recordings. Development will therefore begin with public movement and rehabilitation datasets, plus synthetic variations such as limb masking, reduced range of motion, occlusion, altered tempo, and seated posture.

Synthetic data can test the architecture, but it cannot establish reliable performance for amputees, people with limb differences, or wheelchair users. Target-population data or appropriately licensed datasets will be needed before making strong claims about those groups.

## Working location

All future AdaptFit work belongs in:

`/Users/devk/AdaptFit/`

## Historical interpretation

- **True for the session:** this brief records the product scope, initial
  exercise idea, and public-data bootstrap assumption at the session date.
- **Still current:** capability-focused onboarding, on-device intent, and the
  prohibition on medical/clinical claims remain reflected in the canonical
  product and contract documents.
- **Superseded:** its unversioned roadmap and model assumptions are replaced by
  [current-state.md](current-state.md), [contracts-and-schemas.md](contracts-and-schemas.md),
  and [project-forward-plan.md](project-forward-plan.md).
- **Evidence:** this is a planning record and contains no checkpoint or
  participant-evaluation artifact.
- **Cannot prove:** implementation, model accuracy, target-population support,
  mobile parity, safety, or deployment readiness.
