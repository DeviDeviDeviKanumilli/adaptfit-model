# AdaptFit Recommendation Model Plan

> **Documentation metadata**
> - **Status:** research-backlog
> - **Authority:** proposed workout-selection, substitution, and personalization architecture; reviewed recipe and safety rules remain authoritative
> - **Last verified:** 2026-09-08
> - **Source commit:** `613ff12` (repository base; recommendation system is not implemented)
> - **Owner:** AdaptFit product and ML engineering
> - **Supersedes or supports:** supports `product-scope.md`, `exercise-and-capability-schema.md`, `contracts-and-schemas.md`, `data-and-training-plan.md`, and `project-forward-plan.md`
> - **Review trigger:** recipe schema, capability/equipment rule, feedback-log design, ranking experiment, or safety review

The cross-subsystem runtime boundary is in [system context and data flow](system-context-and-dataflow.md).
Use [recipe catalog and review](recipe-catalog-and-review.md) for approval and
catalog hashes, [privacy and data lifecycle](privacy-and-data-lifecycle.md) for
local history, and the versioned fixtures under
[`schemas/`](schemas/) and [`examples/contracts/`](examples/contracts/) for
payload examples. This plan remains research-backlog until the deterministic
candidate service and its fixtures exist.

> **Current-state boundary:** The current repository has no recommendation
> model, recommendation checkpoint, workout-history store, learned substitution
> ranker, or production exercise-selection service. Capability fields exist in
> the profile and pose contracts, but a complete candidate-selection system is
> not implemented.

## Purpose and product boundary

The recommender selects a workout from reviewed `ExerciseRecipeV1` records. It
does not analyze camera frames, count repetitions, estimate movement phase, or
replace the causal TCN. The movement model handles live exercise tracking; the
recommender handles what to do next and which compatible alternative to offer.

The product path is:

```text
CapabilityProfileV1 + goals + preferences + equipment + history
                              |
                              v
        deterministic feasibility and safety filter
                              |
                     eligible recipe set
                              |
                              v
              recommendation ranker / routine planner
                              |
                 ranked exercises and substitutions
                              |
                              v
               reviewed WorkoutPlan / manual fallback
```

The neural ranker may order **eligible** recipes. It may not make an impossible
or unreviewed recipe eligible, infer disability from appearance, waive a
required equipment or posture rule, or make a clinical safety decision.

## Components

### 1. Recipe catalog and hard feasibility mask

The catalog remains the source of truth for stable exercise and variant IDs,
required and optional limbs, posture, equipment, movement family, substitution
group, dose limits, camera requirements, allowed feedback, confidence floor,
review status, and reviewer metadata.

The feasibility filter removes a candidate when any required condition fails:

- recipe is not `approved` or is retired;
- required position is incompatible with the declared profile;
- a required limb is absent, unknown, or not approved for the variant;
- required equipment is unavailable or not confirmed for the session;
- a movement appears in the user's explicit avoid list;
- required camera support is unavailable for a camera-tracked workout;
- recipe dose, rest, or session constraints cannot be satisfied.

`unknown` capability is unresolved information, not permission to assume
availability. Camera occlusion does not change the capability profile. If no
approved candidate remains, the product must explain the conflict and offer
profile editing, manual selection, or a safe stop.

### 2. Candidate generator

The first candidate generator should be deterministic and catalog-backed. It
should return the eligible recipe IDs, the failed-rule reasons for excluded
recipes, and the feature values used by the ranker. This makes an empty result,
substitution, and human review explainable before any neural model is added.

Candidate generation should support:

- a fresh-session set of compatible exercises;
- continuation of a routine while respecting dose and rest constraints;
- a replacement restricted to the same reviewed `substitution_group`;
- a no-equipment or alternate-posture fallback only when a recipe explicitly
  approves it.

### 3. Neural workout ranker (planned)

The recommender should be a separate user/context-to-exercise model. A compact
two-tower or hybrid model is the preferred first neural experiment:

```text
user/context encoder  -> user/session embedding --+
                                                   +--> compatibility score
recipe encoder        -> exercise embedding ------+
```

The user/context tower may encode capability states, position, equipment,
goals, preferred difficulty, available time, recent workload, and explicit
preferences. The recipe tower may encode movement family, posture, body-demand
metadata, difficulty, equipment, substitution group, dose, camera support, and
reviewed feedback dimensions. All categorical fields use versioned IDs and
embeddings; numerical fields carry units and bounded normalization.

The ranker receives only the feasible candidate set. Its score may combine:

```text
score(recipe | user, session) =
    goal_fit
  + preference_fit
  + capability/context fit
  + appropriate novelty and routine variety
  + history-based expected usefulness
```

The terms and weights must be versioned. A score is a ranking signal, not a
probability of safety, medical benefit, or successful rehabilitation.

### 4. Routine planner (planned, deterministic first)

Ranking individual exercises is not the same as assembling a safe routine. A
constrained planner should choose a sequence subject to approved dose, rest,
time, posture, equipment, movement-family variety, and substitution rules. Start
with a deterministic greedy or integer-programming-style planner over ranked
recipes. Add a learned sequence policy only if the deterministic planner and
feedback data show a measurable limitation.

The planner must expose why an item was selected, skipped, or substituted. It
must preserve user edits and never silently replace an exercise after the
session begins.

### 5. Substitution ranker (planned)

Substitution is a constrained use of the same ranker, not an unrestricted
similarity search. When a user skips an exercise or equipment becomes
unavailable, filter to recipes in the approved substitution group, re-run the
hard feasibility mask, and rank the remaining alternatives. Require explicit
confirmation before changing the session. Record the original recipe, proposed
replacement, reason, and profile/equipment version.

### 6. Personalization layer (planned)

Begin with explicit onboarding preferences and non-gradient local history. A
future user embedding can learn from completed, skipped, swapped, repeated,
favorited, paused, and manually rejected exercises. These events are ambiguous:
a skip may indicate pain, equipment loss, fatigue, time pressure, or dislike.
The event schema must retain the user's stated reason where available instead of
turning every skip into a negative label.

Personalization should be local-first. A model update must not require raw
camera frames, raw pose, or cloud workout history. If a future sync path is
added, it requires explicit consent, retention rules, deletion behavior, and a
separate privacy review.

## Cold start and rollout

The recommender must be useful before it has behavioral history:

1. **R0 — catalog and rules:** approved recipes plus deterministic feasibility.
2. **R1 — content ranker:** a transparent linear or small MLP score using goals,
   capabilities, equipment, posture, difficulty, and recipe metadata.
3. **R2 — feedback logging:** versioned exposure, selection, completion, skip,
   swap, pause, and user-reason events with no raw media.
4. **R3 — neural ranker:** two-tower or hybrid ranking trained only on valid
   eligible-candidate exposures and time/user-held-out splits.
5. **R4 — local personalization:** update a small user representation from
   consented local history; retain the content/rules fallback.
6. **R5 — routine optimization:** add sequence-level planning only after
   candidate ranking and constraint behavior are stable.

The first release can ship R0 or R1. A neural ranker is not a prerequisite for
the movement TCN or for a safe demo. The product must retain a deterministic
fallback if the ranker is unavailable, uncertain, or incompatible with the
recipe/runtime version.

## Training data and objective

The recommendation dataset is separate from pose-training windows. Each logged
example records the request context, eligible candidate set, exposure position,
selected recipe, outcome, reason codes, recipe/profile versions, and timestamp.
Do not train on candidates that the hard filter would have rejected.

Useful signals include:

- explicit goal, difficulty, equipment, and preference choices;
- selected, started, completed, paused, skipped, swapped, repeated, or manually
  rejected recipes;
- user-stated comfort, difficulty, or “not available today” reasons;
- session duration, rest adherence, and dose completion;
- recipe review status and deterministic feasibility reasons.

Do not use unreviewed camera quality or pose predictions as a proxy for medical
benefit. Movement-model outputs may later provide session summaries such as
observed count or tracking confidence, but the integration must be declared and
evaluated separately from the recommendation label.

Start with a pointwise or pairwise objective on eligible candidates. A listwise
loss or contextual bandit is a later option after exposure logging is reliable.
Negative sampling must respect what the user could actually choose and must
distinguish unavailable, skipped, rejected, and merely unexposed recipes.

Training splits must be user-level and time-aware. Keep a future-period holdout
for behavioral generalization, a cold-start holdout for users with no history,
and a recipe/source holdout for catalog expansion. Do not use test-period
outcomes to tune ranking weights or feasibility rules.

## Evaluation and release gates

Evaluate the system in layers:

| Gate | Required evidence |
|---|---|
| Feasibility safety | Zero recommendations that violate approved capability, posture, equipment, avoid-list, or review-status rules in exhaustive fixtures. |
| Empty-candidate behavior | Clear profile-edit, manual-selection, or safe-stop path with no silent substitution. |
| Cold-start ranking | Top-k or NDCG/Recall@k on a user/time-held-out set with explicit candidate-set size and exposure policy. |
| Personalization | Improvement over the content/rules baseline on future user periods without subgroup or novelty collapse. |
| Substitution | Accepted replacement rate and reason-coded failures within approved substitution groups. |
| Routine quality | Dose/time/rest constraint satisfaction, variety/coverage, and user-edit preservation. |
| Calibration | Score/recommendation confidence is evaluated separately from movement tracking confidence. |
| Privacy | No raw frames/raw pose in recommendation inputs or logs; consent, deletion, and local fallback are tested. |

Required breakdowns include capability profile, seated/wheelchair context,
equipment availability, new versus returning user, exercise family, and empty
candidate cases. A recommendation metric cannot establish clinical benefit or
exercise safety by itself.

## Interfaces and artifacts

The planned interfaces are defined in
[contracts-and-schemas.md](contracts-and-schemas.md):

- `RecommendationRequestV1` — profile, goals, preferences, equipment, session
  constraints, and history summary;
- `EligibleRecipeSetV1` — recipe IDs, feasibility decisions, failed-rule
  reasons, and catalog/version hashes;
- `WorkoutRecommendationV1` — ranked recipe/variant IDs, score components,
  reason codes, model/rules versions, and fallback state;
- `WorkoutFeedbackEventV1` — exposure, selection, completion, skip, swap,
  pause, rejection, and user-provided reason with consent metadata.

Each recommender artifact must include the recipe catalog hash, capability and
equipment enum versions, feature/normalization version, training split,
checkpoint, config hash, candidate-generator version, ranking metrics,
subgroup metrics, and known limitations. A model bundle must not load a ranker
whose recipe or capability schema is newer than the product runtime.

## Open questions and stop conditions

- Which first-release goals and difficulty vocabulary are approved by product?
- Which user-history events may be stored locally, and for how long?
- Is a local-only ranker sufficient for the challenge demo, or is sync needed?
- What is the approved minimum candidate-set size before showing a routine?
- Which recipe metadata is human-reviewed and safe to expose as an explanation?
- Does a neural ranker beat the content/rules baseline after accounting for data
  collection and privacy cost?

Stop and fall back to rules when recipe hashes, capability/equipment enums,
consent state, or candidate feasibility cannot be verified. Defer neural
personalization when history is sparse, exposure is biased, or the model
increases constraint violations, subgroup disparity, repetitive routines, or
unexplained substitutions.

## Source context

The product intent is also recorded in the [Congressional App Challenge
planning document](https://docs.google.com/document/d/12Kxnl1_AVT4UkomWXMYT8k7-e1vCFo0nHpe-5YWBf0I/edit?tab=t.w4bv5fg0djh),
which describes capability/equipment-aware routines, exercise substitution,
and a separate neural workout recommender. That document is product context;
the repository contracts, recipe review state, manifests, and evaluation gates
remain the engineering source of truth.
