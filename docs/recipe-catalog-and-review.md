# Recipe catalog and review protocol

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** reviewed exercise recipes, approval states, catalog hashes, and deterministic eligibility rules
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `613ff12` (documentation revision base)
> - **Owner:** AdaptFit product and safety review
> - **Supersedes or supports:** supports `exercise-and-capability-schema.md`, `recommendation-model-plan.md`, and `contracts-and-schemas.md`
> - **Review trigger:** recipe field, capability rule, equipment requirement, substitution group, or approval decision change

The catalog answers which exercise variants exist and which are eligible for a
profile. It is separate from the movement model's ability to recognize or count
an exercise. Catalog support is not camera validation or clinical suitability.

## Catalog record requirements

Each record is an `ExerciseRecipeV1` with a stable `exercise_id`, `variant_id`,
semantic `recipe_version`, movement family, posture, required/optional limbs,
relevant joints, equipment, unilateral/bilateral support, substitution group,
phase sequence, dose constraints, allowed feedback dimensions, camera
requirements, confidence floor, and review metadata.

Recommendation metadata may add goals, difficulty, duration, and demand tags,
but those are reviewed product metadata. They are not claims about force,
muscle activation, clinical benefit, or medical suitability.

## Initial recipe registry

| Stable exercise ID | Initial variants | Required context | Status |
|---|---|---|---|
| `seated_one_arm_biceps_curl` | `left_no_equipment`, `right_no_equipment` | seated; one usable arm | draft until human review |
| `seated_one_arm_band_row` | left/right approved anchor and band variants | seated; usable arm; reviewed band/anchor | draft until setup review |
| `seated_single_leg_knee_extension` | left/right | seated; one usable leg; stable support | draft until human review |
| `seated_single_leg_march` | left/right/alternating | seated; usable leg; stable support | draft until human review |
| `seated_forward_reach` | one-arm and bilateral reviewed variants | stable seat; reachable target; camera view | new tracking recipe; draft |

Standing, weight-bearing, and wall-supported exercises remain deferred from the
first inclusive release until their posture and capability review is complete.

## Approval lifecycle

`draft → safety-review → catalog-approved → camera-validated → retired`.

- `draft`: shape is valid but must not be recommended.
- `safety-review`: product/safety reviewer is checking posture, capability,
  equipment, dose, substitution, and language.
- `catalog-approved`: eligible for deterministic recommendation if all hard
  rules pass; camera support may still be unavailable.
- `camera-validated`: the supported view, feature dependencies, and tracking
  behavior have evidence for the declared use.
- `retired`: preserved for history but cannot enter a new candidate set.

ML training cannot promote a recipe. Every approval stores reviewer role, date,
review notes, catalog hash, and affected schema version.

## Deterministic eligibility

The candidate generator runs before ranking and applies, in order:

1. recipe approval and retirement state;
2. required posture and mobility context;
3. required limb capability (`absent` never satisfies `required`);
4. equipment availability and setup requirements;
5. movements-to-avoid and self-reported range constraints;
6. camera/view and minimum-joint-visibility requirements;
7. variant and substitution-group constraints.

An unknown capability remains unknown. An occluded joint changes observation
state and tracking confidence; it does not mutate the user's capability profile.
If no candidate survives, return an explicit empty `EligibleRecipeSetV1` with a
reason code and reviewed next action. Never ask the ranker to recover an
infeasible set.

## Substitution and catalog hashing

Substitution is allowed only inside the recipe's reviewed substitution group.
The replacement is re-run through the entire feasibility filter and requires
explicit confirmation when equipment or posture changes. Requests,
recommendations, feedback events, and model bundles carry the catalog hash so a
ranker cannot select against stale recipe requirements.

## Required review fixtures

Maintain fixtures for one-arm, one-leg, seated/wheelchair, assisted-limb,
unknown-capability, occluded-joint, unavailable-equipment, and empty-candidate
cases. These fixtures prove rule behavior; they do not prove exercise safety or
target-population outcomes.
