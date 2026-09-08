# Exercise and Capability Schema

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** product recipe schema, capability rules, and reviewed exercise catalog
> - **Last verified:** 2026-09-07
> - **Source commit:** `e75ba65`
> - **Owner:** AdaptFit product and safety review
> - **Supersedes or supports:** canonical product-facing schema; contracts-and-schemas defines serialized compatibility fields
> - **Review trigger:** recipe, capability enum, equipment rule, safety review, or feedback-dimension change

## Initial exercise set

These are the recommended first-release exercises. They are deliberately seated and support unilateral execution where possible.

| Exercise | Primary use | Supported variants | Main constraints |
|---|---|---|---|
| Seated one-arm biceps curl | Upper-limb flexion and rep tracking | Left-only, right-only, bilateral if available | Requires one usable arm and safe elbow movement |
| Seated one-arm band row | Upper-limb pulling and posture | Left-only, right-only, bilateral if available | Requires an approved anchor/band setup |
| Seated single-leg knee extension | Lower-limb extension | Left-only, right-only | Requires one usable leg and stable seating |
| Seated single-leg march or hip lift | Hip flexion and lower-limb movement | Left-only, right-only, alternating | Requires stable seated posture and available leg movement |
| Seated forward reach | Reach and trunk-control movement | One-arm or bilateral | Requires a stable seat and an appropriate reach target |

The forward-reach recipe is a new tracking recipe relative to the current project and will need to be added to the canonical catalog.

Sit-to-stand and wall push-up should be deferred from the first inclusive release because they require standing, weight-bearing, or capabilities that many wheelchair and lower-limb users may not have.

All exercises need expert review before being presented as suitable for a specific limitation.

## Capability profile

Suggested canonical representation:

```json
{
  "position": "seated",
  "mobility_context": "wheelchair",
  "limbs": {
    "left_arm": "available",
    "right_arm": "absent",
    "left_leg": "limited",
    "right_leg": "available"
  },
  "equipment": ["none", "resistance_band"],
  "user_limits": {
    "range_of_motion": "self_reported",
    "movements_to_avoid": []
  }
}
```

Allowed limb states should include:

- `available`
- `limited`
- `absent`
- `assisted`
- `unknown`

`unknown` means the user has not declared a capability. Camera observation uses
an `observed_mask` plus an observation state such as `occluded_or_unknown`.
Keeping those fields separate prevents the model from confusing a camera
failure with a user capability: `absent` is a declared capability, while
`occluded_or_unknown` is a frame-level observation problem.

## Exercise recipe

Every exercise should have one canonical recipe with:

- Stable exercise identifier.
- Movement family.
- Required position.
- Required and optional limbs.
- Required joints.
- Primary and secondary body roles.
- Equipment options.
- Allowed unilateral and seated variants.
- Expected phase sequence.
- ROM and tempo targets.
- Confidence floor.
- Form rules.
- Supported capability profiles.
- Human review status.

The Python catalog, mobile recipes, TypeScript domain catalog, and database records should eventually be generated from or validated against this schema.

## Versioned recipe shape

The product-facing serialized form is `ExerciseRecipeV1` in
[contracts-and-schemas.md](contracts-and-schemas.md). At minimum, each record
must contain:

```json
{
  "exercise_id": "seated_one_arm_biceps_curl",
  "variant_id": "left_no_equipment",
  "recipe_version": "1.0.0",
  "movement_family": "upper_limb_flexion",
  "required_position": ["seated", "wheelchair"],
  "required_limbs": ["left_arm"],
  "optional_limbs": [],
  "relevant_joints": ["left_shoulder", "left_elbow", "left_wrist"],
  "required_equipment": [],
  "optional_equipment": [],
  "unilateral_support": true,
  "bilateral_support": false,
  "substitution_group": "seated_elbow_flexion",
  "expected_phase_sequence": ["rest", "concentric", "hold", "eccentric"],
  "dose_constraints": {"min_reps": 1, "max_reps": 30, "rest_seconds": 30},
  "allowed_feedback_dimensions": ["count", "phase", "tracking_confidence"],
  "camera_requirements": {"view": "front_or_oblique", "min_visibility": 0.75},
  "confidence_floor": 0.70,
  "review_status": "draft",
  "reviewer": null,
  "reviewed_at": null
}
```

The example is a shape example, not approval or evidence that all listed
feedback is currently supported. Stable IDs are database/API keys; display
names can change. Enum changes require a schema version and migration note.
Numeric values carry units and inclusive bounds. Missing optional fields are
`null` or an empty list according to the contract; an omitted field is invalid.

## Required examples and edge cases

| Profile/input | Expected catalog and runtime behavior |
|---|---|
| One-arm user (`right_arm=absent`) | Offer only recipes whose `required_limbs` can be met by the left arm or a reviewed substitution; never create a bilateral requirement from an observed right-arm pose. |
| One-leg user (`left_leg=absent`) | Remove bilateral lower-limb recipes; retain a reviewed right-only variant if its posture and balance rules permit it. |
| Seated user/wheelchair | Exclude standing/weight-bearing recipes; require stable-seat and camera-view checks. |
| Assisted limb | Use only variants whose review names assisted use; do not convert assisted to available. |
| Unknown capability | Ask or keep the candidate unavailable when the limb is required. |
| Occluded joint | Keep the profile unchanged; suppress dependent feedback and emit a tracking reason code. |
| Equipment lost mid-workout | Pause and revalidate; switch only to an approved substitution after explicit confirmation. |

## Ownership and synchronization

The Python catalog is the reference during model development. A generated or
checked TypeScript/mobile/database representation must preserve IDs, versions,
enums, required joints, and review status. CI or a release checklist should
compare catalog hashes and reject a mobile recipe whose schema version is newer
than the model/runtime it invokes. Human safety review owns recipe approval;
ML training cannot promote a recipe to `approved`.
