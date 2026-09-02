# Exercise and Capability Schema

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
- `occluded_or_unknown`

The distinction between `absent` and `occluded_or_unknown` prevents the model from confusing a camera failure with a user capability.

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

