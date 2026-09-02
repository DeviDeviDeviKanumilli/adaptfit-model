# Decision Log and Open Questions

## Confirmed decisions

| Area | Decision |
|---|---|
| Initial users | People with missing or unavailable arms/legs, one-arm or one-leg users, and wheelchair/seated users |
| Expansion | Other bodily limitations come after the initial profiles |
| Onboarding | Users self-report their limitations and available movements |
| Inference | On-device |
| Initial exercises | Seated one-arm biceps curl, seated one-arm band row, seated single-leg knee extension, seated single-leg march/hip lift, seated forward reach |
| Architecture direction | Shared model across exercises and capability profiles |
| Scalable production encoder | Causal TCN |
| Baseline encoder | Small causal GRU |
| Work location | `/Users/devk/AdaptFit/` |
| Current data access | No direct access to target-group participants |

## Engineering recommendations

- Keep anatomy and safety constraints outside the neural network.
- Use explicit limb-availability and confidence masks.
- Normalize pose features relative to the torso or pelvis.
- Treat exercise definitions as versioned recipes.
- Evaluate by participant, not by random frame.
- Quantize for on-device inference.
- Add target-population data before making strong performance or safety claims.

## Open questions

These do not block the initial architecture and preprocessing work, but they must be resolved before a public release:

1. Which Android devices are the minimum supported hardware?
2. Should the first model runtime be TFLite only, or should cross-platform export be required immediately?
3. Which physical therapist, trainer, or clinical reviewer will approve exercise recipes and quality labels?
4. Which public datasets have terms compatible with the intended project and distribution model?
5. How should the app phrase safety warnings and escalation guidance?
6. Which equipment configurations should be supported for the first release?
7. How will target-population data eventually be recruited, consented, stored, and reviewed?

## Immediate next engineering task

Create the canonical capability and exercise schemas, then build the public-dataset preprocessing pipeline. The model should not be trained until the feature contract can represent optional limbs, seated posture, visibility, and confidence.

