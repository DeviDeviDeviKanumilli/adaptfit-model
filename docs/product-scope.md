# Product Scope and Safety Boundaries

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** approved product scope and safety review
> - **Last verified:** 2026-09-07
> - **Source commit:** `e75ba65`
> - **Owner:** AdaptFit product and safety review
> - **Supersedes or supports:** canonical user journeys, claim boundaries, and fallback behavior
> - **Review trigger:** new user profile, exercise, feedback claim, data behavior, or safety review

## Problem

Standard fitness applications often assume a fixed body configuration and a narrow set of movements. AdaptFit should select and modify exercise routines according to a user’s available movements rather than forcing every user through the same exercise template.

## Initial user profiles

The first release should explicitly support these profiles:

- One available arm or an unavailable/missing arm.
- One available leg or an unavailable/missing leg.
- Wheelchair or seated movement.
- Combinations of the above where the selected exercise remains appropriate.

The terms “unavailable” and “assisted” are intentional: the system should not assume that a missing limb, a prosthesis, an injury, or a temporary limitation produces the same movement pattern.

## Onboarding

The user should provide capability information rather than having the model infer a disability from appearance.

Recommended fields:

- Left arm: available, limited, absent, assisted, or unknown.
- Right arm: available, limited, absent, assisted, or unknown.
- Left leg: available, limited, absent, assisted, or unknown.
- Right leg: available, limited, absent, assisted, or unknown.
- Preferred position: seated, wheelchair, standing, or other supported position.
- Equipment: none, resistance band, light weight, or other approved equipment.
- Optional movement limitations: reduced range, balance limitation, or movement that the user should avoid.

The onboarding language should be capability-focused and non-diagnostic. Users should be able to change their profile later.

## Product behavior

AdaptFit should:

- Filter out exercises that require unavailable capabilities.
- Offer unilateral or seated alternatives when supported.
- Calibrate expected range and tempo to the individual.
- Track only the joints and limbs relevant to the selected exercise.
- Show low-confidence feedback when pose quality is poor.
- Allow the user to stop, skip, or manually mark an exercise.

## Safety boundaries

The system should not:

- Diagnose disability, injury, disease, or rehabilitation status.
- Infer that a person is safe to perform a movement solely from camera data.
- Treat an absent or occluded limb as a badly performed normal limb.
- Claim to measure muscle activation, force, joint loading, or clinical recovery from monocular pose.
- Generate unrestricted exercise prescriptions without curated exercise rules and human review.

The model can detect observable movement properties such as approximate joint angle, tempo, range of motion, smoothness, and tracking confidence. Anatomy metadata and safety rules must constrain what the product recommends.

## User journeys and validation

### Onboarding

1. Explain that the user is describing available movement and equipment, not
   receiving a diagnosis.
2. Collect the `CapabilityProfileV1` fields, including position and consent
   settings. Every answer has an explicit “I do not know” option.
3. Validate conflicts before showing exercises: a seated position cannot be
   paired with a standing-only recipe; an `absent` required limb cannot be
   satisfied by a camera observation; required equipment must be confirmed.
4. Show the user the interpreted profile and let them edit it. Store the
   profile version and local change time.
5. Run a short camera readiness check only for visibility and framing. Never
   overwrite a user-declared capability from that check.

### Exercise selection and workout

1. Filter the reviewed recipe catalog by capability, position, equipment,
   movement-avoidance rules, and recipe approval state.
2. If no approved candidate remains, explain why and offer manual selection,
   profile editing, or a safe stop. Do not substitute a visually similar
   exercise.
3. Before the first set, confirm the side/variant and the relevant camera view.
4. During tracking, show feedback only for dimensions allowed by the recipe and
   supported by labels. Low tracking confidence produces a pause/reposition
   message or abstention, not a strong form correction.
5. Let the user pause, skip, stop, reset, or manually mark a repetition. Manual
   corrections are events with a reason code and are not silently used as model
   labels.
6. At the end, summarize observed counts and confidence. Do not imply that a
   count or quality score certifies safety or clinical progress.

## Capability and equipment conflict behavior

| Situation | Required behavior |
|---|---|
| Required limb is `absent` | Remove the recipe unless an approved unilateral/substitution variant exists. |
| Required limb is `limited` or `assisted` | Keep only recipes whose review explicitly permits that state; apply the recipe’s range/stop rules. |
| Capability is `unknown` | Ask before selecting a recipe that depends on it; do not treat it as available. |
| Equipment unavailable at workout time | Revalidate inventory and return to a reviewed no-equipment or substitution recipe. Never silently remove a band/anchor requirement. |
| Camera cannot see a required joint | Keep the user capability unchanged, abstain from dependent feedback, and offer reposition/manual fallback. |
| Profile changes mid-session | Finish or explicitly cancel the current event, reset decoder state, re-filter candidates, and record the new profile version. |

## Accessibility and fallback requirements

- All critical actions (start, pause, skip, manual count, stop, retry) must be
  usable without precise touch or spoken instructions.
- Feedback must have text and visual alternatives, controllable cadence, and no
  color-only meaning. It must not require the user to look away during a
  movement.
- The product must support seated/wheelchair onboarding and one-sided use of
  the interface.
- Camera denial, low light, occlusion, unsupported device, and model load
  failure must lead to a clear manual workout path or a safe stop.
- Offline operation must not block profile editing or deletion. Raw frames and
  raw pose remain local unless an explicitly versioned consent setting says
  otherwise.

## Exercise review workflow

Recipes move through `draft → safety_review → approved → retired`. A reviewer
must confirm required capabilities, equipment, posture, camera view, allowed
feedback dimensions, stop conditions, and prohibited claims. A catalog entry may
be visible as “planned” while `draft`, but it is not camera-validated support.
Review evidence belongs in the recipe manifest with reviewer identity, date,
version, and unresolved questions.

## Claim boundary

Allowed language describes observable behavior: “the camera could not track the
left elbow,” “one repetition was counted,” or “feedback was withheld because
confidence was low.” Forbidden language says or implies diagnosis, injury
assessment, rehabilitation outcome, safety clearance, muscle activation, force,
joint loading, or clinical validation. Any challenge-submission claim must state
the actual participant population, labels, split, and artifact evidence.
