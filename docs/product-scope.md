# Product Scope and Safety Boundaries

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

