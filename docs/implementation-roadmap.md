# Implementation Roadmap

All implementation work should occur under `/Users/devk/AdaptFit/`.

## Phase 0: Freeze scope and schemas

Deliverables:

- Canonical capability profile.
- Canonical exercise recipe.
- Canonical skeleton and feature schema.
- Explicit privacy and safety boundaries.
- Mapping from current Python, mobile, TypeScript, and database catalogs.

Exit condition: the same exercise and capability identifiers can be used by preprocessing, training, mobile runtime, and adaptation rules.

## Phase 1: Build the data pipeline

Deliverables:

- Dataset adapters for selected public sources.
- Joint mapping and torso/pelvis normalization.
- Feature extraction with confidence and missing-limb masks.
- Synthetic augmentation for occlusion, asymmetry, reduced ROM, tempo, and missing limbs.
- Subject-level split generation.
- Data schema documentation and tests.

Exit condition: a reproducible command can turn an approved dataset into training, validation, and test sequences.

## Phase 2: Establish baselines

Deliverables:

- Existing deterministic tracker benchmark.
- Small causal GRU baseline.
- Repetition and phase metrics.
- Latency benchmark on representative mobile hardware.

Exit condition: the learned baseline is measurable against the current deterministic system.

## Phase 3: Train the scalable temporal model

Deliverables:

- Shared causal TCN.
- Multi-task phase, repetition, quality, and observability heads.
- Comparison against the GRU baseline.
- Quantization-aware validation (after the reference float32 artifact).
- Model card documenting training data, limitations, and supported profiles.

Exit condition: the TCN meets the agreed accuracy, confidence, and latency thresholds on the available evaluation data.

## Phase 4: Integrate Android inference

Deliverables:

- Native feature extraction or native temporal inference.
- Versioned on-device model bundle.
- Session reset and timestamp handling.
- One-sided tracking support.
- Confidence-aware UI feedback.

Exit condition: the five initial exercises can be selected, tracked, and evaluated without sending camera data to a server.

## Phase 5: Add adaptation rules

Deliverables:

- Capability-profile filtering.
- Exercise variants and unilateral recipes.
- ROM and tempo calibration.
- Exercise-specific form rules.
- Abstention and manual override behavior.

Exit condition: a user profile never receives an exercise whose required capabilities are unavailable unless a reviewed alternative is selected.

## Phase 6: Evaluate and harden

Deliverables:

- Participant-level evaluation reports.
- Missing-limb and occlusion stress tests.
- Failure-case review.
- Device performance report.
- Privacy and data-retention review.

Exit condition: known limitations are visible in the product and documentation, and low-confidence cases fail safely.

## Phase 7: Acquire target-population data

Future sources may include a rehabilitation center, adaptive sports organization, physical therapist, or licensed dataset partner.

The target-population dataset should be collected with consent, clear labeling, and qualified movement review. It should be used to fine-tune and evaluate the model separately for upper-limb, lower-limb, and wheelchair/seated profiles.
