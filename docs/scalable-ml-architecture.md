# Scalable, Anatomy-Informed ML Architecture

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** implemented model code plus explicitly labeled future architecture
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `d5362ee` (documentation revision)
> - **Owner:** AdaptFit ML engineering
> - **Supersedes or supports:** canonical model/data-flow explanation; contracts-and-schemas owns exact serialized interfaces
> - **Review trigger:** architecture, feature schema, head, runtime, or deployment change

## Model choice

For a scalable on-device system, the planned production path is a **shared
causal dilated TCN** with a small causal GRU retained as a baseline.

A GRU is still useful for comparison and may win on a very small dataset. The TCN is the better long-term default because it trains in parallel, has a predictable receptive field, is straightforward to quantize, and can process a rolling buffer without relying on a long-lived hidden state.

## High-level pipeline

```text
Camera frames
    ↓
On-device pose estimator
    ↓
Canonical anatomy feature encoder
    ↓
Joint confidence + limb-availability masks
    ↓
Shared causal temporal encoder
    ↓
Phase | repetitions | quality | uncertainty
    ↓
Capability-aware exercise rules and adaptation
```

## Anatomy-informed feature representation

The temporal model should not consume raw pixel coordinates as its primary input. The feature encoder should derive:

- Relative segment vectors normalized to torso or pelvis coordinates.
- Joint angles for the joints relevant to the exercise.
- Angular velocity and acceleration.
- Range-of-motion and tempo features over a rolling window.
- Smoothness and stability indicators.
- Left/right asymmetry where both sides exist.
- Trunk compensation and posture indicators.
- Per-joint visibility and confidence.
- Separate capability state (`available`, `limited`, `absent`, `assisted`, or
  `unknown`) from camera observability (`observed`, `occluded_or_unknown`).

The current v1 model keeps the training/reference feature width stable. A
separate diagnostics configuration can append acceleration, rolling-ROM,
smoothness, and related summaries for experiments; that variant is not the
v1/mobile contract. Those summaries are not converted into quality labels when
a source does not provide them. They are a possible bridge to a richer schema
or self-supervised auxiliary task once sufficient reviewed data exists.

The v1 capability channel uses explicit per-joint weights for available,
limited, assisted, unknown, and absent states. This keeps a limited or
assisted limb from being treated exactly like an unconstrained limb while
preserving the profile one-hot context.

The availability mask is essential. A missing or untracked limb must not be represented as a normal limb with a low-quality pose score.

Anatomy is encoded through the body-segment feature layout, exercise-specific joint requirements, and curated body-demand metadata. A graph neural network can be considered later, after enough data exists to justify a learned spatial layer.

## Shared model design

There should be one shared encoder across exercises and user profiles. The
profile and exercise recipe condition the outputs rather than creating a
separate network for every disability.

### Implemented now: v1 contract

- Input: exactly 283 ordered features from `FeatureSchemaV1`.
- Window: 128 frames at approximately 30 FPS; current training stride is 8.
- Temporal encoder: causal residual TCN, 96 channels, five blocks, dilations
  1/2/4/8/16, 125-frame causal receptive field.
- Baseline: small causal GRU retained for comparison.
- Heads: six family logits, five phase logits, per-frame start/end logits, four
  pooled binary quality logits, and a per-frame tracking-confidence output.
- Runtime state: the streaming runtime retains the previous 124 feature frames,
  rejects non-finite/out-of-order timestamps, and resets on session or exercise
  changes.
- Weight count evidence: corrected-v1 TCN 307,410 parameters; GRU 68,178.

The 283-input contract and tensor semantics are authoritative in
[contracts-and-schemas.md](contracts-and-schemas.md). The implementation values
above must be changed in code/config and this document together; generic ranges
are not valid substitutes for the current contract.

### Planned next

- Add reviewed repetition-density supervision only if boundary/decoder analysis
  shows a measured counting gap.
- Add controlled warm-start and staged fine-tuning, then optionally a single
  teacher-distillation experiment.
- Add reviewed quality targets and calibration before enabling quality feedback.
- Define export, quantization, native parity, and recipe integration.

### Deferred research

Graph spatial layers, a larger exercise-ID catalog head, learned personalized
adapters, rich uncertainty heads, and broad teacher ensembles are deferred until
the corresponding data, labels, or deployment evidence exists.

### Future design envelope (not a current contract)

Earlier planning used a 48–90 frame window, 64–128 channels, a two-to-four
second receptive field, and approximately 0.2–1 MB of temporal weights. These
are experiment ranges only. They must not be used to configure v1 or cited as
implemented behavior. Int8 quantization is also a planned deployment step, not
an available artifact.

The checked-in v1 TCN has a 125-frame causal receptive field with the default
kernel and dilations. `CausalStreamingRuntime` keeps only the preceding 124
feature frames, so live inference does not grow memory with session length.
The GRU baseline keeps its hidden state instead. Both runtimes accept chunks,
reject non-finite or out-of-order timestamps, and reset state when the session
or exercise identifier changes.

## Prediction heads

The shared encoder should support multiple small heads:

- Exercise family or movement primitive.
- Movement phase.
- Repetition start/end or repetition confidence.
- Range-of-motion quality.
- Tempo control.
- Smoothness or compensation indicators.
- Overall tracking and prediction confidence.

The quality heads should be treated as movement observations, not medical judgments.
For v1, family and quality heads use a masked mean over the observed temporal
context rather than a single final frame. The evaluator also reports calibrated
classification confidence and abstention rates; a learned uncertainty head is
deferred until uncertainty labels are available.

The current implementation adds a small per-frame tracking-confidence head. Its
target is generated from observed landmarks, pose confidence, and the explicit
capability profile, so a declared absent limb is not treated as a camera failure.
This is a self-supervised observability signal for deciding when to abstain from
feedback. It must not be described as clinical uncertainty or movement quality.

## Data-flow and ownership boundary

```text
camera
  → native pose estimator
  → canonical 33-joint pose
  → FeatureSchemaV1 (283 features, masks, profile context)
  → causal TCN/GRU
  → prediction heads
  → calibrated confidence and abstention
  → deterministic recipe, capability, and safety rules
  → WorkoutEventV1
```

The neural network describes observable motion from its feature window. The
recipe/rule layer owns eligibility, capability conflicts, equipment checks,
pause/reset behavior, feedback allowlists, and manual fallback. Keeping safety
rules outside the network makes them inspectable and lets a reviewed recipe
change without silently changing the learned representation.

Training validation merges overlapping windows back to sequence coordinates
before selecting the best checkpoint. Window metrics are retained for
diagnostics, but sequence-level metrics are the primary validation score because
the default stride creates correlated windows. If an older NPZ-only prepared
split has no JSONL offsets, training explicitly falls back to window scoring and
records that limitation.

Unsegmented source clips and procedural quality templates are masked by default
in `training/configs/v1.yaml`. They remain available as opt-in ablations, but
must not be presented as validated repetition or movement-quality labels.

## Where adaptation belongs

The neural network should describe what it sees. A curated rule layer should decide what is eligible and how it can be modified.

Examples of deterministic constraints:

- A movement requiring bilateral grip is incompatible with a one-arm profile unless a supported unilateral version exists.
- A standing exercise is incompatible with a seated-only profile unless a seated recipe exists.
- A low-confidence or occluded joint should suppress form feedback rather than produce a strong correction.
- A user-specific range calibration should adjust the expected target without changing the underlying anatomy metadata.

This separation allows new exercises and adaptations to be added through recipes and rules without retraining the entire model.

## Scaling dimensions

### More exercises

Use movement families, exercise recipes, and body-demand metadata rather than a large monolithic exercise classifier. Adding an exercise should primarily involve a recipe, relevant joints, labels, and a small head or conditioning vector.

### More user profiles

Use the same encoder with explicit masks and capability embeddings. Do not create a separate model for each disability category.

### More data

Pretrain the motion encoder on unlabeled pose sequences, then fine-tune with labels for phase, repetitions, and quality. This reduces dependence on manually labeled clips.

### More devices

Keep a stable feature contract and export an int8 model. Android can use an on-device mobile runtime first; iOS can share the same weights later through a compatible mobile runtime or platform-specific conversion.

### Personalization

Personalization should begin with local calibration values such as comfortable ROM, tempo range, and confidence thresholds. Later, a small per-user adapter can be updated on-device without retraining the shared backbone.

## Teacher-student option

As the dataset grows, a larger offline teacher model can learn richer temporal representations. Its predictions can be distilled into a compact causal TCN student for on-device use. This provides a path to improve accuracy without shipping a large model to the phone.

## Model limitations

Monocular pose cannot directly measure muscle activation, force, joint loading, or clinical recovery. The system should describe observable movement and uncertainty, then use anatomy-informed metadata and expert-reviewed rules for exercise selection.
