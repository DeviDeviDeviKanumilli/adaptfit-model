# Scalable, Anatomy-Informed ML Architecture

## Model choice

For a scalable on-device system, use a **shared causal dilated TCN** as the production temporal encoder and retain a small causal GRU as a baseline.

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
- Explicit limb state: available, limited, absent, assisted, or occluded.

The current v1 model keeps the deployed feature width stable and exposes
additional acceleration, rolling-ROM, smoothness, trunk-lean, and tracking
confidence summaries through the observable-diagnostics API. Those summaries
are not converted into quality labels when a source does not provide them.
They are the bridge to a richer feature schema or self-supervised auxiliary
task once sufficient reviewed data exists.

The v1 capability channel uses explicit per-joint weights for available,
limited, assisted, unknown, and absent states. This keeps a limited or
assisted limb from being treated exactly like an unconstrained limb while
preserving the profile one-hot context.

The availability mask is essential. A missing or untracked limb must not be represented as a normal limb with a low-quality pose score.

Anatomy is encoded through the body-segment feature layout, exercise-specific joint requirements, and curated body-demand metadata. A graph neural network can be considered later, after enough data exists to justify a learned spatial layer.

## Shared model design

There should be one shared encoder across exercises and user profiles. The profile and exercise recipe condition the outputs rather than creating a separate network for every disability.

Recommended initial configuration:

- A versioned anatomy feature contract (283 inputs in v1; 378 inputs in the
  optional diagnostics variant).
- A 48–90 frame rolling window at approximately 30 FPS.
- Four to six causal dilated residual TCN blocks.
- 64–128 channels.
- Receptive field of roughly 2–4 seconds.
- Int8 quantization for the mobile build.

Expected model-weight size is approximately 0.2–1 MB depending on width and head count. Runtime memory and pose-estimator size are separate from the temporal model’s weight size.

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
