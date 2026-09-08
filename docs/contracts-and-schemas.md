# AdaptFit contracts and schemas

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** source schemas, model heads, config, and runtime interfaces
> - **Last verified:** 2026-09-08
> - **Source commit:** `f890101`
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** resolves contract detail previously scattered across product, architecture, and data plans
> - **Review trigger:** any field, enum, tensor shape, unit, mask, decoder, or cross-platform interface change

This document defines the versioned interfaces that Python preparation and
training, the streaming runtime, future native clients, and product rules must
share. A field is not part of a contract merely because a proposal mentions it;
it must have a producer, consumer, validation rule, and version.

## Contract rules

- Version every serialized contract independently from the model checkpoint.
- Preserve field order and units for feature tensors.
- Distinguish an unavailable capability from an unobserved camera landmark.
- Use explicit masks for missing labels. Do not encode “unknown” as a negative
  quality target.
- Reject non-finite values and out-of-order timestamps at runtime.
- A schema/model/normalization/decoder mismatch is a load error, not a best-effort
  conversion.
- Python is the reference implementation until native parity fixtures pass.

## `CapabilityProfileV1`

Purpose: describe what the user says they can do. It is user-provided context,
not a disability classifier.

| Field | Type / allowed values | Semantics |
|---|---|---|
| `profile_version` | string, `capability.v1` | Version of this JSON contract. |
| `position` | `seated`, `wheelchair`, `standing`, `supported`, `unknown` | Preferred or permitted starting context. |
| `mobility_context` | controlled string | Product context; never a clinical diagnosis. |
| `left_arm`, `right_arm`, `left_leg`, `right_leg` | `available`, `limited`, `absent`, `assisted`, `unknown` | Declared capability. `absent` is not camera failure. |
| `equipment` | array of controlled IDs | Equipment available at onboarding; recipes revalidate at use time. |
| `range_of_motion` | optional self-reported value/object | Personal constraint; never interpreted as measured clinical ROM. |
| `movements_to_avoid` | array of stable movement IDs or user text | User safety preference/constraint, passed to rules. |
| `consent` | versioned local-data settings | Controls local recording, diagnostics, and deletion behavior. |

`unknown` means the user has not supplied a capability answer. Camera
visibility belongs to pose observations and uses `occluded_or_unknown` (or an
equivalent observation mask), never a capability state. A declared `absent`
limb receives a capability mask of zero while an occluded available limb has a
camera-observation mask of zero; these cases must remain distinguishable.

## `ExerciseRecipeV1`

Every product exercise is a reviewed, stable recipe, not an unconstrained model
output. Required fields:

| Field | Requirement |
|---|---|
| `exercise_id`, `variant_id`, `recipe_version` | Stable IDs; never use display names as joins. |
| `movement_family` | One of the model’s six family IDs or an explicitly unmapped catalog value. |
| `required_position` | Controlled posture/position set. |
| `required_limbs`, `optional_limbs`, `relevant_joints` | Explicit capability and feature dependencies. |
| `required_equipment`, `optional_equipment` | IDs checked against the current equipment inventory. |
| `unilateral_support`, `bilateral_support`, `substitution_group` | Eligibility and alternative behavior. |
| `expected_phase_sequence` | Ordered phase IDs; missing/uncertain phases are allowed only if declared. |
| `dose_constraints` | Reps, time, rest, ROM/tempo ranges, and stop conditions with units. |
| `allowed_feedback_dimensions` | Only dimensions with reviewed labels and product approval. |
| `camera_requirements` | View, minimum visibility, and relevant-joint rules. |
| `confidence_floor` | Versioned threshold plus abstention behavior. |
| `review_status`, `reviewer`, `reviewed_at` | `draft`, `safety_review`, `approved`, `retired`; approval is human-owned. |

Catalog support means a recipe can be displayed after capability filtering.
Camera-validated support additionally requires an evaluated tracking/decoding
path for that variant. Empty candidate sets must produce a clear manual-fallback
state; the product must never silently substitute an unsafe exercise.

## `FeatureSchemaV1`

The current v1 training/reference input is exactly 283 ordered values. The layout
is generated in `training/src/features/anatomy.py` and must be treated as a
versioned ABI:

| Block | Width | Notes |
|---|---:|---|
| normalized joint positions | 66 | 33 joints × `(x, y)` after subtracting the hip/shoulder anchor and dividing by hip/shoulder/torso scale; dimensionless |
| joint velocities | 66 | Same order; normalized-position units per second at the configured FPS |
| joint confidence | 33 | Pose confidence clipped to `[0, 1]` and multiplied by the capability weight |
| joint/segment angles | 14 | Twelve arccos angles divided by π (`[0, 1]`) plus trunk-lean and pelvis-tilt values divided by π (approximately `[-1, 1]`) |
| angular velocities | 14 | Per-second differences of the normalized angle values |
| observed-joint mask | 33 | Binary camera/data observability; `1` means finite and usable for that frame |
| capability mask | 33 | Per-joint profile weights: available `1.0`, limited `0.65`, absent `0.0`, assisted `0.8`, unknown `0.5` |
| profile context | 20 | Four limbs × five capability-state one-hot values |
| posture/position context | 4 | One-hot `seated`, `wheelchair`, `standing`, or `unknown` |
| **Total** | **283** | No optional diagnostics are concatenated into this tensor |

The encoder normalizes translation and scale using the configured torso/pelvis
reference. When hips are unavailable it falls back to shoulders, torso scale,
or a finite-joint anchor according to the source implementation. Invalid values
are zeroed before the masks are applied; short gaps may be interpolated by the
preparation config, while long gaps remain masked. Non-finite features are
rejected before batching. Normalization statistics have their own version and
are stored with each model artifact. Frame rate, window length, stride, causal
receptive field, and coordinate convention are mandatory manifest fields.

The optional observable-diagnostics variant (for example acceleration,
rolling-ROM, smoothness, trunk lean, or tracking summaries) is not a compatible
replacement for the 283-input contract. It requires a new schema version,
checkpoint, normalization file, and parity fixtures.

## `MovementPredictionV1`

Current outputs:

| Output | Shape/role | Label and mask rule |
|---|---|---|
| movement family | six pooled logits | Train only on source family labels. |
| phase | five per-frame logits | Train only where phase labels are source-backed or explicitly approved. |
| repetition boundaries | per-frame start/end logits | One start and one end channel; decoder owns event matching. |
| dimension-specific quality | four pooled binary logits | Current coverage is zero; keep masked/disabled until reviewed targets exist. Candidate datasets with frame-level quality annotations (such as SERE) require an explicit window/sequence temporal aggregation contract (e.g. window-level pooling or thresholded active-frame fraction) before mapping into pooled logits; without an explicit aggregation contract, these heads must remain masked. |
| tracking confidence | per-frame observability output | Self-supervised from pose/capability observations; not clinical uncertainty. |
| expert quality | optional five-class pooled logits | UCO/composite target only; never relabel as ROM, tempo, or smoothness. |

Planned outputs—density, exercise-ID, repetition confidence, calibrated
abstention, and richer quality targets—are future schema additions. They must
not be serialized under `MovementPredictionV1`.

## `WorkoutEventV1`

The deterministic decoder emits one event only after it has reconciled
overlapping windows and runtime state:

```json
{
  "event_version": "workout-event.v1",
  "session_id": "…",
  "exercise_id": "…",
  "variant_id": "…",
  "side": "left|right|bilateral|alternating|unknown",
  "start_timestamp_ms": 0,
  "end_timestamp_ms": 0,
  "count_delta": 1,
  "event_confidence": 0.0,
  "tracking_confidence": 0.0,
  "abstained": false,
  "reason_code": "accepted|low_tracking|ambiguous_boundary|paused|reset|manual",
  "model_version": "…",
  "feature_schema_version": "feature.v1",
  "decoder_version": "decoder.v1"
}
```

The decoder must define, in versioned configuration:

- timestamp monotonicity and maximum tolerated gap;
- reset on session/exercise/variant change;
- pause and rest behavior;
- dropped-frame bridging versus forced abstention;
- partial-repetition and manual-correction semantics;
- one-to-one deduplication of overlapping-window predictions;
- event confidence and tracking-confidence floors.

No event is emitted for an unavailable capability. Camera occlusion should
produce abstention or a reason code rather than an inferred failed repetition.

## Provenance contracts

### `DatasetManifestV1`

Minimum fields: `dataset_id`, source URL or accession, source version,
checksum, license/access state, raw and prepared paths, adapter version,
participant/session IDs, modality and joint map, label roles and masks,
augmentation lineage, privacy/retention rule, split assignment, and known
prohibited interpretations.

### `ExperimentRecordV1`

Minimum fields: experiment ID, Git commit, config hash, dataset-manifest IDs,
split policy, seed, parent checkpoint, trainable layers, optimizer and loss
weights, effective labeled frames/windows, validation cadence, stopping reason,
wall time, peak memory, metrics by source/profile/exercise, and failure notes.

### `ModelArtifactManifestV1`

Minimum fields: model/schema/normalization/decoder versions, checkpoint hash and
parent, architecture and parameter count, input/output shapes, frame rate/window
/stride/receptive field, export/runtime, quantization state, calibration data
provenance, golden-fixture result, artifact paths, creation commit, and known
limitations. A mobile bundle cannot be called deployable until this manifest
and native parity evidence exist.

## Ownership and compatibility

Python schemas and feature builders own reference serialization. The model
runner owns tensor shapes and checkpoint loading. The decoder owns event
semantics. Mobile/native code must consume the manifest and golden fixtures;
it may not duplicate undocumented constants. TypeScript/database representations
must use the same stable IDs and version fields. Any incompatible change needs a
new version and a migration note in the decision ledger.
