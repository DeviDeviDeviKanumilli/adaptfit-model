# On-Device Deployment Plan

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** model/runtime contracts and future export/parity acceptance criteria
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `613ff12` (documentation revision base)
> - **Owner:** AdaptFit runtime engineering
> - **Supersedes or supports:** canonical deployment contract; current-state records that deployment is not yet available
> - **Review trigger:** exporter, native bridge, schema, device target, privacy behavior, or quantization change

The end-to-end boundary is in [system context and data flow](system-context-and-dataflow.md).
Use the [failure and recovery matrix](failure-and-recovery-matrix.md), [privacy
and data lifecycle](privacy-and-data-lifecycle.md), and [artifact registry](artifact-registry.md)
for acceptance evidence that is still unavailable until a native bridge exists.

## Privacy requirement

Camera frames, raw video, audio, and raw pose streams should remain on the device. Training data should be collected only under explicit consent and stored separately from the production runtime.

For training workflows, a raw recording may be processed locally into derived skeleton features and then deleted if the consent and retention policy permits. Derived data can still be sensitive and must be handled accordingly.

## Runtime pipeline

```text
Native camera capture
    ↓
Native pose estimation
    ↓
Native canonical feature extraction
    ↓
Rolling temporal buffer
    ↓
Quantized temporal model
    ↓
Small prediction object to the UI
```

The current mobile bridge returns only a small set of angles and confidence. An anatomy-informed model needs either:

- A richer allowlisted feature vector from native code, without passing raw landmarks to JavaScript; or
- The temporal model to run natively and return only derived predictions.

The second option gives the strongest privacy boundary and reduces JavaScript workload.

The current reference implementation for this contract is
`training/src/models/streaming.py`. It is deliberately kept separate from the
future native bridge so Python can be used as a golden causal-parity harness.
The mobile implementation should match its chunk semantics, state reset events,
timestamp checks, and feature width before model conversion.

Workout recommendation runs before this camera pipeline. The planned
recommendation model consumes the local capability profile, equipment, goals,
recipe catalog, and consented history; it does not require raw frames or raw
pose. Its hard feasibility rules and catalog version must be validated before a
ranker result is shown. A future local ranker would need its own versioned
manifest and fallback behavior; it is not part of the current TCN runtime.

## Model format

Android is the first proposed deployment target because the PeddieHacks
reference app has an Android MediaPipe path. The active AdaptFit model
repository does not yet contain that bridge. Export a float model first, then
an int8 mobile model through a runtime supported by the native module, with
TFLite as an initial candidate rather than an available artifact.

The model interface should remain independent of the runtime so that the same canonical weights can later be converted or adapted for iOS.

The proposed AF-MJEPA is training-only. It is not part of the mobile bundle,
native inference path, latency budget, or privacy release claim. Only a selected
student checkpoint derived from the canonical 283-feature contract may proceed
to export, and distillation does not waive any float/native/quantized parity
gate.

## Streaming behavior

The runtime should:

- Process frames causally.
- Maintain a short rolling feature buffer.
- Reset state when a session or exercise changes.
- Handle dropped frames and timestamps.
- Propagate joint confidence and limb masks.
- Abstain from form feedback when confidence is too low.
- Never convert missing data into a confident “bad form” result.

The reference runtime treats a timestamp gap as a dropped-frame gap that can
continue the state, while a non-increasing timestamp is rejected. A new session
or exercise ID clears the temporal state before the first new prediction. A
native runtime may choose a stricter gap policy, but that policy must be tested
against the same golden sequences.

## Mobile acceptance criteria

- Runs fully on-device.
- Maintains real-time responsiveness on the target Android device.
- Does not transmit raw frames or raw landmarks.
- Has deterministic behavior for session reset and exercise changes.
- Produces a confidence value with every movement-quality output.
- Supports one-sided inputs without requiring a bilateral completion.
- Has a versioned model and feature schema.

## Model bundle contract

No bundle currently passes this contract. A future release artifact should be
an immutable directory or archive with:

```text
adaptfit-bundle/
  manifest.json                 # ModelArtifactManifestV1
  model.float32.<runtime>       # reference/exported float model
  model.int8.<runtime>           # optional post-training quantized model
  normalization.npz             # training-only fitted statistics
  feature_schema.json            # FeatureSchemaV1 and ordering
  decoder.json                   # thresholds, tolerance, reset/gap policy
  recipe_compatibility.json     # supported exercise/variant IDs and versions
  golden-fixtures/              # input/output/event traces and hashes
  model-card.md                 # limits, licenses, and claim boundary
```

The manifest must include model, feature, normalization, decoder, recipe, and
bundle versions; input/output shapes; frame rate/window/stride/receptive field;
parameter count; source commit/config hash; checkpoint parent; export runtime;
quantization/calibration provenance; artifact checksums; and known limitations.
The loader rejects an incompatible schema, normalization version, decoder, or
recipe version. It must not silently pad, reorder, or reinterpret 283 inputs.

## Native boundary and golden parity

The native pose estimator may produce a full 33-joint canonical pose internally.
Only the allowlisted, versioned 283-feature vector crosses into the temporal
model, or the complete temporal model runs inside the native process. Raw frames,
raw pose, and unreviewed debug traces do not cross into JavaScript or a network
service.

Golden fixtures must cover bilateral, unilateral, declared-absent capability,
occluded joints, dropped frames, slow/partial reps, non-monotonic timestamps,
session reset, exercise change, pause, and duplicate overlapping windows. For
each fixture compare feature values, every prediction head, confidence,
abstention, reason code, and emitted `WorkoutEventV1` sequence against Python
within a versioned numeric tolerance. Store fixture hashes in the bundle
manifest.

## Export and quantization sequence

1. Validate the float model in Python and test operator/export coverage with a
   small existing checkpoint.
2. Export the selected float checkpoint and compare outputs on validation
   fixtures before running a device session.
3. Apply post-training quantization using a representative validation/calibration
   subset, never the locked test set. Re-run head, event, and abstention parity.
4. Use quantization-aware fine-tuning only if post-training quantization fails a
   declared gate and the extra compute is justified.
5. Select float or quantized deployment before final test reporting. Record
   calibration data, runtime version, tolerances, and rollback artifact.

## Runtime failure and fallback behavior

- **Camera denied/unavailable:** show a manual workout or safe-stop path; never
  claim that no movement occurred.
- **Pose estimator unavailable:** return `abstention=true` with a reason code and
  keep the profile unchanged.
- **Dropped frames:** follow the manifest gap policy; bridge only permitted
  gaps, otherwise abstain/reset. A timestamp gap must not create extra reps.
- **Out-of-order timestamp:** reject the frame and log a local diagnostic; do
  not rewind state.
- **Session/exercise/variant change:** flush the temporal buffer and decoder
  before accepting new predictions.
- **Model load/schema mismatch:** refuse to run the model and use a versioned
  fallback/manual path. Never load a “close enough” bundle.
- **Thermal/memory pressure:** use an explicitly tested smaller fallback model or
  stop feedback; disclose that fallback in the event and diagnostics.

## Device validation and platform parity

Android is first for end-to-end validation. Measure cold/warm startup, per-frame
and p95 latency, memory, dropped-frame rate, thermal throttling, battery impact,
and complete-session event traces on each minimum device. Repeat the same golden
fixtures and acceptance thresholds on iOS before claiming iOS support; shared
weights do not prove shared preprocessing or runtime behavior.

## Privacy acceptance

The release checklist must verify permission denial, local-only processing,
temporary-buffer lifetime, deletion, crash/log redaction, offline behavior, and
that model bundles contain no raw frames, raw pose, or participant identifiers.
Derived features and event histories remain sensitive and follow the consent
and retention settings in `CapabilityProfileV1`. A design intention is not a
passed privacy gate until a device test and review record exist.
