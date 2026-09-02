# On-Device Deployment Plan

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

## Model format

Android should be the first deployment target because the repository already contains an Android MediaPipe path. Export an int8 mobile model through a runtime supported by the native module, with TFLite as the initial practical option.

The model interface should remain independent of the runtime so that the same canonical weights can later be converted or adapted for iOS.

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
