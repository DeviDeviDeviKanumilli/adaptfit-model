# AdaptFit failure and recovery matrix

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** detection, reason codes, user behavior, fallback, telemetry, and test ownership
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `613ff12` (documentation revision base)
> - **Owner:** AdaptFit runtime and product engineering
> - **Supersedes or supports:** supports `system-context-and-dataflow.md`, `on-device-deployment.md`, `product-scope.md`, and `privacy-and-data-lifecycle.md`
> - **Review trigger:** new failure mode, reason code, fallback, runtime, or user-visible behavior

Every failure must have an explicit state. A failed model load or missing
observation is not converted into a confident prediction.

## Implementation boundary

This document is the canonical failure and recovery **contract**, not a claim
that every row is already shipped. As of code baseline `ba8bf1a`,
`training/src/models/streaming.py` only validates feature chunks and timestamps,
maintains causal TCN/GRU state, resets on session or exercise changes, and
returns model tensors. It does not implement a decoder/FSM, `WorkoutEventV1`
emission, camera UI, product fallback, or real-time quality guidance. Rows below
marked `planned` are acceptance behavior for the future decoder/product path;
they must not be described as current runtime behavior until code, fixtures, and
evidence exist.

| Failure | Detection | Reason code | User behavior/fallback | Telemetry | Fixture/owner |
|---|---|---|---|---|---|
| Planned — camera denied | permission result | `camera_denied` | explain setup; no movement feedback | aggregate denial state | mobile fixture / runtime |
| Planned — required joint occluded | observed mask below recipe floor | `joint_occluded` | abstain dependent feedback; request view change | joint mask and duration only | feature/streaming tests / runtime |
| Planned — pose estimator dropout | missing or stale timestamp | `pose_dropout` | pause decoder; resume after stable frames | dropout duration | streaming fixture / runtime |
| Planned — unknown required capability | profile state `unknown` | `capability_unknown` | keep recipe unavailable; ask user | no inferred capability | recipe fixture / product |
| Planned — equipment unavailable | user/session update | `equipment_missing` | pause and offer reviewed substitution | equipment reason | recommendation fixture / product |
| Planned — no eligible recipe | empty deterministic candidate set | `no_eligible_candidate` | show empty state and next action | candidate count and rule reason | recipe fixture / product |
| Planned — draft/retired recipe | approval check | `recipe_not_approved` | omit from candidates | catalog hash/status | catalog fixture / safety |
| Planned — bundle/schema mismatch | load-time compatibility check | `bundle_incompatible` | fail closed; use explicit compatible fallback | versions/hashes, no pose | bundle fixture / release |
| Planned — dropped frames | timestamp gap | `timestamp_gap` | preserve time; pause or abstain per decoder policy | gap duration | runtime fixture / runtime |
| Planned — duplicate overlapping windows | decoder event identity | `duplicate_event` | suppress duplicate count delta | decoder version and event key | decoder fixture / runtime |
| Partial — decoder reset/pause | state transition | `decoder_reset` | close or pause event with explicit state | reset reason | decoder fixture / runtime |
| Planned — corrupt dataset/manifest | checksum/schema/identity check | `data_invalid` | stop preparation/training; no artifact promotion | manifest/checksum | preflight fixture / data |
| Implemented in training — missing label | label mask | `label_unavailable` | zero loss weight; report coverage | task valid count | training tests / evaluation |
| Planned — low tracking confidence | `tracking_confidence` drops below recipe floor (`confidence_floor`) or $p_{track} < 0.60$ for $\ge 30$ frames (1.0s) during active rep | `low_tracking` | planned decoder emits abstention; freezes FSM in `PAUSED`; suppresses rep count; prompts camera repositioning | confidence floor delta and duration only; no raw pose | streaming fixture / runtime |
| Planned — ambiguous repetition boundary | Start/end logits exceed threshold simultaneously or peak separation < minimum window | `ambiguous_boundary` | suppress count increment; maintain active set count until unambiguous boundary is observed | boundary event confidence and peak distance | decoder fixture / runtime |
| Planned — cadence out of bounds | Repetition cycle duration faster than minimum physiological duration or slower than maximum duration in recipe dose constraints | `cadence_out_of_bounds` | disregard spurious cycle or emit cadence pacing prompt; do not increment valid rep count | measured repetition cycle duration and recipe ID | decoder fixture / runtime |
| Unavailable/planned — excessive compensation | Biomechanical compensation metric (e.g., trunk lean $\theta_{trunk} \ge \tau_{trunk}$) exceeds recipe safety threshold | `excessive_compensation` | future quality/product path may provide form guidance; unavailable while labels and runtime are missing | compensation dimension and threshold delta only | quality fixture / product |
| Planned — consent withdrawn | User toggles off telemetry/diagnostics in profile or requests session data deletion | `consent_withdrawn` | immediately purge local session cache, disable diagnostic recording, fail-safe to strictly stateless on-device execution | zero telemetry emitted (all logging disabled) | privacy fixture / privacy review |

## Planned runtime camera occlusion and tracking abstention specification

When a decoder is implemented, it should prevent false-positive repetitions and
corrupted quality feedback during transient camera occlusions or participant
framing loss by applying the following versioned acceptance rules. These rules
are not currently enforced by the Python streaming runtime.

1. **Active Repetition Occlusion Trigger**:
   If the landmark tracking confidence $p_{track} < 0.60$ for $\ge 30$ consecutive frames (1.0 second at 30 FPS) while the decoder is in an active repetition state (`CONCENTRIC_DRIVE`, `APEX_HOLD`, or `ECCENTRIC_RETURN`):
   - The planned decoder suppresses repetition count accumulation.
   - The planned FSM transitions into the `PAUSED` state.
   - The planned runtime emits a valid `WorkoutEventV1` with:
     ```json contract=workout-event-v1
     {
       "schema_version": "workout-event.v1",
       "session_id": "session-demo-001",
       "exercise_id": "seated_one_arm_biceps_curl",
       "variant_id": "left_no_equipment",
       "side": "left",
       "start_timestamp_ms": 30000,
       "end_timestamp_ms": 30000,
       "count_delta": 0,
       "event_confidence": 0.0,
       "abstention": true,
       "reason_code": "low_tracking",
       "tracking_confidence": 0.42,
       "model_version": "movement-tcn-v1",
       "feature_schema_version": "feature.v1",
       "decoder_version": "decoder.v1"
     }
     ```
     The consecutive-frame counter and FSM state are decoder-internal; they are
     not serialized in `WorkoutEventV1`.
   - The user interface displays a guidance prompt: *"Tracking lost: please adjust camera or step back into frame"*.
   - Angular displacement accumulation for range-of-motion (ROM) is frozen to prevent corrupted joint predictions from inflating or failing the repetition quality score.

2. **Recovery & Resumption Protocol**:
   - The planned FSM remains in `PAUSED` until $p_{track} \ge 0.60$ persists for a stabilization warmup period of $\ge 5$ consecutive frames.
   - Upon recovery within 300 frames (10.0 seconds), the planned decoder restores the prior active state and unfreezes the timer.
   - If tracking confidence remains below $0.60$ for $> 300$ consecutive frames (10.0 seconds), the planned decoder marks the current repetition aborted internally with `reason_code = "tracking_timeout"`, flushes the active repetition buffer, and transitions to `IDLE_REST`. `aborted` is not a `WorkoutEventV1` field; adding it requires a schema revision.

## Recovery rules

- Safety and feasibility failures fail closed.
- Observation failures abstain only for dependent outputs and preserve the
  declared capability profile.
- Data and artifact failures stop the run before compute is spent.
- Recovery may resume only after the same versioned contract and state are
  revalidated; do not silently continue across a schema or decoder change.

See [privacy and data lifecycle](privacy-and-data-lifecycle.md) for what may be
logged during recovery and [test matrix](test-and-fixture-matrix.md) for the
required fixture coverage.
