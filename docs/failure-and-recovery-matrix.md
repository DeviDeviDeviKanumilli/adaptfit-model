# AdaptFit failure and recovery matrix

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** detection, reason codes, user behavior, fallback, telemetry, and test ownership
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `1a46f38` (documentation revision base)
> - **Owner:** AdaptFit runtime and product engineering
> - **Supersedes or supports:** supports `system-context-and-dataflow.md`, `on-device-deployment.md`, `product-scope.md`, and `privacy-and-data-lifecycle.md`
> - **Review trigger:** new failure mode, reason code, fallback, runtime, or user-visible behavior

Every failure must have an explicit state. A failed model load or missing
observation is not converted into a confident prediction.

| Failure | Detection | Reason code | User behavior/fallback | Telemetry | Fixture/owner |
|---|---|---|---|---|---|
| Camera denied | permission result | `camera_denied` | explain setup; no movement feedback | aggregate denial state | mobile fixture / runtime |
| Required joint occluded | observed mask below recipe floor | `joint_occluded` | abstain dependent feedback; request view change | joint mask and duration only | feature/streaming tests / runtime |
| Pose estimator dropout | missing or stale timestamp | `pose_dropout` | pause decoder; resume after stable frames | dropout duration | streaming fixture / runtime |
| Unknown required capability | profile state `unknown` | `capability_unknown` | keep recipe unavailable; ask user | no inferred capability | recipe fixture / product |
| Equipment unavailable | user/session update | `equipment_missing` | pause and offer reviewed substitution | equipment reason | recommendation fixture / product |
| No eligible recipe | empty deterministic candidate set | `no_eligible_candidate` | show empty state and next action | candidate count and rule reason | recipe fixture / product |
| Draft/retired recipe | approval check | `recipe_not_approved` | omit from candidates | catalog hash/status | catalog fixture / safety |
| Bundle/schema mismatch | load-time compatibility check | `bundle_incompatible` | fail closed; use explicit compatible fallback | versions/hashes, no pose | bundle fixture / release |
| Dropped frames | timestamp gap | `timestamp_gap` | preserve time; pause or abstain per decoder policy | gap duration | runtime fixture / runtime |
| Duplicate overlapping windows | decoder event identity | `duplicate_event` | suppress duplicate count delta | decoder version and event key | decoder fixture / runtime |
| Decoder reset/pause | state transition | `decoder_reset` | close or pause event with explicit state | reset reason | decoder fixture / runtime |
| Corrupt dataset/manifest | checksum/schema/identity check | `data_invalid` | stop preparation/training; no artifact promotion | manifest/checksum | preflight fixture / data |
| Missing label | label mask | `label_unavailable` | zero loss weight; report coverage | task valid count | training tests / evaluation |

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
