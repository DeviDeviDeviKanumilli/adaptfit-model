# AdaptFit privacy and data lifecycle

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** local-first data handling, consent, retention, deletion, and logging boundaries
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `613ff12` (documentation revision base)
> - **Owner:** AdaptFit product and privacy review
> - **Supersedes or supports:** supports `product-scope.md`, `recommendation-model-plan.md`, `on-device-deployment.md`, and `contracts-and-schemas.md`
> - **Review trigger:** new storage, sync, telemetry, consent field, model input, or data-retention policy

AdaptFit's v1 privacy boundary is local-first. The repository does not yet
contain a production mobile implementation, so these are required runtime
contracts and acceptance criteria, not a claim that a shipped app has passed
the audit.

## Data lifecycle

```text
camera frames (ephemeral)
  → pose landmarks (ephemeral/local)
  → normalized features and masks (local session)
  → predictions and WorkoutEventV1 (local)
  → consented feedback/history (local)
  → recommendation candidate/exposure records (local, redacted)
  → deletion/reset or explicit user export
```

Raw frames and raw pose are not recommendation inputs or recommendation logs.
The ranker uses declared capability, equipment, goals, recipe metadata, and
consented history summaries. Tracking confidence may be retained with a workout
event; it is not a diagnosis or a quality claim.

## Consent and retention classes

| Data | Default | Retention | Deletion/reset |
|---|---|---|---|
| Raw camera frames | never persisted | ephemeral processing buffer | discarded after inference or error |
| Raw pose landmarks | local only | session lifetime unless explicitly needed for debugging consent | removed with session reset |
| Features/predictions | local session | until session summary is complete | removed with session reset |
| Workout events | local history | user-controlled history setting | delete individual session or all history |
| Recommendation feedback/exposure | local, redacted | user-controlled history setting | delete with recommendation history reset |
| Training research data | consented, access-controlled | dataset/DUA policy | source-specific deletion and manifest update |

No sync or cloud retention is part of v1. Any future opt-in sync must add a new
consent state, transport/storage owner, encryption/retention policy, export and
deletion contract, and a new review decision before implementation.

## Reset, denial, and redaction behavior

- Camera denial results in a clear setup state; the app must not fall back to
  unobserved movement claims.
- A user can reset workout and recommendation history independently from their
  capability profile. A profile reset invalidates candidate caches.
- Logs contain event IDs, schema/model versions, reason codes, coarse timing,
  and aggregate diagnostics only. They do not contain frames, raw pose, or
  unnecessary free text.
- Consent withdrawal stops collection and marks affected history unavailable;
  it does not silently retain a training copy.

The mobile implementation must prove these behaviors with deletion and redaction
fixtures before claiming “on-device privacy.”
