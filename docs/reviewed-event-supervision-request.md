# AdaptFit reviewed event-supervision intake

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** external label request and acceptance contract for the blocked R1 decoder gate
> - **Last verified:** 2026-09-11
> - **Source commit:** `b3926d8` plus the current follow-up working tree
> - **Owner:** AdaptFit data and evaluation engineering
> - **Supersedes or supports:** supports `HANDOFF.md`, `pretraining-readiness.md`, `current-state.md`, and `training-execution-log.md`
> - **Review trigger:** receipt of new labels, source annotations, adjudication results, or a split-policy change

This document defines the smallest external input that can unblock the AdaptFit
movement-model completion loop. It is intentionally narrower than a new data
collection project: the immediate need is reviewed temporal supervision for
repetition start, hold/apex, and end behavior in representative seated,
unilateral-friendly recordings.

## Why this is required

The current local evidence has passed the unavailable-boundary mask correction
the partial-pose tracking-target correction, and the local UL-RED adapter audit
now recovers explicit markerless 3Rep spans for `219` R3 recordings, but the
validation decoder gate still fails. The UCO exact-boundary follow-up contains
`507` strong target pairs but no per-frame phase/hold labels; its accepted
checkpoint has `0/507` nearby starts, `1/507` nearby ends, and `0/507`
apex-qualified intervals. The UL-RED spans are boundary-only and provide no
reviewed hold/apex labels. A sparse boundary-positive-weight cap of `64.0` was
tested in a new artifact root and rejected because it lowered validation
sequence score and increased false events. The locked test split remains
unused.

The missing evidence is therefore not a request to infer labels from pose,
lower the tracking floor, remove apex confirmation, or tune thresholds. It is a
request for reviewed temporal annotations with source identity and adjudication
provenance.

## Requested scope

The preferred sources are:

1. UCOPhyRehab++ recordings already staged under `data/raw/ucophyrehabpp/`; and
2. REHAB24-6 recordings already staged under `data/raw/rehab24_6/`.

Additional recordings are acceptable when they are seated and unilateral-
friendly, carry participant/session identity, and can be split without
participant leakage. The first delivery should cover the launch exercise
families rather than broad unrelated movements:

- seated one-arm biceps curl;
- seated one-arm band row;
- seated single-leg knee extension;
- seated single-leg march or hip lift; and
- seated forward reach.

At minimum, each reviewed repetition must have a start event, an end event, and
either a reviewed apex/hold interval or an explicit reviewed statement that the
apex/hold is not observable. To pass the current decoder gate, the training
cohort must include enough observable apex/hold examples for the decoder to
learn the required two-frame apex confirmation; unobservable examples remain
masked and do not count as negative apex labels.

## Accepted row contract

One JSON object per reviewed repetition is preferred. All frame indices are
inclusive source-frame IDs; timestamps may be supplied as an additional check.

```json
{
  "source_dataset": "ucophyrehabpp",
  "participant_id": "subject1",
  "session_id": "exercise_01",
  "exercise_id": "01",
  "side": "left",
  "rep_index": 0,
  "start_frame": 35,
  "apex_start_frame": 120,
  "apex_end_frame": 124,
  "end_frame": 282,
  "phase_labels_available": true,
  "annotator_id": "reviewer_a",
  "review_status": "adjudicated",
  "reviewed_at": "2026-09-11",
  "source_license_or_access_note": "..."
}
```

Required fields:

- `source_dataset`, `participant_id`, `session_id`, `exercise_id`, and `rep_index`;
- `start_frame` and `end_frame`, or an explicit reviewed unavailable value;
- `apex_start_frame` and `apex_end_frame` when a hold/apex is observable;
- `phase_labels_available`, distinguishing unavailable labels from reviewed
  negative evidence;
- annotator identity, review status, review date, and source/license access
  note.

Optional but strongly preferred fields:

- frame-level phase spans (`rest`, `concentric`, `hold`, `eccentric`);
- timestamp and source frame-rate fields;
- annotation confidence and disagreement notes;
- original annotation IDs and links to the source recording;
- a second independent annotation or an adjudication record.

Do not submit labels generated solely from velocity extrema, angle thresholds,
the existing weak displacement heuristic, or a model prediction. Such values
may be retained as diagnostic proposals, but they must remain masked and cannot
be used as reviewed supervision.

## Semantics and conventions

The accepted semantics follow [the annotation handbook](annotation-handbook.md):

- `rep_start` and `rep_end` are reviewed turnaround boundaries in sequence time;
- the apex/hold is the reviewed inflection or stable interval between
  concentric and eccentric motion;
- a partial repetition, interrupted set, or uncertain boundary is explicitly
  marked partial/uncertain and excluded from count-loss supervision;
- unavailable phase or apex labels are masked, not converted to class zero or a
  negative apex target;
- source frame IDs must be preserved before any resampling to the 30 FPS model
  contract;
- if two reviewers disagree, retain both annotations and provide the declared
  adjudication or consensus rule.

## Acceptance checklist before training

When the files arrive, the next run must not start until all of these checks
pass:

1. source and access/license status are recorded;
2. every row resolves to exactly one staged source sequence;
3. frame IDs are monotonic and within the source recording bounds;
4. starts precede apex/hold, which precedes ends, with no duplicate repetition
   index inside a session;
5. participant/session identities are disjoint across train, validation, and
   locked test splits;
6. reviewed unavailable fields remain masked;
7. reviewed phase labels map only to the canonical five-class vocabulary;
8. the adapter and one positive/one masked regression fixture pass;
9. the prepared root has finite features, correct `[128, 283]` windows, zero
   unknown-boundary endpoint leakage, and an auditable label-coverage report;
10. the test split is not read during preparation, training, calibration, or
    failure analysis.

## Resume sequence after acceptance

Use a new prepared root and a new artifact root. Preserve the corrected-v1
baseline, `data/processed-r1-uco-tracking-target-correction/`, and
`data/processed-r1-ulred-boundary-support/` unchanged.

```text
receive reviewed rows
  → verify provenance and source-frame alignment
  → add adapter/fixture coverage
  → run docs validator and full tests
  → prepare a new isolated root
  → run preflight and runtime checks
  → train one bounded validation-only candidate
  → calibrate decoder.v1 on validation only
  → run failure analysis and update the execution log
  → if and only if the gate passes, train seed 43
  → if and only if both validation runs pass, evaluate the locked test once
```

The validation gate remains:

- count MAE `<= 0.40`;
- repetition-end F1 `>= 0.60`;
- zero false events on empty-target sequences.

No test evaluation, quality-head activation, clinical claim, target-population
claim, native export, or mobile release decision is authorized before the
validation and provenance gates pass.
