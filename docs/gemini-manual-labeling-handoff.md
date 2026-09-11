# Gemini handoff: reviewed repetition-event labeling

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** instructions for Gemini-assisted temporal annotation before AdaptFit training
> - **Last verified:** 2026-09-11
> - **Source commit:** `b3926d8` plus the current follow-up working tree
> - **Owner:** AdaptFit data and evaluation engineering
> - **Supersedes or supports:** supports [`reviewed-event-supervision-request.md`](reviewed-event-supervision-request.md), [`annotation-handbook.md`](annotation-handbook.md), and [`production-completion-plan.md`](production-completion-plan.md)
> - **Review trigger:** annotation rubric, source-frame convention, label schema, or training acceptance-policy change

## Copy this instruction into Gemini

You are an annotation assistant for the AdaptFit movement model. Your task is
to propose frame-accurate repetition-event labels from the supplied recording
or frame-indexed pose visualization. You are not the final annotator. Every
proposal must be reviewed and approved by a human before it can be used for
training.

The target launch exercises are:

1. seated one-arm biceps curl;
2. seated one-arm band row;
3. seated single-leg knee extension;
4. seated single-leg march or hip lift; and
5. seated forward reach.

Label logical repetitions, not generated 128-frame training windows. Preserve
the source recording’s native frame IDs, frame rate, and index base. Do not
resample frame numbers to 30 FPS. If the frame index base is unknown, stop and
report it instead of guessing.

Do not infer labels from a filename, an existing model prediction, a velocity
extremum, an angle threshold, `Segmentation.csv`, or a UL-RED `3Rep` row alone.
Those may be diagnostic context, but they are not proof of a reviewed apex or
event boundary.

## Annotation definitions

For each complete repetition:

- `start_frame`: the first source frame showing deliberate movement out of
  stable rest toward the exercise. Ignore pose jitter, camera motion, and
  isolated one-frame spikes.
- `apex_start_frame` / `apex_end_frame`: the observable maximum-contraction or
  turnaround interval between the concentric and eccentric portions. If the
  apex is a single clear inflection frame, set both fields to that frame. Do
  not confuse the apex with the repetition end.
- `end_frame`: the source frame at the reviewed return/turnaround boundary
  after the eccentric portion, using the same convention for every clip.
- `phase_labels_available`: `true` only when the apex/phase evidence is
  visually defensible. If the apex is not observable because of occlusion,
  poor pose, a cut, or ambiguity, set it to `false` and leave both apex fields
  null. Never convert an unobservable apex into a negative apex label.

Use inclusive source-frame IDs. The expected temporal order is:

```text
start_frame <= apex_start_frame <= apex_end_frame <= end_frame
```

If a repetition is partial, interrupted, duplicated, or ambiguous, preserve it
as a proposal only when useful for review and set `partial` or `uncertain` to
`true`. Such a repetition must not be treated as an exact count target.

## What to inspect

For every supplied clip or sequence:

1. Identify the source dataset, participant, session, exercise, side, native
   frame rate, and frame-index base from the supplied metadata.
2. Watch the whole clip once to understand rest, repetitions, pauses, and any
   camera or pose failures.
3. Replay at reduced speed and inspect each candidate boundary with surrounding
   frames, not just one isolated image.
4. Mark starts, apex/hold intervals, and ends for complete repetitions.
5. Mark pauses, partial repetitions, dropped frames, occlusions, and unclear
   events in the notes.
6. Report clips that cannot be identified or whose source frame IDs cannot be
   established. Do not silently discard them or invent metadata.

Give special attention to the cases AdaptFit must support: seated posture,
unilateral movement, limited range of motion, assistive support, slow motion,
pauses, partial repetitions, and temporary occlusion. These are reasons to
record uncertainty, not reasons to force a label.

## Gemini proposal output

Return one JSON object per proposed repetition in a UTF-8 JSONL file named
`gemini_event_proposals.jsonl`. Use this shape:

```json
{
  "source_dataset": "ucophyrehabpp",
  "source_sequence_id": "subject1/exercise_01",
  "participant_id": "subject1",
  "session_id": "exercise_01",
  "exercise_id": "01",
  "side": "left",
  "rep_index": 0,
  "frame_index_base": 0,
  "source_fps": 30.0,
  "start_frame": 35,
  "apex_start_frame": 120,
  "apex_end_frame": 124,
  "end_frame": 282,
  "phase_labels_available": true,
  "partial": false,
  "uncertain": false,
  "gemini_confidence": 0.86,
  "uncertainty_notes": "Clear hold interval; end follows return to rest.",
  "annotator_id": "gemini-proposal",
  "review_status": "ai_proposed",
  "reviewed_at": null,
  "source_license_or_access_note": "provided recording; access provenance supplied by owner"
}
```

For an unobservable apex, use:

```json
{
  "start_frame": 35,
  "apex_start_frame": null,
  "apex_end_frame": null,
  "end_frame": 282,
  "phase_labels_available": false,
  "uncertain": true,
  "uncertainty_notes": "Apex is occluded between frames 118 and 130."
}
```

`gemini_confidence` is a diagnostic confidence value, not a reviewed label.
Gemini must use `review_status: "ai_proposed"`; it must never claim that its
own output is `reviewed` or `adjudicated`.

Also return a Markdown review log named `gemini_event_review_log.md` containing:

- one section per source sequence;
- the source metadata and frame-index convention;
- the number of proposed complete, partial, and uncertain repetitions;
- clips with missing or ambiguous metadata;
- clips with no observable apex;
- any disagreement between an existing source boundary and visual evidence;
- a short explanation for every omitted or rejected candidate.

## Human approval procedure

The project owner must review every proposed JSONL row before training:

1. Open the source clip or frame-indexed visualization.
2. Check at least 10 frames before and after each start, apex, and end.
3. Correct frame IDs, side, exercise, and uncertainty fields as needed.
4. Confirm that partial or uncertain repetitions are not exact count targets.
5. Keep unavailable apex fields null and `phase_labels_available=false`.
6. Change `annotator_id` to the human reviewer and change
   `review_status` to `reviewed` after individual approval.
7. Use `review_status: "adjudicated"` only after a second reviewer or named
   adjudication rule resolves a disagreement.
8. Preserve the original `gemini_event_proposals.jsonl` unchanged alongside
   the human-approved file.

The final approved file should be named
`reviewed_event_supervision.jsonl`. The approved rows must retain Gemini’s
proposal provenance and add human review notes where corrections were made.
Do not edit prepared `.npy` arrays by hand and do not copy labels into every
overlapping training window.

## Recommended first batch

Begin with approximately 20 complete repetitions per launch exercise—about
100 repetitions total—spread across different participants, sessions, sides,
speeds, and camera conditions when available. This is a first coverage batch,
not a guaranteed pass threshold. The training team will run validation and
request another targeted batch if the decoder still lacks event evidence.

If fewer examples are available for an exercise, label all eligible examples
and report the shortfall. Do not replace the missing exercise or population
with fabricated or procedurally inferred labels.

## Approval checklist before handoff to training

The human-approved file is ready for the AdaptFit team only when:

- every row identifies exactly one source sequence;
- native frame IDs and frame-index base are recorded;
- start and end frames are present for complete repetitions;
- apex fields are both present or both null;
- temporal order is valid whenever fields are available;
- no repetition identity is duplicated;
- partial and uncertain rows are explicitly marked;
- unavailable apexes remain masked;
- source, participant, session, exercise, and side metadata are preserved;
- reviewer identity, review status, review date, and access/license note exist;
- the original Gemini proposal file is preserved; and
- no row is marked `reviewed` without human inspection.

Send the approved JSONL and review log to the AdaptFit project owner. The next
engineering sequence is:

```text
approved JSONL
  -> provenance and frame-bound audit
  -> adapter and positive/masked regression fixtures
  -> new isolated prepared-data root
  -> preflight and runtime checks
  -> bounded seed-42 training with test disabled
  -> validation-only decoder calibration
  -> failure analysis
  -> seed 43 only if seed 42 passes
  -> one locked-test evaluation only if both validation runs pass
```

The full acceptance contract is in
[`reviewed-event-supervision-request.md`](reviewed-event-supervision-request.md).
Do not use the Gemini proposal file directly to unlock the decoder gate.
