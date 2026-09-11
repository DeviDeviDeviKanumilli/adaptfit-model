# AdaptFit production completion plan

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** dependency-ordered execution plan for completing and releasing the movement model
> - **Last verified:** 2026-09-11
> - **Source commit:** `b3926d8` plus the current follow-up working tree
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** supports [`HANDOFF.md`](../HANDOFF.md), [`current-state.md`](current-state.md), [`efficient-training-strategy.md`](efficient-training-strategy.md), and [`evaluation-protocol.md`](evaluation-protocol.md)
> - **Review trigger:** any new label source, checkpoint, decoder result, test evaluation, export, parity result, device result, or release decision

This is the copy-pasteable execution plan for the AdaptFit `/goal`. It is
ordered deliberately: each gate must pass before the next dependent action is
allowed. The agent should keep working through safe local actions, but must
stop and report an exact external dependency when the plan says that reviewed
labels, consented recordings, native/mobile source, or a physical device is
required.

## 1. Definition of done

The work is complete only when all of the following evidence exists:

1. A reproducible candidate movement model is trained from the immutable
   corrected-v1 parent or from a later candidate whose complete provenance is
   recorded.
2. The candidate passes the validation-only decoder gate on the fixed
   validation split:
   - repetition-count MAE `<= 0.40`;
   - repetition-end F1 `>= 0.60`;
   - zero false events on sequences whose target count is empty.
3. A second-seed confirmation passes the same validation gate without changing
   the data split, label policy, decoder policy, or test protocol.
4. Exactly one locked-test evaluation is run after the validation gates pass.
   The report includes sequence-level and window-level metrics, the candidate
   hash, config hash, split manifest, decoder version, and `test_split_used`.
5. The model bundle is exportable and loadable by the intended runtime, with
   the 283-feature contract, normalization, model weights, decoder version,
   metadata, and provenance bundled together.
6. Python reference inference and the native/mobile implementation agree on
   golden fixtures, including masks, logits/probabilities, event ordering,
   abstention, reset, timestamp gaps, and count behavior.
7. Android-first export, memory, latency, thermal, battery, and quantized
   parity checks pass on the intended device class, or the exact missing
   device/native dependency is explicitly recorded.
8. Quality, target-population, privacy, and recommendation claims are not made
   unless their separate evidence gates also pass.

Passing a training metric alone is not completion. A blocked decoder, absent
   label coverage, missing export, missing native parity, or unverified device
   gate keeps the model from being a release candidate.

## 2. Non-negotiable invariants

Keep these rules in force for the entire run:

- Work in `/Users/devk/AdaptFit`, not the empty Documents checkout.
- Preserve `artifacts/corrected-v1/` and its checkpoint unchanged. The
  corrected-v1 TCN SHA-256 is
  `e69ff69d2eaad85319bec80f25a5f2a8cb3d415169c5c4ef2441a8bc7f99011b`.
- Keep the participant/session-disjoint train, validation, and locked-test
  split fixed until validation passes.
- Never read, calibrate on, tune on, or mine the locked test split before the
  validation gate passes.
- Give every new data change a new prepared-data root and every new model or
  decoder intervention a new artifact root. Do not overwrite old evidence.
- Preserve the 283-feature, 128-frame, causal-TCN contract unless a separately
  justified contract experiment is created and fully revalidated.
- Keep unavailable labels masked. Do not turn missing boundaries, missing
  phase, missing apex, or missing quality labels into negatives or procedural
  first/last-frame labels.
- Do not fabricate clinical, phase, apex, repetition, quality, or
  target-population labels. Diagnostic proposals may be stored only as
  diagnostics and must remain masked for supervised training.
- Keep the four dimension-specific quality heads disabled while their label
  coverage is zero. UCO's composite expert score is a separate auxiliary
  target, not ROM, tempo, smoothness, or trunk-control supervision.
- Keep `decoder.v1` apex confirmation and tracking abstention enabled. Never
  relax thresholds or remove apex confirmation merely to force a pass.
- Record source licenses/access, participant and session identities, config
  hashes, checkpoint hashes, source commit, split manifest, device, seed, and
  test-lock status for every run.

## 3. Current starting point

Before doing anything else, read [`HANDOFF.md`](../HANDOFF.md),
[`current-state.md`](current-state.md), [`training-execution-log.md`](training-execution-log.md),
[`pretraining-readiness.md`](pretraining-readiness.md), and the relevant
contracts. Inspect the working tree before changing it:

```bash
cd /Users/devk/AdaptFit
git status --short
git log -1 --oneline
python3 scripts/validate_docs.py
python3 -m pytest -q
```

The latest local UL-RED boundary-source correction is retained in
`data/processed-r1-ulred-boundary-support/` and
`artifacts/r1-tcn-ulred-boundary-support/`. It recovered 219 markerless R3
three-repetition span sources, but its validation-only decoder result is still
blocked: end F1 `0.070718`, count MAE `0.684`, and 8 empty-target false events.
The strong UL-RED spans contain no reviewed hold/apex labels. The archive-wide
audit also found no additional local phase/apex source in the staged MM-Fit,
UCO, REHAB24-6, IntelliRehabDS, or UL-RED files. Therefore the current next
action is the reviewed-event-supervision intake, not another blind model-only
run.

## 4. Phase 0 — establish a clean evidence baseline

Run this phase after any new checkout or resumed task:

1. Confirm the repository path, branch, current commit, and dirty files.
2. Verify the corrected-v1 checkpoint hash and confirm no files under its
   artifact root changed.
3. Read the current artifact registry and execution log. Treat historical
   metrics as historical; do not promote a blocked checkpoint.
4. Run the documentation validator, full test suite, `compileall`, and
   `git diff --check`.
5. Run strict source/config preflight for the selected experiment. Confirm all
   enabled sources decode, licenses/access notes are present, participant
   groups are disjoint, and quality heads are disabled unless labels exist.
6. Run a runtime smoke check on the selected device. Confirm input shape
   `[batch, 128, 283]`, finite outputs, causal model behavior, masks, and all
   expected output heads.
7. If any check fails, fix the smallest source/test/config issue, add a
   regression test, rerun the checks, and record the result before training.

Do not spend training time on a configuration that fails preflight or runtime
validation.

## 5. Phase 1 — acquire and accept trustworthy supervision

The immediate missing input is reviewed temporal supervision for representative
seated, unilateral-friendly recordings. Preferred existing sources are UCO
and REHAB24-6; additional recordings are acceptable if their provenance,
license/access, participant identity, and split eligibility are clear. The
launch set is:

- seated one-arm biceps curl;
- seated one-arm band row;
- seated single-leg knee extension;
- seated single-leg march or hip lift; and
- seated forward reach.

Each reviewed repetition should provide a start, an end, and an observable
hold/apex interval, or an explicit reviewed-unavailable value for the apex.
Unavailable is a mask, not a negative label. Partial or uncertain repetitions
must be explicitly marked and excluded from exact count supervision.

Use one JSON object per repetition where possible. Required provenance and
label fields are defined in [`reviewed-event-supervision-request.md`](reviewed-event-supervision-request.md):

- source dataset, participant, session, exercise, side, and repetition index;
- source-frame start and end IDs;
- apex/hold source-frame interval when observable;
- phase-label availability and uncertainty/partial status;
- annotator, review status, review date, and source license/access note.

Before training, accept the delivery only if:

1. every row resolves to exactly one staged source sequence;
2. source frame IDs are monotonic and in bounds;
3. starts precede apex/hold, which precedes ends;
4. participant/session identities remain disjoint across all splits;
5. reviewed unavailable fields remain masked;
6. canonical phase vocabulary and frame-rate/resampling rules are respected;
7. one positive and one masked adapter regression fixture pass;
8. prepared arrays are finite and have the `[128, 283]` contract;
9. the prepared manifest reports label coverage and zero unknown-boundary
   endpoint leakage; and
10. the locked test split has not been read.

If these conditions cannot be met, stop and request the missing provenance or
review rather than filling the gap procedurally.

## 6. Phase 2 — one isolated supervised candidate

When accepted supervision exists:

1. Preserve the existing corrected roots unchanged.
2. Add or update only the required adapter and focused regression tests.
3. Run documentation validation and the full test suite.
4. Generate a new prepared root with a new manifest. Never modify an existing
   prepared root in place.
5. Run preflight and runtime checks again.
6. Start from the strongest prior validation candidate or the immutable
   corrected-v1 parent, recording the parent hash and normalization source.
7. Use a bounded training budget, seed 42 first, and keep test evaluation
   disabled (`--skip-test` or equivalent).
8. Preserve the same feature/window contract, label masks, decoder policy,
   participant split, and quality-head policy unless the new labels explicitly
   justify one isolated change.
9. Train only the outputs supported by trustworthy labels. Monitor per-task
   loss contributions so sparse boundary supervision is not hidden by abundant
   family labels.
10. Write checkpoints, metrics, manifest, effective config, hashes, and a
    human-readable run summary to the new artifact root.

The default escalation order is:

1. validation-only decoder calibration;
2. heads-first fine-tuning with the backbone frozen;
3. final-block fine-tuning;
4. final-two-block fine-tuning;
5. a short full-TCN pilot only if representation transfer is demonstrated;
6. one teacher experiment only if a measured remaining gap justifies it.

Change one factor at a time. Keep matched data, budgets, seeds, decoder
settings, and validation reconstruction for comparisons.

## 7. Phase 2A — validate and diagnose before escalating

Calibrate `decoder.v1` on validation only. Record the selected start, end,
phase, tracking-floor, apex, debounce, reset, timestamp-gap, and duration
settings. The decoder must retain tracking abstention and apex confirmation.

Then run the failure analyzer and classify the result:

- If labels or masks are wrong, correct the source-backed mapping, add a
  regression test, regenerate a new prepared root, and rerun the bounded
  candidate.
- If raw model outputs already contain the required event evidence, a narrow
  decoder-only change may be evaluated in a separately named artifact root.
  Do not change thresholds just because the model lacks evidence.
- If tracking, end, or apex signals are absent, do not force the decoder to
  accept them. Acquire reviewed supervision or representative recordings.
- If one staged fine-tuning level improves the diagnostic but does not pass,
  advance exactly one level in the escalation order.
- Reject runs that worsen empty-target false events, abstention safety,
  forgetting, or subgroup behavior even if a composite score rises.

Use the current failure-analysis fields as the decision evidence: target-pair
coverage, nearby start/end crossings, tracking-floor passes, qualified apex
intervals, predicted hold frames, abstention reasons, empty-target false
events, and source-specific results.

## 8. Optional density/count branch

A density head is allowed only after valid boundary supervision is verified and
the residual failure is specifically counting. It must be a separate isolated
experiment, not combined with a new sampler, new backbone, teacher, and new
decoder simultaneously.

The target must be nonnegative, expressed as repetitions per frame, sum to the
labeled repetition count over fully labeled intervals, preserve fractional
mass in cropped windows, and be masked where labels are unavailable. Causal
inference must accumulate each prediction exactly once across overlapping
windows. The density branch cannot replace the required start/end/apex event
gate; do not start it merely to avoid requesting missing hold/apex labels.

If implemented, add the target contract, prepared-array provenance, model and
streaming output tests, density-specific validation metrics, and a new
artifact root. Compare it against the same supervised-only candidate and keep
it only if it materially improves count without regressing event F1,
abstention, empty-target false events, or subgroup behavior.

## 9. Optional teacher branch

Do not begin teacher work while trustworthy supervised escalation is still
unresolved. If a measured gap remains after supervised candidates:

1. choose one teacher only;
2. verify license, compatible checkpoint, input mapping, timestamps, masks,
   and split provenance;
3. generate teacher targets from training data only;
4. compare teacher-assisted and supervised-only students with matched budgets;
5. use SSTRAC first for a demonstrated density/count gap;
6. use AF-MJEPA only for a demonstrated representation-transfer gap; and
7. reject the teacher if total compute is unjustified or count, subgroup,
   abstention, or streaming behavior regresses.

Motion-JEPA, recommender learning, and broad representation pretraining are
not prerequisites for the first successful supervised movement candidate.

## 10. Phase 3 — reproducibility and locked test

Do not touch test until one seed passes the complete validation decoder gate.
Then:

1. run seed 43 from the same starting point and prepared root, with the same
   validation reconstruction and decoder settings;
2. calibrate and analyze seed 43 on validation only;
3. require seed 43 to pass all three decoder conditions;
4. compare seed 42 and seed 43 for family, phase, boundary, end events,
   count, tracking, abstention, and source/profile slices;
5. select the candidate by the predeclared rule and record both runs;
6. run exactly one locked-test evaluation using the selected candidate;
7. never use test results to change the model, decoder, split, or claims; and
8. write the test manifest, report hashes, and final decision to the registry
   and execution log.

If either validation seed fails, return to the failure-driven escalation. Do
not average away a failed gate or choose the seed using test results.

## 11. Phase 4 — runtime, export, and Android-first release gates

After the supervised validation and single locked-test evaluation pass, build
the deployable bundle in a new release artifact root. Include:

- model weights and architecture identifier;
- feature schema and normalization statistics;
- exercise/variant/side metadata and capability assumptions;
- decoder version and all thresholds/state-machine constants;
- provenance, source commit, config hash, checkpoint hash, and license notes;
- model input/output tensor signatures; and
- compatibility/version checks.

Verify the following in order:

1. Python save/load round trip;
2. streaming state reset, timestamp gap, pause, low-tracking, and context
   change behavior;
3. Python golden fixtures for logits/probabilities and serialized events;
4. native bridge input normalization, masks, tensor layouts, and event
   serialization;
5. Python/native numerical parity on float fixtures;
6. export loadability and output-shape checks;
7. float-versus-quantized parity with declared tolerances;
8. Android inference latency, memory, thermal, battery, and long-session
   stability on the intended physical device; and
9. packaging, version mismatch, missing-input, and safe-abstention behavior.

If the native repository or target device is unavailable, stop at that gate
and report the exact dependency. Do not describe Python success as mobile
availability.

## 12. Quality, target population, privacy, and recommendation gates

These are separate from the core movement-model gate:

- Obtain reviewed labels before enabling the four dimension-specific quality
  outputs. Keep UCO composite expert quality separate.
- Obtain consented recordings from actual target populations before making
  amputee, limb-difference, wheelchair-user, or clinical claims. Public seated
  posture proxies and synthetic occlusions do not satisfy this gate.
- Verify raw camera/pose locality, retention, deletion, consent, and telemetry
  behavior in the production mobile build.
- Build recommendation in dependency order: reviewed recipe metadata, hard
  feasibility mask, content/rules baseline, feedback logging, then a neural
  ranker only after user/time-held-out evaluation data exists.

Do not let a future quality, recommender, or world-model experiment obscure a
blocked movement decoder or substitute for missing target-population evidence.

## 13. Stopping, retry, and evidence rules

After every intervention:

1. run the focused regression tests;
2. run docs validation, full tests, compile checks, and diff checks as
   appropriate;
3. verify old roots and locked-test files were not modified;
4. record the new root, config, parent, seed, device, checkpoint/report hashes,
   metrics, gate result, and failure analysis;
5. compare the relevant diagnostic, not only the composite score; and
6. update [`HANDOFF.md`](../HANDOFF.md), [`current-state.md`](current-state.md),
   [`artifact-registry.md`](artifact-registry.md), and the execution log.

If three consecutive isolated experiments show the same failure mechanism
without a material diagnostic change, stop training and request the missing
external input. Do not respond by increasing epochs indefinitely. If a required
external input is absent, leave the goal active only while safe local checks or
documentation work can make progress; otherwise report the exact blocker and
the resume command/order.

## 14. Resume sequence after the current blocker clears

Use this exact order when reviewed rows or new recordings arrive:

```text
receive reviewed rows
  -> verify provenance, license/access, identity, frame bounds, and adjudication
  -> add adapter and positive/masked regression fixtures
  -> run docs validator, compile checks, diff check, and full tests
  -> prepare a new isolated root; preserve corrected roots unchanged
  -> run strict preflight and runtime smoke checks
  -> train one bounded seed-42 validation-only candidate
  -> calibrate decoder.v1 on validation only
  -> run failure analysis and update the execution log
  -> if and only if the gate passes, train seed 43
  -> if and only if both validation runs pass, evaluate the locked test once
  -> build export bundle and Python/native golden fixtures
  -> pass Android export, parity, quantization, and physical-device gates
  -> complete privacy, target-population, quality, and recommendation evidence
  -> call the model complete only after every applicable gate is evidenced
```

The current blocker is at the first line of this resume sequence: reviewed
start, hold/apex, and end supervision (or additional representative recordings
with those labels) is not yet available locally. Until it arrives, preserve the
UL-RED and UCO corrected roots, keep quality heads disabled, keep the test split
locked, and do not promote a blocked checkpoint.
