# AdaptFit handoff

> **Status:** canonical-active
> **Authority:** continuation instructions for the next implementation or evaluation agent
> **Last verified:** 2026-09-11
> **Source commit:** `b3926d8` (current handoff and calibration run); R1 evidence at `d7848e2`
> **Owner:** AdaptFit training and evaluation engineering
> **Review trigger:** new checkpoint, decoder result, test evaluation, schema/config change, or repository revision

This is the short operational handoff for `/Users/devk/AdaptFit`. Read it
before running commands. The longer contracts and rationale remain in
[`docs/README.md`](docs/README.md), [`docs/current-state.md`](docs/current-state.md),
and [`docs/pretraining-readiness.md`](docs/pretraining-readiness.md).

## Repository state

- Source repository: `/Users/devk/AdaptFit`
- The empty `/Users/devk/Documents/ChatGPT/AdaptFit` directory is not the source of truth.
- Branch: `main`
- Last pushed documentation commit: `b3926d8` (`Add current project handoff`)
- Remote: `github/main`; verify with `git status --short` and `git ls-remote github refs/heads/main`.
- Training artifacts are local and ignored by Git. Their manifests, hashes, and limitations are recorded in the execution log.

## What is complete

The corrected-v1 baseline remains immutable:

- TCN parent: `artifacts/corrected-v1/checkpoints/tcn_best.pt`
- Parent SHA-256: `e69ff69d2eaad85319bec80f25a5f2a8cb3d415169c5c4ef2441a8bc7f99011b`
- Contract: 283 features, 128 frames, 96-channel causal TCN, five dilated blocks, 125-frame receptive field.
- Prepared split: `data/processed-corrected-v1`; 160,813 train windows, 5,204 validation windows, 7,592 test windows; participant overlap is zero.

The R1 heads-only run is complete in `artifacts/r1-tcn-boundary/`:

- Config: `training/configs/experiments/r1_tcn_boundary_finetune.yaml`
- Parent: corrected-v1 TCN above
- Device: Apple MPS, float32
- Backbone: frozen; 1,746 output-head parameters trained
- 27 epochs completed; early stopping selected epoch 12
- Best checkpoint: `artifacts/r1-tcn-boundary/checkpoints/tcn_best.pt`
- Best checkpoint SHA-256: `e74679f9ebe0b012d4ed8729120aa8e3489c960ea7941a24d7e94f9e8ff45cbf`
- Run report: `artifacts/r1-tcn-boundary/metrics.json`
- Run report SHA-256: `7d340a204b924d1d46bffa68f2833a83cdcaf2df3be764b985856c77cccd3c4d`
- Source commit recorded by the run: `d7848e2b12322e40e83758abd5e12ae6c7727e80`
- Test evaluation: **not performed**

Best validation sequence metrics:

| Metric | Value |
|---|---:|
| Composite validation score | `0.6603763` |
| Family macro-F1 | `0.902795` |
| Phase macro-F1 | `0.569988` |
| Boundary F1 | `0.548749` |
| Repetition-start F1 | `0.988439` |
| Repetition-end F1 | `0.109059` |
| Count MAE | `0.155134` |
| Four-head quality-label coverage | `0.0%` |

The composite score is not accuracy. `training/src/metrics.py` combines family,
phase, boundary, and quality metrics; because quality labels are unavailable,
the quality term is omitted and the remaining weights are renormalized. The
family result is strong, but boundary and repetition-end performance are still
weak. This checkpoint is not a release candidate.

## R1 decoder calibration result

Validation-only decoder calibration was run on the R1 epoch-12 checkpoint. It
read the validation split only, tried 18 configurations in the versioned
threshold neighborhood, and wrote this ignored local report:

```bash
cd /Users/devk/AdaptFit
python3 -m training.calibrate_decoder \
  --config training/configs/experiments/r1_tcn_boundary_finetune.yaml \
  --checkpoint artifacts/r1-tcn-boundary/checkpoints/tcn_best.pt \
  --decoder-config training/configs/decoder_v1.yaml \
  --output artifacts/r1-tcn-boundary/decoder/decoder_calibration.json \
  --project-root /Users/devk/AdaptFit \
  --device mps
```

The selected configuration was `start_threshold=0.55`,
`end_threshold=0.55`, `tracking_floor=0.50`, with apex confirmation enabled.
The report records:

- checkpoint SHA-256: `e74679f9ebe0b012d4ed8729120aa8e3489c960ea7941a24d7e94f9e8ff45cbf`;
- source commit: `b3926d864c1d976504c2d7131db8d4992cf2ef45`;
- 896 validation sequences and 327 target boundary pairs;
- predicted count `0`, missed events `327`, repetition-end F1 `0.0`, count MAE
  `0.3649554`, and zero false events on empty sequences;
- `test_split_used: false`.

The provisional decoder gate is:

- count MAE ≤ `0.40`;
- repetition-end F1 ≥ `0.60`;
- false events on empty target sequences = `0`.

Do not change thresholds, remove apex confirmation, or read the test split to
force a pass. Record the selected thresholds, gate status, checkpoint hash,
source commit, and report path.

The gate is **blocked**: count MAE and empty-sequence false events pass, but
repetition-end F1 fails. The R1 checkpoint is not promoted and has no test
metrics.

## Prior next action (completed)

Inspection of representative misses and the label/mask path found a targeted
boundary-data correction:

- the decoder emitted 266 `low_tracking` abstentions, 55
  `partial_without_apex` events, and no positive events;
- high raw boundary probabilities exist, but the selected decoder rarely sees
  the required tracking floor and two high-confidence hold/apex frames;
- 219 target pairs come from REHAB24-6 segmentation, 87 from UL-RED records
  whose boundary source is `unknown`, and 21 from procedural templates;
- `training/src/data/adapters.py::_make_sequence` currently defaults a missing
  `rep_boundary` to first/last-frame labels. UL-RED uses that default while
  declaring its boundary source unknown, so those 87 validation sequences are
  incorrectly supervised as endpoint-boundary examples.

The chosen follow-up is an isolated label/mask correction: make unavailable
source boundaries remain masked, add a regression test, regenerate a new
prepared-data root, and run one bounded boundary experiment from the immutable
corrected-v1 parent. Do not alter thresholds, overwrite corrected-v1, or start
JEPA, recommender, mobile, or broad retraining work.

The correction is implemented and the isolated follow-up has completed its
bounded heads-only training and validation-only decoder calibration:

- Code/config: `training/src/data/adapters.py`,
  `training/tests/test_data_contracts.py`, and
  `training/configs/experiments/r1_tcn_boundary_mask_correction.yaml`;
- Prepared root: `data/processed-r1-boundary-mask-correction`;
- Boundary coverage: `1,372` labeled sequences; UL-RED boundaries remain
  masked, while REHAB24-6 segmentation boundaries are explicit;
- Artifact root: `artifacts/r1-tcn-boundary-mask-correction`;
- Training: completed epoch `68`; best validation checkpoint is epoch `53`,
  with `1,746` trainable head parameters and exact resume from the paused epoch
  10 checkpoint;
- Best checkpoint SHA-256:
  `317bbbf782f5c69f2888737e8bdcec914c8d812e71001c4f8c718fe861479505`;
- Validation calibration report:
  `artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_calibration.json`;
- Calibration: `896` validation sequences, `240` target boundary pairs,
  predicted count `0`, missed events `240`, count MAE `0.2678571`, end F1
  `0.0`, and zero false events on empty sequences;
- Calibration report SHA-256:
  `855b325d27b3ec7fd1e7124a6c57daaafd2ee1413de2b4323297756f25ec937d`;
- Failure analysis: [`decoder_failure_analysis.md`](artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_failure_analysis.md)
  and [`decoder_failure_analysis.json`](artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_failure_analysis.json);
  the analysis finds `0/240` qualified apex intervals, `178/219` segmentation
  starts below the tracking floor, and only `88/240` target ends with a nearby
  end-probability crossing;
- Failure-analysis JSON SHA-256:
  `868cb2327554cf0e2e124f937b838f81a343c223ef54460298e5ffc158766199`;
- Test evaluation: **not performed**; `test_split_used` is false.

The phase-weight correction follow-up has now completed its isolated
training-only run:

- Code/config: `training/src/runner.py`, `training/tests/test_models.py`, and
  `training/configs/experiments/r1_tcn_phase_weight_correction.yaml`;
- The runner now ignores absent classes when computing normalized class weights;
  the absent `unknown` phase class receives zero weight instead of suppressing
  the rare but valid `hold` class;
- Prepared root: the existing verified
  `data/processed-r1-boundary-mask-correction` root was reused unchanged;
- Artifact root: `artifacts/r1-tcn-phase-weight-correction`;
- Parent checkpoint:
  `artifacts/r1-tcn-boundary-mask-correction/checkpoints/tcn_best.pt`;
- Training: Apple MPS, frozen backbone, 1,746 trainable head parameters;
  completed 24 epochs and early-stopped at epoch 24, with best epoch 9;
- Best validation sequence score: `0.6679357`;
- Best validation sequence metrics: family macro-F1 `0.827578`, phase
  macro-F1 `0.588146`, boundary F1 `0.614690`, repetition-start F1 `0.996094`,
  repetition-end F1 `0.233286`, and count MAE `1.155134`;
- Best checkpoint SHA-256:
  `2bac2156ae7ef85b39e65ce5e1d8eff9fe4ea5799ce6b28c2f5ffcf9d30ebf5a`;
- Latest checkpoint SHA-256:
  `eb0c14ad8c4320d4bed460168c0e2ca1d92deba891663669a7015ba5e654dc79`;
- Run report: `artifacts/r1-tcn-phase-weight-correction/metrics.json`;
  SHA-256 `72dfbfe56f9347306774969c05727ca46991d18afc3b0e8640bb34f42cc12f20`;
- Test evaluation: **not performed**; decoder calibration is recorded below.

## Phase-weight correction validation calibration

The phase-weight checkpoint was calibrated with `decoder.v1` on the validation
split only. The report is
`artifacts/r1-tcn-phase-weight-correction/decoder/decoder_calibration.json`.
The selected thresholds were start `0.65`, end `0.65`, phase `0.45`, and
tracking floor `0.50`, with apex confirmation enabled. It records:

- 896 validation sequences and 240 target boundary pairs;
- predicted count `20`, matched count `4`, missed events `236`;
- event F1 `0.030769`, start F1 `0.115385`, end F1 `0.153846`;
- count MAE `0.279018`;
- false events on empty target sequences `15`;
- calibration report SHA-256:
  `a4df7614d2f2932ba4093352d20c19e83a355ac1cc9e4b4be98e213cfee18b40`;
- `test_split_used: false`.

## Decision after calibration

The corrected boundary-mask follow-up calibration is **blocked**: count MAE and
empty-sequence false-event checks pass, but end F1 remains `0.0` against the
required `0.60`. The phase-weight correction calibration is also **blocked**:
count MAE passes, but end F1 is `0.153846` and false events on empty sequences
are `15`, both failing their gates. Do not promote either checkpoint or run the
locked test evaluation. The corrected label/mask path is now verified. The
detailed validation-only diagnosis is recorded in
[`decoder_failure_analysis.md`](artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_failure_analysis.md):
the dominant blocker is missing model-predicted hold/apex support, with
tracking-floor and end-signal alignment failures concentrated in REHAB24-6.

The phase-weight checkpoint was replayed on validation only. The detailed
reports are
[`decoder_failure_analysis.md`](artifacts/r1-tcn-phase-weight-correction/decoder/decoder_failure_analysis.md)
and
[`decoder_failure_analysis.json`](artifacts/r1-tcn-phase-weight-correction/decoder/decoder_failure_analysis.json).
They find:

- only `22/240` target intervals satisfy the two-frame apex rule, and only
  `31/240` contain any predicted hold;
- `154/219` explicit REHAB24-6 segmentation starts are below the `0.50`
  tracking floor;
- only `83/240` target ends have an end-probability crossing within ±250 ms;
- procedural templates qualify `21/21` target intervals for apex support, while
  REHAB24-6 qualifies only `1/219`;
- the corrected path contains `219` segmentation plus `21` procedural target
  pairs and `0` unknown-boundary target pairs, so unavailable-boundary leakage
  is not the remaining failure.

The report SHA-256 values are JSON
`9d940f237bb630fb28d1ac1af4fdc4db4d2e01d91c76d47d0c64f7f0ac700076` and
Markdown
`0c73586023e9a2b5b568d2a664b3d54d82e53a1d84a24b87eee95e74ebeeb29e`.

The phase-weight checkpoint still does not pass its decoder gate. The selected
last-block follow-up was trained from
`artifacts/r1-tcn-phase-weight-correction/checkpoints/tcn_best.pt`, with only
`blocks.4.*` and the existing heads trainable (`57,426` parameters). It selected
epoch 8 after early stopping at epoch 23 and reached validation sequence score
`0.6791891`, but its validation-only decoder gate remains blocked: end F1
`0.094862`, count MAE `0.271205`, and 8 empty-target false events. Its config is
`training/configs/experiments/r1_tcn_last_block_event_support.yaml`; its best
checkpoint is
`artifacts/r1-tcn-last-block-event-support/checkpoints/tcn_best.pt`, SHA-256
`e1ac1ea978b323e128d9e966b067f8c5af073358e86846e8e7c046ec11a156de`.
Validation-only failure analysis still finds `21/240` apex-qualified intervals,
`219/219` segmentation starts below the selected tracking floor, and `92/240`
target ends with nearby end signal. Test remains locked.

Following the escalation ladder, the final-two-block fine-tuning intervention
was run from that best last-block checkpoint, with only `blocks.3.*`,
`blocks.4.*`, and the existing heads trainable (`113,106` parameters). Its config is
`training/configs/experiments/r1_tcn_last_two_blocks_event_support.yaml` with
SHA-256 `10a4df87a984f937202ab26d1ca247afca1245b5e57ba47141428cf1ea8d467c`,
and its isolated artifact root is
`artifacts/r1-tcn-last-two-blocks-event-support/`. Reuse
`data/processed-r1-boundary-mask-correction/` unchanged, keep seed 42, loss
weights, label masks, decoder thresholds, and
`evaluate_test_after_training=false`. It selected epoch 42 of 57 with validation
sequence score `0.7040537`; validation-only calibration was blocked at end F1
`0.299674`, count MAE `0.333705`, and `63` empty-target false events. Its
failure analysis finds `49/240` apex-qualified intervals, `219/219` segmentation
starts below the tracking floor, and `91/240` target ends with nearby end
signal. Test evaluation remains locked.

The short full-TCN fine-tuning pilot was then run from the final-two-block best
checkpoint. Its config is
`training/configs/experiments/r1_tcn_full_finetune_event_support_pilot.yaml`
with SHA-256 `4023e3b277ad874f1c9ad2492dff587e6d4e2e1b0f2f16019f96444bc22471cd`,
and its artifact root is
`artifacts/r1-tcn-full-finetune-event-support-pilot/`. All `307,410` model
parameters were trainable; the backbone/head learning rates were `1e-5`/`3e-4`,
and the maximum budget was 8 epochs with patience 3. Preflight and runtime
checks passed. Best epoch 6 reached validation sequence score `0.7176679`; the
validation-only decoder gate remained blocked (end F1 `0.296053`, count MAE
`0.330357`, and 60 empty-target false events). Failure analysis still finds
`219/219` segmentation starts below the tracking floor and only `101/240`
target ends with nearby end signal. Stop model-only escalation and request
reviewed REHAB24-6 boundary labels or additional representative labeled
recordings; do not read the locked test split.

The UCO exact-boundary follow-up was then completed in two isolated steps. The
partial-pose tracking-target correction records only the canonical joints
structurally represented by each source, fixing the artificial tracking-floor
failure without changing boundary labels or decoder policy. Its config is
`training/configs/experiments/r1_tcn_uco_tracking_target_correction.yaml`, its
artifact root is `artifacts/r1-tcn-uco-tracking-target-correction/`, and its
best checkpoint is epoch 2 with SHA-256
`bc69117c1a287a27fcf63466238ab97da812eda3ff92e5a4db968f94ef80d20`. Its
validation-only decoder gate remains blocked: end F1 `0.070130`, count MAE
`0.584`, and 7 false events on empty-target sequences. Tracking-floor
abstentions are no longer the explanation: UCO still has `0/507` nearby
starts, `1/507` nearby ends, and `0/507` apex-qualified targets, and its exact
recording boundaries carry no per-frame hold/apex labels.

One loss-only intervention then raised the sparse-boundary positive-weight cap
to `64.0` in a new artifact root
`artifacts/r1-tcn-uco-boundary-positive-weight/`. It selected epoch 1 with
validation sequence score `0.5252875`, down from `0.6176060`, and remained
blocked at end F1 `0.142322`, count MAE `0.5864`, and 24 empty-target false
events. Reject the cap increase and do not repeat it. The current dependency
is reviewed UCO and/or REHAB24-6 start, hold/apex, and end supervision, or
additional representative seated unilateral recordings with those labels.
Preserve the corrected prepared root and keep the locked test split
unevaluated until the supervision is available.
The exact intake contract is [reviewed-event-supervision-request.md](docs/reviewed-event-supervision-request.md).

If a later candidate passes all decoder requirements, freeze it and run the
locked test evaluation exactly once with its own config and checkpoint:

```bash
python3 -m training.evaluate \
  --config <passing-candidate-config> \
  --checkpoint <passing-candidate-checkpoint> \
  --project-root /Users/devk/AdaptFit \
  --device mps
```

Compare R1 with corrected-v1 using the same split, sequence aggregation,
decoder version, and provenance fields. Update the artifact registry before
describing any result as an improvement.

If the configured run is executed, recalibrate its checkpoint on validation only
with the decoder thresholds held fixed for the first comparison. Do not add
event sampling, label edits, threshold changes, or another loss-weight change
to this intervention. Promote or read the locked test split only if the
declared decoder gate passes. Do not start JEPA, recommender learning, or mobile
export until the supervised movement comparison and decoder gate establish a
measured need and a valid artifact.

## Latest local-source follow-up: UL-RED 3Rep spans

The prior handoff requested reviewed event supervision. Before stopping, a raw
source audit found a valid local label source that the previous adapter had
discarded: S01.zip through S10.zip contain marker-less/3Rep_Sxx.csv, and 219
of 219 matching R3 AMC recordings have explicit start/end spans. CSV indices
are zero-based; AMC frame IDs are one-based, so the adapter now maps them
explicitly and masks malformed or out-of-bounds rows. No phase, hold, or apex
labels are fabricated.

The isolated follow-up is complete:

- Config: training/configs/experiments/r1_tcn_ulred_boundary_support.yaml
  (file SHA-256 4a8fa4def53d686ca57c29a8958f6eae1dca7a5ff77b38b684d74601dd1ad198;
  effective hash sha256:320a23fd5cf43a79ed821e08b90a419b221b9ec87f57156be77561c9115bfc04).
- Prepared root: data/processed-r1-ulred-boundary-support/; 7,315
  sequences, 86,076 train windows, 16,483 validation windows, 7,536 test
  windows; 219 explicit UL-RED R3 sequences. The overlay provenance is in
  overlay_provenance.json; the previous root is unchanged.
- Parent: UCO tracking-target correction checkpoint
  bc69117c1a287a27fcf63466238ab97da812eda3ff92e5a4db968f94ef80d20.
- Artifact: artifacts/r1-tcn-ulred-boundary-support/; full TCN, MPS,
  307,410 trainable parameters, 5 epochs, best epoch 2, validation sequence
  score 0.6013545; checkpoint SHA-256
  7ed351765ad1c943d525721c03fdcec65c1fdd06df2966c59971ca934b5d4ce1.
- Validation-only decoder calibration: target pairs 872, predicted 33,
  matched 22, missed 850, event F1 0.048619, end F1 0.070718, count MAE
  0.684, and 8 false events on empty-target sequences; gate blocked.
  Calibration SHA-256 c7fb7b28e9d92f496d0b38c1cf3e44614129d5113ee809a06ddd47844fdedac1.
- Failure analysis: JSON SHA-256
  4715afb36b623ec437ca3256d4444f4f41d8ba7bb016a660cd57dfe7c63ced52;
  Markdown SHA-256 6476e330b670189d46a22af30fc733658d527e575a7f0d6ad40d24e048e0e66e.
- Test evaluation: not performed; test_split_used=false.

The current next action is therefore reviewed UL-RED/UCO/REHAB24-6 start,
hold/apex, and end supervision, or additional representative seated unilateral
recordings with those labels. Preserve both corrected prepared roots, keep
quality heads disabled, and do not lower tracking/apex decoder gates. After
reviewed labels arrive, create a new prepared and artifact root, rerun
preflight, runtime, bounded training, validation-only calibration, and failure
analysis. A passing validation gate is required before a second seed and
exactly one locked-test evaluation; then complete export, Python/native parity,
device, quantization, privacy, and target-population gates before calling the
model finished.

## Hard boundaries

- No test metrics exist for any R1 candidate; all R1 follow-ups are
  validation-only and their decoder gates are blocked.
- R1 decoder calibration exists at `artifacts/r1-tcn-boundary/decoder/decoder_calibration.json` and is **blocked**.
- Corrected mask follow-up calibration exists at
  `artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_calibration.json`
  and is **blocked**; it records zero accepted events and `240` missed target
  events on validation.
- Phase-weight correction training exists at
  `artifacts/r1-tcn-phase-weight-correction/`; its best checkpoint is
  validation-calibrated but its decoder gate is **blocked**; it has not been
  test-evaluated.
- Last-block event-support intervention is complete at
  `training/configs/experiments/r1_tcn_last_block_event_support.yaml` and points
  to `artifacts/r1-tcn-last-block-event-support/`; its validation-only decoder
  gate is **blocked** and it has no test metrics.
- Last-two-block event-support intervention is complete at
  `training/configs/experiments/r1_tcn_last_two_blocks_event_support.yaml` and
  points to `artifacts/r1-tcn-last-two-blocks-event-support/`; its validation-only
  decoder gate is **blocked** (end F1 `0.299674`; count MAE `0.333705`; 63
  empty-target false events) and it has no test metrics.
- Full-TCN event-support pilot is complete at
  `training/configs/experiments/r1_tcn_full_finetune_event_support_pilot.yaml`
  and points to `artifacts/r1-tcn-full-finetune-event-support-pilot/`; all
  parameters were trainable, its validation-only decoder gate is **blocked**,
  and test remains locked.
- UCO tracking-target correction is complete at
  `training/configs/experiments/r1_tcn_uco_tracking_target_correction.yaml`
  and points to `artifacts/r1-tcn-uco-tracking-target-correction/`; its
  validation-only decoder gate is **blocked** (end F1 `0.070130`; count MAE
  `0.584`; 7 empty-target false events) and it has no test metrics.
- UCO boundary-positive-weight intervention is complete at
  `training/configs/experiments/r1_tcn_uco_boundary_positive_weight.yaml` and
  points to `artifacts/r1-tcn-uco-boundary-positive-weight/`; cap `64.0` was
  rejected after validation sequence score fell to `0.5252875`, and its
  validation-only decoder gate is **blocked** (end F1 `0.142322`; count MAE
  `0.5864`; 24 empty-target false events). Do not repeat the cap increase.
- The next required input is reviewed UCO/REHAB24-6 start, hold/apex, and end
  supervision or additional representative labeled recordings. Until then,
  no further blind model-only escalation is authorized.
- Four dimension-specific quality heads have zero reviewed label coverage.
- No real amputee, limb-difference, or wheelchair-user validation cohort exists.
- No production exporter, native bridge, quantized bundle, or mobile parity suite exists.
- The Python decoder is a reference implementation, not a production mobile runtime.
- Do not make clinical, medical, safety, privacy, or target-population claims.
- Do not overwrite `artifacts/corrected-v1/` or the locked test split.

## Required checks after any change

```bash
python3 scripts/validate_docs.py
python3 -m pytest -q
git diff --check
git status --short
```

The canonical execution record is
[`docs/training-execution-log.md`](docs/training-execution-log.md). Update it
after every calibration, evaluation, checkpoint, or decision.
