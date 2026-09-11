# AdaptFit pre-training readiness

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** reproducibility gate for the next training run; source code, configs, and generated reports remain authoritative
> - **Last verified:** 2026-09-11
> - **Source commit:** `b3926d864c1d976504c2d7131db8d4992cf2ef45` (R1 calibration and phase-weight correction training)
> - **Owner:** AdaptFit training and evaluation engineering
> - **Supersedes or supports:** supports `current-state.md`, `efficient-training-strategy.md`, `training-execution-log.md`, and `artifact-registry.md`
> - **Review trigger:** any change to the selected config, prepared split, checkpoint, decoder, label policy, or training runner

This document is the handoff between preparation and the next authorized model
training run. It freezes the evidence already checked and gives Luna one
ordered path. It does **not** select a product release or promote the decoder.
The generated artifacts referenced here are local, ignored R0/R1 reports and
checkpoints; they must be regenerated when absent.

## Current decision

The bounded smoke run, original R1 heads-only run, corrected boundary-mask
follow-up, phase-weight correction, last-block representation training, and
final-two-block representation training have completed. Every completed R1
decoder gate remains **blocked**: the final-two-block candidate reached end F1
`0.299674` with `63` false events on empty-target sequences; the last-block
candidate reached end F1 `0.094862` with `8`; the phase-weight candidate reached
end F1 `0.153846` with `15`; and the corrected-mask candidate reached end F1
`0.0`. These fail the required end F1 `0.60` and empty-target false-event
checks. The unavailable-boundary label/mask correction is implemented and
regression-tested; the corrected-v1 checkpoint, corrected prepared data, and
locked test split remain immutable. Product release remains blocked until the
decoder, quality supervision, target-population evidence, and native parity
gates pass in their own artifacts.

The final-two-block run improved the validation sequence score to `0.7040537`,
and the short full-TCN pilot reached `0.7176679`, but both remained blocked by
the decoder gate. A separate UCO exact-boundary path then corrected the
partial-pose tracking denominator: the tracking-target correction reached
validation sequence score `0.6176060`, but calibration remained blocked at end
F1 `0.070130`, count MAE `0.584`, and `7` empty-target false events. The
tracking correction removed the artificial tracking-floor failure, yet UCO
still supplied `0/507` nearby starts, `1/507` nearby ends, and `0/507`
apex-qualified targets; its exact boundaries have no per-frame hold/apex
labels. A single sparse-boundary positive-weight cap intervention (`64.0`)
reduced the validation sequence score to `0.5252875` and increased
empty-target false events to `24`. Reject that cap and do not read the test
split or start another blind model-only intervention without reviewed event
supervision.

A final local-source audit recovered explicit UL-RED `marker-less/3Rep_Sxx.csv`
spans for 219 of 219 available R3 recordings. The isolated UL-RED boundary
follow-up reached validation sequence score `0.6013545`, but its validation-only
decoder calibration remained blocked at end F1 `0.070718`, count MAE `0.684`,
and 8 empty-target false events. The spans are boundary-only and provide no
hold/apex labels; keep `data/processed-r1-ulred-boundary-support/` and its
artifact root immutable and request reviewed start, hold/apex, and end
supervision before another model-only run.

## Frozen baseline and provenance

| Item | Frozen value | Evidence |
|---|---|---|
| Active repository | `/Users/devk/AdaptFit` | `git rev-parse --show-toplevel` |
| Baseline checkpoint | `artifacts/corrected-v1/checkpoints/tcn_best.pt` | SHA-256 `e69ff69d2eaad85319bec80f25a5f2a8cb3d415169c5c4ef2441a8bc7f99011b` |
| GRU comparison checkpoint | `artifacts/corrected-v1/checkpoints/gru_baseline.pt` | SHA-256 `4105bf23b2bcbfea768a5abe15cf38d5277e341bbad889456ca567625036ea27` |
| Feature schema | `adaptfit.features.v1`, 283 inputs | `artifacts/corrected-v1/feature_schema.json`, SHA-256 `c4f5a4a2dc983ef0d338377655f75d7c96732d3c4fcc15dbc268be48925499d6` |
| Prepared split | `data/processed-corrected-v1` | `index.json`, SHA-256 `e68b88e000dd1e1b88b94f0bf98a138ee6d192a169ae6340c323c84b4cd06bee` |
| Window contract | 128 frames, approximately 30 FPS, stride 8 | `training/configs/v1_corrected.yaml` and R0 config |
| TCN architecture | 96 channels; kernel 3; dilations 1/2/4/8/16; causal receptive field 125 frames | source/config inspection |
| R0 config | `training/configs/experiments/r0_baseline.yaml` | SHA-256 `bc76c5de5bc7c4f9cfd85e8770aa9faf2f3454b004b3d622ed828dc99abebce9` |
| R1 config | `training/configs/experiments/r1_tcn_boundary_finetune.yaml` | SHA-256 `5a3682e943a50902e50e137dd3202e239013f9589f49401467e07636b26e196e` |
| Decoder contract | `decoder.v1`; validation-only calibration | `training/src/decoder.py`, `training/configs/decoder_v1.yaml` |

The current prepared audit has 160,813 train windows across 50 groups, 5,204
validation windows across 11 groups, and 7,592 test windows across 11 groups,
with zero participant overlap and zero identity collisions. There are no real
amputee, limb-difference, or wheelchair-user recordings in these splits. The
quality-head coverage is zero. These facts are constraints on interpretation,
not evidence that a product gate has passed.

## R0 verification already completed

R0 evaluates the immutable corrected-v1 checkpoints with the corrected label
policy: 1,042 `multi_rep_weak_phase` validation/test windows are masked for
phase, and no quality labels are fabricated. The test report is an audit of the
locked baseline, not a tuning set.

The local reports are:

- `artifacts/r0-baseline/metrics/tcn_evaluation.json`;
- `artifacts/r0-baseline/metrics/gru_evaluation.json`;
- `artifacts/r0-baseline/manifests/tcn_model_artifact_manifest.json`;
- `artifacts/r0-baseline/manifests/gru_model_artifact_manifest.json`.
- `artifacts/r0-baseline/prepared_audit.json` (strict storage and split audit);

The corrected-v1 TCN test sequence metrics in that report are family macro-F1
`0.8757019`, phase macro-F1 `0.6024214` after the mask, boundary F1 `0.5900653`,
repetition-count MAE `0.3111831`, repetition-start F1 `0.9953917`, and
repetition-end F1 `0.1847390`. The GRU is a baseline comparison only; its test
count MAE is `30.6272` and end F1 is `0.05534`. These are benchmark values with
split, source, commit, config, checkpoint, and artifact provenance in the JSON
reports. They are not clinical, safety, target-population, or mobile claims.

Reproduce the audit without changing `artifacts/corrected-v1/`:

```bash
python3 -m training.preflight \
  --config training/configs/experiments/r0_baseline.yaml \
  --project-root /Users/devk/AdaptFit --mode full

python3 -m training.evaluate \
  --config training/configs/experiments/r0_baseline.yaml \
  --checkpoint artifacts/corrected-v1/checkpoints/tcn_best.pt \
  --project-root /Users/devk/AdaptFit --device cpu

python3 -m training.evaluate \
  --config training/configs/experiments/r0_baseline.yaml \
  --checkpoint artifacts/corrected-v1/checkpoints/gru_baseline.pt \
  --project-root /Users/devk/AdaptFit --device cpu
```

## Decoder calibration result

Calibration uses the validation split only. It searches the versioned
threshold neighborhood and writes
`artifacts/r0-baseline/decoder/decoder_calibration.json`. It never reads the
test split and cannot create repetition labels that the model does not predict.

```bash
python3 -m training.calibrate_decoder \
  --config training/configs/experiments/r0_baseline.yaml \
  --checkpoint artifacts/corrected-v1/checkpoints/tcn_best.pt \
  --decoder-config training/configs/decoder_v1.yaml \
  --output artifacts/r0-baseline/decoder/decoder_calibration.json \
  --project-root /Users/devk/AdaptFit --device cpu
```

The selected baseline operating point is `start_threshold=0.55`,
`end_threshold=0.55`, `tracking_floor=0.50`, with apex confirmation enabled.
On 896 validation sequences it predicts zero accepted repetitions, against 327
target boundary pairs: count MAE `0.3649554` passes the provisional count
screen, but event F1 and end F1 are `0.0`, with 327 missed events. The declared
gate is therefore `blocked`:

| Requirement | Result | Interpretation |
|---|---:|---|
| Count MAE ≤ 0.40 | pass | insufficient by itself because abstention can make count error look small |
| End F1 ≥ 0.60 | fail (`0.0`) | baseline end evidence is inadequate |
| False events on empty target sequences = 0 | pass (`0`) | the selected operating point is conservative |

The calibration artifact is evidence of a decoder/model limitation, not a
release calibration. Do not loosen thresholds or disable apex confirmation and
then call the result a pass without a separately named experiment and the same
validation-only provenance.

The regenerated local calibration report is currently hashed
`sha256:c776e159c941d0bb16e45c3debf9189743a4899cc1e5f17c07e6bf86ab80375d`.
Because the report is ignored by Git, the command above is the source of truth
when the working tree or commit changes.

The unavailable-boundary correction has now been implemented and covered by a
UL-RED regression test. The isolated prepared root is
`data/processed-r1-boundary-mask-correction/`; its bounded heads-only follow-up
completed epoch 68 with best epoch 53 at
`artifacts/r1-tcn-boundary-mask-correction/checkpoints/tcn_best.pt`. Validation
calibration is complete but blocked: end F1 is `0.0`, count MAE is `0.2678571`,
and all `240` target events were missed. The test split was not read.

The validation-only failure analysis is recorded in
`artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_failure_analysis.md`
and its JSON companion. It found zero qualified apex intervals across 240
target boundary pairs, 178 of 219 explicit segmentation starts below the
tracking floor, and a nearby end-probability crossing for only 88 of 240 target
ends. The evidence supports one event-focused phase/apex and boundary
intervention, with decoder thresholds held fixed for the first comparison.

## Corrected boundary-mask follow-up result

The isolated run used Apple MPS, kept the backbone frozen, trained only `1,746`
head parameters, and resumed exactly from the epoch-10 checkpoint. The best
validation sequence metrics at epoch 53 were family macro-F1 `0.831182`, phase
macro-F1 `0.530337`, boundary F1 `0.614100`, repetition-start F1 `0.996109`,
repetition-end F1 `0.232092`, and count MAE `1.1875`. These are training-time
validation metrics, not test results.

The validation-only `decoder.v1` report is
`artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_calibration.json`.
It covers 896 sequences and 240 target boundary pairs; the selected decoder
accepted zero events, produced 499 abstention events, and missed all 240 target
events. Its checkpoint hash is
`sha256:317bbbf782f5c69f2888737e8bdcec914c8d812e71001c4f8c718fe861479505`,
and its report hash is
`sha256:855b325d27b3ec7fd1e7124a6c57daaafd2ee1413de2b4323297756f25ec937d`.
The gate is blocked and `test_split_used` is false.

## Phase-weight correction training result

The failure analysis showed that `_class_weights` treated an absent phase class
as one sample before normalization. Because `unknown` is absent while `hold`
is rare but valid, that behavior suppressed the hold-class weight. The runner
now assigns zero weight to absent classes and normalizes only among present
classes; a regression test covers this contract.

The isolated training command used
`training/configs/experiments/r1_tcn_phase_weight_correction.yaml`, initialized
from the corrected-mask best checkpoint, and reused
`data/processed-r1-boundary-mask-correction/` without changing it.

| Item | Result |
|---|---|
| Artifact root | `artifacts/r1-tcn-phase-weight-correction/` |
| Parent checkpoint | `artifacts/r1-tcn-boundary-mask-correction/checkpoints/tcn_best.pt` |
| Device/training | Apple MPS; frozen backbone; 1,746 trainable head parameters |
| Epochs | 24 completed; early stopping at epoch 24; best epoch 9 |
| Best validation sequence score | `0.6679357` |
| Best validation sequence metrics | family macro-F1 `0.827578`; phase macro-F1 `0.588146`; boundary F1 `0.614690`; repetition-start F1 `0.996094`; repetition-end F1 `0.233286`; count MAE `1.155134` |
| Training/validation compute | `4,168.44` / `110.80` seconds |
| Best checkpoint | `tcn_best.pt`; SHA-256 `2bac2156ae7ef85b39e65ce5e1d8eff9fe4ea5799ce6b28c2f5ffcf9d30ebf5a` |
| Run report | `metrics.json`; SHA-256 `72dfbfe56f9347306774969c05727ca46991d18afc3b0e8640bb34f42cc12f20` |
| Decoder calibration | Report SHA-256 `a4df7614d2f2932ba4093352d20c19e83a355ac1cc9e4b4be98e213cfee18b40`; gate **BLOCKED** |
| Selected decoder result | 896 validation sequences; predicted `20`, matched `4`, missed `236`; end F1 `0.153846`; count MAE `0.279018`; empty-target false events `15` |
| Test evaluation | **Not performed**; `test_split_used=false` |

This is validation-only decoder evidence, not a product result. The candidate
fails the declared event gate; do not evaluate the locked test split.

## Validation-only failure analysis and selected intervention

The phase-weight checkpoint was replayed with the selected decoder thresholds
unchanged. The reports are
`artifacts/r1-tcn-phase-weight-correction/decoder/decoder_failure_analysis.json`
and its Markdown companion. Their SHA-256 values are
`9d940f237bb630fb28d1ac1af4fdc4db4d2e01d91c76d47d0c64f7f0ac700076` and
`0c73586023e9a2b5b568d2a664b3d54d82e53a1d84a24b87eee95e74ebeeb29e`.

The failure pattern is specific enough for one controlled representation
intervention:

- only `22/240` target intervals satisfy the two-frame apex rule, and only
  `31/240` contain any predicted hold;
- `154/219` explicit REHAB24-6 segmentation starts are below the `0.50`
  tracking floor;
- only `83/240` target ends have an end-probability crossing within ±250 ms,
  while procedural templates qualify `21/21` intervals and REHAB24-6 qualifies
  only `1/219`;
- the corrected mask path leaves `219` segmentation plus `21` procedural
  target pairs and `0` unknown-boundary target pairs, so unavailable-boundary
  leakage is not the remaining failure.

The last-block intervention was completed from the phase-weight best
checkpoint in `artifacts/r1-tcn-last-block-event-support/`. It unfreezes only
`blocks.4.*` plus the existing output heads (`57,426` trainable parameters),
selected epoch 8 after early stopping at epoch 23, and reached validation
sequence score `0.6791891`. Its validation-only decoder gate is blocked (end
F1 `0.094862`; count MAE `0.271205`; 8 empty-target false events); no test
evaluation was performed.

The final-two-block intervention was then run from that best last-block
checkpoint. It is defined in
`training/configs/experiments/r1_tcn_last_two_blocks_event_support.yaml` with
config SHA-256
`10a4df87a984f937202ab26d1ca247afca1245b5e57ba47141428cf1ea8d467c` and writes
to `artifacts/r1-tcn-last-two-blocks-event-support/`. It unfreezes only
`blocks.3.*`, `blocks.4.*`, and the existing heads (`113,106` trainable
parameters). The corrected-mask prepared root, seed, feature/window contract,
label masks, loss weights, optimizer rates, decoder version/thresholds, and
`evaluate_test_after_training=false` remained fixed. Training selected epoch 42
of 57 with validation sequence score `0.7040537`; validation-only calibration
was blocked at end F1 `0.299674`, count MAE `0.333705`, and `63` empty-target
false events. Failure analysis confirms that decoder-only threshold changes
cannot reach the end-F1 gate because `149/240` target ends have no nearby raw
end signal. The locked test split remains unused.

The short full-TCN fine-tuning pilot was then run from the final-two-block best
checkpoint. It is defined in
`training/configs/experiments/r1_tcn_full_finetune_event_support_pilot.yaml`
with config SHA-256
`4023e3b277ad874f1c9ad2492dff587e6d4e2e1b0f2f16019f96444bc22471cd` and writes
to `artifacts/r1-tcn-full-finetune-event-support-pilot/`. All `307,410` model
parameters were trainable, the backbone learning rate was `1e-5`, the head rate
was `3e-4`, and the pilot was limited to 8 epochs with patience 3. The corrected
prepared root, seed 42, label masks, loss weights, decoder settings, and
`evaluate_test_after_training=false` remained fixed. Preflight and runtime
checks passed. Best epoch 6 reached validation sequence score `0.7176679`; the
validation-only decoder gate then remained blocked (end F1 `0.296053`; 60
empty-target false events). Failure analysis is recorded in the pilot artifact.

This concluded the prior model-only supervised escalation for the current
labels. Stop training and request reviewed REHAB24-6 boundary labels and/or additional
representative seated unilateral recordings with explicit start, hold/apex, and
end supervision. Do not tune against test or invent labels.

## UCO follow-up result and next gate

The UCO boundary-support run, partial-pose tracking-target correction, and one
boundary-positive-weight intervention are complete in isolated roots. The
tracking correction is the accepted data-contract fix: partial sources now
declare the canonical joints they structurally represent, and their tracking
targets are no longer diluted by unlisted joints. It does not create phase,
hold, or apex supervision.

| Artifact | Validation-only result | Decision |
|---|---|---|
| `artifacts/r1-tcn-uco-tracking-target-correction/` | Best epoch 2; sequence score `0.6176060`; end F1 `0.070130`; count MAE `0.584`; 7 empty-target false events | Gate blocked; tracking correction retained |
| `artifacts/r1-tcn-uco-boundary-positive-weight/` | Cap `64.0`; best epoch 1; sequence score `0.5252875`; end F1 `0.142322`; count MAE `0.5864`; 24 empty-target false events | Gate blocked; cap rejected |

Failure analysis shows that UCO contributes `507` strong target pairs but no
per-frame phase/hold labels; only `1/507` has a nearby end signal and `0/507`
have nearby starts or apex-qualified intervals in the accepted tracking-target
run. The next dependency is reviewed UCO and/or REHAB24-6 start, hold/apex, and
end supervision, or additional representative labeled seated unilateral
recordings. Do not lower the tracking floor, fabricate labels, increase the
positive-weight cap again, or evaluate the locked test split.

After reviewed supervision is available, create a new prepared root and
artifact root, rerun preflight and runtime checks, train/calibrate on
validation only, and regenerate failure analysis. Only a passing validation
gate unlocks second-seed confirmation and exactly one locked test evaluation.

## UL-RED markerless boundary follow-up

A raw-source audit then found explicit UL-RED `marker-less/3Rep_Sxx.csv` spans
for 219 of 219 available R3 recordings. The adapter correction maps the
zero-based CSV indices to the one-based AMC frame IDs and adds a regression
test. It does not infer phase, hold, or apex labels. The isolated bounded run
used `data/processed-r1-ulred-boundary-support/` and
`artifacts/r1-tcn-ulred-boundary-support/`; unchanged prepared arrays were
hard-linked into the new root because the existing 15 GB root could not be
duplicated on the available volume, while boundary arrays and metadata were
regenerated and the parent root remained unchanged.

| Artifact | Validation-only result | Decision |
|---|---|---|
| `artifacts/r1-tcn-ulred-boundary-support/` | Best epoch 2 of 5; sequence score `0.6013545`; target pairs `872`; end F1 `0.070718`; count MAE `0.684`; 8 empty-target false events | Gate blocked; local boundary correction retained, but reviewed hold/apex/end supervision required |

The decoder selected start `0.65`, end `0.55`, phase `0.45`, tracking floor
`0.50`, with apex confirmation enabled. The validation failure analysis finds
`636` strong UL-RED target pairs but zero target hold labels and only `1/636`
apex-qualified intervals. Keep the new root immutable, do not remove apex
confirmation or lower the tracking floor, and do not evaluate the locked test.
The next step is the reviewed-event-supervision intake documented in
`reviewed-event-supervision-request.md`; only after the new labels are audited
and prepared does another isolated training/calibration run make sense.

## R1 smoke result

The required two-epoch CPU smoke command completed after strict preflight. It
trained only the 1,746 output-head parameters from the corrected-v1 parent and
wrote `artifacts/r1-tcn-boundary/`. Epoch 1 was selected on validation with
sequence score `0.6484946`; epoch 2 scored `0.6476718`. The best checkpoint had
sequence boundary F1 `0.5561438`, repetition-end F1 `0.1180846`, and count MAE
`0.1071429`. Losses were finite. The run took `848.13` training seconds plus
`23.98` validation seconds on CPU.

The checkpoints contain model/optimizer state, Python/NumPy/Torch RNG state,
sampler epoch, effective `config`, trainable layers, parent checkpoint, and
source commit. `test_evaluation_performed` is false and the metrics report has
no test evaluation. Full checkpoint and report hashes are recorded in
[training-execution-log.md](training-execution-log.md).

## R1 full-run result

The full heads-only R1 run completed on Apple MPS with 27 epochs before early
stopping; epoch 12 was selected on the validation sequence score `0.6603763`.
The selected checkpoint has sequence family macro-F1 `0.902795`, phase
macro-F1 `0.569988`, boundary F1 `0.548749`, repetition-end F1 `0.109059`, and
count MAE `0.155134`. These are validation-only training metrics. Test
evaluation remains locked, quality-head coverage remains zero, and the
checkpoint is not a release candidate.

## R1 decoder calibration result

The R1 epoch-12 checkpoint was calibrated on validation only using
`decoder.v1`. The report is
`artifacts/r1-tcn-boundary/decoder/decoder_calibration.json` and records source
commit `b3926d864c1d976504c2d7131db8d4992cf2ef45`, checkpoint hash
`sha256:e74679f9ebe0b012d4ed8729120aa8e3489c960ea7941a24d7e94f9e8ff45cbf`,
896 validation sequences, and 327 target boundary pairs. The selected
thresholds are start `0.55`, end `0.55`, tracking floor `0.50`, with apex
confirmation enabled. It predicts zero positive events, has end F1 `0.0`,
count MAE `0.3649554`, and zero false events on empty sequences. The gate is
blocked and `test_split_used` is false.

Diagnostics found 266 `low_tracking` abstentions, 55
`partial_without_apex` events, and no positive events. The validation labels
include 87 UL-RED sequences with `boundary_label_source=unknown`; the adapter
currently supplies first/last-frame labels when `rep_boundary` is omitted.
This conflicts with the documented unavailable-boundary policy and is the
selected label/mask correction before another bounded experiment.

## R1 training handoff

R1 is an isolated, validation-selected warm-start experiment. It retains the
283/128/96-channel causal contract, uses the corrected-v1 normalization and
prepared split, and writes only to `artifacts/r1-tcn-boundary/`.

`training/configs/experiments/r1_tcn_boundary_finetune.yaml` specifies:

- parent checkpoint `artifacts/corrected-v1/checkpoints/tcn_best.pt`;
- head-only initialization (`freeze_backbone: true`);
- head learning rate `3e-4`, backbone rate `3e-5` (the latter is inactive until
  a later partial-unfreeze experiment);
- batch size 64, maximum 100 epochs, patience 15, float32, seed 42;
- test evaluation disabled during training;
- latest resumable checkpoint enabled;
- strict source paths with optional quality sources disabled.

The full R1 budget has completed and passed its checkpoint/provenance gate.
Its decoder calibration is complete but blocked; keep
`evaluate_test_after_training: false` until the boundary-label correction,
candidate selection, and decoder are frozen.

The corrected boundary-mask follow-up also completed its bounded budget and
passed checkpoint/provenance checks, but its validation decoder gate is blocked.
The phase-weight correction follow-up has now completed its bounded budget and
passed checkpoint/provenance checks; keep test evaluation disabled until its
validation-only decoder gate passes.

The completed smoke command was:

```bash
python3 -m training.preflight \
  --config training/configs/experiments/r1_tcn_boundary_finetune.yaml \
  --project-root /Users/devk/AdaptFit --mode full

python3 -m training.train \
  --config training/configs/experiments/r1_tcn_boundary_finetune.yaml \
  --project-root /Users/devk/AdaptFit --models tcn --device cpu \
  --max-epochs 2 --skip-test
```

The full R1 command that produced the current artifact was:

```bash
python3 -m training.train \
  --config training/configs/experiments/r1_tcn_boundary_finetune.yaml \
  --project-root /Users/devk/AdaptFit \
  --models tcn --device auto --skip-test
```

The smoke, full-R1, corrected-mask, phase-weight, last-block, final-two-block,
full-TCN, and UL-RED checkpoint/provenance checks are recorded. Their
calibrations and failure analyses are recorded and blocked. The next evidence
action is reviewed boundary/phase supervision or additional representative
labeled recordings, followed by a new prepared root and validation-only
calibration. Evaluate the locked test split only once a candidate and decoder
actually pass the declared gate.

## Stop conditions

Stop before or during R1 if any of the following occurs:

- a source enabled by strict preflight is missing or decodes to zero sequences;
- the prepared index, normalization hash, feature width, window length, split
  identity, or label masks differ from this handoff;
- a checkpoint is written under `artifacts/corrected-v1/` or another existing
  registry root;
- model, optimizer, or gradient values become non-finite;
- a warm-start or resume model/schema/decoder/normalization mismatch is found;
- the configured full-TCN pilot does not train all model parameters, or changes
  the prepared root, label masks, loss weights, decoder settings, or declared
  pilot budget;
- exact resume lacks optimizer, RNG, sampler, or trainable-layer provenance;
- validation worsens for the declared patience or the compute budget expires;
- a quality or target-population claim is inferred from unavailable labels.

## What remains after the training handoff

The original, corrected-mask, phase-weight, last-block, final-two-block, and
full-TCN decoder calibrations are complete but blocked, and decoder calibration
is not the final product gate. The immediate remaining evidence work is
reviewed REHAB24-6 boundary supervision or additional representative labeled
recordings, followed by a new prepared root, validation-only calibration, and
failure analysis. Only after the decoder gate passes may a second seed and one
locked test evaluation run.
Later work is the supervised-only versus teacher-assisted comparison if a
measured gap remains, quality-label acquisition, target-population collection,
native export and golden parity, and recommendation implementation.
