# AdaptFit Training Execution Log

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** training execution ledger, checkpoint provenance, and audit record
> - **Last verified:** 2026-09-11
> - **Source commit:** `b3926d8` (R1 full-run evidence and decoder calibration)
> - **Owner:** AdaptFit training engineering
> - **Supersedes or supports:** satisfies the execution log requirement from [documentation-validation.md](documentation-validation.md) and [efficient-training-strategy.md](efficient-training-strategy.md)
> - **Review trigger:** after every experiment run, checkpoint evaluation, data preparation, or task transition

This document is the operational execution log and run inventory for AdaptFit.
It records verified checkouts, candidate checkpoint status, evaluation findings,
known blockers, and task progression following the Luna execution checklist in
[efficient-training-strategy.md](efficient-training-strategy.md).

The normalized checkpoint inventory is also recorded in the [artifact
registry](artifact-registry.md). Add a manifest entry after every run; do not
use this log to promote a partial checkpoint or overwrite a historical result.

## 2A. Pre-training gate verification (2026-09-08 working tree)

This gate was completed before the R1 smoke run. The detailed handoff is
[pretraining-readiness.md](pretraining-readiness.md). The smoke and full-run
results are recorded below; subsequent corrected-mask execution is recorded in
sections 2E through 2H.

| Gate | Result | Evidence |
|---|---|---|
| Immutable corrected-v1 hashes | **PASS** | TCN `e69ff69d…`, GRU `4105bf23…`; no files under `artifacts/corrected-v1/` changed |
| Strict source/config preflight | **PASS** | `r0_baseline.yaml` and `r1_tcn_boundary_finetune.yaml`; enabled sources decode successfully; optional quality sources are disabled |
| R0 test audit | **PASS (audit only)** | `artifacts/r0-baseline/metrics/*_evaluation.json` and model manifests; test was read, never used for calibration |
| Validation-only decoder calibration | **BLOCKED** | `artifacts/r0-baseline/decoder/decoder_calibration.json`; count MAE `0.3649554`, end F1 `0.0`, false empty-sequence events `0` |
| Warm-start/freeze/resume controls | **PASS (full run exercised)** | `training/src/runner.py`, `training/train.py`, staged config, staged-training tests, and the R1 full-run checkpoint; exact interruption/resume not run |
| Next eligible action | **Validation-only decoder calibration** | Full R1 completed with test evaluation locked; calibrate `decoder.v1` before test evaluation |

The decoder block is a measured baseline limitation. It does not authorize
changing the test split, relaxing label masks, or presenting the decoder as a
release component. Validation-only decoder calibration is now the next
computational action; product release remains blocked until the decoder gate is
rechecked and all other release gates pass.

## 2B. R1 smoke training verification (2026-09-08)

The bounded R1 smoke command completed on CPU for two epochs after strict
preflight passed for the four enabled sources (`rehab24_6`, `intellirehabds`,
`mmfit`, and `ul_red`). It used the corrected-v1 TCN checkpoint as a warm-start,
froze the temporal backbone, trained 1,746 head parameters, and wrote only to
the ignored `artifacts/r1-tcn-boundary/` root. No test evaluation was performed.

| Item | Result |
|---|---|
| Effective source commit | `627283bec6851d95cba988a0235931c158465eab` |
| Config hash | `sha256:0537376cac3337a95277b2b063a6549c13ded9e1b548b2a00266e96827302962` |
| Parent checkpoint | `/Users/devk/AdaptFit/artifacts/corrected-v1/checkpoints/tcn_best.pt` |
| Best epoch | 1 of 2 |
| Epoch 1 | loss `0.199823`; validation sequence score `0.648495`; sequence boundary F1 `0.556144`; sequence repetition-end F1 `0.118085`; count MAE `0.107143` |
| Epoch 2 | loss `0.198885`; validation sequence score `0.647672`; sequence boundary F1 `0.554943`; sequence repetition-end F1 `0.117122`; count MAE `0.131696` |
| Training compute | `848.13` seconds on CPU; validation `23.98` seconds |
| Test evaluation | **Not performed** (`test_evaluation_performed=false`) |
| Best checkpoint | `artifacts/r1-tcn-boundary/checkpoints/tcn_best.pt`; SHA-256 `9c3098b2633292136df99f9c4c52a4623be2c7137bd05bcc661d9412ae6fc772` |
| Latest checkpoint | `artifacts/r1-tcn-boundary/checkpoints/tcn_latest.pt`; SHA-256 `d342b78fbfff55b2d872a70c3f18342ffd19ad4b7175ff4b0d1b8f3b8f5ca017` |
| Run report | `artifacts/r1-tcn-boundary/metrics.json`; SHA-256 `e25849f805c9f2454144b1c160adabb0396e21083e67235c2ba516e089d37d25` |
| Run status | **Smoke complete; full R1 not run** |

Both checkpoints contain model and optimizer state, Python/NumPy/Torch RNG
state, sampler epoch, effective `config`, trainable-layer list, parent
checkpoint, and source commit. The smoke result is a readiness/provenance gate,
not a model-selection result or product claim. The full R1 run is recorded
below; its checkpoint still needs validation-only decoder recalibration before
any locked-test evaluation.

The local artifact root was subsequently reused for the full R1 run. The smoke
hashes above preserve that earlier state; the files currently present under
`artifacts/r1-tcn-boundary/` are the full-run files recorded below.

## 2C. R1 full heads-only training (2026-09-09)

The full R1 training command completed on Apple MPS with the same corrected-v1
parent, frozen temporal backbone, and 1,746 trainable head parameters. The
configured maximum was 100 epochs with patience 15; validation stopped the run
at epoch 27 and selected epoch 12. Test evaluation remained disabled.

| Item | Result |
|---|---|
| Effective source commit | `d7848e2b12322e40e83758abd5e12ae6c7727e80` |
| Config hash | `sha256:73dadb643781fd39630729d8a5e347ea98384b9455d1894589d3061c8d031c04` |
| Device | Apple MPS; float32 |
| Epochs | 27 completed; early stopping at epoch 27; best epoch 12 |
| Best validation sequence score | `0.6603763` |
| Best validation sequence metrics | family macro-F1 `0.902795`; phase macro-F1 `0.569988`; boundary F1 `0.548749`; repetition-start F1 `0.988439`; repetition-end F1 `0.109059`; count MAE `0.155134` |
| Quality-label coverage | `0.0%` for the four dimension-specific quality heads |
| Training/validation compute | `4,455.76` / `111.25` seconds |
| Test evaluation | **Not performed** (`test_evaluation_performed=false`) |
| Best checkpoint | `artifacts/r1-tcn-boundary/checkpoints/tcn_best.pt`; SHA-256 `e74679f9ebe0b012d4ed8729120aa8e3489c960ea7941a24d7e94f9e8ff45cbf` |
| Latest checkpoint | `artifacts/r1-tcn-boundary/checkpoints/tcn_latest.pt`; SHA-256 `d0c4a369d0229db4d1932ed763abaf688b29e1ebf1197124fba02624f9f57810` |
| Current run report | `artifacts/r1-tcn-boundary/metrics.json`; SHA-256 `7d340a204b924d1d46bffa68f2833a83cdcaf2df3be764b985856c77cccd3c4d` |
| Run status | **Full training complete; validation-only; not a release candidate** |

The full run is a training result, not a decoder or product result. The best
checkpoint must go through validation-only decoder recalibration before any
locked-test evaluation. The validation boundary/end metrics do not by
themselves establish an improvement over corrected-v1 because the comparison
must use the same split, decoder, and artifact protocol.

## 2D. R1 validation-only decoder calibration (2026-09-09)

Calibration ran against the R1 epoch-12 checkpoint on the validation split
only. It used the versioned `decoder.v1` neighborhood and wrote an ignored
report without reading the locked test split.

| Item | Result |
|---|---|
| Source commit | `b3926d864c1d976504c2d7131db8d4992cf2ef45` |
| Config hash | `sha256:73dadb643781fd39630729d8a5e347ea98384b9455d1894589d3061c8d031c04` |
| Checkpoint | `artifacts/r1-tcn-boundary/checkpoints/tcn_best.pt`; SHA-256 `e74679f9ebe0b012d4ed8729120aa8e3489c960ea7941a24d7e94f9e8ff45cbf` |
| Decoder report | `artifacts/r1-tcn-boundary/decoder/decoder_calibration.json` |
| Split | Validation only; 896 sequences, 327 target boundary pairs |
| Selected decoder | start `0.55`, end `0.55`, tracking floor `0.50`, apex confirmation enabled |
| Result | Predicted count `0`; missed events `327`; end F1 `0.0`; count MAE `0.3649554`; empty-target false events `0` |
| Gate | **BLOCKED**: count MAE and empty-target safety pass; end F1 `0.0` fails the `0.60` requirement |
| Test split | **Not used** (`test_split_used=false`) |

The 18 searched configurations all remained conservative and produced no
positive decoded repetition. Representative event reasons were 266
`low_tracking` abstentions, 55 `partial_without_apex` events, and one
`partial_timeout` event. Raw boundary logits can be high, but the decoder's
tracking and apex requirements are not met reliably.

The label audit found 219 target pairs from REHAB24-6 segmentation, 87 from
UL-RED records with `boundary_label_source=unknown`, and 21 from procedural
templates. `training/src/data/adapters.py::_make_sequence` currently turns a
missing `rep_boundary` into first/last-frame labels, which contradicts the
UL-RED unknown-boundary declaration. The next targeted follow-up is therefore
to correct that unavailable-label default, add a regression test, regenerate
an isolated prepared split, and run one bounded boundary experiment. The
corrected-v1 data and checkpoint remain immutable.

## 2E. Boundary-mask correction and bounded follow-up (2026-09-09–2026-09-10)

The adapter correction is implemented: `_make_sequence` now passes omitted
boundary labels through to `CanonicalSequence`, whose contract masks them as
`-1`; REHAB24-6 explicitly supplies first/last-frame labels for its segmented
clips. The UL-RED adapter therefore keeps its `boundary_label_source=unknown`
records masked. Regression coverage asserts both behaviors.

| Item | Result |
|---|---|
| Config | `training/configs/experiments/r1_tcn_boundary_mask_correction.yaml` |
| Prepared root | `data/processed-r1-boundary-mask-correction/` |
| Artifact root | `artifacts/r1-tcn-boundary-mask-correction/` |
| Prepared data | 5,004 sequences; 160,813 train / 5,204 validation / 7,592 test windows; zero participant overlap |
| Boundary coverage | 1,372 sequences; UL-RED unavailable boundaries are masked |
| Parent checkpoint | `artifacts/corrected-v1/checkpoints/tcn_best.pt` |
| Training | Apple MPS, frozen backbone, 1,746 trainable head parameters; completed epoch 68; best epoch 53 |
| Checkpoint state | `tcn_latest.pt` and `tcn_best.pt` contain optimizer, Python/NumPy/Torch RNG, and sampler state |
| Best checkpoint | `tcn_best.pt` (epoch 53); SHA-256 `317bbbf782f5c69f2888737e8bdcec914c8d812e71001c4f8c718fe861479505` |
| Training resume | Exact resume from epoch 10 completed after normalizing accelerator-loaded Torch RNG state to a CPU byte tensor |
| Test evaluation | **Not performed**; `test_evaluation_performed=false` |

The process was initially interrupted after epoch 10 had been checkpointed. The
corrected-v1 prepared data, checkpoint, and test split were not modified or
read by this follow-up. The resumed run completed without reading the test
split.

## 2F. Corrected boundary-mask validation calibration (2026-09-10)

The best epoch-53 checkpoint was calibrated with `decoder.v1` on the validation
split only. The report is
`artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_calibration.json`.

| Item | Result |
|---|---|
| Source commit | `b3926d864c1d976504c2d7131db8d4992cf2ef45` |
| Config hash | `sha256:d421d3c2e081b9ee0f99e6a55b1ff7628dea360026fb2e7eae7de5209988ab28` |
| Checkpoint hash | `sha256:317bbbf782f5c69f2888737e8bdcec914c8d812e71001c4f8c718fe861479505` |
| Report hash | `sha256:855b325d27b3ec7fd1e7124a6c57daaafd2ee1413de2b4323297756f25ec937d` |
| Calibration scope | 896 validation sequences; 240 target boundary pairs; test split unused |
| Selected decoder | `start=0.55`, `end=0.55`, `phase=0.45`, `tracking_floor=0.50`; apex confirmation enabled |
| Selected metrics | predicted count `0`; missed events `240`; event F1 `0.0`; start F1 `0.0`; end F1 `0.0`; count MAE `0.2678571`; abstentions `499`; false events on empty sequences `0` |
| Gate | **BLOCKED**: count MAE and empty-sequence false-event checks pass; end F1 `0.0` fails the `0.60` threshold |

The corrected unavailable-boundary mask path is therefore verified, but the
model/decoder still fails the event gate. Do not evaluate the locked test split
or promote this checkpoint. The next action is representative missed-event
review followed by one targeted event-boundary intervention in a new artifact
root.

## 2G. Corrected-mask decoder failure analysis (2026-09-10)

The validation-only analysis replays the selected calibration configuration and
records source-level abstention reasons, target-boundary signal statistics,
phase/apex support, and representative cases. It was run with:

```bash
python3 -m training.analyze_decoder_failures \
  --config training/configs/experiments/r1_tcn_boundary_mask_correction.yaml \
  --checkpoint artifacts/r1-tcn-boundary-mask-correction/checkpoints/tcn_best.pt \
  --calibration-report artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_calibration.json \
  --output artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_failure_analysis.json \
  --markdown-output artifacts/r1-tcn-boundary-mask-correction/decoder/decoder_failure_analysis.md \
  --project-root /Users/devk/AdaptFit --device mps
```

| Finding | Evidence |
|---|---|
| Phase/apex support | Target intervals contain 342 hold-label frames; the model predicts 1 hold frame; 0/240 intervals satisfy the two-frame apex requirement |
| Start tracking | 178/219 explicit segmentation starts are below the `0.50` tracking floor |
| End alignment | 88/240 target ends have an end-probability crossing within ±250 ms; 84/240 also satisfy the tracking floor |
| Decoder reasons | 226 `low_tracking`, 262 `partial_without_apex`, and 11 `partial_too_short` abstentions; zero positive events |
| Label/mask status | 219 segmentation and 21 procedural target pairs; no unknown-boundary target pairs |
| Decision | **Blocked**; use an isolated event-focused phase/apex and boundary intervention, holding decoder thresholds fixed initially |

The JSON report SHA-256 is
`868cb2327554cf0e2e124f937b838f81a343c223ef54460298e5ffc158766199`; the
Markdown report SHA-256 is
`ab3b738bf3d03aefe5501bbac4676c40a7b450484e4bf8952e34b16565fbdf4a`.
The report is validation-only and does not unlock test evaluation.

## 2H. Phase-weight correction training (2026-09-10)

The failure analysis identified an absent-class normalization bug in
`training/src/runner.py`: an absent phase class was treated as one sample when
normalizing weights. The correction assigns zero weight to absent classes and
normalizes only across present classes; `training/tests/test_models.py` adds a
regression test. The run reused the verified corrected-mask prepared root and
initialized from its best checkpoint, so this is a separate loss-weighting
lineage rather than a data regeneration.

The command was:

```bash
python3 -m training.train \
  --config training/configs/experiments/r1_tcn_phase_weight_correction.yaml \
  --project-root /Users/devk/AdaptFit \
  --models tcn --device mps --seed 42 --skip-test
```

| Item | Result |
|---|---|
| Effective source commit | `b3926d864c1d976504c2d7131db8d4992cf2ef45` |
| Config path hash | `sha256:28b762164e9340bd1b37de1c042cce5f8e95d0de1c75662162a44525cebd1694` |
| Effective config hash | `sha256:3d0f8e312db4d58a79c45fa25f7c8e13a59f91b31a0fcc8ce18352f24c5ae7f5` |
| Parent checkpoint | `artifacts/r1-tcn-boundary-mask-correction/checkpoints/tcn_best.pt` |
| Prepared root | `data/processed-r1-boundary-mask-correction/` (reused unchanged) |
| Artifact root | `artifacts/r1-tcn-phase-weight-correction/` |
| Device/training | Apple MPS; frozen backbone; 1,746 trainable head parameters |
| Epochs | 24 completed; early stopping at epoch 24; best epoch 9 |
| Best validation sequence score | `0.6679357` |
| Best validation sequence metrics | family macro-F1 `0.827578`; phase macro-F1 `0.588146`; boundary F1 `0.614690`; repetition-start F1 `0.996094`; repetition-end F1 `0.233286`; count MAE `1.155134` |
| Training/validation compute | `4,168.44` / `110.80` seconds |
| Best checkpoint | `tcn_best.pt`; SHA-256 `2bac2156ae7ef85b39e65ce5e1d8eff9fe4ea5799ce6b28c2f5ffcf9d30ebf5a` |
| Latest checkpoint | `tcn_latest.pt`; SHA-256 `eb0c14ad8c4320d4bed460168c0e2ca1d92deba891663669a7015ba5e654dc79` |
| Run report | `metrics.json`; SHA-256 `72dfbfe56f9347306774969c05727ca46991d18afc3b0e8640bb34f42cc12f20` |
| Test evaluation | **Not performed** (`test_evaluation_performed=false`) |
| Decoder calibration | **Blocked**; report records end F1 `0.153846` and 15 false events on empty sequences |
| Run status | **Training complete; validation-only decoder gate blocked; not a release candidate** |

The best epoch-9 checkpoint improved the validation sequence score over the
corrected-mask best (`0.6483843`) by `0.0195514`. Its decoder calibration is
recorded below and is blocked; no test decision is permitted.

## 2I. Phase-weight correction validation-only decoder calibration (2026-09-10)

The phase-weight correction best checkpoint was calibrated with `decoder.v1` on
the validation split only. The command was:

```bash
python3 -m training.calibrate_decoder \
  --config training/configs/experiments/r1_tcn_phase_weight_correction.yaml \
  --checkpoint artifacts/r1-tcn-phase-weight-correction/checkpoints/tcn_best.pt \
  --decoder-config training/configs/decoder_v1.yaml \
  --output artifacts/r1-tcn-phase-weight-correction/decoder/decoder_calibration.json \
  --project-root /Users/devk/AdaptFit --device mps
```

| Item | Result |
|---|---|
| Checkpoint | `artifacts/r1-tcn-phase-weight-correction/checkpoints/tcn_best.pt`; SHA-256 `2bac2156ae7ef85b39e65ce5e1d8eff9fe4ea5799ce6b28c2f5ffcf9d30ebf5a` |
| Calibration report | `artifacts/r1-tcn-phase-weight-correction/decoder/decoder_calibration.json`; SHA-256 `a4df7614d2f2932ba4093352d20c19e83a355ac1cc9e4b4be98e213cfee18b40` |
| Scope | 896 validation sequences; 240 target boundary pairs; test split unused |
| Selected decoder | start `0.65`, end `0.65`, phase `0.45`, tracking floor `0.50`; apex confirmation enabled |
| Result | Predicted `20`; matched `4`; missed `236`; event F1 `0.030769`; start F1 `0.115385`; end F1 `0.153846`; count MAE `0.279018` |
| Empty-target false events | `15` |
| Gate | **BLOCKED**: count MAE passes; end F1 and empty-target false-event requirements fail |
| Test split | **Not used** (`test_split_used=false`) |

The phase-weight correction improved training-time validation, but it did not
meet the decoder event gate. The next action is validation-only failure
analysis followed by one narrow intervention in a new artifact root; do not
evaluate the locked test split or begin broad training.

## 2J. Phase-weight validation-only failure analysis and last-block intervention definition (2026-09-10)

The phase-weight calibration failure was replayed on validation only with the
selected thresholds unchanged. The command was:

```bash
python3 -m training.analyze_decoder_failures \
  --config training/configs/experiments/r1_tcn_phase_weight_correction.yaml \
  --checkpoint artifacts/r1-tcn-phase-weight-correction/checkpoints/tcn_best.pt \
  --calibration-report artifacts/r1-tcn-phase-weight-correction/decoder/decoder_calibration.json \
  --output artifacts/r1-tcn-phase-weight-correction/decoder/decoder_failure_analysis.json \
  --markdown-output artifacts/r1-tcn-phase-weight-correction/decoder/decoder_failure_analysis.md \
  --project-root /Users/devk/AdaptFit --device mps
```

| Finding | Evidence |
|---|---|
| Qualified phase/apex support | Only `22/240` target intervals satisfy the two-frame apex rule; `31/240` contain any predicted hold. The aggregate predicted hold-frame count (`659`) is not usable apex evidence inside most target intervals. |
| Start tracking | `154/219` explicit REHAB24-6 segmentation starts are below the `0.50` tracking floor. |
| End alignment | `83/240` target ends have an end-probability crossing within ±250 ms; `81/240` also satisfy the tracking floor. |
| Source concentration | Procedural templates qualify all `21/21` target intervals for apex support, while REHAB24-6 qualifies only `1/219`; the remaining failures are therefore concentrated in the real segmentation source rather than unknown-boundary masking. |
| Label/mask status | The corrected path leaves `219` segmentation and `21` procedural target pairs, with `0` unknown-boundary target pairs. |

The JSON report SHA-256 is
`9d940f237bb630fb28d1ac1af4fdc4db4d2e01d91c76d47d0c64f7f0ac700076`; the
Markdown report SHA-256 is
`0c73586023e9a2b5b568d2a664b3d54d82e53a1d84a24b87eee95e74ebeeb29e`.
These reports are validation-only and do not unlock test evaluation.

The selected intervention was a representation-only, partial last-block
fine-tune:

| Item | Definition |
|---|---|
| Config | `training/configs/experiments/r1_tcn_last_block_event_support.yaml`; SHA-256 `708a2b835958c6369a18333dfd00854fccc5fef2106d5324d5475449c46a72e1` |
| Artifact root | `artifacts/r1-tcn-last-block-event-support/` |
| Parent checkpoint | `artifacts/r1-tcn-phase-weight-correction/checkpoints/tcn_best.pt`; SHA-256 `2bac2156ae7ef85b39e65ce5e1d8eff9fe4ea5799ce6b28c2f5ffcf9d30ebf5a` |
| Trainable surface | Existing `heads.*` plus exactly `blocks.4.*`; `57,426` parameters across 16 tensors (10 head, 6 final-block) |
| Held constant | `data/processed-r1-boundary-mask-correction/`, seed `42`, 283-feature/128-frame contract, label masks, loss weights, optimizer rates, `decoder.v1`, selected decoder thresholds, and `evaluate_test_after_training=false` |
| Preflight | **PASS**; all four enabled sources decoded successfully; no test data was read |
| Status | **Training complete; validation decoder gate blocked; test locked** |

This intervention was selected because the phase-weight checkpoint already
showed strong local end logits in many REHAB24-6 misses, while the frozen
representation supplied insufficient qualified phase/apex support and tracking
alignment. It changed one model degree of freedom and left labels and decoder
state untouched.

## 2K. Last-block event-support training (2026-09-10)

The configured last-block intervention was run with the prepared corrected-mask
root unchanged and test evaluation disabled:

```bash
python3 -m training.train \
  --config training/configs/experiments/r1_tcn_last_block_event_support.yaml \
  --project-root /Users/devk/AdaptFit \
  --models tcn --device mps --seed 42 --skip-test
```

| Item | Result |
|---|---|
| Effective source commit | `b3926d864c1d976504c2d7131db8d4992cf2ef45` |
| Config path hash | `sha256:708a2b835958c6369a18333dfd00854fccc5fef2106d5324d5475449c46a72e1` |
| Effective config hash | `sha256:32cf4a96dff744a8c99b2dbcffd912e7783500e49a9d343a2bcb5b0fe1cbc362` |
| Parent checkpoint | `artifacts/r1-tcn-phase-weight-correction/checkpoints/tcn_best.pt`; SHA-256 `2bac2156ae7ef85b39e65ce5e1d8eff9fe4ea5799ce6b28c2f5ffcf9d30ebf5a` |
| Prepared root | `data/processed-r1-boundary-mask-correction/` (reused unchanged) |
| Device/training | Apple MPS; 57,426 trainable parameters; `blocks.4.*` plus heads |
| Epochs | 23 completed; early stopping at epoch 23; best epoch 8 |
| Best validation sequence score | `0.6791891` |
| Best validation sequence metrics | family macro-F1 `0.866847`; phase macro-F1 `0.591009`; boundary F1 `0.610987`; repetition-start F1 `0.996124`; repetition-end F1 `0.225850`; count MAE `1.414063` |
| Training/validation compute | `4,384.41` / `113.89` seconds |
| Best checkpoint | `tcn_best.pt`; SHA-256 `e1ac1ea978b323e128d9e966b067f8c5af073358e86846e8e7c046ec11a156de` |
| Latest checkpoint | `tcn_latest.pt`; SHA-256 `39fe56ed6486510dd3e6dae0a3169760c96c927942ad20e74c9611a1a794abef` |
| Run history | `metrics/tcn_history.json`; SHA-256 `287db7b74c65751d88da15951a12143eab6ebbfbd917544846603c6bfd91e581` |
| Test evaluation | **Not performed** (`test_evaluation_performed=false`) |
| Run status | **Training complete; validation-only decoder gate pending** |

The representation intervention improved the training-time validation sequence
score over the phase-weight parent (`0.6679357` to `0.6791891`) but did not by
itself establish event reliability. The best checkpoint remained the only
candidate eligible for validation-only decoder calibration.

## 2L. Last-block validation-only decoder calibration (2026-09-10)

The epoch-8 checkpoint was calibrated with `decoder.v1` on validation only,
with apex confirmation retained. The report is
`artifacts/r1-tcn-last-block-event-support/decoder/decoder_calibration.json`.

| Item | Result |
|---|---|
| Checkpoint | `artifacts/r1-tcn-last-block-event-support/checkpoints/tcn_best.pt`; SHA-256 `e1ac1ea978b323e128d9e966b067f8c5af073358e86846e8e7c046ec11a156de` |
| Calibration report | `artifacts/r1-tcn-last-block-event-support/decoder/decoder_calibration.json`; SHA-256 `d495be8ae07d80c7b0b215922b15998d6a44311cd08218148bf10ba2b3ebfaf1` |
| Scope | 896 validation sequences; 240 target boundary pairs; test split unused |
| Selected decoder | start `0.55`, end `0.55`, phase `0.45`, tracking floor `0.60`; apex confirmation enabled |
| Result | Predicted `13`; matched `3`; missed `237`; event F1 `0.023715`; start F1 `0.086957`; end F1 `0.094862`; count MAE `0.271205` |
| Empty-target false events | `8` |
| Gate | **BLOCKED**: count MAE passes; repetition-end F1 and empty-target false-event requirements fail |
| Test split | **Not used** (`test_split_used=false`) |

The candidate retained good count MAE by abstaining, but it did not provide the
event evidence required by the gate. It is not promoted and has no test
metrics.

## 2M. Last-block validation-only failure analysis and next intervention (2026-09-10)

The failed calibration was replayed and analyzed without reading test:

```bash
python3 -m training.analyze_decoder_failures \
  --config training/configs/experiments/r1_tcn_last_block_event_support.yaml \
  --checkpoint artifacts/r1-tcn-last-block-event-support/checkpoints/tcn_best.pt \
  --calibration-report artifacts/r1-tcn-last-block-event-support/decoder/decoder_calibration.json \
  --output artifacts/r1-tcn-last-block-event-support/decoder/decoder_failure_analysis.json \
  --markdown-output artifacts/r1-tcn-last-block-event-support/decoder/decoder_failure_analysis.md \
  --project-root /Users/devk/AdaptFit --device mps
```

| Finding | Evidence |
|---|---|
| Qualified phase/apex support | `21/240` target intervals satisfy the two-frame apex rule; `46/240` contain any predicted hold. |
| Start tracking | `219/219` explicit segmentation starts are below the selected `0.60` tracking floor; tracking passes on none of those intervals. |
| End alignment | `92/240` target ends have an end-probability crossing within ±250 ms; `148/240` have no nearby end signal. |
| Source concentration | Procedural templates qualify `21/21` intervals, while segmentation qualifies `0/219`; the remaining failure is concentrated in REHAB24-6. |
| Label/mask status | `219` segmentation plus `21` procedural target pairs; `0` unknown-boundary target pairs. The unavailable-boundary fix is not the remaining cause. |

The JSON report SHA-256 is
`2ef9df4873e4684f94e0ad4a686aefb8ef0a8e6bc98c033eacd0d709a0b41b83`; the
Markdown report SHA-256 is
`5a045fa4e1e68408754d3358e59f35c0baa7af04261500581e2eeb2e1a95775d`.
The reports are validation-only and do not unlock test evaluation.

The last-block intervention improved the sequence score but did not pass the
event gate. Following the escalation ladder, the next isolated experiment is
to unfreeze exactly the final two TCN residual blocks from the best last-block
checkpoint. All labels, data, loss weights, decoder policy, seed, and test lock
remain fixed:

| Item | Definition |
|---|---|
| Config | `training/configs/experiments/r1_tcn_last_two_blocks_event_support.yaml`; SHA-256 `10a4df87a984f937202ab26d1ca247afca1245b5e57ba47141428cf1ea8d467c` |
| Artifact root | `artifacts/r1-tcn-last-two-blocks-event-support/` |
| Parent checkpoint | `artifacts/r1-tcn-last-block-event-support/checkpoints/tcn_best.pt`; SHA-256 `e1ac1ea978b323e128d9e966b067f8c5af073358e86846e8e7c046ec11a156de` |
| Trainable surface | Existing `heads.*` plus exactly `blocks.3.*` and `blocks.4.*`; `113,106` parameters across 22 tensors |
| Held constant | Corrected-mask prepared root, seed `42`, 283-feature/128-frame contract, label masks, loss weights, decoder.v1, selected policy, and `evaluate_test_after_training=false` |
| Preflight/runtime | **PASS**; all enabled sources available, finite forward outputs, no training or test evaluation |
| Status | **Configured; training not yet launched** |

Do not change decoder thresholds, remove apex confirmation, add event sampling,
edit labels, or run a locked-test evaluation for this experiment.

## 2N. Final-two-block event-support training (2026-09-10–2026-09-11)

The final-two-block intervention was run from the best last-block checkpoint
with the corrected-mask prepared root unchanged and test evaluation disabled:

```bash
python3 -m training.train \
  --config training/configs/experiments/r1_tcn_last_two_blocks_event_support.yaml \
  --project-root /Users/devk/AdaptFit \
  --models tcn --device mps --seed 42 --skip-test
```

| Item | Result |
|---|---|
| Effective source commit | `b3926d864c1d976504c2d7131db8d4992cf2ef45` |
| Config path hash | `sha256:10a4df87a984f937202ab26d1ca247afca1245b5e57ba47141428cf1ea8d467c` |
| Effective config hash | `sha256:335628c5285c3fc26de37a337cf088171bee0a59479ae7c846333ec618a2d79e` |
| Parent checkpoint | `artifacts/r1-tcn-last-block-event-support/checkpoints/tcn_best.pt`; SHA-256 `e1ac1ea978b323e128d9e966b067f8c5af073358e86846e8e7c046ec11a156de` |
| Prepared root | `data/processed-r1-boundary-mask-correction/` (reused unchanged) |
| Device/training | Apple MPS; 113,106 trainable parameters; `blocks.3.*`, `blocks.4.*`, plus heads |
| Epochs | 57 completed; early stopping at epoch 57; best epoch 42 |
| Best validation sequence score | `0.7040537` |
| Best validation sequence metrics | family macro-F1 `0.937317`; phase macro-F1 `0.603465`; boundary F1 `0.610256`; repetition-start F1 `0.996109`; repetition-end F1 `0.224404`; count MAE `1.295759` |
| Training/validation compute | `10,839.78` / `258.19` seconds |
| Best checkpoint | `tcn_best.pt`; SHA-256 `18c2e3dcb27c64af76b73e8431df8f9eb59939cad9f156b16dc8f3fa24c16599` |
| Latest checkpoint | `tcn_latest.pt`; SHA-256 `ca1ec059c8c1427289b38c03316fdc232cec25f89d3d79d612dc9bc6081585c0` |
| Run history | `metrics/tcn_history.json`; SHA-256 `01d7a72ac6c027c5e00b821a24b10adedb6d48f988e45f8e43c588923a9eab2f` |
| Test evaluation | **Not performed** (`test_evaluation_performed=false`) |
| Run status | **Training complete; validation-only decoder gate pending** |

The final-two-block intervention improved the training-time validation sequence
score over the last-block candidate (`0.6791891` to `0.7040537`) and materially
increased predicted hold/apex support, but it did not establish the decoder
event gate.

## 2O. Final-two-block validation-only decoder calibration and failure analysis (2026-09-11)

The epoch-42 checkpoint was calibrated and replayed with `decoder.v1` on
validation only. The selected decoder retained apex confirmation:

| Item | Result |
|---|---|
| Checkpoint | `artifacts/r1-tcn-last-two-blocks-event-support/checkpoints/tcn_best.pt`; SHA-256 `18c2e3dcb27c64af76b73e8431df8f9eb59939cad9f156b16dc8f3fa24c16599` |
| Calibration report | `artifacts/r1-tcn-last-two-blocks-event-support/decoder/decoder_calibration.json`; SHA-256 `16b1ffb4cd62c9759b02964b26bf744e54bc2e35b56f4a6067d8b3f92712a1cc` |
| Scope | 896 validation sequences; 240 target boundary pairs; test split unused |
| Selected decoder | start `0.65`, end `0.55`, phase `0.45`, tracking floor `0.60`; apex confirmation enabled |
| Result | Predicted `67`; matched `3`; missed `237`; event F1 `0.019544`; start F1 `0.273616`; end F1 `0.299674`; count MAE `0.333705` |
| Empty-target false events | `63` |
| Gate | **BLOCKED**: repetition-end F1 and empty-target false-event requirements fail |
| Test split | **Not used** (`test_split_used=false`) |

Failure analysis is recorded at
`artifacts/r1-tcn-last-two-blocks-event-support/decoder/decoder_failure_analysis.json`
and `.md`. The JSON SHA-256 is
`42a78973dc990df7feaae27c48c3d8d5d6748e22cb135a7d6e9a7d9d166107d4`; the
Markdown SHA-256 is
`cf697258b8e1ea4d89bad261eaadba900e9abe14c428659a0567a7d504f1ce5a`.

The model now supplies `49/240` apex-qualified intervals and `110/240`
intervals with any predicted hold, up from the last-block candidate. However,
all `219/219` REHAB24-6 segmentation starts remain below the selected tracking
floor, and only `91/240` target ends have a nearby end-probability crossing.
Because `149/240` target ends have no raw end signal, a decoder-only threshold
change cannot reach the declared end-F1 gate. There are still `219` segmentation
plus `21` procedural target pairs and `0` unknown-boundary pairs, so the
unavailable-boundary fix is not the remaining cause.

## 2P. Full-TCN fine-tuning pilot, calibration, and stop decision (2026-09-11)

The two-block representation experiment improved sequence metrics but failed
the decoder gate. The authorized short full-TCN pilot was therefore run before
requesting new reviewed boundary data. It used a fresh root and changed only
the trainable surface plus the intentionally short pilot budget:

| Item | Definition |
|---|---|
| Config | `training/configs/experiments/r1_tcn_full_finetune_event_support_pilot.yaml`; SHA-256 `4023e3b277ad874f1c9ad2492dff587e6d4e2e1b0f2f16019f96444bc22471cd` |
| Artifact root | `artifacts/r1-tcn-full-finetune-event-support-pilot/` |
| Parent checkpoint | `artifacts/r1-tcn-last-two-blocks-event-support/checkpoints/tcn_best.pt`; SHA-256 `18c2e3dcb27c64af76b73e8431df8f9eb59939cad9f156b16dc8f3fa24c16599` |
| Trainable surface | Full TCN and heads; all `307,410` parameters |
| Pilot budget | Maximum 8 epochs; patience 3; backbone learning rate `1e-5`; head learning rate `3e-4` |
| Held constant | Corrected-mask prepared root, seed `42`, feature/window contract, label masks, loss weights, decoder.v1, apex confirmation, and `evaluate_test_after_training=false` |
| Preflight/runtime | **PASS**; all enabled sources available, finite outputs, no test evaluation |
| Training | MPS; 8 epochs completed; best epoch 6; train `1,455.94` s; validation `39.21` s |
| Best validation sequence score | `0.7176679` |
| Best checkpoint | `tcn_best.pt`; SHA-256 `1695ab70241e3f058164e82281583b89a899907636e4e43c1f2c94a5761e91b2` |
| Latest checkpoint | `tcn_latest.pt`; SHA-256 `cd260d97a827c066e0e9a45d44cf0cdbd38dbcddc3806ed8a66dbb790e4305c8` |
| Metrics/report hashes | `metrics.json` `7ce08a8d284f4129ac8c95ea0d6e3db3d551caa4e664d3e830c0bcfc0eb7cacc`; `tcn_history.json` `64088aac11dd2c0d7e164ceba9ffca35e4daa796aaccbb4a9f28ade771accaef`; `experiment_record.json` `a9da698db53dcec4b5473a7933634eb7ec9abb8cd3358147dc9d4f6d77082628` |
| Validation calibration | Selected start `0.75`, end `0.65`, phase `0.45`, tracking floor `0.60`, apex required; report SHA-256 `5a37ff1b1e8c3965b622fba8b4aeb0a8a14fceb47d0aaaf849f8cce7d5fce3cd` |
| Decoder result | Predicted `64`, matched `3`, missed `237`; event F1 `0.019737`; end F1 `0.296053`; count MAE `0.330357`; `60` false events on empty-target sequences; **gate BLOCKED** |
| Failure analysis | JSON SHA-256 `192bb944ff6c29e2af503e542774bae9e34e3521f54826ecf7bac0f8bb17e89b`; Markdown SHA-256 `cbe990c531dbb0e8f3912aa29d2a0a41ee3685cc5a91eaa58ade1a29abf70917`; `43/240` apex-qualified, `103/240` predicted-hold-present, `219/219` segmentation starts below tracking floor, `101/240` end-signal crossings |
| Test evaluation | **Not performed**; `test_split_used=false` |

The pilot improved the aggregate validation sequence score from `0.7040537` to
`0.7176679`, but it did not improve the decisive tracking-floor mechanism:
`219/219` explicit segmentation starts remain below the `0.60` floor. Apex
support also remains sparse (`43/240` qualified intervals), and only `101/240`
target ends have a nearby end-probability crossing; therefore a decoder-only
threshold change cannot reach end F1 `0.60`. The unavailable-boundary fix is
not implicated: the target set remains `219` segmentation plus `21` procedural
pairs and `0` unknown-boundary pairs.

This is the final authorized supervised model-only escalation in the current
data regime. Stop further training and request reviewed REHAB24-6 boundary
labels and/or additional representative seated unilateral recordings with
explicit start, hold/apex, and end supervision. Do not tune against test or
invent labels. After external supervision is available, audit it, regenerate a
new prepared root, and begin again with preflight and validation-only
calibration before considering a second seed or locked test evaluation.

## 2Q. UCO boundary-support data gate and explicit normalization-transfer path (2026-09-11)

The next eligible change is an isolated data-support experiment using the
locally staged UCOPhyRehab++ recordings. This is not a repeat of the exhausted
corrected-mask model-only escalation. The UCO source is enabled for exact
recording-boundary supervision; its composite quality labels remain disabled
for the four dimension-specific quality heads, and synthetic augmentation is
disabled for this storage-constrained run.

Before training, the working tree was audited and the required controls passed:

| Item | Result |
|---|---|
| Repository | `/Users/devk/AdaptFit`; branch `main`; source commit `b3926d864c1d976504c2d7131db8d4992cf2ef45` |
| Immutable corrected-v1 TCN SHA-256 | `e69ff69d2eaad85319bec80f25a5f2a8cb3d415169c5c4ef2441a8bc7f99011b` |
| UCO boundary-support config | `training/configs/experiments/r1_tcn_uco_boundary_support.yaml`; SHA-256 `04ea6b6cb5e474207a0882073d79e183ec31de7c63da74dd852474cfcd507fa4` |
| Prepared root | `data/processed-r1-uco-boundary-support/`; approximately 7,315 real sequences, 86,076 train windows, 16,483 validation windows, and 7,536 test windows |
| Storage and numeric audit | float32; train/validation features finite; test predictions not collected |
| Boundary audit | unknown/unavailable rows contain no endpoint labels; UCO strong-boundary rows are present |
| Preflight | **PASS**; all required enabled sources decode; UCO has 393 exact-boundary recordings |
| Runtime check | **PASS**; input `[64,128,283]`, finite outputs, test loader instantiated but no test inference |
| Documentation validation | **PASS** |
| Test suite | **PASS**; 135 tests |
| Diff check | **PASS** |
| UCO artifact root | `artifacts/r1-tcn-uco-boundary-support/` contains preparation metadata only; no checkpoint exists |

The parent checkpoint uses a different declared normalization version from the
new prepared root. The runner therefore has an explicit, provenance-recorded
warm-start normalization-transfer path. Default checkpoint compatibility and
exact resume remain strict; the transfer flag is only used for the declared
initialization checkpoint and is covered by a regression test. Training must
not begin until the targeted and full test suites remain green after this
change.

The next command is the bounded UCO training run with test evaluation disabled:

```bash
python3 -m training.train \
  --config training/configs/experiments/r1_tcn_uco_boundary_support.yaml \
  --project-root /Users/devk/AdaptFit \
  --models tcn --device mps --seed 42 --skip-test
```

The only eligible follow-up after training is validation-only decoder
calibration and failure analysis in this new artifact root. The decoder gate
remains count MAE `<= 0.40`, repetition-end F1 `>= 0.60`, and zero false events
on empty target sequences. Test evaluation remains locked until that gate
passes.

## 2R. Partial-pose tracking-target correction and regenerated prepared root (2026-09-11)

The UCO validation failure analysis showed that the tracking target was
structurally diluted for partial-pose sources: UCO records expose three
exercise-relevant joints, but the target denominator included all canonical
joints. The resulting UCO target was `0.090909`, forcing every labeled UCO
boundary start below the decoder tracking floor. REHAB24-6 and UL-RED showed
the same partial-source denominator effect (`0.515151` and `0.454545` in the
prior prepared root).

The narrow correction records the canonical joints structurally represented by
each adapter sequence and excludes unlisted joints from the self-supervised
tracking-target denominator. It does not change boundary labels, phase labels,
decoder thresholds, model architecture, or quality supervision. Regression
coverage was added for a three-joint UCO-style sequence and the UCO adapter
metadata.

| Item | Result |
|---|---|
| Code | `training/src/features/anatomy.py`, `training/src/data/adapters.py`, and `training/analyze_decoder_failures.py` updated |
| Regression tests | Partial-pose tracking target and UCO metadata tests added |
| Test suite after correction | **PASS**; 136 tests |
| New config | `training/configs/experiments/r1_tcn_uco_tracking_target_correction.yaml` |
| New prepared root | `data/processed-r1-uco-tracking-target-correction/` |
| New artifact root | `artifacts/r1-tcn-uco-tracking-target-correction/` |
| Prepared data | 7,315 sequences; 86,076 train windows; 16,483 validation windows; 7,536 test windows; float32 |
| Finite-feature audit | **PASS**; train and validation features contain zero non-finite values |
| Boundary-mask audit | **PASS**; unknown-boundary rows contain zero endpoint labels |
| Tracking-target audit | UCO, REHAB24-6, UL-RED, and IntelliRehabDS partial-source targets rise to approximately `1.0` when their represented joints are observed; MM-Fit remains unchanged because its canonical-source metadata does not declare a partial expected set |
| Runtime check | **PASS**; finite TCN outputs with `[64,128,283]` input; no test inference |

The new experiment will warm-start from the UCO boundary-support checkpoint
with the same normalization version and otherwise unchanged training recipe.
Its only changed factor is the regenerated tracking-target contract. Test
evaluation remains disabled.

## 2S. UCO partial-pose tracking-target correction training and calibration (2026-09-11)

The regenerated UCO root was used for one bounded experiment. The only
experimental factor was the corrected partial-pose tracking-target denominator;
the UCO exact boundaries, phase-label masks, model contract, decoder policy,
seed, and quality-head disablement were held constant.

| Item | Result |
|---|---|
| Config | `training/configs/experiments/r1_tcn_uco_tracking_target_correction.yaml`; file SHA-256 `194f5edc194d822c2d2fd7f566b6441cf3c114b3c73a16e94ab745fa87e8ef27` |
| Effective config hash | `sha256:697ca78b451c3c2a2ed2d1b755f0b652947aaf8b2c352dd4562582c97f57e55c` |
| Prepared root | `data/processed-r1-uco-tracking-target-correction/`; 7,315 sequences; 86,076 train windows; 16,483 validation windows; 7,536 test windows |
| Artifact root | `artifacts/r1-tcn-uco-tracking-target-correction/` |
| Parent checkpoint | UCO boundary-support best checkpoint; SHA-256 `9c69b9aa8173da7f10140779ec70f87c012ae151519e3f2375e61b0dd09327a0`; no normalization transfer was required |
| Training | MPS; all `307,410` parameters trainable; 5 epochs completed; early stopping; best epoch 2; train `573.18` s; validation `135.70` s |
| Best validation sequence score | `0.6176060` |
| Best checkpoint | `tcn_best.pt`; SHA-256 `bc69117c1a287a27fcf63466238ab97da812eda3ff92e5a4db968f94ef80d20` |
| Run report | `metrics.json`; SHA-256 `d2b03276d03d38ca102c817b2fa06916235512372d1a57672a2716560fbbcde3` |
| Validation calibration | Selected start `0.75`, end `0.55`, phase `0.45`, tracking floor `0.50`, apex required; report SHA-256 `140595d5f986b97960132517fdaeba7f6511f03962f112eec47f061414534045` |
| Decoder result | Predicted `27`, matched `17`, missed `726`; event F1 `0.044156`; start/end F1 `0.070130`; count MAE `0.584`; `7` false events on empty-target sequences; **gate BLOCKED** |
| Failure analysis | JSON SHA-256 `8f674a3cd4d43a131b4b93885c49dab2c6ca48b76b55fa1d493dbcb78b388b17`; Markdown SHA-256 `bb10ce9703ea6751f4e55dd8dc66535d97237ae25e2ccedf663e367cbeaabcac` |
| Test evaluation | **Not performed**; `test_split_used=false` |

The correction fixed the tracking-floor mechanism: `0/236` segmentation starts
and `0/507` UCO strong-boundary starts were below the tracking floor. It did not
fix the event path. REHAB24-6 supplied a nearby end signal for `119/236`
targets and qualified apex support for `29/236`; UCO supplied a nearby start
signal for `0/507`, a nearby end signal for `1/507`, and apex support for
`0/507`. UCO’s exact recording spans intentionally have no per-frame phase or
hold labels, so the decoder’s apex-confirmation rule cannot be satisfied by
those targets alone. This is a supervision/signal limitation, not a reason to
lower the tracking floor or fabricate phase labels.

## 2T. UCO sparse-boundary positive-weight intervention (2026-09-11)

Because boundary positives are sparse (approximately `186:1` negative to
positive among valid training frames), one bounded loss-only intervention was
tested. It added a configurable positive-weight cap and changed only that cap
from the default `10.0` to `64.0`; the corrected prepared root and all decoder
settings remained unchanged.

| Item | Result |
|---|---|
| Config | `training/configs/experiments/r1_tcn_uco_boundary_positive_weight.yaml`; file SHA-256 `8254b86ab77aa7f3b21d4f9f96c47803ce377b74c1e1a4c339e6947955e8da52` |
| Effective config hash | `sha256:115cdad392207c069378ad6c644edcc04a5c099652f96113b90b53a1b153953b` |
| Prepared root | Reused unchanged: `data/processed-r1-uco-tracking-target-correction/` |
| Artifact root | `artifacts/r1-tcn-uco-boundary-positive-weight/` |
| Parent checkpoint | Tracking-target-correction best checkpoint; SHA-256 `bc69117c1a287a27fcf63466238ab97da812eda3ff92e5a4db968f94ef80d20` |
| Changed factor | `training.boundary_positive_weight_cap: 64.0`; default is `10.0` |
| Training | MPS; all `307,410` parameters trainable; 4 epochs completed; early stopping; best epoch 1; train `464.57` s; validation `110.09` s |
| Best validation sequence score | `0.5252875`, down from `0.6176060` in the tracking-target correction run |
| Best checkpoint | `tcn_best.pt`; SHA-256 `1e2bdcdcb6870ddaa215ce6d406c2ab0b01f95361fb0ce16a1528df00c0d3c52` |
| Run report | `metrics.json`; SHA-256 `60808ad1dcbd5790b7557144525c7fecc1799c82cc6f7d029d123d8824cd207d` |
| Validation calibration | Selected start `0.65`, end `0.75`, phase `0.45`, tracking floor `0.50`, apex required; report SHA-256 `140a779da9c481bcb62f67350630fb312d52a2fdf364588124af265d95e4d28a` |
| Decoder result | Predicted `58`, matched `21`, missed `722`; event F1 `0.052434`; start F1 `0.144819`; end F1 `0.142322`; count MAE `0.5864`; `24` false events on empty-target sequences; **gate BLOCKED** |
| Failure analysis | JSON SHA-256 `f425955f980a883af36f813da62147aea5cb99a23d14c79cfe2b812e6436aa3c`; Markdown SHA-256 `9f3344882af4ed1db4930e064bd8466049914e0429ea3d144ed25a6241c828ba` |
| Test evaluation | **Not performed**; `test_split_used=false` |

The cap increase is rejected. It lowered the aggregate validation score, left
the decoder gate blocked, and increased false events on empty-target sequences.
The raw-signal diagnosis remains source-specific: the corrected tracking
contract passes, but UCO still lacks learned start/end/apex support and has no
hold/apex supervision. Do not repeat this cap increase or use decoder-threshold
changes to force a pass.

## 2U. Current decision and external dependency (2026-09-11)

The UCO data-support path and one narrow loss intervention are now exhausted
without a validation gate pass. The next action is not another blind model-only
run. Request one of the following reviewed inputs:

1. reviewed UCO and/or REHAB24-6 start, hold/apex, and end labels for the
   launch-exercise recordings; or
2. additional representative seated unilateral recordings carrying those same
   labels.

When that supervision is available, audit licenses and participant/source
splits, regenerate a new prepared root, and run preflight, runtime checks,
validation-only training/calibration, and failure analysis in a new artifact
root. Keep the locked test split unevaluated. Only after the validation gate
passes may the second seed and exactly one locked test evaluation proceed.

Until then, preserve the corrected tracking-target root unchanged, keep quality
supervision disabled, and do not promote either UCO checkpoint.

The external-label intake contract is documented in
[`docs/reviewed-event-supervision-request.md`](reviewed-event-supervision-request.md).

## 2V. UL-RED markerless 3Rep boundary-source correction and bounded rerun (2026-09-11)

A raw-source audit found a local supervision source that the previous adapter
discarded: every UL-RED subject archive contains a `marker-less/3Rep_Sxx.csv`
file, and 219 of the 219 available R3 AMC recordings have matching rows. The
CSV indices are zero-based while the AMC parser exposes one-based frame IDs;
the adapter now performs that explicit conversion and leaves malformed or
out-of-bounds rows unavailable. The correction is boundary-only: UL-RED R3
recordings still do not receive fabricated per-frame phase, hold, or apex
labels.

Because the existing float32 prepared root is approximately 15 GB and the
development volume had approximately 4.5 GB free, the new prepared root was
regenerated as an isolated hard-link overlay: immutable feature and auxiliary
arrays are linked read-only, while boundary arrays, JSONL metadata, index, and
provenance are newly written. The parent root remains unchanged. This is
recorded in `overlay_provenance.json` and
`prepared_overlay_provenance.json`.

| Item | Result |
|---|---|
| Regression and contract tests | **PASS**; focused UL-RED/data-contract suite `20 passed`; full suite `138 passed` |
| Docs/compile checks | **PASS**; `python3 scripts/validate_docs.py`, `python3 -m compileall -q training`, and `git diff --check` |
| Config | `training/configs/experiments/r1_tcn_ulred_boundary_support.yaml`; file SHA-256 `4a8fa4def53d686ca57c29a8958f6eae1dca7a5ff77b38b684d74601dd1ad198`; effective config hash `sha256:320a23fd5cf43a79ed821e08b90a419b221b9ec87f57156be77561c9115bfc04` |
| Prepared root | `data/processed-r1-ulred-boundary-support/`; 7,315 sequences; 86,076 train windows; 16,483 validation windows; 7,536 test windows; 219 explicit UL-RED R3 boundary sequences; 12,091 affected windows |
| Artifact root | `artifacts/r1-tcn-ulred-boundary-support/` |
| Parent checkpoint | `artifacts/r1-tcn-uco-tracking-target-correction/checkpoints/tcn_best.pt`; SHA-256 `bc69117c1a287a27fcf63466238ab97da812eda3ff92e5a4db968f94ef80d20`; no normalization transfer |
| Preflight/runtime | **PASS**; all enabled sources decoded; finite TCN outputs on MPS with `[64,128,283]` input; test inference not run |
| Training | MPS full TCN; all `307,410` parameters trainable; 5 epochs completed; early stopping; best epoch 2; train `624.81` s and validation `139.03` s |
| Best validation sequence score | `0.6013545` |
| Best checkpoint | `tcn_best.pt`; SHA-256 `7ed351765ad1c943d525721c03fdcec65c1fdd06df2966c59971ca934b5d4ce1` |
| Run report | `metrics.json`; SHA-256 `6bbf36e99b6f6ca122a6d127bf2db6992687f2c3f8d05743f70670fffdaba3ac` |
| Validation calibration | Selected start `0.65`, end `0.55`, phase `0.45`, tracking floor `0.50`, apex required; report SHA-256 `c7fb7b28e9d92f496d0b38c1cf3e44614129d5113ee809a06ddd47844fdedac1` |
| Decoder result | Target pairs `872` (`636` strong UL-RED plus `236` segmentation); predicted `33`, matched `22`, missed `850`; event F1 `0.048619`; start F1 `0.072928`; end F1 `0.070718`; count MAE `0.684`; `8` false events on empty-target sequences; **gate BLOCKED** |
| Failure analysis | JSON SHA-256 `4715afb36b623ec437ca3256d4444f4f41d8ba7bb016a660cd57dfe7c63ced52`; Markdown SHA-256 `6476e330b670189d46a22af30fc733658d527e575a7f0d6ad40d24e048e0e66e` |
| Test evaluation | **Not performed**; `test_split_used=false` |

The intervention is retained as valid data-source evidence but does not pass
the decoder gate. The strong UL-RED target group has no target hold/apex
labels, and only `1/636` strong target pairs has an apex-qualified interval in
the validation failure analysis. Do not lower the tracking floor, remove apex
confirmation, tune against test, or fabricate phase labels. The next required
input is reviewed UL-RED/UCO/REHAB24-6 start, hold/apex, and end supervision or
additional representative seated unilateral recordings with those labels.

---

## 1. Candidate Checkpoint & Run Inventory (Task A1)

| Run / Artifact Root | Config | Checkpoint | Training Status | Labeled Evaluation Status | Model Architecture & Params | Split & Sources |
|---|---|---|---|---|---|---|
| `artifacts/corrected-v1/` | `training/configs/v1_corrected.yaml` | `tcn_best.pt` (epoch 35)<br>`gru_baseline.pt` (epoch 34) | **Training complete** (50 / 49 epochs) | **Evaluated** (Sequence & Window)<br>Family Acc: 91.25% (TCN) / 89.47% (GRU)<br>Start F1: 99.54% (TCN) / 83.50% (GRU)<br>End F1: 18.47% (TCN) / 5.53% (GRU)<br>Rep Count MAE: 0.31 (TCN) / 30.63 (GRU) | Causal TCN (307,410)<br>Causal GRU (68,178)<br>283 inputs, 128 frames | 50 train / 11 val / 11 test participant groups.<br>Zero identity collisions.<br>Sources: REHAB24-6, IntelliRehabDS, MM-Fit, UL-RED, procedural seed. |
| `artifacts/v2-quality/` | `training/configs/v2_quality.yaml` | None | **Prepared data only** | **Not trained / not evaluated** | Target: Causal TCN / GRU with expert quality head | 69 train / 15 val / 15 test groups.<br>Adds UCOPhyRehab++ (exact spans, composite score). |
| `artifacts/v2-quality-fixed/` | `training/configs/v2_quality_fixed.yaml` | `tcn_best.pt` (epoch 42 / 72)<br>`gru_baseline.pt` (interrupted epoch 52) | **TCN training complete** (72 epochs, best 42).<br>**GRU interrupted** (epoch 52). | **TCN Evaluated on Test** (Sequence & Window):<br>Family Acc: 90.47%, Macro-F1: 83.91%<br>Phase Acc: 81.36%, Macro-F1: 54.92%<br>Start F1: 66.96%, End F1: 12.26%<br>Rep Count MAE: 2.03<br>Expert Quality Acc: 56.92% (MAE: 0.43)<br>Quality-4 heads: 0% coverage (masked) | Causal TCN (307,410)<br>Causal GRU (68,178)<br>283 inputs, float16 memmaps | Reuses `data/processed-v2-quality`.<br>1,007 logical test sequences.<br>Tested on UL-RED, UCO, wheelchair-positions. |
| `artifacts/r1-tcn-boundary/` | `training/configs/experiments/r1_tcn_boundary_finetune.yaml` | `tcn_best.pt` (epoch 12 / 27)<br>`tcn_latest.pt` (epoch 27) | **Full training complete; decoder gate blocked; test locked** | Validation-only sequence metrics plus `decoder/decoder_calibration.json`. End F1 `0.0` after calibration; no test evaluation. | Causal TCN (307,410)<br>1,746 trainable head parameters<br>283 inputs, 128 frames | Corrected-v1 prepared split; parent `corrected-v1/tcn_best.pt`; Apple MPS heads-only run. |
| `artifacts/r1-tcn-boundary-mask-correction/` | `training/configs/experiments/r1_tcn_boundary_mask_correction.yaml` | `tcn_best.pt` (epoch 53 / 68)<br>`tcn_latest.pt` (epoch 68) | **Training complete; decoder gate blocked; test locked** | Validation-only sequence metrics plus `decoder/decoder_calibration.json`. End F1 `0.0`; count MAE `0.2678571`; 240 target events missed; no test evaluation. | Causal TCN (307,410)<br>1,746 trainable head parameters<br>283 inputs, 128 frames | Isolated corrected-mask prepared split; parent `corrected-v1/tcn_best.pt`; Apple MPS heads-only run with exact resume. |
| `artifacts/r1-tcn-phase-weight-correction/` | `training/configs/experiments/r1_tcn_phase_weight_correction.yaml` | `tcn_best.pt` (epoch 9 / 24)<br>`tcn_latest.pt` (epoch 24) | **Training complete; decoder gate blocked; test locked** | Training-time validation sequence score `0.6679357`; validation-only decoder calibration is blocked with end F1 `0.153846` and 15 empty-target false events; no test evaluation. | Causal TCN (307,410)<br>1,746 trainable head parameters<br>283 inputs, 128 frames | Reuses isolated corrected-mask prepared split; parent `r1-tcn-boundary-mask-correction/tcn_best.pt`; Apple MPS heads-only run with corrected absent-class phase weighting. |
| `artifacts/r1-tcn-last-block-event-support/` | `training/configs/experiments/r1_tcn_last_block_event_support.yaml` | `tcn_best.pt` (epoch 8 / 23)<br>`tcn_latest.pt` (epoch 23) | **Training complete; decoder gate blocked; test locked** | Validation sequence score `0.6791891`; validation-only decoder calibration blocked with end F1 `0.094862`, count MAE `0.271205`, and 8 empty-target false events; failure analysis recorded; no test evaluation. | Causal TCN (307,410)<br>57,426 trainable parameters<br>283 inputs, 128 frames | Reuses isolated corrected-mask prepared split; parent `r1-tcn-phase-weight-correction/tcn_best.pt`; final block `blocks.4.*` plus heads only. |
| `artifacts/r1-tcn-last-two-blocks-event-support/` | `training/configs/experiments/r1_tcn_last_two_blocks_event_support.yaml` | `tcn_best.pt` (epoch 42 / 57)<br>`tcn_latest.pt` (epoch 57) | **Training complete; decoder gate blocked; test locked** | Validation sequence score `0.7040537`; validation-only decoder calibration blocked with end F1 `0.299674`, count MAE `0.333705`, and 63 empty-target false events; failure analysis recorded; no test evaluation. | Causal TCN (307,410)<br>113,106 trainable parameters<br>283 inputs, 128 frames | Reuses isolated corrected-mask prepared split; parent `r1-tcn-last-block-event-support/tcn_best.pt`; final two blocks `blocks.3.*` and `blocks.4.*` plus heads only. |
| `artifacts/r1-tcn-full-finetune-event-support-pilot/` | `training/configs/experiments/r1_tcn_full_finetune_event_support_pilot.yaml` | `tcn_best.pt` (epoch 6 / 8)<br>`tcn_latest.pt` (epoch 8) | **Training complete; decoder gate blocked; test locked** | Validation sequence score `0.7176679`; validation-only decoder calibration blocked with end F1 `0.296053`, count MAE `0.330357`, and 60 empty-target false events; failure analysis recorded; no test evaluation. | Causal TCN (307,410)<br>307,410 trainable parameters<br>283 inputs, 128 frames | Reuses isolated corrected-mask prepared split; parent `r1-tcn-last-two-blocks-event-support/tcn_best.pt`; full TCN plus heads; pilot max 8 epochs/patience 3. |
| `artifacts/r1-tcn-ulred-boundary-support/` | `training/configs/experiments/r1_tcn_ulred_boundary_support.yaml` | `tcn_best.pt` (epoch 2 / 5)<br>`tcn_latest.pt` (epoch 5) | **Training complete; decoder gate blocked; test locked** | Validation sequence score `0.6013545`; validation-only calibration blocked with end F1 `0.070718`, count MAE `0.684`, and 8 empty-target false events; failure analysis recorded; no test evaluation. | Causal TCN (307,410)<br>307,410 trainable parameters<br>283 inputs, 128 frames | Isolated `data/processed-r1-ulred-boundary-support` UL-RED 3Rep boundary overlay; parent `r1-tcn-uco-tracking-target-correction/tcn_best.pt`; boundary-only source correction. |

---

## 2. Key Findings & Baseline Comparison

### A. Repetition-End Bottleneck
Across both `corrected-v1` and `v2-quality-fixed`:
- **Repetition Start** is reliably detected (99.54% F1 on v1, 66.96% F1 on v2 test).
- **Repetition End** is materially weaker (18.47% F1 on v1, 12.26% F1 on v2 test).
- **Root Cause**: Phase transitions to rest/eccentric boundary have wide source annotation disagreement and loose boundary definitions. The validation-only failure analysis also found missing hold/apex support and tracking-floor misalignment; the phase-weight correction improved training-time validation but still needs decoder calibration (Task C1).

### B. Quality Head Coverage Reality
- All four dimension-specific quality outputs (ROM, tempo, smoothness, trunk compensation) currently have **0.0% labeled coverage** across all existing datasets.
- UCOPhyRehab++ provides an ordinal 1–5 physiotherapist composite execution score mapped to `expert_quality_logits` (5-class). On 65 evaluated test sequences with expert ratings, TCN achieves **56.92% accuracy and 0.43 MAE**. This is a useful auxiliary task, but must never be presented as four independent clinical quality scores.

### C. Target-Population Data Gap
- There are **no real amputee, congenital limb-difference, or wheelchair-user recordings** in the training or test splits.
- Wheelchair-position recordings in IntelliRehabDS are posture proxies by non-wheelchair users.
- Procedural seeds and synthetic limb-occlusion masks are engineering aids for network gradient stability; they are not target-population validation.

---

## 3. Luna Execution Task Progress Tracker

| Task ID | Description | Status | Verification & Evidence |
|---|---|---|---|
| **A1** | Inventory existing work & artifacts | **COMPLETED** | Verified configs, checkpoints, memmap shapes, parameter counts, and histories across `corrected-v1`, `v2-quality`, and `v2-quality-fixed`. |
| **A2** | Define valid comparison cohort | **IN PROGRESS** | Training participant IDs isolated; test splits locked. Union overlap checks enforced by `training.audit`. |
| **A3** | Establish baseline & failure inventory | **COMPLETED** | Sequence evaluation generated for `v2-quality-fixed/tcn_best.pt` on test split (1,007 sequences); compared with `corrected-v1`. |
| **A4** | Prepare focused launch supervision | **PENDING** | Define launch-exercise rep boundary conventions (curls, rows, extensions, marches, reach). |
| **B1** | Safe warm-start weight initialization | **COMPLETED (smoke verified)** | `--init-checkpoint` and `training.init_checkpoint` loaded the corrected-v1 parent and recorded its path and source commit in the R1 checkpoint. |
| **B2** | Trainable-layer selection & resume | **COMPLETED (full run and exact resume verified)** | Frozen-backbone head training produced 1,746 trainable parameters; the epoch-10 interruption was resumed to completion after normalizing accelerator-loaded RNG state to CPU. |
| **B3** | Redundant work reduction (stride/sampler) | **PENDING** | Evaluate stride-16/32 training sampling while preserving validation reconstruction. |
| **B4** | Bounded experiment configuration | **COMPLETED** | `training/configs/experiments/r0_baseline.yaml` and `r1_tcn_boundary_finetune.yaml`; both validate and use isolated roots. |
| **C1** | Improve decoding without gradient updates | **COMPLETED (blocked gate)** | Original R1, corrected-mask, phase-weight, last-block, last-two-block, and full-TCN validation-only calibrations are recorded and fail the event gate. The full-TCN pilot leaves 219/219 segmentation starts below the tracking floor and only 101/240 end-signal crossings; request reviewed boundary labels/additional representative recordings before further model-only work. |

---

## 4. Verification History

| Date | Commit | Role | Action | Result | Notes |
|---|---|---|---|---|---|
| 2026-09-01 | `e75ba65` | Historical baseline | Initial training & corrected-v1 benchmark | Completed | Full TCN/GRU trained; 123 tests passing. |
| 2026-09-07 | `b6ac20c` | Documentation revision | Documentation contracts and runbooks refined | Completed | Unified metadata, schemas, and roadmap across 23 files. |
| 2026-09-07 | `ba8bf1a` | **Code baseline** | Repository-wide audit & TCN v2 evaluation | Passed (124 tests) | Evaluated `v2-quality-fixed` TCN; fixed TCN streaming expert head; added CLI scripts and execution log; last commit modifying model/runtime/test code. |
| 2026-09-08 | `f890101` / `686218a` | Documentation revision | Roadmap alignment & canonical metadata synchronization | Passed (124 tests) | Aligned roadmap statuses, audited dataset mappings, and synchronized canonical documentation metadata. |
| 2026-09-08 | `d5362ee` | Documentation revision | Execution provenance audit & contract grounding | Passed (124 tests) | Updated prepared-manifest counts, labeled planned paths and candidate datasets, grounded speedup hypotheses. |
| 2026-09-08 | `1a46f38` | Documentation revision | Phase 5 planned-metric labeling | Passed (124 tests) | Marked mobile calibration, parity, latency, memory, and thermal values as planned acceptance criteria. |
| 2026-09-08 | `613ff12` | Documentation revision (prior canonical HEAD) | Expand dataset catalog and formalize operational documentation contracts | Passed (124 tests) | Pushed baseline audited before the documentation correctness repair; expanded dataset registry and operational contracts. |
| 2026-09-08 | `847fd5d` | Pre-training gate implementation | Decoder, provenance, staged runner controls, R0/R1 configs, and readiness docs | Passed (132 tests; docs validator) | Corrected-v1 hashes unchanged; R0 manifests/calibration regenerated; R1 training intentionally not launched. |
| 2026-09-08 | `627283b` | R1 smoke verification | Two-epoch CPU heads-only warm-start with test evaluation disabled | Passed | Finite losses; validation-only metrics and provenance complete; full R1 remains unrun. |
| 2026-09-09 | `d7848e2` | R1 full heads-only training | 27-epoch Apple MPS run with early stopping and test lock | Completed | Best epoch 12; validation-only metrics and provenance complete; later calibration is recorded as blocked; test evaluation remains locked. |
| 2026-09-09 | `b3926d8` | R1 validation-only decoder calibration | Versioned threshold neighborhood on validation split only | Blocked | End F1 `0.0`; test split unused; UL-RED unavailable-boundary mask mismatch identified for targeted correction. |
| 2026-09-10 | `b3926d8` | Corrected-mask exact resume and validation-only calibration | Resumed epoch-10 checkpoint to epoch 68, selected epoch 53, and calibrated `decoder.v1` on validation only | Completed training; blocked calibration | Checkpoint and report hashes recorded above; end F1 `0.0`; 240 events missed; test split unused. |
| 2026-09-10 | `b3926d8` | Phase-weight correction training | Corrected absent-class normalization, ran isolated MPS heads-only training from the corrected-mask checkpoint, and early-stopped at epoch 24 | Completed training; calibration later blocked | Best epoch 9; sequence score `0.6679357`; checkpoint/report hashes recorded above; test split unused. |
| 2026-09-10 | `b3926d8` | Phase-weight correction validation-only decoder calibration | Calibrated the epoch-9 checkpoint with `decoder.v1` on validation only | Blocked | End F1 `0.153846`; count MAE `0.279018`; 15 false events on empty sequences; report hash recorded above; test split unused. |
| 2026-09-10 | `b3926d8` | Phase-weight validation-only failure analysis and last-block intervention definition | Replayed decoder failure modes and defined an isolated final-TCN-block fine-tune config | Passed definition/preflight; training not launched | 22/240 qualified apex intervals, 154/219 segmentation starts below tracking floor, 83/240 end-signal crossings; new artifact root/config recorded above; test split unused. |
| 2026-09-10 | `b3926d8` | Last-block event-support training | Ran the isolated final-block fine-tune with test disabled | Completed training; decoder gate later blocked | Best epoch 8 of 23; sequence score `0.6791891`; checkpoint/report hashes recorded above; test split unused. |
| 2026-09-10 | `b3926d8` | Last-block validation-only calibration and failure analysis | Calibrated and replayed `decoder.v1` on validation only | Blocked | End F1 `0.094862`; count MAE `0.271205`; 8 empty-target false events; `21/240` apex-qualified intervals and `219/219` segmentation starts below tracking floor; test split unused. |
| 2026-09-10 | `b3926d8` | Final-two-block intervention definition | Configured exactly `blocks.3.*` plus `blocks.4.*` and heads from the best last-block checkpoint | Passed definition/preflight; training not launched | 22 trainable tensors / `113,106` parameters; new artifact root/config recorded above; test split unused. |
| 2026-09-10–2026-09-11 | `b3926d8` | Final-two-block event-support training | Ran the isolated final-two-block fine-tune with test disabled | Completed training; decoder gate later blocked | Best epoch 42 of 57; sequence score `0.7040537`; checkpoint/report hashes recorded above; test split unused. |
| 2026-09-11 | `b3926d8` | Final-two-block validation-only calibration and failure analysis | Calibrated and replayed `decoder.v1` on validation only | Blocked | End F1 `0.299674`; count MAE `0.333705`; 63 empty-target false events; `49/240` apex-qualified intervals, `219/219` segmentation starts below tracking floor, and `91/240` end-signal crossings; test split unused. |
| 2026-09-11 | `b3926d8` | Full-TCN fine-tuning pilot definition | Configured full-TCN pilot from the best final-two-block checkpoint | Passed definition/preflight; training not launched | All `307,410` parameters trainable; max 8 epochs/patience 3; new artifact root/config recorded above; test split unused. |
| 2026-09-11 | `b3926d8` | Full-TCN fine-tuning pilot training | Ran the bounded full-TCN pilot with test disabled | Completed training; decoder gate later blocked | Best epoch 6 of 8; sequence score `0.7176679`; checkpoint/report hashes recorded above; test split unused. |
| 2026-09-11 | `b3926d8` | Full-TCN validation-only calibration and failure analysis | Calibrated and replayed `decoder.v1` on validation only | Blocked; supervised escalation stopped | End F1 `0.296053`; count MAE `0.330357`; 60 empty-target false events; `43/240` apex-qualified intervals, `219/219` segmentation starts below tracking floor, and `101/240` end-signal crossings; reviewed labels/additional representative recordings requested; test split unused. |
| 2026-09-11 | `b3926d8` | UL-RED markerless 3Rep source audit, boundary overlay, training, calibration, and failure analysis | Recovered local zero-based 3Rep spans, mapped them to AMC frame IDs, trained one bounded full-TCN candidate, and calibrated `decoder.v1` on validation only | Blocked; local boundary correction retained, reviewed phase/apex supervision required | 219 explicit R3 recordings; best epoch 2 of 5; sequence score `0.6013545`; end F1 `0.070718`; count MAE `0.684`; 8 empty-target false events; test split unused. |

## 2W. Archive-wide local-supervision audit and escalation boundary (2026-09-11)

After the UL-RED rerun, every staged label-bearing archive was inspected for
additional source-backed temporal supervision before requesting external input.
The audit covered IntelliRehabDS `SkeletonData.zip`, MM-Fit `mm-fit.zip`, the
REHAB24-6 joint and segmentation files, UCO JSON/JSONL labels, and UL-RED
subject archives. It found no unconsumed local hold/apex source:

- MM-Fit label rows are set-level spans with an activity/repetition count, and
  the adapter correctly keeps them out of individual repetition-boundary and
  phase supervision.
- UCO supplies exact recording/repetition boundary rows and composite expert
  scores, but no per-frame phase or hold/apex labels.
- REHAB24-6 supplies its existing segmentation rows; no additional reviewed
  phase/apex file was present in the staged files.
- UL-RED supplies marker and marker-less `3Rep` rows. The marker-less spans
  are now consumed for 219 matching R3 recordings, but remain boundary-only.

The density-head fallback in `efficient-training-strategy.md` is therefore not
started. It is authorized only as a secondary experiment for a counting gap
after valid boundary evidence is present, while the current validation failure
also requires start/end and two-frame hold/apex evidence. A density head would
require a new target contract and artifact root and cannot, on its own, clear
the repetition-end or apex-confirmation gates. Starting it now would not
resolve the measured blocker and would consume a training branch without the
required supervision.

The escalation boundary is consequently reached: request reviewed
UCO/UL-RED/REHAB24-6 start, hold/apex, and end rows, or additional representative
seated unilateral recordings carrying those labels. Keep all corrected roots
and the locked test split unchanged. Once accepted rows exist, follow the
intake sequence in `docs/reviewed-event-supervision-request.md`; only a passing
validation gate may unlock seed 43 and the single locked-test evaluation.
