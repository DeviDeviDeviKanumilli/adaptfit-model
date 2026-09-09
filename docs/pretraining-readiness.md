# AdaptFit pre-training readiness

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** reproducibility gate for the next training run; source code, configs, and generated reports remain authoritative
> - **Last verified:** 2026-09-08
> - **Source commit:** `627283bec6851d95cba988a0235931c158465eab` (R1 smoke evidence; pre-training gate baseline)
> - **Owner:** AdaptFit training and evaluation engineering
> - **Supersedes or supports:** supports `current-state.md`, `efficient-training-strategy.md`, `training-execution-log.md`, and `artifact-registry.md`
> - **Review trigger:** any change to the selected config, prepared split, checkpoint, decoder, label policy, or training runner

This document is the handoff between preparation and the next authorized model
training run. It freezes the evidence already checked and gives Luna one
ordered path. It does **not** start training, select a product release, or
promote the decoder. The only generated artifacts referenced here are local
evaluation reports under `artifacts/r0-baseline/`; that directory is ignored by
Git and must be regenerated when it is absent.

## Current decision

The repository is mechanically ready for an isolated training experiment, and
the bounded R1 smoke run has completed successfully. The next computational
action is the full R1 boundary-focused TCN warm-start described below. The
corrected-v1 checkpoint and prepared data remain immutable. The decoder gate is
recorded as **blocked** because the current TCN does not emit validated end
events on the validation split; this is the measured training target, not a
reason to claim product readiness.

The full R1 run remains the next action after this handoff. Product release remains
blocked until the decoder, quality supervision, target-population evidence, and
native parity gates pass in their own artifacts.

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

The bounded smoke run is complete and passed its provenance gate. The full R1
budget is now eligible, but it remains a separate action. Run it with the same
config and `--models tcn` without the two-epoch override; keep
`evaluate_test_after_training: false` so test evaluation remains locked.

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

The full R1 command, when authorized, is:

```bash
python3 -m training.train \
  --config training/configs/experiments/r1_tcn_boundary_finetune.yaml \
  --project-root /Users/devk/AdaptFit \
  --models tcn --device auto --skip-test
```

The smoke checkpoint and provenance checks passed. Select the best full-R1
checkpoint on validation, calibrate the decoder again on validation, and
evaluate the locked test split only once the candidate and decoder are frozen.

## Stop conditions

Stop before or during R1 if any of the following occurs:

- a source enabled by strict preflight is missing or decodes to zero sequences;
- the prepared index, normalization hash, feature width, window length, split
  identity, or label masks differ from this handoff;
- a checkpoint is written under `artifacts/corrected-v1/` or another existing
  registry root;
- model, optimizer, or gradient values become non-finite;
- a warm-start or resume model/schema/decoder/normalization mismatch is found;
- exact resume lacks optimizer, RNG, sampler, or trainable-layer provenance;
- validation worsens for the declared patience or the compute budget expires;
- a quality or target-population claim is inferred from unavailable labels.

## What remains after the training handoff

The full R1 run is the next action, but it is not the final product gate. After R1,
the remaining evidence work is: validation-only decoder recalibration;
supervised-only versus teacher-assisted comparison if a measured gap remains;
quality-label acquisition; target-population collection; native export and
golden parity; and recommendation implementation. None of those states exists
in the current repository.
