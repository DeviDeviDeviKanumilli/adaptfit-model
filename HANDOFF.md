# AdaptFit handoff

> **Status:** canonical-active
> **Authority:** continuation instructions for the next implementation or evaluation agent
> **Last verified:** 2026-09-09
> **Source commit:** `c766ca1` (documentation state); R1 evidence at `d7848e2`
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
- Last pushed documentation commit: `c766ca1` (`Record full R1 training results`)
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

## The next action

Run validation-only decoder calibration. This reads the validation split only,
tries the versioned threshold neighborhood, and writes an ignored local report:

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

The provisional decoder gate is:

- count MAE ≤ `0.40`;
- repetition-end F1 ≥ `0.60`;
- false events on empty target sequences = `0`.

Do not change thresholds, remove apex confirmation, or read the test split to
force a pass. Record the selected thresholds, gate status, checkpoint hash,
source commit, and report path.

## Decision after calibration

If all decoder requirements pass, freeze the epoch-12 checkpoint and decoder,
then run the locked test evaluation exactly once:

```bash
python3 -m training.evaluate \
  --config training/configs/experiments/r1_tcn_boundary_finetune.yaml \
  --checkpoint artifacts/r1-tcn-boundary/checkpoints/tcn_best.pt \
  --project-root /Users/devk/AdaptFit \
  --device mps
```

Compare R1 with corrected-v1 using the same split, sequence aggregation,
decoder version, and provenance fields. Update the artifact registry before
describing any result as an improvement.

If calibration is blocked, do not promote the checkpoint or start another broad
training run. Inspect representative missed and false end events, verify the
boundary annotation convention and masks, and choose one targeted follow-up:

1. decoder/label correction if errors are temporal-boundary errors;
2. event-focused sampling or reviewed boundary supervision if labels are valid;
3. partial last-block fine-tuning only if the frozen representation is shown to
   miss the motion.

Keep each follow-up in a new artifact root. Do not start JEPA, recommender
learning, or mobile export until the supervised movement comparison and decoder
gate establish a measured need and a valid artifact.

## Hard boundaries

- No test metrics exist for R1 yet.
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
