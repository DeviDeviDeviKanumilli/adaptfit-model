# Audit fixes and overnight run

This document records the fixes made after the training hot-path audit. The
changes preserve the existing v2 prepared data and artifacts while making the
next run more honest, faster, and reproducible.

## Preserved data and new outputs

The following existing paths are read-only inputs for the fixed run:

- `/Users/devk/AdaptFit/data/processed-v2-quality/`
- `/Users/devk/AdaptFit/artifacts/v2-quality/`

The fixed configuration writes new run artifacts to:

- `/Users/devk/AdaptFit/artifacts/v2-quality-fixed/`

The fixed run reuses the existing prepared memmaps. It does not create another
large prepared-data tree.

## Label correctness

The UCO composite-quality policy is now passed from configuration into the
adapter. When disabled, UCO repetition scores are masked and marked
`disabled_by_policy`. When enabled, they remain available but are reported as
weak composite physiotherapist labels.

Weak displacement phase labels are not considered valid for a recording-level
sequence containing multiple repetitions. This rule is applied both when new
data is prepared and as a read-only runtime mask over the current v2 metadata.
The source memmaps are never edited.

ROM, tempo, smoothness, and trunk-compensation labels remain unavailable when
the source does not provide those labels. UCO's composite score is not treated
as four independent clinical quality labels.

## Data-loading performance

The training loaders now:

- Materialize one batch instead of copying and converting every feature window.
- Keep float16 feature storage but cast the batch to float32 before inference.
- Use length-bucketed batches so short right-padded windows are trimmed before
  the TCN runs.
- Preserve sample indices and restore source order before evaluation.
- Reopen memmaps safely in spawned macOS worker processes.
- Support two persistent workers and prefetching in the fixed configuration.

The dataset no longer scans the complete feature memmap whenever it opens for
training. Strict finite-value scanning is available as a separate chunked
prepared-data audit.

## Preparation and diagnostics

Future preparation extracts features once per sequence. It writes raw features
to a staging memmap, fits normalization statistics during that pass, then
normalizes continuous columns in place in bounded chunks. Existing prepared
trees are not replaced unless `--replace-existing` is explicitly supplied; an
old tree is moved to a timestamped backup before replacement.

Rolling-range diagnostics now use vectorized causal sliding windows and are
covered by an equivalence test against the previous reference behavior.

## Metrics and provenance

The combined report uses the `adaptfit.metrics.v3` schema:

```text
schema_version
config
seed
data_summary
run
models
evaluations
artifacts
```

Training initializes the root report. Evaluation adds model-specific reports
under `evaluations` without replacing another model's result. JSON writes are
strict and atomic; non-finite metric values are represented as `null`.

Reports include effective phase-label status, label provenance, identity
audits, source and position coverage, projection metadata, and performance
timings.

UCO 3D poses explicitly record:

```text
source_coordinate_dim: 3
model_coordinate_dim: 2
coordinate_projection: xy_from_3d
depth_discarded: true
projection_reason: mobile_2d_contract_without_calibration
```

The model still consumes the 283-value 2D feature contract. Depth is not used
by the current mobile-parity model.

## Validation cadence

The fixed configuration validates on the first epoch, every second epoch, and
the final epoch. Early-stopping patience counts validation checks. Skipped
epochs are recorded in the model history, along with training and validation
timings.

Model computation remains float32. Float16 is a storage optimization only;
unsupported half-precision training is rejected by configuration validation.

## No-training verification

Run this before an overnight training run:

```bash
cd /Users/devk/AdaptFit
ADAPTFIT_RUN_TRAINING=0 ./run_v2_quality_fixed_overnight.sh
```

This verifies:

- The legacy 745-entry and 700/745 family audit.
- UCO checksums.
- Required staged source files.
- Prepared memmap finiteness and shape contracts.
- One-batch float32 TCN forward through the configured worker/bucket loader;
  this check performs no optimization or parameter updates.
- Participant-level split isolation.
- Logical identity collision counts.
- UL-RED and wheelchair-position test coverage.
- Compilation and non-training tests.

The command must finish with `verification complete; training was not run`.

## Latest no-training verification

The fixed verification completed successfully on September 1, 2026. It
reproduced the legacy audit at 745 logical sequences with 700 correct family
predictions, found zero logical-identity collisions, found no participant
overlap between splits, and confirmed UL-RED plus wheelchair-position coverage
in the test split. The strict storage check scanned 291,459 prepared rows and
found finite features in every row. The worker-backed TCN forward check used a
64-by-128-by-283 batch on MPS and produced finite outputs without
backpropagation.

The preserved prepared tree contains 118,581 UCO windows whose older metadata
has the source dimension and `xy_from_3d` marker but not the newer explanatory
fields. The audit labels these rows `legacy_inferred` and counts the discarded
depth explicitly. Newly prepared data will carry the complete projection
metadata contract; the preserved tree was not rewritten.

The non-training contract suite currently passes 121 tests, with two
training-only tests intentionally deselected by the verification script.

## Overnight training

After the no-training verification passes, launch the actual run with:

```bash
cd /Users/devk/AdaptFit
ADAPTFIT_RUN_TRAINING=1 ./run_v2_quality_fixed_overnight.sh
```

The script optionally performs a two-epoch smoke run first, then starts fresh
TCN and GRU training. Set `ADAPTFIT_RUN_SMOKE=0` to omit the smoke pass. Set
`ADAPTFIT_NUM_WORKERS=0` if worker startup causes a local macOS issue.

Expected final files include:

- `artifacts/v2-quality-fixed/checkpoints/tcn_best.pt`
- `artifacts/v2-quality-fixed/checkpoints/gru_baseline.pt`
- `artifacts/v2-quality-fixed/metrics/tcn_history.json`
- `artifacts/v2-quality-fixed/metrics/gru_history.json`
- `artifacts/v2-quality-fixed/metrics/tcn_evaluation.json`
- `artifacts/v2-quality-fixed/metrics/gru_evaluation.json`
- `artifacts/v2-quality-fixed/metrics.json`
- `artifacts/v2-quality-fixed/model_card.md`

## Interpretation boundary

The benchmark remains a public-data research result. Synthetic occlusion,
limb masking, seated examples, wheelchair-position examples, and UCO expert
scores do not constitute clinical validation of amputees, people with limb
differences, wheelchair users, safety, or individualized exercise adaptation.
Real consented target-population recordings and expert labels are still
required.
