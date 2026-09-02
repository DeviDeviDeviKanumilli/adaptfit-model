# Quality benchmark v2

This document describes the prepared, but not yet trained, v2 run in
`/Users/devk/AdaptFit`. The run adds UCOPhyRehab++ supervision while keeping
the existing on-device feature contract and leaving the corrected v1 data and
artifacts untouched.

## Why this run exists

The corrected v1 benchmark has useful movement-family and phase labels, but it
has no reliable dimension-specific form labels and its repetition-end task is
weak. v2 adds a source with explicit repetition spans and a composite expert
execution score so the overnight model can learn a more defensible quality
auxiliary task.

The v2 model still does not infer a disability from a person’s appearance.
Capability information remains an onboarding input, and missing or unobserved
limbs remain masked in the pose contract.

## UCOPhyRehab++ source

The staged files came from the official [UCOPhyRehab++ Zenodo record](https://zenodo.org/records/17935737):

```text
data/raw/ucophyrehabpp/ucophyrehab2_data.jsonl
data/raw/ucophyrehabpp/dataset_3d_with_angles.json
```

The official [Scientific Data paper](https://www.nature.com/articles/s41597-026-07362-5)
describes 27 healthy participants performing 16 rehabilitation exercises. The
released metadata associates repetitions with inclusive initial and final
frames and a 1–5 physiotherapist execution score. The released pose file
contains partial 3D landmarks and the analyzed joint angle. The adapter uses
the x/y projection of those landmarks so the existing 283-value feature
schema remains unchanged.

This source is useful supervision, not target-population evidence. It does not
contain real amputee, congenital limb-difference, or wheelchair-user
participants. Its expert score is a composite assessment involving observable
execution, not four separate labels for ROM, tempo, smoothness, and trunk
compensation. The raw release and article terms must be reviewed before any
redistribution or commercial use.

## Adapter behavior

The UCO adapter emits two logical sequence types:

- Recording sequences retain the entire source recording and exact repetition
  start/end boundary labels. Their phase labels are masked because a single
  weak displacement rule would be misleading across multiple repetitions.
- Repetition sequences slice each scored inclusive frame range. They carry the
  optional five-class ordinal target and leave repetition boundaries masked so
  a one-repetition clip does not create a trivial start/end target.

The internal expert target classes are zero-based `0..4`, corresponding to the
raw source score range `1..5`. A score list is averaged before rounding; a
scalar score is accepted and recorded with `expert_quality_rater_count: 1`.
The raw score, source range, exercise ID, side, position, and source member
remain in every prepared metadata row.

The four existing quality heads are still deliberately unavailable when their
specific labels are absent. The new head is named `expert_quality_logits` and
is reported separately as `weak_composite` quality supervision.

Exercise IDs are mapped conservatively into the existing broad movement
families. The original exercise ID and name are always retained, so this
mapping can be replaced by an exercise-specific taxonomy later.

## Corrected v2 layout

```text
/Users/devk/AdaptFit/
  data/raw/ucophyrehabpp/
  data/processed-v2-quality/
  artifacts/v2-quality/
  training/configs/v2_quality.yaml
  run_v2_overnight.sh
  scripts/verify_ucophyrehabpp.sh
```

The existing paths are not overwritten:

```text
data/processed-corrected-v1/
artifacts/corrected-v1/
data/processed/
artifacts/
```

The v2 feature memmaps use float16 storage to keep the duplicated prepared
tree practical on the development machine. `PreparedWindowDataset` converts
each sample back to float32 before it reaches PyTorch. Labels, masks, and
tracking targets retain their explicit dtypes.

## Prepared data verification

The completed non-training preparation produced:

| Item | Result |
| --- | ---: |
| Source sequences before synthetic variants | 7,315 |
| Training/validation/test windows | 267,440 / 16,483 / 7,536 |
| Feature shape per frame | 283 |
| Storage dtype for features | float16 |
| UCO usable sequences | 2,311 |
| UCO scored repetitions | 1,918 |
| UCO exact-boundary recordings | 393 |
| Dimension-specific quality sequences | 0 |
| Expert-quality sequences | 1,918 |
| Train/validation/test participant groups | 69 / 15 / 15 |
| Identity collisions | 0 in every split |
| Test contains UCO | yes |
| Test contains UL-RED | yes |
| Test contains wheelchair-position examples | yes |
| Real target-population validation | no |

The prepared audit is saved at
`/Users/devk/AdaptFit/artifacts/v2-quality/prepared_audit.json`. The source
checksums are recorded in
`/Users/devk/AdaptFit/data/manifests/ucophyrehabpp.md5`.

## Overnight execution

The script runs checksum verification, legacy identity auditing, v2 preflight,
preparation, participant/identity auditing, compilation, and non-training
contract tests. It then runs a two-epoch smoke pass followed by full TCN and
GRU training and evaluation when launched normally:

```bash
cd /Users/devk/AdaptFit
./run_v2_overnight.sh
```

To skip the short smoke pass and start the configured full run after the
verification stages:

```bash
ADAPTFIT_RUN_SMOKE=0 ./run_v2_overnight.sh
```

To repeat only the preparation and verification stages without any model
training:

```bash
ADAPTFIT_RUN_TRAINING=0 ./run_v2_overnight.sh
```

If the v2 prepared tree already exists and only the checks or training need to
be repeated, set `ADAPTFIT_SKIP_PREPARE=1`. The normal overnight command is
safe to use from a fresh checkout because it regenerates only the v2 output
roots.

## Outputs after training

When the overnight training completes, it will write model-specific outputs
under `artifacts/v2-quality/`:

```text
checkpoints/tcn_best.pt
checkpoints/gru_baseline.pt
metrics/tcn_history.json
metrics/gru_history.json
metrics/tcn_evaluation.json
metrics/gru_evaluation.json
metrics.json
predictions/
plots/
model_card.md
```

Both model reports use the same corrected test split and include family,
phase, repetition boundary, repetition-count, tracking, and optional expert
quality metrics. The four dimension-specific quality metrics remain explicitly
marked unavailable. All results must be described as public-data research
benchmarks, not clinical validation.

## What this still cannot prove

The UCO source improves label quality for controlled healthy-person
rehabilitation movements, but it does not prove that the model adapts safely
for an amputee, a person with a congenital limb difference, a person with one
available arm or leg, or a wheelchair user. Those claims require consented,
participant-level recordings from each target profile, exact movement and phase
labels, expert review, and a held-out target-population test set.
