# Data and Training Plan

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** preparation code, dataset manifests, training configuration, and approved label policy
> - **Last verified:** 2026-09-07
> - **Source commit:** `e75ba65`
> - **Owner:** AdaptFit data and training engineering
> - **Supersedes or supports:** canonical data/label protocol; supports dataset-catalog, efficient-training-strategy, and contracts-and-schemas
> - **Review trigger:** adapter, label mask, split, normalization, license, or retention change

## Current data constraint

There is no direct access to target-group participants for consented recordings. This means the first model can be a well-engineered prototype, but it cannot yet provide strong evidence for amputees, people with limb differences, or wheelchair users.

## Public data bootstrap

Potential starting sources include:

- [UI-PRMD](https://pmc.ncbi.nlm.nih.gov/articles/PMC5773117/): rehabilitation movement recordings with correct and non-optimal examples.
- [KIMORE](https://doi.org/10.1109/TNSRE.2019.2923060): rehabilitation exercises with skeleton data and clinical-quality context.
- [IntelliRehabDS](https://zenodo.org/records/4610859): skeleton-based rehabilitation gestures including seated, wheelchair, and standing contexts.
- [REHAB24-6](https://zenodo.org/records/13305826): exercise recordings with skeleton data, repetitions, segmentation, and correctness information.
- [MM-Fit](https://mmfit.github.io/): synchronized exercise pose data with exercise-set spans and repetition counts; the current adapter uses its pose and family labels while masking unavailable phase and per-repetition targets.
- [Fitness-AQA](https://github.com/ParitoshParmar/Fitness-AQA): exercise-quality data for squats, overhead press, and barbell row.
- [WheelPose](https://github.com/hilab-open-source/wheelpose): wheelchair pose robustness data; useful for tracking robustness but not a complete exercise-quality dataset.
- [SAFER-Activities](https://safer-activities.github.io/): wheelchair-related pose and activity data; useful for robustness and context, not necessarily for repetition labels.

Dataset terms, research restrictions, and redistribution rights must be checked before any data is used in training or shipped with the project.

## Preprocessing pipeline

Each source should be converted into a canonical sequence format:

- Load pose or skeleton data.
- Map source joints to the canonical body-joint layout.
- Resample to the configured frame rate and interpolate only short gaps; long
  gaps remain masked.
- Normalize translation and scale using torso or pelvis landmarks.
- Compute angles, segment vectors, velocities, and optional accelerations.
- Preserve visibility, confidence, and missing-joint masks.
- Attach exercise, phase, repetition, and quality labels where available.
- Store participant and session identifiers for subject-level splitting.

Each prepared window also records label provenance. For example, a source
correctness label can be distinguished from a weak displacement-derived phase
label, a single-clip repetition boundary assumption, or a procedural seed
label. By default, the single-clip assumption and procedural quality labels are
masked; they can be enabled only for explicit ablation runs. Source correctness
is not projected into a specific ROM, tempo, smoothness, or trunk target.
Evaluation reports both overlapping-window
metrics and sequence-level metrics after merging windows from the same
sequence.

Raw source data should remain outside the tracked application repository when licensing or privacy requires it.

## Synthetic augmentation

Synthetic transformations can help train missing-data behavior and improve robustness:

- Mask one arm or one leg.
- Simulate partial camera occlusion.
- Remove or corrupt individual joints.
- Reduce or expand range of motion.
- Change tempo.
- Add left/right asymmetry.
- Apply seated posture transformations where biomechanically reasonable.
- Add realistic landmark noise and dropped frames.

These transformations are useful for engineering and pretraining. They do not substitute for real target-user movement patterns or safety validation.

## Labels

Useful labels include:

- Exercise identifier and movement family.
- Participant and session identifier.
- Repetition start and end.
- Movement phase.
- Valid, modified, incomplete, or uncertain movement.
- Approximate ROM.
- Tempo.
- Smoothness.
- Trunk compensation.
- Left/right asymmetry.
- Camera view and visibility quality.
- Equipment and position.
- User comfort or difficulty when ethically and safely collected.

Labels for unsafe or clinically inappropriate movement should come from qualified reviewers. Do not ask participants to intentionally perform dangerous form for the sake of collecting negative examples.

## Training strategy

Recommended sequence:

1. Train a simple deterministic baseline using the existing feature engine.
2. Train a small GRU baseline for movement phase and repetition detection.
3. Train the shared causal TCN using the same feature contract.
4. Add quality heads only where labels are reliable.
5. Use self-supervised temporal pretraining as the dataset grows.
6. Compare models using subject-level splits and on-device latency.

The **proposed pilot target** is approximately 15–25 participants, five
exercises, and 50–100 repetitions per exercise if a consented dataset becomes
available. This is a recruitment planning estimate, not a validated sample-size
or release gate. Diversity of participants matters more than collecting many
clips from the same person.

## Evaluation

Do not randomly split frames from the same person between training and test. Use participant-level or session-level splits.

Track at least:

- Repetition-count error.
- Repetition boundary F1.
- Phase-classification F1.
- Quality-classification precision/recall.
- Calibration of confidence and abstention.
- Performance under missing limbs, occlusion, and dropped frames.
- Per-profile performance for upper-limb, lower-limb, and wheelchair/seated cases.
- On-device latency and memory.

Until target-user data exists, results should be reported as public-data and synthetic-robustness results, not as validated performance for the target populations.

## Canonical ingestion lifecycle

Every source follows the same state machine:

1. **Candidate:** record the source citation, owner, terms, modality, and
   intended task in [dataset-catalog.md](dataset-catalog.md).
2. **Access review:** verify the license, download permissions, privacy/consent
   restrictions, redistribution rule, retention/deletion contact, and permitted
   research or product uses.
3. **Acquisition:** store raw data outside the tracked repository with a source
   version, checksum, download date, and immutable read-only copy where terms
   allow.
4. **Adapter:** map joints, timestamps, labels, participants, and units into
   the canonical schema. Record adapter version and source checksum in
   `DatasetManifestV1`.
5. **Quality review:** check finite values, timestamp order, joint-map coverage,
   label ranges, missingness, duplicate identities, and sample counts.
6. **Prepared:** write sequence/window manifests with lineage, masks, split
   assignment, and normalization-statistics version.
7. **Approved for a task:** enable only label roles whose provenance and license
   permit that task. Adapter completion does not approve every model head.
8. **Retired/deleted:** remove or quarantine data when a license, consent, or
   retention rule requires it and preserve a manifest tombstone without raw
   content.

## Identity, splits, and normalization

- `participant_id` is stable within a source and must not be regenerated per
  clip. If a source supplies no identity, mark it `unknown` and prohibit
  participant-level claims from that split.
- `session_id` distinguishes recordings by the same participant. Windows keep
  `source_id`, `participant_id`, `session_id`, sequence offsets, and augmentation
  lineage so overlaps cannot cross a split.
- Generate train/validation/test partitions by participant first, then enforce
  source-held-out partitions for transfer experiments. Lock the test manifest
  before tuning thresholds or selecting a checkpoint.
- Fit torso/pelvis normalization statistics on training participants only. Reuse
  the exact statistics for validation, test, replay, and deployment. A changed
  statistic is a new normalization version and invalidates direct checkpoint
  comparisons.
- Keep frame-rate conversion and interpolation settings in the manifest. Short
  gaps may be interpolated only under the configured limit; longer gaps remain
  observed-mask zeros.

## Label provenance and disagreement

Each label field carries a role and mask, for example `source_expert`,
`source_automatic`, `derived_weak`, `synthetic`, `procedural`, or `unknown`.
Losses activate only when the field’s task mask is valid. A source correctness
label is not silently converted into ROM, tempo, smoothness, or trunk-control
targets. A single-clip repetition assumption and procedural quality template
remain masked by default.

When annotators disagree, retain both annotations, reviewer IDs, timestamps,
and an adjudication status. Define the repetition boundary convention per
exercise before computing metrics. Ambiguous intervals may be masked or given
an explicit uncertain label; deleting disagreements hides a data problem.
Report agreement and unresolved coverage alongside model scores.

## Augmentation lineage

Every synthetic sample points to one real parent sequence and records the
transformation, seed, parameter bounds, and whether timestamps/boundaries were
transformed consistently. Augmentations may simulate occlusion, joint noise,
limited visibility, tempo, ROM, asymmetry, or dropped frames. They cannot create
target-population evidence, new participants, or labels for a task the parent
does not support. Validation and test remain unaugmented except for an explicit
robustness stress-test split.

## License, privacy, and retention gate

Before a dataset enters a training manifest, the owner must record license name,
access state (`verified`, `restricted`, `pending`, or `prohibited`), permitted
uses, attribution, redistribution rule, consent/IRB note where applicable,
raw-data location, retention/deletion deadline, and a contact for removal. A
dataset with pending terms may be used for local inspection only, not training
or shipping. Raw frames, raw pose, and participant identifiers stay outside the
application repository unless the terms explicitly allow otherwise.

## Prepared-data acceptance checklist

An adapter is ready only when a reviewer can reproduce:

- source checksum and adapter/config versions;
- participant/session counts and duplicate checks;
- canonical joint mapping and unit conversion;
- missingness and confidence distributions;
- label-role coverage and masks by task;
- participant/source split files;
- training-only normalization statistics;
- representative sequence/window fixtures;
- license/access and retention decision;
- a `DatasetManifestV1` path consumed by the experiment record.

The concise source registry is [dataset-catalog.md](dataset-catalog.md). Proposed
sources remain in [dataset-expansion-plan.md](dataset-expansion-plan.md) until
this checklist and the license gate pass.
