# Verified Research and Run Findings

This document consolidates the AdaptFit decisions, data research, training
pipeline findings, and the corrected analysis of the latest saved run.

Last verified: 2026-09-01

The detailed acquisition notes and the larger research longlist remain in the
[dataset expansion and ingestion plan](dataset-expansion-plan.md). This
document records the conclusions that should guide implementation.

## Product direction

AdaptFit is intended to adapt fitness routines around the movements a person
can safely perform. Many existing fitness applications assume that a user can
stand, run, jump, or use both arms. AdaptFit should use an explicit capability
profile and movement observations to select an appropriate exercise variant.

The initial target profiles are:

- A missing or unavailable arm, including one-arm users.
- A missing or unavailable leg, including one-leg users.
- Wheelchair and seated users.

Users should self-report available limbs, limitations, position, equipment,
and comfortable movement during onboarding. The camera model should not infer
or diagnose a disability from appearance.

The first release should support these seated and unilateral-friendly
exercises:

- Seated one-arm biceps curl.
- Seated one-arm band row.
- Seated single-leg knee extension.
- Seated single-leg march or hip lift.
- Seated forward reach.

Sit-to-stand and wall push-ups are deferred because they require weight-bearing
or bilateral capability that many target users may not have.

All future AdaptFit implementation belongs under `/Users/devk/AdaptFit/`.
The upstream [PeddieHacks26 repository](https://github.com/DeviDeviDeviKanumilli/PeddieHacks26)
is reference material only and is not being copied into `/app` or preserved as
a nested application.

## Current data reality

There is no direct access yet to consented recordings from the target groups.
Public datasets and synthetic augmentation can bootstrap engineering, but they
cannot establish that the product works safely for amputees, people with limb
differences, or wheelchair users.

The current prepared data contains:

- 5,004 canonical sequences before synthetic augmentation.
- 11,986 sequence entries after training-only augmentation.
- 160,813 training windows, 6,333 validation windows, and 6,463 test windows.
- 283 normalized features per frame.
- 4,388 sequences with phase labels.
- 1,811 sequences with boundary labels.
- Zero sequences with dimension-specific quality labels.

The ten UL-RED subject archives are staged under
`data/raw/ul_red/`. Their checksums are recorded in
`data/manifests/ul_red.sha256`. The current parser produces 439 canonical
sequences across ten subjects, with 299 standing and 140 seated sequences.
UL-RED preserves normal, fast, and slow recording metadata but does not supply
the expert quality labels required for the quality heads.

## Machine-learning design

### Shared anatomy-informed model

The scalable production direction is one shared causal temporal encoder across
exercises and capability profiles. Exercise-specific requirements and the
user's capability profile condition the model and rule layer; a separate neural
network should not be created for every disability or exercise.

The primary model is a causal dilated TCN:

- Input width: 283 features.
- Input projection: 96 channels.
- Five causal residual blocks.
- Kernel size: 3.
- Dilations: 1, 2, 4, 8, and 16.
- Dropout: 0.1.
- Window: 128 frames at approximately 30 FPS.
- Causal receptive field: 125 frames, or about 4.2 seconds.
- Actual parameter count in the saved v1 model: 307,410.

The GRU is retained as a baseline:

- One causal layer.
- Hidden size: 64.
- Actual parameter count in the saved baseline: 68,178.

Approximate weight sizes are 1.2 MB for the TCN and 0.27 MB for the GRU in
float32. Int8 weights would be approximately 0.31 MB and 0.07 MB respectively,
before runtime and metadata overhead. Pose estimation, camera buffers, and
runtime memory are separate from these temporal-model sizes.

The model outputs:

- Movement family.
- Per-frame movement phase.
- Repetition start and end.
- Range-of-motion quality.
- Tempo quality.
- Smoothness quality.
- Trunk-compensation flag.
- Tracking observability confidence.

The quality heads must remain masked unless a source provides reliable labels.
The tracking-confidence head is a self-supervised observability signal; it is
not clinical uncertainty and is not movement quality.

### Anatomy-informed features

The canonical feature representation includes:

- Translation-normalized joint coordinates.
- Scale-normalized segment vectors.
- Joint velocities.
- Fourteen anatomy-defined joint angles.
- Angular velocities.
- Pose confidence.
- Camera-observed masks.
- Explicit capability masks.
- Four-limb capability context.
- Seated or wheelchair position context.
- Left/right asymmetry where both sides are available.
- Trunk posture and compensation indicators.

The capability mask distinguishes available, limited, assisted, absent, and
unknown-visibility states. A missing limb must not be represented as a normal
limb with a low-confidence camera observation. The model must not infer a
disability from appearance.

### Training labels

The intended labels are:

- Movement family: arm curl/flexion, row/pull, knee extension, hip flexion or
  march, reach, and other/unknown.
- Phase: unknown, rest, concentric, hold, and eccentric.
- Repetition start and end.
- Observable quality: ROM, tempo, smoothness, and trunk compensation.

When a source does not provide a label, that loss is skipped. A set-level count,
one-clip boundary, generic correctness score, or procedural template must not be
silently converted into a frame-level clinical or quality label.

### Training settings

- Optimizer: AdamW.
- Learning rate: `3e-4`.
- Weight decay: `1e-4`.
- Batch size: 64, reduced to 32 only if memory requires it.
- Maximum epochs: 100.
- Early-stopping patience: 15.
- Gradient clipping: 1.0.
- Seed: 42.
- Float32 training.
- Apple MPS when available, CPU fallback otherwise.
- DataLoader workers: 0 on macOS.

Loss weights are family 0.5, phase 1.0, boundary 2.0, quality 1.0, and
self-supervised tracking 0.2.

## Training pipeline

The intended reproducible sequence is:

```bash
cd /Users/devk/AdaptFit

python3 -m training.preflight \
  --config training/configs/v1.yaml

python3 -m training.prepare_data \
  --config training/configs/v1.yaml

python3 -m training.train \
  --config training/configs/v1.yaml \
  --models tcn,gru \
  --device auto \
  --seed 42

python3 -m training.evaluate \
  --config training/configs/v1.yaml \
  --checkpoint artifacts/checkpoints/tcn_best.pt
```

The stages are:

- Preflight checks source presence, licenses, required data roles, and the
  feature contract.
- Preparation maps each source into canonical joints, normalizes coordinates,
  resamples to 30 FPS, preserves masks, creates windows, and writes
  participant-level splits.
- Training fits the shared TCN and GRU using masked multi-task losses.
- Evaluation computes window-level diagnostics and sequence-level metrics after
  merging overlapping windows.
- Artifact checks record the configuration, feature schema, normalization
  statistics, checkpoints, plots, and model card.

Synthetic augmentation is training-only. It can simulate joint occlusion,
dropped frames, landmark noise, reduced ROM, changed tempo, left/right
asymmetry, opposite-limb masking, and limited visibility. It cannot validate
real amputee, limb-difference, or wheelchair movement.

Splits must be participant-level. Synthetic variants remain in the same split as
their original participant. Random frame splitting is prohibited.

## Dataset research conclusion

### The exact-match question

The requested ideal dataset would contain:

- Many unique participants in every target profile.
- 50–100 repetitions per exercise per participant.
- Video or pose sequences with exact repetition and phase labels.
- Expert labels for ROM, tempo, smoothness, and compensation.
- A participant-level test set containing every target profile.

No single public dataset found meets all of these requirements. The largest
gap is the combination of real limb-difference or wheelchair participants,
high repetition volume, exact phase labels, and expert multidimensional quality
labels.

The most useful sources are complementary rather than interchangeable:

| Source | Strongest contribution | Important gap |
|---|---|---|
| [UCOPhyRehab++](https://pmc.ncbi.nlm.nih.gov/articles/PMC13365497/) and its [Zenodo release](https://zenodo.org/records/17935737) | Exact repetition start/end frames, 3D pose, and two physiotherapist scores based on ROM, postural control or compensation, and execution velocity | 27 participants, controlled mostly healthy population, roughly four repetitions per exercise in the original protocol, no wheelchair or limb-absence profiles, and no direct smoothness label |
| [NIAID stroke rehabilitation data](https://data.niaid.nih.gov/resources?id=mendeley_ygpdzx52g2) | 128 participants, 631 Kinect skeleton files, five exercises, and clinician performance scores | Scores are not documented as exact frame-level phase, boundary, ROM, tempo, smoothness, and compensation labels |
| [FineRehab](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Li_FineRehab_A_Multi-modality_and_Multi-task_Dataset_for_Rehabilitation_Analysis_CVPRW_2024_paper.html) | 50 participants, including 30 musculoskeletal patients and 20 healthy participants, Kinect and 17-IMU data, 16 actions, and a clinician-informed quality framework | Not focused on amputees or wheelchair users; exact per-repetition label mapping and access terms require an audit |
| [StrokeRehab](https://strokerehabdata.github.io/dataset.html) | 51 stroke-impaired and 20 healthy participants, 3,372 trials, video/IMU data, and high-resolution functional primitive labels | Primitive labels are not the same as exercise repetitions, phase classes, or ROM/tempo/smoothness/compensation scores |
| [UL-RED](https://datacat.liverpool.ac.uk/2729/) | 22 rehabilitation exercises, ten subjects, markerless and marker-based pose, and normal/fast/slow conditions | Small population, one- or three-repetition recordings, no target profiles, and no expert multidimensional quality labels |
| [KERAAL](https://keraal.enstb.org/KeraalDataset.html) | Kinect skeleton, RGB, 2D pose, and physician labels for correctness, error type, body part, and error span | Three low-back-pain exercises; not limb absence or wheelchair data; CC BY-NC-SA terms |
| [KIMORE](https://doi.org/10.1109/TNSRE.2019.2923060) | Five physician-selected rehabilitation exercises, skeleton data, physician-derived features, and clinical scores | Access and reuse terms require confirmation; labels are not the complete requested quality vector |
| [Toronto Rehab Stroke Pose](https://github.com/zhiderek/TRSPD) and [Kaggle record](https://www.kaggle.com/datasets/derekdb/toronto-robot-stroke-posture-dataset) | Stroke and healthy seated Kinect movements with expert compensation annotations | Small cohort and insufficient repetition volume; no limb-absence or wheelchair target coverage |
| [ROAG](https://zenodo.org/records/13908725) | 2,450 reaching trajectories from seven able-bodied participants and two transradial prosthesis users | Motion-capture reach data, not exercise repetitions or expert quality labels |
| [ProGait](https://github.com/pittisl/ProGait) | 412 video clips from four transfemoral prosthesis users with 2D pose | Walking/gait only; no exercise repetition or quality labels |
| [SAFER-Activities](https://safer-activities.github.io/) | More than 66 hours of video, wheelchair recordings, 2D/3D poses, timestamps, and subject/view splits | Activity segments rather than exercise repetition and expert quality labels |
| [WheelPose](https://github.com/hilab-open-source/wheelpose) | Wheelchair-person pose generation and approximately 2,464 RGB images from 84 public videos | Image-level pose data, not temporal exercise data |
| [InfiniteRep](https://paperswithcode.com/dataset/infiniterep) | Synthetic exercise videos with frame-level repetition counts and high quantity | Synthetic only; no real target population or expert quality labels |
| [SERE](https://arxiv.org/abs/2506.03752) | Post-stroke exercise and compensation-label direction that may be useful for targeted research | Reported trial volume is far below 50–100 repetitions per exercise per participant; access and exact label package need verification |
| [Upper-limb stroke exercise videos](https://data.mendeley.com/datasets/49h9dcwx5v/1) | 491 videos, four upper-limb exercises, 30 FPS, one cycle per video, and complete/incomplete labels | Ten healthy volunteers rather than stroke participants; no expert four-dimensional quality labels |
| [REHAB sensor dataset](https://doi.org/10.1038/s41597-026-07802-2) | 120 post-stroke participants and standardized assessment/training movements | IMU and flex sensors rather than video or camera pose; no exact camera repetition/phase labels |
| [PrimSeq / NYU stroke data](https://datacatalog.med.nyu.edu/dataset/10595) | 41 chronic stroke participants, cameras, IMUs, functional primitives, and clinical information | Not the requested exercise-repetition and quality-label package |

### What each source should do

- UCOPhyRehab++ should supply quality-head pretraining and exact boundary
  supervision.
- NIAID, StrokeRehab, Toronto Rehab, and SERE should supply impaired-motion or
  compensation pretraining where their labels are actually supported.
- UL-RED should continue to bootstrap seated/standing temporal modeling and
  pace variation.
- ROAG and ProGait should support transradial and lower-limb prosthesis pose or
  compensation evaluation.
- SAFER-Activities and WheelPose should support wheelchair pose robustness and
  held-out profile evaluation.
- InfiniteRep should be used only for synthetic quantity and augmentation
  experiments.

The broader researched longlist includes REHAB24-6, IntelliRehabDS, MM-Fit,
UI-PRMD, Fitness-AQA, MM-Fi, QEVD, GAITEX, K2MUSE, Kuopio, MyPredict/MyLeg,
CeTI-Age-Kinematics, DUO-GAIT, NONSD-Gait, OpenLimbTT, WheelArm, Fit3D, FLEX,
FLAG3D, Countix/QUVA, MEx, Olympic action quality, TaiChi-AQA, RepDB, KIT,
JRDB-Pose, Human Data Corpus, H3WB, the University of Michigan locomotor
datasets, and additional prosthesis and gait sources. Their roles and links are
recorded in the [dataset expansion plan](dataset-expansion-plan.md).

### What still has to be collected

Public data can bootstrap the model, but the exact first-release benchmark
requires a purpose-built, consented study with:

- One-arm or upper-limb-difference participants.
- One-leg or lower-limb-difference participants.
- Wheelchair and seated participants.
- The five supported exercises.
- Repetition and phase timestamps.
- Expert review of ROM, tempo, smoothness, and trunk compensation.
- A participant-level test cohort containing every profile.

The 50–100 repetition target is a research target, not a property of the
currently available public datasets. Recruitment could eventually involve
rehabilitation centers, prosthetics clinics, adaptive sports organizations,
physical therapists, and wheelchair-user organizations. Consent, institutional
review, privacy, and participant safety must be established before collection.

## Latest saved run

### What the original artifacts report

The saved TCN checkpoint is
[`artifacts/checkpoints/tcn_best.pt`](../artifacts/checkpoints/tcn_best.pt).
The saved evaluation is in
[`artifacts/metrics.json`](../artifacts/metrics.json), with the detailed TCN
copy in [`artifacts/metrics/tcn_evaluation.json`](../artifacts/metrics/tcn_evaluation.json).
The supporting records are the [TCN history](../artifacts/metrics/tcn_history.json),
[GRU history](../artifacts/metrics/gru_history.json),
[feature schema](../artifacts/feature_schema.json),
[training configuration](../artifacts/training_config.yaml), and
[prepared-data index](../data/processed/index.json).

The TCN history records:

- MPS device.
- 307,410 parameters.
- Best epoch: 35.
- Last recorded epoch: 50.
- Best validation sequence score: 0.618893.
- Training loss: 0.488970 at epoch 1 and 0.207971 at epoch 50.

The run ending at epoch 50 is consistent with the best epoch at 35 plus the
configured patience of 15.

The original evaluator reports 434 sequence groups. That grouping is not the
correct logical-sequence benchmark because distinct source clips are merged.

### Corrected logical-sequence evaluation

The saved TCN window predictions were re-aggregated using the complete source
metadata identity. This produces 745 logical test entries and exactly matches
the independently reported corrected results:

| Metric | Corrected result |
|---|---:|
| Movement-family accuracy | 700/745 = 94.0% |
| Movement-family macro-F1 | 89.7% |
| Phase frame accuracy | 88.6% |
| Phase macro-F1 | 61.9% |
| Hold-phase F1 | 1.1% |
| Repetition-start F1 | 100.0% |
| Repetition-end F1 | 29.3% |
| Quality-label coverage | 0.0% |

The phase accuracy is computed only on 62,547 labeled phase frames. It is not
accuracy over every frame in every test window. The hold F1 exposes the severe
class imbalance and weak learning of the hold phase.

The 100% repetition-start F1 is not meaningful production evidence because
starts are largely placed at frame zero after each clip is treated as a logical
sequence. Repetition-end F1 is the more informative boundary result and is not
production-ready.

The original 434-group report is still useful for identifying the saved
artifact, but it must not be used as the final logical-sequence benchmark. Its
reported sequence-level family result is 400/434 = 92.2% accuracy and 87.9%
macro-F1.

### Why the grouping was wrong

The current sequence-ID builder in
[`training/src/data/prepare.py`](../training/src/data/prepare.py) includes the
source, participant, session, video ID, repetition number, source frame range,
and family. However, some source-specific identifiers are nested in
`source_metadata` and are not included in the generated ID:

- IntelliRehabDS `gesture_id` is omitted.
- MM-Fit `set_index`, set frame range, activity, and other set identifiers are
  omitted.

The aggregation code in
[`training/src/evaluation.py`](../training/src/evaluation.py) correctly groups
by the supplied `sequence_id`; it is receiving an incomplete identity. The
current test metadata contains 434 generated IDs but 745 unique identities when
the full source metadata is included.

The proper fix is to create a source-stable logical identity in preparation,
include the required source-specific fields, regenerate prepared splits, and
rerun checkpoint selection and evaluation. Both TCN and GRU must be evaluated
with the same corrected identity logic before making a final comparison.

### Split and label findings

- Participant groups do not overlap between train, validation, and test.
- The test split contains no UL-RED sequences.
- The test split contains no wheelchair-position sequences.
- Validation contains UL-RED and wheelchair examples, but this does not make
  the test benchmark representative of those profiles.
- 5,029 of 6,463 test windows have unknown phase labels.
- 5,029 of 6,463 test windows have unknown boundary labels.
- All four quality targets are unlabeled.
- No quality model performance can be claimed from this run.

The saved artifacts therefore measure public-data movement classification and a
limited phase/boundary task, not target-population adaptation or clinical form
quality.

### TCN versus GRU

The TCN clearly outperformed the saved GRU baseline under the original
evaluation setup. The saved GRU sequence results are approximately 64.1%
family macro-F1, 46.0% phase macro-F1, 55.4% boundary F1, and 8.76 repetition
count MAE. The comparison was rerun after the sequence-ID fix; the corrected
results below use the same logical test entries and checkpoint-selection
metric for both models.

### Tests and integrity checks

The original implementation passed 103 tests. The corrected implementation
adds identity, split, reporting, and artifact-isolation regressions; the full
suite now passes with 110 tests. The checks include:

- Canonical feature shapes and finite values.
- Missing and occluded joint behavior.
- Dropped frames and low-confidence poses.
- Session reset and exercise-change behavior.
- Participant split leakage.
- Model save/load correctness.
- One-batch training and evaluation.
- Streaming runtime behavior.
- UL-RED archive parsing and checksum validation.

Passing tests establish engineering integrity, not clinical validity.

## Corrected benchmark rerun

The corrected rerun is isolated from the original data and artifacts:

- Prepared data: `/Users/devk/AdaptFit/data/processed-corrected-v1/`
- Artifacts: `/Users/devk/AdaptFit/artifacts/corrected-v1/`
- Configuration: `/Users/devk/AdaptFit/training/configs/v1_corrected.yaml`
- Runner: `/Users/devk/AdaptFit/run_corrected_overnight.sh`

The shared identity version is `adaptfit.sequence.v2`. It includes source-
specific clip, gesture, set, repetition, and frame-range fields. The legacy
audit still reproduces `745` corrected logical entries and `700 / 745` correct
family predictions. The corrected prepared data has `5,004` source sequences,
`11,986` entries including training-only synthetic variants, and `0` identity
collisions in every split.

The corrected participant split contains `50` training groups, `11`
validation groups, and `11` test groups with no overlap. The test set contains
`617` logical sequences and `7,592` windows, including `44` UL-RED sequences
and `52` wheelchair-position sequences. Wheelchair-position data is a
public-data proxy; the run contains no real amputee, limb-difference, or
wheelchair-user participant recordings.

The final fresh models ran on MPS. TCN selected epoch 35 after 50 epochs and
has 307,410 parameters. GRU selected epoch 34 after 49 epochs and has 68,178
parameters. Sequence-level metrics are primary because overlapping windows
are correlated:

| Metric | TCN | GRU |
|---|---:|---:|
| Family accuracy | 91.25% | 89.47% |
| Family macro-F1 | 87.57% | 84.37% |
| Phase frame accuracy | 82.48% | 82.25% |
| Phase macro-F1 | 58.26% | 54.63% |
| Hold F1 | 31.25% | 0.00% |
| Repetition-start F1 | 99.54% | 83.50% |
| Repetition-end F1 | 18.47% | 5.53% |
| Repetition-count MAE | 0.31 | 30.63 |

The phase metrics use `64,064` labeled sequence-level frames. Quality-label
coverage is `0.0%` for both models, so ROM, tempo, smoothness, and
trunk-compensation performance is unavailable. The low repetition-end scores
mean the temporal boundary task is not production-ready even though TCN is a
stronger research candidate than GRU on this split.

The model-specific reports, combined report, predictions, and traceable
sequence metadata are under
`/Users/devk/AdaptFit/artifacts/corrected-v1/`. These are public-data
research and synthetic-robustness results, not clinical validation.

## Required next work

### Use the corrected benchmark

- Keep the corrected run in `data/processed-corrected-v1/` and
  `artifacts/corrected-v1/`; do not overwrite the original benchmark.
- Treat the sequence-level corrected metrics as the current engineering
  baseline, not as clinical or target-population evidence.
- Use the staged UCOPhyRehab++ release through the isolated v2 quality run,
  subject to license and label audit. It supplies exact repetition spans and a
  separate composite expert-quality target; it does not fill the four
  dimension-specific quality heads.

### Add useful supervision

- The v2 adapter for UCOPhyRehab++ is implemented and prepared in
  `data/processed-v2-quality/`; run `run_v2_overnight.sh` for training and
  evaluation.
- Audit NIAID, FineRehab, StrokeRehab, Toronto Rehab, KERAAL, and KIMORE
  licenses and label definitions.
- Keep source-specific quality labels masked when a source does not measure the
  requested dimension.
- Add source-aware and profile-aware metrics rather than one pooled score.
- Collect real target-population recordings before claiming profile support.

### Prepare deployment later

- Quantize the validated float32 TCN to int8.
- Create golden feature vectors for Python/mobile parity.
- Rewrite the mobile runtime directly in the root project.
- Run pose extraction and temporal inference on-device.
- Add capability-aware exercise recipes, safety rules, abstention, and manual
  override behavior.

## Claims that are currently allowed

It is accurate to say that the current TCN is a promising research prototype
for movement-family classification on the available public-data split. It is
also accurate to say that it beats the saved GRU baseline in that prototype
experiment.

It is not accurate to claim:

- Clinical validation.
- Reliable form or movement-quality assessment.
- Reliable hold-phase recognition.
- Production-ready repetition-end detection.
- Generalization to amputees, people with limb differences, or wheelchair users.
- Safety for the first-release target profiles.

All results should be labeled as public-data and synthetic-robustness results,
not clinical validation.
