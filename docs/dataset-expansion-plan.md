# AdaptFit dataset expansion and ingestion plan

> **Documentation metadata**
> - **Status:** research-backlog
> - **Authority:** research notes and proposed acquisition work
> - **Last verified:** 2026-09-07
> - **Source commit:** `ba8bf1a`
> - **Owner:** AdaptFit data research
> - **Supersedes or supports:** supports future acquisition decisions; dataset-catalog and current-state override availability and training inclusion
> - **Review trigger:** source access/license verification, adapter completion, or change in the canonical data protocol

This document is the working plan for expanding AdaptFit's training data. It
records the additional datasets researched after the initial MM-Fit integration,
what each source can teach the system, the access and licensing blockers, and
the implementation work needed to make the data useful without contaminating
the labels or participant-level splits.

> **Non-authoritative research note:** this file records proposals and dated
> ingestion observations. It must not be used to decide whether a source is
> currently available, licensed, integrated, or included in a benchmark. Use
> [dataset-catalog.md](dataset-catalog.md) for the current registry and
> [current-state.md](current-state.md) for artifact status. Any count below is
> valid only for the run/date and source commit named with it.

The central rule is that a large dataset is not automatically useful for the
temporal model. AdaptFit consumes a sequence of camera-pose-like observations,
but many valuable sources provide images, RGB video, optical motion capture,
IMUs, EMG, force plates, or anatomy models instead. Each source is therefore
assigned a role before any data is downloaded or merged.

## Current status

The current project now has working adapters for REHAB24-6, IntelliRehabDS,
MM-Fit, and UL-RED. The local MM-Fit archive is integrated into the canonical
schema, and its workout-to-participant mapping prevents repeated sessions from
leaking across the participant split.

The current prepared data contains:

- 5,004 canonical sequences before synthetic augmentation.
- 11,986 sequence entries after training-only augmentation.
- 160,813 training windows, 6,333 validation windows, and 6,463 test windows.
- 283 features per frame, including anatomy-derived angles, velocities,
  confidence, observation masks, capability context, and position context.
- No dimension-specific quality labels yet. Quality heads must remain masked
  until a source supplies ROM, tempo, smoothness, or compensation annotations.

All ten UL-RED subject archives are now staged under `data/raw/ul_red/` and
their SHA-256 checksums are recorded in `data/manifests/ul_red.sha256`. The
adapter reads the archives without requiring the raw data to be extracted into
the repository. The current parse produces 439 clean canonical sequences
across ten subjects: 44 sequences for each of S01 through S09 and 43 for S10.
There are 299 standing and 140 seated sequences, with the source's normal and
normal/fast/slow recording protocols preserved as metadata.

The one-epoch TCN smoke artifact reaches sequence-level movement-family
macro-F1 0.814, phase macro-F1 0.551, boundary F1 0.577, and repetition-count
MAE 0.276 on the held-out test split. These figures are a reproducibility
baseline for the expanded data, not a final overnight result or clinical
validation.

## What the model actually needs

The highest-value data has several of these properties:

- Time-ordered 2D or 3D joint positions at a known sampling rate.
- Repeated movements or transitions with start/end timestamps.
- A stable subject identifier for participant-level splitting.
- Seated, wheelchair, unilateral, prosthetic, or residual-limb examples.
- Expert or clearly defined labels for observable errors.
- Explicit visibility or absence information, rather than silently missing
  coordinates.
- A license that permits the intended research or product use.

Data that lacks those properties can still be valuable, but it must be routed
to the pose front end, self-supervised pretraining, anatomy prior, or evaluation
track rather than being presented as direct repetition supervision.

## Dataset roles

### Direct temporal training

These sources can become canonical sequences after a coordinate and label
adapter. They are candidates for movement-family, phase, repetition, tempo, or
quality training.

### Pose-front-end training

These sources primarily contain still images, video, boxes, or keypoints. They
can improve the camera model's ability to see wheelchairs, prostheses, residual
limbs, occlusions, and unusual body topology. They do not directly train the
TCN or GRU until framewise poses are extracted and temporal labels exist.

### Biomechanics and sensor pretraining

These sources contain motion capture, IMU, EMG, kinetics, or anatomical shapes.
They are useful for auxiliary representation learning, joint-angle sanity
checks, compensation priors, and a future sensor-fusion branch. They must not
be treated as camera data or used to claim camera-model accuracy.

### Evaluation and safety evidence

These sources are best kept as held-out challenge sets. They can reveal where a
pose model hallucinates a missing joint, where a wheelchair user is mistaken
for a standing user, or where a prosthesis is incorrectly interpreted as a
biological limb.

## Limb difference, amputation, and prosthetic movement

### ROAG

[Imperial College project page](https://www.imperial.ac.uk/manipulation-touch/open-source/dataset/roag-dataset/)
and [Zenodo record](https://zenodo.org/records/13908725)

ROAG contains reach-to-grasp motion across a 7-by-7 target grid from seven
able-bodied participants and two transradial prosthesis users. Bracing was also
used to simulate restricted wrist motion. It is the closest discovered source
to the first missing-arm profile because it contains compensatory arm and torso
motion rather than only appearance labels.

Use it for reach geometry, trunk compensation, asymmetry, and a
capability-conditioned reach evaluator. It is motion-capture data, so the
adapter must map its joint names and coordinate system into the canonical
layout. The Zenodo record indicates CC BY 4.0; preserve attribution.

### InclusiveVidPose

[Project page](https://anonymous-accept.github.io/inclusivevidpose/) and the
[ICLR paper and data-use terms](https://openreview.net/pdf?id=SyQqXAdWUq)

InclusiveVidPose reports 313 videos, more than 327,000 annotated frames, and
398 individuals with amputations, congenital limb differences, or prosthetic
limbs. Its 25-keypoint representation includes residual-limb endpoints,
boxes, masks, tracking IDs, and prosthesis information.

Use it for pose-front-end fine-tuning and absent-versus-unobserved anatomy
handling. The data-use agreement restricts it to approved, non-commercial,
ethical research and prohibits product development without written permission.
It should not be downloaded into the product training set until the custodians
approve the intended use.

### LDPose

[Research project](https://akitaraphael.github.io/ldpose/) and [ICCV open-access
paper](https://openaccess.thecvf.com/content/ICCV2025/html/Ying_LDPose_Towards_Inclusive_Human_Pose_Estimation_for_Limb-Deficient_Individuals_in_ICCV2025_paper.html)

LDPose reports 28,065 images and approximately 72,716 people, with 25
keypoints: the standard body points plus eight residual-limb endpoints. It
covers upper-limb, lower-limb, and bilateral deficiencies in sports, training,
and daily-life imagery.

This is a high-priority pose-front-end source because the annotation protocol
explicitly represents non-existent joints. The project page exposes model and
dataset access links, but the access terms need to be recorded before use. It
does not provide temporal repetitions by itself.

### ProPose

[Hugging Face dataset card](https://huggingface.co/datasets/Soralink/ProPose)
and [official code](https://github.com/SoraLink/ProPose)

ProPose contains approximately 34,692 annotated images in train, validation,
and test splits and uses an Omni-Pose topology. Each keypoint is labeled as
biological, prosthetic, or physically absent. The dataset card lists an
approximately 20 GB archive and a nonstandard license field, so the exact
terms must be reviewed before acquisition.

Use it to train an anatomy-aware pose front end and to evaluate whether a
detector hallucinates an elbow, wrist, knee, or ankle that does not exist. Keep
its official splits and do not convert still images into fake temporal labels.

### ProGait

[Official repository](https://github.com/pittisl/ProGait), [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/html/Yin_ProGait_A_Multi-Purpose_Video_Dataset_and_Benchmark_for_Transfemoral_Prosthesis_ICCV_2025_paper.html),
and [Hugging Face data release](https://huggingface.co/datasets/ericyxy98/ProGait)

ProGait contains 412 walking clips from four above-knee amputees using
multiple prosthetic legs. It includes 2D keypoints in NumPy arrays, confidence
scores, masks, CVAT annotations, and inside/outside parallel-bar scenarios.

Use it for lower-limb prosthesis pose robustness, gait-state representation,
and a held-out test of the missing-knee or missing-ankle handling. It is not an
exercise-repetition dataset, so it should not receive invented curl, squat, or
rep-boundary labels.

### Open dataset of above-knee amputee stand-up and sit-down biomechanics

[Dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11903848/)

This source contains kinetics, kinematics, EMG, and video recordings from nine
above-knee amputees during stand-up and sit-down. The data can inform
sit-to-stand transitions, asymmetric loading, and trunk compensation.

The source is high value for anatomy-informed priors, but the raw modality is
motion capture and laboratory biomechanics rather than phone pose. Inspect the
repository license and any raw-video restrictions before staging it.

### Open 18-person above-knee amputee gait dataset

[Scientific Data record](https://pmc.ncbi.nlm.nih.gov/articles/PMC7242470/)

This open biomechanics source contains full-body data from 18 people with
unilateral above-knee amputations walking at several speeds. It is useful for
prosthesis-side asymmetry, intact-side compensation, and gait phase priors.
The data should be routed through a gait/biomechanics adapter and held out from
exercise-quality claims.

### Syrian above-knee amputee gait dataset

[Dataset record](https://data.mendeley.com/datasets/k5y9jkx87y/1) and [data
descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC8449166/)

This dataset contains 14 unilateral above-knee amputees and 20 healthy
participants, with gait parameters, lower-limb angles, moments, and ground
reaction forces. It is useful for geographic and device-domain variation, but
it is not a phone-camera pose dataset.

### Thirty transtibial prosthesis-user kinematics dataset

[Figshare record](https://doi.org/10.6084/m9.figshare.25698006.v1) and [data
descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC11344789/)

This source contains optical-motion-capture and IMU-derived kinematics from 30
transtibial prosthesis users walking in a laboratory. It provides joint angles,
raw marker trajectories, and IMU-derived angles.

Use it to improve lower-limb asymmetry and prosthesis-side priors. The adapter
should keep the source in a gait-only task family or auxiliary loss rather than
forcing it into the five first-release exercise families.

### GaitIntent

[Scientific Data article](https://www.nature.com/articles/s41597-026-07799-8)

GaitIntent provides inertial kinematics from 11 participants, including one
transtibial amputee, and includes steady-state and transitional locomotion. The
study also uses a hands-free crutch to approximate unilateral lower-limb loss
in some healthy participants. The data are numerical, de-identified streams
with consent for open academic use.

Use it for transition detection and future lower-limb motion-intent
pretraining. It is not a substitute for real amputee exercise recordings.

### GAITEX

[Scientific Data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC12775516/),
[Zenodo record](https://zenodo.org/records/15729055), and [processing
repository](https://github.com/ai-for-sensor-data-analytics-ulm/aisd_ortho_ki_dataset)

GAITEX contains synchronized optical motion capture and IMU recordings from 19
participants. It includes rehabilitation exercises with correct and clinically
relevant incorrect variants, plus normal and impaired-gait tasks. The release
contains marker trajectories, processed IMU data, and timestamp files that
identify repetition or walking-condition intervals.

This is one of the strongest candidates for a future quality adapter because
the incorrect variants are defined by the protocol rather than invented from a
generic fitness score. First audit the exact exercise taxonomy and source
license, then map only the supported error dimensions to AdaptFit quality
targets. Keep the source in an exercise-quality track separate from the
disability-profile split.

### K2MUSE

[Official project page](https://k2muse.github.io/), [dataset page](https://k2muse.github.io/datasets/),
and [paper](https://arxiv.org/abs/2504.14602)

K2MUSE contains lower-limb kinematics and kinetics from 30 young and 12 older
adults walking at multiple speeds and inclines. It also synchronizes force
plates, surface EMG, and amplitude-mode ultrasound. The data are laboratory
biomechanics rather than phone-camera pose, but the varied speeds, inclines,
age groups, and acquisition interference are valuable for lower-limb phase and
tempo representation learning.

Use the kinematic subset for anatomy and phase sanity checks. Do not merge EMG,
ultrasound, or force-plate channels into the camera feature vector unless a
separate sensor-fusion model is explicitly introduced.

### Kuopio gait dataset

[Zenodo record](https://zenodo.org/records/10559504) and [data
descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC11385067/)

Kuopio contains 51 participants with 3D marker trajectories, ground-reaction
measurements, six lower-body IMUs, and sagittal-plane OpenPose keypoints from
walking trials. The release includes participant metadata, timestamps, and
multiple data modalities for the same trials.

The OpenPose trajectories make this a useful camera-pose parity and confidence
stress-test source. The full archive is large, so stage the OpenPose and
participant metadata first; add the raw motion-capture and IMU archives only
when a sensor-baseline experiment is scheduled.

### MyPredict / MyLeg

[University of Twente dataset page](https://research.utwente.nl/en/datasets/roessingh-research-development-myleg-database-for-activity-predic/)

MyPredict contains kinematics and surface EMG from 55 able-bodied subjects over
85 measurement sessions. Participants transition freely between gait-related
activities, which makes the source useful for activity recognition and
transition modeling. It is not a disability dataset and does not provide the
camera pose representation needed by the primary model.

Use the kinematics as an auxiliary lower-limb transition source only after
reviewing the download terms. Keep the EMG in a separate modality branch.

### CeTI-Age-Kinematics

[Scientific Data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC11954993/),
[Figshare record](https://figshare.com/articles/dataset/The_i_CeTI-Age-Kinematics_i_dataset_A_Full-Body_IMU-Based_Motion_Dataset_of_Daily_Tasks_by_Older_and_Younger_Adults/26983645)

CeTI-Age-Kinematics contains full-body IMU motion from 32 participants split
between older and younger adults. It covers 30 daily tasks in nine categories,
including reaching, object interaction, walking, and a five-repetition
sit-to-stand protocol. The records include anthropometry, task variations,
procedures, repetitions, and BIDS-organized files.

Use it for age-related tempo and movement-variability pretraining, seated-to-
standing transitions, and a held-out domain test. It does not represent limb
absence or wheelchair propulsion.

### Open multi-sensor gait dataset

[Figshare record](https://figshare.com/articles/dataset/A_multi-sensor_and_cross-session_human_gait_dataset_captured_through_an_optical_system_and_inertial_measurement_units/14727231)

This CC BY 4.0 dataset contains 25 healthy participants, 500 trials across two
sessions, 42 full-body optical markers, a smartphone accelerometer, and a
wearable IMU. It is useful for cross-session robustness and for checking how
much information is lost when a rich motion-capture signal is reduced to a
camera-like 2D observation.

Use the optical marker trajectories to build a controlled camera-projection
benchmark. Split by participant and session; never place both sessions of one
participant in different evaluation groups.

### DUO-GAIT

[Scientific Data descriptor](https://www.nature.com/articles/s41597-023-02391-w),
[PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC10442385/), and [code
repository](https://github.com/HPI-CH/fatigue-dual-task-data)

DUO-GAIT contains four six-minute walking conditions from 16 healthy adults:
single-task and cognitive dual-task walking, each before and after fatigue.
Nine IMUs cover the head, trunk, wrists, legs, and feet, and the release also
contains spatiotemporal gait parameters and fatigue-related measurements.

Use it for robustness to fatigue, divided attention, pace variability, and
upper/lower-body coordination. It is an auxiliary sensor dataset, not direct
evidence for wheelchair, amputation, or exercise-quality performance.

### NONSD-Gait

[Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.2rbnzs7z3)

NONSD-Gait contains 23 healthy adults walking under non-standardized dual-task
conditions with optical motion capture, a depth camera, and IMUs. It is useful
for camera-domain shift, depth-versus-RGB pose comparison, and transition
robustness in less controlled protocols.

Treat it as a held-out sensor-transfer source until its labels and license are
reviewed. It does not supply target-population examples.

### FineRehab

[CVPR Workshop paper](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Li_FineRehab_A_Multi-modality_and_Multi-task_Dataset_for_Rehabilitation_Analysis_CVPRW_2024_paper.pdf)

FineRehab combines two Azure Kinect cameras with 17 IMUs for rehabilitation
movements and crops long recordings into repetition-level clips. It is a
promising multi-view and sensor-fusion source for exercise segmentation, but
the exact distribution package, participant split, and usage terms must be
verified before acquisition.

Do not use its IMUs as labels for a camera-only model. If the skeleton export is
available, adapt it as an auxiliary temporal source and preserve the original
repetition clips.

### Comprehensive full-body IMU activity dataset

[Scientific Data descriptor](https://www.nature.com/articles/s41597-026-06710-9)

This newer dataset records 30 participants performing 12 daily activities with
17 full-body IMUs. It is useful for sensor-layout ablations, missing-sensor
robustness, and pretraining a future optional IMU branch. It should not be
counted as exercise or disability data.

### Powered-prosthesis and exoskeleton datasets from the University of Michigan

[Locomotor Control Systems Laboratory data page](https://locolab.robotics.umich.edu/data.html)

The page provides several motion-capture and OpenSim datasets, including
seven above-knee amputees in sit/stand and walking experiments, powered
prosthesis hip-compensation data from three participants, and able-bodied
locomotion with transitions, ramps, stairs, and crouching.

These are valuable for biomechanical sanity checks and compensation modeling.
They should be acquired one study at a time because each associated dataset
has its own publication and licensing conditions.

### OpenLimbTT

[Open-source repository](https://github.com/abel-research/OpenLimbTT)

OpenLimbTT is not movement data. It is a statistical, privacy-preserving model
of transtibial residual-limb anatomy built from MRI and CT-derived shape data.
The released mean and virtual patient shapes can be used to make anatomically
constrained synthetic residual-limb geometries.

Use it only for anatomy priors, visualization, and synthetic pose/front-end
stress tests. It cannot validate movement quality or clinical outcomes.

## Wheelchair and seated movement

### SAFER-Activities

[Official project page](https://safer-activities.github.io/) and [dataset
card](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities)

SAFER-Activities reports more than 66 hours of untrimmed video from 46
participants, frame-level labels for 30 action classes, a wheelchair subset,
2D and lifted 3D poses, boxes, frozen visual features, and subject/view splits.
It also provides a wheelchair keypoint image subset.

This is one of the best wheelchair pose and activity sources. Use the official
subject and view splits, and reserve its non-lab test set as an out-of-domain
evaluation. It requires access approval and is CC BY-NC-SA 4.0. Falls and
routine activities are not exercise-quality labels.

### WheelPose and Users in Wheelchairs

[Official repository](https://github.com/hilab-open-source/wheelpose)

WheelPose contains a synthetic wheelchair-person data generator and a UIW
image set of approximately 2,464 annotated RGB images from 84 public videos in
16 action classes. The full image set is available by request; the code and
the source images have different legal conditions.

Use it for wheelchair detection, occlusion stress tests, and pose-front-end
training. It is not direct temporal supervision.

### WheelPoser-IMU

[Official repository](https://github.com/axle-lab/WheelPoser) and [paper](https://arxiv.org/abs/2409.08494)

WheelPoser-IMU contains 167 minutes of paired sparse IMU and motion-capture
data from wheelchair users, including propulsion and pressure relief. It is a
strong future sensor-fusion and wheelchair-motion-prior source, but it cannot
be mixed into a camera-pose feature tensor without an explicit sensor branch.

### Stroke rehabilitation exercise data with Kinect and IMU

[NIAID data record](https://data.niaid.nih.gov/resources?id=mendeley_ygpdzx52g2)

This dataset reports 128 participants, 631 Kinect skeleton files, two-IMU
recordings, performance scores, and five exercises involving arm lifting,
trunk tilt, trunk rotation, pelvis rotation, and squatting. The portal lists
the data under CC BY 4.0, but the source package and exact skeleton parser
must be checked before ingestion.

This is a promising bridge between rehabilitation exercise and sensor fusion.
It needs a parser for the `.skeleton` format and a careful review of the
performance-score definitions before its labels enter the quality heads.

### WheelArm Synchronized Dataset

[Dataset card](https://huggingface.co/datasets/Cordelia/WheelArm_WoZ_Pilot_Dataset)

WheelArm contains 53 episodes from five subjects performing five assistive
daily-living tasks with a wheelchair-mounted robot arm. It includes RGB,
depth, robot kinematics, wheelchair base states, IMU, audio, and dialogue.

Use it for future assistive-task context and camera/robot interaction research,
not for human exercise labels. It is approximately 47 GB and should remain a
separate multimodal project.

## Stroke, motor impairment, and clinical quality

### StrokeRehab

[Official dataset page](https://strokerehabdata.github.io/dataset.html)

StrokeRehab contains 3,372 trials from 51 stroke-impaired and 20 healthy
participants, with 120,891 annotated functional primitives such as reach,
transport, reposition, stabilize, and idle. It provides IMU and video-derived
data, with access through SimTK.

Use it for primitive recognition, unilateral upper-body movement, and
compensation representation learning. It is not a direct camera-pose source in
the current release form.

### Toronto Rehab Stroke Pose Dataset

[Dataset reference and code](https://github.com/zhiderek/TRSPD) and [Kaggle
record](https://www.kaggle.com/datasets/derekdb/toronto-robot-stroke-posture-dataset)

This source contains 25-joint Kinect pose estimates from stroke survivors and
healthy participants performing robot-assisted tasks, with posture-compensation
labels. It is well matched to a trunk-compensation head, subject to Kaggle
terms and source availability.

### KIMORE

[Paper and dataset access](https://doi.org/10.1109/TNSRE.2019.2923060)

KIMORE contains RGB-D and skeleton recordings for five rehabilitation
exercises, with 78 subjects and physician-derived features and clinical scores.
It is a high-value quality source, but access and reuse terms must be confirmed
before staging or commercial use.

### KERAAL

[Official dataset page](https://keraal.enstb.org/KeraalDataset.html)

KERAAL contains low-back-pain rehabilitation recordings from nine healthy
subjects and twelve patients, with Kinect 3D skeletons, RGB, OpenPose or
BlazePose 2D poses, and physician annotations for correctness, error type,
body part, and error timespan. It is CC BY-NC-SA.

Use it to train observable error localization and trunk compensation for
research. Do not silently use the labels as clinical diagnoses.

### University of Idaho UI-PRMD

[Open data descriptor](https://doi.org/10.3390/data3010002)

UI-PRMD contains ten healthy subjects performing ten repetitions of ten
physical-therapy movements with Vicon and Kinect capture. It provides joint
positions and angles and is described as public domain under the Open Data
Commons PDDL.

Use it as a clean skeleton benchmark for repetition structure, alignment, and
cross-dataset evaluation. It has no target-group participants, so it cannot
validate capability-aware adaptation.

### UCO Physical Rehabilitation

[Paper and repository](https://github.com/AVAuco/ucophyrehab)

UCO Physical Rehabilitation contains 2,160 videos from 27 people performing
eight rehabilitation exercises from multiple RGB views with OptiTrack ground
truth. The authors describe access by email and research-purpose request.

Use it for pose-extractor evaluation and viewpoint robustness after the
maintainers confirm file format and license.

### Upper-limb stroke rehabilitation exercise video dataset

[Open-access data article](https://www.sciencedirect.com/science/article/pii/S2352340926003719)

This newer video source reports 491 videos covering four upper-limb
strengthening exercises performed by ten volunteers at 30 FPS. It is useful for
viewpoint, lighting, and temporal exercise classification, but the population
is not a target disability group and the released annotations must be checked.

### Sensor comparison for upper-body stroke rehabilitation

[DataverseNO record](https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/4EWI6I)

This replication dataset contains motion data from 16 healthy participants
performing upper-body movements relevant to stroke rehabilitation and range of
motion assessment. The record provides a 1.2 GB download.

Use it for sensor/pose cross-modal checks and ROM feature validation. It is not
stroke-participant evidence.

## General fitness, repetition, and movement quality

### Fit3D

[Official project page](https://fit3d.imar.ro/) and [exercise and annotation
details](https://fit3d.imar.ro/fit3d)

Fit3D contains 611 multi-view sequences, more than 2.9 million ground-truth
3D skeletons and meshes, 47 exercise types, four camera views, and per-
recording repetition segmentations. The public download page lists an 18 GB
training set and a 1.4 GB test set and requires an account.

This is the strongest general source for repetition boundaries and 3D pose
pretraining. Use it to learn exercise mechanics, then evaluate separately on
seated, unilateral, and limb-difference sources.

### FLEX

[Paper and dataset project](https://arxiv.org/abs/2506.03198)

FLEX reports more than 7,500 multiview recordings of 20 weight-loaded actions
from 38 subjects across three skill levels, with synchronized RGB, 3D pose,
sEMG, physiological signals, and a fitness knowledge graph connecting actions,
keysteps, error types, and feedback.

This is the best candidate discovered for interpretable fitness quality
pretraining. The full access and license terms must be checked. It is not a
disability dataset and should not be used to define what a disabled user's
movement ought to look like.

### FLAG3D

[Official project page](https://andytang15.github.io/FLAG3D/)

FLAG3D reports 180,000 sequences across 60 fitness and daily-activity
categories, including MoCap skeletons, SMPL data, natural-scene video,
rendered video, and sentence-level exercise instructions. The full dataset is
distributed after a signed license agreement; an 1,800-video rendering subset
is described on the project page.

Use it for broad movement-family and instruction-conditioned pretraining. Its
access agreement and synthetic/rendered subsets must be tracked separately.

### Countix and QUVA Repetition

[Countix paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.pdf)
and [RepNet release information](https://research.google/blog/repnet-counting-repetitions-in-videos/)

Countix contains in-the-wild repetition videos with counts and periodic
segments across workouts, sports, music, tools, and other repetitive actions.
It is useful for repetition-counting pretraining under camera motion and speed
changes, but its source videos are not canonical pose and may have separate
copyright conditions.

### Fitness-AQA

[Research repository](https://github.com/bk225/4990-Fitness-AQA)

Fitness-AQA focuses on quality assessment for back squat, overhead press, and
barbell row. It is useful for row and trunk-quality representation learning
after access approval, but it is video-first and non-commercial terms should be
assumed until verified.

### MEx

[Dataset paper](https://arxiv.org/abs/1908.08992)

MEx is a multimodal exercise dataset with numerical time series, video, and
pressure-sensor data for human activity recognition and exercise quality
assessment. It is useful for a future sensor-fusion branch and for studying
whether pressure changes add information that camera pose cannot observe.

### Olympic action-quality dataset

[Dataset and precomputed pose release](https://people.csail.mit.edu/hpirsiav/quality.html)

This action-quality source provides Olympic sports video, annotations, and
precomputed tracked human pose. It is not rehabilitation data, but it can
pretrain generic quality-ranking and pose-temporal encoders. Keep it as a
research-only auxiliary source because sports quality is not the same as safe
exercise quality.

### TaiChi-AQA

[Research article](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.70053)

TaiChi-AQA is a fine-grained action-quality source covering 24 Tai Chi
postures with action labels and multiple quality scores. It can teach smoothness,
tempo, and sequence-level quality ranking, but its movement semantics and
population differ from AdaptFit's first release.

### Exercise ontology and instruction data

[RepDB exercise dataset](https://github.com/sergei-argutin/exercise-dataset)

This is not a movement recording dataset. It provides exercise names,
instructions, target muscles, equipment, goals, tags, MET values, and
illustrations. Its stated free tier is usable in commercial apps with
attribution. It can seed AdaptFit's exercise/capability recipe catalog, but it
must never be treated as evidence about how a body with a limitation should
move.

## General anatomy, pose, and occlusion pretraining

### KIT Whole-Body Human Motion Database

[Official database](https://motion-database.humanoids.kit.edu/)

KIT provides whole-body motion capture and object-interaction sequences with a
unifying human-body representation. It is useful for anatomy-aware motion
priors, but not for disability-specific validation.

### JRDB-Pose

[Official pose release](https://jrdb.erc.monash.edu.au/dataset/pose)

JRDB-Pose provides approximately 600,000 pose annotations with track IDs and
explicit visibility states for occluded keypoints. Use it for generic occlusion
and temporal tracking stress tests, not as rehabilitation or exercise-quality
data.

### Human Data Corpus

[Open repository](https://humandatacorpus.org/)

The Human Data Corpus contains several open multimodal human-motion and
ergonomics collections, including kinematics, kinetics, and EMG. It can supply
anatomy and ergonomic risk priors, but each subcollection must be evaluated
individually for relevance and license.

### H3WB and standard human pose sets

[Human3.6M WholeBody extension](https://github.com/wholebody3d/wholebody3d)

H3WB extends standard human pose with body, feet, face, and hand keypoints. It
is valuable for improving landmark coverage and cross-skeleton mapping, but it
contains intact bodies and does not model limb absence.

## Data acquisition backlog

### Acquire and adapt now

- Use the ten staged UL-RED subject archives and the AMC parser. Start with
  marker-less clean AMC files because they are the closest to a camera-pose
  input and have known 30 FPS timing.
- Keep the acquisition reproducible with
  `scripts/download_ul_red.sh`; validate the downloaded archives against
  `data/manifests/ul_red.sha256` before conversion.
- Keep the source manifest and preflight check for UL-RED. Require at least
  three subjects before enabling it in the default combined training run.
- Adapt GAITEX next if the Zenodo package and terms expose the timestamped
  exercise files. It is the best newly found source for protocol-defined
  incorrect movement variants.
- Request or download Fit3D if the account and research terms are acceptable.
  Start with the skeleton and repetition-segmentation files rather than all
  videos.
- Request ROAG and ProGait access for amputee and prosthetic evaluation. Keep
  them in a separate limb-difference evaluation split until the adapter is
  reviewed.
- Request SAFER-Activities access and use its wheelchair subject split as a
  dedicated pose/front-end benchmark.

### Adapt after access approval

- InclusiveVidPose, LDPose, and ProPose: extend the pose schema with residual
  endpoint and topology labels, then export pose-front-end training shards.
- KERAAL, KIMORE, Toronto Rehab Stroke Pose, and the new stroke exercise
  sources: map clinician labels into only the quality dimensions they actually
  measure.
- WheelPoser-IMU, StrokeRehab, K2MUSE, GaitIntent, and the prosthesis
  biomechanics sources: build a separate sensor/biomechanics representation
  task rather than concatenating non-camera signals into v1.

### Keep as auxiliary or held out

- FLEX, FLAG3D, Countix, Fitness-AQA, and the Olympic action-quality set are
  useful for broad quality and repetition pretraining, but should not dominate
  the disability-aware fine-tuning mix.
- OpenLimbTT is an anatomy prior, not a movement dataset.
- RepDB is exercise metadata, not evidence about human movement.
- General pose and motion sources such as KIT, JRDB-Pose, H3WB, and Human Data
  Corpus should be used to improve visibility and mapping, not to create
  disability labels.

## Implementation work

### 3D Mocap to 2D Canonical Virtual Camera Projection Specification

Several high-fidelity biomechanical datasets (ROAG, UI-PRMD, Pipelines, Arm-CODA, AddBiomechanics) provide 3D Cartesian coordinates in millimeters $(X, Y, Z)$ rather than monocular 2D camera coordinates $(x, y) \in [0, 1]^2$. To integrate these datasets into AdaptFit's 283-feature pipeline without corrupting the camera-space input contract, every 3D mocap adapter must pass coordinates through a standardized virtual camera projection pipeline:

#### 1. Pinhole Perspective Camera Model
For a 3D joint coordinate in world frame $\mathbf{P}_w = [X_w, Y_w, Z_w]^T$:
$$\mathbf{P}_c = \mathbf{R}_{c} (\mathbf{P}_w - \mathbf{T}_c)$$
where:
- $\mathbf{T}_c = [0, h_{cam}, -d_{cam}]^T$ positions the virtual camera at a realistic standing smartphone tripod distance $d_{cam} \in [1.8\text{m}, 2.4\text{m}]$ and elevation $h_{cam} \in [0.9\text{m}, 1.2\text{m}]$ from the subject.
- $\mathbf{R}_c = \mathbf{R}_x(\phi_{tilt})$ applies a downward camera tilt angle $\phi_{tilt} \in [5^\circ, 15^\circ]$, matching typical real-world mobile device placement.

The 2D projected sensor coordinates $(u, v)$ in pixels are computed via the camera intrinsics matrix $\mathbf{K}$:
$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \mathbf{K} \mathbf{P}_c = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$
with standard mobile parameters: image resolution $W=1080$, $H=1920$, focal length $f_x = f_y \approx 800\text{px}$, and principal point $(c_x, c_y) = (W/2, H/2)$.

#### 2. Normalization to Canonical 2D Contract
The projected pixel coordinates $(u, v)$ are mapped directly into the $[0, 1]$ normalized image frame:
$$x = \frac{u}{W}, \quad y = \frac{v}{H}$$
To match AdaptFit's `training/src/features/anatomy.py` normalization:
- **Hip-Centered Origin**: Subtract the midpoint between left and right hip landmarks:
  $$x_{norm} = x - \frac{x_{left\_hip} + x_{right\_hip}}{2}, \quad y_{norm} = y - \frac{y_{left\_hip} + y_{right\_hip}}{2}$$
- **Torso-Length Scale Invariance**: Normalize distances by the Euclidean torso scale:
  $$S_{torso} = \|\mathbf{P}_{mid\_hip} - \mathbf{P}_{mid\_shoulder}\|_2$$
- **Velocity Derivation**: Compute finite-difference temporal derivatives:
  $$v_x(t) = \frac{x(t) - x(t-1)}{\Delta t}, \quad v_y(t) = \frac{y(t) - y(t-1)}{\Delta t}$$
- **Confidence & Capability Masking**: Compute camera-space observability $observed \in \{0.0, 1.0\}$ strictly from marker availability and camera view frustum: set $observed = 1.0$ only if the marker is present/finite in the mocap trajectory AND the projected pixel coordinates lie within the sensor viewport ($0 \le u \le W$, $0 \le v \le H$, and depth $Z_c > 0$); set $observed = 0.0$ for occluded, dropped, or out-of-frame markers. Apply canonical profile capability weights $w_{capability} \in \{1.0 \text{ (available)}, 0.8 \text{ (assisted)}, 0.65 \text{ (limited)}, 0.5 \text{ (unknown)}, 0.0 \text{ (absent)}\}$ per the feature contract in [contracts-and-schemas.md](contracts-and-schemas.md) (never restricted to binary $\{0.0, 1.0\}$). Compute joint confidence as camera observability multiplied by capability weight ($confidence = observed \cdot w_{capability}$ clipped to $[0.0, 1.0]$), preserving the `FeatureSchemaV1` ABI.

---

### Adapter Specifications for Priority Ingestions

#### 1. DynTherapy Adapter (`training/src/data/adapters.py#load_dyntherapy`)
- **Source Layout**: 33 MediaPipe Pose landmarks $(x, y, z, visibility)$ at 30 FPS.
- **Mapping**: Exact 1:1 joint index mapping (Joints 0–32 correspond identically to MediaPipe standard).
- **Target Mapping**:
  - `family`: Map 7 exercises into AdaptFit families (knee raises $\rightarrow$ `single_leg_march`, seated shoulder press $\rightarrow$ `overhead_press`, arm curls $\rightarrow$ `arm_curl`).
  - `rep_start` / `rep_end`: Map explicit source "Start" and "End" frame annotations directly to boundary heads.
  - `phase`: Synthesize temporal phases between boundaries: concentric (start to peak velocity), hold (apex inflection), eccentric (descent), rest (inter-cycle stillness).
- **Masks**: Quality heads masked (`-1`), boundary and family heads fully active (`1`).

#### 2. UI-PRMD Adapter (`training/src/data/adapters.py#load_uiprmd`)
- **Source Layout**: Vicon 3D optical marker trajectories across 10 physical therapy movements.
- **Mapping**: Transform 39 Vicon markers to canonical 33 joints via anatomical centroid estimation; apply the 3D-to-2D virtual camera projection.
- **Target Mapping**:
  - `family`: Map exercises (e.g. seated sit-to-stand, shoulder abduction).
  - `rep_start` / `rep_end`: Map repetition cycle boundaries where verified segmentation timestamps exist.
  - `quality_logits`: Kept masked (`-1` with `quality_mask = 0.0`). UI-PRMD provides overall binary correctness ("optimal" vs. "non-optimal"), which must NOT be mapped directly into dimension-specific ROM (`quality[:, 0]`) or trunk compensation (`quality[:, 3]`) heads. Per the feature contract and line 844, correctness scores must never become ROM or trunk quality; dimension-specific heads remain zero-coverage until verified sub-dimension annotations exist.
- **Masks**: `quality_mask` strictly masked (`0.0`) across all 4 dimension-specific heads; boundary and family heads active (`1.0`) where verified.

#### 3. Pipelines Wheelchair Adapter (`training/src/data/adapters.py#load_pipelines`)
- **Source Layout**: Synchronized 8-camera markerless video paired with 14-camera Vicon optical mocap.
- **Mapping**: Ingest markerless 2D pose keypoints directly, with 3D optical mocap serving as numerical verification.
- **Target Mapping**:
  - `family`: `seated_mobility`.
  - `phase`: Map propulsion push phase to `concentric` and recovery phase to `eccentric`.
  - `boundary`: Mark push contact onset as `rep_start` and push release as `rep_end`.
- **Masks**: Set capability masks for legs to $0.0$ when wheelchair propulsion restricts lower-body movement.

#### 4. Ottobock #DearAI Visual Ingestion Pipeline
- **Source Layout**: Curated community video clips of upper-limb and lower-limb amputees.
- **Mapping**: Feed frames through MediaPipe 33-keypoint detector. Compare extracted landmarks against ground truth residual limb endpoint annotations.
- **Target Mapping**:
  - Does not emit temporal repetition labels. Emits `PoseFrame` robustness shards.
  - Calibrates absent-limb capability weighting vectors ($w_c$) to verify that zero confidence on absent joints does not degrade torso tracking.

#### 5. ROAG Transradial Ingestion Adapter (`training/src/data/adapters.py#load_roag`)
- **Source Layout**: 3D optical marker time series of 7 controls and 2 transradial amputees across 2,450 reaching trajectories.
- **Mapping**: Apply 3D-to-2D virtual camera projection. Set intact arm $w_c = 1.0$, transradial arm $w_{c, wrist} = 0.0$ (or prosthesis flag).
- **Target Mapping**:
  - `family`: `forward_reach`.
  - `quality_logits[:, 3]`: Binary trunk compensation logit. To prevent a schema mismatch between ROAG's continuous trunk tilt angle $\theta_{trunk}$ and the binary pooled classification head (`quality_logits[:, 3]`), $\theta_{trunk}$ must be thresholded against a validated biomechanical compensation threshold (e.g. compensatory trunk lean $\theta_{trunk} \ge \tau_{trunk} = 15^\circ \implies 1$, normal $\implies 0$). Continuous angle values must never be directly placed into the binary quality logit; continuous regression is reserved for a future schema version.
  - `boundary`: Reach initiation $\rightarrow$ `rep_start`, target contact $\rightarrow$ `rep_end`.
- **Masks**: `quality_mask[:, 3]` active (`1.0`) only when thresholded compensation is evaluated; other quality dimensions remain masked (`0.0`).

#### 6. SERE / TRSPD Stroke Adaptation Adapter (`training/src/data/adapters.py#load_sere`)
- **Source Layout**: ZED 3D skeletons + Kinect v2 25-joint skeletons for post-stroke hemiparetic patients.
- **Mapping**: Map Kinect 25 joints to canonical 33 joints by interpolating hip/torso midpoints.
- **Target Mapping**:
  - `quality_logits[:, 3]`: Therapist-graded frame-level trunk lean and shoulder hiking annotations mapped directly to binary compensation threshold $\ge 1$.
  - `quality_logits[:, 0]`: Range of motion deficit scores thresholded to binary ROM deficit quality logit (or held masked if continuous, preserving the binary schema of `MovementPredictionV1`).
  - `expert_quality_logits`: Therapist composite score (1–5 ordinal).

### Canonical adapters

Every adapter must emit the existing `CanonicalSequence` contract and record:

- source dataset and exact source file;
- participant, session, and trial identifiers;
- coordinate convention and source frame rate;
- original exercise/action label;
- label provenance and source-specific label masks;
- camera visibility versus anatomical absence;
- whether a sequence is real, synthetic, or a transformed modality.

The adapter must never turn “one clip” into individual repetition boundaries,
turn a correctness score into ROM quality, or turn a prosthesis into an intact
biological joint. Unsupported targets remain `-1` or masked.

### UL-RED adapter design

The first code addition in this expansion is the UL-RED adapter:

- read `S01.zip` through `S10.zip` without extracting raw archives;
- prefer `marker-less/clean/*.amc` files;
- parse the AMC positional frames and use the accompanying ASF only for audit
  metadata when present;
- map the UL-RED `Waist`, `Neck`, shoulders, elbows, wrists, hands, hips,
  knees, and ankles to the 33-joint canonical layout;
- infer a broad movement family from the exercise name;
- preserve the repetition number and pace (`normal`, `fast`, `slow`) in
  metadata;
- use one-recording boundaries only when an explicit repetition interval is
  present; otherwise mask boundaries;
- retain normal/fast/slow as a tempo-condition field, not as a quality label;
- test an archive with multiple subjects and prove that subject IDs do not
  cross splits.

### Pose-front-end data format

Image and video sources should be exported separately as shards containing:

```text
image_or_video_id
frame_index
keypoints_xy
keypoint_confidence
keypoint_visibility
anatomical_status
prosthesis_status
person_id
source_dataset
license_tag
```

`anatomical_status` must distinguish `present`, `absent`, `residual_endpoint`,
and `unknown_visibility`. That distinction is required for missing-limb
robustness.

### Evaluation matrix

The expanded evaluation should report results by source and profile:

- intact standing;
- seated;
- wheelchair;
- unilateral upper-limb limitation;
- unilateral lower-limb limitation;
- prosthetic limb;
- occluded but anatomically present joints;
- absent joints represented by explicit capability masks.

Every report must include participant-level splits, label coverage, calibration,
abstention rate, and failure examples. Public-data results are not clinical
validation.

## Storage and compute planning

At the latest preparation run, the processed arrays occupy about 25 GiB and
the ten UL-RED subject archives occupy about 5.3 GB combined. The GAITEX
Zenodo API lists a single 17.66 GB `data.zip`, so it is intentionally not
staged on this machine until processed-data storage is moved or expanded.
Fit3D, ProPose, SAFER-Activities, QEVD, and WheelArm can also require large
storage or controlled access.

Do not download every candidate at once. Use this order:

- UL-RED skeleton archives and code.
- Fit3D skeleton/repetition files if approved.
- ROAG, ProGait, and SAFER-Activities access requests.
- Clinician-labeled sources after license review.
- Video-first or sensor-first sources only when a specific pretraining task is
  defined.

Raw data must remain outside version control. Store checksums, source URLs,
download date, license text, and adapter version in the manifest.

## Acceptance tests for this expansion

- UL-RED AMC parsing produces finite canonical coordinates and monotonically
  increasing timestamps.
- Multiple UL-RED subject archives produce distinct participant groups.
- Pace metadata survives conversion and is not used as a correctness label.
- Explicit missing joints remain masked rather than interpolated as normal
  anatomy.
- Prosthesis and residual-limb labels are retained as metadata or front-end
  targets rather than collapsed into “bad pose.”
- No source-specific participant appears in more than one split.
- All source adapters report label coverage before training.
- A source with no quality labels leaves all four quality dimensions masked.
- A source with only set-level or clip-level counts does not receive invented
  frame-level boundaries.
- The combined data preparation remains reproducible from the manifest and
  configuration.
- The expanded TCN and GRU checkpoints remain loadable on CPU and MPS.

## Definition of done

This expansion is complete for a source only when the source has a recorded
license/access decision, a checksum, a canonical adapter or explicit reason it
cannot be adapted, participant-safe splits, source-specific tests, and a
metrics entry that identifies exactly which labels were evaluated.

The target-population gap remains a data-collection problem. Public data can
make the pose front end and temporal model stronger, but it cannot replace
consented recordings from people with one-arm use, limb absence, prostheses,
or wheelchair use performing the exact AdaptFit routines.
