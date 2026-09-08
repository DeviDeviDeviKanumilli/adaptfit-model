# AdaptFit public dataset catalog

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** dataset registry, adapter code, manifests, and license/access evidence
> - **Last verified:** 2026-09-07
> - **Source commit:** `e75ba65`
> - **Owner:** AdaptFit data engineering
> - **Supersedes or supports:** concise canonical registry; detailed acquisition proposals live in dataset-expansion-plan
> - **Review trigger:** adapter or checksum change, license/access update, or label-role change

This catalog records public datasets that can help AdaptFit learn movement
recognition, temporal structure, observable movement quality, pose robustness,
and capability-aware adaptation. It was researched on 2026-08-31. A dataset
being public does not automatically grant permission to redistribute it, train
a commercial product on it, or use it for clinical claims. Every source needs a
license and ethics review before it is added to a released model.

The larger researched longlist, acquisition statuses, and adapter plan are in
the [dataset expansion and ingestion plan](dataset-expansion-plan.md).

The exact-match analysis for the requested target profiles, repetition volume,
phase labels, expert quality labels, and participant-level test coverage is in
the [verified research and run findings](verified-research-and-run-findings.md).
No single public dataset found so far satisfies all of those requirements.

## How to use this catalog

The current AdaptFit model consumes normalized 2D pose sequences with explicit
observed and capability masks. Datasets therefore fall into different roles:

- Direct temporal supervision can train the shared TCN or GRU.
- Quality and compensation labels can train the observable-quality heads.
- Image/keypoint datasets can improve the camera pose front end, but do not
  directly train the temporal model until pose sequences are extracted.
- IMU or motion-capture datasets can support auxiliary pretraining and physics
  checks, but cannot be treated as camera-pose data.

Participant or subject identifiers must stay intact so all splits remain
participant-level. Synthetic limb masking is useful for engineering tests, but
it is not evidence that a model works for a real limb difference or wheelchair
user.

## Canonical registry

The registry below is the source of truth for current role and availability.
`Integrated` means an adapter exists and a prepared path can be audited; it
does not mean every label or task is valid. `Pending` and `restricted` sources
must not enter a release artifact until the access gate passes. Checksums are
recorded in `data/manifests/` when present; “manifest pending” is an explicit
provenance gap.

| Dataset | Status / adapter | Modality and participant profile | Labels and usable tasks | Prohibited interpretation | License/access | Checksum/path | Next action |
|---|---|---|---|---|---|---|---|
| REHAB24-6 | Integrated / `rehab24_6` | 2D pose; general rehabilitation participants | Exercise family, repetition spans; family/phase/boundary experiments where source labels exist | Not target-population validation; no unprovided quality dimensions | Verify source terms before redistribution | `data/raw/rehab24_6/`; manifest pending | Record source checksum and task masks |
| IntelliRehabDS | Integrated / `intellirehabds` | Skeleton; seated, wheelchair, standing contexts | Gesture/family and clip boundaries; robustness/context | Not a repetition-quality or clinical outcome label | Verify access and redistribution | `data/raw/intellirehabds/`; manifest pending | Complete license review and participant split audit |
| MM-Fit | Integrated / `mmfit` | 2D/3D pose and RGB-D; general exercise participants | Family and set-level count; temporal context | No per-repetition boundaries or phase from released set CSV | Terms pending for product use | `data/raw/mmfit/`; manifest pending | Preserve count/set provenance; do not invent boundaries |
| UL-RED | Integrated / `ul_red` | Marker-less/mocap; general rehabilitation participants | Exercise, speed, sequence structure; phase/tempo pretraining | Not limb-difference or wheelchair evidence; no clinical safety claim | CC BY 4.0 as recorded; verify archive terms | `data/raw/ul_red/`; `data/manifests/ul_red.sha256` | Audit archive checksum and label map |
| UCOPhyRehab++ | Integrated / `ucophyrehabpp` | 3D pose; controlled healthy participants | Exact repetition spans and composite 1–5 expert score | Composite score is not four dimension-specific quality labels or target-population evidence | License review pending | `data/raw/ucophyrehabpp/`; `data/manifests/ucophyrehabpp.md5` | Finish v2 sequence evaluation and license decision |
| ROAG | Candidate / no adapter | Motion capture; seven able-bodied and two transradial prosthesis users | Reach geometry, asymmetry, trunk compensation | Not camera-pose or AdaptFit repetition labels without conversion | CC BY 4.0 reported; verify record | Not staged; manifest pending | Build a dedicated coordinate/participant adapter |
| InclusiveVidPose | Candidate / no adapter | Video/keypoints; amputations, limb differences, prostheses | Pose-front-end robustness and absent/unobserved distinction | Not temporal repetition or exercise-quality supervision | Data-use agreement/restrictions | Not staged; manifest pending | Obtain custodian approval before access |
| KIMORE | Candidate / no adapter | RGB-D/skeleton; healthy and motor-dysfunction participants | Physician features and clinical-context research | Not amputee/wheelchair validation; no commercial use assumption | Access/license pending | Not staged; manifest pending | Verify terms and label mapping |
| KERAAL | Candidate / no adapter | RGB-D/video/pose; healthy and low-back-pain participants | Expert correctness, error type/time, trunk compensation | Not clinical safety or target-profile validation | CC BY-NC-SA reported | Not staged; manifest pending | Research-only access review and adapter design |
| Toronto Rehab Stroke Pose | Candidate / no adapter | Kinect pose; stroke survivors and controls | Frame-level compensation/context | Not generic exercise correctness or AdaptFit capability support | Kaggle/access terms pending | Not staged; manifest pending | Verify release terms and source labels |
| StrokeRehab | Candidate / no adapter | Video features/IMU/kinematics; stroke and healthy participants | Functional primitives, upper-body representation | Not canonical 2D camera-pose or repetition labels | SimTK account/access required | Not staged; manifest pending | Decide whether feature modality is useful |
| SAFER-Activities | Candidate / no adapter | 2D/3D pose and images; normal/wheelchair recordings | Pose robustness, action segments, held-out subject/view tests | Not exercise-quality, clinical, or fall-safety labels | CC BY-NC-SA reported; access gate pending | Not staged; manifest pending | Verify terms and create wheelchair robustness split |
| WheelPose / Users in Wheelchairs | Candidate / no adapter | RGB images/keypoints; 84 public-video subjects | Pose detection and wheelchair robustness | Not temporal count or quality supervision | Code MIT; images have separate restrictions | Not staged; manifest pending | Obtain image-data permission |
| WheelPoser-IMU | Candidate / no adapter | IMU/mocap; wheelchair users | Sensor-fusion/motion priors | Not camera-pose or exercise feedback labels | CC BY-NC reported; request/access pending | Not staged; manifest pending | Keep separate from pose-only training |
| MM-Fi | Candidate / no adapter | Multimodal keypoints; 40 subjects | Representation/action pretraining | Taxonomy does not equal AdaptFit families; no target-profile claim | Access/license pending | Not staged; manifest pending | Verify download and source mapping |
| QEVD/FIT-300K | Candidate / no adapter | Video; many exercise variants | Video-first form/feedback pretraining after pose extraction | Not clinical judgment or direct pose labels | Qualcomm research license | Not staged; manifest pending | Request terms; estimate pose extraction cost |
| Fitness-AQA | Candidate / no adapter | In-the-wild video; squat/press/row | Fine-grained quality pretraining | Non-commercial and taxonomy cannot be copied as clinical labels | Access request/non-commercial | Not staged; manifest pending | Request access and map quality dimensions |
| WLU Rehabilitation Posture | Candidate / no adapter | Privacy-blurred video; post-stroke exercises | Verification/counting research | Terms and canonical pose format unclear | Access/license pending | Not staged; manifest pending | Do not download until terms are recorded |

The detailed research backlog remains in [dataset-expansion-plan.md](dataset-expansion-plan.md).

## Already integrated

### MM-Fit

[Official project page](https://mmfit.github.io/)

MM-Fit contains synchronized RGB-D video, 2D and 3D pose estimates, and more
than 800 minutes of multimodal exercise data. Its ten exercise classes include
bicep curls, rows, seated shoulder presses, seated lateral raises, squats,
lunges, sit-ups, tricep extensions, push-ups, and jumping jacks.

The adapter in `training/src/data/adapters.py` reads the pose-only archive and
exercise-set CSV files. It contributes explicit movement-family labels and
set-level repetition counts. MM-Fit does not provide individual repetition
boundaries or phase labels in the released CSV format, so those targets remain
masked instead of being invented. The archive is kept local until its dataset
terms are verified for the intended use.

## Highest-priority additions for AdaptFit

### ROAG

[Imperial College project page](https://www.imperial.ac.uk/manipulation-touch/open-source/dataset/roag-dataset/)
and [Zenodo record](https://zenodo.org/records/13908725)

ROAG, or Reaching Over a Grid, contains 2,450 reaching trajectories from seven
able-bodied participants and two transradial amputees using prosthetic devices.
It includes arm and torso motion across 49 targets and was designed to study
compensatory motion from upper-limb differences.

This is the best discovered match for the first missing-arm capability profile.
Use it to train or evaluate reach geometry, trunk compensation, asymmetry, and
capability-conditioned feedback. It is motion-capture data rather than the
current camera-pose format, so it needs a dedicated conversion adapter and
careful coordinate mapping. The Zenodo record lists CC BY 4.0; retain the
required attribution.

### InclusiveVidPose

[Project and dataset page](https://anonymous-accept.github.io/inclusivevidpose/)

InclusiveVidPose is a video pose-estimation dataset focused on individuals
with amputations, congenital limb differences, and prosthetic limbs. The
project reports 313 video sequences, more than 327,000 annotated frames, 398
individuals, 25 keypoints including residual-limb endpoints, segmentation
masks, bounding boxes, tracking IDs, and prosthesis information.

This should be used to improve the pose front end and the absent-versus-
unobserved anatomy distinction, not as direct repetition supervision. Its
download is governed by a data-use agreement and additional ethical and
research-use restrictions. Do not download, redistribute, or use it in a
product without approval from the dataset custodians.

### KIMORE

[Paper and dataset access link](https://doi.org/10.1109/TNSRE.2019.2923060)

KIMORE contains RGB-D and skeleton recordings for five physician-selected
rehabilitation exercises. It covers 78 subjects, including healthy subjects
and subjects with motor dysfunctions, and includes physician-defined movement
features and clinical scores.

This is a strong candidate for the quality and clinician-context heads. The
paper links to a Google Drive download, but the project does not provide a
simple permissive software-style license in the paper. Confirm the dataset
terms and access conditions before staging it. It is not a substitute for
amputee or wheelchair data.

### KERAAL low-back-pain rehabilitation dataset

[Official dataset page](https://keraal.enstb.org/KeraalDataset.html)

KERAAL contains three low-back-pain rehabilitation exercises from nine healthy
subjects and twelve patients. It provides Kinect 3D skeletons, RGB video,
OpenPose or BlazePose 2D poses, and physician annotations for correctness,
error type, body part, and error time span.

This is one of the most useful sources for observable quality, trunk
compensation, error localization, and clinician-reviewed labels. The official
page states CC BY-NC-SA, so it is appropriate for research experiments but not
automatically for a commercial mobile release.

### University of Liverpool Rehabilitation Exercise Dataset

[University data record](https://datacat.liverpool.ac.uk/2729/)

UL-RED contains 22 rehabilitation-oriented exercises across ten subjects, with
marker-based motion capture, marker-less tracking, and depth data. It includes
single- and three-repetition recordings at normal, fast, and slow speeds. The
data record provides subject archives and identifies them as CC BY 4.0.

This is a good addition for tempo robustness, phase modeling, and validating
marker-less pose extraction. The subjects are not the target disability
groups, so it should be treated as general rehabilitation pretraining.

### Toronto Rehab Stroke Pose Dataset

[Research group release page](https://www.cs.toronto.edu/~taati/index.htm)
and [dataset/code reference](https://github.com/zhiderek/TRSPD)

The Toronto Rehab Stroke Pose Dataset contains 25-joint Kinect pose data from
stroke survivors and healthy participants, with frame-level posture
compensation ratings and participant demographics. The referenced release is
available through Kaggle.

This is directly relevant to upper-body compensation and impaired movement. It
should be converted into canonical sequences with a source-specific label map;
the compensation labels should feed the trunk or observable-compensation head,
not be relabeled as generic exercise correctness. Verify Kaggle terms before
use.

### StrokeRehab

[Official dataset page](https://strokerehabdata.github.io/dataset.html)

StrokeRehab contains 3,372 trials from 51 stroke-impaired and 20 healthy
subjects, with high-resolution labels for short functional primitives such as
reach, transport, reposition, stabilize, and idle. It provides synchronized
IMU and video data; the released video side is feature data rather than raw
video, while the kinematic data is available through SimTK.

Use it for upper-body impaired-motion representation learning and primitive
recognition. It is not a direct replacement for camera pose because raw video
and canonical 2D landmarks are not the released modality. Access requires a
SimTK account.

### SAFER-Activities

[Dataset card](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities)
and [project page](https://safer-activities.github.io/)

SAFER-Activities includes normal and wheelchair recordings, non-lab held-out
tests, action segments, 2D poses, lifted 3D poses, boxes, subject/view splits,
and a wheelchair keypoint image set. The dataset card reports separate
wheelchair subject splits and an out-of-distribution non-lab test set.

This is a strong wheelchair pose-front-end and robustness benchmark. It is a
large download, requires agreeing to access conditions, and is released under
CC BY-NC-SA 4.0. Its simulated falls and activity segments should not be
treated as exercise-quality or clinical safety labels.

### WheelPose and the Users in Wheelchairs dataset

[Official repository](https://github.com/hilab-open-source/wheelpose)

WheelPose provides synthetic wheelchair-person pose generation and a Users in
Wheelchairs image dataset with roughly 2,464 annotated RGB images from 84
public videos, covering 16 action classes with boxes and keypoints. The full
image dataset is available on request; the repository code is MIT-licensed,
but the image dataset has separate source and fair-use constraints.

Use it to test and improve wheelchair detection and 2D keypoint extraction.
It is image-level rather than a temporal exercise dataset, so it should not be
used to claim better repetition counting by itself.

### WheelPoser-IMU

[Official repository](https://github.com/axle-lab/WheelPoser)

WheelPoser-IMU contains 167 minutes of paired IMU and motion-capture data from
wheelchair users, including propulsion and pressure-relief motions. The data
is obtained through a request form and the repository states a CC BY-NC 4.0
license for the project.

This is useful for a future sensor-fusion or physics-consistency branch, and
for learning wheelchair-specific motion priors. It is not camera data and
should not be mixed directly into the current pose-only training set.

## Additional useful sources

### MM-Fi

[Official toolbox and access instructions](https://github.com/ybhbingo/MMFi_dataset)

MM-Fi contains more than 320,000 synchronized frames from 40 subjects, 27
daily or rehabilitation action categories, and 2D/3D pose keypoints alongside
other sensing modalities. The release provides extracted keypoints while raw
RGB images are restricted for privacy.

Use its rehabilitation actions for representation pretraining and action
recognition. The action taxonomy is not the same as AdaptFit's exercise
families, and the access page points to Google Drive or Baidu Netdisk without a
simple license statement, so verify terms before use.

### Qualcomm Exercise Video Dataset

[Qualcomm dataset page](https://www.qualcomm.com/developer/software/qevd-dataset)

QEVD contains more than 474 hours of exercise and fitness-coaching video. Its
FIT-300K subset has about 289,000 short clips covering 148 exercises and
variations including different pacing, common mistakes, and modified form.
The FIT-Coach subsets add longer workouts and coach-style corrective feedback.

This is attractive for quality-language and form-variation pretraining after
running a pose extractor over the videos. It is not a skeleton dataset, access
uses a Qualcomm research license agreement, and the feedback labels are not
clinical judgments.

### Fitness-AQA

[Official code and access instructions](https://github.com/ParitoshParmar/Fitness-AQA)

Fitness-AQA targets fine-grained quality assessment for back squat, overhead
press, and barbell row. The dataset is distributed through an access request
form and the repository states that it is for non-commercial use.

Use it for row and trunk-quality representation learning if permission is
granted. It is in-the-wild video rather than canonical pose, and its quality
taxonomy must be mapped carefully into ROM, tempo, smoothness, and trunk
compensation rather than copied as a clinical label.

### UCOPhyRehab++

[Official Zenodo release](https://zenodo.org/records/17935737) and
[Scientific Data paper](https://www.nature.com/articles/s41597-026-07362-5)

UCOPhyRehab++ is now staged under `data/raw/ucophyrehabpp/`. Its 3D pose and
metadata release provides exact repetition frame ranges and a 1–5 composite
physiotherapist execution score for rehabilitation exercises. The new v2
adapter uses it for exact recording boundaries and an optional ordinal expert
quality head.

The release contains healthy, controlled participants rather than amputee,
limb-difference, or wheelchair-user participants. The score is not four
independent ROM, tempo, smoothness, and trunk-compensation labels. Keep the
source marked for license review before redistribution or commercial use. See
`docs/quality-benchmark-v2.md` for the adapter and overnight run.

### WLU Rehabilitation Posture

[Dataset page](https://www.kaggle.com/datasets/sulaimanmuhammed/wlu-rehabilitation-posture)

WLU Rehabilitation Posture focuses on post-stroke arm raise, knee extension,
and sit-to-stand recordings. Its videos are blurred for privacy and include
multiple angles and devices.

It may help with exercise verification and repetition-count experiments, but
the page does not establish a clear redistribution license or a canonical pose
format. Treat it as a request-and-review candidate rather than an immediate
download.

## Recommended acquisition order

Start with ROAG, InclusiveVidPose, KERAAL, UL-RED, and SAFER-Activities. These
cover the largest current gaps: real upper-limb difference, limb-deficiency
pose estimation, clinician-reviewed quality, tempo variation, and wheelchair
robustness.

Request KIMORE and Toronto Rehab Stroke Pose next for clinical and stroke
compensation labels. Add MM-Fi, QEVD, and Fitness-AQA only as carefully
licensed pretraining sources because they are broader or video-first.

Do not merge all sources into one undifferentiated loss. Each adapter should
declare exactly which labels it supplies, preserve label provenance, and leave
unsupported phase, boundary, quality, and clinical targets masked.

## What these datasets still cannot provide

No discovered public dataset covers the full first-release population of
people with missing limbs, one-arm use, lower-limb absence, and wheelchair use
performing the same five AdaptFit exercises under consented product-testing
conditions. Public data can bootstrap the pose front end and general movement
model, but the safety-critical adaptation layer still needs consented,
participant-reviewed recordings and clinician or trained-expert labeling.
