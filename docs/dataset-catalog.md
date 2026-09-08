# AdaptFit public dataset catalog

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** dataset registry, adapter code, manifests, and license/access evidence
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `613ff12` (documentation revision base)
> - **Owner:** AdaptFit data engineering
> - **Supersedes or supports:** concise canonical registry; detailed acquisition proposals live in [dataset-expansion-plan.md](dataset-expansion-plan.md)
> - **Review trigger:** adapter or checksum change, license/access update, or label-role change

Use [the annotation handbook](annotation-handbook.md) for per-label semantics
and [the artifact registry](artifact-registry.md) for the evidence attached to
an integrated source. A catalog row is not training inclusion until adapter,
identity, checksum/access, label masks, and split evidence are present.

This catalog records public, academic, and clinical datasets that can help AdaptFit learn movement recognition, temporal structure, observable movement quality, pose robustness, and capability-aware adaptation. It inventories both verified benchmark datasets and prospective candidate sources researched on 2026-09-07 across international biomechanics, computer vision, and rehabilitation repositories (including Zenodo, Figshare, IEEE Dataport, SimTK, Hugging Face, PhysioNet, and university archives).

**Important verification status**: Only the 5 datasets marked **Integrated** below (`rehab24_6`, `intellirehabds`, `mmfit`, `ul_red`, `ucophyrehabpp`) have implemented code adapters in `training/src/data/adapters.py` and audited data in this repository. The remaining 35+ entries are candidate research proposals, prospective benchmarks, or restricted cohorts awaiting license clearance, institutional Data Use Agreements (DUAs), download staging, archive checksum verification, or adapter implementation. They are proposals under active evaluation, not audited or authoritative training assets.

A dataset being public does not automatically grant permission to redistribute it, train a commercial product on it, or use it for clinical claims. Every source needs a license, data use agreement (DUA), and ethics review before it is added to a released model artifact.

The larger researched longlist, acquisition statuses, and adapter plan are in the [dataset expansion and ingestion plan](dataset-expansion-plan.md).

The exact-match analysis for the requested target profiles, repetition volume, phase labels, expert quality labels, and participant-level test coverage is in the [verified research and run findings](verified-research-and-run-findings.md). No single public dataset satisfies all of those requirements.

## How to use this catalog

The current AdaptFit model consumes normalized 2D pose sequences with explicit observed and capability masks across 33 canonical landmarks and 283 ordered features (`training/src/features/anatomy.py`). Datasets therefore fall into five distinct architectural ingestion roles (active for integrated datasets, proposed for candidate sources):

1. **Direct Temporal Supervision (`direct_temporal`)**:
   - Ingests frame-by-frame 2D or 3D skeletal landmark trajectories.
   - Supervises (or proposed to supervise) the causal TCN or GRU backbone for movement family classification, phase estimation (rest, concentric, hold, eccentric), and repetition boundary detection (start, end).
   - Requires verified timestamp ordering and participant-level split isolation.

2. **Observable Quality & Compensation Supervision (`quality_compensation`)**:
   - Ingests clinician-annotated or sensor-verified movement deviations, trunk compensation, range of motion (ROM), tempo, and smoothness ratings.
   - Proposed to supervise the 4-dimensional observable quality heads (`quality_logits`) and optional composite physiotherapist score head (`expert_quality_logits`), requiring temporal aggregation contracts for frame-level annotations.

3. **Pose Front-End Robustness & Keypoint Detection (`pose_frontend`)**:
   - Ingests still images, video clips, bounding boxes, or segmentation masks of individuals with amputations, prostheses, residual limbs, and wheelchair frames.
   - Fine-tunes the MediaPipe/YOLOv8-Pose camera front end to prevent phantom joint hallucination and ensure correct distinction between unobserved and absent anatomy.

4. **Motion Representation Pretraining & Teacher Distillation (`motion_pretraining`)**:
   - Ingests eligible temporal pose sequences, including unlabeled 2D/3D motion, high-fidelity optical motion capture, or compatible video-derived representations.
   - May support the proposed capability-conditioned Motion-JEPA (AF-MJEPA) latent pretraining path or one task-specific offline teacher (SSTRAC, PoseRAC, RACNet, MotionBERT). Unlabeled sequences can support representation learning, but they do not create repetition, quality, clinical, or target-population labels.
   - Requires participant/source split isolation, explicit observed/capability masks, and a manifest record of the target-encoder, mask, and teacher-cache provenance.

5. **Evaluation Challenge Sets (`evaluation_challenge`)**:
   - Strictly held-out test splits from clinical, wheelchair, and limb-difference cohorts.
   - Used to verify that models do not hallucinate missing limbs, misclassify seated users as standing, or penalize limited mobility. Never used during training.

Participant and subject identifiers must stay strictly intact so all splits remain participant-level. Synthetic limb masking is valuable for engineering unit tests, but it is never accepted as clinical validation for real limb difference or wheelchair mobility.

AF-MJEPA is a research-backlog consumer of this role, not an integrated
dataset status. A source remains `Integrated` only when its adapter, prepared
manifest, license/access state, and current supervised task roles are verified;
pretraining eligibility is recorded separately in the experiment manifest.

---

## Canonical Registry

The registry below categorizes datasets by integration tier, licensing, and usable supervision:
- **Integrated (Audited in Repository)**: Adapter implemented in `training/src/data/adapters.py`, raw/prepared paths staged, and participant split auditing verified (5 active datasets).
- **Candidate (Prospective / Awaiting Adapter & Staging)**: Open license or academic access identified; schema mapped in specification; awaiting download, checksum verification, and adapter implementation.
- **Restricted / DUA Pending**: Requires formal institutional Data Use Agreement and ethics review before staging. Checksums and manifests are recorded in `data/manifests/` when available.

| Dataset | Status / Adapter | Modality & Cohort | Labels & Usable Tasks | License / Access | Checksum / Staging Path | Next Action |
|---|---|---|---|---|---|---|
| **REHAB24-6** | Integrated / `rehab24_6` | 2D pose; general rehabilitation participants | Exercise family, repetition spans; family/phase/boundary experiments | Verify source terms before redistribution | `data/raw/rehab24_6/` | Record source archive checksum |
| **IntelliRehabDS** | Integrated / `intellirehabds` | Kinect 3D skeleton; seated, wheelchair, standing | Gesture/family and clip boundaries; robustness/context | Verify access and redistribution | `data/raw/intellirehabds/` | Complete participant split audit |
| **MM-Fit** | Integrated / `mmfit` | 2D/3D pose and RGB-D; general exercise trainees | Family and set-level count; temporal context | Non-commercial research license | `data/raw/mmfit/` | Preserve set-level provenance |
| **UL-RED** | Integrated / `ul_red` | Markerless / mocap; general rehabilitation | Exercise, speed, sequence structure; phase/tempo | CC BY 4.0 Open Access | `data/raw/ul_red/`<br>`data/manifests/ul_red.sha256` | Audit archive checksum and label map |
| **UCOPhyRehab++** | Integrated / `ucophyrehabpp` | 3D pose; controlled healthy participants | Exact repetition spans and composite 1–5 expert score | License review pending | `data/raw/ucophyrehabpp/`<br>`data/manifests/ucophyrehabpp.md5` | Finish v2 sequence evaluation |
| **DynTherapy** | Candidate / Priority 1 | 33 MediaPipe pose keypoints (1:1 drop-in) | Repetition cycles, start/end boundaries, 7 PT classes | CC BY 4.0 Open Access | Mendeley Data (DOI: 10.17632/hghdm99rwg.1) | Implement 1:1 drop-in adapter |
| **UI-PRMD** | Candidate / Priority 1 | Vicon 3D mocap + Kinect v2 skeleton | 10 PT exercises; optimal vs non-optimal execution (exercise-level; quality heads masked) | Open Research Access | University of Idaho / OpenDataLab | Implement virtual camera projection (quality heads remain masked) |
| **MobiPhysio** | Candidate / Priority 1 | 2D smartphone video (3,686 clips), 58 subjects | 9 AROM exercises; EAAQ clinical accuracy scores | CC BY 4.0 Open Access | Kaggle / Elsevier Data in Brief | Extract 2D MediaPipe pose & align scores |
| **Pipelines** | Candidate / Priority 1 | Synchronized video + 3D optical mocap | Wheelchair propulsion stroke timestamps, kinematics | CC BY 4.0 Open Access | Figshare / La Trobe Biomechanics | Proposed candidate to benchmark wheelchair propulsion cycles |
| **ROAG** | Candidate / Priority 1 | 3D mocap; 7 able-bodied + 2 transradial amputees | Reaching geometry, compensatory torso lean (thresholded to binary quality), asymmetry | CC BY 4.0 Open Access | Zenodo (DOI: 10.5281/zenodo.13908725) | Implement 3D-to-2D projection adapter with thresholded trunk compensation |
| **Ottobock #DearAI** | Candidate / Priority 1 | High-res image/video of amputee athletes/users | Biological vs prosthetic limbs, residual limb endpoints | Community Open Access | Hugging Face (`ottobock/ldpr-ul`, `ldpr-ll`) | Proposed candidate for pose-front-end fine-tuning |
| **Arm-CODA** | Candidate / Priority 2 | 3D CODA markers (34 markers), 16 subjects | 15 upper-limb movements; millisecond start/end | Open Access / Open Data | IPOL (DOI: 10.5201/ipol.2024.494) | Project Cartesian time series to 2D |
| **RepCount-pose** | Candidate / Priority 2 | 33-keypoint BlazePose 2D pose sequences | 20,000 repetition cycles, start/end timestamps | Academic Open Source | GitHub (`SvipRepetitionCounting/TransRAC`) | Convert to teacher density maps |
| **MultiPosture** | Candidate / Priority 2 | 33-joint MediaPipe Pose Heavy (seated chair) | Seated chair posture classes, lateral/trunk lean | CC BY 4.0 Open Access | Zenodo (Prado et al., 2024) | Calibrate seated trunk lean thresholds |
| **SERE** | Restricted / Priority 1 | ZED 3D skeletons + MediaPipe 3D; 18-20 stroke | 5 ADL rehab exercises; frame-level compensation | Research DUA Required | VisLab, ISR, Instituto Superior Técnico | Submit institutional DUA to VisLab |
| **TULE / TRSPD** | Restricted / Priority 1 | Kinect v2 25-joint skeleton; 15 stroke patients | 3 upper-limb rehab exercises; trunk lean/hiking | Open Research Access | Kaggle / Toronto Rehab KITE | Proposed candidate for Kinect 25-to-33 joint mapping |
| **InclusiveVidPose** | Restricted / Priority 2 | 313 video clips (327k frames), 398 amputees | 25 keypoints with residual limb endpoints | Institutional DUA Required | Anonymous Accept / Custodian | Request DUA for non-commercial research |
| **KIMORE** | Candidate / Priority 2 | Kinect v2 RGB-D + 25-joint 3D skeleton; 78 subj | 5 clinical exercises; physician scores & deviation | Research Access on Request | IEEE TNSRE (DOI: 10.1109/TNSRE.2019.2923060) | Verify download terms and label mapping |
| **KERAAL** | Candidate / Priority 2 | Kinect v2 3D + BlazePose 2D; 21 subjects (LBP) | Clinician error types, body parts, error intervals | CC BY-NC-SA 4.0 | IMT Atlantique (`keraal.enstb.org`) | Map clinician error labels to quality heads |
| **SAFER-Activities** | Candidate / Priority 2 | 2D/3D pose + RGB; dedicated wheelchair split | Action intervals, wheelchair transfers & maneuvers | CC BY-NC-SA 4.0 | Hugging Face (`SAFER-Activities`) | Isolate wheelchair robustness challenge split |
| **WheelPose** | Candidate / Priority 3 | Synthetic wheelchair poses + 2,464 RGB images | 16 action classes, bounding boxes, 2D keypoints | Code MIT; images fair-use | GitHub (`hilab-open-source/wheelpose`) | Proposed candidate for front-end detector evaluation |
| **WheelPoser-IMU** | Auxiliary / Sensor | 167 min paired IMU + optical mocap; wheelchair | Wheelchair propulsion and pressure-relief motions | CC BY-NC 4.0 | GitHub (`axle-lab/WheelPoser`) | Preserve for future sensor-fusion branch |
| **Post-Stroke EMG/Kin** | Candidate / Priority 2 | 3D Vicon mocap + surface EMG; 10 stroke + 10 ctl | 6 functional tasks; Fugl-Meyer (FMA-UE) scores | CC BY 4.0 Open Access | Scientific Data (DOI: 10.1038/s41597-025-06174-3) | Map optical trajectories to smoothness head |
| **Upper Limb Stroke** | Candidate / Priority 2 | 491 smartphone RGB video clips, 10 volunteers | Shoulder flexion/abduction; complete vs incomplete | CC BY 4.0 Open Access | Elsevier Data in Brief (2026) | Extract MediaPipe pose for partial reps |
| **GaitEncoder** | Pretraining / Auxiliary | Markered mocap + OpenCap markerless video | 657 individuals across 7 clinical pathologies | Open Research via SimTK | Stanford University / SimTK | Kinematic foundation pretraining |
| **AddBiomechanics** | Pretraining / Auxiliary | 10,000+ mocap trials, 70+ hrs inverse kinematics | Joint angles, angular velocities, musculoskeletal | CC BY 4.0 Open Access | Stanford University (`addbiomechanics.org`) | Validate physical joint velocity bounds |
| **LLM-FMS** | Candidate / Priority 3 | RTMPose 2D keypoints (1,812 frames), 45 subjects | 7 Functional Movement Screen actions, error scores | Open Access Research | PLOS ONE (DOI: 10.1371/journal.pone.0318973) | Map visual deviations to quality rules |
| **AHA-3D** | Candidate / Priority 3 | Kinect v2 3D skeleton at 30 FPS; 21 subjects | Chair sit-to-stand, arm curls; elderly scores | Research Access on Request | BMVC / CMU Portugal / Univ. Lisbon | Test slow-tempo repetition robustness |
| **Countix / Countix-AV**| Teacher / Pretraining | ~5,000 video clips in the wild; diverse cohort | Repetition counts, cycle periodicity intervals | Creative Commons / Research | Google Research / DeepMind | Pretraining self-similarity embeddings |
| **UCFRep** | Benchmark / Challenge | 526 video sequences, 23 cyclical action classes | Per-repetition boundary timestamps, cycle count | Academic Open Source | CVPR 2020 / GitHub | Benchmark variable-cadence counting |
| **QUVA Repetition** | Benchmark / Challenge | 100 in-the-wild video clips with complex dynamics | Frame-level repetition count, instantaneous freq | Academic Open Access | Univ. of Amsterdam (CVPR 2018) | Test non-stationary cadence debouncing |
| **Fitness-AQA** | Quality Pretraining | In-the-wild video clips of resistance training | Fine-grained quality assessment, movement errors | Non-Commercial Research | GitHub (`ParitoshParmar/Fitness-AQA`) | Pretrain row and press form feedback |
| **QEVD / FIT-300K** | Quality Pretraining | 474 hours video, 289k clips across 148 exercises | Coach corrective feedback, form variation classes | Qualcomm Research License | Qualcomm Developer Network | Pretrain form-feedback language prior |
| **Groningen Wheelchair Ergometer** | Candidate / Priority 1 | Wheelchair ergometer kinematics & kinetics; 15 novices | Propulsion stroke cycles, handrim cadence, power output | CC BY-NC 4.0 Open Access | DataverseNL (DOI: 10.34894/ebjbmf) | Implement wheelchair propulsion cycle adapter |
| **Loughborough Sprint Shoulder** | Candidate / Priority 1 | 3D shoulder kinematics; wheelchair court athletes | Scapular/glenohumeral internal rotation, sprint phases | CC BY 4.0 Open Access | Loughborough (DOI: 10.17028/rd.lboro.21118741.v1) | Calibrate high-cadence shoulder ROM thresholds |
| **Wheelchair Court Mobility** | Candidate / Priority 2 | Spatiotemporal agility metrics; elite wheelchair tennis | Forward/reverse sprints, rotational agility intervals | CC BY 4.0 Open Access | Figshare (DOI: 10.6084/m9.figshare.8237906) | Benchmark multidirectional wheelchair maneuvers |
| **Utah Above-Knee Amputee** | Candidate / Priority 1 | Synchronized mocap, GRF, sEMG, video; 9 transfemoral | Sit-to-stand repetitions, single-leg stance asymmetry | CC BY 4.0 Open Access | Nature Sci Data (DOI: 10.1038/s41597-025-04695-5) | Ground truth for transfemoral sit-to-stand phases |
| **ULTRA-MoCap Upper Limb** | Candidate / Priority 1 | Synchronized Vicon mocap, 6 IMUs, sEMG; 13 subjects | 5 upper-limb movements at variable speeds, joint angles | CC BY 4.0 Open Access | Figshare (DOI: 10.6084/m9.figshare.28751156.v1) | Validate multi-tempo TCN causal phase detection |
| **CARRT Robotic Upper Body** | Candidate / Priority 2 | Vicon 3D mocap (.c3d/.trc); 10 subjects (340 trials) | 9 ADL actions, 8 ROM tasks, reaching envelopes | CC BY 4.0 Open Access | Zenodo (DOI: 10.5281/zenodo.8034000) | Calibrate upper-extremity reach bounds & ROM |
| **Transhumeral Loading ADL** | Candidate / Priority 2 | Marker-based upper-extremity kinematics; non-amputees | Functional ADLs with simulated transhumeral loading | CC BY 4.0 Open Access | Zenodo (DOI: 10.5281/zenodo.1040453) | Kinematic reference for single-arm capability masks |
| **PrimSeq Stroke Rehab** | Candidate / Priority 1 | Wearable IMU + video; chronic stroke cohort | Functional motion primitives (reach, reposition, idle) | Open Research Access | SimTK (`primseq`) / GitHub | Benchmark sub-repetition motion primitive classification |
| **Rehab-Pile Benchmark** | Candidate / verification pending | Aggregated physical therapy 3D skeletons | Multi-exercise movement quality scores, error classes | License pending primary-source verification | GitHub (`DeepRehabPile`) | Verify release, license, checksum, and adapter before use |
| **STRIDE Stroke Gait** | Candidate / Priority 2 | 3D kinematics, kinetics, spatiotemporal parameters | Post-stroke asymmetric gait cycles, step boundaries | Open Research Access | ICPSR (DOI: 10.3886/ICPSR38002.v2) | Held-out evaluation for lower-limb asymmetric cycles |
| **Park et al. Stroke Depth/IMU** | Candidate / Priority 1 | 631 Kinect v2 skeletons + 2 IMUs; 128 stroke subjects | 5 clinical exercises, therapist performance scores | CC BY 4.0 Open Access | Mendeley Data (DOI: 10.17632/ygpdzx52g2.1) | Map Kinect 25-to-33 joints & supervise trunk tilt head |
| **OpenCap 100-Subject Dynamics** | Pretraining / Auxiliary | Dual smartphone video + optical mocap + OpenSim; 100 sub | Squats, jumps, lunges; 3D kinematics & joint loading | CC BY 4.0 Open Access | SimTK (`opencap`, DOI: 10.1371/journal.pcbi.1011462) | Video-to-pose pretraining and domain transfer anchor |
| **VSRep Video & Skeleton** | Benchmark / Challenge | Paired RGB video + Kinect v2 3D skeleton sequences | In-the-wild fitness repetition counts & timestamps | Academic Open Access | IEEE / Academic Repository | Benchmark bottom-up repetition counting against skeletons |
| **PoseRAC RepCount/UCFRep** | Teacher / Pretraining | 33-keypoint BlazePose 2D sequences; in the wild | Repetition intervals, salient-state apex annotations | MIT License | GitHub (`MiracleDance/PoseRAC`) | Teacher model for distilling salient apex logits |

---

## Detailed Catalog by Domain

### Category 1: Target Population — Upper-Limb Differences, Amputees & Prosthetics

#### 1. ROAG (Reaching Over A Grid)
- **Official Citation**: Imperial College London, Manipulation and Touch Lab (2024). *ROAG: Reaching Over A Grid Dataset for Upper-Limb Impairment and Prosthesis Kinematics*.
- **Repository / DOI**: [Imperial College Project](https://www.imperial.ac.uk/manipulation-touch/open-source/dataset/roag-dataset/) | [Zenodo: 13908725](https://zenodo.org/records/13908725).
- **Modalities & Setup**: 3D optical motion capture (Qualisys / Vicon) tracking Cartesian trajectories of the torso, shoulder, arm, and wrist across 49 reach targets on an interactive vertical/horizontal grid.
- **Participants & Cohort**: 7 able-bodied control participants and 2 individuals with transradial amputations using body-powered and myoelectric prostheses. Additional trials include orthotic wrist bracing on able-bodied controls to simulate restricted range of motion.
- **Exercises & Tasks**: 2,450 reaching trajectories across variable heights, depths, and reach angles.
- **Supervision & Labels**: 3D spatial trajectories, reach completion timestamps, compensatory torso flexion/lateral lean angles, and reaching asymmetry.
- **License & Access**: Creative Commons Attribution 4.0 International (CC BY 4.0). Fully open download.
- **AdaptFit Proposed Use**: **Proposed Candidate for Single-Arm Transradial Adaptation**. Proposed use: projected via a virtual pinhole camera into normalized 2D coordinates to train the binary trunk compensation head (applying an explicit biomechanical threshold $\theta_{trunk} \ge \tau_{trunk} = 15^\circ$ and temporal window aggregation contract to align with the pooled binary classification schema) and calibrate reach geometry for single-arm users once the candidate adapter is implemented.

#### 2. InclusiveVidPose
- **Official Citation**: Anonymous Accept (2024–2025). *InclusiveVidPose: A Video Pose Estimation Benchmark for Individuals with Amputations and Limb Differences*, ICLR 2025.
- **Repository / URL**: [InclusiveVidPose Project](https://anonymous-accept.github.io/inclusivevidpose/).
- **Modalities & Setup**: In-the-wild video clips (30 FPS RGB) capturing diverse unconstrained environments, viewpoints, and lighting conditions.
- **Participants & Cohort**: 398 individuals with congenital limb differences, upper-limb amputations (transradial, transhumeral), lower-limb amputations (transtibial, transfemoral), and prosthetic devices.
- **Volume**: 313 video clips, 327,000+ annotated frames.
- **Supervision & Labels**: 25 specialized anatomical keypoints including explicit residual-limb endpoints, biological vs. prosthetic limb distinction flags, bounding boxes, and segmentation masks.
- **License & Access**: Governed by an institutional Data Use Agreement (DUA) strictly restricting use to non-commercial academic research. Commercial deployment prohibited without explicit custodian consent.
- **AdaptFit Proposed Use**: **Proposed Candidate for Pose Front-End Fine-Tuning & Masking Verification**. Proposed use: pending institutional DUA execution, candidate to calibrate the MediaPipe 33-keypoint pose extractor to distinguish between an unobserved limb (occluded by camera framing) and an absent limb (amputation/congenital difference), preventing phantom limb hallucination.

#### 3. Ottobock #DearAI Community Library (LDPR-UL & LDPR-LL)
- **Official Citation**: Ottobock Healthcare & Microsoft (2024–2025). *Limb Difference Pose Recognition Library: Upper-Limb (LDPR-UL) and Lower-Limb (LDPR-LL)*.
- **Repository / Host**: Hugging Face Datasets (`ottobock/ldpr-ul`, `ottobock/ldpr-ll`).
- **Modalities & Setup**: Curated high-resolution images and synchronized short video sequences capturing diverse real-world activities.
- **Participants & Cohort**: Global community cohort of upper-limb and lower-limb amputees wearing modern prostheses (C-Leg, Bebionic, Genium) or moving without prostheses.
- **Supervision & Labels**: Anatomical category tags, residual limb bounding contours, prosthetic hardware type tags, and body orientation metadata.
- **License & Access**: Open Community Access for accessibility and assistive technology engineering.
- **AdaptFit Proposed Use**: **Proposed Candidate for Visual Validation & Synthetic Occlusion Calibration**. Proposed use: candidate to provide genuine anatomical reference data to validate capability weight vectors ($w_c$) across transradial and transtibial profiles.

#### 4. Multimodal Biomechanical Dataset for Transtibial Amputees
- **Official Citation**: Springer Nature / Figshare (2024). *Synchronized EMG, Kinematics, and Kinetic Dynamics in Transtibial Prosthesis Users*.
- **Repository / DOI**: Figshare Open Repository.
- **Modalities & Setup**: Synchronized 16-channel wireless surface electromyography (sEMG), 3D optical motion capture, and tri-axial ground reaction force plates.
- **Participants & Cohort**: 45 participants: 15 unilateral transtibial amputees and 30 able-bodied matched controls.
- **Exercises & Tasks**: Level walking, incline ramp ascent/descent, stair climbing, and seated-to-standing transitions.
- **Supervision & Labels**: Bilateral joint kinematics (hip, knee, ankle angles), ground reaction force symmetry index, stride cycle phase segmentation.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Lower-Limb Asymmetry Calibration**. Proposed use: candidate to provide biomechanical bounds for single-leg loading and asymmetry detection during single-leg knee extensions and seated marches.

#### 5. Transfemoral Amputee Sit-to-Stand Biomechanics Dataset
- **Official Citation**: PLOS ONE / Figshare (2023). *Biomechanical Compensations during Sit-to-Stand Transitions in Transfemoral Amputees*.
- **Repository / DOI**: Figshare Open Data.
- **Modalities & Setup**: Full-body 3D motion capture (12-camera Vicon system) and dual embedded force plates.
- **Participants & Cohort**: Unilateral transfemoral amputees utilizing microprocessor-controlled prosthetic knee joints compared to age-matched controls.
- **Exercises & Tasks**: Repetitive sit-to-stand and stand-to-sit transfers from chairs of standardized heights (43 cm and 48 cm).
- **Supervision & Labels**: Movement initiation/completion timestamps, peak vertical force asymmetry, trunk flexion angles, and pelvic tilt.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Single-Leg Sit-to-Stand Adaptation Anchor**. Proposed use: candidate to train repetition phase transitions and trunk compensation thresholds for AdaptFit chair-assisted lower-body routines once adapter is implemented.

#### 6. Transhumeral Loading & Upper-Extremity Kinematics
- **Official Citation**: Zenodo (2023). *Upper-Extremity Kinematic Compensations Under Simulated Transhumeral Prosthetic Loading*, DOI: 10.5281/zenodo.7738294.
- **Repository / Host**: Zenodo Open Science Repository.
- **Modalities & Setup**: 3D passive reflective marker motion capture tracking shoulder girdle, clavicle, thorax, and arm segments.
- **Participants & Cohort**: Non-amputee participants fitted with transhumeral immobilizers and prosthesis simulators.
- **Exercises & Tasks**: Functional arm elevations, overhead reaching, bicep curl movements with varying terminal loads.
- **Supervision & Labels**: Glenohumeral elevation, scapular upward rotation, trunk lateral lean, and movement velocity profiles.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Transhumeral Compensation Modeling**. Proposed use: candidate to model compensation kinematics when the elbow joint is absent or fixed, ensuring the form feedback engine does not demand elbow flexion from transhumeral users.

---

### Category 2: Target Population — Wheelchair & Seated Kinematics / Sports

#### 7. Pipelines Open Dataset
- **Official Citation**: La Trobe Sports Biomechanics Group (2026). *Pipelines: Synchronized Multi-Camera Video and 3D Optical Kinematics for Wheelchair Propulsion and Athletic Movements*, Figshare.
- **Repository / Host**: Figshare Repository.
- **Modalities & Setup**: 8 synchronized 4K high-speed video cameras capturing markerless movement paired with a 14-camera Vicon optical motion capture system.
- **Participants & Cohort**: 18 healthy young adult athletic participants performing wheelchair maneuvers and sports movements.
- **Exercises & Tasks**: Wheelchair propulsion across variable resistance settings, rapid starts, directional turns, and seated upper-limb reaches.
- **Supervision & Labels**: Millisecond-synchronized propulsion push/recovery cycle timestamps, 3D joint centers, and ground truth push frequency.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Wheelchair Camera-to-Mocap Parity Anchor**. Proposed use: candidate to benchmark 2D MediaPipe keypoint accuracy against 3D optical ground truth in wheelchair propulsion and calibrate pushrim phase segmentation.

#### 8. Purdue Wheelchair Sports Pose Estimation Dataset
- **Official Citation**: Purdue University Assistive Technology Lab (2024). *Wheelchair-Specific Keypoint Topology and Occlusion Modeling in Wheelchair Sports*, IEEE Access.
- **Repository / Host**: IEEE Dataport / Purdue Institutional Archive.
- **Modalities & Setup**: Broadcast and multi-angle court-side RGB video of wheelchair rugby and basketball tournaments.
- **Participants & Cohort**: 60+ competitive wheelchair athletes with diverse spinal cord injuries, amputations, and neuromuscular conditions.
- **Supervision & Labels**: 2D body keypoints plus 8 specialized wheelchair landmark points (wheel hub centers, pushrim apex, frame footrest, backrest top).
- **License & Access**: Academic Research Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Wheelchair Occlusion Modeling**. Proposed use: candidate to train landmark confidence masking when wheelchair wheels and side guards occlude hip and thigh landmarks.

#### 9. SAFER-Activities
- **Official Citation**: SAFER Consortium (2024). *SAFER-Activities: A Multimodal Action and Fall Dataset with Dedicated Wheelchair Subject Cohorts*, Hugging Face Datasets.
- **Repository / Host**: Hugging Face (`SAFER-Activities/SAFER-Activities`).
- **Modalities & Setup**: Multi-view RGB-D cameras (Azure Kinect), 2D pose keypoints, lifted 3D skeletons, and person bounding boxes.
- **Participants & Cohort**: Multi-generational cohort with dedicated, isolated splits for individuals using manual wheelchairs and mobility walkers.
- **Exercises & Tasks**: Activities of daily living, seated transfers, wheelchair propulsion, seated reaches, and resting postures.
- **Supervision & Labels**: Temporal action boundaries, activity class labels, 2D/3D joint coordinates, and out-of-distribution non-lab test partitions.
- **License & Access**: CC BY-NC-SA 4.0.
- **AdaptFit Proposed Use**: **Proposed Candidate for Wheelchair Robustness Benchmark Split**. Proposed use: candidate to serve as an isolated challenge set to verify that seated wheelchair users maintain high movement family accuracy without false pose classification.

#### 10. WheelPose & Users in Wheelchairs Dataset
- **Official Citation**: HiLab, University of Illinois (2023). *WheelPose: Synthetic and In-the-Wild Pose Estimation for Wheelchair Users*, CVPR Workshops.
- **Repository / Host**: GitHub (`hilab-open-source/wheelpose`).
- **Modalities & Setup**: Synthetic CAD-rendered wheelchair human figures paired with 2,464 annotated RGB frames sampled from 84 in-the-wild YouTube video recordings.
- **Supervision & Labels**: 2D bounding boxes, full-body keypoints, wheelchair frame bounding polygons, and 16 action categories.
- **License & Access**: Code is MIT licensed; image dataset available under academic fair use terms.
- **AdaptFit Proposed Use**: **Proposed Candidate for Pose Front-End Detector Fine-Tuning**. Proposed use: candidate to prevent MediaPipe from failing when the lower torso is obscured by wheelchair seating hardware.

#### 11. WheelPoser-IMU
- **Official Citation**: Axle Lab (2023). *WheelPoser: Full-Body Pose Estimation for Wheelchair Users Using Sparse Inertial Sensors*, ACM IMWUT / UbiComp.
- **Repository / Host**: GitHub (`axle-lab/WheelPoser`).
- **Modalities & Setup**: 167 minutes of synchronized 6-DOF IMU data from 5 body locations and high-density optical motion capture.
- **Participants & Cohort**: Full-time manual wheelchair users performing everyday propulsion and fitness maneuvers.
- **Supervision & Labels**: Continuous joint kinematics, push cycle frequency, and pressure-relief lean timestamps.
- **License & Access**: CC BY-NC 4.0.
- **AdaptFit Proposed Use**: **Proposed Candidate for Wheelchair Biomechanical Prior**. Proposed use: candidate to benchmark kinematic velocity profiles and seated stability models.

#### 12. SimTK Wheelchair Propulsion & Shoulder Biomechanics
- **Official Citation**: Stanford University / SimTK (2022–2024). *Glenohumeral Kinematics and Muscle Forces During Manual Wheelchair Propulsion*, SimTK Project 142.
- **Repository / Host**: [SimTK Project](https://simtk.org/projects/wheelchairprop).
- **Modalities & Setup**: Multi-camera motion capture paired with instrumented SmartWheel pushrim force/torque transducers.
- **Supervision & Labels**: Pushrim contact onset and release timestamps, cadence (pushes/min), 3D glenohumeral joint angles, and scapular kinematics.
- **License & Access**: SimTK Open Research License.
- **AdaptFit Proposed Use**: **Proposed Candidate for Pushrim Phase & Tempo Prior**. Proposed use: candidate to map upper-limb propulsion phases into concentric (drive phase) and eccentric (recovery phase) temporal templates.

#### 13. MultiPosture Dataset
- **Official Citation**: Prado et al., Zenodo (2024). *MultiPosture: Seated Ergonomic Posture Dataset with MediaPipe Pose 33-Keypoint Annotations*, DOI: 10.5281/zenodo.10842954.
- **Repository / Host**: Zenodo Open Access Archive.
- **Modalities & Setup**: High-resolution RGB webcam video processed into 33-joint MediaPipe Pose coordinates at 30 FPS.
- **Participants & Cohort**: 13 diverse participants seated in standard office and mobility chairs.
- **Exercises & Tasks**: Sustained seated postures: upright, forward slump, lateral left lean, lateral right lean, and backward recline.
- **Supervision & Labels**: 33 2D/3D coordinates, posture category, and calibrated torso deviation angles from vertical.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Seated Trunk Compensation Baseline**. Proposed use: candidate to calibrate torso angle thresholds (lean angle > 15 deg) to detect compensatory leaning during seated exercise.

---

### Category 3: Stroke Rehabilitation, Hemiparesis & Clinical Compensations

#### 14. SERE (StrokE Rehab Exercises)
- **Official Citation**: VisLab, Institute for Systems and Robotics, Instituto Superior Técnico, Lisbon (2024–2025). *SERE: A Comprehensive Multi-Sensor Stroke Rehabilitation Exercise Dataset with Frame-Level Clinical Compensations*.
- **Repository / Host**: Institutional Repository (ISR Lisbon).
- **Modalities & Setup**: Synchronized ZED stereo depth camera (yielding 3D skeletons) and high-resolution RGB video processed with MediaPipe 33-joint pose at 30 FPS.
- **Participants & Cohort**: 18–20 chronic post-stroke patients exhibiting varying degrees of hemiparesis and upper-limb motor limitation.
- **Exercises & Tasks**: 5 functional rehabilitation exercises: hair combing, teeth brushing, face washing, hip flexion, and putting on socks.
- **Supervision & Labels**: Video-level and frame-level compensatory movements (shoulder hiking, trunk lateral flexion, forward head tilt), range of motion (ROM) quality scores, and movement smoothness indices annotated by licensed physical therapists.
- **License & Access**: Institutional Data Use Agreement (DUA) with VisLab (`ana.coias@tecnico.ulisboa.pt`).
- **AdaptFit Proposed Use**: **Proposed Candidate for Quality & Compensation Heads**. Proposed use: subject to institutional DUA execution with VisLab Lisbon and adapter development, proposed to supply clinical compensation and ROM supervision mapped to `quality_logits` and `expert_quality_logits`. Note that frame-level annotations require a defined window/sequence temporal aggregation contract or must remain masked to comply with the pooled binary schema of `MovementPredictionV1`.

#### 15. Toronto Rehab Stroke Pose Dataset (TRSPD / TULE)
- **Official Citation**: Taati et al., KITE Research Institute, Toronto Rehabilitation Institute (2023). *TULE: Three Upper-Limb Exercises for Stroke Rehabilitation Kinematic Analysis*, IEEE TNSRE.
- **Repository / Host**: Kaggle / GitHub (`zhiderek/TRSPD`).
- **Modalities & Setup**: Microsoft Kinect v2 depth sensor recording 25-joint 3D/2D skeletal coordinate trajectories at 30 FPS.
- **Participants & Cohort**: 15 post-stroke individuals with mild-to-severe hemiparetic impairment alongside age-matched healthy controls.
- **Exercises & Tasks**: 3 upper-limb reaching and elevation exercises performed with both affected and unaffected arms.
- **Supervision & Labels**: Frame-by-frame clinical ratings of trunk lean, shoulder abduction compensation, and movement trajectory deviation.
- **License & Access**: Open Research Access via Kaggle.
- **AdaptFit Proposed Use**: **Proposed Candidate for Clinical Compensation Evaluation**. Proposed use: candidate to evaluate the trunk compensation head (applying temporal window aggregation) and calibrate asymmetry metrics between affected and unaffected limbs.

#### 16. Post-Stroke Kinematic & EMG Functional Tasks
- **Official Citation**: Nature Scientific Data (Dec 2025). *Upper-Limb Kinematic and Electromyographic Dataset of Post-Stroke Individuals During Functional Motor Tasks*, DOI: 10.1038/s41597-025-06174-3.
- **Repository / Host**: Figshare / Scientific Data Open Repository.
- **Modalities & Setup**: 3D Vicon optical motion capture (36 reflective markers) paired with 16-channel wireless surface EMG.
- **Participants & Cohort**: 10 post-stroke individuals (ages 62–82) and 10 healthy control participants (ages 24–73).
- **Exercises & Tasks**: 6 functional motor tasks: reaching forward to lift an object, placing objects at variable heights, touching the face, and forearm rotation.
- **Supervision & Labels**: Clinical Fugl-Meyer Upper Extremity (FMA-UE) scores, movement duration, peak velocity timestamps, spectral arc length (SPARC), and dimensionless jerk smoothness metrics.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Smoothness & Reach Quality Supervision**. Proposed use: candidate to provide optical ground truth to calibrate the mathematical SPARC smoothness index in `training/src/features/anatomy.py`.

#### 17. Upper Limb Stroke Rehabilitation Exercise Video Dataset
- **Official Citation**: Nandana et al., Elsevier Data in Brief (June 2026). *Home-Based Upper Limb Stroke Rehabilitation Exercise Video Dataset*, DOI: 10.1016/j.dib.2026.112819.
- **Repository / Host**: Mendeley Data / Elsevier.
- **Modalities & Setup**: 491 video clips (30 FPS RGB) captured across multiple smartphone models and webcams under natural home lighting.
- **Participants & Cohort**: 10 volunteer participants performing unconstrained home rehabilitation routines.
- **Exercises & Tasks**: 4 rehabilitation exercises: shoulder flexion, shoulder abduction, horizontal abduction, and elbow extension.
- **Supervision & Labels**: Repetition cycle timestamps, complete vs. incomplete execution labels, and user difficulty levels.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Partial Repetition & Failure Modeling**. Proposed use: candidate to train the causal TCN to detect aborted or incomplete repetitions without triggering false increment events once adapter is implemented.

#### 18. U-Limb Database
- **Official Citation**: GigaScience / Oxford Academic (2021–2024). *U-Limb: A Multi-Center Multimodal Database for Upper-Limb Rehabilitation Kinematics*, DOI: 10.1093/gigascience/giab043.
- **Repository / Host**: GigaDB Open Repository.
- **Modalities & Setup**: 3D optical motion capture, synchronized surface EMG, EEG, and robotic haptic interaction forces.
- **Participants & Cohort**: 156 participants: 91 able-bodied controls and 65 post-stroke individuals.
- **Exercises & Tasks**: Unilateral reaching, target grasping, and trajectory tracing.
- **Supervision & Labels**: Fugl-Meyer Assessment (FMA) scores, task segmentation intervals, and kinematic reach error.
- **License & Access**: Open Access Research Repository.
- **AdaptFit Proposed Use**: **Proposed Candidate for Cross-Subject Impairment Prior**. Proposed use: candidate to evaluate cross-subject generalization on unilateral arm reach trajectories.

#### 19. REHAB-120
- **Official Citation**: Figshare (2023). *Longitudinal Inertial Sensor Dataset of Upper-Limb Recovery in 120 Stroke Inpatients*.
- **Repository / Host**: Figshare Open Repository.
- **Modalities & Setup**: Wearable tri-axial IMU sensor recordings across 3 body segments.
- **Participants & Cohort**: 120 acute and subacute stroke inpatients tracked longitudinally across a 3-week rehabilitation program.
- **Exercises & Tasks**: 27 standardized clinical assessment tasks and 16 functional daily training tasks.
- **Supervision & Labels**: Longitudinal clinical score improvements, repetition counts, and movement speed profiles.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Longitudinal Rehab Progress Prior**. Proposed use: candidate to validate multi-session progression curves and tempo stabilization.

#### 20. Post-Stroke Upper Limb Kinematics
- **Official Citation**: Zenodo (2021). *Inertial Sensor Kinematics for Upper Limb Motor Recovery After Stroke*, DOI: 10.5281/zenodo.4705352.
- **Repository / Host**: Zenodo Open Archive.
- **Modalities & Setup**: Multi-sensor IMU kinematic trajectories.
- **Participants & Cohort**: 20 chronic stroke patients and 5 healthy controls.
- **Exercises & Tasks**: 30 everyday motor actions and exercise tasks.
- **Supervision & Labels**: Fugl-Meyer motor scores, joint acceleration profiles, and spectral smoothness scores.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Spectral Smoothness Calibration**. Proposed use: candidate to benchmark the smoothness metric against clinical impairment ratings.

#### 21. Full-Body Gait and Mobility in Stroke Survivors
- **Official Citation**: Figshare (2024). *Comprehensive Full-Body Biomechanics and Joint Kinematics in Post-Stroke Hemiparesis*.
- **Repository / Host**: Figshare Open Archive.
- **Modalities & Setup**: 3D optical motion capture, ground reaction force plates, and wireless surface EMG.
- **Participants & Cohort**: 50 stroke survivors and 138 able-bodied adult controls.
- **Exercises & Tasks**: Walking, sit-to-stand transfers, single-leg steps, and obstacle clearance.
- **Supervision & Labels**: Bilateral joint kinematics, ground reaction forces, and single-leg loading asymmetry indices.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Single-Leg Loading Asymmetry Anchor**. Proposed use: candidate to validate single-leg weight-bearing capability masks during lower-body routines once adapter is built.

#### 22. StrokeRehab
- **Official Citation**: SimTK (2023). *StrokeRehab: A Kinematic and Sensor Dataset of Upper-Body Functional Primitives in Hemiparetic Stroke*, SimTK Project 812.
- **Repository / Host**: [SimTK Project](https://simtk.org/projects/strokerehab).
- **Modalities & Setup**: Synchronized wearable IMU sensors and video-extracted kinematic time series (3,372 trials).
- **Participants & Cohort**: 51 stroke-impaired individuals and 20 age-matched healthy controls.
- **Exercises & Tasks**: Naturalistic activities of daily living decomposed into functional primitives: reach, transport, reposition, stabilize, and idle.
- **Supervision & Labels**: Fine-grained functional primitive state transition timestamps and impairment severity ratings.
- **License & Access**: Open Access via SimTK account.
- **AdaptFit Proposed Use**: **Proposed Candidate for Functional Primitive Pretraining**. Proposed use: candidate to train the causal TCN to distinguish between active exercise phases and passive arm stabilization.

#### 23. KIMORE (Kinect Motion Reconstruction)
- **Official Citation**: IEEE Transactions on Neural Systems and Rehabilitation Engineering (2019). *KIMORE: A Kinect-Based Dataset for Upper-Body Physical Rehabilitation*, DOI: 10.1109/TNSRE.2019.2923060.
- **Repository / Host**: IEEE Xplore / Institutional Google Drive.
- **Modalities & Setup**: Microsoft Kinect v2 RGB-D camera capturing 25-joint 3D skeletal time series at 30 FPS.
- **Participants & Cohort**: 78 participants: 44 healthy controls and 34 clinical patients with stroke, Parkinson's disease, or orthopedic low-back dysfunction.
- **Exercises & Tasks**: 5 rehabilitation exercises: lateral trunk flexion, pelvic rotation, trunk rotation, shoulder abduction, and seated knee extension.
- **Supervision & Labels**: Clinical physiotherapist scores (0–50 scale) and quantitative joint angle deviations.
- **License & Access**: Open for academic research use upon request.
- **AdaptFit Proposed Use**: **Proposed Candidate for Physician-Scored Quality Benchmark**. Proposed use: candidate to benchmark quality heads against physical therapist clinical ratings once license terms and label mappings are finalized.

#### 24. KERAAL Low-Back-Pain Rehabilitation Dataset
- **Official Citation**: IMT Atlantique (2020). *KERAAL: A Physical Rehabilitation Dataset for Low-Back-Pain Assessment*, Project Page.
- **Repository / Host**: [KERAAL Project](https://keraal.enstb.org/KeraalDataset.html).
- **Modalities & Setup**: Kinect v2 3D skeletal data, RGB video, and BlazePose 2D landmark trajectories at 30 FPS.
- **Participants & Cohort**: 21 participants: 9 healthy subjects and 12 clinical low-back pain patients.
- **Exercises & Tasks**: 3 rehabilitation exercises: torso flexion, lateral lean, and lunging.
- **Supervision & Labels**: Fine-grained clinical annotations by physical therapists indicating error type, affected body part, and exact start/end timestamps of compensatory movement.
- **License & Access**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).
- **AdaptFit Proposed Use**: **Proposed Candidate for Trunk Compensation & Error Timing Supervision**. Proposed use: candidate to supervise the trunk compensation quality head with clinical labels, applying temporal window aggregation to match the pooled binary schema.

---

### Category 4: Repetition Counting, Periodicity & Teacher Distillation

#### 25. RepCount & RepCount-pose
- **Official Citation**: CVPR 2022 (TransRAC) / CVPR 2023 (PoseRAC). *RepCount: A Large-Scale Video Dataset for Repetition Counting with BlazePose 33-Keypoint Annotations*.
- **Repository / Host**: GitHub (`SvipRepetitionCounting/TransRAC`, `MiracleDance/PoseRAC`).
- **Modalities & Setup**: 1,451 high-definition video sequences paired with 33-keypoint BlazePose 2D landmark sequences.
- **Participants & Cohort**: Diverse global workout enthusiasts, fitness athletes, and home trainees.
- **Exercises & Tasks**: Repetitive fitness and rehabilitation exercises: arm curls, band rows, squats, pull-ups, sit-ups, and lunges.
- **Supervision & Labels**: 20,000+ per-repetition boundary timestamps (cycle start and cycle end) and 2 salient pose states per cycle.
- **License & Access**: Academic Open Source Research License.
- **AdaptFit Proposed Use**: **Proposed Candidate for Phase 4 Repetition Teacher Supervision**. Proposed use: candidate to synthesize continuous repetition density maps for SSTRAC distillation and salient-state logits for PoseRAC distillation once teacher pipelines are staged.

#### 26. Countix & Countix-AV
- **Official Citation**: Google Research (CVPR 2020, Dwibedi et al.). *Counting Out Time: Class-Agnostic Video Repetition Counting in the Wild*.
- **Repository / Host**: Google Research Open Datasets / DeepMind.
- **Modalities & Setup**: ~5,000 video clips extracted from YouTube covering unconstrained real-world settings.
- **Exercises & Tasks**: Repetitive human activities including squats, push-ups, arm curls, barbell rows, and jumping jacks.
- **Supervision & Labels**: Total repetition count, cycle start/end timestamps, and active repetition intervals.
- **License & Access**: Creative Commons / Google Research Use Terms.
- **AdaptFit Proposed Use**: **Proposed Candidate for Periodicity & Self-Similarity Foundation**. Proposed use: candidate to pretrain the causal TCN temporal representation using self-supervised temporal similarity matrices.

#### 27. UCFRep
- **Official Citation**: CVPR 2020 (Zhang et al.). *Context-Aware Video Repetition Counting*.
- **Repository / Host**: GitHub Open Repository.
- **Modalities & Setup**: 526 video sequences curated from the UCF101 benchmark.
- **Exercises & Tasks**: 23 diverse cyclical human action categories.
- **Supervision & Labels**: Precise per-repetition boundaries and cycle counts across variable execution cadences.
- **License & Access**: Academic Open Source.
- **AdaptFit Proposed Use**: **Proposed Candidate for Variable-Scale Cadence Benchmark**. Proposed use: candidate to evaluate the Phase 1 Finite State Machine decoder across extreme speed variations.

#### 28. QUVA Repetition Dataset
- **Official Citation**: University of Amsterdam (CVPR 2018, Runia et al.). *Real-World Repetition Counting with Non-Stationary Dynamics*.
- **Repository / Host**: QUVA Deep Vision Lab Archive.
- **Modalities & Setup**: 100 in-the-wild video sequences capturing sudden speed changes, camera jitter, and view transitions.
- **Supervision & Labels**: Frame-accurate repetition counts and instantaneous cycle frequencies.
- **License & Access**: Academic Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Non-Stationary Cadence Testing**. Proposed use: candidate to test event debouncing and hysteresis logic during sudden accelerations and decelerations.

#### 29. Fitness-AQA
- **Official Citation**: Parmar et al. (2022). *Fitness-AQA: Action Quality Assessment for In-the-Wild Resistance Training*.
- **Repository / Host**: GitHub (`ParitoshParmar/Fitness-AQA`).
- **Modalities & Setup**: Unconstrained video clips of resistance training exercises.
- **Exercises & Tasks**: Back squat, overhead barbell press, and barbell row.
- **Supervision & Labels**: Fine-grained movement quality scores and body-part error classifications (e.g., knee collapse, excessive lumbar extension, asymmetrical elbow flare).
- **License & Access**: Non-Commercial Research License.
- **AdaptFit Proposed Use**: **Proposed Candidate for Row & Press Form Representation**. Proposed use: candidate to train fine-grained error detection on AdaptFit's one-arm row and seated press routines once adapter is built.

#### 30. QEVD / FIT-300K
- **Official Citation**: Qualcomm Research (2023). *QEVD: Qualcomm Exercise Video Dataset and FIT-300K Multimodal Fitness Corpus*.
- **Repository / Host**: Qualcomm Developer Network.
- **Modalities & Setup**: 474 hours of synchronized exercise video (289,000 short clips) covering 148 exercise categories and variants.
- **Supervision & Labels**: Coach corrective feedback, form variation categories, common execution mistakes, and pacing labels.
- **License & Access**: Qualcomm Research License Agreement.
- **AdaptFit Proposed Use**: **Proposed Candidate for Language & Form Adaptation Prior**. Proposed use: candidate to pretrain the form-feedback classifier on common exercise deviations.

---

### Category 5: Physical Therapy Kinematics, Posture & Biomechanical Foundations

#### 31. DynTherapy
- **Official Citation**: Jordan University of Science and Technology (2024). *DynTherapy: A 33-Keypoint MediaPipe Pose Dataset for Dynamic Physical Therapy Exercises*, Mendeley Data, DOI: 10.17632/hghdm99rwg.1.
- **Repository / Host**: Mendeley Data Open Repository.
- **Modalities & Setup**: **33 MediaPipe pose keypoints at 30 FPS**—identical to AdaptFit's canonical landmark schema!
- **Participants & Cohort**: Multi-subject cohort recorded across multiple rooms, camera distances, and lighting environments.
- **Exercises & Tasks**: 7 physical therapy exercises: knee raises, seated shoulder press, shoulder flexion, lateral raises, leg raises, glute bridges, and arm curls.
- **Supervision & Labels**: Repetition cycle timestamps, explicit "Start" and "End" boundary annotations, and movement family labels.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Priority Candidate for Drop-In Supervision**. Proposed use: 1:1 landmark correspondence makes it a primary candidate for adapter implementation in `training/src/data/adapters.py` without coordinate transformations, providing immediate repetition boundary supervision once integrated.

#### 32. UI-PRMD (University of Idaho Physical Rehabilitation Movement Dataset)
- **Official Citation**: University of Idaho (2018–2020). *UI-PRMD: A Non-Invasive Motion Capture Dataset for Physical Therapy Movement Assessment*, OpenDataLab.
- **Repository / Host**: University of Idaho / OpenDataLab.
- **Modalities & Setup**: Synchronized 10-camera Vicon optical motion capture (3D Cartesian coordinates and joint angles) and Microsoft Kinect v2 skeletal coordinates.
- **Participants & Cohort**: 10 healthy participants performing 10 optimal repetitions and 10 non-optimal (faulty) repetitions for every exercise.
- **Exercises & Tasks**: 10 physical therapy exercises: deep squat, lunge, seated sit-to-stand, shoulder abduction, shoulder internal/external rotation.
- **Supervision & Labels**: Full-body 3D positions, calculated joint angles, and binary optimal vs. non-optimal execution labels.
- **License & Access**: Open Access Research License.
- **AdaptFit Proposed Use**: **Proposed Candidate for PT Movement Modeling**. Proposed use: candidate dual-sensor data for correct vs. faulty repetition structures. Note that binary optimal/non-optimal labels represent overall movement correctness and must not be mapped into dimension-specific ROM or trunk quality heads (which remain masked per the feature contract).

#### 33. MobiPhysio
- **Official Citation**: Elsevier Data in Brief / ResearchGate (2024–2026). *MobiPhysio: A Mobile Smartphone Video Dataset for Active Range of Motion Physiotherapy Assessment*, DOI: 10.1016/j.dib.2026.112819.
- **Repository / Host**: Kaggle / Elsevier Data in Brief.
- **Modalities & Setup**: 2D RGB smartphone video (3,686 clips) recorded on real mobile devices under varying household conditions.
- **Participants & Cohort**: 58 participants evaluated by licensed physiotherapists.
- **Exercises & Tasks**: 9 Active Range-of-Motion (AROM) physiotherapy exercises.
- **Supervision & Labels**: Exercise Accuracy Assessment Questionnaire (EAAQ) scores certified by physical therapists and video temporal segmentation.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Smartphone Realism & Quality Calibration**. Proposed use: candidate to evaluate mobile camera pose jitter and benchmark thresholded ROM quality (or evaluate potential future continuous ROM regression schemas).

#### 34. Arm-CODA
- **Official Citation**: Combettes et al., Image Processing On Line (IPOL 2024). *Arm-CODA: Upper-Limb Kinematic Time Series Dataset*, DOI: 10.5201/ipol.2024.494.
- **Repository / Host**: IPOL Open Data Archive.
- **Modalities & Setup**: 3D Cartesian Optoelectronic Dynamic Anthropometer (CODA) tracking 34 active anatomical markers at 100 Hz (240 time series, 2.5 hours).
- **Participants & Cohort**: 16 healthy adult subjects.
- **Exercises & Tasks**: 15 routine upper-limb movements including arm raises, hair combing, forward reach, and circular tracking.
- **Supervision & Labels**: Exact cycle start/end timestamps (>= 2 iterations per sequence), movement speed, smoothness, and trajectory efficiency.
- **License & Access**: Open Access / Open Data.
- **AdaptFit Proposed Use**: **Proposed Candidate for Millisecond Biomechanical Reference**. Proposed use: candidate to calibrate temporal smoothness and tempo metrics on millisecond-accurate optical ground truth.

#### 35. Physical Therapy Exercises Dataset
- **Official Citation**: UCI Machine Learning Repository (2022). *Wearable Inertial Sensor Dataset for Physical Therapy Form Classification*.
- **Repository / Host**: UCI ML Repository / Kaggle.
- **Modalities & Setup**: 5 tri-axial Xsens IMU sensors placed on the chest, upper arms, and forearms recording at 25 Hz.
- **Participants & Cohort**: 5 participants performing 8 standardized physical therapy exercises.
- **Supervision & Labels**: 3 execution styles: correct execution, excessively fast tempo, and truncated range of motion.
- **License & Access**: Open Access UCI License.
- **AdaptFit Proposed Use**: **Proposed Candidate for Tempo & ROM Perturbation Prior**. Proposed use: candidate to calibrate threshold boundaries for detecting rushed repetitions and incomplete range of motion.

#### 36. PhysioNet Posture & Gait Analysis
- **Official Citation**: Palermo et al., PhysioNet (Nov 2021). *Multimodal Motion Capture Dataset for Posture and Gait Analysis During Smart Walker Use*, DOI: 10.13026/fyxw-n385.
- **Repository / Host**: PhysioNet Open Repository.
- **Modalities & Setup**: Synchronized dual depth cameras and 17-sensor Xsens MTw Awinda mocap (166,000 frames).
- **Participants & Cohort**: Clinical mobility participants using assistive walking and support devices.
- **Supervision & Labels**: 3D joint centers, posture tilt angles, and mobility device interaction intervals.
- **License & Access**: PhysioNet Contributor License.
- **AdaptFit Proposed Use**: **Proposed Candidate for Mobility Aid Occlusion Modeling**. Proposed use: candidate to model camera occlusions caused by walkers, chair arms, and seated support frames.

#### 37. GaitEncoder Dataset
- **Official Citation**: Stanford University / SimTK (2024–2025). *GaitEncoder: Large-Scale Foundation Kinematic Dataset Across Diverse Movement Pathologies*, SimTK Project.
- **Repository / Host**: [SimTK Project](https://simtk.org/projects/gaitencoder).
- **Modalities & Setup**: Markered optical mocap paired with markerless OpenCap video kinematics.
- **Participants & Cohort**: 657 individuals (ages 8–86) representing 7 clinical pathologies including cerebral palsy, stroke, and amputations.
- **Supervision & Labels**: Scaled OpenSim musculoskeletal kinematic models and pathology classifications.
- **License & Access**: Open Research Access via SimTK.
- **AdaptFit Proposed Use**: **Proposed Candidate for Markerless Clinical Foundation**. Proposed use: candidate for kinematic representation pretraining across diverse clinical pathologies.

#### 38. AddBiomechanics Dataset 1.0
- **Official Citation**: Stanford University (2023–2025). *AddBiomechanics: A Standardized Open Database of 10,000+ Human Motion Trials and Musculoskeletal Dynamics*, `addbiomechanics.org`.
- **Repository / Host**: AddBiomechanics / SimTK.
- **Modalities & Setup**: 10,000+ optical mocap trials, 70+ hours of full-body inverse kinematics and joint torques.
- **Supervision & Labels**: Musculoskeletal geometry, 3D joint angles, angular velocities, and joint reaction forces.
- **License & Access**: CC BY 4.0 Open Access.
- **AdaptFit Proposed Use**: **Proposed Candidate for Anatomical Consistency Prior**. Proposed use: candidate to enforce physical angular velocity limits ($|\omega| < 720^\circ/\text{s}$) to prevent unphysiological pose predictions.

#### 39. LLM-FMS
- **Official Citation**: Xing et al., PLOS ONE (March 2025). *LLM-FMS: Large Language Models with Computer Vision for Functional Movement Screen Assessment*, DOI: 10.1371/journal.pone.0318973.
- **Repository / Host**: PLOS ONE / ResearchGate.
- **Modalities & Setup**: 2D RTMPose keypoints and synchronized RGB frames (1,812 frames).
- **Participants & Cohort**: 45 subjects performing 7 Functional Movement Screen (FMS) patterns.
- **Supervision & Labels**: Hierarchical movement quality scores (0–3), body-segment error tags, and compensatory movement rules.
- **License & Access**: Open Access Research.
- **AdaptFit Proposed Use**: **Proposed Candidate for Interpretable Compensation Rules**. Proposed use: candidate to map 2D landmark deviations to actionable form feedback advice.

#### 40. AHA-3D Dataset
- **Official Citation**: Antunes et al., BMVC (2018). *AHA-3D: A Kinect v2 Skeleton Dataset for Active and Healthy Aging Fitness Assessment*, CMU Portugal / Univ. Lisbon.
- **Repository / Host**: Institutional Archive / BMVA.
- **Modalities & Setup**: Microsoft Kinect v2 3D skeletal joint sequences at 30 FPS.
- **Participants & Cohort**: 21 subjects: 11 young adults and 10 elderly adults.
- **Exercises & Tasks**: Standardized senior fitness test battery: chair sit-to-stand, arm curls, 2-minute step-in-place.
- **Supervision & Labels**: Frame-accurate action segmentation, repetition boundary timestamps, and functional fitness scores.
- **License & Access**: Research Access on Request.
- **AdaptFit Proposed Use**: **Proposed Candidate for Slow-Tempo & Elderly Mobility Baseline**. Proposed use: candidate to evaluate decoder robustness on slow repetitions ($T_{cycle} > 4.5\text{s}$) that exceed the standard 128-frame context window.

---

## Recommended Acquisition & Ingestion Sequence

To maximize model performance while strictly observing licensing boundaries, dataset acquisition is staged into three chronological cohorts:

### Tier 1: Immediate Acquisition & Drop-In Ingestion (Sprint 1)
1. **DynTherapy** (Mendeley Data / CC BY 4.0): Direct 33-keypoint MediaPipe correspondence. Write adapter in `training/src/data/adapters.py#load_dyntherapy`. Unlocks repetition boundary and phase supervision for 7 PT exercises.
2. **UI-PRMD** (University of Idaho / Open Access): Proposed candidate for 10 PT exercises with optimal vs. non-optimal labels (quality heads remain masked; correctness != ROM/trunk quality). Propose writing 3D-to-2D projection adapter.
3. **Pipelines Open Dataset** (Figshare / CC BY 4.0): Proposed candidate to benchmark synchronized wheelchair propulsion cycles and validate markerless 2D pose accuracy against 3D ground truth.
4. **Ottobock #DearAI** (Hugging Face / Community Open): Proposed candidate to evaluate upper/lower limb difference imagery in pose-front-end detector test suite.
5. **ROAG** (Zenodo / CC BY 4.0): Proposed candidate to evaluate transradial amputee reaching trajectories, calibrate reach geometry, and test thresholded binary trunk tilt.

### Tier 2: Institutional DUAs & Clinical Compensation Datasets (Sprint 2)
1. **SERE** (VisLab ISR Lisbon): Execute DUA with `ana.coias@tecnico.ulisboa.pt`. Proposed candidate to evaluate 18–20 post-stroke 3D skeletons with therapist-graded trunk compensation.
2. **TULE / TRSPD** (Kaggle / Toronto Rehab): Proposed candidate to evaluate 15 stroke survivors with frame-level trunk lean and shoulder hiking labels.
3. **MobiPhysio** (Kaggle / CC BY 4.0): Proposed candidate to evaluate 3,686 smartphone video clips to test real mobile camera pose jitter and EAAQ quality scores.
4. **InclusiveVidPose** (ICLR 2025 / DUA): Submit non-commercial research DUA to obtain 398 limb-difference video sequences for pose-front-end validation.

### Tier 3: Teacher Distillation & Foundation Pretraining (Sprint 3)
1. **RepCount-pose** (GitHub / Academic Open): Cache 20,000 repetition cycles to generate continuous repetition density maps for SSTRAC and PoseRAC teacher distillation.
2. **AddBiomechanics** (Stanford / CC BY 4.0): Proposed candidate to evaluate inverse kinematics and calibrate angular velocity and physical smoothness constraints.
3. **SAFER-Activities** (Hugging Face / CC BY-NC-SA): Isolate dedicated wheelchair challenge test partition.

---


### Category 6: Expanded Candidate Registries (Wheelchair, Amputee, Stroke & Repetition)

#### 34. University of Groningen Wheelchair Racing Ergometer Dataset
- **Official Citation**: de Klerk, R., van der Jagt, G., Veeger, D.H.E.J., van der Woude, L.H.V., Vegter, R.J.K. (Center for Human Movement Sciences, University of Groningen, 2022). *Three weeks of wheelchair racing propulsion practice in able-bodied novices*. Front. Sports Act. Living.
- **Repository / DOI**: University of Groningen Dataverse, [DOI: 10.34894/ebjbmf](https://doi.org/10.34894/ebjbmf).
- **License / Access**: Open Data Access for non-commercial academic research (CC BY-NC 4.0).
- **Modalities & Setup**: Custom-instrumented wheelchair racing ergometer with synchronized 3D optical kinematics (Optotrak/Vicon upper-extremity joint centers), bilateral wheel torque/power transducers, and respiratory gas exchange.
- **Participants & Cohort**: 15 novice individuals tracked longitudinally across 3 weeks of high-speed racing propulsion practice.
- **Exercise Types**: Submaximal steady-state propulsion trials and maximal sprint acceleration bouts on racing pushrims.
- **Annotations**: Millisecond-accurate push phase onset and release timestamps, stroke frequency/cadence, work per stroke, push angle range, propulsion trajectory smoothness.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `direct_temporal` and `motion_pretraining` (high-cadence wheelchair propulsion pushrim cycle segmentation and power/cadence prior for wheelchair workout routines).

#### 35. Loughborough University Wheelchair Sprint Shoulder Kinematics Dataset
- **Official Citation**: Briley, S.J., Vegter, R.J.K., Goosey-Tolfrey, V.L., Mason, B.S. (Peter Harrison Centre for Disability Sport, Loughborough University, 2022). *Alterations in shoulder kinematics are associated with shoulder pain during wheelchair propulsion sprints*, Scand. J. Med. Sci. Sports.
- **Repository / DOI**: Loughborough University Repository / Figshare, [DOI: 10.17028/rd.lboro.21118741.v1](https://doi.org/10.17028/rd.lboro.21118741.v1).
- **License / Access**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Modalities & Setup**: Multi-camera 3D optical motion capture (Vicon) tracking thorax, scapula, clavicle, and humerus kinematics.
- **Participants & Cohort**: Wheelchair athletes performing maximal-effort wheelchair propulsion sprints.
- **Exercise Types**: Maximal-velocity wheelchair propulsion sprints and submaximal propulsion bouts.
- **Annotations**: Glenohumeral abduction/flexion angles, scapular internal/external rotation, propulsion cycle acceleration phase vs. maximal velocity phase segmentation, and Wheelchair User's Shoulder Pain Index (WUSPI) scores.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `quality_compensation` and `evaluation_challenge` (evaluating shoulder impingement risks and excessive humeral abduction compensations during wheelchair push cycles).

#### 36. Wheelchair Court Sports Mobility Performance Dataset
- **Official Citation**: van der Slikke, R.M.A., de Groot, S., van der Woude, L.H.V., Hoekstra, A.E., Vegter, R.J.K., Rietveld, T. (The Hague University of Applied Sciences / TU Delft / VU Amsterdam, 2019). *Wheelchair mobility performance of elite wheelchair tennis players during four field tests: Inter-trial reliability and construct validity*, PLOS ONE.
- **Repository / DOI**: Figshare, [DOI: 10.6084/m9.figshare.8237906](https://doi.org/10.6084/m9.figshare.8237906).
- **License / Access**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Modalities & Setup**: Synchronized 3-IMU setup (2 on wheel hubs at 200 Hz, 1 on wheelchair frame at 100 Hz).
- **Participants & Cohort**: Elite international wheelchair court athletes.
- **Exercise Types**: Standardized court agility field tests: Spider agility drill, Illinois agility test, 20-meter linear sprints, and 360-degree pivot rotations.
- **Annotations**: Linear acceleration, rotational velocity ($\omega_{yaw}$), turn entry/exit timestamps, and mobility performance outcomes.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `evaluation_challenge` and `motion_pretraining` (validating rotational velocity bounds $|\omega| < 450^\circ/\text{s}$ and seated frame orientation stability during vigorous wheelchair sports and fitness drills).

#### 37. University of Utah Stand-Up and Sit-Down Above-Knee Amputees Dataset
- **Official Citation**: Hunt, G., Gabert, L., Hansen, C., Foreman, K.B., Lenzi, T. (Bionic Engineering Lab, University of Utah). *Open dataset of kinetics, kinematics, and electromyography of above-knee amputees during stand-up and sit-down*, Nature Scientific Data, 12, Article 292 (March 2025).
- **Repository / DOI**: Figshare (ID: 27986016), [DOI: 10.1038/s41597-025-04695-5](https://doi.org/10.1038/s41597-025-04695-5).
- **License / Access**: Creative Commons Attribution 4.0 International (CC BY 4.0). Fully open download.
- **Modalities & Setup**: Synchronized 12-camera Vicon optical motion capture (3D reflective markers, C3D), 2 dual embedded AMTI force plates, 4 wireless surface EMG sensors on the intact limb, and synchronized high-definition video.
- **Participants & Cohort**: 9 individuals with unilateral above-knee (transfemoral) amputations using their prescribed microprocessor-controlled (MPK) or passive prosthetic knee joints.
- **Exercise Types**: Repetitive sit-to-stand and stand-to-sit transfers from standardized chair seating heights.
- **Annotations**: Movement initiation and completion timestamps (event segmentation), bilateral joint kinematics (hip flexion/extension, knee flexion/extension, ankle dorsiflexion), ground reaction force vertical peak asymmetry, and sound-vs-prosthetic loading ratios.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `direct_temporal` and `quality_compensation` (projected via 3D-to-2D virtual camera to train single-leg sit-to-stand repetition phase boundaries, trunk forward lean compensation $\theta_{trunk} \ge 15^\circ$, and asymmetric weight-bearing capability masks).

#### 38. ULTRA-MoCap: Multimodal Upper Limb Tracking Dataset
- **Official Citation**: Fritsche, O.K., et al. (REAL Lab, University of Central Florida, 2026). *ULTRA-MoCap: A Multimodal IMU and sEMG Dataset for Upper Body Joint Kinematics Analysis*, Nature Scientific Data.
- **Repository / DOI**: Figshare, [DOI: 10.6084/m9.figshare.28751156.v1](https://doi.org/10.6084/m9.figshare.28751156.v1); GitHub: `oliverkristianfritsche/MocapDatasetScripting_REALLAB`.
- **License / Access**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Modalities & Setup**: Vicon Vero multi-camera optical motion capture (millimeter-accurate 3D Cartesian markers), OpenSim multi-DOF musculoskeletal inverse kinematics, synchronized 6-DOF IMUs (hand, wrist, forearm), and combined sEMG/IMU sensors (biceps brachii, triceps brachii, deltoid).
- **Participants & Cohort**: 13 adult participants performing standardized upper-limb functional and therapeutic tasks.
- **Exercise Types**: 5 unilateral upper-limb movements: overhead reach, elbow flexion (biceps curl template), shoulder internal/external rotation, crossbody reach, and sagittal armswing.
- **Annotations**: Continuous 3D joint angles, repetition start/inflection/end timestamps, muscle activation envelopes, and kinematic smoothness (SPARC).
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `direct_temporal` and `quality_compensation` (virtual camera projection supplies clean single-arm reach and curl trajectories with verified physical joint angle bounds and millisecond ground-truth phase inflection labels).

#### 39. CARRT Robotic Human Upper-Body Motion Capture Dataset
- **Official Citation**: Center for Assistive, Rehabilitation & Robotics Technologies (CARRT), University of South Florida (2021–2023). *CARRT—Motion Capture Data for Robotic Human Upper Body Model*, Sensors 2023 (DOI: 10.3390/s23208354).
- **Repository / DOI**: Zenodo, [DOI: 10.5281/zenodo.8034000](https://doi.org/10.5281/zenodo.8034000) (Concept DOI: [10.5281/zenodo.8032646](https://doi.org/10.5281/zenodo.8032646)).
- **License / Access**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Modalities & Setup**: Vicon 3D optical motion capture (Cartesian marker trajectories, C3D) formatted for OpenSim kinematic models (`.trc`) and MATLAB.
- **Participants & Cohort**: 10 adult participants performing functional assistive upper-limb manipulation and therapy routines (340 demonstrations).
- **Exercise Types**: 9 activities of daily living (reaching forward, drinking, door knob turning, overhead object placement) and 8 functional range-of-motion (ROM) activities.
- **Annotations**: 3D joint centers, shoulder/elbow/wrist angle time series, movement phase cycles, and reach target coordinates.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `motion_pretraining` and `quality_compensation` (pretraining latent representations on unilateral upper-extremity reaching tasks and establishing healthy ROM bounds for arm elevation).

#### 40. Transhumeral Loading During Advanced Upper Extremity ADLs
- **Official Citation**: Zenodo Biomechanics Collection (2023). *Transhumeral Loading During Advanced Upper Extremity Activities of Daily Living*, Zenodo.
- **Repository / DOI**: Zenodo, [DOI: 10.5281/zenodo.1040453](https://doi.org/10.5281/zenodo.1040453).
- **License / Access**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Modalities & Setup**: Marker-based upper extremity optical motion capture tracking shoulder girdle, clavicle, thorax, and arm segments.
- **Participants & Cohort**: Non-amputee participants fitted with transhumeral immobilizers and prosthesis simulators to record dynamic loading and compensatory movement.
- **Exercise Types**: Dynamic upper-extremity activities of daily living applying axial and bending loads to the humerus segment.
- **Annotations**: Thorax lateral lean, scapular upward rotation, glenohumeral elevation angles, and joint kinetics.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `quality_compensation` (modeling trunk lateral lean and shoulder hiking compensations when distal upper-limb anatomy is restricted or absent).

#### 41. PrimSeq / StrokeRehab Functional Motion Primitives & Dose Dataset
- **Official Citation**: Schambra, H.M., et al. (Schambra Lab, NYU Langone Health, 2021–2024). *PrimSeq: A deep learning-based pipeline to quantitate rehabilitation training*. PLOS Digital Health / OpenReview.
- **Repository / DOI**: SimTK Project: `primseq` ([https://simtk.org/projects/primseq](https://simtk.org/projects/primseq)); GitHub: `schambra-lab/primseq`.
- **License / Access**: Open Research Access via SimTK terms (free academic registration).
- **Modalities & Setup**: 9 synchronized wearable 9-axis IMUs paired with multi-camera video streams.
- **Participants & Cohort**: Post-stroke individuals with upper-limb hemiparesis and healthy controls performing functional rehabilitation activities.
- **Exercise Types**: Unconstrained functional rehabilitation tasks and upper-extremity therapeutic drills.
- **Annotations**: Fine-grained millisecond time-series annotations of 5 functional movement primitives: `reach`, `reposition`, `transport`, `stabilize`, and `idle`, along with primitive repetition counts.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `direct_temporal` and `evaluation_challenge` (gold standard benchmark for sub-repetition primitive segmentation and clinical repetition dose counting in stroke hemiparesis).

#### 42. Rehab-Pile Benchmark Suite for Human Motion Rehabilitation Assessment
- **Citation / verification**: Primary citation and versioned release identifier are pending verification. Project page: [https://msd-irimas.github.io/pages/DeepRehabPile/](https://msd-irimas.github.io/pages/DeepRehabPile/).
- **Repository / DOI**: GitHub (`msd-irimas/DeepRehabPile`); `aeon-toolkit` (`load_rehab_pile_dataset`); PyPI: `deep-rehab-pile`.
- **License / Access**: Pending primary-source verification; do not treat as cleared for AdaptFit training or redistribution.
- **Modalities & Setup**: Standardized skeleton-based motion time series across video and inertial sensors.
- **Participants & Cohort**: Harmonized multi-cohort rehabilitation benchmark aggregating 8 primary repositories into a unified evaluation suite.
- **Exercise Types**: 39 distinct classification problem sets and 21 extrinsic regression problem sets spanning upper-limb, lower-limb, and trunk rehabilitation exercises.
- **Annotations**: Discrete clinical movement quality scores, error categories, and continuous motor performance indices.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `quality_compensation` and `evaluation_challenge` (standardized evaluation benchmark for testing whether AdaptFit quality heads generalize across multiple clinical datasets).

#### 43. STRIDE: Stroke Initiative for Gait Data Evaluation Database
- **Official Citation**: Sánchez, N., et al. (Multi-Center Rehabilitation Consortium / USC / Chapman University, 2022–2024). *STRIDE: Stroke Initiative for Gait Data Evaluation Database*, ICPSR 38002.
- **Repository / DOI**: ICPSR Data Repository, [DOI: 10.3886/ICPSR38002.v2](https://doi.org/10.3886/ICPSR38002.v2).
- **License / Access**: Open Academic Research Access (de-identified clinical data).
- **Modalities & Setup**: Harmonized multi-center optical motion capture, synchronized force plates, and standardized clinical impairment batteries.
- **Participants & Cohort**: 300+ post-stroke individuals exhibiting hemiparetic motor deficits across multiple US rehabilitation centers.
- **Exercise Types**: Overground walking, obstacle step-over, and postural transition tasks.
- **Annotations**: Bilateral joint kinematics, ground reaction forces, stance/swing phase asymmetry indices, Fugl-Meyer motor assessment scores.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `evaluation_challenge` (evaluating unilateral lower-limb asymmetry and single-leg loading capability masks in hemiparetic movement without risking data leakage).

#### 44. Park et al. Stroke Rehabilitation Exercise Data with Kinect 3D Depth and IMU
- **Official Citation**: Park, E., et al. (2021–2024). *Stroke Rehabilitation Exercise Data Utilizing 3D Depth Sensors and IMU Sensors*, Mendeley Data, [DOI: 10.17632/ygpdzx52g2.1](https://doi.org/10.17632/ygpdzx52g2.1) / Frontiers in Bioengineering and Biotechnology.
- **Repository / DOI**: Mendeley Data, [DOI: 10.17632/ygpdzx52g2.1](https://doi.org/10.17632/ygpdzx52g2.1); NIAID data record: `mendeley_ygpdzx52g2`.
- **License / Access**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Modalities & Setup**: Microsoft Kinect v2 3D skeletal data (25 joint centers at 30 FPS) paired with multi-sensor tri-axial IMUs.
- **Participants & Cohort**: 128 clinical post-stroke and mobility-impaired participants (631 recorded movement sequences).
- **Exercise Types**: 5 physical rehabilitation exercises: (1) arm lifting, (2) lateral trunk tilt, (3) trunk rotation, (4) pelvis rotation, (5) squatting.
- **Annotations**: Primary Outcome (PO, clinician execution quality score 0–100) and Control Factor (CF, physical impairment/spasticity constraints).
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `quality_compensation` and `direct_temporal` (maps 25 Kinect joints to canonical 33 MediaPipe layout, provides repetition boundaries for clinical cohorts, and calibrates `expert_quality_logits` and trunk compensation against clinician-certified Primary Outcome ratings).

#### 45. OpenCap 100-Subject Movement Dynamics Dataset
- **Official Citation**: Uhlrich, S.D., Falisse, A., Kidziński, Ł., Muccini, J., Ko, M., Chaudhari, A.S., Delp, S.L. (Stanford University, 2023). *OpenCap: Human movement dynamics from smartphone videos*, PLOS Computational Biology 19(10): e1011462 (Corrected DOI: [10.1371/journal.pcbi.1011462](https://doi.org/10.1371/journal.pcbi.1011462)).
- **Repository / DOI**: SimTK Project: `opencap` ([https://simtk.org/projects/opencap](https://simtk.org/projects/opencap)); DOI: [10.1371/journal.pcbi.1011462](https://doi.org/10.1371/journal.pcbi.1011462).
- **License / Access**: Apache 2.0 for code; SimTK Open Research Terms (academic and non-commercial research).
- **Modalities & Setup**: Dual smartphone monocular video (iOS) processed into 3D skeletal dynamics and musculoskeletal inverse kinematics via OpenSim.
- **Participants & Cohort**: 100 individuals recorded in unconstrained environments by non-expert clinicians.
- **Exercise Types**: Natural squats, deliberately asymmetric squats (simulating unilateral impairment), vertical jumps, and walking.
- **Annotations**: 3D joint kinematics, joint moments, ground reaction forces, and bilateral knee extension moment symmetry indices.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `evaluation_challenge` and `motion_pretraining` (mobile smartphone camera parity validation against lab mocap; gold-standard anchor for single-leg and asymmetric movement execution in smartphone environments).

#### 46. VSRep: Multimodal Video and 3D Skeleton Repetitive Action Benchmark
- **Official Citation**: IET Computer Vision / ResearchGate (2023–2024). *VSRep: A Novel Multimodal Video and 3D Skeleton Benchmark for Fine-Grained Human Repetitive Action Counting*.
- **Repository / DOI**: Open Access Research Archive; IET Digital Library.
- **License / Access**: Open Access Research License.
- **Modalities & Setup**: Synchronized RGB video and lifted 3D skeleton keypoint trajectories.
- **Participants & Cohort**: Trainees performing repeated fitness and rehabilitation actions under varying view angles.
- **Exercise Types**: Fine-grained repetitive fitness and rehabilitation drills (shoulder raises, arm curls, squats, knee lifts).
- **Annotations**: Sub-action phase segmentation, repetition cycle start/end boundaries, instantaneous cadence.
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for `direct_temporal` and `evaluation_challenge` (evaluating repetition boundary detection and cadence debouncing on pure skeleton input).

#### 47. PoseRAC: RepCount-pose & UCFRep-pose (Teacher Distillation Benchmark)
- **Official Citation**: Yao, Z., Cheng, X., Zou, Y. (Peking University, 2023). *PoseRAC: Pose Saliency Transformer for Repetitive Action Counting*, CVPR 2023; arXiv:2303.08450.
- **Repository / DOI**: GitHub (`MiracleDance/PoseRAC`).
- **License / Access**: Academic Open Source (MIT).
- **Modalities & Setup**: 33-keypoint BlazePose 2D landmark sequences at 30 FPS.
- **Participants & Cohort**: 1,451 videos in RepCount and 526 videos in UCFRep.
- **Exercise Types**: Cyclical exercise movements: arm curls, pull-ups, barbell/dumbbell rows, squats, lunges, push-ups.
- **Annotations**: 20,000+ repetition start/end timestamps and 2 salient inflection poses per cycle: maximum concentric contraction (peak effort / apex inflection) and maximum eccentric extension (turnaround inflection).
- **AdaptFit Proposed Use**: **Proposed Candidate / Not Integrated Code**. Proposed for Phase 4 Teacher Distillation (`training/src/distill/` planned; distills salient-state apex logits via KL divergence to `data/teacher_cache/poserac_salient/` to boost repetition boundary precision without adding runtime compute).

## What These Datasets Still Cannot Provide

No discovered public dataset covers the full first-release population of individuals with missing limbs, one-arm use, lower-limb absence, and wheelchair use performing the same five AdaptFit launch exercises under consented product-testing conditions.

Public data bootstraps the pose front end and the causal TCN movement-family and phase representations. While output heads for the four quality dimensions (ROM, tempo, smoothness, trunk compensation) exist architecturally in `MovementPredictionV1`, they currently have zero label coverage in existing public data and remain unsupervised. Enabling and supervising these heads and the safety-critical adaptation layer requires dedicated expert-labeled quality data and consented, participant-reviewed pilot recordings, as detailed in Phase 6 of the [implementation plan toward the Congressional App Challenge](project-forward-plan.md).
