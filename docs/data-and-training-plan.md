# Data and Training Plan

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

The initial prototype target is approximately 15–25 participants, five exercises, and 50–100 repetitions per exercise if a consented dataset becomes available. Diversity of participants matters more than collecting many clips from the same person.

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
