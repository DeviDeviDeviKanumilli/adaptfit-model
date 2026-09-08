# AdaptFit: implementation plan toward the Congressional App Challenge

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** dependency-ordered delivery plan grounded in current-state evidence
> - **Last verified:** 2026-09-07
> - **Source commit:** `e75ba65`
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** canonical roadmap; implementation-roadmap is retained as a superseded redirect
> - **Review trigger:** work-package completion, dependency change, gate result, or evidence that changes scope

Planning review: September 7, 2026. Audited model checkout: `e75ba65`.

## Outcome and scope

Build a demonstrable on-device workout flow: self-reported capabilities and equipment → compatible routine → camera-assisted exercise → reliable rep events or explicit manual fallback → compatible substitution → saved progress. The broader goal includes disabilities, injuries, pain, and mobility limits; the recorded first-release scope is five seated, unilateral-friendly exercises: one-arm curl, one-arm band row, single-leg knee extension, single-leg march, and forward reach. These are candidate recipes requiring review, not assertions of suitability for every user.

The supplied PDF and the Google Doc's **Final Idea** tab supply the problem statement; **Technical Impl** supplies the intended AI pipeline. There is no separate tab named “Problem Doc” in the matching Google Doc. Their proposed work is planning input, not authorization to train, deploy, recruit participants, or publish a submission during this review.

The active workspace `/Users/devk/Documents/ChatGPT/AdaptFit` is an empty Git repository with no remote or commits. The actual model project is `/Users/devk/AdaptFit`, remote `DeviDeviDeviKanumilli/adaptfit-model`. The earlier application is `/Users/devk/Downloads/PeddieHacks`, remote `DeviDeviDeviKanumilli/PeddieHacks26`. Follow the active model repo's README and recorded decisions: build the AdaptFit application under `/Users/devk/AdaptFit`; use PeddieHacks as reference, without nesting or modifying it as part of this plan.

## What exists, and what remains unproven

| Area | Verified state | Next action |
| --- | --- | --- |
| Pose features | Canonical 33-joint **2D** contract; 283 features including positions, velocities, angles/angular velocities, confidence, observed/capability masks, profile and posture | Freeze v1 ordering, normalization and missing-value semantics; preserve full native landmark input |
| Movement networks | Shared causal TCN, 96 channels, five blocks; causal GRU baseline; multi-task heads; Python streaming harness | Extend and validate these models rather than create a model per exercise |
| Output heads | Six families, five phases, start/end boundaries, four quality dimensions, tracking; optional five-class composite quality head | Add density and exercise conditioning; restrict feedback to validated targets |
| Research benchmark | Corrected-v1 artifacts exist; TCN has 307,410 parameters | Preserve this baseline and distinguish it from later incomplete runs |
| Quality supervision | Corrected-v1 dimension-specific label coverage is zero | Do not turn output tensors or UCO composite labels into claims of validated form feedback |
| Recommendation | No separate trained neural recommender in the model project | Build catalog, hard compatibility rules, history events, then embedding ranker |
| Deployment | Python causal inference reference exists | Native mobile feature parity, export, quantization and device benchmarks still required |

The corrected-v1 TCN report covers 617 logical test sequences and 7,592 windows: family macro-F1 87.57%, phase macro-F1 58.26%, repetition-start F1 99.54%, repetition-end F1 18.47%, count MAE 0.31. These are saved public-data research results, not fresh training or real target-population validation. The strong count figure does not establish reliable end detection or online counting. The previous sequence-identity collision issue was corrected; retain its regression checks instead of treating it as an unfixed task.

Fresh artifact inspection resolves the newer runs: `v2-quality` has preparation artifacts only. `v2-quality-fixed` has a completed 72-epoch TCN training loop (best epoch 42) and window-level results, but no merged sequence evaluation or complete comparison report. Its GRU checkpoint is intermediate: the log ends after interruption at epoch 52, with no final GRU history/evaluation. Evaluate the completed TCN under the matching feature/label policy; treat GRU completion or retraining as separate work. Do not describe either v2 directory as a completed TCN/GRU benchmark.

The previous app explains the integration gap: Android emits selected angles and aggregate confidence, iOS pose availability is stubbed, the counter is bilateral-first, and only the seated curl recipe is marked calibrated. Its catalog's curl/row requirements include both grips. Relevant **reference-only** files are under `/Users/devk/Downloads/PeddieHacks`: `apps/mobile/modules/adaptfit-pose`, `apps/mobile/src/lib/tracking/{analyzer,recipes}.ts`, and `packages/domain/src/{catalog,compatibility,generation}.ts`. These paths are not present in the active model repository. The new app must deliberately support one-sided capabilities rather than inherit those assumptions. The model repo itself consumes pose data; it does not currently run MediaPipe.

Code anchors: `training/src/data/schema.py`, `training/src/features/anatomy.py`, `training/src/models/{tcn,gru,heads,streaming}.py`, `training/src/losses.py`. Benchmark evidence: `artifacts/corrected-v1/metrics/tcn_evaluation.json`; provenance and limitations: `docs/verified-research-and-run-findings.md` and `docs/audit-fixes-and-overnight.md`. Existing tests passed: 123, as run during this review. Passing tests do not establish model accuracy on intended users.

## Work packages and dependency order

Role labels below are responsibilities to assign, not named commitments. Dates and numeric gates are proposed planning targets unless explicitly attributed to the competition.

### P0 — Establish one reproducible baseline and one product contract

**Owner: ML + product. First 2–3 working days.**

1. Inventory corrected-v1, v2-quality and v2-quality-fixed separately. Record checkpoint/config/schema/normalization hashes, completed epochs, selected epoch, evaluation status and label-policy version. Checkpoint presence alone is not proof of a completed comparison. Evaluate existing usable checkpoints before launching fresh training; use a new artifact destination so historical results survive.
2. Re-run the bounded preflight and prepared-data audit for the chosen config. Preserve participant-level splits, source-stable sequence IDs, parent/augmentation grouping, train-only normalization and label provenance. Exclude procedural/synthetic examples from the headline real-data results and report them separately.
3. Freeze `PoseFrameV1`, `FeatureSchemaV1`, `CapabilityProfileV1`, `ExerciseRecipeV1`, `MovementPredictionV1` and `WorkoutEventV1`. Specify units, timestamps, frame cadence, joint order, mask meaning, model/schema versions and reset behavior. Keep inferred tracking visibility separate from self-reported ability.
4. Assign a minimum Android test device, a recipe reviewer and a small consented pilot recruitment route. Start a conversion feasibility spike immediately; avoid discovering unsupported model operators in the final week.

**Exit:** clean-environment inference from a pinned artifact; no split/identity collisions; report states which heads are labeled and evaluated; one selected deployment path. A failed audit blocks claims based on that run, not unrelated catalog/UI work.

### P1 — Curated catalog, capabilities and equipment before ranking

**Owner: product/domain. Week 1–2; parallel with P0 after schema agreement.**

Create versioned recipes with ID/variant/side, family, posture, required capabilities, reviewed limitations, required versus optional equipment, substitution group, goal/muscle tags, difficulty, dose bounds, relevant joints, supported camera view and allowed feedback dimensions. Represent equipment alternatives explicitly (for example, one of several band types), rather than requiring every listed item.

Build a deterministic feasibility predicate that every recommendation and swap must pass. Cover unilateral variants, seated/wheelchair posture, assisted and limited states, unknown capability, unavailable equipment, changing equipment mid-session, explicit avoided movements and an empty candidate set. Unknown suitability should request clarification or offer a manual choice; ranking must never override a failed hard constraint. Maintain usable no-equipment choices where reviewed recipes permit them.

Build a simple baseline ranker and bounded workout assembler: order, duration, sets/reps/rest, side, diversity and reviewed dose rules. Add swap reasons and recheck the entire replacement against current profile/equipment. This creates a usable product and a reference against which the neural recommender must improve.

**Exit:** reviewed initial recipes; zero known constraint violations in exhaustive catalog/profile fixtures; clear explanations for exclusion and no-compatible-option states.

### P2 — Movement student: counting first, then richer outputs

**Owner: ML. Weeks 2–4. Depends on P0 and usable repetition labels.**

Retain the 283-input causal TCN as baseline and the GRU as control. The existing 128-frame window at 30 FPS is roughly 4.3 seconds; test slower repetitions that exceed this context. Separate a longer-lived count decoder from bounded model context, and evaluate longer receptive fields only if failure analysis warrants them. Richer optional motion features need a versioned ablation, not an untracked input-width change.

Build these additions to the shared student:

- **Exercise conditioning/recognition:** add a small learned embedding for the selected recipe/variant and a separately supervised exercise-ID head as labels permit. Keep broad-family and unknown/unsupported outputs. Supplying an exercise ID is conditioning, not evidence that the model recognized it; do not leak the target ID into recognition evaluation.
- **Repetition density:** add a nonnegative per-frame density head with explicit target units. Its integral/sum over a complete labeled interval must correspond to count. Train with masked density and count-consistency losses only where genuine repetition supervision exists; partial windows require correct fractional mass. Keep boundary and phase heads as complementary outputs.
- **Streaming decoder:** consume each timestamp once; calibrate boundary/density fusion, debounce, warm-up, pause/resume, exercise changes and tracking gaps. Never accumulate overlapping-window predictions twice. Emit a finalized event once, with abstention and count correction behavior specified.
- **Quality:** retain masked ROM, tempo, smoothness and trunk outputs. Train each only with its own valid target; a five-level UCO composite score remains a separate auxiliary task. Show reviewed observable feedback only for supported exercise/view/dimension combinations. Personal ROM calibration must not penalize a limited range against an assumed standard body.
- **Confidence:** expose separate `repConfidence` and `trackingConfidence` values in the prediction/event contract. Calibrate rep-event reliability against validation event matches, alongside observed-landmark checks; use a learned auxiliary confidence head only if training event-correctness targets can be generated without test leakage. The current observability target is not automatically a probability that a rep or quality judgment is correct. During poor tracking, pause automatic events and give useful framing guidance or manual input.

Use masked, provenance-aware losses: family/exercise/phase classification, boundary classification, density/count consistency, and individually labeled quality losses (masked binary classification with the current four heads; regression only after an explicit head/loss/schema change). Keep tracking supervision and the optional composite task separate; an ordinal composite loss is a proposed extension to the current class-based implementation. Tune weights on validation only. Run one change at a time against the unchanged TCN, with model size, latency and subgroup results reported together.

**Exit:** continuous held-out recordings show improved rep/boundary performance without hidden double counts; unsupported outputs are suppressed. Proposed pilot targets: count error no greater than one rep on at least 90% of labeled sets of roughly ten reps, and zero false rep events in the curated pause/occlusion regression set. These are release targets to validate, not achieved scores or population guarantees.

### P2b — Teacher distillation experiments, with corrected target semantics

**Owner: ML research. Start after the supervised baseline; bounded experiments in Weeks 3–5 if time permits, otherwise after submission.** All four requested teacher candidates remain on the roadmap, but each supplies a different kind of target:

| Teacher | Verified output and correct integration | Gate before training the student |
| --- | --- | --- |
| [SSTRAC](https://github.com/imjjun/SSTRAC_public) | Offline repetition density/count teacher; sum the density to obtain count. Distill aligned valid-frame density and count consistency | Check 33-to-17 pose mapping, coordinate conventions, full-window behavior and teacher count against real annotations |
| [PoseRAC](https://github.com/MiracleDance/PoseRAC) | Salient-pose class scores with threshold-triggered count events; optional exercise-specific salient-state auxiliary target | Obtain matching salient-state annotations and validate trigger semantics; do not relabel these as biomechanical phases |
| [RACNet](https://github.com/Luoadore/RACnet) | Full-resolution action-start probabilities; optional start-logit teacher through its RGB/video-feature path | Verify feature extractor/checkpoint availability, exact time alignment and cycle-start conventions; it does not supply end-boundary pairs |
| [MotionBERT](https://github.com/Walter0807/MotionBERT) | Frozen motion representations; use a projection from student hidden features to a compatible teacher representation | Validate H36M-17 input mapping, confidence channel, missing/absent limb handling and checkpoint sequence limits; reconstructed joints are not quality labels |

None supplies AdaptFit's rest/concentric/hold/eccentric labels directly. Obtain explicit phase annotation or retain honestly labeled exercise-specific weak supervision with unknown/transition masking. Never expand a count label into invented exact boundaries. MotionBERT input's three channels must follow its documented 2D-pose/confidence contract; do not assume they mean x/y/z simply from tensor shape.

Implementation sequence: shared pose-adapter smoke test using MotionBERT; SSTRAC density experiment; then separate RACNet start and PoseRAC salient-state experiments. Add teacher cache metadata containing checkpoint hash, preprocessing version, source/participant/split, frame timestamps, target meaning and confidence. Use training-split teacher outputs for optimization; preserve validation for selection and the locked test for final reporting. Future-aware offline teachers are permissible, but the student must remain causal and its online timing must be evaluated independently.

Compare supervised-only, each teacher separately, then only useful combinations with fixed splits/training budgets. Distillation loss is the supervised loss plus separately weighted, masked density, start-logit, salient-state or embedding losses. Select weights on validation. Drop teachers that do not improve the relevant task or that degrade subgroup performance/online timing. No teacher runs in the shipping app. The challenge release must not depend on all four experiments succeeding.

### Data acquisition and annotation plan

Only REHAB24-6, IntelliRehabDS, MM-Fit, UL-RED and UCOPhyRehab++ have explicit raw-source adapters in the audited repo. A configured source name or canonical-NPZ fallback is not a working adapter for a dataset's original release. Expand sources by missing supervision, not by the length of the source list.

| Priority | Sources from the Technical Impl | Intended contribution and next action |
| --- | --- | --- |
| First: audit existing inputs | [REHAB24-6](https://zenodo.org/records/13305826), [IntelliRehabDS](https://zenodo.org/records/4610859), [MM-Fit](https://mmfit.github.io/), [UL-RED / Liverpool](https://datacat.liverpool.ac.uk/2729/), [UCOPhyRehab++](https://zenodo.org/records/17935737) | Preserve acquisition/identity records; audit actual annotation meaning, usable exercises and test coverage. UCO exact spans and composite scores remain separate from dimension-specific quality |
| Next: genuine repetition supervision | [RepCount / TransRAC](https://github.com/SvipRepetitionCounting/TransRAC) and [KERAAL](https://keraal.enstb.org/KeraalDataset.html) | Audit access/terms and frame conventions; add raw adapters and alignment fixtures. Any RepCount-pose derivative must retain original video/subject identity and annotation provenance. KERAAL error spans are source-specific, not automatically full phase labels |
| Next: target-relevant robustness | [ROAG project](https://www.imperial.ac.uk/manipulation-touch/open-source/dataset/roag-dataset/) and [release](https://zenodo.org/records/13908725), [InclusiveVidPose](https://xiaobaishu0097.github.io/InclusiveVidPose/), [StrokeRehab](https://strokerehabdata.github.io/dataset.html), [SAFER-Activities](https://safer-activities.github.io/), [WheelPose](https://github.com/hilab-open-source/wheelpose) | Check acquisition/terms and task fit. Reaching, activity, pose and synthetic wheelchair data can help representation/front-end tests; they are not interchangeable with rep, quality or real target-user validation data |
| Audit before scheduling | [KIMORE](https://doi.org/10.1109/TNSRE.2019.2923060), [Toronto stroke pose code](https://github.com/zhiderek/TRSPD), [MM-Fi](https://github.com/ybhbingo/MMFi_dataset), [QEVD](https://www.qualcomm.com/developer/software/qevd-dataset), [Fitness-AQA](https://github.com/ParitoshParmar/Fitness-AQA), [WLU rehabilitation posture](https://www.kaggle.com/datasets/sulaimanmuhammed/wlu-rehabilitation-posture), [WheelPoser-IMU](https://github.com/axle-lab/WheelPoser) | Uncommitted acquisition backlog. Verify canonical release, license, modalities, subject IDs, useful targets and adapter effort. IMU modalities are not drop-in monocular pose inputs; generic fitness quality scores are not clinical labels |

For every new source, record terms/access, checksums, participants, modality/camera, skeleton mapping, coordinate units, FPS/timebase, left/right conventions, task labels and masks. Add a tiny audited adapter fixture, visualize aligned pose/annotations, then prepare isolated data. Route exact spans to boundaries/density, counts to count loss, composite scores to composite loss, and missing targets to masks. Preserve consistent source participant identity across tasks, crops, clips and augmentations; audit overlapping releases before merging datasets.

Add an annotation protocol for the five launch exercises: rep start/end definition, partial-rep handling, concentric/eccentric direction, rests/holds, permitted adaptations, uncertain labels and reviewer disagreement. For actual quality dimensions, first define whether the target is binary, ordinal or continuous; the current four outputs are pooled binary logits, so continuous ROM/tempo estimates require an explicit head/loss/schema change. Geometric measurements may be displayed only within their reviewed limits and must be distinguished from learned quality judgments.

Source checks for teacher semantics: [SSTRAC paper](https://doi.org/10.1109/access.2025.3624029), [PoseRAC paper](https://arxiv.org/abs/2303.08450), [RACNet paper](https://arxiv.org/abs/2407.09431), [MotionBERT paper](https://arxiv.org/abs/2210.06551). Dataset links in the acquisition backlog identify candidates; they do not assert that access or training rights have been cleared.

### P3 — Separate neural workout recommender and substitution ranking

**Owner: ML + domain. Weeks 3–5. Depends on P1 and an event contract.**

Build a small two-tower network: user/session encoder (capability states, posture, equipment, goals, preferences, recent tolerability and history) and exercise encoder (recipe metadata plus learned exercise ID). Start with a proposed 32-dimensional embedding and simple MLPs; select size from validation. Rank **only feasible** candidates. Keep routine assembly and hard dose/compatibility constraints outside the network.

Log local impressions/candidate sets, selection, completion fraction, skip/swap reason, repeats and explicit preference. Distinguish pain or equipment-related skips from dislike. Non-selection is not automatically a negative label; do not train on unavailable exercises as though they were rejected preferences. Begin with reviewed pairwise preferences if real history is scarce, label that supervision honestly, and retain the deterministic cold-start fallback.

Train with pairwise ranking or contrastive loss on valid candidate sets; split by user and time and compare NDCG@k/Recall@k, acceptance and constraint violations to the baseline. Hold out users and separately assess unseen recipe metadata. Learn personalization from voluntary history without weakening capability constraints.

For substitution, reuse embeddings plus original exercise, remaining routine and swap reason. Restrict to compatible substitution groups, then reapply feasibility. Add a small reranker only if baseline reuse fails on labeled pairs; a third independent large network is not a prerequisite.

**Exit:** the neural prototype trains reproducibly and ranks valid candidates; release activation requires improvement over the baseline on held-out evidence. If the data is insufficient, ship the clearly labeled rule-based fallback and keep the neural ranker experimental.

### P4 — Native mobile vertical slice and user experience

**Owner: mobile. Weeks 1–5, in parallel after contracts stabilize.**

Create the mobile application in the active AdaptFit project. Study the earlier app's native camera, permission and session boundaries; implement the new contract deliberately. Keep raw frames and full pose processing native/on-device. Feed the full landmark stream to native feature extraction and temporal inference, then send compact derived events to the UI.

Produce Python/native golden fixtures for bilateral, unilateral, absent, assisted, limited and occluded limbs, camera mirroring, coordinate scaling, irregular timestamps, dropped frames and session/exercise resets. Match the exact 2D feature normalization and ordering before adding depth or richer features. Reject incompatible schema/model bundles visibly.

Compare supported conversion/runtime options on the actual checkpoint; choose one for Android. Validate float32 export numerically, then calibrate quantization on representative training/validation samples, and evaluate the quantized model on the locked test set. Keep a validated float fallback if integer quantization regresses performance. Version weights, labels, feature schema, normalization and decoder thresholds as one artifact bundle.

Implement onboarding, equipment per session, workout selector/assembly, guided workout, camera permission/setup, pause/stop/manual count, reasoned swap, summary/history and honest progress graphs. Include screen-reader labels, scalable text, accessible controls, non-color-only cues, audio/haptic alternatives and a fully usable camera-denied path. Settings cover capabilities, equipment, consent and local data deletion. No fabricated progress or form scores.

**Exit:** complete offline workout and swap flow on the minimum device; network inspection confirms no raw camera/pose uploads. Measure p50/p95 latency, memory, dropped frames, thermal behavior and battery over a full session. Proposed responsiveness budget: p95 camera-to-feedback below 150 ms at the selected supported cadence; declare hardware and test conditions.

### P5 — Target-user validation and challenge delivery

**Owner: product/reviewer + mobile + ML. Recruitment starts Week 1; validation Weeks 4–6.**

Prepare consent, retention/deletion choices, annotation instructions and a participant/session split before collecting recordings. Seek seated/wheelchair and unilateral/limb-difference participants relevant to the intended claims. Do not equate synthetic limb masking or a wheelchair-position field with recordings of those users. Pilot size depends on recruitment and cannot justify broad population claims; record counts and limitations explicitly.

Annotate rep events, pauses, partial reps, camera failures and valid feedback dimensions; double-review a subset and resolve disagreement. Test reduced ROM, slow tempo, assistance, prostheses when relevant and voluntarily represented, clutter, camera angles and lighting. Report per-source, exercise, posture, capability and device results with participant-level uncertainty and abstention coverage. Retest only changes that can affect these outcomes.

Freeze features after Week 5. Use Week 6 for pilot corrections and accessibility; Week 7 for regression checks, reproducible demo data, installation instructions, source/AI attribution and submission material. Proposed internal submission target: October 23, leaving a buffer before the official **October 26, 2026, 12:00 pm EDT** deadline. Confirm participating district and eligibility early. The official rulebook requires a 1–3 minute public YouTube/Vimeo demonstration and disclosure of AI/tool use with meaningful student contribution. Preserve a record of student implementation and understanding. Source: [2026 CAC rulebook](https://www.congressionalappchallenge.us/wp-content/uploads/2026/05/2026-CAC-Rules.pdf).

**Demo:** capabilities → available equipment → compatible routine → supported rep tracking → deliberate equipment swap → saved progress. Show limitations honestly. watchOS, hundreds of camera-recognized exercises and broad clinical claims are outside the critical first-release path; retain them as later milestones.

## Decisions that must be resolved, without blocking independent work

- Minimum Android device and whether iOS is a first-release requirement: decide during P0; existing repo direction favors Android first.
- Recipe/label reviewer and participant access: start now; absence blocks validated target-user claims and unsupervised quality feedback, not engineering fixtures.
- Data/weight licenses and reuse restrictions: verify before each acquisition, training use or redistribution; code and dataset licenses are separate checks.
- Neural ranking data: retain a deterministic baseline until enough reviewed preference/history evidence exists.
- Scope expansion: distinguish catalog-compatible exercises from camera-validated exercises. Grow each only after its own coverage checks.

## Immediate next implementation batch

1. Reconcile completed versus partial training artifacts and publish a current status report.
2. Freeze golden feature/prediction fixtures and run an early mobile conversion spike.
3. Implement the five-recipe capability/equipment catalog and deterministic feasibility tests.
4. Audit exact repetition labels and evaluate continuous-stream counting before expanding model heads.
5. Build the mobile onboarding → routine → manual session → summary flow while neural work proceeds.

These tasks can move the project forward immediately while preserving the full neural roadmap.

## Work-package execution contract

The package sections above explain the work. This table makes the handoff
machine-readable for Luna: do not mark a package complete without its artifact
and gate, and stop only the dependent path when its failure is isolated.

| Package | Prerequisites | Responsible role | Files/interfaces affected | Expected artifact | Acceptance gate | Failure/stop condition | Downstream unlocked |
|---|---|---|---|---|---|---|---|
| P0 baseline/contracts | Current-state inventory, code/config access | ML + product | `contracts-and-schemas.md`, configs, manifests, runtime | Reproducible inventory, pinned baseline, schema fixtures | No identity collisions; finite inference; head/label coverage explicit | Stop benchmark claims on leakage/schema failure; catalog work may continue | P1 recipes, P2 experiments, P4 parity |
| P1 catalog/compatibility | Contract versions and reviewer route | Product/domain + safety | Recipe catalog, capability profile, compatibility rules | Versioned recipes, exhaustive profile/equipment fixtures, empty-candidate behavior | Zero known hard-constraint violations; review states recorded | Stop ranking or camera claims if no reviewed recipe; keep manual fallback | P3 ranker, P4 UI, P5 target pilot |
| P2 movement student | P0, valid repetition labels, locked comparison cohort | ML | Model heads, losses, decoder, evaluation | Isolated pilot report and `WorkoutEventV1` replay traces | Sequence/event gains without duplicate counts or unsupported feedback | Stop on leakage, missing masks, non-finite loss, or no meaningful improvement | P2b teacher choice, P4 integration |
| P2b teachers | P2 supervised baseline and source/terms review | ML research | Teacher adapters/cache, loss config | One teacher comparison with cost and provenance | Matched-budget held-out benefit without subgroup/latency regression | Defer teacher; never block deterministic baseline | Optional P2 extension |
| Data acquisition | Candidate source, license/access gate | Data/research | Dataset adapter, manifest, split/label map | Checksummed prepared source plus adapter fixture | Participant IDs, masks, labels, license and normalization reproduce | Do not stage/train pending or prohibited source | Valid P2/P5 supervision |
| P3 recommender | P1 feasibility and event/history contract | ML + domain | Catalog embeddings, ranker, logging | Baseline-vs-neural ranking report | Only feasible candidates; held-out improvement or honest fallback | Keep deterministic ranker; do not train on non-selection as dislike | Product personalization |
| P4 mobile vertical slice | Contract fixtures, exporter feasibility, reviewed recipes | Mobile/runtime | Native feature/model bridge, bundle, UI | Offline demo bundle, parity report, device measurements | Golden fixtures, privacy, latency/memory/thermal gates pass | Use manual/float fallback; no native support claim | P5 validation/demo |
| P5 validation/challenge | Consent, reviewer, locked target split, stable build | Product/reviewer + ML/mobile | Consent/retention, annotation, release package | Participant report, limitations, demo and attribution package | Claims match participants/labels; safety/privacy/accessibility review passes | Stop broad claims or submission language; preserve engineering evidence | Public release decision |

Each package owner records the commit, config/schema versions, changed files,
artifact paths, gate result, and next eligible package in the execution log.
