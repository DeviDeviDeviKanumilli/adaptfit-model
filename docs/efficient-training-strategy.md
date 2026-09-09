# AdaptFit: accuracy-first training with minimal wasted compute

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** training runbook; commands must still be verified against the runner and config
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `613ff12` (documentation revision base)
> - **Owner:** AdaptFit training engineering
> - **Supersedes or supports:** canonical staged-training workflow; supports current-state, data-and-training-plan, and evaluation-protocol
> - **Review trigger:** runner/config/schema change, new checkpoint, or changed compute/evaluation budget

## Recommendation

Reuse the existing TCN, repair the supervision around its weakest tasks, and fine-tune only as much of the network as the evidence requires. Begin with the five launch exercises. Expand coverage after those work reliably on real, unseen participants.

The objective is **reliable movement feedback per hour of total experimentation**, including preparation, teacher inference, training and evaluation. Minimum training time and maximum possible accuracy are competing objectives; there is no guaranteed optimal recipe before measuring the failures. The strategy below spends compute in stages and stops when further work does not produce a meaningful improvement.

This is a proposed training strategy, not a report of new experiments. It complements [the project plan](project-forward-plan.md).

For execution, use the **Luna execution TODO list** below. It breaks this strategy into bounded tasks with dependencies, outputs and completion checks.

## 1. Establish what actually needs to improve

The audited corrected-v1 TCN already has useful movement representations: approximately 307k parameters and 87.57% family macro-F1. However, repetition-end F1 is 18.47%, and its four quality dimensions have no labeled evaluation coverage. More training on the same data cannot supply missing supervision.

First evaluate the completed v2-quality-fixed TCN with merged sequence and continuous-stream metrics. Its training finished, but the earlier audit found no completed sequence evaluation; the GRU run was interrupted. Do not spend another full run finishing the GRU unless a smaller deployment alternative is needed.

Compare candidate checkpoints using their own correct preprocessing on a common **validation** cohort that neither saw during training. Existing v1/v2 splits may differ: exclude the union of their historical training participants. Keep the already examined benchmarks as historical diagnostics; reserve newly collected, untouched participants for the final test. If no clean comparison cohort exists, collect one before calling a checkpoint superior.

Inspect a small, diverse sample of failures and classify them:

| Failure | Cheapest useful response |
| --- | --- |
| Pose missing, mirrored or normalized differently from mobile | Fix preprocessing and masks; preserve Python/native parity |
| Peaks detected but counted twice or during pauses | Fix/calibrate the streaming decoder on validation data |
| Labels disagree on where a repetition ends | Correct annotation conventions before retraining |
| Cleanly labeled motions still confuse the model | Targeted fine-tuning |
| A desired quality label does not exist | Obtain reviewed labels or keep that output unavailable |

Choose by online count error, start/end event precision and recall, timing delay, and performance across intended profiles. Track phase/family scores as secondary measures. Report abstention coverage alongside accuracy so a model cannot appear better merely by refusing difficult cases.

## 2. Spend annotation effort where it replaces training effort

Prioritize existing sources with exact repetition spans, especially the integrated UCO data, and a small consented set of actual launch-exercise recordings. Include pauses, partial reps, slow repetitions, reduced ROM, unilateral movement and camera failures. Public seated or wheelchair-position metadata is not equivalent to validation with wheelchair users.

Define one rep-start/end convention per exercise and review ambiguous examples. Keep composite quality scores separate from ROM, tempo, smoothness and trunk-control targets. Missing labels remain masked; teacher guesses and procedural labels must retain their provenance.

The current configuration uses 128-frame windows with stride 8: adjacent windows overlap by about 94%. Start experiments with fewer redundant windows, while retaining sufficient causal context and coverage near repetition events. For example, test stride 32 or randomized training offsets, with extra sampling around underrepresented events. Keep validation and test reconstruction fixed. Fewer windows mean fewer updates, so compare both wall time and the number of valid supervised frames processed—not epoch counts alone.

Sample across participants, exercises and sources so long recordings do not dominate. Keep a representative mix of easy examples and genuine failures. Use light, label-consistent noise, occlusion and tempo augmentation; time warping must transform boundaries/density consistently. Avoid generating large numbers of synthetic variations that repeat the same information.

After each useful experiment, choose the next annotation batch from diverse training-pool failures or disagreements, plus a random sample. Never mine the locked test set for training examples. This focuses new information on the remaining error rather than repeatedly increasing dataset size.

## 3. Fine-tune in progressively more expensive stages

Keep the 283-feature contract and existing backbone initially. Preserve the checkpoint's normalization and label definitions. Adding a density head does not require changing the input representation or retraining pose estimation.

| Stage | What changes | When to advance |
| --- | --- | --- |
| A: no network training | Validation-only decoder thresholds, debouncing, pause/reset behavior and confidence calibration | Residual errors are genuinely model errors |
| B: heads first | Freeze the TCN backbone; update existing boundary heads and, in a separate experiment, a small nonnegative density head | New heads plateau but consistent labeled errors remain |
| C: partial fine-tuning | Unfreeze the last one or two temporal blocks with a lower learning rate; retain representative old-data examples | Features appear unable to represent the new motions |
| D: full fine-tuning | Unfreeze the backbone briefly at a conservative rate | Partial tuning cannot resolve the demonstrated domain shift |

Frozen-feature training and full-network fine-tuning are established transfer-learning options; their relative benefit here must be measured. [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

Proposed starting settings—not established optima—are AdamW with approximately `3e-4` for new heads and `3e-5` for unfrozen backbone layers, retaining batch size 64 if practical and gradient clipping at 1.0. Run a short 2–3-pass pilot over a fixed representative training subset. Continue only promising candidates, checking validation every 1–2 passes and stopping after roughly three checks without meaningful improvement. Increase patience if evaluation noise makes that rule unstable. These budgets screen experiments; they do not guarantee convergence.

Freezing parameters does not remove forward-pass cost. Cache backbone activations only if repeated head experiments justify the storage and extraction time; keep the frozen backbone in evaluation mode. Such a cache becomes invalid when weights, preprocessing or input augmentations change. Frozen blocks must stay in evaluation mode even while the trainable head is training.

Avoid full retraining from random initialization unless the checkpoint is incompatible, contaminated for the intended evaluation, or demonstrably transfers poorly. Preserve an unchanged copy of every starting checkpoint.

## 4. Train the outputs that have trustworthy targets

For the first improvement cycle, prioritize repetition starts/ends and reliable streaming counting. Add density only as a controlled second experiment. Define density in reps per frame so its sum matches the labeled count; cropped windows must retain the appropriate fractional mass. Density targets are offline supervision, but the student's inputs must remain causal. Check online event delay separately from final set-count accuracy, and never sum predictions twice from overlapping windows.

Use a weighted combination of losses, with each averaged over its valid labeled examples:

`loss = repetition supervision + supported auxiliary tasks + optional teacher loss`

Keep exact boundary/count labels stronger than weak labels. Monitor the contribution of each loss so abundant family labels do not overwhelm sparse repetition supervision. Retain some original-task training examples to detect and reduce forgetting. Do not launch a broad loss-weight search; adjust only when the per-task results identify interference.

The current four quality heads are pooled binary logits, not continuous ROM estimators. Train them only when their particular binary labels exist. Continuous or ordinal targets require an explicit head/loss change. Keep UCO's composite expert score as its own optional auxiliary task, and retain it only if it helps a relevant held-out outcome.

Initially use the selected exercise as session context instead of requiring fine-grained exercise recognition to work across a large catalog. A learned exercise embedding is a later targeted extension. Its supplied identity cannot also serve as evidence that the model independently recognized the exercise.

## 5. Use distillation selectively

Do not begin by training or running multiple teachers. Teacher inference, pose-format conversion and tuning are part of the compute budget. First establish whether supervised fine-tuning already solves the problem. The proposed [AF-MJEPA plan](motion-jepa-world-model-plan.md) is a separate offline representation/pretraining experiment, not an automatic replacement for a task-specific teacher.

If counting remains weak, **SSTRAC density/count supervision is the first task-specific teacher experiment I would try**, after verifying that its predictions improve on existing labels for the relevant recordings. Use released compatible weights where available, freeze the teacher, and cache its aligned outputs once. Reliable human/source labels remain the anchor. [SSTRAC implementation](https://github.com/imjjun/SSTRAC_public).

If the measured failure is broader representation transfer and there is enough
eligible unlabeled pose data, evaluate AF-MJEPA instead of adding another task
teacher. Start with one bounded 10M/20M/40M size comparison, future-latent
prediction as the primary objective, and the existing TCN as the student. Do
not run SSTRAC and AF-MJEPA together in the first experiment; the extra cost and
different targets would make the result hard to attribute. The full gate and
masking rules are in [the Motion-JEPA plan](motion-jepa-world-model-plan.md).

MotionBERT is a later representation experiment if the main failure is transfer to unfamiliar motion or incomplete pose. Its 17-joint format needs an explicit adapter; it does not supply repetition or quality labels. RACNet's RGB-derived action-start signals and PoseRAC's salient-pose events introduce additional adaptation work, so defer them unless they address a measured gap. None directly supplies AdaptFit's biomechanical phase labels. [Teacher details and sources](project-forward-plan.md).

Compare one teacher-assisted candidate against supervised-only training with matched data and update budgets. Retain it only if its benefit justifies the **total** extra time and preserves subgroup and streaming performance. Distillation can improve a small student; it does not inherently reduce total training cost.

## 6. Make a small number of decisive experiments

My first experiment sequence would be:

1. **Evaluate and calibrate existing TCNs.** No gradient updates; select a defensible starting point.
2. **Fine-tune boundary heads on corrected, balanced supervision.** This isolates the value of better data and inexpensive adaptation.
3. **Choose one next change from the errors:** unfreeze late blocks if representations are inadequate, or add density if boundary-only counting remains brittle. Do not change both at once.
4. **Optional teacher experiment.** Attempt only if the simpler route leaves an important measurable gap.

Use one seed for initial screening. Confirm the selected improvement against its baseline with a second seed if the difference could plausibly be training noise. Compare errors on the same participants and report participant-level uncertainty. The final test is for reporting the selected configuration, not repeatedly choosing the next experiment.

Record data/schema/checkpoint versions, trainable layers, supervised frames, updates, wall time, peak memory and validation results. Continue a run while it makes worthwhile progress; stop experiments whose additional compute produces no material improvement. Do not promise a fixed hour count before profiling this machine.

## 7. Preserve accuracy through deployment

Reuse the existing memmaps, batch conversion and length bucketing. Profile preparation, loading, forward/backward passes and validation before changing infrastructure. The current config uses float16 storage with float32 model computation on MPS; storage precision is not mixed-precision training. Any lower-precision compute path requires separate support and numerical checks. Avoid adding compilation complexity unless its warm-up cost is recovered over the planned runs. [PyTorch performance guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

Export and check feature/output parity early. Try post-training quantization after choosing the float model; assess it on validation and use quantization-aware fine-tuning only if needed. Choose the deployment variant before final test reporting. Calibration samples must not come from the test set. Verify accuracy and timing on complete device sessions, including slow reps beyond the model's approximately four-second receptive field.

Personalize initial range/tempo and decoding through a short user calibration, without per-user gradient updates. Keep capability/equipment feasibility deterministic. Train the separate small workout recommender only once useful reviewed preferences or consented history exist; pose training data does not teach exercise preference.

## Changes needed before executing this strategy

The staged fine-tuning workflow is now implemented for the next isolated run:
checkpoint initialization, schema/normalization checks, trainable-layer
selection, separate learning rates, strict source paths, test locking, and
resumable optimizer/RNG/sampler state. The R1 config has not been trained yet;
use [pretraining-readiness.md](pretraining-readiness.md) for the bounded smoke
command and its stop conditions.

Add density supervision only after the simpler experiment warrants it. Keep artifact directories isolated and verify weight loading, frozen-layer behavior and causal streaming parity. No training or source-code changes were performed to create this document.

The practical priority is: **clean supervision → reuse learned features → targeted fine-tuning → selective distillation**. For AdaptFit's current state, this is my strongest hypothesis for improving useful accuracy with the least wasted training; the staged comparisons will establish whether it holds.

## Operational runbook

## Reproducibility and command/output contract

Record the following before each run in `ExperimentRecordV1`: repository
checkout and dirty paths, Python/PyTorch/NumPy versions, operating system and
device (`cpu`, CUDA, or MPS), available disk, config hash, dataset-manifest
IDs/checksums, feature/normalization versions, seed, parent checkpoint, and
declared wall-time/update budget. The current baseline was verified on Python
3.13 with PyTorch MPS support; this is an environment observation, not a
portable hardware guarantee.

| Command | Prerequisites | Expected output | Stop condition |
|---|---|---|---|
| `python3 -m training.preflight --config <cfg>` | raw paths, manifest, license/access state | preflight report and resolved config | any missing path, checksum, schema, or split failure |
| `python3 -m training.prepare_data --config <cfg>` | passing preflight and isolated output root | prepared windows, manifest, normalization, feature schema, audit | identity collision, non-finite features, or unmasked label error |
| `python3 -m training.train ...` | prepared data, new artifact root, selected seed/device | checkpoints, histories, model card, experiment record | non-finite loss, leakage, incompatible keys, budget/patience stop |
| `python3 -m training.evaluate ...` | immutable checkpoint and locked split | sequence/window metrics, predictions, provenance report | missing offsets, split mismatch, or untraceable metric |
| `python3 scripts/validate_docs.py` | repository paths and JSON fixtures | read-only documentation/fixture report | any link, metadata, schema, status, or claim-trace failure |

Warm-start and exact resume remain different contracts. A warm-start restores
weights only and records a parent checkpoint; an exact resume restores the
model, optimizer, RNG streams, sampler epoch, update history, and compatible
data/code environment (scheduler/scaler state are recorded when such state is
introduced). The current runner exposes `--init-checkpoint`,
`--resume-checkpoint`, `--freeze-backbone`, `--unfreeze-last-blocks`,
`--skip-test`, and isolated experiment configs. Every run writes to a new
artifact root and updates the [artifact registry](artifact-registry.md) after
evaluation. The exact next command is in
[pretraining-readiness.md](pretraining-readiness.md).

This section turns the strategy into a repeatable gate sequence. A Luna agent
must stop at the first failed gate, record the evidence, and avoid spending
training compute to work around a provenance or schema failure.

### Preflight and preparation

1. Confirm repository, Git commit, dirty paths, Python environment, device,
   available disk, and expected compute/time budget.
2. Read `docs/current-state.md`, `docs/contracts-and-schemas.md`, the selected
   dataset manifest, and the experiment configuration. Resolve every path.
3. Verify feature width 283, frame rate, 128-frame window, stride, masks,
   normalization-statistics hash, label dimensions, and model head shapes.
4. Verify that participant/source IDs and split manifests are present and that
   no validation/test participant occurs in the training set.
5. Run a no-gradient batch through the selected checkpoint. Stop on non-finite
   features, logits, losses, missing labels that are not masked, or a schema
   mismatch.
6. Create a new artifact directory containing a copied config, source commit,
   input manifest IDs, environment summary, and parent-checkpoint hash. Never
   write into `artifacts/corrected-v1/`, `artifacts/v2-quality/`, or
   `artifacts/v2-quality-fixed/`.

### Checkpoint and resume semantics

- **Warm-start:** load model weights only; use when changing optimizer, sampler,
  trainable layers, or task heads. Record missing/unexpected keys and require
  an explicit compatibility check.
- **Exact resume:** restore model, optimizer, scheduler, AMP/scaler state when
  used, RNG states, sampler position, update count, and config hash. It is valid
  only when the data order, code, preprocessing, and hardware assumptions are
  compatible.
- **Best checkpoint:** selected by the primary validation sequence metric and
  saved separately from the latest resumable state. Do not select by training
  loss alone.
- Preserve an immutable copy of every parent checkpoint. A filename such as
  `best.pt` does not prove what run or schema produced it.

### Training stages and activation

1. **Decoder-only:** calibrate thresholds, debouncing, pause/reset, and
   abstention on validation. No gradients.
2. **Heads-only:** freeze the TCN; update only selected heads. Keep frozen
   modules in evaluation mode and optimizer parameter groups explicit.
3. **Partial backbone:** unfreeze the final one or two temporal blocks with a
   lower learning rate while retaining representative original data.
4. **Full fine-tuning:** use only when the measured domain shift remains after
   the cheaper stages. Freeze architecture and schema before starting.

Losses are averaged over valid examples for each task. A missing or
   unsupported label contributes zero weight, not a fabricated negative. Log
   per-task valid counts and loss contributions so abundant family labels cannot
   drown sparse boundary labels. Teacher losses are optional and must remain
   separate from reliable supervised targets.

### Sampler, validation, and stopping

- The current 128-frame/stride-8 data has highly correlated windows. A training
  sampler may reduce redundant overlaps or balance participants/sources, but
  validation/test reconstruction and sequence offsets remain unchanged.
- Keep event neighborhoods, pauses, slow/partial repetitions, and representative
  easy examples. Do not sample only errors or boundaries.
- Run validation at a fixed update/pass cadence recorded in the experiment
  record. Select on sequence-level metrics; retain window metrics as diagnostics.
- Stop on non-finite values, schema drift, data leakage, repeated validation
  degradation, exhausted wall-time/update budget, or no meaningful improvement
  for the declared patience. A stop is a reportable outcome, not a reason to
  silently extend the budget.

### Compute and failure triage

Record preparation time, data-loader time, forward/backward time, validation
time, total wall time, peak memory, device, effective labeled frames/windows,
updates, and teacher inference time. When a run fails, classify it before
retrying: environment/path, data/provenance, numerical, checkpoint/schema,
decoder/evaluation, or model capacity. Fix the cheapest upstream cause first;
do not mask an evaluation failure by launching a longer run.

### Teacher and deployment gates

Run a teacher only after supervised fine-tuning and decoder work leave a measured
gap. Verify access terms, pose mapping, time alignment, compatible weights, and
cached-output hashes. Compare teacher-assisted and supervised-only candidates
with matched student updates and include teacher preparation/inference in the
compute report. Retain it only when the held-out outcome improves without
subgroup or causal-streaming regressions.

Before final test reporting, freeze preprocessing, decoder, calibration,
architecture, checkpoint, and deployment variant. Run the golden Python/native
fixtures and float/quantized parity checks in [evaluation-protocol.md](evaluation-protocol.md).

### Motion-JEPA gate (planned)

Treat AF-MJEPA as a research backlog item until C1 and the supervised baseline
are reproducible. Before any run, create a versioned manifest and config that
fixes the participant/source split, target-encoder update rule, prediction
horizons, masking policy, loss weights, model size, and wall-time budget. Add a
finite-value/collapse check and a frozen-probe evaluation before scaling the
run.

Use only canonical pose sequences and explicit capability/observed masks. Keep
the 283-feature student and all reliable supervised losses unchanged. Cache
teacher latents or soft targets with model, preprocessing, split, timestamp,
and generation-commit hashes. Accept the teacher only when a matched held-out
student improves a predeclared sequence or robustness metric without regressions
in count, phase, abstention, subgroup, or causal streaming behavior. If the
representation collapses, split provenance is incomplete, or the student does
not improve at the declared budget, stop and retain the supervised baseline.

## Luna execution TODO list

### How to use this list

Work in `/Users/devk/AdaptFit`. Paths below are relative to that repository. Read applicable `AGENTS.md` instructions and inspect the current checkout before editing. The artifact status above is an audit snapshot; verify it again rather than assuming it is still current.

Execute one numbered task at a time. Do not implement the optional experiments automatically. After each task, record changed files, verification results, unresolved issues and the next eligible task in `docs/training-execution-log.md` (new file). Check a box only when its completion check passes. Record blocked tasks with the missing input; continue independent eligible work.

This TODO list is a handoff specification, not an instruction to start training during this document-editing request. When implementation is requested, follow that request's scope and compute budget. Do not interpret a prepared configuration as a reason to launch an overnight run. Preserve existing datasets/checkpoints and write each experiment to a distinct artifact directory.

**Terminology:** the backbone is the shared TCN feature extractor; a head maps its features to a task output; warm-start means loading existing weights; exact resume also restores optimizer and random-number state; a gate is a completion condition, not a request for additional permission.

### Phase A — Establish trustworthy inputs and evaluation

#### A1. Inventory the existing work

- [ ] Inspect Git status, configs, processed-data indexes and artifacts for corrected-v1, v2-quality and v2-quality-fixed.
- [ ] Record each checkpoint's schema, normalization, source split, architecture, completed training status and available evaluation files.
- [ ] Mark runs as prepared, partially trained, training-complete or evaluated. Do not infer completion from a checkpoint filename.

**Read:** `training/configs/`, `artifacts/`, `data/processed*/`, `docs/audit-fixes-and-overnight.md`.

**Output:** the inventory section in `docs/training-execution-log.md`.

**Done when:** every candidate checkpoint has an explicit provenance/status entry, including unknown fields. No training has been launched to discover its status.

#### A2. Define valid comparison data

- [ ] Recover training participant identities for each candidate checkpoint, including original participants behind clips and synthetic variants.
- [ ] Identify a common validation cohort outside the union of those training participants. Audit duplicate recordings across sources.
- [ ] Keep normalization fitted only on training data. Preserve each checkpoint's original preprocessing during comparison.
- [ ] Define a separate final test cohort that will not be inspected during tuning. If new participant data is required, record that dependency.

**Depends on:** A1. **Read:** `training/src/data/{identity,prepare,dataset}.py`, prepared metadata.

**Output:** versioned participant/split manifests and an overlap report in a new experiment directory.

**Done when:** comparison data has zero known training overlap, or the report explicitly says a fair comparison is blocked. Old public test results remain historical diagnostics.

#### A3. Establish the baseline and inspect failures

- [ ] Evaluate usable checkpoints with matching configs and preprocessing; complete missing merged-sequence reporting for the finished v2 TCN.
- [ ] Replay continuous sequences through the causal runtime. Measure count error, separately matched start/end events, false counts during rest, event delay and abstention coverage.
- [ ] Report results by exercise/source/profile where labels and sample counts permit. Label synthetic and public-position proxy results separately.
- [ ] Review diverse failures and label them as preprocessing, annotation, decoder or representation failures.

**Depends on:** A1; A2 for comparative claims. **Read:** `training/evaluate.py`, `training/src/{evaluation,metrics}.py`, `training/src/models/streaming.py`.

**Output:** baseline report and short failure inventory. New report/runtime evaluation code may be needed; do not invent existing CLI flags.

**Done when:** the next experiment addresses an observed failure, and missing labels are reported as unavailable rather than zero error.

#### A4. Prepare focused supervision

- [ ] Write rep start/end, partial-rep and pause conventions for each launch exercise.
- [ ] Review source labels against those conventions. Retain exact labels, weak labels and missing labels distinctly.
- [ ] Create a small representative training subset with multiple participants, clean cycles, pauses, slow reps and relevant adaptations.
- [ ] Define any additional consented recording/annotation needs. Keep unavailable quality dimensions masked; retain UCO composite scores separately.
- [ ] Add a bounded adapter/label fixture for each changed source and verify time alignment after resampling.

**Depends on:** A3. **Read:** `training/src/data/{adapters,schema,temporal,provenance}.py`, `training/src/losses.py`.

**Output:** annotation notes, subset manifest and verified label changes.

**Done when:** every active loss has a meaningful target, validity mask and provenance. Do not rewrite the complete prepared-data tree unless a verified change requires it.

### Phase B — Build the smallest useful fine-tuning workflow

#### B1. Add safe weight initialization

The implementation is present in the working tree and covered by staged
training tests. The checklist below now describes the verification expected
for each future run; it is not a request to retrofit a legacy checkpoint.

- [ ] Add an explicit warm-start option to the training configuration/CLI and load the selected checkpoint before optimization.
- [ ] Check input schema, normalization, architecture and label meanings. Reject incompatible shared layers.
- [ ] Allow intentionally new heads only through an explicit allowlist of missing/new parameters; report all loaded and initialized parameters.
- [ ] Record the starting checkpoint identity in the new run's output.

**Depends on:** A1. **Edit:** `training/train.py`, `training/src/{config,runner}.py`, model construction as needed.

**Done when:** a test shows loaded shared weights match the source, an incompatible checkpoint fails clearly, and unexpected missing keys cannot silently pass.

#### B2. Add trainable-layer stages and recoverable checkpoints

The runner now supports the requested modes and stores the resumable state in
`adaptfit.checkpoint.v2`. No full R1 training run has been launched yet.

- [ ] Support heads-only, heads-plus-last-blocks and full fine-tuning modes.
- [ ] Keep frozen blocks in evaluation mode; configure the optimizer with only intended trainable parameters and separate head/backbone learning rates.
- [ ] Save model, optimizer, scheduler when used, random-number state, sampler progress and completed update count for future resumption.
- [ ] Distinguish best-validation weights from the latest resumable training state. Document the exact-resume boundary supported by the loader.

**Depends on:** B1. **Edit:** `training/src/runner.py`, `training/src/models/tcn.py` only if necessary.

**Done when:** a one-step test proves frozen weights do not change and intended weights do; a small save/resume test verifies the supported continuation behavior. Old weight-only checkpoints remain warm starts.

#### B3. Reduce redundant work without changing evaluation

- [ ] Add a reproducible training-only sampler that reduces repeated overlapping windows and balances participants/sources/exercises.
- [ ] Retain event neighborhoods, pauses and representative easy examples; do not train exclusively on boundaries or failures.
- [ ] Keep sufficient causal context for sampled windows. Leave validation/test manifests and reconstruction unchanged.
- [ ] Log sampled windows, labeled frames, updates and elapsed time. Reuse existing memmaps and length bucketing.

**Depends on:** A4 and B1. **Read/edit:** `training/src/data/dataset.py`, loader construction in `training/src/runner.py`.

**Done when:** sampling is reproducible, affects training only and preserves useful label coverage. Larger stride is an experiment, not an automatic replacement of all prepared data.

#### B4. Prepare a bounded experiment configuration

- [ ] Create a new fine-tuning config derived from the compatible existing config; use a new artifact destination.
- [ ] Start with the learning rates and short pilot suggested above, retaining float32 model computation and gradient clipping.
- [ ] Set validation cadence, early stopping, maximum updates/passes and a wall-time budget appropriate to the execution request.
- [ ] Run a tiny forward/backward check and relevant existing tests before the actual pilot.

**Depends on:** B1–B3. **Output:** new config under `training/configs/` and recorded preflight results.

**Done when:** the config resolves correctly, the checkpoint loads, outputs/losses are finite and no existing run will be overwritten.

### Phase C — Run only experiments justified by results

#### C1. Improve decoding without network updates

- [ ] Fix duplicate events, reset behavior or timestamp issues found in A3.
- [ ] Calibrate thresholds/debouncing on validation data and compare with the unchanged baseline.
- [ ] Verify pauses, occlusion, dropped frames, exercise changes and manual corrections with replay fixtures.

**Depends on:** A3. **Output:** versioned decoder settings and comparison report.

**Done when:** accepted changes improve the intended behavior without hiding errors through excessive abstention. This task can precede the fine-tuning implementation.

#### C2. Run the heads-only pilot

- [ ] Warm-start the selected TCN; train boundary heads on the reviewed subset with the backbone frozen.
- [ ] Keep comparison conditions fixed and record total preparation/training/evaluation time.
- [ ] Compare against the calibrated baseline on the valid comparison cohort. Check original-task performance for regressions.
- [ ] Continue a promising run only within its declared budget; otherwise record why it stopped.

**Depends on:** A2–A4, B4 and C1. **Output:** isolated pilot artifacts and an accept/reject/insufficient-evidence decision.

**Done when:** the report identifies whether the cheap adaptation helped and which failure remains. An improved training loss alone is insufficient.

#### C3. Select one next experiment

- [ ] If clean motion representations remain inadequate, fine-tune the last one or two blocks at the lower rate while retaining representative original data.
- [ ] If boundary-only counting remains brittle, take the optional density task below instead.
- [ ] If preprocessing, annotations or decoder behavior still explain failures, return to those tasks.
- [ ] If performance is adequate for the declared scope, proceed to confirmation; do not expand the model automatically.

**Depends on:** C2. **Output:** one justified experiment choice in the execution log.

**Done when:** only one new factor changes and the reason is tied to observed errors. Full-network tuning is a later escalation, not a mandatory stage.

### Phase D — Optional model extensions

#### D1. Add density only if C3 selects it

- [ ] Add a nonnegative per-frame density output alongside existing boundary outputs.
- [ ] Define targets whose sum equals count over fully labeled intervals; preserve fractional mass in crops and mask unknown regions.
- [ ] Add masked density/count losses and initialize only the new head before considering backbone changes.
- [ ] Verify target mass, resampling/cropping, causal inputs and single-count accumulation in streaming replay.
- [ ] Compare with the matched boundary-only model; reject the extension if it fails to improve useful counting behavior.

**Edit:** `training/src/models/heads.py`, data target generation/schema, `training/src/losses.py`, evaluation and runtime code.

**Done when:** tests establish target/count consistency and validation demonstrates a worthwhile improvement at acceptable latency.

#### D2. Try one teacher only if supervised improvement is insufficient

- [ ] Pick a teacher for the remaining task; SSTRAC is the first density/count candidate, not a source of phase labels.
- [ ] If the remaining gap is representation transfer rather than density, evaluate one bounded AF-MJEPA pilot instead of adding another task teacher. Read `docs/motion-jepa-world-model-plan.md` first.
- [ ] Verify access/terms, checkpoint, pose mapping, time alignment and teacher behavior on relevant examples.
- [ ] Cache frozen teacher targets with model/preprocessing hashes, timestamps, masks and split provenance. Do not optimize on test data.
- [ ] Add one separately weighted distillation loss while retaining reliable supervised labels.
- [ ] Compare supervised-only and teacher-assisted students with matched update budgets, including teacher inference in the cost report.

**Depends on:** C3 and the appropriate target contract; D1 for density distillation.

**Done when:** the teacher's total cost buys a measurable benefit without subgroup or causal-streaming regressions. Otherwise defer it. MotionBERT, RACNet, PoseRAC, SSTRAC, and AF-MJEPA remain separate optional experiments; do not implement multiple teachers together in the first comparison.

#### D3. Defer unsupported outputs and unrelated learning tasks

- [ ] Keep each quality output disabled until its reviewed target exists; match binary/ordinal/continuous losses to the actual target semantics.
- [ ] Add exercise embeddings only for a measured coverage gap with stable exercise IDs and no recognition-label leakage.
- [ ] Use non-gradient user calibration initially. Keep the separate recommender on its own preference/history data, after deterministic compatibility filtering exists.

**Done when:** every deferred feature has its missing prerequisite recorded. “Deferred” does not mean implemented or validated.

### Phase E — Confirm, export and hand off

#### E1. Confirm the chosen improvement

- [ ] Compare the selected candidate with its baseline on the same validation participants and report subgroup sample sizes and uncertainty.
- [ ] Use a second seed for both when training variability could explain the claimed gain.
- [ ] Freeze architecture, preprocessing, decoder and calibration choices before final test reporting.

**Depends on:** the selected Phase C/D path. **Done when:** improvement is reproducible enough for the claim being made, or uncertainty is explicitly disclosed.

#### E2. Check deployment fidelity

- [ ] Perform a float export feasibility check early using an existing checkpoint; repeat parity checks on the selected final model.
- [ ] Compare Python/native feature values and outputs on golden unilateral, missing-limb, occlusion and timestamp fixtures.
- [ ] Try post-training quantization using non-test calibration data. Use validation to select float or quantized deployment.
- [ ] Measure complete-session latency, memory and counting behavior on the intended device. Use quantization-aware tuning only if needed.

**Depends on:** A1 for the early spike; E1 for final candidate verification.

**Done when:** the versioned deployable bundle preserves acceptable accuracy and streaming behavior, or the exact conversion/device blocker is documented.

#### E3. Produce the final evidence package

- [ ] Evaluate the frozen deployment candidate on the untouched test cohort once for final reporting; do not tune against its errors.
- [ ] Save model/schema/normalization/decoder versions, split provenance, metrics, abstention coverage, total experiment time and known limitations.
- [ ] Update the execution log with completed, rejected, deferred and blocked tasks, plus exact reproduction commands that were actually verified.

**Depends on:** E1–E2 and available final-test data.

**Done when:** another developer can reproduce the result and distinguish tested support from proposed support. No fresh target-user test means no fresh target-user accuracy claim.

### Suggested first assignment to Luna

“Read this strategy and applicable repo instructions. Complete A1 and identify the evidence needed for A2. Inspect existing files without launching training. Write the inventory, unresolved provenance questions and next eligible tasks to `docs/training-execution-log.md`. Report the exact checkpoint and data paths you verified. Do not implement optional model extensions.”

After that bounded assignment, use the dependencies above to choose the next task. A1–A4 and C1 establish whether more training is necessary; B1–B4 make the first fine-tuning run controlled and reviewable.

## Research-backed reuse order

Use the [research-backed methodology reuse report](research-method-reuse-report.md)
to choose one bounded external method at a time. The current student contract and
training runbook remain unchanged.

1. Freeze corrected-v1 and audit decoder/counting errors.
2. Choose one cheap movement intervention: pose-preserving synthetic repeats,
   PoseRAC-style salient anchors, or a density/temporal-correlation teacher.
3. Compare that intervention against the matched supervised baseline on locked
   sequence-level metrics.
4. Only if the remaining error is representation transfer, run one structured
   SSTRAC/TransRAC-style teacher or one AF-MJEPA pilot.
5. Distill cached targets into the existing TCN with the same 283-feature input,
   split, and causal runtime contract.
6. Keep all other teachers deferred until the selected intervention either passes
   or is rejected.

Teacher work must include teacher inference time, cache size, preprocessing and
model hashes, participant/source split, and a stop reason. Do not run multiple
teachers together in the first comparison, and do not use test data to choose a
teacher, mask, decoder, or loss weight.
