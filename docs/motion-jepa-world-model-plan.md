# AdaptFit Capability-Conditioned Motion-JEPA

> **Documentation metadata**
> - **Status:** research-backlog
> - **Authority:** proposed self-supervised world-model and teacher architecture; it does not override implemented model contracts
> - **Last verified:** 2026-09-08
> - **Source commit:** `613ff12` (repository base; proposal is not implemented)
> - **Owner:** AdaptFit ML research
> - **Supersedes or supports:** supports `scalable-ml-architecture.md`, `efficient-training-strategy.md`, `data-and-training-plan.md`, and Phase 4 of `project-forward-plan.md`
> - **Review trigger:** baseline result, data-volume change, compute-budget change, or self-supervised experiment

AF-MJEPA is listed in the [model registry](model-registry.md) as a research-only
teacher. Follow the [repository and implementation map](repository-and-implementation-map.md)
for the current training entrypoints and the [artifact registry](artifact-registry.md)
for the provenance required before any teacher result can be compared with the
movement baseline.

> **Proposal boundary:** No Motion-JEPA model, checkpoint, training entrypoint,
> teacher cache, export, or mobile runtime exists in the current repository.
> The current model remains the 283-feature causal TCN with the GRU baseline.
> This document defines a bounded research path that may improve the existing
> student after the decoder and supervised baseline gates pass.

## Verdict and role

An AdaptFit-specific JEPA is a strong fit for the offline representation and
teacher layer. It should learn predictable movement state in latent space rather
than reconstructing every future landmark coordinate or generating video. The
useful prediction target is a representation of exercise dynamics: phase,
repetition progression, posture, velocity, capability constraints, equipment
effects, and movement quality where reviewed targets exist.

JEPA is therefore an optional pretraining and distillation strategy, not a
replacement for supervised heads, deterministic safety rules, the current
causal TCN, or the `WorkoutEventV1` decoder. It must not delay Phase 1 decoder
calibration or the controlled supervised baseline comparison.

The working name is **AdaptFit Capability-Conditioned Motion-JEPA (AF-MJEPA)**.
It is a training-only teacher/world model. Its representations or soft targets
may be distilled into the existing approximately 308K-parameter TCN, which
remains the deployment candidate.

## Why this scope is preferred

The current supervised TCN remains the first-line baseline because it is small,
causal, testable, and already aligned with the product contract. A JEPA pilot
is justified only when unlabeled temporal data or representation-transfer
failures create a gap that supervised fine-tuning cannot close cheaply.

The proposed pose-level approach avoids the main costs of the alternatives:

- exact-coordinate future regression would spend capacity on viewpoint,
  proportions, estimator jitter, occlusion, and style that are not the product
  target;
- pixel reconstruction or a generative video world model would add large data,
  compute, and privacy requirements without being needed for the causal student;
- contrastive or generic masked reconstruction objectives remain possible
  baselines, but require their own augmentation and shortcut audits;
- raw RGB V-JEPA-scale training is deferred until pose-only experiments show a
  clear, measurable limitation.

The desired scale is a shared latent movement model across exercise families,
equipment variants, postures, and capability profiles. This is a hypothesis to
test, not evidence that one model already supports hundreds of exercises.

## Proposed data flow

```text
TRAINING ONLY

33-joint pose sequence
+ pose confidence and observed masks
+ capability profile and capability masks
+ posture, exercise, and equipment context
        |
        v
  context encoder  ------------------------------+
        |                                         |
   latent context                          target encoder
        |                                  (EMA/stop-gradient)
        v                                         ^
     predictor ---- predicts future or hidden ---+
        |
        +-- latent JEPA loss
        +-- optional exercise/phase/rep/quality/confidence heads
        |
        v
  distillation targets for the 283-feature causal TCN student
```

The target encoder must not receive gradient updates from the predictor loss.
An EMA target or another explicitly tested anti-collapse mechanism is required;
the choice is an experiment, not a current implementation fact.

## Representation contract

The `FeatureSchemaV1` 283-feature vector remains the stable student and mobile
contract. The larger teacher should preserve more structure instead of treating
one frame as an undifferentiated vector:

- joint/time tokens contain canonical landmark position, velocity, confidence,
  observed state, and capability state;
- global tokens contain posture, exercise/variant, equipment, and profile
  context;
- engineered angles, angular velocities, and diagnostic features may be
  attached to relevant joint or global tokens;
- 2D projection, 3D mocap, and source-specific skeletons are mapped into the
  canonical 33-joint layout before the JEPA objective is applied.

The model must preserve the distinction between an available limb that is
occluded and a limb declared absent or assisted. JEPA masking may simulate
occlusion or sensor loss, but it must not rewrite capability metadata or turn a
declared absence into a failed camera observation.

## Conditioning variables

The predictor may condition on:

- stable exercise and variant IDs;
- movement family and expected posture;
- equipment and, only when reliably recorded, resistance/load;
- capability profile, limb state, and assistance mode;
- camera/view metadata when it affects observability.

Equipment is a physical context variable, not only recommender metadata. A curl
with a band, dumbbell, cable, or machine can share an exercise family while
having different dynamics. The conditioning representation must not be used to
claim that the model understands force, load, muscle activation, or clinical
appropriateness; those remain outside the pose model and inside reviewed rules.

## Self-supervised objectives

Future latent prediction is the primary objective. A context window should
predict one or more later latent intervals, with horizons selected from the
available sequence length and recorded in the experiment manifest. Candidate
horizons are approximately 0.25, 0.5, 1, and 2 seconds at the configured frame
rate; they are starting points, not fixed requirements.

Use complementary tasks only when each task has a clear purpose:

1. **Past-to-future prediction:** predict future latent movement states from
   observed history and conditioning variables. This is the closest analogue
   to the proposed movement world model.
2. **Temporal interval masking:** hide a contiguous interval and predict its
   latent state from the surrounding context. Use this to test phase and cycle
   structure without requiring labels at every frame.
3. **Joint/anatomy masking:** hide selected wrists, elbows, legs, or confidence
   channels to improve robustness to occlusion and missing landmarks. Keep
   capability masks explicit.
4. **Cross-view consistency:** when paired 3D and projected 2D views or
   multi-view recordings are available, require compatible latent movement
   states. Do not manufacture paired views from unrelated sequences.
5. **Confidence degradation:** use dropped-frame and confidence corruption
   augmentations only as robustness stressors. They do not create quality or
   target-population labels.

Random patch masking copied from an RGB video model is not the default. Masks
must test motion continuation, body-configuration robustness, or view
invariance. Poorly chosen masks can teach the wrong invariances or let the
model solve the task from trivial local shortcuts.

The full multitask objective is represented as:

```text
L = λ_jepa L_latent
  + λ_future L_future
  + λ_exercise L_exercise
  + λ_phase L_phase
  + λ_rep L_rep
  + λ_quality L_quality
  + λ_confidence L_confidence
```

The weights, latent distance, horizons, and supervised heads must be declared
per experiment. Missing labels contribute zero through the existing task masks;
they are never treated as negative labels. Quality terms remain disabled when
reviewed quality coverage is zero.

## Dataset use and provenance

Every clean temporal sequence can contribute to the JEPA objective even when it
does not have repetition or quality labels. A dataset without a valid label for
a downstream head may still be useful for representation pretraining, subject
to the same license, privacy, identity, and split rules as supervised data.

The experiment manifest must record, for every sequence:

- source/version/checksum, license and permitted use;
- participant/session identity and split;
- canonical joint mapping, units, frame rate, and view metadata;
- observed and capability masks;
- augmentation lineage and mask type;
- available downstream label roles and valid masks;
- teacher/model/config hashes and target-encoder version.

For a held-out evaluation claim, self-supervised pretraining must obey the
declared participant/source split. Do not pretrain on held-out test participants
and then report the resulting downstream score as if the representation were
unexposed. Cross-view pretraining may use 3D motion data as a biomechanical or
view-invariance prior, but it does not create amputee, limb-difference,
wheelchair-user, or clinical validation evidence.

## Capacity and compute plan

Start with a bounded size comparison rather than assuming that a large model is
better:

- encoder: approximately 8–20M parameters;
- predictor: approximately 2–8M parameters;
- total pilot candidates: about 10M, 20M, and 40M;
- 30–80M is a later option only if data volume and held-out gains justify it.

These are engineering starting points inferred from the lower dimensionality of
canonical pose relative to RGB video, not a published optimum for exercise
motion. The teacher may be much larger than the student, but its offline
training and inference cost must be included in the experiment record.

Use existing prepared pose sequences first. Do not begin RGB V-JEPA training or
pixel/video generation; that would add compute and privacy scope without being
necessary for the current movement contract.

## Distillation into the phone model

The student continues to consume the exact 283-feature stream and 128-frame
causal windows. Candidate distillation targets are:

- pooled or per-frame latent features aligned to the student window;
- future-state or phase representations;
- teacher soft logits for exercise, phase, repetition boundaries, or confidence;
- repetition density only when a target contract and evaluation gate exist.

The student must retain reliable supervised losses. A teacher loss is an
additional signal, not a substitute for valid labels. Cache teacher outputs
offline with sequence IDs, timestamps, masks, split provenance, preprocessing
hashes, model hash, and generation commit. The teacher never ships in the
mobile bundle.

## Gates and stop conditions

Do not implement or launch this path until:

1. C1 decoder calibration and the current supervised baseline are reproducible.
2. The pretraining manifest has valid participant/source identity and license
   state.
3. The target encoder, masking policy, loss weights, and compute budget are
   written into a versioned experiment config.
4. A small collapse/finite-value test and a frozen-probe evaluation exist.

Accept a JEPA experiment only if, at a matched student budget, it improves a
predeclared held-out sequence metric or robustness metric without regressing
movement family, phase, count, abstention, subgroup, or causal-streaming
behavior. Report the full teacher cost and compare against supervised-only and
the current TCN baseline.

Stop or defer when the representation collapses, masks are solved by leakage,
teacher cost exceeds the declared budget, labels/splits are not traceable, or
the student does not improve a product-relevant metric. No JEPA result permits
medical, quality, or target-population claims without the corresponding labels
and participants.

## Research references and interpretation limits

The proposal is informed by the latent-prediction framing in [I-JEPA](https://arxiv.org/abs/2301.08243),
future-state and action-conditioned work in [V-JEPA 2](https://arxiv.org/abs/2506.09985),
and the recent past-to-future human-video direction in
[Human-JEPA](https://arxiv.org/abs/2608.21160). Those papers show that JEPA-style
objectives can support useful representations or anticipation in their own
settings; they do not establish AdaptFit accuracy, safety, or deployment
readiness. Any AdaptFit claim must come from its own versioned manifest,
checkpoint, split, metrics, and artifact.

## Research-backed objective and reuse boundary

The [research-backed methodology reuse report](research-method-reuse-report.md)
grounds this proposal in skeleton representation research:

- Skeleton2vec supports contextualized teacher targets and motion-aware tube masking;
- MotionBERT supports learning from incomplete pose observations but requires a deliberate
  3D/skeleton adapter;
- I-JEPA and V-JEPA support predictor/target latent prediction and anti-collapse patterns;
- V-JEPA2 is a permissively licensed conceptual reference whose video scale is outside
  the first AdaptFit pilot.

These sources inform the experiment design; they do not add code, checkpoints, or
deployment support to this repository. The older I-JEPA/V-JEPA repositories have
non-commercial research licensing, and Skeleton2vec does not provide a clearly reusable
implementation license in its public README. Do not vendor their code or weights.

The first AF-MJEPA pilot must choose one target-encoder/masking recipe, record the
license and commit for every imported component, and compare against a supervised-only
TCN with matched split and compute budget. A falling latent loss alone is not a pass.
