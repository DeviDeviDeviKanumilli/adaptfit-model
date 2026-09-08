# AdaptFit research-backed methodology reuse

> **Documentation metadata**
> - **Status:** supporting
> - **Authority:** cited research synthesis and proposed reuse boundaries; source code, configuration, artifacts, and canonical contracts remain authoritative
> - **Last verified:** 2026-09-08
> - **Source commit:** 0ef219d
> - **Owner:** AdaptFit ML research and engineering
> - **Supersedes or supports:** supports model-registry.md, efficient-training-strategy.md, recommendation-model-plan.md, motion-jepa-world-model-plan.md, project-forward-plan.md, and artifact-registry.md
> - **Review trigger:** new baseline result, external source/license change, teacher experiment, recommender data collection, or deployment decision

This report translates external research into bounded AdaptFit experiments. It separates
published or repository facts from AdaptFit inferences and implementation decisions.
External benchmark numbers do not transfer to AdaptFit and are not product evidence.

## Decision

Keep the current 283-feature, 128-frame causal TCN as the product student. Improve it
through the cheapest measured intervention first:

1. verify the current sequence-level baseline and decoder behavior;
2. test one label-efficient movement method or offline counting teacher;
3. test structured AF-MJEPA only if the remaining error is representation transfer;
4. build recommendation rules and a small ranker only after the catalog and feedback
   contracts exist;
5. export and validate only the selected student.

External transformers, video models, JEPA teachers, and recommendation frameworks remain
offline research tools. The phone model must continue to use the FeatureSchemaV1 contract,
capability-aware masks, causal streaming state, and deterministic safety/decoder rules.

## Current AdaptFit boundary

The current repository implements:

- a causal TCN with 283 inputs, 96 channels, five dilated residual blocks, and a
  125-frame receptive field;
- 128-frame training windows with the versioned feature and normalization contracts;
- a smaller causal GRU comparison model;
- family, phase, repetition-boundary, pooled quality, and tracking-confidence heads;
- participant/source-aware preparation, masked losses, sequence aggregation, and Python
  streaming tests.

The repository does not currently contain:

- an implemented WorkoutEvent decoder or production event state machine;
- a learned recommendation model or behavioral-history store;
- a native Android/iOS model bridge, exporter, quantized bundle, or parity fixture;
- an AF-MJEPA model, teacher cache, or distillation path;
- reviewed labels for the four dimension-specific quality heads;
- real amputee, limb-difference, wheelchair-user, or clinical target-population validation.

These facts are verified in [current-state](current-state.md), [model registry](model-registry.md),
and the [repository map](repository-and-implementation-map.md).

## How to interpret the evidence

### Sourced facts

A sourced fact describes what a paper, official repository, license file, or official
documentation says. Reported metrics remain measurements on the source method and source
dataset.

### AdaptFit inference

An inference maps a sourced method to the current AdaptFit contracts. It is a hypothesis
until an isolated experiment produces an artifact with a participant/source split, config,
checkpoint, and evaluation report.

### Recommended action

A recommendation is an implementation choice for AdaptFit. It is not evidence that the
method will improve the current benchmark.

### Rights rule

Code, weights, annotations, and datasets have separate rights. A permissive code license
does not grant permission to redistribute a dataset or checkpoint. Before reuse, record
the exact repository commit, license text, dependency versions, weight terms, dataset
terms, checksum, and intended use in the experiment manifest. If any of these are
unavailable, use the method as a paper-only reference.

## Movement and repetition methods

### TransRAC: cycle supervision, density, and multi-scale temporal correlation

TransRAC introduced the RepCount dataset with long and short videos, interruptions,
inconsistent cycles, and fine-grained start/end annotations. Its model encodes temporal
correlations at multiple scales and predicts a density map whose sum represents the
repetition count.[^1]

**AdaptFit reuse**

- Convert reviewed start/end spans into a nonnegative density target whose total mass
  equals the labeled count.
- Train an offline, noncausal teacher or an isolated auxiliary head to expose cadence,
  interruptions, and long/short cycle structure.
- Distill only causal per-frame targets or aligned soft boundary targets into the
  existing TCN.
- Keep event emission, refractory timing, reset behavior, and abstention in the
  deterministic decoder.
- Do not claim that density prediction alone solves online latency or duplicate-window
  accumulation.

The official TransRAC repository is Apache-2.0 licensed, but its Video Swin input path is
not a drop-in replacement for canonical pose features. The first AdaptFit experiment
should reuse its target construction and temporal-correlation idea, not import the full
RGB backbone.[^2]

### SSTRAC: skeleton reconstruction and density prediction

SSTRAC uses skeleton sequences, reconstructs defective skeletons, applies a dual-stream
spatio-temporal transformer, builds a multi-scale self-attention representation, and
predicts a repetition density map. The authors explicitly target body-size and
occlusion robustness.[^3]

**AdaptFit reuse**

- Use canonical 33-joint sequences plus observed and capability masks as the teacher
  input.
- Add a corruption/reconstruction pretext task that removes or degrades observable
  joints while preserving declared capability state.
- Compare a compact skeleton teacher against the current TCN on sequence count MAE,
  start/end F1, false events during rest, and event latency.
- Cache teacher density or boundary targets with timestamps, masks, split, preprocessing
  hash, teacher hash, and generation commit.
- Treat the MIT repository as a code candidate, not an automatic dependency; isolate
  its preprocessing and verify every transitive dependency before copying code.[^4]

SSTRAC is the preferred first external implementation candidate because its input modality
is closer to AdaptFit than an RGB repetition model. It remains an offline teacher and
does not become a mobile runtime model.

### RepNet: self-similarity and synthetic repetition

RepNet uses a temporal self-similarity matrix and synthetic repeated clips made by
sampling unlabeled videos and repeating them with varied periods and counts. The paper
reports that synthetic training can generalize to unseen repetition classes and that the
self-similarity representation is useful for cadence analysis.[^5]

**AdaptFit reuse**

- Generate synthetic repetitions from eligible pose sequences rather than raw RGB.
- Transform timestamps, masks, phase labels, and density mass consistently when clips are
  repeated, cropped, or time-scaled.
- Use temporal self-similarity plots as a diagnostic for cadence changes, pauses, and
  false periodicity.
- Use the synthetic data only for representation or boundary/count experiments; it does
  not create clinical, quality, or target-population labels.
- Preserve participant lineage so repeated copies cannot inflate participant counts.

Google Research source code is Apache-2.0, while data and pretrained weights require
their own terms review. The method is suitable for augmentation and diagnostics, not a
reason to add a raw-video model to the product.[^6]

### PoseRAC: salient-pose annotation

PoseRAC represents each repetition with two exercise-specific salient poses instead of
annotating every redundant frame. Its paper and repository describe a lightweight
pose-level model and a RepCount-pose annotation set.[^7]

**AdaptFit reuse**

- For each launch exercise, define reviewed anchor states such as the contracted and
  returned positions.
- Use anchor labels to reduce annotation cost and to create auxiliary phase/apex
  supervision.
- Retain AdaptFit's explicit start/end boundary labels; two salient poses are not a
  substitute for pauses, partial reps, aborted reps, or event timing.
- Use the MIT repository only as optional annotation/model tooling after checking its
  data terms and input mapping.

This is the cheapest candidate when the main bottleneck is annotation time rather than
representation capacity.

### RACnet: TSM and start-probability learning

RACnet learns frame embeddings, predicts action-start probabilities at full temporal
resolution, and imposes consistency with a generated temporal self-similarity matrix.[^8]

The concept is useful for a boundary teacher or diagnostic, but the official repository
is CC-BY-NC and relies on RGB-derived Video Swin features. It is therefore paper-only for
a commercial or distributable product unless legal review authorizes a specific use. Do
not vendor its code, weights, or dataset into AdaptFit.

### MotionBERT and Skeleton2vec: representation pretraining

MotionBERT pretrains a motion encoder to recover underlying 3D motion from noisy or
partial 2D observations, using a dual-stream spatio-temporal transformer.[^9]
Skeleton2vec predicts contextualized teacher representations at masked skeleton positions
and uses motion-aware temporal tube masking to force longer-range reasoning.[^10]

**AdaptFit reuse**

- Treat incomplete pose and confidence corruption as explicit training conditions.
- Preserve the distinction between camera occlusion and declared absent anatomy.
- Prefer contextual latent targets over raw-coordinate reconstruction for AF-MJEPA.
- Use MotionBERT only as a later transfer experiment because its 3D skeleton and model
  assumptions require a deliberate adapter.
- Do not copy Skeleton2vec code: the public repository says the implementation is not
  generally available under a clearly stated reuse license.

## Recommendation methodology

The recommender is a separate system. It selects reviewed recipes; it does not count
repetitions, infer phase, inspect camera frames, or override safety rules.

### Stage R4: deterministic feasibility and content ranking

The first implementation should:

1. validate the profile, equipment, posture, recipe approval state, and avoid list;
2. compute an eligible candidate set with reason codes;
3. return a safe empty-candidate response when the set is empty;
4. rank eligible recipes using deterministic goals, variety, dose, history, and user edits;
5. log catalog hash, profile version, model/rules version, shown candidates, chosen item,
   and fallback reason without raw frames or pose.

A ranker must never receive an ineligible candidate. Feasibility remains outside the
learned score.

### Stage R5: small learned ranker

When consented exposure and feedback data exist, begin with a compact pointwise or BPR
pairwise model over eligible candidates. Use user-level splits, future-period holdouts,
cold-start evaluation, exposure-aware negatives, and separate metrics for completion,
skip, swap, rejection, and manual selection.

The two-stage retrieval/ranking pattern from YouTube and TensorFlow Recommenders is a
future scaling pattern. The initial catalog is small enough that deterministic filtering
and a local content ranker are preferable to ANN retrieval.[^11] RecBole may be used as a
research benchmark harness, but its datasets and production suitability must be assessed
separately.[^12]

SASRec and BERT4Rec remain method references for sequential history modeling. Their
official repositories are Apache-2.0 but depend on obsolete TensorFlow/Python versions;
reimplement the idea in the AdaptFit stack only after history volume justifies it.

### Deferred contextual bandit and slate optimization

Do not implement contextual bandits or slate RL in the first recommender. First establish
a safe no-op/manual baseline, reliable exposure logging, consent, and offline policy
evaluation. Action-centered bandits are relevant because mHealth datasets can contain
only a few hundred noisy decisions and benefit from modeling incremental treatment
effects rather than a fully nonlinear changing baseline.[^13] Safety-aware bandit work
provides a useful risk-constraint framing, but it does not establish exercise safety for
AdaptFit.[^14]

## AF-MJEPA teacher

AF-MJEPA remains a training-only research model.

### Input representation

Use structured tokens from:

- canonical 33-joint positions or normalized coordinates;
- velocities and selected angles;
- pose confidence and observed masks;
- capability state and capability masks;
- posture, exercise/variant, equipment, and camera-view context.

A declared absent limb must stay absent in capability metadata. Synthetic masking may
simulate camera occlusion, but must not convert absence into an observation failure.

### Objective

Use future latent prediction as the primary loss. Evaluate temporal interval masking,
motion-aware joint/tube masking, and paired-view consistency as separate ablations. Use
an EMA or stop-gradient target encoder and record collapse diagnostics. Supervised phase,
boundary, family, or quality heads are optional auxiliary losses and remain masked when
labels are unavailable.

I-JEPA and V-JEPA support the general predictor/target principle: predict latent
representations instead of pixels. Their older repositories are research-only
CC-BY-NC; use their published method, not their code, in a distributable product.[^15]
V-JEPA2 is MIT licensed but is a large video-domain reference whose scale is not
appropriate for the first pose-only pilot.[^16]

### Distillation

Teacher outputs must be cached with:

- sequence and participant IDs;
- source and split;
- timestamps and frame rate;
- feature, normalization, mask, and schema hashes;
- teacher model/config/checkpoint hashes;
- generation commit and command;
- target type and horizon;
- artifact limitations.

Distillation must add a teacher loss to valid supervised losses. The student remains the
283-feature causal TCN and must pass the existing sequence-level, subgroup, confidence,
abstention, and latency gates.

## Reuse matrix

| Candidate | Exact reuse | AdaptFit compatibility | Rights/limits | First gate |
|---|---|---|---|---|
| TransRAC | density targets, cycle spans, multi-scale temporal relation | offline teacher; RGB backbone is not direct-fit | Apache code; dataset/weights separate | density mass and count/boundary improvement |
| SSTRAC | skeleton corruption, dual-stream temporal teacher, density map | closest external teacher input | MIT code; verify dependencies/data | occlusion robustness and causal distillation |
| RepNet | synthetic repeats and TSM diagnostic | pose adaptation required | Apache code; source data/weights separate | lineage-safe augmentation benefit |
| PoseRAC | salient phase/apex annotations | direct annotation concept; preserve full boundaries | MIT code; annotation terms separate | label-cost reduction without timing loss |
| RACnet | TSM/start probability loss | RGB and CC-BY-NC boundary | research-only unless legal approval | paper-only comparison |
| MotionBERT | incomplete-pose pretraining | 3D/17-joint adapter required | Apache code | transfer gain justifies adapter |
| Skeleton2vec | contextual targets and tube masks | strong AF-MJEPA design fit | no clear reusable code release | collapse/probe checks |
| RecBole | offline recommender comparisons | separate research harness | MIT code; data terms separate | behavioral dataset sufficiency |
| I-JEPA/V-JEPA | target/predictor and EMA principles | conceptual pose adaptation | CC-BY-NC code | method reference only |
| V-JEPA2 | latent world-model design patterns | domain/scale mismatch | MIT code; do not vendor first | pose-only pilot first |

## Gates, artifacts, and stop conditions

- **R0 baseline freeze:** reproduce corrected-v1 and lock split/config/schema evidence.
  Stop if any metric or artifact cannot be traced.
- **R1 cheap movement experiment:** choose exactly one of synthetic repetition
  augmentation, salient-pose labels, or density/TSM teacher. Stop if sequence count,
  boundary, or rest-false-event metrics do not improve over the matched baseline.
- **R2 structured teacher:** choose one SSTRAC/TransRAC-style teacher or AF-MJEPA. Require
  split isolation, finite losses, target-encoder/collapse checks, and total compute/time.
  Stop on leakage, collapse, or no useful frozen-probe result.
- **R3 distillation:** compare supervised-only and teacher-assisted students with matched
  budgets. Stop on primary-metric regression, subgroup regression, worse abstention, or
  causal latency regression.
- **R4 recommendation baseline:** require exhaustive hard-rule fixtures, empty-candidate
  fallback, catalog/version compatibility, and no unsafe candidate.
- **R5 learned recommendation:** require consented exposure data, cold-start and
  future-time evaluation, calibration, subgroup reporting, and zero feasibility
  violations. Stop if learned ranking worsens fallback or hard constraints.
- **R6 deployment:** export only the selected student. Require Python/native golden
  fixtures, float/quantized parity, latency/memory/thermal measurements, privacy checks,
  and rollback artifacts.

## Validation and claim boundaries

After this documentation pass:

~~~text
python3 scripts/validate_docs.py
python3 -m pytest -q
~~~

The documentation validator must continue to reject broken relative links, missing
metadata, stale current facts, invalid fixtures, and unsupported claims. Research links
and license records should be reviewed manually because availability and terms can change.

No external benchmark result may be described as an AdaptFit result. No method described
here establishes clinical safety, medical benefit, target-population performance, quality
supervision, or mobile parity. Missing labels remain unavailable, and synthetic or
public able-bodied data cannot substitute for consented target-population validation.

## Sources

[^1]: Huazhang Hu et al., “TransRAC: Encoding Multi-Scale Temporal Correlation With Transformers for Repetitive Action Counting,” CVPR 2022. [Paper](https://arxiv.org/abs/2204.01018).
[^2]: SvipRepetitionCounting, [official TransRAC repository](https://github.com/SvipRepetitionCounting/TransRAC), Apache-2.0 repository metadata and implementation.
[^3]: Jungjun Lim et al., “SSTRAC: Skeleton-Based Dual-Stream Spatio-Temporal Transformer for Repetitive Action Counting in Videos,” IEEE Access 2025. [Paper](https://doi.org/10.1109/access.2025.3624029).
[^4]: imjjun, [official SSTRAC repository](https://github.com/imjjun/SSTRAC_public), MIT license and implementation.
[^5]: Daniel Dwibedi et al., “Counting Out Time: Class Agnostic Video Repetition Counting in the Wild,” 2020. [Paper](https://arxiv.org/abs/2006.15418).
[^6]: Google Research, [RepNet implementation](https://github.com/google-research/google-research/tree/master/repnet), Apache-2.0 code repository; dataset and checkpoint terms require separate review.
[^7]: Ziyu Yao et al., “PoseRAC: Pose Saliency Transformer for Repetitive Action Counting,” 2023. [Paper](https://arxiv.org/abs/2303.08450); [repository](https://github.com/MiracleDance/PoseRAC).
[^8]: Yanan Luo et al., “Rethinking temporal self-similarity for repetitive action counting,” 2024. [Repository](https://github.com/Luoadore/RACnet), CC-BY-NC repository license.
[^9]: Wentao Zhu et al., “MotionBERT: A Unified Perspective on Learning Human Motion Representations,” ICCV 2023. [Project](https://motionbert.github.io/); [license](https://github.com/Walter0807/MotionBERT/blob/main/LICENSE).
[^10]: Ruizhuo Xu et al., “Skeleton2vec: A Self-supervised Learning Framework with Contextualized Target Representations for Skeleton Sequence,” 2024. [Paper](https://arxiv.org/abs/2401.00921); [repository](https://github.com/Ruizhuo-Xu/Skeleton2vec).
[^11]: Paul Covington et al., “Deep Neural Networks for YouTube Recommendations,” 2016. [Google Research](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/); [TensorFlow Recommenders retrieval guide](https://www.tensorflow.org/recommenders/examples/basic_retrieval).
[^12]: RUCAIBox, [RecBole repository](https://github.com/RUCAIBox/RecBole), MIT research library with separate dataset terms.
[^13]: Susan A. Murphy et al., “Action-Centered Contextual Bandits,” NeurIPS 2017. [Paper](https://papers.neurips.cc/paper/7179-action-centered-contextual-bandits.pdf).
[^14]: Wen Sun, Debadeepta Dey, and Ashish Kapoor, “Safety-Aware Algorithms for Adversarial Contextual Bandit,” ICML 2017. [PMLR](https://proceedings.mlr.press/v70/sun17a.html).
[^15]: Mahmoud Assran et al., “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture,” CVPR 2023. [I-JEPA repository](https://github.com/facebookresearch/ijepa); [V-JEPA repository](https://github.com/facebookresearch/jepa).
[^16]: Meta FAIR, [V-JEPA2 repository](https://github.com/facebookresearch/vjepa2), MIT code license and video-domain implementation.
