# AdaptFit annotation handbook

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** label definitions, temporal boundaries, masking, adjudication, and annotation provenance
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `9fe47fb` (documentation revision base)
> - **Owner:** AdaptFit data and evaluation engineering
> - **Supersedes or supports:** supports `data-and-training-plan.md`, `dataset-catalog.md`, and `evaluation-protocol.md`
> - **Review trigger:** label policy, source adapter, quality target, split policy, or temporal aggregation change

This handbook prevents an implementation agent from treating every visible
motion or source-provided score as a valid AdaptFit target. Each label has a
role, provenance, mask, unit, and temporal resolution.

## Label roles

| Role | Meaning | Valid target | Missing behavior |
|---|---|---|---|
| `movement_family` | coarse movement class | categorical per frame/sequence | mask, never infer from filename alone |
| `phase` | ordered observable movement state | rest/concentric/hold/eccentric or source mapping | mask unsupported states |
| `rep_start` / `rep_end` | temporal event boundary | event timestamp/frame with source tolerance | mask absent boundaries |
| `repetition_count` | count for a complete logical sequence | integer with sequence identity | exclude incomplete or unknown sequence |
| `quality_*` | named reviewed quality dimension | only when source label semantics match | zero task weight and report coverage |
| `expert_quality` | source-provided composite score | documented source scale and mapping | do not equate to four quality heads |
| `tracking_confidence` | observability/input quality | pose/visibility-derived signal | never use as movement quality |

## Temporal conventions

- `rest` is a valid phase only when the source defines a stable pre/post-motion
  interval; arbitrary gaps are not automatically rest.
- `concentric`, `hold`, and `eccentric` boundaries follow the source's stated
  movement direction and are mapped to the canonical phase vocabulary in the
  adapter. A source without a defensible mapping is masked.
- Start and end events are stored in sequence time, not independently copied
  into every overlapping window. Window labels carry offsets back to the logical
  sequence.
- A partial repetition, interrupted set, or unknown boundary is marked partial
  and excluded from count-loss supervision unless the experiment explicitly
  declares a partial-label policy.
- Pauses and dropped frames remain explicit events/observation gaps. They must
  not be compressed into a faster repetition.

## Quality and disagreement policy

Quality dimensions require a written rubric, annotator/source identity, scale,
and aggregation rule. Two labels that sound similar are not merged without a
versioned mapping. Expert composite quality remains a separate target from ROM,
tempo, smoothness, or compensation.

When annotators disagree, preserve all source labels, record the disagreement
type, and adjudicate with a named reviewer or a declared consensus rule. Report
agreement and unresolved masks. Never replace disagreement with a majority label
without recording the policy version.

### Temporal window aggregation protocol for quality logits

Because `MovementPredictionV1.quality_logits` is a pooled representation per temporal window of $W = 128$ frames (tensor shape `[batch, 4]`), frame-level clinical ratings $q_t^{(k)} \in \{0, 1\}$ or continuous kinematic deviations $\theta(t)$ require an explicit window pooling contract. For active repetition frames $W_{active} \subseteq W$:
$$Y_{W, k} = \begin{cases} 1 & \text{if } \frac{1}{|W_{active}|} \sum_{t \in W_{active}} q_t^{(k)} \ge \alpha_{active} \\ 0 & \text{otherwise} \end{cases}$$
where the default threshold is $\alpha_{active} = 0.20$ (at least 20% of active repetition frames exhibit the designated compensation or form error). For continuous trunk lean, the window binary label is $1$ if $\max_{t \in W_{active}} \theta_{trunk}(t) \ge 15^\circ$ and $0$ otherwise. If no clinical aggregation rubric or verified threshold exists, the quality target must remain strictly masked (`quality_mask[:, k] = 0.0`). Continuous angle values must never be passed directly into binary quality logits.

### Absence vs. occlusion vs. assisted protocol

- **Absence (`absent`, capability mask = 0.0)**: Anatomical limb absence (e.g., transradial/transhumeral or transfemoral/transtibial amputation declared in `CapabilityProfileV1`). In `training/src/features/anatomy.py`, missing joints are masked from loss computation and excluded from expected tracking targets; they are not penalized or flagged as occluded or incorrect form.
- **Occlusion (`occluded`, observation mask = 0.0, capability mask > 0.0)**: Anatomically present limb temporarily obscured from camera view. Generates a `joint_occluded` abstention rather than a form error.
- **Assisted (`assisted`, capability mask = 0.80)**: Limb supported by orthosis, strap, or caregiver. Movement is tracked with adjusted kinematics and not penalized for assistive stabilization.

### Salient pose inflection point protocol (apex vs. turnaround)

- **Turnaround Boundary (`rep_start` / `rep_end`)**: Directional reversal marking transition from rest to concentric motion (`rep_start`) or return to rest (`rep_end`).
- **Apex Inflection (`hold` / peak contraction)**: Frame of maximum joint excursion or zero velocity ($\dot{\theta} = 0, \ddot{\theta} < 0$) between concentric and eccentric phases.
- **Enforcement**: The temporal decoder requires an apex detection between `rep_start` and `rep_end`. Any end-boundary candidate not preceded by an apex inflection is rejected as motion jitter.

## Adapter obligations

Every integrated source adapter must provide:

- source/version/checksum and access state;
- participant and session identity;
- coordinate units and frame rate;
- exercise and movement mapping;
- phase and boundary mapping, or explicit masks;
- quality label semantics, scale, and coverage;
- augmentation lineage and synthetic flags;
- split assignment before window generation.

Synthetic occlusion, reduced range, noise, or procedural labels carry lineage
back to the real source sequence and cannot be counted as independent target
population evidence.

## Review checklist

Before a label enters a training manifest, verify that the adapter test passes,
the label semantics match the selected loss, missing values are masked, source
identity is preserved, and the participant split is assigned before overlap
windowing. If any item is unknown, mark the role unavailable and do not publish
a metric for it.
