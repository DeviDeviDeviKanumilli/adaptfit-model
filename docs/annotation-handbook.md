# AdaptFit annotation handbook

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** label definitions, temporal boundaries, masking, adjudication, and annotation provenance
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `1a46f38` (documentation revision base)
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
