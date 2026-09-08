# AdaptFit Documentation

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** documentation index; source code/config/artifacts remain authoritative for implementation facts
> - **Last verified:** 2026-09-07
> - **Source commit:** `e75ba65`
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** replaces the previous undifferentiated document list
> - **Review trigger:** any new canonical document, status change, schema change, or roadmap change

This index is the entry point for Luna implementation agents. The active model
repository is `/Users/devk/AdaptFit`. The separate `/Users/devk/Documents/ChatGPT/AdaptFit`
checkout and the PeddieHacks application are not implementation sources.

## Status vocabulary

- **`canonical-active`** — current engineering contract or decision. Use it to
  answer implementation questions, then verify source code/config/artifacts.
- **`supporting`** — useful context that supports a canonical document but does
  not override it.
- **`historical`** — a dated experiment, audit, or session record. It describes
  what was true for that run and cannot establish current behavior.
- **`research-backlog`** — proposed investigation or acquisition work. It is not
  evidence that a dataset, model, or feature is available.
- **`superseded`** — retained for traceability and redirects only. Follow the
  linked canonical source instead.

Every refined document starts with the metadata fields: status, authority, last
verified date, source commit, owner, relationship to other documents, and review
trigger.

## Authority hierarchy

When sources disagree, use this order:

1. Source code and schemas.
2. Versioned configuration.
3. Generated artifacts and manifests.
4. Canonical engineering documents.
5. Historical reports and planning notes.

Every current metric, command, path, and model claim must name the source
commit/config/artifact that supports it. If evidence is missing, write
“unavailable” or “not evaluated”; do not infer a result from a proposal.

## Start here

| Question | Canonical answer |
|---|---|
| What is implemented right now? | [Current state](current-state.md) |
| What fields, tensors, masks, and event semantics must agree? | [Contracts and schemas](contracts-and-schemas.md) |
| How should data and labels be ingested? | [Data and training plan](data-and-training-plan.md) and [dataset catalog](dataset-catalog.md) |
| How should a training run be prepared and stopped? | [Efficient training strategy](efficient-training-strategy.md) and [training quickstart](../training/README.md) |
| How is a model evaluated and released? | [Evaluation protocol](evaluation-protocol.md) |
| How does a future mobile bundle work? | [On-device deployment](on-device-deployment.md) |
| What is the dependency-ordered delivery plan? | [Project forward plan](project-forward-plan.md) |
| What is the training execution state and checkpoint inventory? | [Training execution log](training-execution-log.md) |
| Why was a decision made? | [Decision ledger](decisions-and-open-questions.md) |
| What is a prior experiment or audit? | Use the [historical reports](#historical-and-supporting-records), not current-state claims. |

## Canonical active documents

### Product and safety

- [Product scope and safety boundaries](product-scope.md)
- [Exercise and capability schema](exercise-and-capability-schema.md)

### Model, data, and training

- [Scalable ML architecture](scalable-ml-architecture.md)
- [Data and training plan](data-and-training-plan.md)
- [Public dataset catalog](dataset-catalog.md)
- [Efficient training strategy and Luna TODO](efficient-training-strategy.md)

### Runtime and delivery

- [Evaluation protocol](evaluation-protocol.md)
- [On-device deployment plan](on-device-deployment.md)
- [Project forward plan](project-forward-plan.md)
- [Training execution log and run inventory](training-execution-log.md)
- [Decision ledger](decisions-and-open-questions.md)
- [Training implementation quickstart](../training/README.md)

## Supporting and research-backlog documents

- [Dataset expansion and ingestion backlog](dataset-expansion-plan.md) —
  proposed sources and adapters; not the current availability registry.
- [PeddieHacks reference audit](existing-project-audit.md) — supporting
  reference application context, not the active AdaptFit implementation.

## Historical and supporting records

These are preserved for evidence and reproducibility. Each has a status banner,
source metadata, and interpretation limits:

- [Corrected benchmark rerun](corrected-benchmark-rerun.md)
- [Quality benchmark v2](quality-benchmark-v2.md)
- [Audit fixes and overnight run](audit-fixes-and-overnight.md)
- [Verified research and run findings](verified-research-and-run-findings.md)
- [Complete session brief](session-brief.md)
- [Superseded implementation roadmap](implementation-roadmap.md)

## Glossary

- **Capability state:** user-declared `available`, `limited`, `absent`,
  `assisted`, or `unknown`; it is separate from camera observation.
- **Camera visibility:** whether a required joint is observable and usable in a
  frame. An occluded available limb is not an absent limb.
- **Exercise ID / variant ID:** stable machine identifiers for a catalog recipe
  and its unilateral, equipment, or posture variant.
- **Movement family:** coarse model class shared by related exercise recipes.
- **Phase:** an ordered observable movement state used for temporal feedback.
- **Repetition boundary:** a decoded start or end event in sequence time.
- **Density:** nonnegative repetitions-per-frame mass whose sum matches count on
  a fully labeled interval.
- **Tracking confidence:** observability of pose/input quality; not clinical or
  movement-quality confidence.
- **Prediction confidence:** calibrated model/decoder confidence for an output.
- **Abstention:** an explicit decision to withhold feedback because evidence is
  insufficient or incompatible.
- **Composite quality:** a source-provided overall label; it is not automatically
  ROM, tempo, smoothness, or compensation supervision.
- **Target-population validation:** evaluation on consented people in the
  intended population. Public seated data and synthetic masks do not satisfy it.

## Documentation validation

Follow the [documentation validation workflow](documentation-validation.md)
for the read-only inventory, path, contract, artifact, metric, dataset,
command, and terminology checks below.

Before merging a documentation change, verify:

- every relative Markdown link resolves;
- every code/config/artifact path exists or is marked historical/unavailable;
- current dimensions and parameter counts match code/config;
- artifact status matches filesystem evidence;
- each metric names split, source, commit, and artifact;
- integrated datasets have a real adapter plus license/access state;
- historical documents are not linked as the current implementation guide;
- quality and target-population claims have label and participant evidence;
- training commands state prerequisites, outputs, and stop conditions;
- canonical documents contain the metadata block;
- contract terminology is consistent with code and the roadmap.

Human review is required for safety, capability language, privacy/consent,
exercise-review boundaries, challenge-submission claims, and statements that
could be interpreted as medical or clinical validation.

## Scope boundary

AdaptFit is an adaptive fitness and movement-feedback system, not a diagnostic
or medical device. It must not infer disability from appearance, diagnose a
condition, estimate muscle activation or force from monocular video, or make
independent clinical safety decisions.
