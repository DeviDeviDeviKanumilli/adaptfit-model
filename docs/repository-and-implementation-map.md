# Repository and implementation map

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** repository entrypoints, source/config/test ownership, and artifact handoff map
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `613ff12` (documentation revision base)
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** supports `system-context-and-dataflow.md`, `current-state.md`, and the training/deployment runbooks
> - **Review trigger:** repository layout, entrypoint, config, test, or artifact ownership change

Use this map before editing. Source code, configuration, generated artifacts,
and manifests outrank prose. A path marked planned is an interface to add; it
is not evidence that a component exists.

## Subsystem map

| Subsystem | Source of truth | Entrypoint/config | Output | Existing tests | Owner/status |
|---|---|---|---|---|---|
| Data contracts | `training/src/data/schema.py` | `training/configs/*.yaml` | prepared sequence records and manifests | `training/tests/test_data_contracts.py` | Data engineering / active |
| Dataset adapters | `training/src/data/adapters.py` | `data/manifests/sources.yaml` | canonical source records | `training/tests/test_pipeline.py` | Data engineering / adapter-specific |
| Identity and splits | `training/src/data/identity.py` and `provenance.py` | dataset manifest split fields | participant/source split metadata | `test_data_contracts.py`, `test_integrity_and_artifacts.py` | Evaluation / active |
| Feature construction | `training/src/features/anatomy.py` | feature section in config | 283-feature tensors and schema JSON | `training/tests/test_features.py` | ML engineering / active |
| Diagnostic features | `training/src/features/diagnostics.py` | `training/configs/v2_diagnostics.yaml` | 378-feature diagnostic tensors | `training/tests/test_features.py` | Research / separate variant |
| Model construction | `training/src/models/build.py`, `tcn.py`, `gru.py`, `heads.py` | model section in config | TCN/GRU modules and parameter counts | `training/tests/test_models.py` | ML engineering / active |
| Training | `training/train.py`, `training/src/runner.py` | `training/configs/*.yaml` | checkpoints, histories, model card | `training/tests/test_streaming_and_training.py` | Training engineering / active |
| Preparation | `training/prepare_data.py`, `training/src/data/prepare.py` | config and manifest | processed data, normalization, audit | `training/tests/test_pipeline.py` | Data engineering / active |
| Evaluation | `training/evaluate.py`, `training/src/evaluation.py` | checkpoint plus split manifest | metrics, predictions, reports | `training/tests/test_corrected_benchmark.py`, `test_v2_quality_pipeline.py` | Evaluation / active |
| Streaming | `training/src/models/streaming.py` | feature/model schema | rolling predictions and runtime state | `training/tests/test_streaming_runtime.py` | Runtime engineering / Python only |
| Decoder | project-forward decoder work package | planned decoder manifest | `WorkoutEventV1` events | planned fixtures | Runtime engineering / planned |
| Recipe feasibility | `docs/exercise-and-capability-schema.md` | recipe catalog/hash | eligible candidate set | planned recommendation fixtures | Product/safety / planned |
| Recommendation | `docs/recommendation-model-plan.md` | planned request/config | ranked workout recommendation | planned recommendation fixtures | Product/ML / planned |
| Mobile bundle | `docs/on-device-deployment.md` | planned artifact manifest | native model bundle | no native tests | Release engineering / unavailable |

## Luna call paths

### Preparation

```text
training/prepare_data.py
  → training/src/config.py
  → training/src/data/prepare.py
  → training/src/data/adapters.py
  → training/src/data/identity.py + provenance.py
  → training/src/features/anatomy.py
  → processed data + manifest + normalization + audit
```

### Training

```text
training/train.py
  → training/src/config.py
  → training/src/data/dataset.py
  → training/src/models/build.py
  → training/src/models/tcn.py or gru.py
  → training/src/models/heads.py
  → training/src/runner.py
  → checkpoint + history + model card
```

### Evaluation

```text
training/evaluate.py
  → training/src/evaluation.py
  → training/src/metrics.py
  → training/src/reporting.py
  → sequence/window metrics + predictions + provenance record
```

### Runtime

```text
pose adapter
  → canonical pose schema
  → features/anatomy.py-compatible 283 layout
  → models/streaming.py
  → prediction heads
  → planned decoder and WorkoutEventV1
```

## Edit boundaries

- Change source code/config first when implementation facts differ; then update
  canonical docs and record the source commit.
- Never overwrite `artifacts/corrected-v1/`, `artifacts/v2-quality/`, or
  `artifacts/v2-quality-fixed/` during a new experiment.
- Do not edit historical reports to make old metrics look current; add a dated
  correction or update `current-state.md`.
- Do not add a recipe to `approved` from ML training alone. Human safety review
  owns approval.
- Do not add a mobile or target-population claim without the corresponding
  artifact and review evidence.

## Canonical question routing

| Question | Start here | Verify against |
|---|---|---|
| What exists now? | `docs/current-state.md` | code, configs, filesystem |
| What fields and versions agree? | `docs/contracts-and-schemas.md` | Python schemas and fixtures |
| Which model should be used? | `docs/model-registry.md` | checkpoint manifest and evaluation |
| How is data labeled? | `docs/annotation-handbook.md` | adapter and label masks |
| Can a recipe be offered? | `docs/recipe-catalog-and-review.md` | reviewed catalog and capability profile |
| Why did a run or event fail? | `docs/failure-and-recovery-matrix.md` | logs and fixture |
| Is a claim releasable? | `docs/requirements-traceability.md` | artifact, split, and gate |
