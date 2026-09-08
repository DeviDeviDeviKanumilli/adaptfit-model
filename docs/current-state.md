# AdaptFit current state

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** source code, versioned configuration, filesystem artifacts, and verified reports
> - **Last verified:** 2026-09-07
> - **Source commit:** `e75ba65`
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** supersedes scattered “current status” statements in run reports; supports the contracts, training, evaluation, and deployment documents
> - **Review trigger:** any code/config/schema change, new checkpoint, completed evaluation, adapter change, or deployment conversion

This is the single snapshot of what is implemented and evidenced in the
canonical repository at `/Users/devk/AdaptFit`. A statement in another
document may propose or explain work, but it cannot override this file without
an update here and a source artifact.

## Repository and verification

| Item | Current value | Evidence or limit |
|---|---|---|
| Active repository | `/Users/devk/AdaptFit` | The empty `/Users/devk/Documents/ChatGPT/AdaptFit` checkout is not the model source of truth. |
| Commit | `e75ba65` | Re-verify with `git rev-parse HEAD` before reproducing a run. |
| Branch state | Clean except the documentation files being refined | Do not treat uncommitted documentation as model implementation. |
| Test suite | 123 tests passed in the last verified run (`python3 -m pytest -q`) | Re-run after code changes; this documentation pass does not change code. |
| Runtime scope | Python training and streaming runtime | No production mobile bridge is present. |

## Implemented model contract

- Canonical pose input is the 33-joint layout in
  `training/src/data/schema.py`.
- `FeatureSchemaV1` has 283 ordered inputs produced by
  `training/src/features/anatomy.py`. The exact ordering and masks are defined
  in [contracts-and-schemas.md](contracts-and-schemas.md).
- The default temporal window is 128 frames at approximately 30 FPS with
  stride 8 in the current training configuration.
- The TCN uses 96 channels and five causal residual blocks with dilations
  1, 2, 4, 8, and 16. Its causal receptive field is 125 frames. The measured
  corrected-v1 TCN has 307,410 parameters.
- The small GRU is a baseline; the measured corrected-v1 baseline has 68,178
  parameters.
- Current prediction heads are six movement-family logits, five phase logits,
  per-frame repetition start/end logits, four pooled binary quality logits, and
  a per-frame tracking-confidence output. A five-class expert-quality output
  is optional and must remain masked unless its target is present.
- Capability-aware filtering, exercise eligibility, confidence thresholds,
  abstention, decoding, and event emission are deterministic product/runtime
  responsibilities. They are not silently learned by the current network.

## Data adapters and label state

The integrated preparation path has adapters for REHAB24-6, IntelliRehabDS,
MM-Fit, UL-RED, and UCOPhyRehab++. Adapter presence means data can be converted
to the canonical prepared format; it does not mean every task has valid labels.
The registry in [dataset-catalog.md](dataset-catalog.md) is authoritative for
license/access status, label roles, and intended use.

Current label facts:

- Source-level family, phase, repetition, and correctness labels are retained
  only where the source actually supplies them.
- Weak displacement-derived labels, single-clip repetition assumptions, and
  procedural quality templates are masked by default.
- Coverage for the four dimension-specific quality heads is currently **zero**.
  No quality claim may be made from those heads until reviewed targets exist.
- There is no real amputee, limb-difference, or wheelchair-user validation
  cohort. Public seated metadata and synthetic limb masking are engineering
  aids, not target-population evidence.

## Artifact state

| Artifact directory | State | What it proves |
|---|---|---|
| `artifacts/corrected-v1/` | Complete benchmark | TCN and GRU were trained/evaluated and overlapping windows were merged to sequence metrics. This is the current baseline. |
| `artifacts/v2-quality/` | Prepared data only | Preparation and manifests exist; no completed model comparison. |
| `artifacts/v2-quality-fixed/` | Partial | TCN training completed through 72 epochs with best epoch 42, but only window metrics are available; the GRU checkpoint is an interrupted intermediate run at epoch 52. It is not a complete TCN/GRU comparison. |

Checkpoint-level status at this verification:

- Corrected-v1: `artifacts/corrected-v1/checkpoints/tcn_best.pt` and
  `gru_baseline.pt`, with model-specific histories, evaluations, sequence
  predictions, feature schema, normalization statistics, and model card.
- V2-quality: no checkpoint files; only preparation/config/schema/audit files.
- V2-quality-fixed: `checkpoints/tcn_best.pt` and `gru_baseline.pt` exist, but
  the TCN has no merged sequence evaluation in this artifact and the GRU is an
  interrupted intermediate state. Do not select either as a complete v2
  comparison without new evaluation evidence.

Corrected-v1 evidence includes 617 logical test sequences and 7,592 windows
with zero identity collisions. Reported baseline metrics are family accuracy
91.25%, family macro-F1 87.57%, phase accuracy 82.48%, phase macro-F1 58.26%,
repetition-start F1 99.54%, repetition-end F1 18.47%, and repetition-count MAE
0.31. These values are historical run results, not a release claim; use the
artifact manifest and [evaluation-protocol.md](evaluation-protocol.md) for
the required split and metric provenance.

## Product and deployment status

- The model repository does not contain a production exporter, TFLite/Core ML
  conversion, native model bridge, quantized artifact, or mobile golden-fixture
  suite.
- PeddieHacks is a reference application only. Its Android path emits selected
  angles and aggregate confidence, its iOS pose path is stubbed, and its
  deterministic counter is bilateral-first. It is not the active AdaptFit
  implementation.
- Raw camera frames and raw pose should remain local by design, but this has
  not been demonstrated by a production mobile build.
- No mobile latency, memory, thermal, battery, Python/native parity, or
  float/quantized parity gate has passed.

## Known blockers and unsupported claims

1. Quality supervision is absent for all four dimension-specific quality heads.
2. Target-population validation is absent.
3. The current runner builds fresh models and does not yet expose exact resume,
   warm-start, frozen-layer selection, or staged fine-tuning as a documented
   command contract.
4. Repetition-end performance is materially weaker than repetition-start
   performance and needs boundary/decoder review before product use.
5. A production model bundle and native runtime contract do not exist.
6. No document may claim medical, clinical, force, muscle-activation, joint-load,
   or safety certification from the current evidence.

## Next validated actions

1. Freeze this state and the [contracts](contracts-and-schemas.md) before
   changing training or data schemas.
2. Run the evaluation and decoder audit on corrected-v1 without touching the
   locked test set.
3. Add provenance-aware fine-tuning controls and an isolated pilot before any
   teacher or density experiment.
4. Obtain reviewed quality labels and consented target-population recordings.
5. Define and test the model bundle, Python/native golden fixtures, and mobile
   release gates before describing deployment as available.

## Planned work (not current behavior)

The roadmap may propose a density head, teacher distillation, richer quality
targets, personalized calibration, quantization, and native mobile inference.
Those are planned experiments or deliverables. They become current only when
their code/config, artifact, evaluation, and manifest evidence are recorded
here.
