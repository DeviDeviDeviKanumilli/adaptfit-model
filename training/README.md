# AdaptFit training

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** training entry point; [the full runbook](../docs/efficient-training-strategy.md) and runner/config are authoritative for commands
> - **Last verified:** 2026-09-08
> - **Source commit:** `f890101`
> - **Owner:** AdaptFit training engineering
> - **Supersedes or supports:** short operational quickstart; rationale and staged workflow live in the linked runbook
> - **Review trigger:** command/config/runner change or new artifact layout

This is the training-first rewrite. It is intentionally independent of the
PeddieHacks application and lives directly in the AdaptFit project root.

Use this file for the shortest operational path. Use
[the canonical training runbook](../docs/efficient-training-strategy.md) for
why a run is justified, warm-start versus exact-resume semantics, staged
fine-tuning, sampler/loss rules, stopping gates, and the Luna TODO list. Read
[current-state](../docs/current-state.md),
[contracts](../docs/contracts-and-schemas.md), and
[evaluation-protocol](../docs/evaluation-protocol.md) before changing a run.

This quickstart does not authorize an overnight run. A command is eligible only
after preflight passes, the source/license gate passes, a locked comparison
split exists, and a new artifact directory is selected.

## Environment

The current development machine has Python 3.13, PyTorch with MPS support,
NumPy, pandas, scikit-learn, PyYAML, and tqdm. Install the project dependencies
with:

```bash
python3 -m pip install -e .
```

## Data placement

The configured public archives belong under `data/raw/`:

```text
data/raw/rehab24_6/2d_joints.zip
data/raw/rehab24_6/Segmentation.csv
data/raw/intellirehabds/SkeletonData.zip
data/raw/mmfit/mm-fit.zip
data/raw/ul_red/S01.zip ... data/raw/ul_red/S10.zip
data/raw/ucophyrehabpp/ucophyrehab2_data.jsonl
data/raw/ucophyrehabpp/dataset_3d_with_angles.json
```

The repository includes adapters for the downloaded REHAB24-6 and
IntelliRehabDS layouts, plus the optional MM-Fit pose archive. MM-Fit's
exercise-set labels provide movement-family supervision and set-level
repetition counts; its individual phase and repetition-boundary labels remain
masked because the source does not provide those targets. Additional sources
can provide canonical `.npz` files
with `joints[T, 33, 2]`, `pose_confidence[T, 33]`, `observed_mask[T, 33]`,
and optional labels. Use a `meta_json` field following the schema in
`training/src/data/schema.py`.

UL-RED is also supported directly. The staged subject archives contain
marker-less clean AMC sequences from ten subjects; the adapter maps their
19-joint hierarchy into the canonical 33-joint layout, preserves the one- or
three-repetition and pace protocol in metadata, and leaves unsupported quality
and boundary labels masked. Verify the downloaded archives with
`data/manifests/ul_red.sha256` before preparing data.

Missing family and phase labels are stored as `-1` and skipped by the loss;
the `Other` and `Unknown` classes are reserved for explicit source labels.

The v2 quality configuration also supports UCOPhyRehab++. It creates
recording-level exact repetition-boundary targets and repetition-level,
five-class composite physiotherapist execution scores. Those scores are kept
in an optional `expert_quality_logits` head; they are not mapped into the four
dimension-specific quality outputs. See
`docs/quality-benchmark-v2.md` for source provenance, limitations, and the
overnight command.

## Run

From the project root:

```bash
python3 -m training.preflight --config training/configs/v1.yaml
python3 -m training.prepare_data --config training/configs/v1.yaml
python3 -m training.train --config training/configs/v1.yaml --models tcn,gru --device auto --seed 42
python3 -m training.evaluate --config training/configs/v1.yaml --checkpoint artifacts/checkpoints/tcn_best.pt
```

The commands above are the current fresh-training path. They do not implement
warm-start or exact interrupted-run resume; do not add `--resume` by inference.
Copy the config and write to an isolated artifact root for every experiment.
The run must record the Git commit, parent checkpoint (if any), dataset
manifest, normalization/schema versions, seed, device, stopping reason, and
metrics required by the evaluation protocol.

The data preparation stage produces fixed windows under `data/processed/` as
per-array `.npy` memmaps and saves normalization statistics and the feature
schema under `artifacts/`. `PreparedWindowDataset` still reads legacy split
`.npz` files for compatibility. Training writes checkpoints and histories;
evaluation writes window-level and sequence-level metrics, predictions, label
provenance summaries, and optional confusion-matrix plots.

Validation is sequence-aware when the prepared JSONL metadata is present:
overlapping windows are merged by sequence and early stopping uses the merged
score. Legacy NPZ-only prepared data remains trainable, but the runner reports
that it is using a window-level fallback because it cannot safely reconstruct
sequence offsets without metadata.

The reference streaming runtime is
`training/src/models/streaming.py`. It accepts normalized feature chunks,
retains the TCN receptive-field context or GRU hidden state, and resets on
explicit session or exercise changes. It is a Python parity harness for the
future native mobile implementation, not the mobile export itself.

The default `v1.yaml` keeps the 283-value on-device feature contract. The
optional `v2_diagnostics.yaml` uses a separate output root and appends causal
acceleration, angular acceleration, rolling-ROM, and smoothness features for
experiments that need richer temporal inputs.

The v2 quality run keeps the same 283-value contract and stores feature
memmaps as float16 to reduce disk use; batches are converted to float32 before
model execution. The fixed run reuses the existing prepared data, applies the
effective phase-label policy at runtime, and writes isolated outputs under
`artifacts/v2-quality-fixed/`. Run
`ADAPTFIT_RUN_TRAINING=0 ./run_v2_quality_fixed_overnight.sh` for verification
without training. See `docs/audit-fixes-and-overnight.md` for the full run
contract.

## Stop conditions

Stop before training when feature width, labels, masks, normalization, split
identity, license/access, or checkpoint compatibility does not match the
selected manifest. Stop during training on non-finite values, leakage,
unexpected missing/unexpected keys, validation degradation beyond the declared
patience, exhausted compute budget, or writes into an existing artifact.
Record the failure and next eligible task in
`docs/training-execution-log.md`; do not hide it by extending the run.

## Verification

Run the full contract suite before an overnight job:

```bash
python3 -m pytest -q
python3 -m unittest discover -s training/tests -p 'test_*.py'
python3 -m compileall -q training
bash -n run_overnight.sh
bash -n run_v2_quality_fixed_overnight.sh
```

The tests cover canonical data validation, participant-level split behavior,
normalization leakage, missing-joint and occlusion handling, memmap integrity,
right-padded streaming inference, masked and provenance-weighted losses,
capability-aware tracking-confidence targets and metrics, checkpoint reloads,
artifact/index consistency, checkpoint/schema compatibility, preflight source
inspection, malformed prediction contracts, legacy metadata fallback, and
sequence-level evaluation aggregation.

## Important interpretation

The procedural seed data and synthetic augmentations make exact v1 exercise
families and missing-joint cases trainable. They are not human target-group
validation. Public phase labels are marked as weak, IntelliRehabDS clip
boundaries are marked as single-clip assumptions, and procedural labels are
reported separately. By default, unsegmented Intelli clips do not contribute
repetition-boundary targets and procedural clips do not contribute quality
targets; both policies can be enabled explicitly for ablation experiments.
Source-level correctness is retained as metadata rather than incorrectly
mapped to ROM quality. Current results must be described as public-data and
synthetic-robustness results, not clinical validation.
