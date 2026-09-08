# Documentation validation workflow

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** documentation review procedure
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `613ff12` (documentation revision base)
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** supports the validation checklist in `docs/README.md`
> - **Review trigger:** canonical document, contract, artifact, or repository-layout change

This is a lightweight, read-only validation workflow. It is deliberately a
documentation procedure rather than a model/code change. Run it from
`/Users/devk/AdaptFit` before merging a documentation update or asking Luna to
implement a training/deployment task.

The executable structural check is `python3 scripts/validate_docs.py`. It uses
only the standard library, reads source/config/artifact paths, validates the
versioned JSON fixtures, and never prepares data, trains, exports, or edits a
file. The repository CI job runs this command; human review is still required
for safety, privacy, capability, recipe-approval, and challenge language.

## 1. Inventory and metadata

```bash
find docs -type f -name '*.md' -print | sort
test -f training/README.md
for f in docs/*.md training/README.md; do
  rg -q 'Documentation metadata' "$f" || echo "missing metadata: $f"
done
python3 scripts/validate_docs.py
git rev-parse HEAD
git status --short
```

Every Markdown file under `docs/` and `training/README.md` must have the seven
metadata fields. The source commit in a document is a verification point, not a
substitute for checking the current checkout.

## 2. Relative-link and path checks

Review every relative Markdown link from the directory containing its source.
Links to historical or unavailable paths must say so in nearby text. Code,
config, checkpoint, and artifact references must resolve or be marked
`historical`, `unavailable`, or `not staged`.

```bash
rg -n '\]\([^https:#][^)]*\)' docs training/README.md
rg -n '`(/Users/devk/AdaptFit|training/|artifacts/|data/)[^`]*`' docs training/README.md
```

The validator checks links and the existence of canonical artifact directories;
planned or historical paths must be labeled nearby rather than represented by
placeholder files.

When a link contains an anchor, validate the file first and then verify the
heading anchor manually. Do not turn an unavailable artifact into a placeholder
file merely to make a link pass.

## 3. Current-contract checks

Compare [current-state.md](current-state.md) and
[contracts-and-schemas.md](contracts-and-schemas.md) against:

- `training/src/data/schema.py`;
- `training/src/features/anatomy.py`;
- `training/src/models/tcn.py`, `gru.py`, and `heads.py`;
- `training/configs/`;
- `artifacts/*/feature_schema.json` and `training_config.yaml`.
- `docs/motion-jepa-world-model-plan.md` for the explicitly planned, training-only
  AF-MJEPA path. Confirm it has no current checkpoint, export, or mobile claim.
- `docs/recommendation-model-plan.md` for the explicitly planned recommendation
  path. Confirm hard feasibility precedes ranking and no neural recommender is
  claimed as implemented.

The review must confirm 283 inputs, 128-frame windows, the configured stride,
96 TCN channels, five dilations, the 125-frame receptive field, head dimensions,
and normalization/version semantics. If any value differs, update the source
code/config and the canonical docs together; label the old value historical.

## 4. Artifact and metric traceability

For each published metric, record dataset manifest/split, source and participant
scope, Git commit, config hash, checkpoint, schema/normalization/decoder versions,
and artifact path. Check that:

- corrected-v1 is complete;
- v2-quality is prepared-only;
- v2-quality-fixed is partial;
- quality-head coverage is zero unless a new manifest proves otherwise;
- no target-population claim lacks real participant evidence;
- sequence metrics are primary when offsets exist and window fallback is stated
  when they do not.
- any Motion-JEPA experiment records target-encoder/mask/horizon settings,
  participant/source split isolation, collapse diagnostics, and teacher-cache
  provenance; a latent loss alone is not a product result.
- any recommendation claim identifies the eligible candidate set, recipe/catalog
  version, exposure policy, user/time split, fallback behavior, and hard-rule
  audit; ranking metrics alone do not establish safety or clinical benefit.

## 5. Dataset and license checks

For every dataset marked integrated in [dataset-catalog.md](dataset-catalog.md),
verify an adapter, raw/prepared path, checksum or an explicit pending checksum,
participant/session identity handling, label masks, and license/access state.
Move a source back to candidate/research-backlog when any of these is unknown.

## 6. Command and experiment checks

Do not launch training as a documentation check. Read each command in
[training/README.md](../training/README.md) and the runbook and confirm:

- prerequisites and input paths exist;
- output directories are isolated;
- current commands are fresh-training commands, not implied resume commands;
- the Motion-JEPA document is a research contract only and does not imply a
  runnable pretraining command or authorize a training launch;
- preflight, validation cadence, stopping rule, and artifact manifest are
  specified;
- the command does not overwrite a historical benchmark.

If a command needs a code/config change before it can run, label it planned and
record the missing interface in the decision ledger.

## 7. Terminology and safety review

Search for inconsistent or unsafe claims:

```bash
rg -n -i 'clinical|diagnos|safe for|validated amput|wheelchair validation|muscle activation|force|joint loading|quality coverage|production ready' docs training/README.md
```

Have a human review safety wording, capability-focused language, privacy and
consent, exercise approval, challenge claims, and any sentence that could be
read as medical validation. A passing link/path check cannot approve those
claims.

## 8. Machine-readable contract fixtures

Schemas under `docs/schemas/` use `schema_version` and valid/invalid examples
under `docs/examples/contracts/` are checked by the validator. Invalid fixtures
must fail either structural schema validation or an explicit cross-field policy
such as empty-candidate consistency, catalog compatibility, consent state, or
model/feature compatibility.

Normative JSON examples embedded in Markdown use a fence whose info string is
`json contract=<schema-stem>`; the event example in
`contracts-and-schemas.md` uses `json contract=workout-event-v1`. These blocks
are validated against the matching schema. Illustrative snippets without a
`contract=` marker may contain placeholders or type descriptions and are not
treated as executable payloads.

## Completion record

Record the validation date, checkout commit, reviewer, changed documents,
missing/unavailable paths, metric/artifact checks, and unresolved questions in
the pull request or `docs/training-execution-log.md`. The docs are complete only
when Luna can answer an engineering question by following one canonical link
without inferring critical behavior from a historical report.
