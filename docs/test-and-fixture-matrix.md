# AdaptFit test and fixture matrix

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** mapping from contracts/failures to verification evidence
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `9fe47fb` (documentation revision base)
> - **Owner:** AdaptFit engineering and evaluation review
> - **Supersedes or supports:** supports `documentation-validation.md`, `evaluation-protocol.md`, and `failure-and-recovery-matrix.md`
> - **Review trigger:** contract, failure behavior, test suite, runtime, or release-gate change

The existing Python suite proves training/data invariants. This matrix makes
the untested product and deployment boundaries visible instead of counting all
tests as product validation.

| Area | Existing evidence | Required fixture or test | Current gap |
|---|---|---|---|
| Data schemas and masks | `test_data_contracts.py` | invalid units, missing masks, identity collision | extend invalid-fixture coverage |
| 283-feature layout | `test_features.py` | ordering, units, absent-vs-occluded masks | native parity unavailable |
| TCN/GRU shapes | `test_models.py` | head dimensions and receptive-field assertion | current Python only |
| Causal streaming | `test_streaming_runtime.py` | chunked versus full-sequence parity, timestamp gaps | native parity unavailable |
| Training/artifact isolation | `test_integrity_and_artifacts.py` | parent hash, manifest completeness, no overwrite | registry validator planned |
| Sequence evaluation | corrected/v2 pipeline tests | participant/source split, event tolerance, count metrics | target-population cohort absent |
| Recipe feasibility | none | one-arm, one-leg, assisted, unknown, occluded, equipment loss | recommender not implemented |
| Recommendation ranking | none | empty set, hard-mask audit, exposure/history leakage | no behavioral dataset/ranker |
| Privacy lifecycle | none | frame/pose redaction, reset, consent withdrawal | no mobile implementation |
| Bundle compatibility | none | model/schema/normalization/decoder/catalog mismatch | exporter/native bridge absent |
| Motion-JEPA | none | collapse, mask/horizon, teacher-cache provenance, matched student | research-only proposal |

## Release interpretation

Passing a Python unit test proves the tested invariant only. It does not prove
mobile parity, privacy behavior, clinical safety, target-population performance,
or recommendation usefulness. Each new fixture must identify its contract
version, expected failure/success, and owner.

## Required additions before release claims

1. Add machine-readable contract fixtures and validate them in CI.
2. Add exhaustive deterministic recipe-feasibility tests before any ranker.
3. Add Python/native golden fixtures before mobile parity claims.
4. Add deletion/redaction tests before on-device privacy claims.
5. Add participant-held-out, source-held-out, and target-population evidence
   before any corresponding performance claim.
