# Decision Log and Open Questions

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** decision ledger with evidence links
> - **Last verified:** 2026-09-08
> - **Source commit:** `ba8bf1a` (code baseline) / `9fe47fb` (documentation revision base)
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** converts the earlier prose log into a source for roadmap and contract decisions
> - **Review trigger:** any decision, evidence change, or revisit trigger firing

## Decision ledger

Use one row per decision. `Accepted` means the current implementation or plan
must follow it; `Proposed` means it still needs evidence; `Blocked` means the
decision cannot be made until the listed prerequisite exists. Dates refer to the
last review, not to the original idea.

| ID | Decision text | Status | Date | Owner | Rationale | Evidence | Affected docs/code | Revisit trigger |
|---|---|---|---|---|---|---|---|---|
| D-001 | AdaptFit starts with capability-focused support for one-arm/one-leg, limb-absence/asymmetry, and seated/wheelchair profiles. | Accepted | 2026-09-07 | Product/safety | Scope is meaningful while avoiding unsupported generalization. | Product scope and current-state limitations | `product-scope.md`, `exercise-and-capability-schema.md` | New approved population or safety review |
| D-002 | Users declare capability; the model does not infer disability from appearance. | Accepted | 2026-09-07 | Product/safety | Separates consented capability from camera observability. | `CapabilityProfileV1` and safety boundary | `contracts-and-schemas.md`, product docs | Consent/usability evidence |
| D-003 | Use a shared 283-feature causal TCN as the scalable v1 path and a small causal GRU as baseline. | Accepted | 2026-09-07 | ML engineering | Shared representation supports recipes and capabilities; TCN is parallelizable/streamable. | Code/config and corrected-v1 artifacts | `scalable-ml-architecture.md`, `current-state.md` | Architecture or schema change |
| D-004 | Keep capability/safety eligibility and event decoding outside the neural network. | Accepted | 2026-09-07 | ML/runtime | Rules remain inspectable and can change without silently retraining behavior. | Runtime design and recipe contract | `scalable-ml-architecture.md`, `contracts-and-schemas.md` | Evidence that a learned rule is safer and reviewable |
| D-005 | Treat corrected-v1 as the complete benchmark; v2-quality is prepared-only and v2-quality-fixed is partial. | Accepted | 2026-09-07 | Evaluation | Prevents incomplete artifacts from being cited as comparisons. | Artifact directories and reports | `current-state.md`, `training-execution-log.md`, evaluation docs | Complete sequence evaluation and manifests |
| D-006 | Keep dimension-specific quality heads disabled until reviewed labels exist. | Accepted | 2026-09-07 | Evaluation/safety | Current label coverage is zero; a score would be fabricated. | Prepared masks and audit reports | `contracts-and-schemas.md`, `evaluation-protocol.md` | Nonzero reviewed coverage and safety review |
| D-007 | Do not claim target-population validation without consented participant evidence. | Accepted | 2026-09-07 | Research/safety | Public seated data and synthetic masks are not target-population evidence. | Current-state and data catalog | All reports and challenge claims | New consented cohort and locked evaluation |
| D-008 | Use staged training: decoder audit, heads-only warm start, partial tuning, then optional teacher. | Proposed | 2026-09-07 | Training engineering | Maximizes useful accuracy per compute; hypotheses require matched experiments. | Efficient training strategy | `efficient-training-strategy.md`, roadmap | Pilot results contradicting the order |
| D-009 | Android is the first deployment validation target; iOS follows native parity evidence. | Proposed | 2026-09-07 | Runtime engineering | Limits device matrix while keeping cross-platform contract explicit. | Deployment plan; no exporter yet | `on-device-deployment.md` | Device constraints or native runtime choice |
| D-010 | Use a versioned model bundle with schema, normalization, decoder, and golden fixtures. | Proposed | 2026-09-07 | Runtime engineering | Prevents silent Python/native drift. | `ModelArtifactManifestV1` | Deployment/evaluation docs | Exporter implementation |
| D-011 | Treat a capability- and equipment-conditioned pose Motion-JEPA as an optional offline pretraining/teacher path; preserve the 283-feature causal TCN as the student and deployment candidate. | Proposed | 2026-09-08 | ML research | Latent future-state prediction may use heterogeneous unlabeled sequences, but the cost and benefit are unproven and it must not change safety or mobile contracts. | `motion-jepa-world-model-plan.md`; I-JEPA/V-JEPA 2/Human-JEPA references | `scalable-ml-architecture.md`, `efficient-training-strategy.md`, `project-forward-plan.md` | Reproducible matched pilot, data-volume change, or evidence that supervised training is sufficient |
| D-012 | Keep workout recommendation separate from movement tracking: deterministic feasibility first, then optional content/neural ranking over eligible reviewed recipes. | Proposed | 2026-09-08 | Product/safety + ML | Prevents a ranker from overriding capability/equipment/approval rules and allows a useful cold-start fallback. | `recommendation-model-plan.md`, `RecommendationRequestV1`, `ExerciseRecipeV1` | `product-scope.md`, `exercise-and-capability-schema.md`, `project-forward-plan.md` | Recipe catalog approval, feedback-log evidence, or safety review |
| D-013 | Use one repository-wide model registry that distinguishes neural families, checkpoint variants, deterministic components, and external pose dependencies. | Accepted | 2026-09-08 | ML engineering | Prevents “how many models exist?” from being answered by mixing current baselines with planned teachers or rules. | `model-registry.md`, current artifacts | `current-state.md`, `scalable-ml-architecture.md`, `project-forward-plan.md` | New model family or artifact lifecycle |
| D-014 | Use normative versioned JSON fixtures under `docs/schemas/` and valid/invalid examples under `docs/examples/contracts/`; `schema_version` is the serialized version field. | Accepted | 2026-09-08 | Engineering | Gives Luna concrete payloads and makes absent-vs-occluded, empty-candidate, and compatibility failures testable. | Contract schemas and examples | `contracts-and-schemas.md`, documentation validator | Contract migration or native adoption |
| D-015 | Keep v1 data handling local-first with no sync or cloud retention by default; any future sync requires a new consent and privacy decision. | Accepted | 2026-09-08 | Product/privacy | Recommendation history needs a lifecycle contract without implying that a production mobile privacy audit has passed. | `privacy-and-data-lifecycle.md` | `on-device-deployment.md`, `recommendation-model-plan.md` | Explicit opt-in sync proposal and privacy review |
| D-016 | Treat the five seated launch recipes as draft until product/safety review; catalog support and camera validation remain separate states. | Accepted | 2026-09-08 | Product/safety | Prevents a catalog record or model metric from being read as exercise suitability evidence. | `recipe-catalog-and-review.md`, exercise schema | `product-scope.md`, recommendation docs | Reviewer approval and camera evidence |
| D-017 | Run structural documentation checks in CI and require human review for safety, privacy, capability, recipe, and challenge claims. | Accepted | 2026-09-08 | Engineering + safety | Links and schema drift are automatable; language and safety boundaries require review. | `documentation-validation.md`, docs validator | repository CI and canonical docs | CI/tooling change |

## Open questions with gates

| ID | Question | Blocking prerequisite | Decision owner | Required evidence |
|---|---|---|---|---|
| Q-001 | Which Android devices and OS versions are minimum supported hardware? | Product demo scope and device access | Runtime/product | Complete-session latency, memory, thermal, and battery measurements |
| Q-002 | Which native runtime/export path is supported first? | Float export spike on a real checkpoint | Runtime | Python/native golden fixtures and operator coverage |
| Q-003 | Who approves recipes, labels, and safety wording? | Named qualified reviewers and review workflow | Product/safety | Signed recipe/label review records |
| Q-004 | Which datasets can be used for the intended distribution? | License/access review per manifest | Data owner | Terms, checksum, retention and attribution record |
| Q-005 | What is the approved user-facing abstention and escalation wording? | Accessibility and safety review | Product/safety | Reviewed copy and fallback usability test |
| Q-006 | Which equipment and substitutions are in the first release? | Recipe inventory and safety review | Product | Approved recipe matrix and empty-candidate behavior |
| Q-007 | How will target-population data be recruited, consented, stored, and reviewed? | Partner and privacy process | Research/safety | Consent, retention, annotation agreement, locked split |
| Q-008 | Does AF-MJEPA improve a held-out student metric enough to justify its added compute? | Reproducible C1/supervised baseline, eligible pose manifest, and bounded compute budget | ML research | Matched supervised-only versus JEPA-assisted student, collapse diagnostics, total-cost report, subgroup and streaming checks |
| Q-009 | Which masking, target-encoder, conditioning, and prediction-horizon choices are valid for AdaptFit motion? | Small finite-value/collapse pilot with no split leakage | ML research | Versioned ablation manifest, frozen probes, future/temporal/joint masking comparison, and failure analysis |
| Q-010 | What recommendation goals, difficulty values, explanation fields, and substitution groups are approved for the first release? | Recipe review and product vocabulary | Product/safety | Versioned catalog metadata, reviewed examples, and empty-candidate behavior |
| Q-011 | Is there enough consented, eligible exposure history to train a neural ranker? | Local feedback schema, exposure logging, and future-period holdout | Product/data/ML | Content/rules baseline, user/time-held-out ranking metrics, constraint audit, privacy review |

## Next eligible work

The next eligible engineering sequence is: freeze contracts; inventory and
evaluate existing checkpoints; resolve decoder/data failures; implement an
isolated warm-start pilot only if the valid comparison cohort exists; then make
one evidence-backed extension choice. Do not begin density, quality, AF-MJEPA,
another teacher, or mobile work by assuming its prerequisite has passed.
