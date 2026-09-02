# Corrected benchmark rerun

## Purpose

This document records the corrected identity, participant split, training, and
evaluation run for AdaptFit. It is a separate research benchmark. The original
prepared data and original artifacts remain available for comparison and are
not used for corrected checkpoint selection.

## Artifact isolation

The corrected run uses:

- Prepared windows: `/Users/devk/AdaptFit/data/processed-corrected-v1/`
- Model and report artifacts: `/Users/devk/AdaptFit/artifacts/corrected-v1/`
- Configuration: `/Users/devk/AdaptFit/training/configs/v1_corrected.yaml`
- Reproducible runner: `/Users/devk/AdaptFit/run_corrected_overnight.sh`

The legacy directories remain:

- `/Users/devk/AdaptFit/data/processed/`
- `/Users/devk/AdaptFit/artifacts/`

## Logical identity

The shared identity implementation is in
`/Users/devk/AdaptFit/training/src/data/identity.py`. Its version is
`adaptfit.sequence.v2`.

Identity fields include the dataset, participant, session or workout, and
source-specific clip identity. IntelliRehabDS includes the source member,
gesture, repetition, position, and correctness variant. MM-Fit includes the
workout, activity, set index, and set frame range. REHAB24-6 includes the video,
exercise, camera/orientation, and source frame range. UL-RED includes the
archive, source member, exercise, recording repetition protocol, and pace.
Procedural sequences use their family, repetition, active-side, and session
metadata. Movement family is not used as an identity field.

Evaluation recomputes the corrected identity from `source_metadata`, so legacy
rows with a short or incomplete `sequence_id` can still be evaluated without
merging distinct clips.

## Legacy audit

Before creating the corrected split, the new identity function was applied to
the saved legacy test predictions. It reproduced the previously verified
benchmark:

- Legacy test windows: `6,463`
- Corrected logical test sequences: `745`
- Correct family predictions: `700 / 745`
- Corrected family accuracy: approximately `93.96%`
- Corrected family macro-F1: approximately `89.75%`
- Identity collisions: `0`

## Corrected prepared data

The corrected preparation produced `5,004` source sequences and `11,986`
entries after training-only synthetic variants. The split uses participant
groups, with `50` groups in training, `11` in validation, and `11` in test.
There is no participant-group overlap between splits.

The corrected test split contains:

- `617` logical sequences
- `7,592` windows
- `44` UL-RED logical sequences
- `52` wheelchair-position logical sequences
- `122` REHAB24-6 logical sequences
- `322` IntelliRehabDS logical sequences
- `79` MM-Fit logical sequences
- `50` procedural logical sequences

The prepared-data audit reports zero identity collisions in every split. All
synthetic variants remain in the training split.

Wheelchair-position examples are public-data position proxies, not recordings
identified as wheelchair users. The available data has no real amputee,
limb-difference, or wheelchair-user participant recordings. Those populations
still require consented target-population data before the product can claim
profile generalization.

## Labels and evaluation

The corrected reports include family accuracy and macro-F1, labeled-frame
phase accuracy, per-phase F1 and support, repetition-start and repetition-end
precision/recall/F1, repetition-count MAE, logical sequence counts, identity
collision counts, source and position coverage, calibration, and quality-label
coverage.

ROM, tempo, smoothness, and trunk-compensation metrics remain unavailable. The
current evaluated datasets contain zero reliable labels for those dimensions;
the model must not be described as a form-quality or clinical assessment
system.

Corrected model reports are kept separately:

- `/Users/devk/AdaptFit/artifacts/corrected-v1/metrics/tcn_evaluation.json`
- `/Users/devk/AdaptFit/artifacts/corrected-v1/metrics/gru_evaluation.json`
- `/Users/devk/AdaptFit/artifacts/corrected-v1/metrics/tcn_history.json`
- `/Users/devk/AdaptFit/artifacts/corrected-v1/metrics/gru_history.json`
- `/Users/devk/AdaptFit/artifacts/corrected-v1/metrics.json`

Per-model test predictions and corrected sequence metadata are stored under
`/Users/devk/AdaptFit/artifacts/corrected-v1/predictions/`.

## Smoke run

The two-epoch smoke run completed on MPS for both models. It produced finite
losses, checkpoints, model-specific predictions, and corrected sequence
reports. The smoke evaluation confirmed `617` logical test sequences and zero
identity collisions for both models.

The smoke run is only an artifact and plumbing check. It is not the final
benchmark and is not used to make product claims.

## Final full-run benchmark

The full fresh TCN/GRU run uses seed `42`, float32, MPS when available, the
configured batch size, a maximum of `100` epochs, and early-stopping patience
of `15`. It starts both models from fresh initialization and selects each
checkpoint using corrected validation aggregation.

The full run completed with fresh initialization. TCN training stopped after
50 epochs with its best checkpoint at epoch 35. GRU training stopped after 49
epochs with its best checkpoint at epoch 34. The saved parameter counts are
307,410 for TCN and 68,178 for GRU. Both final evaluations ran on MPS.

Both models use exactly the same corrected test set: `7,592` windows merged
into `617` logical sequences, with zero identity collisions. Sequence-level
metrics are the primary comparison because overlapping windows are correlated.

| Sequence-level metric | TCN | GRU |
|---|---:|---:|
| Movement-family accuracy | 91.25% | 89.47% |
| Movement-family macro-F1 | 87.57% | 84.37% |
| Phase frame accuracy | 82.48% | 82.25% |
| Phase macro-F1 | 58.26% | 54.63% |
| Hold-phase F1 | 31.25% | 0.00% |
| Repetition-start F1 | 99.54% | 83.50% |
| Repetition-end F1 | 18.47% | 5.53% |
| Repetition-count MAE | 0.31 | 30.63 |
| Family calibration ECE | 0.046 | 0.039 |

The sequence-level phase metrics use `64,064` labeled frames. At the window
level, phase accuracy is `72.17%` for TCN and `71.69%` for GRU over `307,642`
labeled frames. Window-level family accuracy is `93.31%` for TCN and `91.75%`
for GRU. These window-level values should not replace the sequence-level
comparison.

TCN per-phase F1 is `69.91%` rest, `75.63%` concentric, `34.88%` hold, and
`68.64%` eccentric. GRU per-phase F1 is `55.57%` rest, `76.32%` concentric,
`0.00%` hold, and `67.49%` eccentric. The unknown phase has no labeled support
in this test split. The low repetition-end scores show that boundary detection
is not production-ready, and the GRU count result is especially poor.

Quality-label coverage is `0.0%` for both models, so ROM, tempo, smoothness,
and trunk-compensation metrics are unavailable rather than negative results.
The test set has public-data coverage from REHAB24-6, IntelliRehabDS, MM-Fit,
UL-RED, and procedural examples, plus seated and wheelchair-position examples.
It has no real amputee, limb-difference, or wheelchair-user participant
recordings. Wheelchair-position examples are public proxies, not target-user
validation.

The final reports and traceable predictions are saved at:

- `/Users/devk/AdaptFit/artifacts/corrected-v1/metrics/tcn_evaluation.json`
- `/Users/devk/AdaptFit/artifacts/corrected-v1/metrics/gru_evaluation.json`
- `/Users/devk/AdaptFit/artifacts/corrected-v1/metrics.json`
- `/Users/devk/AdaptFit/artifacts/corrected-v1/predictions/`

The full test suite passes with `110` tests. The legacy audit still reproduces
`745` corrected logical entries and `700 / 745` correct family predictions.

## Acceptance interpretation

These results are public-data research and synthetic-robustness results. They
are not clinical validation. They do not validate real amputee, limb-
difference, or wheelchair-user movement adaptation, and they do not validate
safety or movement-quality coaching. The next evidence requirement is
consented target-population recordings with exact repetition/phase labels and
expert labels for the quality dimensions.
