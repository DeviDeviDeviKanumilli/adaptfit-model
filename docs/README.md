# AdaptFit Documentation

This folder records the product and machine-learning design decisions from the AdaptFit planning session.

## Current direction

- Product: fitness routines that adapt to a person’s available movements.
- Initial users: people with limb absence or asymmetry, people with one available arm or leg, and wheelchair or seated users.
- Onboarding: users self-report available limbs, limitations, position, and equipment.
- Inference: on-device; raw camera frames should not leave the device.
- Temporal model: shared causal TCN for the scalable production path, with a small GRU retained as a baseline.
- Initial exercises: five seated, unilateral-friendly exercises.
- Data status: no direct access to target-group participants yet; public datasets and synthetic augmentation will bootstrap development.

## Categories

### Product and scope

- [Product scope and safety boundaries](product-scope.md)
- [Exercise and capability schema](exercise-and-capability-schema.md)

### Existing codebase

- [PeddieHacks26 project audit](existing-project-audit.md)

### Machine learning

- [Scalable, anatomy-informed ML architecture](scalable-ml-architecture.md)
- [Data and training plan](data-and-training-plan.md)
- [Public dataset catalog](dataset-catalog.md)
- [Dataset expansion and ingestion plan](dataset-expansion-plan.md)
- [Verified research and run findings](verified-research-and-run-findings.md)
- [Corrected benchmark rerun](corrected-benchmark-rerun.md)
- [Quality benchmark v2 and overnight run](quality-benchmark-v2.md)
- [Audit fixes and fixed overnight run](audit-fixes-and-overnight.md)
- [On-device deployment plan](on-device-deployment.md)
- [Training implementation guide](../training/README.md)

### Execution

- [Implementation roadmap](implementation-roadmap.md)
- [Decision log and open questions](decisions-and-open-questions.md)
- [Complete session brief](session-brief.md)

## Scope boundary

AdaptFit is being designed as an adaptive fitness and movement-feedback system, not as a diagnostic or medical device. The model should not infer a disability from appearance, diagnose a condition, estimate muscle activation or force from monocular video, or independently make clinical safety decisions.
