# PeddieHacks26 Project Audit

Source repository: [DeviDeviDeviKanumilli/PeddieHacks26](https://github.com/DeviDeviDeviKanumilli/PeddieHacks26)

This audit reflects the repository state examined during the planning session.

## Repository structure

- `apps/mobile`: React Native/Expo mobile application.
- `apps/web`: legacy web application.
- `apps/api`: Fastify API.
- `packages/contracts`: shared API contracts.
- `packages/domain`: exercise catalog, capability compatibility, and workout generation.
- `packages/intelligence`: deterministic pose-feature and workout-intelligence runtime.
- `supabase`: database schema and seed data.
- `prisma`: database-related definitions.
- `model`: Python MediaPipe calibration and exercise-analysis lab.

## Python model folder

The [model folder](https://github.com/DeviDeviDeviKanumilli/PeddieHacks26/tree/main/model) is currently a desktop calibration and analysis lab rather than a trained production ML model.

### `vision_model.py`

- Uses MediaPipe Pose Landmarker Lite.
- Tracks one pose.
- Computes angles from three landmarks.
- Applies visibility thresholds.
- Uses 2D image/pixel coordinates, which are not yet anatomy-normalized body measurements.

### `exercise_analyzer.py`

- Computes minimum, maximum, mean, and range of motion.
- Contains a target-and-return repetition state machine.
- `ExerciseSetTracker` currently counts bilateral completion based on configured limbs.
- Tracks sets, rest, and terminal summaries.

### `exercise_catalog.py`

- Contains static muscle and joint bitmasks.
- Defines approximately 10 exercise entries in the Python catalog.
- The masks are useful metadata but do not yet encode full biomechanics or user-specific capability.

### `exercise_selector.py`

- Filters exercises using target-muscle and unavailable-muscle/joint bitmasks.
- This is a deterministic selector, not a learned recommendation model.

### `main.py`

- Webcam loop defaults to a biceps-curl workflow.
- Uses hardcoded left and right elbow-related landmark triples.
- The exercise argument changes the summary label but does not yet fully select tracking logic.

The repository’s Python tests passed during the inspection, but the Python model did not have a dedicated training pipeline or learned weights.

## Mobile and native tracking

The Android native module in [`apps/mobile/modules/adaptfit-pose`](https://github.com/DeviDeviDeviKanumilli/PeddieHacks26/tree/main/apps/mobile/modules/adaptfit-pose) runs MediaPipe locally and returns a small signal set including left angle, right angle, and confidence.

The mobile tracking recipes currently include six keys:

- Seated biceps curl.
- Seated band row.
- Seated march.
- Seated knee extension.
- Sit-to-stand.
- Wall push-up.

Only the seated biceps-curl path is currently marked as calibrated. iOS pose support is incomplete compared with Android.

The current bridge exposes too little information for an anatomy-informed temporal model. Trunk compensation, asymmetry, optional limbs, and joint-level visibility require a richer native feature contract or native temporal inference.

## TypeScript intelligence package

The [`packages/intelligence`](https://github.com/DeviDeviDeviKanumilli/PeddieHacks26/tree/main/packages/intelligence) package is deterministic. It includes:

- Rolling angle features.
- Velocity and range calculations.
- Variance and stability features.
- Confidence handling.
- Repetition tracking.
- Threshold-based motion inspection.
- Privacy and validation utilities.

It does not currently load network weights and is not the learned ML layer.

## Domain and database metadata

The domain catalog and Supabase schema contain richer exercise metadata than the Python catalog. The project already describes:

- Primary, secondary, and stabilizing body roles.
- Body demand intensity.
- Required or limited capabilities.
- Equipment options.
- Goals.
- Muscles and tracking profiles.
- Confidence floors, ROM targets, tempo targets, and form rules.

This metadata should become the anatomy and safety constraint layer around the learned temporal model.

## Important gaps

1. Multiple catalogs exist: Python, mobile, TypeScript domain, and database. They need one canonical exercise and tracking schema.
2. The current tracker is bilateral-first, which does not support one-arm or one-leg use as a first-class case.
3. Only a small set of angles reaches the mobile application.
4. There is no learned temporal model or model export pipeline.
5. There is no target-population training dataset.
6. iOS does not yet have the same native pose path.
7. Static muscle/joint masks are not enough to represent biomechanics, compensation, or uncertainty.

## Recommended foundation work

Before training, unify the exercise recipe format and define a canonical pose-feature contract. This prevents the neural model, native bridge, and adaptation rules from developing incompatible representations.

