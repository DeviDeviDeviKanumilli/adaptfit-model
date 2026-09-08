# AdaptFit

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** top-level repository index and operational entry point
> - **Last verified:** 2026-09-07
> - **Source commit:** `b6ac20c`
> - **Owner:** AdaptFit engineering
> - **Supersedes or supports:** project root README; points to canonical docs/ and training/
> - **Review trigger:** new run, artifact layout, or entry point change

AdaptFit is being rewritten as a training-first, on-device movement adaptation
project. The PeddieHacks repository is reference material only; this project
does not nest it under `/app`.

## Training quick start

From `/Users/devk/AdaptFit`:

```bash
python3 -m pip install -e .
python3 -m training.preflight --config training/configs/v1.yaml
python3 -m training.prepare_data --config training/configs/v1.yaml
python3 -m training.train --config training/configs/v1.yaml --models tcn,gru --device auto --seed 42
python3 -m training.evaluate --config training/configs/v1.yaml --checkpoint artifacts/checkpoints/tcn_best.pt
```

The configured bootstrap sources are REHAB24-6, IntelliRehabDS, and the
optional MM-Fit pose archive. Place their archives under `data/raw/` before
running preparation. The preparation stage
also adds a small procedural seed set so all five AdaptFit movement families
have labeled examples. Procedural and synthetic data are not target-population
validation. Prepared splits are written as memory-mapped `.npy` arrays so
training does not load the complete window set into RAM; legacy `.npz` splits
remain readable.

## Documentation

- [Documentation index](docs/README.md)
- [Training guide](training/README.md)
- [Scalable ML architecture](docs/scalable-ml-architecture.md)
- [Data and training plan](docs/data-and-training-plan.md)
- [Public dataset catalog](docs/dataset-catalog.md)
- [On-device deployment plan](docs/on-device-deployment.md)
- [Quality benchmark v2 and overnight run](docs/quality-benchmark-v2.md)

## v2 overnight run

UCOPhyRehab++ is staged under `data/raw/ucophyrehabpp/` and the isolated v2
pipeline is ready. To run the checksum, preparation, smoke test, full TCN/GRU
training, and evaluation overnight:

```bash
cd /Users/devk/AdaptFit
./run_v2_overnight.sh
```

Use `ADAPTFIT_RUN_TRAINING=0 ./run_v2_overnight.sh` for preparation and
verification only.
