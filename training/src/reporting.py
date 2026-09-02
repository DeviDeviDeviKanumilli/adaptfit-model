"""Shared JSON/reporting helpers for reproducible AdaptFit runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RUN_METRICS_SCHEMA_VERSION = "adaptfit.metrics.v3"


def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return json_safe(value.item())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json_write(path: str | Path, value: Any) -> None:
    """Write a JSON artifact atomically within its destination directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def initial_run_report(
    *,
    config: dict[str, Any],
    seed: int,
    data_summary: dict[str, Any],
    artifacts_root: Path,
    processed_root: Path,
) -> dict[str, Any]:
    """Create the single root schema shared by training and evaluation."""

    return {
        "schema_version": RUN_METRICS_SCHEMA_VERSION,
        "config": config,
        "seed": int(seed),
        "data_summary": data_summary,
        "run": {
            "artifact_root": str(artifacts_root),
            "processed_root": str(processed_root),
            "status": "training_complete",
            "sequence_identity_version": data_summary.get("sequence_identity_version"),
        },
        "models": {},
        "evaluations": {},
        "artifacts": {},
    }
