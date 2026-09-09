"""Small, deterministic provenance helpers for training and evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    """Return a content hash without loading the whole artifact into memory."""

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_value(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def git_commit(project_root: str | Path) -> str:
    """Resolve the checked-out commit, or return an explicit unavailable marker."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    commit = completed.stdout.strip()
    return commit or "unavailable"


def environment_summary() -> dict[str, str]:
    """Record the runtime versions needed to interpret a run."""

    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "torch", "yaml"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # pragma: no cover - optional environment metadata
            versions[module_name] = "unavailable"
    return versions


def relative_or_absolute(path: str | Path, root: str | Path) -> str:
    """Prefer stable repository-relative paths while retaining external paths."""

    target = Path(path).resolve()
    base = Path(root).resolve()
    try:
        return str(target.relative_to(base))
    except ValueError:
        return str(target)


def artifact_hashes(
    *,
    project_root: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    feature_schema_path: str | Path | None = None,
    normalization_path: str | Path | None = None,
    prepared_index_path: str | Path | None = None,
) -> dict[str, str]:
    """Hash every available input artifact and mark missing files explicitly."""

    paths = {
        "checkpoint": checkpoint_path,
        "config": config_path,
        "feature_schema": feature_schema_path,
        "normalization": normalization_path,
        "prepared_index": prepared_index_path,
    }
    result: dict[str, str] = {}
    for name, raw_path in paths.items():
        if raw_path is None:
            result[name] = "unavailable"
            continue
        path = Path(raw_path)
        result[name] = sha256_file(path) if path.exists() else "unavailable"
    result["source_commit"] = git_commit(project_root)
    return result


def model_artifact_manifest(
    *,
    project_root: str | Path,
    model_id: str,
    model_version: str,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    config_path: str | Path | None,
    normalization_version: str,
    decoder_version: str,
    status: str,
    metrics: Mapping[str, Any],
    known_limitations: list[str],
    feature_schema_path: str | Path | None = None,
    normalization_path: str | Path | None = None,
    prepared_index_path: str | Path | None = None,
    parent_checkpoint: str | None = None,
    trainable_layers: list[str] | None = None,
    quantization: str | None = None,
) -> dict[str, Any]:
    """Build a manifest that satisfies the checked-in ModelArtifact contract."""

    root = Path(project_root)
    hashes = artifact_hashes(
        project_root=root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        feature_schema_path=feature_schema_path,
        normalization_path=normalization_path,
        prepared_index_path=prepared_index_path,
    )
    manifest_metrics = dict(metrics)
    manifest_metrics.update(
        {
            "artifact_hashes": hashes,
            "environment": environment_summary(),
            "parent_checkpoint": parent_checkpoint,
            "trainable_layers": trainable_layers or [],
        }
    )
    return {
        "schema_version": "model-artifact-manifest.v1",
        "model_id": model_id,
        "model_version": model_version,
        "source_commit": hashes["source_commit"],
        "config_hash": hashes["config"],
        "feature_schema_version": str(config.get("project", {}).get("feature_schema_version", "unavailable")),
        "normalization_version": normalization_version,
        "decoder_version": decoder_version,
        "checkpoint_path": relative_or_absolute(checkpoint_path, root),
        "status": status,
        "known_limitations": list(known_limitations),
        "quantization": quantization,
        "metrics": manifest_metrics,
    }


__all__ = [
    "artifact_hashes",
    "environment_summary",
    "git_commit",
    "model_artifact_manifest",
    "relative_or_absolute",
    "sha256_file",
    "sha256_value",
]
