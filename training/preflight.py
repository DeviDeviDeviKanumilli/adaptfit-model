"""Preflight checks for data availability and training configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

from .src.config import Config, load_config, resolve_project_root
from .src.data.adapters import (
    load_canonical_npz_files,
    load_intellirehabds,
    load_mmfit,
    load_rehab24_6,
    load_ul_red,
    load_ucophyrehabpp,
)
from .src.features.anatomy import feature_dimensions


def _source_has_files(source: Mapping[str, Any], project_root: Path) -> bool:
    path = project_root / source["path"]
    if not path.exists():
        return False
    name = source["name"]
    if name == "rehab24_6":
        return (path / "2d_joints.zip").exists() and (path / "Segmentation.csv").exists()
    if name == "intellirehabds":
        return (path / "SkeletonData.zip").exists() or any(path.rglob("*.txt"))
    if name == "mmfit":
        return (path / "mm-fit.zip").exists() or any(path.rglob("*_labels.csv"))
    if name == "ul_red":
        return any(path.glob("S*.zip")) or any(path.rglob("*.amc"))
    if name == "ucophyrehabpp":
        return (
            (path / str(source.get("metadata_file", "ucophyrehab2_data.jsonl"))).exists()
            and (path / str(source.get("pose_file", "dataset_3d_with_angles.json"))).exists()
        )
    # The generic adapter consumes canonical NPZ files. A loose NPY file is
    # not enough to prove that the source can be loaded into the contract.
    return any(path.rglob("*.npz"))


def _load_source_sequences(
    source: Mapping[str, Any],
    project_root: Path,
    config: Config,
) -> list:
    """Load one source for a truthful preflight availability check."""

    path = project_root / source["path"]
    name = source["name"]
    if name == "rehab24_6":
        return load_rehab24_6(path)
    if name == "intellirehabds":
        return load_intellirehabds(path)
    if name == "mmfit":
        return load_mmfit(path)
    if name == "ul_red":
        return load_ul_red(path)
    if name == "ucophyrehabpp":
        policy = config["data"].get("label_policy", {})
        return load_ucophyrehabpp(
            path,
            allow_composite_quality=bool(policy.get("allow_uco_composite_quality", True)),
        )
    return load_canonical_npz_files(path)


def _inspect_source(
    source: Mapping[str, Any],
    project_root: Path,
    config: Config,
    inspect_sequences: bool = True,
) -> dict[str, Any]:
    enabled = bool(source.get("enabled", True))
    status: dict[str, Any] = {
        "name": source["name"],
        "enabled": enabled,
        "file_present": False,
        "available": False,
        "sequence_count": 0,
        "license_review_required": source.get("license_review_required", False),
    }
    if not enabled:
        return status

    status["file_present"] = _source_has_files(source, project_root)
    if not status["file_present"]:
        return status

    if not inspect_sequences:
        status["inspection_mode"] = "files"
        status["available"] = True
        status["sequence_count"] = None
        return status

    try:
        sequences = _load_source_sequences(source, project_root, config)
    except Exception as error:  # pragma: no cover - exercised through source-specific failures
        status["inspection_error"] = f"{type(error).__name__}: {error}"
        return status
    status["inspection_mode"] = "full"
    status["sequence_count"] = len(sequences)
    status["available"] = bool(sequences)
    if source["name"] == "ucophyrehabpp":
        status["expert_quality_sequence_count"] = sum(
            bool(sequence.expert_quality_mask) for sequence in sequences
        )
        status["exact_boundary_sequence_count"] = sum(
            sequence.metadata.get("boundary_label_source") == "strong"
            for sequence in sequences
        )
        status["source_positions"] = sorted(
            str(sequence.metadata.get("source_position", sequence.position))
            for sequence in sequences
        )
    if not status["available"]:
        status["inspection_error"] = "files were found but no valid canonical sequences were loaded"
    return status


def run_preflight(
    config: Config,
    project_root: Path,
    inspect_sequences: bool = True,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    roles_present: set[str] = set()
    source_status = []
    strict_source_paths = bool(config["data"].get("strict_source_paths", False))
    for source in config["data"]["sources"]:
        status = _inspect_source(source, project_root, config, inspect_sequences=inspect_sequences)
        source_status.append(status)
        if status["available"]:
            roles_present.add(source["role"])
        elif strict_source_paths and source.get("enabled", True):
            detail = status.get("inspection_error") or "enabled source files are missing or invalid"
            errors.append(
                f"Enabled source is unavailable under strict_source_paths: {source['name']} at "
                f"{project_root / source['path']} ({detail})"
            )
        elif source.get("required", False):
            detail = status.get("inspection_error") or "required files are missing"
            errors.append(
                f"Required source is unavailable: {source['name']} at "
                f"{project_root / source['path']} ({detail})"
            )

    required_roles = set(config["data"].get("required_roles", []))
    if "rep_labeled" not in roles_present:
        errors.append("No rep-labeled source is staged; add REHAB24-6 or canonical rep-labeled NPZ files.")
    if "seated_or_wheelchair" not in roles_present:
        errors.append("No seated/wheelchair source is staged; add IntelliRehabDS or a canonical seated source.")
    if required_roles - roles_present:
        warnings.append(f"Configured roles not detected: {sorted(required_roles - roles_present)}")

    device = config["training"].get("device", "auto")
    if device == "auto" and not torch.backends.mps.is_available() and not torch.cuda.is_available():
        warnings.append("No accelerator detected; training will use CPU.")
    if device == "mps" and not torch.backends.mps.is_available():
        errors.append("MPS was requested but is unavailable.")
    expected_feature_dim, expected_continuous_dim = feature_dimensions(
        bool(config["features"].get("include_diagnostics", False))
    )
    if int(config["features"]["input_dim"]) != expected_feature_dim:
        errors.append(f"Selected feature schema requires {expected_feature_dim} input features.")
    if int(config["features"]["continuous_dim"]) != expected_continuous_dim:
        errors.append(f"Selected feature schema requires {expected_continuous_dim} continuous features.")

    if not inspect_sequences:
        warnings.append(
            "File-only preflight checks configured paths and required files; "
            "source decoding is deferred to preparation or a full preflight."
        )
    result = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "inspection_mode": "full" if inspect_sequences else "files",
        "sources": source_status,
    }
    for source in source_status:
        status = (
            "available"
            if source["available"]
            else "unavailable"
            if source["file_present"]
            else "missing"
        )
        print(
            f"source={source['name']} enabled={source['enabled']} status={status} "
            f"sequences={source['sequence_count'] if source['sequence_count'] is not None else 'deferred'}"
        )
        if source.get("name") == "ucophyrehabpp":
            print(
                f"source={source['name']} expert_quality_sequences="
                f"{source.get('expert_quality_sequence_count', 0)} "
                f"exact_boundary_sequences={source.get('exact_boundary_sequence_count', 0)}"
            )
        if source.get("inspection_error"):
            print(f"source={source['name']} detail={source['inspection_error']}")
    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", default=None, type=Path)
    parser.add_argument(
        "--mode",
        choices=("full", "files"),
        default="full",
        help="Decode source sequences in full mode; only check staged files in files mode.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    project_root = resolve_project_root(args.config, args.project_root)
    result = run_preflight(config, project_root, inspect_sequences=args.mode == "full")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
