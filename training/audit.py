"""Audit legacy identity results and corrected prepared-data coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .src.config import Config, load_config, resolve_project_root
from .src.data.dataset import (
    PreparedWindowDataset,
    load_prepared_metadata,
    prepared_split_path,
)
from .src.data.provenance import projection_summary
from .src.evaluation import aggregate_sequence_predictions, audit_sequence_identities
from .src.metrics import compute_metrics


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prediction_parts(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as values:
        outputs = {
            key: values[key]
            for key in ("family_logits", "phase_logits", "boundary_logits", "quality_logits", "tracking_logits")
            if key in values
        }
        targets = {
            key: values[key]
            for key in (
                "family",
                "phase",
                "boundary",
                "quality",
                "quality_mask",
                "tracking_target",
                "time_mask",
            )
            if key in values
        }
    return outputs, targets


def audit_legacy(
    project_root: Path,
    metadata_root: Path,
    predictions_path: Path,
    expected_logical_sequence_count: int | None = 745,
    expected_family_correct: int | None = 700,
) -> dict[str, Any]:
    metadata = load_prepared_metadata(metadata_root, "test")
    outputs, targets = _prediction_parts(predictions_path)
    sequence_outputs, sequence_targets, sequence_metadata = aggregate_sequence_predictions(
        outputs,
        targets,
        metadata,
    )
    metrics = compute_metrics(sequence_outputs, sequence_targets)
    family_correct = int(
        np.sum(sequence_outputs["family_logits"].argmax(axis=-1) == sequence_targets["family"])
    )
    report: dict[str, Any] = {
        "kind": "legacy_identity_audit",
        "metadata_root": str(metadata_root),
        "predictions": str(predictions_path),
        "identity": audit_sequence_identities(metadata),
        "logical_sequence_count": len(sequence_metadata),
        "family_correct": family_correct,
        "family_total": len(sequence_metadata),
        "sequence_metrics": metrics,
        "expected_logical_sequence_count": expected_logical_sequence_count,
        "expected_family_correct": expected_family_correct,
        "matches_previous_audit": (
            (expected_logical_sequence_count is None or len(sequence_metadata) == expected_logical_sequence_count)
            and (expected_family_correct is None or family_correct == expected_family_correct)
        ),
    }
    if project_root:
        report["project_root"] = str(project_root)
    return report


def audit_prepared(
    config: Config,
    project_root: Path,
    processed_root: Path,
    *,
    strict_storage: bool = False,
    chunk_rows: int = 4096,
) -> dict[str, Any]:
    split_metadata = {
        split: load_prepared_metadata(
            processed_root,
            split,
            expected_count=len(PreparedWindowDataset(
                prepared_split_path(processed_root, split),
                validate_features=False,
            )),
        )
        for split in ("train", "validation", "test")
    }
    storage_validation: dict[str, Any] = {}
    if strict_storage:
        for split in ("train", "validation", "test"):
            dataset = PreparedWindowDataset(
                prepared_split_path(processed_root, split),
                validate_features=False,
            )
            storage_validation[split] = dataset.validate_feature_storage(chunk_rows=chunk_rows)
    identities = {split: audit_sequence_identities(rows) for split, rows in split_metadata.items()}
    participant_groups: dict[str, set[str]] = {}
    for split, rows in split_metadata.items():
        participant_groups[split] = {
            f"{row.get('source_dataset', 'unknown')}:{row.get('participant_id', 'unknown')}"
            for row in rows
        }
    overlaps = {
        f"{left}_{right}": sorted(participant_groups[left] & participant_groups[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    split_sources = {
        split: sorted({str(row.get("source_dataset", "unknown")) for row in rows})
        for split, rows in split_metadata.items()
    }
    split_positions = {
        split: sorted({str(row.get("position", "unknown")) for row in rows})
        for split, rows in split_metadata.items()
    }
    required_sources = [str(value) for value in config["data"]["split"].get("required_test_sources", [])]
    required_positions = [str(value) for value in config["data"]["split"].get("required_test_positions", [])]
    coverage_ok = all(source in split_sources["test"] for source in required_sources) and all(
        position in split_positions["test"] for position in required_positions
    )
    index_path = processed_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    report = {
        "kind": "prepared_data_audit",
        "processed_root": str(processed_root),
        "identity_version": identities["test"]["identity_version"],
        "identities_by_split": identities,
        "participant_group_counts": {split: len(groups) for split, groups in participant_groups.items()},
        "participant_group_overlaps": overlaps,
        "split_sources": split_sources,
        "split_positions": split_positions,
        "required_test_sources": required_sources,
        "required_test_positions": required_positions,
        "required_test_coverage_ok": coverage_ok,
        "target_population_coverage": index.get(
            "target_population_coverage",
            {
                "amputee_participants": False,
                "limb_difference_participants": False,
                "wheelchair_user_participants": False,
                "wheelchair_position_public_proxy": False,
                "interpretation": "No target-population claim is supported by this audit.",
            },
        ),
        "index_summary": index,
        "strict_storage_validation": storage_validation,
        "projection_summary": {
            split: projection_summary(rows) for split, rows in split_metadata.items()
        },
        "ok": (
            all(value["identity_collision_count"] == 0 for value in identities.values())
            and not any(overlaps.values())
            and coverage_ok
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "prepared"), required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", default=None, type=Path)
    parser.add_argument("--metadata-root", default=None, type=Path)
    parser.add_argument("--predictions", default=None, type=Path)
    parser.add_argument("--processed-root", default=None, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--strict-storage", action="store_true")
    parser.add_argument("--chunk-rows", default=4096, type=int)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    project_root = resolve_project_root(args.config, args.project_root)
    if args.mode == "legacy":
        metadata_root = args.metadata_root or (project_root / "data/processed")
        predictions_path = args.predictions or (project_root / "artifacts/test_predictions.npz")
        report = audit_legacy(project_root, metadata_root, predictions_path)
        ok = bool(report["matches_previous_audit"])
    else:
        processed_root = args.processed_root or (
            project_root / config["data"].get("processed_root", "data/processed")
        )
        report = audit_prepared(
            config,
            project_root,
            processed_root,
            strict_storage=args.strict_storage,
            chunk_rows=args.chunk_rows,
        )
        ok = bool(report["ok"])

    if args.output is not None:
        output = args.output if args.output.is_absolute() else project_root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(report), indent=2, allow_nan=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
