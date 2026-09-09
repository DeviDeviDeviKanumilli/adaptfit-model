"""Evaluate a trained checkpoint and produce aggregate and per-profile reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .src.config import (
    load_config,
    resolve_project_root,
    validate_checkpoint_compatibility,
)
from .src.data.dataset import (
    PreparedWindowDataset,
    effective_phase_label_status,
    load_prepared_metadata,
)
from .src.data.identity import sequence_identity_for_metadata
from .src.data.provenance import projection_summary
from .src.data.schema import PHASE_NAMES
from .src.evaluation import aggregate_sequence_predictions, audit_sequence_identities
from .src.metrics import compute_metrics
from .src.models import build_model
from .src.provenance import model_artifact_manifest
from .src.reporting import RUN_METRICS_SCHEMA_VERSION, atomic_json_write, json_safe
from .src.runner import collect_predictions, create_loaders, resolve_device


def _json_safe(value: Any) -> Any:
    return json_safe(value)


def _group_metrics(
    outputs: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
    group_name: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, object]]:
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[group_name(item)].append(index)
    result = {}
    for name, indices in groups.items():
        selected = np.asarray(indices, dtype=np.int64)
        result[name] = compute_metrics(
            {key: value[selected] for key, value in outputs.items()},
            {key: value[selected] for key, value in targets.items()},
        )
    return result


def _label_summary(
    metadata: list[dict[str, Any]],
    allow_weak_phase_on_multi_rep_recordings: bool = False,
) -> dict[str, Any]:
    """Summarize label provenance so weak-label scores cannot look clinical."""

    effective_phase_status = Counter()
    for item in metadata:
        usable, reason = effective_phase_label_status(item)
        if allow_weak_phase_on_multi_rep_recordings and reason == "multi_rep_weak_phase":
            usable, reason = True, "allowed_by_policy"
        effective_phase_status[reason if usable else f"masked:{reason}"] += 1
    return {
        "window_count": len(metadata),
        "synthetic_window_count": sum(bool(item.get("synthetic", False)) for item in metadata),
        "by_provenance": dict(Counter(item.get("label_provenance", "unknown") for item in metadata)),
        "by_phase_source": dict(Counter(item.get("phase_label_source", "unknown") for item in metadata)),
        "by_effective_phase_status": dict(effective_phase_status),
        "by_boundary_source": dict(Counter(item.get("boundary_label_source", "unknown") for item in metadata)),
        "by_quality_source": dict(Counter(item.get("quality_label_source", "unknown") for item in metadata)),
        "by_expert_quality_source": dict(
            Counter(item.get("expert_quality_label_source", "unlabeled") for item in metadata)
        ),
        "by_expert_quality_status": dict(
            Counter(item.get("expert_quality_label_status", "unavailable") for item in metadata)
        ),
        "by_tracking_target_source": dict(
            Counter(item.get("tracking_target_source", "unknown") for item in metadata)
        ),
    }


def _quality_interpretation(
    label_summary: dict[str, Any],
    quality_label_coverage: float,
) -> dict[str, Any]:
    """State whether quality metrics have usable labels in this evaluation."""

    quality_sources = set(label_summary.get("by_quality_source", {}))
    if not np.isfinite(quality_label_coverage) or quality_label_coverage <= 0.0:
        status = "unavailable"
        message = "No dimension-specific quality labels are present in this evaluation split."
    elif quality_sources and quality_sources <= {"procedural_template"}:
        status = "procedural_only"
        message = "Quality labels are procedural templates and are not evidence of real movement quality."
    elif quality_label_coverage < 0.5:
        status = "limited_coverage"
        message = "Quality labels cover less than half of the evaluated windows."
    else:
        status = "reported_not_clinical"
        message = "Quality metrics are reported for labeled windows but are not clinical validation."
    return {
        "status": status,
        "label_coverage": float(quality_label_coverage),
        "sources": sorted(quality_sources),
        "message": message,
    }


def _coverage_summary(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "window_count": len(metadata),
        "logical_sequence_count": len({sequence_identity_for_metadata(item) for item in metadata}),
        "source_counts": dict(
            Counter(str(item.get("source_dataset", "unknown")) for item in metadata)
        ),
        "position_counts": dict(
            Counter(str(item.get("position", "unknown")) for item in metadata)
        ),
    }


def _expert_quality_interpretation(
    label_summary: dict[str, Any],
    coverage: float,
) -> dict[str, Any]:
    """Describe the optional UCO score without calling it clinical quality."""

    sources = set(label_summary.get("by_expert_quality_source", {})) - {"unlabeled"}
    if not np.isfinite(coverage) or coverage <= 0.0:
        status = "unavailable"
        message = "No composite expert execution-quality labels are present in this evaluation split."
    elif sources <= {"uco_physiotherapist"}:
        status = "weak_composite"
        message = (
            "The score is a 1-5 composite physiotherapist label covering observable execution; "
            "it is not four independent ROM, tempo, smoothness, or compensation labels."
        )
    else:
        status = "reported_not_clinical"
        message = "Composite expert-quality metrics are reported, but they are not clinical validation."
    return {
        "status": status,
        "label_coverage": float(coverage),
        "sources": sorted(sources),
        "message": message,
    }


def _phase_metrics(metrics: dict[str, float]) -> dict[str, dict[str, float]]:
    """Expose the flat metric contract as a readable per-phase report."""

    return {
        "f1": {name: metrics.get(f"phase_f1_{name}", float("nan")) for name in PHASE_NAMES},
        "support": {
            name: metrics.get(f"phase_support_{name}", float("nan")) for name in PHASE_NAMES
        },
    }


def _update_combined_report(artifacts_root: Path, model_name: str, report: dict[str, Any]) -> None:
    """Merge one model evaluation into the run-level report without overwriting another."""

    path = artifacts_root / "metrics.json"
    combined: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                combined = loaded
        except json.JSONDecodeError:
            combined = {}
    # A legacy evaluation report had ``model_name`` and ``overall`` at the
    # root. Preserve it under an explicit key if this code is used on that
    # artifact layout.
    if "model_name" in combined and "overall" in combined:
        combined = {"legacy_evaluation": combined}
    evaluations = combined.setdefault("evaluations", {})
    if not isinstance(evaluations, dict):
        evaluations = {}
        combined["evaluations"] = evaluations
    evaluations[model_name] = report
    combined["schema_version"] = RUN_METRICS_SCHEMA_VERSION
    combined.setdefault("config", report.get("checkpoint_config", {}))
    combined.setdefault("seed", report.get("seed"))
    combined.setdefault("data_summary", {})
    combined.setdefault("models", {})
    combined.setdefault("run", {})
    combined.setdefault("artifacts", {})
    combined["evaluation_identity_version"] = report.get("identity", {}).get("identity_version")
    combined["latest_evaluation_model"] = model_name
    if isinstance(combined.get("run"), dict):
        combined["run"]["latest_evaluation_model"] = model_name
    atomic_json_write(path, combined)


def _write_plots(outputs: dict[str, np.ndarray], targets: dict[str, np.ndarray], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    except ImportError:
        (output_root / "README.txt").write_text(
            "Install matplotlib to generate confusion-matrix plots.\n", encoding="utf-8"
        )
        return
    output_root.mkdir(parents=True, exist_ok=True)
    family_target = targets["family"]
    family_prediction = outputs["family_logits"].argmax(axis=-1)
    family_valid = family_target >= 0
    if family_valid.any():
        matrix = confusion_matrix(family_target[family_valid], family_prediction[family_valid], labels=list(range(6)))
        ConfusionMatrixDisplay(matrix).plot(values_format="d")
        plt.tight_layout()
        plt.savefig(output_root / "family_confusion.png", dpi=140)
        plt.close()

    phase_target = targets["phase"]
    phase_prediction = outputs["phase_logits"].argmax(axis=-1)
    phase_valid = (phase_target >= 0) & targets["time_mask"]
    if phase_valid.any():
        matrix = confusion_matrix(phase_target[phase_valid], phase_prediction[phase_valid], labels=list(range(5)))
        ConfusionMatrixDisplay(matrix).plot(values_format="d")
        plt.tight_layout()
        plt.savefig(output_root / "phase_confusion.png", dpi=140)
        plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--project-root", default=None, type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--num-workers", default=None, type=int, help="Optional DataLoader worker override")
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    project_root = resolve_project_root(config_path, args.project_root)
    device_name = args.device if args.device != "auto" else config["training"].get("device", "auto")
    device = resolve_device(device_name)
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else project_root / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_compatibility(checkpoint, config)
    model_name = checkpoint["model_name"]
    model = build_model(model_name, config).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint weights for {model_name!r} do not match the requested model schema"
        ) from error
    processed_root = project_root / config["data"].get(
        "processed_root",
        f"{config['project'].get('data_root', 'data')}/processed",
    )
    _, _, loader = create_loaders(config, project_root, num_workers_override=args.num_workers)
    dataset: PreparedWindowDataset = loader.dataset  # type: ignore[assignment]
    outputs, targets = collect_predictions(model, loader, device)

    metadata = load_prepared_metadata(
        processed_root,
        "test",
        expected_count=len(targets["family"]),
    )
    sequence_outputs, sequence_targets, sequence_metadata = aggregate_sequence_predictions(
        outputs,
        targets,
        metadata,
    )
    overall_metrics = compute_metrics(outputs, targets)
    sequence_metrics = compute_metrics(sequence_outputs, sequence_targets)
    identity_audit = audit_sequence_identities(metadata)
    label_policy = checkpoint.get("config", {}).get("data", {}).get("label_policy", {})
    allow_weak_multi_rep_phase = bool(
        label_policy.get("allow_weak_phase_on_multi_rep_recordings", False)
    )
    label_summary = _label_summary(metadata, allow_weak_multi_rep_phase)
    expert_quality_interpretation = _expert_quality_interpretation(
        label_summary,
        overall_metrics.get("expert_quality_label_coverage", 0.0),
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "device": str(device),
        "evaluation_split": "test",
        "evaluation_config": config,
        "checkpoint_config": checkpoint.get("config", {}),
        "evaluation_scope": {
            "window_level_metrics": True,
            "overlapping_windows_may_be_correlated": True,
            "sequence_level_metrics_available": True,
            "clinical_validation": False,
            "label_policy": label_policy,
            "label_summary": label_summary,
            "quality_interpretation": _quality_interpretation(
                label_summary,
                overall_metrics["quality_label_coverage"],
            ),
            "expert_quality_interpretation": expert_quality_interpretation,
            "synthetic_variants_in_evaluation": any(
                bool(item.get("synthetic", False)) for item in metadata
            ),
        },
        "identity": identity_audit,
        "logical_sequence_count": identity_audit["logical_sequence_count"],
        "identity_collision_count": identity_audit["identity_collision_count"],
        "effective_phase_label_policy": {
            "allow_weak_phase_on_multi_rep_recordings": allow_weak_multi_rep_phase,
            "window_status_counts": dict(dataset.phase_policy_reasons),
        },
        "projection_summary": projection_summary(metadata),
        "source_and_position_coverage": {
            "windows": _coverage_summary(metadata),
            "logical_sequences": _coverage_summary(sequence_metadata),
        },
        "quality_label_coverage": overall_metrics["quality_label_coverage"],
        "quality_label_status": _quality_interpretation(
            label_summary,
            overall_metrics["quality_label_coverage"],
        )["status"],
        "expert_quality_label_coverage": overall_metrics.get(
            "expert_quality_label_coverage", 0.0
        ),
        "expert_quality_label_status": expert_quality_interpretation["status"],
        "expert_quality_interpretation": expert_quality_interpretation,
        "phase_metrics": {
            "overall": _phase_metrics(overall_metrics),
            "sequence_level": _phase_metrics(sequence_metrics),
        },
        "overall": overall_metrics,
        "sequence_level": sequence_metrics,
        "by_source_and_position": _group_metrics(
            outputs,
            targets,
            metadata,
            lambda item: f"source={item.get('source_dataset', 'unknown')}|position={item.get('position', 'unknown')}",
        ),
        "by_label_provenance": _group_metrics(
            outputs,
            targets,
            metadata,
            lambda item: f"provenance={item.get('label_provenance', 'unknown')}",
        ),
        "by_label_sources": _group_metrics(
            outputs,
            targets,
            metadata,
            lambda item: (
                f"phase={item.get('phase_label_source', 'unknown')}|"
                f"boundary={item.get('boundary_label_source', 'unknown')}|"
                f"quality={item.get('quality_label_source', 'unknown')}"
            ),
        ),
        "by_source_and_position_sequence": _group_metrics(
            sequence_outputs,
            sequence_targets,
            sequence_metadata,
            lambda item: f"source={item.get('source_dataset', 'unknown')}|position={item.get('position', 'unknown')}",
        ),
        "by_label_provenance_sequence": _group_metrics(
            sequence_outputs,
            sequence_targets,
            sequence_metadata,
            lambda item: f"provenance={item.get('label_provenance', 'unknown')}",
        ),
    }
    artifacts_root = project_root / config["project"].get("artifacts_root", "artifacts")
    metrics_root = artifacts_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    atomic_json_write(metrics_root / f"{model_name}_evaluation.json", report)
    _update_combined_report(artifacts_root, model_name, report)
    prediction_root = artifacts_root / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    test_predictions_path = prediction_root / f"{model_name}_test_predictions.npz"
    sequence_predictions_path = prediction_root / f"{model_name}_sequence_predictions.npz"
    sequence_metadata_path = prediction_root / f"{model_name}_sequence_metadata.jsonl"
    np.savez_compressed(test_predictions_path, **outputs, **targets)
    np.savez_compressed(
        sequence_predictions_path,
        **sequence_outputs,
        **sequence_targets,
    )
    temporary_metadata_path = sequence_metadata_path.with_name(
        f".{sequence_metadata_path.name}.tmp"
    )
    with temporary_metadata_path.open("w", encoding="utf-8") as handle:
        for item in sequence_metadata:
            handle.write(json.dumps(_json_safe(item), sort_keys=True) + "\n")
    temporary_metadata_path.replace(sequence_metadata_path)
    report["prediction_artifacts"] = {
        "window_predictions": str(test_predictions_path),
        "sequence_predictions": str(sequence_predictions_path),
        "sequence_metadata": str(sequence_metadata_path),
    }
    parent_artifact_root = config["project"].get("parent_artifact_root")
    contract_root = artifacts_root
    if parent_artifact_root:
        candidate = project_root / str(parent_artifact_root)
        if candidate.exists():
            contract_root = candidate
    feature_schema_path = contract_root / "feature_schema.json"
    normalization_path = contract_root / "normalization_stats.npz"
    manifest = model_artifact_manifest(
        project_root=project_root,
        model_id=f"movement-{model_name}-v1",
        model_version="v1",
        checkpoint_path=checkpoint_path,
        config=config,
        config_path=config_path,
        normalization_version=str(
            config["data"].get("normalization_version", f"normalization.{artifacts_root.name}")
        ),
        decoder_version=str(config["project"].get("decoder_version", "decoder.pending-v1")),
        status="evaluated",
        metrics={
            "split": "test",
            "logical_sequence_count": identity_audit["logical_sequence_count"],
            "identity_collision_count": identity_audit["identity_collision_count"],
            "sequence_level": sequence_metrics,
            "quality_label_coverage": overall_metrics["quality_label_coverage"],
        },
        known_limitations=[
            "public-data benchmark only",
            "no target-population validation",
            "no native/mobile parity",
            "quality-head coverage is zero",
            "decoder calibration is validation-only and not a release gate",
        ],
        feature_schema_path=feature_schema_path,
        normalization_path=normalization_path,
        prepared_index_path=processed_root / "index.json",
        parent_checkpoint=None,
    )
    manifest_path = artifacts_root / "manifests" / f"{model_name}_model_artifact_manifest.json"
    atomic_json_write(manifest_path, manifest)
    report["artifact_manifest"] = str(manifest_path)
    # Rewrite the model-specific report after adding artifact paths, then
    # merge the final version into the combined comparison report.
    atomic_json_write(metrics_root / f"{model_name}_evaluation.json", report)
    _update_combined_report(artifacts_root, model_name, report)
    _write_plots(outputs, targets, artifacts_root / "plots")
    print(json.dumps(_json_safe(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
