"""Train the TCN and GRU models using prepared fixed-length windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .src.config import load_config, resolve_project_root, validate_config
from .src.provenance import environment_summary, git_commit, sha256_file, sha256_value
from .src.reporting import atomic_json_write, initial_run_report
from .src.runner import train_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", default=None, type=Path)
    parser.add_argument("--models", default="tcn,gru", help="Comma-separated model names")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--max-epochs", default=None, type=int, help="Optional override for a short smoke run")
    parser.add_argument("--batch-size", default=None, type=int, help="Optional batch-size override")
    parser.add_argument("--num-workers", default=None, type=int, help="Optional DataLoader worker override")
    parser.add_argument("--init-checkpoint", default=None, type=Path, help="Warm-start model weights without optimizer state")
    parser.add_argument("--resume-checkpoint", default=None, type=Path, help="Resume an exact interrupted run")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train heads while freezing the temporal backbone")
    parser.add_argument("--unfreeze-last-blocks", default=None, type=int, help="Additionally unfreeze this many final TCN blocks")
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--evaluate-test", dest="evaluate_test", action="store_true")
    test_group.add_argument("--skip-test", dest="evaluate_test", action="store_false")
    parser.set_defaults(evaluate_test=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config_path = args.config.expanduser().resolve()
    project_root = resolve_project_root(args.config, args.project_root)
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.init_checkpoint is not None:
        config["training"]["init_checkpoint"] = str(args.init_checkpoint)
    if args.resume_checkpoint is not None:
        config["training"]["resume_checkpoint"] = str(args.resume_checkpoint)
    if args.freeze_backbone:
        config["training"]["freeze_backbone"] = True
    if args.unfreeze_last_blocks is not None:
        config["training"]["unfreeze_last_blocks"] = args.unfreeze_last_blocks
    if args.evaluate_test is not None:
        config["training"]["evaluate_test_after_training"] = args.evaluate_test
    # CLI overrides are part of the effective run configuration and must pass
    # the same contract checks as the YAML file.
    config = validate_config(config)
    seed = int(args.seed if args.seed is not None else config["project"].get("seed", 42))
    requested_device = args.device if args.device != "auto" else config["training"].get("device", "auto")
    results = {}
    for model_name in [value.strip() for value in args.models.split(",") if value.strip()]:
        results[model_name] = train_model(
            model_name=model_name,
            config=config,
            project_root=project_root,
            seed=seed,
            requested_device=requested_device,
        )

    artifacts_root = project_root / config["project"].get("artifacts_root", "artifacts")
    artifacts_root.mkdir(parents=True, exist_ok=True)
    processed_index = project_root / config["data"].get("processed_root", "data/processed") / "index.json"
    prepared_summary = {}
    if processed_index.exists():
        prepared_summary = json.loads(processed_index.read_text(encoding="utf-8"))

    def format_count(value: object) -> str:
        return f"{value:,}" if isinstance(value, int) else str(value)

    processed_root = project_root / config["data"].get("processed_root", "data/processed")
    report = initial_run_report(
        config=config,
        seed=seed,
        data_summary=prepared_summary,
        artifacts_root=artifacts_root,
        processed_root=processed_root,
    )
    report["run"].update(
        {
            "experiment_id": config["project"].get("experiment_id", config["project"].get("name", "unknown")),
            "source_commit": git_commit(project_root),
            "config_hash": sha256_value(config),
            "config_path_hash": sha256_file(config_path) if config_path.exists() else "unavailable",
            "environment": environment_summary(),
            "test_evaluation_performed": all(
                result.get("test_evaluation_performed", False) for result in results.values()
            ) if results else False,
        }
    )
    report["models"] = results
    report["artifacts"] = {
        "metrics": str(artifacts_root / "metrics.json"),
        "model_card": str(artifacts_root / "model_card.md"),
        "histories": {
            name: str(artifacts_root / "metrics" / f"{name}_history.json")
            for name in results
        },
    }
    atomic_json_write(artifacts_root / "metrics.json", report)
    first_result = next(iter(results.values()), {})
    experiment_record = {
        "schema_version": "experiment-record.v1",
        "experiment_id": config["project"].get("experiment_id", config["project"].get("name", "unknown")),
        "git_commit": git_commit(project_root),
        "config_hash": sha256_value(config),
        "dataset_manifest_ids": [str(config["data"].get("processed_root", "unavailable"))],
        "seed": seed,
        "parent_checkpoint": first_result.get("parent_checkpoint"),
        "trainable_layers": first_result.get("trainable_layers", []),
        "status": "complete",
        "artifact_path": str(artifacts_root),
        "metrics": {
            "models": sorted(results),
            "test_evaluation_performed": report["run"]["test_evaluation_performed"],
            "environment": report["run"]["environment"],
        },
    }
    atomic_json_write(artifacts_root / "experiment_record.json", experiment_record)
    model_card = """# AdaptFit temporal model card\n\n"""
    model_card += "This is a research-only movement model trained on public skeleton data, optional UL-RED marker-less rehabilitation sequences, and procedural/synthetic examples.\n\n"
    model_card += "It is not clinical validation and must not diagnose disability, injury, force, muscle activation, or safety.\n\n"
    model_card += "The tracking-confidence head is a self-supervised observability signal from landmark visibility and confidence; it is not clinical uncertainty or movement-quality supervision.\n\n"
    model_card += "UCOPhyRehab++ 3D coordinates are projected to the model's 2D xy contract; depth is discarded because the current mobile pose interface has no calibrated depth channel.\n\n"
    if int(config["model"].get("expert_quality_classes", 0)) > 0:
        model_card += (
            "This run includes an optional five-class composite execution-quality head trained on "
            "UCOPhyRehab++ repetition-level physiotherapist scores. It is separate from the four "
            "dimension-specific quality heads, which remain masked when ROM, tempo, smoothness, "
            "or trunk-compensation labels are unavailable.\n\n"
        )
    model_card += "## Run provenance\n\n"
    model_card += f"- Seed: `{seed}`\n- Configured maximum epochs: `{config['training']['max_epochs']}`\n- Early-stopping patience: `{config['training']['early_stopping_patience']}`\n"
    label_policy = config["data"].get("label_policy", {})
    model_card += (
        "- Single-clip boundary assumptions: "
        f"`{bool(label_policy.get('allow_single_clip_boundaries', False))}`\n"
        "- Procedural quality labels: "
        f"`{bool(label_policy.get('allow_procedural_quality', False))}`\n"
        "- UCO composite expert labels: "
        f"`{bool(label_policy.get('allow_uco_composite_quality', False))}`\n"
        "- Weak phase on multi-repetition recordings: "
        f"`{bool(label_policy.get('allow_weak_phase_on_multi_rep_recordings', False))}`\n"
    )
    if prepared_summary:
        model_card += (
            f"- Prepared sequences: `{format_count(prepared_summary.get('sequence_count', 'unknown'))}`\n"
            f"- Prepared sequence entries: `{format_count(prepared_summary.get('entry_count_with_synthetic', 'unknown'))}`\n"
            f"- Prepared windows: `{format_count(prepared_summary.get('window_counts', {}).get('train', 'unknown'))}` train, "
            f"`{format_count(prepared_summary.get('window_counts', {}).get('validation', 'unknown'))}` validation, "
            f"`{format_count(prepared_summary.get('window_counts', {}).get('test', 'unknown'))}` test\n"
        )
        ul_red_count = prepared_summary.get("sequence_counts_by_source", {}).get("ul_red")
        if ul_red_count is not None:
            model_card += f"- UL-RED sequences: `{format_count(ul_red_count)}`\n"
        model_card += (
            f"- Sequence identity version: `{prepared_summary.get('sequence_identity_version', 'unknown')}`\n"
            "- Participant-level split isolation: `required`\n"
            f"- Corrected test logical sequences: `{format_count(prepared_summary.get('split_summary', {}).get('test', {}).get('sequence_count', 'unknown'))}`\n"
        )
    if int(config["training"]["max_epochs"]) < 100:
        model_card += "\nThis artifact was produced with a shortened smoke-run override; execute `run_overnight.sh` for the configured overnight training run.\n"
    model_card += "\n## Evaluation interpretation\n\n"
    model_card += "Window-level scores can be optimistic because overlapping windows are correlated. Use the sequence-level report after evaluation. Weak, procedural, and synthetic label provenance is reported separately.\n\n"
    model_card += "## Models\n\n"
    for name, result in results.items():
        model_card += (
            f"- `{name}`: {result['parameter_count']:,} total parameters, "
            f"{result.get('trainable_parameter_count', result['parameter_count']):,} trainable; "
            f"checkpoint `{result['checkpoint']}`.\n"
        )
    model_card += "\n## Data limitations\n\nNo real amputee, limb-difference, or wheelchair-user participant recordings were available for this run. Synthetic limb masking, procedural motion, and wheelchair-position labels are robustness aids or public-data proxies only. These results are public-data research benchmarks, not clinical validation. Target-population recordings and expert labels are required before making claims about individualized exercise adaptation.\n"
    (artifacts_root / "model_card.md").write_text(model_card, encoding="utf-8")
    print(json.dumps({name: result["test_metrics"] for name, result in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
