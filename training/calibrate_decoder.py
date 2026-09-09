"""Calibrate the deterministic repetition decoder on the validation split only."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .src.config import load_config, resolve_project_root, validate_checkpoint_compatibility
from .src.data.dataset import load_prepared_metadata
from .src.evaluation import aggregate_sequence_predictions
from .src.decoder import DecoderConfig, WorkoutEventDecoder, boundary_events_from_targets
from .src.models import build_model
from .src.provenance import git_commit, sha256_file, sha256_value
from .src.runner import collect_predictions, create_loaders, resolve_device
from .src.reporting import atomic_json_write


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _match_times(predicted: list[int], target: list[int], tolerance_ms: int) -> tuple[int, int, int]:
    used: set[int] = set()
    matched = 0
    for value in predicted:
        candidates = [
            (index, abs(value - expected))
            for index, expected in enumerate(target)
            if index not in used and abs(value - expected) <= tolerance_ms
        ]
        if candidates:
            index, _ = min(candidates, key=lambda item: item[1])
            used.add(index)
            matched += 1
    return matched, len(predicted) - matched, len(target) - matched


def _evaluate_config(
    config: DecoderConfig,
    sequence_outputs: dict[str, np.ndarray],
    sequence_targets: dict[str, np.ndarray],
    target_fps: float,
    tolerance_ms: int,
) -> dict[str, float]:
    total_predicted = 0
    total_target = 0
    total_matched = 0
    total_false = 0
    total_missed = 0
    start_predicted: list[int] = []
    start_target: list[int] = []
    end_predicted: list[int] = []
    end_target: list[int] = []
    count_errors: list[float] = []
    latencies: list[float] = []
    abstentions = 0
    empty_target_false_events = 0
    for index in range(len(sequence_targets["family"])):
        valid = np.asarray(sequence_targets["time_mask"][index], dtype=bool)
        length = int(valid.sum())
        if length <= 0:
            continue
        timestamps_ms = np.rint(np.arange(length, dtype=np.float64) * 1000.0 / target_fps).astype(np.int64)
        boundary = sequence_targets["boundary"][index, :length]
        target_pairs = boundary_events_from_targets(boundary, timestamps_ms)
        outputs = sequence_outputs
        decoder = WorkoutEventDecoder(config, model_version="movement-tcn-v1")
        events = decoder.decode_arrays(
            timestamps_ms=timestamps_ms,
            start_probabilities=_sigmoid(outputs["boundary_logits"][index, :length, 0]),
            end_probabilities=_sigmoid(outputs["boundary_logits"][index, :length, 1]),
            phase=outputs["phase_logits"][index, :length],
            tracking_confidences=_sigmoid(outputs["tracking_logits"][index, :length]),
            session_id=f"validation-{index}",
            exercise_id="validation",
            phase_is_logits=True,
        )
        positive_events = [event for event in events if event["count_delta"] > 0]
        abstentions += sum(bool(event["abstention"]) for event in events)
        if not target_pairs:
            empty_target_false_events += len(positive_events)
        predicted_pairs = [
            (int(event["start_timestamp_ms"]), int(event["end_timestamp_ms"]))
            for event in positive_events
        ]
        matched = 0
        used: set[int] = set()
        for start, end in predicted_pairs:
            candidates = [
                (target_index, target_start, target_end)
                for target_index, (target_start, target_end) in enumerate(target_pairs)
                if target_index not in used
                and abs(start - target_start) <= tolerance_ms
                and abs(end - target_end) <= tolerance_ms
            ]
            if candidates:
                target_index, target_start, target_end = min(
                    candidates,
                    key=lambda item: abs(start - item[1]) + abs(end - item[2]),
                )
                used.add(target_index)
                matched += 1
                latencies.append(float(end - target_end))
        total_predicted += len(predicted_pairs)
        total_target += len(target_pairs)
        total_matched += matched
        total_false += len(predicted_pairs) - matched
        total_missed += len(target_pairs) - matched
        count_errors.append(float(abs(len(predicted_pairs) - len(target_pairs))))
        predicted_starts = [start for start, _ in predicted_pairs]
        predicted_ends = [end for _, end in predicted_pairs]
        target_starts = [start for start, _ in target_pairs]
        target_ends = [end for _, end in target_pairs]
        start_predicted.extend(predicted_starts)
        start_target.extend(target_starts)
        end_predicted.extend(predicted_ends)
        end_target.extend(target_ends)

    precision = total_matched / total_predicted if total_predicted else 0.0
    recall = total_matched / total_target if total_target else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    start_matched, start_false, start_missed = _match_times(start_predicted, start_target, tolerance_ms)
    end_matched, end_false, end_missed = _match_times(end_predicted, end_target, tolerance_ms)
    start_precision = start_matched / len(start_predicted) if start_predicted else 0.0
    start_recall = start_matched / len(start_target) if start_target else 0.0
    end_precision = end_matched / len(end_predicted) if end_predicted else 0.0
    end_recall = end_matched / len(end_target) if end_target else 0.0
    start_f1 = 2 * start_precision * start_recall / (start_precision + start_recall) if start_precision + start_recall else 0.0
    end_f1 = 2 * end_precision * end_recall / (end_precision + end_recall) if end_precision + end_recall else 0.0
    return {
        "sequence_count": float(len(count_errors)),
        "predicted_count": float(total_predicted),
        "target_count": float(total_target),
        "matched_count": float(total_matched),
        "false_event_count": float(total_false),
        "missed_event_count": float(total_missed),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "start_f1": float(start_f1),
        "end_f1": float(end_f1),
        "count_mae": float(np.mean(count_errors)) if count_errors else float("nan"),
        "false_events_on_empty_target_sequences": float(empty_target_false_events),
        "abstention_event_count": float(abstentions),
        "mean_end_latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
        "start_false_event_count": float(start_false),
        "start_missed_count": float(start_missed),
        "end_false_event_count": float(end_false),
        "end_missed_count": float(end_missed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--decoder-config", default=Path("training/configs/decoder_v1.yaml"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-root", default=None, type=Path)
    parser.add_argument("--device", default="cpu", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--tolerance-ms", default=250, type=int)
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    project_root = resolve_project_root(config_path, args.project_root)
    device = resolve_device(args.device if args.device != "auto" else config["training"].get("device", "auto"))
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else project_root / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_compatibility(checkpoint, config)
    model = build_model(checkpoint["model_name"], config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    _, validation_loader, _ = create_loaders(config, project_root, num_workers_override=0)
    outputs, targets = collect_predictions(model, validation_loader, device)
    processed_root = project_root / config["data"].get("processed_root", "data/processed")
    metadata = load_prepared_metadata(processed_root, "validation", expected_count=len(targets["family"]))
    sequence_outputs, sequence_targets, _ = aggregate_sequence_predictions(outputs, targets, metadata)
    decoder_values = yaml.safe_load(args.decoder_config.read_text(encoding="utf-8"))
    base = DecoderConfig.from_mapping(decoder_values)
    grid = {
        "start_threshold": [max(0.45, base.start_threshold - 0.10), base.start_threshold, min(0.90, base.start_threshold + 0.10)],
        "end_threshold": [max(0.45, base.end_threshold - 0.10), base.end_threshold, min(0.90, base.end_threshold + 0.10)],
        "tracking_floor": [max(0.40, base.tracking_floor - 0.10), base.tracking_floor],
    }
    candidates: list[dict[str, Any]] = []
    for start_threshold, end_threshold, tracking_floor in itertools.product(
        grid["start_threshold"], grid["end_threshold"], grid["tracking_floor"]
    ):
        candidate = DecoderConfig(
            **{
                **base.to_mapping(),
                "start_threshold": start_threshold,
                "end_threshold": end_threshold,
                "tracking_floor": tracking_floor,
            }
        )
        metrics = _evaluate_config(
            candidate,
            sequence_outputs,
            sequence_targets,
            float(config["data"].get("target_fps", 30.0)),
            args.tolerance_ms,
        )
        candidates.append({"config": candidate.to_mapping(), "metrics": metrics})

    def score(item: dict[str, Any]) -> tuple[float, float, float]:
        metrics = item["metrics"]
        # Count error is primary, with end-event F1 as the tie-breaker and
        # false events on empty recordings as a safety penalty.
        return (
            -float(metrics["count_mae"]),
            float(metrics["end_f1"]),
            -float(metrics["false_events_on_empty_target_sequences"]),
        )

    selected = max(candidates, key=score)
    selected_metrics = selected["metrics"]
    gate = {
        "count_mae": {
            "threshold": 0.40,
            "value": float(selected_metrics["count_mae"]),
            "passed": bool(selected_metrics["count_mae"] <= 0.40),
        },
        "end_f1": {
            "threshold": 0.60,
            "value": float(selected_metrics["end_f1"]),
            "passed": bool(selected_metrics["end_f1"] >= 0.60),
        },
        "false_events_on_empty_target_sequences": {
            "threshold": 0,
            "value": int(selected_metrics["false_events_on_empty_target_sequences"]),
            "passed": bool(selected_metrics["false_events_on_empty_target_sequences"] == 0),
        },
    }
    report = {
        "schema_version": "adaptfit.decoder-calibration.v1",
        "status": "validation_only",
        "split": "validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_hash": sha256_file(checkpoint_path),
        "source_commit": git_commit(project_root),
        "config_hash": sha256_value(config),
        "device": str(device),
        "tolerance_ms": args.tolerance_ms,
        "sequence_count": int(len(sequence_targets["family"])),
        "candidate_count": len(candidates),
        "selected": selected,
        "gate": {
            "status": "pass" if all(item["passed"] for item in gate.values()) else "blocked",
            "requirements": gate,
            "interpretation": (
                "Validation-only decoder calibration passed the declared pre-training gate."
                if all(item["passed"] for item in gate.values())
                else "The baseline decoder does not meet the declared release-oriented event gate; "
                "training or reviewed boundary supervision must improve this before product claims."
            ),
        },
        "candidates": candidates,
        "test_split_used": False,
        "known_limitations": [
            "calibration uses public-data validation sequences only",
            "no target-population validation",
            "no mobile/native parity",
            "quality labels are unavailable",
        ],
    }
    atomic_json_write(args.output, report)
    print(json.dumps({"selected": selected, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
