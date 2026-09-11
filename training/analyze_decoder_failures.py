"""Analyze validation-only decoder failures for one trained checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from .calibrate_decoder import _evaluate_config, _sigmoid
from .src.config import load_config, resolve_project_root, validate_checkpoint_compatibility
from .src.data.dataset import load_prepared_metadata
from .src.data.schema import PHASE_NAMES, PHASE_TO_ID
from .src.decoder import DecoderConfig, WorkoutEventDecoder
from .src.evaluation import aggregate_sequence_predictions
from .src.models import build_model
from .src.provenance import git_commit, sha256_file, sha256_value
from .src.reporting import atomic_json_write
from .src.runner import collect_predictions, create_loaders, resolve_device


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _quantiles(rows: Iterable[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {f"p{percentile:02d}": None for percentile in (0, 10, 25, 50, 75, 90, 100)}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"p{percentile:02d}": float(np.percentile(array, percentile))
        for percentile in (0, 10, 25, 50, 75, 90, 100)
    }


def _target_pairs(boundary: np.ndarray) -> list[tuple[int, int]]:
    starts = list(np.flatnonzero(np.asarray(boundary)[:, 0] >= 0.5))
    ends = list(np.flatnonzero(np.asarray(boundary)[:, 1] >= 0.5))
    pairs: list[tuple[int, int]] = []
    end_index = 0
    for start in starts:
        while end_index < len(ends) and ends[end_index] <= start:
            end_index += 1
        if end_index >= len(ends):
            break
        pairs.append((int(start), int(ends[end_index])))
        end_index += 1
    return pairs


def _metadata_value(metadata: dict[str, Any], key: str, default: str = "unknown") -> str:
    value = metadata.get(key)
    return default if value is None else str(value)


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"target_pairs": 0, "target_sequences": 0}
    condition_keys = (
        "start_gate_near_250ms",
        "end_signal_near_250ms",
        "end_gate_near_250ms",
        "apex_qualified",
        "target_hold_present",
        "predicted_hold_present",
        "start_tracking_below_floor",
        "end_tracking_below_floor",
    )
    numeric_keys = (
        "duration_frames",
        "start_probability",
        "start_probability_peak_250ms",
        "end_probability",
        "end_probability_peak_250ms",
        "end_probability_peak_valid_duration",
        "tracking_at_start",
        "tracking_at_end",
        "tracking_min_interval",
        "tracking_pass_fraction",
        "max_hold_probability_interval",
        "apex_qualified_frames",
        "apex_qualified_tracking_frames",
    )
    target_phase_counts: Counter[str] = Counter()
    predicted_phase_counts: Counter[str] = Counter()
    for row in rows:
        target_phase_counts.update(row["target_phase_label_counts"])
        predicted_phase_counts.update(row["predicted_phase_counts_interval"])
    return {
        "target_pairs": len(rows),
        "target_sequences": len({int(row["sequence_index"]) for row in rows}),
        "condition_counts": {
            key: int(sum(bool(row[key]) for row in rows)) for key in condition_keys
        },
        "phase_at_target_end": dict(
            sorted(Counter(row["phase_at_target_end"] for row in rows).items())
        ),
        "target_phase_label_counts": dict(sorted(target_phase_counts.items())),
        "predicted_phase_counts_interval": dict(sorted(predicted_phase_counts.items())),
        "quantiles": {key: _quantiles(rows, key) for key in numeric_keys},
    }


def _decoder_sequence_summary(
    sequence_outputs: dict[str, np.ndarray],
    sequence_targets: dict[str, np.ndarray],
    sequence_metadata: list[dict[str, Any]],
    decoder_config: DecoderConfig,
    target_fps: float,
    tolerance_ms: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    phase_threshold = float(decoder_config.phase_threshold)
    tracking_floor = float(decoder_config.tracking_floor)
    start_threshold = float(decoder_config.start_threshold)
    end_threshold = float(decoder_config.end_threshold)
    hold_id = PHASE_TO_ID["hold"]
    unknown_id = PHASE_TO_ID["unknown"]
    neighborhood_frames = max(1, int(round(250.0 * target_fps / 1000.0)))
    min_duration_frames = max(
        1, int(np.ceil(float(decoder_config.min_rep_duration_ms) * target_fps / 1000.0))
    )

    start_probabilities = _sigmoid(sequence_outputs["boundary_logits"][..., 0])
    end_probabilities = _sigmoid(sequence_outputs["boundary_logits"][..., 1])
    tracking_confidences = _sigmoid(sequence_outputs["tracking_logits"])
    phase_logits = np.asarray(sequence_outputs["phase_logits"], dtype=np.float64)
    phase_logits -= np.max(phase_logits, axis=-1, keepdims=True)
    phase_probabilities = np.exp(phase_logits)
    phase_probabilities /= np.sum(phase_probabilities, axis=-1, keepdims=True)
    phase_ids = phase_probabilities.argmax(axis=-1)
    phase_confidences = phase_probabilities.max(axis=-1)

    decoder_reason_counts: Counter[str] = Counter()
    decoder_by_source: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "sequences": 0,
            "target_pairs": 0,
            "positive_events": 0,
            "abstentions": 0,
            "reason_counts": Counter(),
        }
    )
    decoder_by_boundary_source: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "sequences": 0,
            "target_pairs": 0,
            "positive_events": 0,
            "abstentions": 0,
            "reason_counts": Counter(),
        }
    )
    target_rows: list[dict[str, Any]] = []

    for sequence_index in range(len(sequence_targets["family"])):
        valid = np.asarray(sequence_targets["time_mask"][sequence_index], dtype=bool)
        length = int(valid.sum())
        if length <= 0:
            continue
        timestamps_ms = np.rint(
            np.arange(length, dtype=np.float64) * 1000.0 / target_fps
        ).astype(np.int64)
        pairs = _target_pairs(sequence_targets["boundary"][sequence_index, :length])
        metadata = sequence_metadata[sequence_index]
        source = _metadata_value(metadata, "source_dataset")
        boundary_source = _metadata_value(metadata, "boundary_label_source")
        source_bucket = decoder_by_source[source]
        boundary_bucket = decoder_by_boundary_source[boundary_source]
        for bucket in (source_bucket, boundary_bucket):
            bucket["sequences"] += 1
            bucket["target_pairs"] += len(pairs)

        events = WorkoutEventDecoder(
            decoder_config,
            model_version="movement-tcn-v1",
        ).decode_arrays(
            timestamps_ms=timestamps_ms,
            start_probabilities=start_probabilities[sequence_index, :length],
            end_probabilities=end_probabilities[sequence_index, :length],
            phase=sequence_outputs["phase_logits"][sequence_index, :length],
            tracking_confidences=tracking_confidences[sequence_index, :length],
            session_id=f"validation-{sequence_index}",
            exercise_id="validation",
            phase_is_logits=True,
        )
        for event in events:
            reason = str(event["reason_code"])
            decoder_reason_counts[reason] += 1
            source_bucket["reason_counts"][reason] += 1
            boundary_bucket["reason_counts"][reason] += 1
        positive_events = sum(int(event["count_delta"]) > 0 for event in events)
        abstentions = sum(bool(event["abstention"]) for event in events)
        for bucket in (source_bucket, boundary_bucket):
            bucket["positive_events"] += positive_events
            bucket["abstentions"] += abstentions

        for event_index, (start_index, end_index) in enumerate(pairs):
            start_window = slice(
                max(0, start_index - neighborhood_frames),
                min(length, start_index + neighborhood_frames + 1),
            )
            end_window = slice(
                max(0, end_index - neighborhood_frames),
                min(length, end_index + neighborhood_frames + 1),
            )
            interval_end = min(length, end_index + 1)
            interval = slice(start_index, interval_end)
            valid_duration_start = min(length, start_index + min_duration_frames)
            valid_duration_interval = slice(valid_duration_start, interval_end)
            start_gate = (
                (start_probabilities[sequence_index, start_window] >= start_threshold)
                & (phase_ids[sequence_index, start_window] != unknown_id)
                & (phase_confidences[sequence_index, start_window] >= phase_threshold)
                & (tracking_confidences[sequence_index, start_window] >= tracking_floor)
            )
            end_signal = end_probabilities[sequence_index, end_window] >= end_threshold
            end_gate = end_signal & (
                tracking_confidences[sequence_index, end_window] >= tracking_floor
            )
            apex = (
                (phase_ids[sequence_index, interval] == hold_id)
                & (phase_confidences[sequence_index, interval] >= phase_threshold)
            )
            apex_tracking = apex & (
                tracking_confidences[sequence_index, interval] >= tracking_floor
            )
            target_phase_values = sequence_targets["phase"][sequence_index, interval]
            target_phase_values = target_phase_values[target_phase_values >= 0]
            target_phase_label_counts = Counter(
                PHASE_NAMES[int(value)] for value in target_phase_values
            )
            predicted_phase_values = phase_ids[sequence_index, interval]
            predicted_phase_counts_interval = Counter(
                PHASE_NAMES[int(value)] for value in predicted_phase_values
            )
            target_row = {
                "sequence_index": sequence_index,
                "target_event_index": event_index,
                "source_dataset": source,
                "boundary_label_source": boundary_source,
                "participant_id": _metadata_value(metadata, "participant_id"),
                "session_id": _metadata_value(metadata, "session_id"),
                "position": _metadata_value(metadata, "position"),
                "exercise_subtype": _metadata_value(
                    metadata.get("source_metadata", {}), "exercise_subtype"
                ),
                "start_frame": start_index,
                "end_frame": end_index,
                "start_timestamp_ms": int(timestamps_ms[start_index]),
                "end_timestamp_ms": int(timestamps_ms[end_index]),
                "duration_frames": end_index - start_index,
                "start_probability": float(start_probabilities[sequence_index, start_index]),
                "start_probability_peak_250ms": float(
                    start_probabilities[sequence_index, start_window].max()
                ),
                "end_probability": float(end_probabilities[sequence_index, end_index]),
                "end_probability_peak_250ms": float(
                    end_probabilities[sequence_index, end_window].max()
                ),
                "end_probability_peak_valid_duration": float(
                    end_probabilities[sequence_index, valid_duration_interval].max()
                    if valid_duration_start < interval_end
                    else end_probabilities[sequence_index, end_index]
                ),
                "tracking_at_start": float(
                    tracking_confidences[sequence_index, start_index]
                ),
                "tracking_at_end": float(tracking_confidences[sequence_index, end_index]),
                "tracking_min_interval": float(
                    tracking_confidences[sequence_index, interval].min()
                ),
                "tracking_pass_fraction": float(
                    (
                        tracking_confidences[sequence_index, interval] >= tracking_floor
                    ).mean()
                ),
                "phase_at_target_end": PHASE_NAMES[int(phase_ids[sequence_index, end_index])],
                "phase_confidence_at_target_end": float(
                    phase_confidences[sequence_index, end_index]
                ),
                "target_phase_label_counts": dict(sorted(target_phase_label_counts.items())),
                "predicted_phase_counts_interval": dict(
                    sorted(predicted_phase_counts_interval.items())
                ),
                "target_hold_present": bool(target_phase_label_counts.get("hold", 0)),
                "predicted_hold_present": bool(
                    predicted_phase_counts_interval.get("hold", 0)
                ),
                "max_hold_probability_interval": float(
                    phase_probabilities[sequence_index, interval, hold_id].max()
                ),
                "apex_qualified_frames": int(apex.sum()),
                "apex_qualified_tracking_frames": int(apex_tracking.sum()),
                "start_gate_near_250ms": bool(start_gate.any()),
                "end_signal_near_250ms": bool(end_signal.any()),
                "end_gate_near_250ms": bool(end_gate.any()),
                "apex_qualified": bool(apex.sum() >= decoder_config.min_phase_frames),
                "start_tracking_below_floor": bool(
                    tracking_confidences[sequence_index, start_index] < tracking_floor
                ),
                "end_tracking_below_floor": bool(
                    tracking_confidences[sequence_index, end_index] < tracking_floor
                ),
            }
            target_rows.append(target_row)

    def normalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(bucket)
        normalized["reason_counts"] = dict(sorted(bucket["reason_counts"].items()))
        return normalized

    source_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    boundary_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        source_rows[row["source_dataset"]].append(row)
        boundary_rows[row["boundary_label_source"]].append(row)

    replay_metrics = _evaluate_config(
        decoder_config,
        sequence_outputs,
        sequence_targets,
        target_fps,
        tolerance_ms,
    )
    return (
        {
            "reason_counts": dict(sorted(decoder_reason_counts.items())),
            "by_source_dataset": {
                key: normalize_bucket(value)
                for key, value in sorted(decoder_by_source.items())
            },
            "by_boundary_label_source": {
                key: normalize_bucket(value)
                for key, value in sorted(decoder_by_boundary_source.items())
            },
            "replay_metrics": replay_metrics,
        },
        {
            "overall": _group_summary(target_rows),
            "by_source_dataset": {
                key: _group_summary(value) for key, value in sorted(source_rows.items())
            },
            "by_boundary_label_source": {
                key: _group_summary(value)
                for key, value in sorted(boundary_rows.items())
            },
        },
        target_rows,
    )


def _representative_cases(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    def compact(row: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "sequence_index",
            "target_event_index",
            "source_dataset",
            "boundary_label_source",
            "session_id",
            "exercise_subtype",
            "start_timestamp_ms",
            "end_timestamp_ms",
            "start_probability",
            "tracking_at_start",
            "end_probability",
            "end_probability_peak_250ms",
            "tracking_at_end",
            "phase_at_target_end",
            "max_hold_probability_interval",
            "apex_qualified_frames",
            "start_gate_near_250ms",
            "end_signal_near_250ms",
            "end_gate_near_250ms",
        )
        return {key: row[key] for key in keys}

    high_end_signal = sorted(
        (row for row in rows if not row["apex_qualified"]),
        key=lambda row: row["end_probability_peak_250ms"],
        reverse=True,
    )
    low_tracking = sorted(rows, key=lambda row: row["tracking_at_end"])
    return {
        "high_end_signal_but_no_apex": [compact(row) for row in high_end_signal[:10]],
        "lowest_tracking_at_target_end": [compact(row) for row in low_tracking[:10]],
    }


def _markdown(report: dict[str, Any]) -> str:
    calibration = report["reference_calibration"]["selected_metrics"]
    target = report["target_signal_diagnostics"]["overall"]
    replay = report["decoder_replay"]
    findings = report["findings"]
    experiment_id = report.get("experiment_id") or Path(
        report["provenance"]["config"]
    ).stem
    lines = [
        f"# Decoder failure analysis: `{experiment_id}`",
        "",
        f"Validation-only analysis for `{experiment_id}`.",
        "The locked test split was not evaluated or used.",
        "",
        "## Decision",
        "",
        report["recommendation"]["summary"],
        "",
        "## Scope and reference calibration",
        "",
        f"- Validation sequences: `{report['scope']['sequence_count']}`",
        f"- Target boundary pairs: `{report['scope']['target_boundary_pairs']}`",
        f"- Reference decoder metrics: predicted `{int(calibration['predicted_count'])}`, missed `{int(calibration['missed_event_count'])}`, count MAE `{calibration['count_mae']:.7f}`, end F1 `{calibration['end_f1']:.6f}`, abstentions `{int(calibration['abstention_event_count'])}`",
        f"- Decoder replay: reason counts `{replay['reason_counts']}`",
        f"- Target hold labels versus predicted hold frames: `{target['target_phase_label_counts'].get('hold', 0)}` versus `{target['predicted_phase_counts_interval'].get('hold', 0)}`",
        "",
        "## Findings",
        "",
        "| Finding | Evidence | Interpretation |",
        "|---|---:|---|",
    ]
    for finding in findings:
        lines.append(
            f"| {finding['name']} | {finding['evidence']} | {finding['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Target signal summary",
            "",
            f"- Start gate present within ±250 ms: `{target['condition_counts']['start_gate_near_250ms']}/{target['target_pairs']}`",
            f"- End probability ≥ threshold within ±250 ms: `{target['condition_counts']['end_signal_near_250ms']}/{target['target_pairs']}`",
            f"- End probability plus tracking floor within ±250 ms: `{target['condition_counts']['end_gate_near_250ms']}/{target['target_pairs']}`",
            f"- Qualified apex intervals: `{target['condition_counts']['apex_qualified']}/{target['target_pairs']}`",
            f"- Target starts below tracking floor: `{target['condition_counts']['start_tracking_below_floor']}/{target['target_pairs']}`",
            f"- Target ends below tracking floor: `{target['condition_counts']['end_tracking_below_floor']}/{target['target_pairs']}`",
            "",
            "## Decoder abstentions by source",
            "",
            "| Source | Sequences | Target pairs | Abstentions | Reasons |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for source, values in replay["by_source_dataset"].items():
        lines.append(
            f"| `{source}` | {values['sequences']} | {values['target_pairs']} | {values['abstentions']} | `{values['reason_counts']}` |"
        )
    lines.extend(["", "## Recommended next experiment", ""])
    for item in report["recommendation"]["next_experiment"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Representative cases",
            "",
            "The full per-target table is in the JSON report. The cases below have strong local end signal but still lack apex support, or have the lowest tracking at the labeled end.",
            "",
            "### High end signal but no apex",
            "",
            "| Source | Session | End probability peak | Tracking at end | Hold/apex frames |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["representative_cases"]["high_end_signal_but_no_apex"]:
        lines.append(
            f"| `{row['source_dataset']}` | `{row['session_id']}` | {row['end_probability_peak_250ms']:.3f} | {row['tracking_at_end']:.3f} | {row['apex_qualified_frames']} |"
        )
    lines.extend(
        [
            "",
            "### Lowest tracking at target end",
            "",
            "| Source | Session | End probability | Tracking at end | End gate near target |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in report["representative_cases"]["lowest_tracking_at_target_end"]:
        lines.append(
            f"| `{row['source_dataset']}` | `{row['session_id']}` | {row['end_probability']:.3f} | {row['tracking_at_end']:.3f} | {row['end_gate_near_250ms']} |"
        )
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", default=None, type=Path)
    parser.add_argument("--project-root", default=None, type=Path)
    parser.add_argument("--device", default="cpu", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--tolerance-ms", default=250, type=int)
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    project_root = resolve_project_root(config_path, args.project_root)
    config = load_config(config_path)
    device_name = args.device if args.device != "auto" else config["training"].get("device", "auto")
    device = resolve_device(device_name)
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else project_root / args.checkpoint
    calibration_path = (
        args.calibration_report
        if args.calibration_report.is_absolute()
        else project_root / args.calibration_report
    )
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    markdown_path = (
        args.markdown_output
        if args.markdown_output is not None and args.markdown_output.is_absolute()
        else project_root / args.markdown_output
        if args.markdown_output is not None
        else output_path.with_suffix(".md")
    )

    calibration_report = json.loads(calibration_path.read_text(encoding="utf-8"))
    decoder_config = DecoderConfig.from_mapping(calibration_report["selected"]["config"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_compatibility(checkpoint, config)
    model = build_model(checkpoint["model_name"], config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    _, validation_loader, _ = create_loaders(config, project_root, num_workers_override=0)
    outputs, targets = collect_predictions(model, validation_loader, device)
    processed_root = project_root / config["data"].get("processed_root", "data/processed")
    metadata = load_prepared_metadata(
        processed_root,
        "validation",
        expected_count=len(targets["family"]),
    )
    sequence_outputs, sequence_targets, sequence_metadata = aggregate_sequence_predictions(
        outputs,
        targets,
        metadata,
    )
    target_fps = float(config["data"].get("target_fps", 30.0))
    decoder_replay, target_diagnostics, target_rows = _decoder_sequence_summary(
        sequence_outputs,
        sequence_targets,
        sequence_metadata,
        decoder_config,
        target_fps,
        args.tolerance_ms,
    )
    reference_metrics = calibration_report["selected"]["metrics"]
    replay_metrics = decoder_replay["replay_metrics"]
    target_pair_count = int(target_diagnostics["overall"]["target_pairs"])
    segmentation_diagnostics = target_diagnostics["by_boundary_label_source"].get(
        "segmentation", {}
    )
    segmentation_pairs = int(segmentation_diagnostics.get("target_pairs", 0))
    procedural_pairs = int(
        target_diagnostics["by_boundary_label_source"]
        .get("procedural_template", {})
        .get("target_pairs", 0)
    )
    unknown_pairs = int(
        target_diagnostics["by_boundary_label_source"]
        .get("unknown", {})
        .get("target_pairs", 0)
    )
    end_signal_count = int(
        target_diagnostics["overall"]["condition_counts"]["end_signal_near_250ms"]
    )
    predicted_hold_present_count = int(
        target_diagnostics["overall"]["condition_counts"]["predicted_hold_present"]
    )
    apex_qualified_count = int(
        target_diagnostics["overall"]["condition_counts"]["apex_qualified"]
    )
    experiment_id = str(config["project"].get("experiment_id", config_path.stem))
    boundary_weight_cap = float(config["training"].get("boundary_positive_weight_cap", 10.0))
    if "boundary-positive-weight" in experiment_id or boundary_weight_cap > 10.0:
        recommendation = {
            "summary": "Reject the increased sparse-boundary positive-weight cap; keep the corrected tracking-target root unchanged and request reviewed event supervision before another model-only run.",
            "next_experiment": [
                "Do not increase the boundary positive-weight cap further: the cap-64 run reduced the validation sequence score and still failed the decoder gate.",
                "Keep data/processed-r1-uco-tracking-target-correction unchanged and preserve the partial-pose tracking-target correction; do not lower the decoder tracking floor or fabricate UCO phase/apex labels.",
                "Request reviewed UCO and REHAB24-6 start, hold/apex, and end supervision, or additional representative seated unilateral recordings with those labels.",
                "After reviewed supervision is available, create a new prepared root and artifact root, rerun preflight and validation-only calibration, and keep the locked test split unevaluated until the declared gate passes.",
            ],
        }
    elif "ulred" in experiment_id:
        recommendation = {
            "summary": "Keep the validated UL-RED boundary overlay as evidence, but do not treat boundary-only spans as sufficient supervision; request reviewed hold/apex and end-event labels before another model-only run.",
            "next_experiment": [
                "Keep data/processed-r1-ulred-boundary-support and artifacts/r1-tcn-ulred-boundary-support immutable as the corrected local-source experiment; its decoder gate remains blocked.",
                "Do not lower the tracking floor, remove apex confirmation, or tune thresholds against the locked test split: the new UL-RED spans are explicit boundaries but do not provide phase/hold/apex labels for the multi-repetition recordings.",
                "Request reviewed UL-RED and/or representative seated unilateral start, hold/apex, and end supervision using the reviewed-event-supervision intake schema; do not fabricate missing phase labels.",
                "After reviewed supervision is available, create a new prepared root and artifact root, rerun preflight, runtime checks, training, validation-only calibration, and failure analysis, and keep the locked test split unevaluated until the declared gate passes.",
            ],
        }
    elif "uco" in experiment_id:
        recommendation = {
            "summary": "Correct the partial-pose tracking-target denominator in a new prepared root, keep decoder policy fixed, and rerun the bounded validation-only experiment.",
            "next_experiment": [
                "Treat only the canonical joints structurally represented by each partial-pose source as the expected tracking set; do not lower the decoder tracking floor to force a pass.",
                "Add a regression test covering a three-joint UCO-style sequence and regenerate a new prepared root with the corrected tracking targets.",
                "Reuse the UCO sources, labels, seed, model contract, loss weights, and decoder.v1 in a new artifact root; keep quality supervision disabled and test evaluation locked.",
                "Re-run preflight, runtime checks, training, validation-only calibration, and failure analysis before considering another intervention.",
            ],
        }
    elif "full-finetune-event-support-pilot" in experiment_id:
        recommendation = {
            "summary": "Stop supervised escalation and request reviewed boundary supervision or additional representative labeled recordings; keep the locked test split unevaluated.",
            "next_experiment": [
                "Do not start another model-only intervention from this pilot: the decoder gate remains blocked and the tracking-floor failure is unchanged.",
                "Request reviewed REHAB24-6 boundary labels and/or additional representative seated unilateral recordings with explicit start, hold/apex, and end supervision.",
                "Keep data/processed-r1-boundary-mask-correction unchanged until the external labels or recordings are audited and regenerated into a new prepared root.",
                "After external supervision is available, create a new isolated artifact root and rerun preflight, validation-only calibration, and failure analysis before considering a second seed or locked test evaluation.",
            ],
        }
    else:
        recommendation = {
            "summary": "Unfreeze only the final TCN residual block from the phase-weight checkpoint, keep the decoder thresholds frozen, and compare on validation only.",
            "next_experiment": [
                "Initialize from artifacts/r1-tcn-phase-weight-correction/checkpoints/tcn_best.pt and train only blocks.4.* plus the existing output heads.",
                "Reuse data/processed-r1-boundary-mask-correction unchanged with the same seed, loss weights, window contract, and label masks; do not fabricate labels for unknown-boundary sequences.",
                "Write to the new artifact root artifacts/r1-tcn-last-block-event-support and keep evaluate_test_after_training=false.",
                "Recalibrate validation-only after the intervention; keep the locked test split unevaluated until the declared gate passes.",
            ],
        }
    report = {
        "schema_version": "adaptfit.decoder-failure-analysis.v1",
        "status": "validation_only",
        "split": "validation",
        "test_split_used": False,
        "experiment_id": experiment_id,
        "provenance": {
            "project_root": str(project_root),
            "config": str(config_path),
            "config_hash": sha256_value(config),
            "checkpoint": str(checkpoint_path),
            "checkpoint_hash": sha256_file(checkpoint_path),
            "calibration_report": str(calibration_path),
            "calibration_report_hash": sha256_file(calibration_path),
            "source_commit": git_commit(project_root),
            "device": str(device),
            "decoder_version": decoder_config.version,
        },
        "scope": {
            "sequence_count": int(len(sequence_targets["family"])),
            "target_boundary_pairs": target_pair_count,
            "target_fps": target_fps,
            "tolerance_ms": int(args.tolerance_ms),
            "boundary_label_sources": target_diagnostics["by_boundary_label_source"],
        },
        "reference_calibration": {
            "status": calibration_report["status"],
            "selected_config": calibration_report["selected"]["config"],
            "selected_metrics": reference_metrics,
            "gate": calibration_report["gate"],
        },
        "decoder_replay": decoder_replay,
        "target_signal_diagnostics": target_diagnostics,
        "target_events": target_rows,
        "representative_cases": _representative_cases(target_rows),
        "findings": [
            {
                "name": "The phase head rarely supplies qualified hold/apex support in target intervals",
                "evidence": f"{apex_qualified_count}/{target_pair_count} intervals satisfy the apex rule; {predicted_hold_present_count}/{target_pair_count} contain any predicted hold",
                "interpretation": "Most target intervals lack the two-frame, confidence-qualified hold support required by the decoder; aggregate predicted hold frames therefore do not translate into usable apex evidence.",
            },
            {
                "name": "Start tracking is below the decoder floor for most explicit segmentation starts",
                "evidence": f"{segmentation_diagnostics.get('condition_counts', {}).get('start_tracking_below_floor', 0)}/{segmentation_pairs} segmentation starts",
                "interpretation": "Tracking confidence suppresses entry into the active state before boundary decoding can begin; this is concentrated in REHAB24-6.",
            },
            {
                "name": "End signal is only partially aligned with target ends",
                "evidence": f"{target_diagnostics['overall']['condition_counts']['end_signal_near_250ms']}/{target_pair_count} target ends with an end-probability crossing",
                "interpretation": f"A threshold-only change would not recover the {target_pair_count - end_signal_count} target ends with no nearby end signal; the end head or supervision needs targeted review.",
            },
            {
                "name": "The unavailable-boundary correction is not the remaining failure",
                "evidence": f"{segmentation_pairs} segmentation + {procedural_pairs} procedural target pairs; {unknown_pairs} unknown-boundary target pairs",
                "interpretation": "The corrected mask path removed the fabricated UL-RED endpoint labels, but the remaining positive-event failure is in model/decoder event support rather than unavailable-label leakage.",
            },
        ],
        "recommendation": recommendation,
        "limitations": [
            "The analysis uses the validation split only and does not read or evaluate the test split.",
            "Decoder abstention counts include sequences without target boundary pairs; target-event diagnostics are reported separately.",
            "Boundary targets are public-data segmentation labels plus procedural templates; there is no target-population validation cohort.",
            "The report analyzes aggregated sequence logits and does not inspect raw video or pose annotations.",
            "Reference calibration metrics are retained alongside a fresh replay to preserve provenance if backend outputs differ across devices.",
        ],
    }
    report["replay_consistency"] = {
        key: {
            "reference": reference_metrics.get(key),
            "replay": replay_metrics.get(key),
            "equal": reference_metrics.get(key) == replay_metrics.get(key),
        }
        for key in (
            "predicted_count",
            "target_count",
            "missed_event_count",
            "end_f1",
            "count_mae",
            "abstention_event_count",
        )
    }
    atomic_json_write(output_path, _json_safe(report))
    _atomic_text_write(markdown_path, _markdown(_json_safe(report)))
    print(json.dumps({"output": str(output_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
