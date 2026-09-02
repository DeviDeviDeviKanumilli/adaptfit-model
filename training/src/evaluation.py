"""Sequence-level aggregation for overlapping prepared windows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .data.identity import (
    SEQUENCE_IDENTITY_VERSION,
    identity_source_signature,
    sequence_identity_for_metadata,
)


def aggregate_sequence_predictions(
    outputs: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Merge overlapping windows before computing sequence-level metrics.

    Window metadata is the source of sequence identity and global frame offset.
    Per-frame logits are averaged only where a window contributes. Labels use
    the first available non-missing value, which preserves masked-label
    semantics while avoiding duplicate supervision from overlap.
    """

    if not metadata:
        raise ValueError("Cannot aggregate sequence predictions without window metadata")
    required_outputs = ("family_logits", "phase_logits", "boundary_logits", "quality_logits")
    required_targets = (
        "family",
        "phase",
        "boundary",
        "quality",
        "quality_mask",
        "time_mask",
    )
    missing_outputs = [key for key in required_outputs if key not in outputs]
    missing_targets = [key for key in required_targets if key not in targets]
    if missing_outputs or missing_targets:
        raise ValueError(
            f"Prediction aggregation is missing outputs={missing_outputs} targets={missing_targets}"
        )
    time_mask = np.asarray(targets["time_mask"])
    if time_mask.ndim != 2:
        raise ValueError("targets.time_mask must have shape [windows, frames]")
    expected_count = len(time_mask)
    if len(metadata) != expected_count:
        raise ValueError(
            f"Metadata count ({len(metadata)}) does not match prediction count ({expected_count})"
        )
    if np.asarray(outputs["family_logits"]).shape[:1] != (expected_count,):
        raise ValueError("outputs.family_logits must have one row per metadata item")
    for name in ("phase_logits", "boundary_logits"):
        values = np.asarray(outputs[name])
        if values.ndim != 3 or values.shape[:2] != time_mask.shape:
            raise ValueError(f"outputs.{name} must have shape [windows, frames, classes]")
    quality_logits = np.asarray(outputs["quality_logits"])
    if quality_logits.ndim != 2 or quality_logits.shape[0] != expected_count:
        raise ValueError("outputs.quality_logits must have one row per metadata item")
    for name in required_outputs:
        if not np.isfinite(np.asarray(outputs[name])).all():
            raise ValueError(f"outputs.{name} must be finite")
    if np.asarray(targets["phase"]).shape != time_mask.shape:
        raise ValueError("targets.phase must have the same [windows, frames] shape as time_mask")
    if np.asarray(targets["boundary"]).shape[:2] != time_mask.shape:
        raise ValueError("targets.boundary must have the same [windows, frames] prefix as time_mask")
    if np.asarray(targets["family"]).shape != (expected_count,):
        raise ValueError("targets.family must have one label per metadata item")
    if np.asarray(targets["quality"]).shape != (expected_count, quality_logits.shape[-1]):
        raise ValueError("targets.quality must have one row per metadata item")
    if np.asarray(targets["quality_mask"]).shape != np.asarray(targets["quality"]).shape:
        raise ValueError("targets.quality_mask must match targets.quality")
    in_time = time_mask.astype(bool)
    if not in_time.any(axis=1).all():
        raise ValueError("each prediction window must contain at least one valid frame")
    if in_time.shape[1] > 1 and np.any(in_time[:, 1:] & ~in_time[:, :-1]):
        raise ValueError("prediction time_mask must be right-padded")
    has_tracking_outputs = "tracking_logits" in outputs
    has_tracking_targets = "tracking_target" in targets
    if has_tracking_outputs != has_tracking_targets:
        raise ValueError("tracking_logits and tracking_target must be provided together")
    has_tracking = has_tracking_outputs and has_tracking_targets
    if has_tracking:
        if np.asarray(outputs["tracking_logits"]).shape != time_mask.shape:
            raise ValueError("outputs.tracking_logits must match targets.time_mask")
        if np.asarray(targets["tracking_target"]).shape != time_mask.shape:
            raise ValueError("targets.tracking_target must match targets.time_mask")
        if not np.isfinite(np.asarray(outputs["tracking_logits"])).all():
            raise ValueError("outputs.tracking_logits must be finite")

    has_expert_output = "expert_quality_logits" in outputs
    if has_expert_output:
        if "expert_quality" not in targets or "expert_quality_mask" not in targets:
            raise ValueError(
                "expert_quality_logits requires expert_quality and expert_quality_mask targets"
            )
        expert_logits = np.asarray(outputs["expert_quality_logits"])
        expert_target = np.asarray(targets["expert_quality"])
        expert_mask = np.asarray(targets["expert_quality_mask"]).astype(bool)
        if expert_logits.ndim != 2 or expert_logits.shape[0] != expected_count or expert_logits.shape[1] < 2:
            raise ValueError("outputs.expert_quality_logits must have shape [windows, classes]")
        if expert_target.shape != (expected_count,) or expert_mask.shape != (expected_count,):
            raise ValueError("expert quality targets and masks must have shape [windows]")
        if not np.isfinite(expert_logits).all():
            raise ValueError("outputs.expert_quality_logits must be finite")

    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[sequence_identity_for_metadata(item)].append(index)

    sequence_values: list[dict[str, np.ndarray]] = []
    sequence_metadata: list[dict[str, Any]] = []
    for indices in groups.values():
        first = metadata[indices[0]]
        lengths = []
        for index in indices:
            if "window_start" not in metadata[index]:
                raise ValueError("Each metadata row needs window_start for sequence aggregation")
            raw_start = metadata[index]["window_start"]
            try:
                start = int(raw_start)
            except (TypeError, ValueError) as error:
                raise ValueError("metadata.window_start must be an integer") from error
            if isinstance(raw_start, bool) or start != raw_start or start < 0:
                raise ValueError("metadata.window_start must be a non-negative integer")
            valid_length = int(in_time[index].sum())
            if "window_end" in metadata[index]:
                try:
                    window_end = int(metadata[index]["window_end"])
                except (TypeError, ValueError) as error:
                    raise ValueError("metadata.window_end must be an integer") from error
                if window_end != start + valid_length:
                    raise ValueError("metadata.window_end must equal window_start plus valid frames")
            lengths.append(start + valid_length)
        length = max(1, max(lengths))
        phase_sum = np.zeros((length, outputs["phase_logits"].shape[-1]), dtype=np.float64)
        phase_count = np.zeros(length, dtype=np.float64)
        boundary_sum = np.zeros((length, outputs["boundary_logits"].shape[-1]), dtype=np.float64)
        boundary_count = np.zeros(length, dtype=np.float64)
        phase_target = np.full(length, -1, dtype=np.int64)
        boundary_target = np.full(
            (length, outputs["boundary_logits"].shape[-1]), -1.0, dtype=np.float32
        )
        if has_tracking:
            tracking_sum = np.zeros(length, dtype=np.float64)
            tracking_count = np.zeros(length, dtype=np.float64)
            tracking_target = np.full(length, -1.0, dtype=np.float32)

        for index in indices:
            start = int(metadata[index]["window_start"])
            valid_length = int(in_time[index].sum())
            end = min(length, start + valid_length)
            if end <= start:
                continue
            local = slice(0, end - start)
            global_slice = slice(start, end)
            phase_sum[global_slice] += outputs["phase_logits"][index, local]
            phase_count[global_slice] += 1.0
            boundary_sum[global_slice] += outputs["boundary_logits"][index, local]
            boundary_count[global_slice] += 1.0
            if has_tracking:
                tracking_sum[global_slice] += outputs["tracking_logits"][index, local]
                tracking_count[global_slice] += 1.0

            local_phase = targets["phase"][index, local]
            local_boundary = targets["boundary"][index, local]
            global_indices = np.arange(start, end)
            phase_slots = (phase_target[global_indices] < 0) & (local_phase >= 0)
            phase_target[global_indices[phase_slots]] = local_phase[phase_slots]
            for boundary_index in range(local_boundary.shape[-1]):
                boundary_slots = (
                    (boundary_target[global_indices, boundary_index] < 0)
                    & (local_boundary[:, boundary_index] >= 0)
                )
                boundary_target[global_indices[boundary_slots], boundary_index] = local_boundary[
                    boundary_slots, boundary_index
                ]
            if has_tracking:
                local_tracking = targets["tracking_target"][index, local]
                tracking_slots = (
                    (tracking_target[global_indices] < 0.0)
                    & np.isfinite(local_tracking)
                    & (local_tracking >= 0.0)
                )
                tracking_target[global_indices[tracking_slots]] = local_tracking[tracking_slots]

        phase_logits = (phase_sum / np.maximum(phase_count[:, None], 1.0)).astype(np.float32)
        boundary_logits = (boundary_sum / np.maximum(boundary_count[:, None], 1.0)).astype(np.float32)
        quality_logits = np.mean(outputs["quality_logits"][indices], axis=0).astype(np.float32)
        family_logits = np.mean(outputs["family_logits"][indices], axis=0).astype(np.float32)
        sequence_item: dict[str, np.ndarray] = {
            "family_logits": family_logits,
            "phase_logits": phase_logits,
            "boundary_logits": boundary_logits,
            "quality_logits": quality_logits,
            "family": np.asarray(targets["family"][indices[0]], dtype=np.int64),
            "phase": phase_target,
            "boundary": boundary_target,
            "quality": targets["quality"][indices[0]].copy(),
            "quality_mask": targets["quality_mask"][indices[0]].copy(),
            "time_mask": phase_count > 0,
        }
        if has_expert_output:
            sequence_item["expert_quality_logits"] = np.mean(
                outputs["expert_quality_logits"][indices], axis=0
            ).astype(np.float32)
            expert_indices = np.asarray(indices, dtype=np.int64)
            valid_expert = np.asarray(targets["expert_quality_mask"])[expert_indices].astype(bool)
            valid_expert &= np.asarray(targets["expert_quality"])[expert_indices] >= 0
            if valid_expert.any():
                first_expert = expert_indices[np.flatnonzero(valid_expert)[0]]
                sequence_item["expert_quality"] = np.asarray(
                    targets["expert_quality"][first_expert], dtype=np.int64
                )
                sequence_item["expert_quality_mask"] = np.asarray(True, dtype=bool)
            else:
                sequence_item["expert_quality"] = np.asarray(-1, dtype=np.int64)
                sequence_item["expert_quality_mask"] = np.asarray(False, dtype=bool)
        if has_tracking:
            sequence_item["tracking_logits"] = (
                tracking_sum / np.maximum(tracking_count, 1.0)
            ).astype(np.float32)
            sequence_item["tracking_target"] = tracking_target
        sequence_values.append(sequence_item)
        canonical_metadata = dict(first)
        canonical_metadata["sequence_id"] = sequence_identity_for_metadata(first)
        canonical_metadata["sequence_identity_version"] = SEQUENCE_IDENTITY_VERSION
        sequence_metadata.append(canonical_metadata)

    max_length = max(item["phase_logits"].shape[0] for item in sequence_values)
    sequence_outputs: dict[str, np.ndarray] = {
        "family_logits": np.stack([item["family_logits"] for item in sequence_values]),
        "phase_logits": np.zeros(
            (len(sequence_values), max_length, outputs["phase_logits"].shape[-1]),
            dtype=np.float32,
        ),
        "boundary_logits": np.zeros(
            (len(sequence_values), max_length, outputs["boundary_logits"].shape[-1]),
            dtype=np.float32,
        ),
        "quality_logits": np.stack([item["quality_logits"] for item in sequence_values]),
    }
    if has_expert_output:
        sequence_outputs["expert_quality_logits"] = np.stack(
            [item["expert_quality_logits"] for item in sequence_values]
        )
    sequence_targets: dict[str, np.ndarray] = {
        "family": np.stack([item["family"] for item in sequence_values]),
        "phase": np.full((len(sequence_values), max_length), -1, dtype=np.int64),
        "boundary": np.full(
            (len(sequence_values), max_length, outputs["boundary_logits"].shape[-1]),
            -1.0,
            dtype=np.float32,
        ),
        "quality": np.stack([item["quality"] for item in sequence_values]),
        "quality_mask": np.stack([item["quality_mask"] for item in sequence_values]),
        "time_mask": np.zeros((len(sequence_values), max_length), dtype=bool),
    }
    if has_expert_output:
        sequence_targets["expert_quality"] = np.asarray(
            [item["expert_quality"] for item in sequence_values], dtype=np.int64
        )
        sequence_targets["expert_quality_mask"] = np.asarray(
            [item["expert_quality_mask"] for item in sequence_values], dtype=bool
        )
    if has_tracking:
        sequence_outputs["tracking_logits"] = np.zeros(
            (len(sequence_values), max_length), dtype=np.float32
        )
        sequence_targets["tracking_target"] = np.full(
            (len(sequence_values), max_length), -1.0, dtype=np.float32
        )

    for index, item in enumerate(sequence_values):
        length = item["phase_logits"].shape[0]
        sequence_outputs["phase_logits"][index, :length] = item["phase_logits"]
        sequence_outputs["boundary_logits"][index, :length] = item["boundary_logits"]
        sequence_targets["phase"][index, :length] = item["phase"]
        sequence_targets["boundary"][index, :length] = item["boundary"]
        sequence_targets["time_mask"][index, :length] = item["time_mask"]
        if has_tracking:
            sequence_outputs["tracking_logits"][index, :length] = item["tracking_logits"]
            sequence_targets["tracking_target"][index, :length] = item["tracking_target"]
    return sequence_outputs, sequence_targets, sequence_metadata


def audit_sequence_identities(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit logical identity uniqueness and coverage for prepared windows."""

    groups: defaultdict[str, set[str]] = defaultdict(set)
    source_counts: defaultdict[str, int] = defaultdict(int)
    position_counts: defaultdict[str, int] = defaultdict(int)
    for item in metadata:
        identity = sequence_identity_for_metadata(item)
        groups[identity].add(identity_source_signature(item))
        source_counts[str(item.get("source_dataset", "unknown"))] += 1
        position_counts[str(item.get("position", "unknown"))] += 1
    collision_groups = {
        identity: len(signatures)
        for identity, signatures in groups.items()
        if len(signatures) > 1
    }
    return {
        "identity_version": SEQUENCE_IDENTITY_VERSION,
        "window_count": len(metadata),
        "logical_sequence_count": len(groups),
        "identity_collision_count": len(collision_groups),
        "identity_collision_groups": collision_groups,
        "source_window_counts": dict(sorted(source_counts.items())),
        "position_window_counts": dict(sorted(position_counts.items())),
    }
