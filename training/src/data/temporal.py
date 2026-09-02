"""Temporal normalization for canonical pose sequences."""

from __future__ import annotations

import numpy as np

from .schema import CanonicalSequence


def _resample_joint(
    values: np.ndarray,
    valid: np.ndarray,
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    max_interpolation_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample one joint coordinate pair while retaining long missing gaps."""

    output = np.full((len(target_positions), 2), np.nan, dtype=np.float32)
    output_valid = np.zeros(len(target_positions), dtype=bool)
    source_indices = np.flatnonzero(valid)
    if source_indices.size == 0:
        return output, output_valid

    valid_positions = source_positions[source_indices]
    valid_values = values[source_indices].astype(np.float32)
    tolerance = 1e-5
    right = np.searchsorted(valid_positions, target_positions, side="left")
    exact = (right < len(valid_positions)) & (
        np.abs(valid_positions[np.minimum(right, len(valid_positions) - 1)] - target_positions)
        <= tolerance
    )
    interior = (right > 0) & (right < len(valid_positions)) & ~exact
    safe_right = np.minimum(right, len(valid_positions) - 1)
    safe_left = np.maximum(right - 1, 0)
    span = valid_positions[safe_right] - valid_positions[safe_left]
    interpolate = interior & (span <= float(max_interpolation_gap + 1) + tolerance)

    if exact.any():
        output[exact] = valid_values[safe_right[exact]]
        output_valid[exact] = True
    if interpolate.any():
        ratio = (target_positions[interpolate] - valid_positions[safe_left[interpolate]]) / np.maximum(
            span[interpolate], tolerance
        )
        output[interpolate] = valid_values[safe_left[interpolate]] + ratio[:, None] * (
            valid_values[safe_right[interpolate]] - valid_values[safe_left[interpolate]]
        )
        output_valid[interpolate] = True
    return output, output_valid


def _resample_labels(
    labels: np.ndarray,
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    frame_valid: np.ndarray,
) -> np.ndarray:
    nearest = np.searchsorted(source_positions, target_positions, side="left")
    nearest = np.clip(nearest, 0, len(source_positions) - 1)
    previous = np.maximum(nearest - 1, 0)
    choose_previous = np.abs(target_positions - source_positions[previous]) < np.abs(
        source_positions[nearest] - target_positions
    )
    nearest[choose_previous] = previous[choose_previous]
    result = np.asarray(labels[nearest]).copy()
    result[~frame_valid] = -1
    return result


def resample_sequence(
    sequence: CanonicalSequence,
    target_fps: float = 30.0,
    max_interpolation_gap: int = 5,
) -> CanonicalSequence:
    """Resample one sequence and interpolate only short landmark gaps.

    Coordinates are interpolated only when both neighboring observations are
    present and their source-frame span contains at most
    ``max_interpolation_gap`` missing frames. Longer gaps remain unobserved.
    Capability absence is never filled because an absent limb has no valid
    source observations to interpolate.
    """

    target_fps = float(target_fps)
    max_interpolation_gap = int(max_interpolation_gap)
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target_fps must be finite and positive")
    if max_interpolation_gap < 0:
        raise ValueError("max_interpolation_gap must be non-negative")

    source_timestamps = sequence.timestamps.astype(np.float64)
    start = float(source_timestamps[0])
    source_positions = (source_timestamps - start) * target_fps
    duration = max(0.0, float(source_positions[-1]))
    target_count = max(1, int(np.floor(duration + 1e-6)) + 1)
    target_positions = np.arange(target_count, dtype=np.float64)
    if duration > target_positions[-1] + 1e-5:
        # Keep the final source observation instead of silently truncating a
        # fractional frame at the end of an irregularly timed clip.
        target_positions = np.append(target_positions, duration)
    target_timestamps = (start + target_positions / target_fps).astype(np.float32)
    # Float32 timestamps can round a numerically tiny endpoint onto the prior
    # frame (common when a source timestamp was produced by float32 division).
    # Remove only duplicate representable timestamps; meaningful fractional
    # endpoints remain in the resampled sequence.
    strictly_increasing = np.r_[True, np.diff(target_timestamps) > 0.0]
    target_positions = target_positions[strictly_increasing]
    target_timestamps = target_timestamps[strictly_increasing]
    target_count = len(target_positions)

    joints = np.full((target_count, sequence.joints.shape[1], 2), np.nan, dtype=np.float32)
    confidence = np.zeros((target_count, sequence.joints.shape[1]), dtype=np.float32)
    observed = np.zeros((target_count, sequence.joints.shape[1]), dtype=bool)
    finite_joints = np.isfinite(sequence.joints).all(axis=-1)
    source_valid = sequence.observed_mask & finite_joints
    for joint_index in range(sequence.joints.shape[1]):
        joint_values, joint_valid = _resample_joint(
            sequence.joints[:, joint_index],
            source_valid[:, joint_index],
            source_positions,
            target_positions,
            max_interpolation_gap,
        )
        joints[:, joint_index] = joint_values
        observed[:, joint_index] = joint_valid
        confidence_values, confidence_valid = _resample_joint(
            np.repeat(sequence.pose_confidence[:, joint_index, None], 2, axis=1),
            source_valid[:, joint_index],
            source_positions,
            target_positions,
            max_interpolation_gap,
        )
        confidence[:, joint_index] = np.clip(
            np.where(confidence_valid, confidence_values[:, 0], 0.0), 0.0, 1.0
        )

    frame_valid = observed.any(axis=1)
    phase = _resample_labels(sequence.phase, source_positions, target_positions, frame_valid)
    rep_boundary = _resample_labels(
        sequence.rep_boundary,
        source_positions,
        target_positions,
        frame_valid,
    ).astype(np.float32)
    metadata = dict(sequence.metadata)
    metadata["resampled_to_fps"] = target_fps
    metadata["max_interpolation_gap"] = max_interpolation_gap
    return sequence.clone(
        joints=joints,
        pose_confidence=confidence,
        observed_mask=observed,
        timestamps=target_timestamps,
        phase=phase,
        rep_boundary=rep_boundary,
        metadata=metadata,
    )
