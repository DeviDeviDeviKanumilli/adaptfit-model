"""Anatomy-aware, fixed-width feature extraction from canonical 2D joints."""

from __future__ import annotations

import numpy as np

from ..data.schema import (
    CANONICAL_INDEX,
    CAPABILITY_STATES,
    CanonicalSequence,
    LIMB_NAMES,
    POSITION_NAMES,
)


CAPABILITY_WEIGHTS = {
    "available": 1.0,
    "limited": 0.65,
    "absent": 0.0,
    "assisted": 0.80,
    "unknown": 0.50,
}


ANGLE_DEFINITIONS = (
    ("left_elbow_flexion", "left_shoulder", "left_elbow", "left_wrist"),
    ("right_elbow_flexion", "right_shoulder", "right_elbow", "right_wrist"),
    ("left_shoulder_flexion", "left_hip", "left_shoulder", "left_elbow"),
    ("right_shoulder_flexion", "right_hip", "right_shoulder", "right_elbow"),
    ("left_shoulder_abduction", "right_shoulder", "left_shoulder", "left_elbow"),
    ("right_shoulder_abduction", "left_shoulder", "right_shoulder", "right_elbow"),
    ("left_hip_flexion", "left_shoulder", "left_hip", "left_knee"),
    ("right_hip_flexion", "right_shoulder", "right_hip", "right_knee"),
    ("left_knee_flexion", "left_hip", "left_knee", "left_ankle"),
    ("right_knee_flexion", "right_hip", "right_knee", "right_ankle"),
    ("left_ankle_angle", "left_knee", "left_ankle", "left_foot_index"),
    ("right_ankle_angle", "right_knee", "right_ankle", "right_foot_index"),
)

ANGLE_COUNT = len(ANGLE_DEFINITIONS) + 2
BASE_CONTINUOUS_FEATURE_DIM = 66 + 66 + ANGLE_COUNT + ANGLE_COUNT + 33
PROFILE_CONTEXT_DIM = len(LIMB_NAMES) * len(CAPABILITY_STATES)
POSITION_CONTEXT_DIM = len(POSITION_NAMES)
BASE_FEATURE_DIM = BASE_CONTINUOUS_FEATURE_DIM + 33 + 33 + PROFILE_CONTEXT_DIM + POSITION_CONTEXT_DIM

# Optional causal motion summaries. They are feature inputs, not labels.
DIAGNOSTIC_CONTINUOUS_FEATURE_DIM = 66 + ANGLE_COUNT + ANGLE_COUNT + 1
CONTINUOUS_FEATURE_DIM = BASE_CONTINUOUS_FEATURE_DIM
FEATURE_DIM = BASE_FEATURE_DIM
DIAGNOSTIC_FEATURE_DIM = DIAGNOSTIC_CONTINUOUS_FEATURE_DIM


LIMB_JOINTS: dict[str, tuple[str, ...]] = {
    "left_arm": (
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "left_pinky",
        "left_index",
        "left_thumb",
    ),
    "right_arm": (
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_pinky",
        "right_index",
        "right_thumb",
    ),
    "left_leg": (
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_heel",
        "left_foot_index",
    ),
    "right_leg": (
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_heel",
        "right_foot_index",
    ),
}


def _safe_midpoint(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros((len(points), 2), dtype=np.float32)
    result_valid = valid.all(axis=1)
    if result_valid.any():
        result[result_valid] = points[result_valid].mean(axis=1)
    return result, result_valid


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = a - b
    second = c - b
    first_norm = np.linalg.norm(first, axis=-1)
    second_norm = np.linalg.norm(second, axis=-1)
    safe = (first_norm > 1e-6) & (second_norm > 1e-6) & valid.all(axis=1)
    cosine = np.zeros(len(a), dtype=np.float32)
    if safe.any():
        cosine[safe] = np.sum(first[safe] * second[safe], axis=-1) / (
            first_norm[safe] * second_norm[safe]
        )
    values = np.zeros(len(a), dtype=np.float32)
    values[safe] = np.arccos(np.clip(cosine[safe], -1.0, 1.0)) / np.pi
    return values, safe


def _capability_mask(sequence: CanonicalSequence) -> np.ndarray:
    mask = np.ones(33, dtype=np.float32)
    for limb, joints in LIMB_JOINTS.items():
        state = sequence.capability_states.get(limb, "unknown")
        weight = CAPABILITY_WEIGHTS.get(state, CAPABILITY_WEIGHTS["unknown"])
        if state == "absent":
            weight = 0.0
        for joint in joints:
            mask[CANONICAL_INDEX[joint]] = weight
    return mask


def tracking_confidence_target(sequence: CanonicalSequence) -> np.ndarray:
    """Return a capability-aware, self-supervised observability target.

    This is a camera/landmark signal, not a movement-quality or disability
    label. Expected-but-absent limbs are excluded using the onboarding profile
    so a known missing limb does not lower tracking confidence by itself.
    """

    capability = _capability_mask(sequence)
    expected = capability > 0.0
    observed = sequence.observed_mask & np.isfinite(sequence.joints).all(axis=-1)
    confidence = np.clip(sequence.pose_confidence, 0.0, 1.0)
    valid_count = int(expected.sum())
    if valid_count == 0:
        return np.zeros(sequence.frames, dtype=np.float32)
    target = (observed & (confidence > 0.0)) * confidence
    return (target[:, expected].mean(axis=1)).astype(np.float32)


def _profile_context(sequence: CanonicalSequence) -> np.ndarray:
    context = np.zeros(len(LIMB_NAMES) * len(CAPABILITY_STATES), dtype=np.float32)
    for limb_index, limb in enumerate(LIMB_NAMES):
        state = sequence.capability_states.get(limb, "unknown")
        state_index = (
            CAPABILITY_STATES.index(state)
            if state in CAPABILITY_STATES
            else CAPABILITY_STATES.index("unknown")
        )
        context[limb_index * len(CAPABILITY_STATES) + state_index] = 1.0
    return context


def _position_context(sequence: CanonicalSequence) -> np.ndarray:
    context = np.zeros(len(POSITION_NAMES), dtype=np.float32)
    index = (
        POSITION_NAMES.index(sequence.position)
        if sequence.position in POSITION_NAMES
        else POSITION_NAMES.index("unknown")
    )
    context[index] = 1.0
    return context


def _velocity(values: np.ndarray, valid: np.ndarray, fps: float) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    if len(values) < 2:
        return output
    output[1:] = (values[1:] - values[:-1]) * float(fps)
    pair_valid = valid[1:] & valid[:-1]
    output[1:] *= pair_valid[..., None] if output.ndim == 3 else pair_valid
    return output


def _rolling_range(values: np.ndarray, valid: np.ndarray, frames: int = 15) -> np.ndarray:
    """Compute a causal rolling range without treating missing angles as data."""

    width = max(1, int(frames))
    if len(values) == 0:
        return np.zeros_like(values, dtype=np.float32)
    padded_values = np.concatenate(
        [np.zeros((width - 1, values.shape[1]), dtype=np.float32), values], axis=0
    )
    padded_valid = np.concatenate(
        [np.zeros((width - 1, valid.shape[1]), dtype=bool), valid.astype(bool)], axis=0
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded_values,
        window_shape=width,
        axis=0,
    )
    valid_windows = np.lib.stride_tricks.sliding_window_view(
        padded_valid,
        window_shape=width,
        axis=0,
    )
    safe_max = np.where(valid_windows, windows, -np.inf).max(axis=-1)
    safe_min = np.where(valid_windows, windows, np.inf).min(axis=-1)
    has_value = valid_windows.any(axis=-1)
    return np.where(has_value, safe_max - safe_min, 0.0).astype(np.float32)


def feature_dimensions(include_diagnostics: bool = False) -> tuple[int, int]:
    """Return ``(feature_dimension, continuous_dimension)`` for a schema variant."""

    if include_diagnostics:
        return (
            BASE_FEATURE_DIM + DIAGNOSTIC_FEATURE_DIM,
            BASE_CONTINUOUS_FEATURE_DIM + DIAGNOSTIC_CONTINUOUS_FEATURE_DIM,
        )
    return FEATURE_DIM, CONTINUOUS_FEATURE_DIM


def continuous_feature_indices(include_diagnostics: bool = False) -> np.ndarray:
    """Return the feature columns that should receive fitted normalization."""

    indices = list(range(BASE_CONTINUOUS_FEATURE_DIM))
    if include_diagnostics:
        indices.extend(range(BASE_FEATURE_DIM, BASE_FEATURE_DIM + DIAGNOSTIC_FEATURE_DIM))
    return np.asarray(indices, dtype=np.int64)


def _custom_angles(normalized: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_shoulder = normalized[:, CANONICAL_INDEX["left_shoulder"]]
    right_shoulder = normalized[:, CANONICAL_INDEX["right_shoulder"]]
    left_hip = normalized[:, CANONICAL_INDEX["left_hip"]]
    right_hip = normalized[:, CANONICAL_INDEX["right_hip"]]
    shoulders, shoulders_valid = _safe_midpoint(
        np.stack([left_shoulder, right_shoulder], axis=1),
        np.stack([valid[:, CANONICAL_INDEX["left_shoulder"]], valid[:, CANONICAL_INDEX["right_shoulder"]]], axis=1),
    )
    hips, hips_valid = _safe_midpoint(
        np.stack([left_hip, right_hip], axis=1),
        np.stack([valid[:, CANONICAL_INDEX["left_hip"]], valid[:, CANONICAL_INDEX["right_hip"]]], axis=1),
    )
    trunk = shoulders - hips
    trunk_norm = np.linalg.norm(trunk, axis=-1)
    trunk_valid = shoulders_valid & hips_valid & (trunk_norm > 1e-6)
    trunk_lean = np.zeros(len(normalized), dtype=np.float32)
    trunk_lean[trunk_valid] = np.arctan2(trunk[trunk_valid, 0], trunk[trunk_valid, 1]) / np.pi

    pelvis = right_hip - left_hip
    pelvis_norm = np.linalg.norm(pelvis, axis=-1)
    pelvis_valid = valid[:, CANONICAL_INDEX["left_hip"]] & valid[:, CANONICAL_INDEX["right_hip"]] & (pelvis_norm > 1e-6)
    pelvis_tilt = np.zeros(len(normalized), dtype=np.float32)
    pelvis_tilt[pelvis_valid] = np.arctan2(pelvis[pelvis_valid, 1], pelvis[pelvis_valid, 0]) / np.pi
    return np.stack([trunk_lean, pelvis_tilt], axis=-1), np.stack([trunk_valid, pelvis_valid], axis=-1)


def extract_features(
    sequence: CanonicalSequence,
    fps: float = 30.0,
    include_diagnostics: bool = False,
) -> np.ndarray:
    """Return anatomy-normalized features for one sequence.

    The default v1 schema is 283 values per frame. The optional diagnostic
    variant appends causal acceleration, angular acceleration, rolling ROM,
    and smoothness summaries while preserving the base feature ordering.
    """

    joints = sequence.joints.astype(np.float32, copy=True)
    observed = sequence.observed_mask & np.isfinite(joints).all(axis=-1)
    capability = _capability_mask(sequence)
    valid = observed & (capability[None, :] > 0.0)

    left_hip = joints[:, CANONICAL_INDEX["left_hip"]]
    right_hip = joints[:, CANONICAL_INDEX["right_hip"]]
    left_shoulder = joints[:, CANONICAL_INDEX["left_shoulder"]]
    right_shoulder = joints[:, CANONICAL_INDEX["right_shoulder"]]
    hip_mid, hip_valid = _safe_midpoint(
        np.stack([left_hip, right_hip], axis=1),
        np.stack([valid[:, CANONICAL_INDEX["left_hip"]], valid[:, CANONICAL_INDEX["right_hip"]]], axis=1),
    )
    shoulder_mid, shoulder_valid = _safe_midpoint(
        np.stack([left_shoulder, right_shoulder], axis=1),
        np.stack([valid[:, CANONICAL_INDEX["left_shoulder"]], valid[:, CANONICAL_INDEX["right_shoulder"]]], axis=1),
    )
    anchor = hip_mid.copy()
    anchor_valid = hip_valid.copy()
    use_shoulders = ~anchor_valid & shoulder_valid
    anchor[use_shoulders] = shoulder_mid[use_shoulders]
    anchor_valid[use_shoulders] = True
    if not anchor_valid.all():
        for frame in np.flatnonzero(~anchor_valid):
            frame_valid = valid[frame]
            if frame_valid.any():
                anchor[frame] = np.nanmean(joints[frame, frame_valid], axis=0)
                anchor_valid[frame] = True

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1)
    hip_width = np.linalg.norm(left_hip - right_hip, axis=-1)
    torso_length = np.linalg.norm(shoulder_mid - hip_mid, axis=-1)
    scale = np.where(
        hip_valid & (hip_width > 1e-3),
        hip_width,
        np.where(shoulder_valid & (shoulder_width > 1e-3), shoulder_width, torso_length),
    )
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0).astype(np.float32)

    normalized = (joints - anchor[:, None, :]) / scale[:, None, None]
    normalized[~valid] = 0.0
    normalized[~np.isfinite(normalized)] = 0.0

    norm_velocity = _velocity(normalized, valid, fps)
    angles = np.zeros((len(joints), ANGLE_COUNT), dtype=np.float32)
    angle_valid = np.zeros((len(joints), ANGLE_COUNT), dtype=bool)
    for angle_index, (_, first, middle, last) in enumerate(ANGLE_DEFINITIONS):
        values, values_valid = _angle(
            normalized[:, CANONICAL_INDEX[first]],
            normalized[:, CANONICAL_INDEX[middle]],
            normalized[:, CANONICAL_INDEX[last]],
            np.stack(
                [
                    valid[:, CANONICAL_INDEX[first]],
                    valid[:, CANONICAL_INDEX[middle]],
                    valid[:, CANONICAL_INDEX[last]],
                ],
                axis=1,
            ),
        )
        angles[:, angle_index] = values
        angle_valid[:, angle_index] = values_valid
    custom, custom_valid = _custom_angles(normalized, valid)
    angles[:, -2:] = custom
    angle_valid[:, -2:] = custom_valid
    angle_velocity = _velocity(angles, angle_valid, fps)

    confidence = np.clip(sequence.pose_confidence, 0.0, 1.0).astype(np.float32)
    confidence *= capability[None, :]
    confidence[~np.isfinite(confidence)] = 0.0
    profile = np.broadcast_to(_profile_context(sequence), (len(joints), PROFILE_CONTEXT_DIM))
    position = np.broadcast_to(_position_context(sequence), (len(joints), POSITION_CONTEXT_DIM))

    pieces: tuple[np.ndarray, ...] = (
        normalized.reshape(len(joints), -1),
        norm_velocity.reshape(len(joints), -1),
        angles,
        angle_velocity,
        confidence,
        valid.astype(np.float32),
        np.broadcast_to(capability.astype(np.float32), (len(joints), 33)),
        profile,
        position,
    )
    if include_diagnostics:
        acceleration_xy = _velocity(norm_velocity, valid, fps)
        angular_acceleration = _velocity(angle_velocity, angle_valid, fps)
        rolling_rom = _rolling_range(angles, angle_valid)
        smoothness = 1.0 / (1.0 + np.mean(np.abs(angular_acceleration), axis=1))
        pieces += (
            acceleration_xy.reshape(len(joints), -1),
            angular_acceleration,
            rolling_rom,
            smoothness[:, None].astype(np.float32),
        )
    features = np.concatenate(pieces, axis=1).astype(np.float32)
    expected_feature_dim, _ = feature_dimensions(include_diagnostics)
    if features.shape[1] != expected_feature_dim:
        raise AssertionError(f"Feature schema expected {expected_feature_dim}, got {features.shape[1]}")
    return features


def feature_schema(include_diagnostics: bool = False, fps: float = 30.0) -> dict:
    """Return a serializable schema that is saved beside every checkpoint."""

    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    feature_dim, continuous_dim = feature_dimensions(include_diagnostics)
    offset = 0
    groups: list[dict[str, int | str]] = []
    group_sizes: list[tuple[str, int]] = [
        ("normalized_xy", 66),
        ("velocity_xy", 66),
        ("angles", ANGLE_COUNT),
        ("angle_velocity", ANGLE_COUNT),
        ("pose_confidence", 33),
        ("observed_mask", 33),
        ("capability_mask", 33),
        ("profile_context", PROFILE_CONTEXT_DIM),
        ("position_context", POSITION_CONTEXT_DIM),
    ]
    if include_diagnostics:
        group_sizes.extend(
            [
                ("acceleration_xy", 66),
                ("angular_acceleration", ANGLE_COUNT),
                ("rolling_rom", ANGLE_COUNT),
                ("smoothness", 1),
            ]
        )
    for name, size in group_sizes:
        groups.append({"name": name, "start": offset, "end": offset + size})
        offset += size
    return {
        "version": "adaptfit.features.v2.diagnostics" if include_diagnostics else "adaptfit.features.v1",
        "dimension": feature_dim,
        "continuous_dimension": continuous_dim,
        "nominal_fps": fps,
        "continuous_indices": continuous_feature_indices(include_diagnostics).tolist(),
        "joint_names": list(CANONICAL_INDEX),
        "angle_names": [item[0] for item in ANGLE_DEFINITIONS] + ["trunk_lean", "pelvis_tilt"],
        "capability_states": list(CAPABILITY_STATES),
        "capability_weights": CAPABILITY_WEIGHTS,
        "limbs": list(LIMB_NAMES),
        "positions": list(POSITION_NAMES),
        "groups": groups,
    }
