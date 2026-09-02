"""Observable motion diagnostics kept separate from weak training labels."""

from __future__ import annotations

import numpy as np

from ..data.schema import CanonicalSequence
from .anatomy import extract_features, feature_schema


def _group(features: np.ndarray, name: str) -> np.ndarray:
    group = next(item for item in feature_schema()["groups"] if item["name"] == name)
    return features[:, group["start"] : group["end"]]


def _differentiate(values: np.ndarray, fps: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        result[1:] = np.diff(values, axis=0) * float(fps)
    return result


def _rolling_range(values: np.ndarray, frames: int) -> np.ndarray:
    width = max(1, int(frames))
    if len(values) == 0:
        return np.zeros_like(values, dtype=np.float32)
    padded = np.concatenate(
        [np.repeat(values[:1], width - 1, axis=0), values], axis=0
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=width,
        axis=0,
    )
    return (windows.max(axis=-1) - windows.min(axis=-1)).astype(np.float32)


def extract_observable_diagnostics(
    sequence: CanonicalSequence,
    fps: float = 30.0,
    rolling_frames: int = 15,
) -> dict[str, np.ndarray]:
    """Return non-clinical motion summaries for analysis or future inference.

    These summaries are intentionally not used to manufacture missing labels.
    A low tracking-confidence or observed-fraction value is an abstention cue,
    not evidence of poor movement quality.
    """

    features = extract_features(sequence, fps=fps)
    velocity_xy = _group(features, "velocity_xy")
    angles = _group(features, "angles")
    angle_velocity = _group(features, "angle_velocity")
    confidence = _group(features, "pose_confidence")
    observed = _group(features, "observed_mask")
    angular_acceleration = _differentiate(angle_velocity, fps)
    smoothness = 1.0 / (1.0 + np.mean(np.abs(angular_acceleration), axis=1))
    diagnostics = {
        "acceleration_xy": _differentiate(velocity_xy, fps),
        "angular_acceleration": angular_acceleration,
        "rolling_rom": _rolling_range(angles, rolling_frames),
        "smoothness": smoothness.astype(np.float32),
        "trunk_lean": np.abs(angles[:, -2]).astype(np.float32),
        "tracking_confidence": np.mean(confidence, axis=1).astype(np.float32),
        "observed_fraction": np.mean(observed, axis=1).astype(np.float32),
    }
    for name, values in diagnostics.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Observable diagnostic {name!r} contains non-finite values")
    return diagnostics
