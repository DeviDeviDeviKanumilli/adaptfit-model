"""Deterministic robustness augmentations for pose sequences."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import numpy as np

from .schema import CANONICAL_INDEX, FAMILY_NAMES, LIMB_NAMES, CanonicalSequence
from ..features.anatomy import LIMB_JOINTS


def _joint_indices(limb: str) -> list[int]:
    return [CANONICAL_INDEX[name] for name in LIMB_JOINTS[limb]]


def _infer_active_side(sequence: CanonicalSequence, limb_group: str) -> str | None:
    text = " ".join(
        str(value).lower()
        for value in (
            sequence.metadata.get("exercise_subtype", ""),
            sequence.metadata.get("position_label", ""),
            sequence.metadata.get("side", ""),
        )
    )
    if "left" in text:
        return "left_" + limb_group
    if "right" in text:
        return "right_" + limb_group
    return None


def _apply_noise(sequence: CanonicalSequence, rng: np.random.Generator, noise_std: float) -> None:
    valid = sequence.observed_mask & np.isfinite(sequence.joints).all(axis=-1)
    if not valid.any():
        return
    points = sequence.joints[valid]
    scale = float(np.nanmedian(np.linalg.norm(points - np.nanmedian(points, axis=0), axis=-1)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    noise = rng.normal(0.0, noise_std * scale, sequence.joints.shape).astype(np.float32)
    sequence.joints[valid] += noise[valid]


def _apply_occlusion(sequence: CanonicalSequence, limb: str) -> None:
    indices = _joint_indices(limb)
    sequence.joints[:, indices] = np.nan
    sequence.pose_confidence[:, indices] = 0.0
    sequence.observed_mask[:, indices] = False
    sequence.quality_mask[:] = False
    sequence.metadata["synthetic_occlusion"] = limb


def _apply_capability_absence(sequence: CanonicalSequence, limb: str) -> None:
    sequence.capability_states[limb] = "absent"
    sequence.metadata["synthetic_absent_limb"] = limb


def _apply_reduced_rom(sequence: CanonicalSequence, scale: float) -> None:
    valid = sequence.observed_mask & np.isfinite(sequence.joints).all(axis=-1)
    if not valid.any():
        return
    anchor = np.nanmedian(sequence.joints, axis=(0, 1))
    if not np.isfinite(anchor).all():
        anchor = np.zeros(2, dtype=np.float32)
    sequence.joints[valid] = anchor + (sequence.joints[valid] - anchor) * float(scale)
    sequence.quality[0] = 0.0
    sequence.quality_mask[0] = True
    sequence.metadata["synthetic_reduced_rom_scale"] = float(scale)


def _resample_array(values: np.ndarray, source_positions: np.ndarray, valid_axis: int = 0) -> np.ndarray:
    values = np.asarray(values)
    target_shape = (len(source_positions),) + values.shape[1:]
    output = np.full(target_shape, np.nan, dtype=np.float32)
    for index in np.ndindex(values.shape[1:]):
        series = values[(slice(None),) + index].astype(np.float32)
        finite = np.isfinite(series)
        if finite.sum() == 0:
            continue
        if finite.sum() == 1:
            output[(slice(None),) + index] = series[finite][0]
        else:
            output[(slice(None),) + index] = np.interp(
                source_positions,
                np.flatnonzero(finite).astype(np.float32),
                series[finite],
            )
    return output


def _apply_tempo(sequence: CanonicalSequence, factor: float) -> CanonicalSequence:
    factor = max(0.5, min(1.5, float(factor)))
    new_frames = max(4, int(round(sequence.frames / factor)))
    positions = np.linspace(0.0, max(0, sequence.frames - 1), new_frames, dtype=np.float32)
    joints = _resample_array(sequence.joints, positions)
    confidence = _resample_array(sequence.pose_confidence, positions)
    observed = confidence > 0.5
    phase_source = np.rint(positions).astype(np.int64)
    phase_source = np.clip(phase_source, 0, sequence.frames - 1)
    phase = sequence.phase[phase_source]
    boundary = np.zeros((new_frames, 2), dtype=np.float32)
    boundary[0, 0] = 1.0
    boundary[-1, 1] = 1.0
    timestamps = np.arange(new_frames, dtype=np.float32) / 30.0
    metadata = dict(sequence.metadata)
    metadata["synthetic_tempo_factor"] = factor
    return sequence.clone(
        joints=joints,
        pose_confidence=np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0),
        observed_mask=observed,
        timestamps=timestamps,
        phase=phase,
        rep_boundary=boundary,
        metadata=metadata,
    )


def _choose_profile_limb(sequence: CanonicalSequence, rng: np.random.Generator) -> str:
    family_name = FAMILY_NAMES[sequence.family] if 0 <= sequence.family < len(FAMILY_NAMES) else "other"
    if family_name in {"arm_flexion", "row_pull", "reach"}:
        active = _infer_active_side(sequence, "arm")
        candidates = [limb for limb in ("left_arm", "right_arm") if limb != active]
        return candidates[0] if active else str(rng.choice(candidates))
    if family_name in {"knee_extension", "hip_flexion"}:
        active = _infer_active_side(sequence, "leg")
        candidates = [limb for limb in ("left_leg", "right_leg") if limb != active]
        return candidates[0] if active else str(rng.choice(candidates))
    return str(rng.choice(LIMB_NAMES))


def generate_augmentations(
    sequence: CanonicalSequence,
    rng: np.random.Generator,
    copies: int,
    occlusion_probability: float = 0.2,
    noise_std: float = 0.008,
    reduced_rom_scale: float = 0.75,
) -> list[CanonicalSequence]:
    """Generate labeled robustness variants without changing the source split."""

    augmented: list[CanonicalSequence] = []
    for index in range(max(0, int(copies))):
        variant = sequence.clone()
        variant.metadata["synthetic"] = True
        variant.metadata["synthetic_variant_index"] = index
        variant.metadata["synthetic_parent_session"] = sequence.session_id
        variant.metadata["label_provenance"] = "synthetic"
        _apply_noise(variant, rng, noise_std)

        profile_limb = _choose_profile_limb(sequence, rng)
        if index % 3 == 0:
            if rng.random() < occlusion_probability:
                _apply_occlusion(variant, profile_limb)
            _apply_capability_absence(variant, profile_limb)
        elif index % 3 == 1:
            _apply_reduced_rom(variant, reduced_rom_scale)
            _apply_capability_absence(variant, profile_limb)
            variant = _apply_tempo(variant, float(rng.choice((0.75, 1.25))))
        else:
            _apply_occlusion(variant, profile_limb)
        # A transformed sequence is no longer guaranteed to preserve the
        # original therapist score. Keep the source expert score only on the
        # untouched repetition-level UCO sequence.
        variant.expert_quality = -1
        variant.expert_quality_mask = False
        variant.metadata["expert_quality_label_status"] = "synthetic_unlabeled"
        augmented.append(variant)
    return augmented
