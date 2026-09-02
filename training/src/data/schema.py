"""Canonical sequence and label schema used by every training source."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_JOINT_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
CANONICAL_INDEX = {name: index for index, name in enumerate(CANONICAL_JOINT_NAMES)}
NUM_JOINTS = len(CANONICAL_JOINT_NAMES)

FAMILY_NAMES = (
    "other",
    "arm_flexion",
    "row_pull",
    "knee_extension",
    "hip_flexion",
    "reach",
)
FAMILY_TO_ID = {name: index for index, name in enumerate(FAMILY_NAMES)}

PHASE_NAMES = (
    "unknown",
    "rest",
    "concentric",
    "hold",
    "eccentric",
)
PHASE_TO_ID = {name: index for index, name in enumerate(PHASE_NAMES)}

QUALITY_NAMES = (
    "rom_ok",
    "tempo_ok",
    "smoothness_ok",
    "trunk_compensation",
)

POSITION_NAMES = (
    "seated",
    "wheelchair",
    "standing",
    "unknown",
)
POSITION_TO_ID = {name: index for index, name in enumerate(POSITION_NAMES)}

LIMB_NAMES = ("left_arm", "right_arm", "left_leg", "right_leg")
CAPABILITY_STATES = ("available", "limited", "absent", "assisted", "unknown")

LABEL_PROVENANCE_WEIGHTS = {
    "strong": 1.0,
    "weak": 0.60,
    "procedural": 0.25,
    "synthetic": 0.40,
    "unknown": 0.0,
}
PHASE_LABEL_SOURCE_WEIGHTS = {
    "strong": 1.0,
    "weak_displacement": 0.35,
    "procedural_template": 0.25,
    "unknown": 0.0,
}
BOUNDARY_LABEL_SOURCE_WEIGHTS = {
    "strong": 1.0,
    "segmentation": 1.0,
    "single_clip_assumption": 0.15,
    "procedural_template": 0.25,
    "unknown": 0.0,
}
QUALITY_LABEL_SOURCE_WEIGHTS = {
    "strong": 1.0,
    "explicit": 1.0,
    "procedural_template": 0.25,
    "unlabeled": 0.0,
    "unknown": 0.0,
}


def _default_capability_states() -> dict[str, str]:
    return {limb: "available" for limb in LIMB_NAMES}


def _default_quality() -> tuple[np.ndarray, np.ndarray]:
    return np.full(4, np.nan, dtype=np.float32), np.zeros(4, dtype=bool)


@dataclass
class CanonicalSequence:
    """One variable-length movement sequence before fixed-window conversion."""

    joints: np.ndarray
    pose_confidence: np.ndarray
    observed_mask: np.ndarray
    timestamps: np.ndarray
    position: str = "unknown"
    participant_id: str = "unknown"
    session_id: str = "unknown"
    source_dataset: str = "unknown"
    # -1 means unavailable; explicit Other can still be supplied as class 0.
    family: int = -1
    phase: np.ndarray | None = None
    rep_boundary: np.ndarray | None = None
    quality: np.ndarray = field(default_factory=lambda: _default_quality()[0])
    quality_mask: np.ndarray = field(default_factory=lambda: _default_quality()[1])
    # Optional composite execution-quality target.  The internal class range
    # is 0..4 and corresponds to the source's ordinal score range 1..5.
    # ``-1`` means that this sequence has no expert-quality target.
    expert_quality: int = -1
    expert_quality_mask: bool = False
    capability_states: dict[str, str] = field(default_factory=_default_capability_states)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joints = np.asarray(self.joints, dtype=np.float32)
        self.pose_confidence = np.asarray(self.pose_confidence, dtype=np.float32)
        self.observed_mask = np.asarray(self.observed_mask, dtype=bool)
        self.timestamps = np.asarray(self.timestamps, dtype=np.float32)
        self.quality = np.asarray(self.quality, dtype=np.float32)
        self.quality_mask = np.asarray(self.quality_mask, dtype=bool)
        try:
            self.expert_quality = int(self.expert_quality)
        except (TypeError, ValueError) as error:
            raise ValueError("expert_quality must be -1 or an integer in the 0..4 range") from error
        self.expert_quality_mask = bool(self.expert_quality_mask)

        if self.phase is None:
            # Unknown is an explicit class; missing phase supervision is -1.
            self.phase = np.full(len(self.joints), -1, dtype=np.int64)
        else:
            self.phase = np.asarray(self.phase, dtype=np.int64)

        if self.rep_boundary is None:
            self.rep_boundary = np.full((len(self.joints), 2), -1.0, dtype=np.float32)
        else:
            self.rep_boundary = np.asarray(self.rep_boundary, dtype=np.float32)

        self.validate()

    @property
    def frames(self) -> int:
        return int(self.joints.shape[0])

    def validate(self) -> None:
        if self.joints.ndim != 3 or self.joints.shape[1:] != (NUM_JOINTS, 2):
            raise ValueError(f"joints must have shape [T, {NUM_JOINTS}, 2], got {self.joints.shape}")
        if self.frames == 0:
            raise ValueError("canonical sequences must contain at least one frame")
        expected_frame_shapes = {
            "pose_confidence": self.pose_confidence.shape,
            "observed_mask": self.observed_mask.shape,
            "timestamps": self.timestamps.shape,
            "phase": self.phase.shape,
        }
        for name, shape in expected_frame_shapes.items():
            if name in {"pose_confidence", "observed_mask"} and shape != (self.frames, NUM_JOINTS):
                raise ValueError(f"{name} must have shape [{self.frames}, {NUM_JOINTS}], got {shape}")
            if name in {"timestamps", "phase"} and shape != (self.frames,):
                raise ValueError(f"{name} must have shape [{self.frames}], got {shape}")
        if self.rep_boundary.shape != (self.frames, 2):
            raise ValueError(f"rep_boundary must have shape [{self.frames}, 2], got {self.rep_boundary.shape}")
        if self.quality.shape != (4,) or self.quality_mask.shape != (4,):
            raise ValueError("quality and quality_mask must have shape [4]")
        if not np.isfinite(self.timestamps).all():
            raise ValueError("timestamps must be finite")
        if self.frames > 1 and not np.all(np.diff(self.timestamps) > 0.0):
            raise ValueError("timestamps must be strictly increasing")
        finite_joints = np.isfinite(self.joints).all(axis=-1)
        if np.any(self.observed_mask & ~finite_joints):
            raise ValueError("observed joints must have finite x/y coordinates")
        if not np.isfinite(self.pose_confidence).all() or np.any(
            (self.pose_confidence < 0.0) | (self.pose_confidence > 1.0)
        ):
            raise ValueError("pose_confidence must be finite and within [0, 1]")
        finite_boundaries = np.isfinite(self.rep_boundary)
        valid_boundaries = np.isin(self.rep_boundary, (-1.0, 0.0, 1.0))
        if np.any(~finite_boundaries) or np.any(~valid_boundaries):
            raise ValueError("rep_boundary labels must be -1, 0, or 1")
        finite_quality = np.isfinite(self.quality)
        if np.any(self.quality_mask & ~finite_quality):
            raise ValueError("masked quality labels must be finite")
        if np.any(finite_quality & ((self.quality < 0.0) | (self.quality > 1.0))):
            raise ValueError("quality labels must be within [0, 1]")
        if self.expert_quality < -1 or self.expert_quality >= 5:
            raise ValueError("expert_quality must be -1 or within the 0..4 class range")
        if self.expert_quality_mask and self.expert_quality < 0:
            raise ValueError("expert_quality_mask cannot be true when expert_quality is unavailable")
        try:
            self.family = int(self.family)
        except (TypeError, ValueError) as error:
            raise ValueError("family must be -1 or a valid FAMILY_NAMES index") from error
        if np.any((self.phase < -1) | (self.phase >= len(PHASE_NAMES))):
            raise ValueError("phase labels must be -1 or valid PHASE_NAMES indices")
        if np.any((self.family < -1) | (self.family >= len(FAMILY_NAMES))):
            raise ValueError("family must be -1 or a valid FAMILY_NAMES index")
        if self.position not in POSITION_TO_ID:
            self.position = "unknown"
        self.capability_states = {
            limb: self.capability_states.get(limb, "unknown")
            if self.capability_states.get(limb, "unknown") in CAPABILITY_STATES
            else "unknown"
            for limb in LIMB_NAMES
        }

    def clone(self, **updates: Any) -> "CanonicalSequence":
        values: dict[str, Any] = {
            "joints": self.joints.copy(),
            "pose_confidence": self.pose_confidence.copy(),
            "observed_mask": self.observed_mask.copy(),
            "timestamps": self.timestamps.copy(),
            "position": self.position,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "source_dataset": self.source_dataset,
            "family": self.family,
            "phase": self.phase.copy(),
            "rep_boundary": self.rep_boundary.copy(),
            "quality": self.quality.copy(),
            "quality_mask": self.quality_mask.copy(),
            "expert_quality": self.expert_quality,
            "expert_quality_mask": self.expert_quality_mask,
            "capability_states": dict(self.capability_states),
            "metadata": dict(self.metadata),
        }
        values.update(updates)
        return CanonicalSequence(**values)

    def to_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "position": self.position,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "source_dataset": self.source_dataset,
            "family": self.family,
            "capability_states": self.capability_states,
            "metadata": self.metadata,
        }
        np.savez_compressed(
            path,
            joints=self.joints,
            pose_confidence=self.pose_confidence,
            observed_mask=self.observed_mask,
            timestamps=self.timestamps,
            phase=self.phase,
            rep_boundary=self.rep_boundary,
            quality=self.quality,
            quality_mask=self.quality_mask,
            expert_quality=np.asarray(self.expert_quality, dtype=np.int64),
            expert_quality_mask=np.asarray(self.expert_quality_mask, dtype=bool),
            meta_json=np.asarray(json.dumps(meta)),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "CanonicalSequence":
        with np.load(path, allow_pickle=False) as data:
            meta_raw = data.get("meta_json", np.asarray("{}"))
            meta = json.loads(str(meta_raw.item()))
            confidence = data.get("pose_confidence")
            if confidence is None:
                confidence = np.ones(data["joints"].shape[:2], dtype=np.float32)
            observed = data.get("observed_mask")
            if observed is None:
                observed = np.isfinite(data["joints"]).all(axis=-1)
            timestamps = data.get("timestamps")
            if timestamps is None:
                timestamps = np.arange(data["joints"].shape[0], dtype=np.float32) / 30.0
            return cls(
                joints=data["joints"],
                pose_confidence=confidence,
                observed_mask=observed,
                timestamps=timestamps,
                position=meta.get("position", "unknown"),
                participant_id=meta.get("participant_id", "unknown"),
                session_id=meta.get("session_id", "unknown"),
                source_dataset=meta.get("source_dataset", "unknown"),
                family=int(meta.get("family", -1)),
                phase=data.get("phase"),
                rep_boundary=data.get("rep_boundary"),
                quality=data.get("quality", _default_quality()[0]),
                quality_mask=data.get("quality_mask", _default_quality()[1]),
                expert_quality=int(data.get("expert_quality", np.asarray(-1)).item()),
                expert_quality_mask=bool(data.get("expert_quality_mask", np.asarray(False)).item()),
                capability_states=meta.get("capability_states", _default_capability_states()),
                metadata=meta.get("metadata", {}),
            )


def empty_quality_labels() -> tuple[np.ndarray, np.ndarray]:
    """Return quality targets when a source has no dimension-specific labels."""

    quality, mask = _default_quality()
    return quality, mask


def label_loss_weights(sequence: CanonicalSequence) -> np.ndarray:
    """Return weights for family, phase, boundary, and quality supervision."""

    provenance = str(sequence.metadata.get("label_provenance", "unknown"))
    provenance_weight = LABEL_PROVENANCE_WEIGHTS.get(provenance, 0.0)
    if provenance == "synthetic":
        synthetic_factor = 0.5
    else:
        synthetic_factor = 1.0
    weights = np.asarray(
        [
            provenance_weight if sequence.family >= 0 else 0.0,
            PHASE_LABEL_SOURCE_WEIGHTS.get(
                str(sequence.metadata.get("phase_label_source", "unknown")),
                0.0,
            ) if np.any(sequence.phase >= 0) else 0.0,
            BOUNDARY_LABEL_SOURCE_WEIGHTS.get(
                str(sequence.metadata.get("boundary_label_source", "unknown")),
                0.0,
            ) if np.any(sequence.rep_boundary >= 0) else 0.0,
            QUALITY_LABEL_SOURCE_WEIGHTS.get(
                str(sequence.metadata.get("quality_label_source", "unknown")),
                0.0,
            ) if sequence.quality_mask.any() else 0.0,
        ],
        dtype=np.float32,
    )
    weights *= synthetic_factor
    return weights


def expert_quality_loss_weight(sequence: CanonicalSequence) -> float:
    """Return the provenance weight for the optional composite score target."""

    if not sequence.expert_quality_mask or sequence.expert_quality < 0:
        return 0.0
    provenance = str(sequence.metadata.get("label_provenance", "unknown"))
    weight = LABEL_PROVENANCE_WEIGHTS.get(provenance, 0.0)
    if provenance == "synthetic":
        weight *= 0.5
    return float(weight)


def quality_from_correctness(correct: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Keep source correctness out of dimension-specific quality targets.

    A correctness score is retained in source metadata, but it does not
    identify ROM, tempo, smoothness, or trunk compensation. Returning masked
    targets prevents the training loop from silently treating it as ROM.
    """

    del correct
    return empty_quality_labels()


def derive_phase_labels(joints: np.ndarray) -> np.ndarray:
    """Create a transparent weak phase label for a single-repetition sequence.

    Public sources provide repetition segmentation but not a universal concentric/
    eccentric annotation. We use the frame with the largest displacement from the
    first pose as a turning point. This is a bootstrap label, not a clinical label.
    """

    joints = np.asarray(joints, dtype=np.float32)
    frames = len(joints)
    labels = np.full(frames, PHASE_TO_ID["unknown"], dtype=np.int64)
    if frames < 4:
        return labels
    finite = np.isfinite(joints).all(axis=-1)
    safe = np.where(finite[..., None], joints, 0.0)
    base = safe[0]
    displacement = np.linalg.norm(safe - base, axis=-1)
    displacement *= finite
    score = displacement.sum(axis=1)
    turn = int(np.argmax(score[1:-1]) + 1)
    if turn < 1 or turn >= frames - 1:
        turn = max(1, frames // 2)
    labels[: min(2, frames)] = PHASE_TO_ID["rest"]
    labels[min(2, turn) : max(min(2, turn) + 1, turn)] = PHASE_TO_ID["concentric"]
    labels[max(min(2, turn) + 1, turn) : max(2, frames - 2)] = PHASE_TO_ID["eccentric"]
    labels[max(0, frames - 2) :] = PHASE_TO_ID["rest"]
    if 0 < turn < frames - 1:
        labels[turn] = PHASE_TO_ID["hold"]
    return labels


def rep_boundary_labels(frames: int) -> np.ndarray:
    labels = np.full((frames, 2), -1.0, dtype=np.float32)
    if frames > 0:
        labels[:, :] = 0.0
        labels[0, 0] = 1.0
        labels[-1, 1] = 1.0
    return labels
