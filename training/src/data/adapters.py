"""Adapters for the downloaded public skeleton datasets.

The adapters emit :class:`CanonicalSequence` objects. They intentionally retain
source metadata so every downstream result can be traced back to a participant,
session, and source file.
"""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO, StringIO
import csv
import json
import logging
from pathlib import Path
import re
from typing import Iterable
from zipfile import ZipFile

import numpy as np

from ..config import Config
from .schema import (
    CANONICAL_INDEX,
    CanonicalSequence,
    empty_quality_labels,
    FAMILY_TO_ID,
    derive_phase_labels,
    LIMB_NAMES,
    rep_boundary_labels,
)


LOGGER = logging.getLogger(__name__)


REHAB24_SOURCE_NAMES = (
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "Head_end",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHand_end",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHand_end",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeBase_end",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "RightToeBase_end",
)

REHAB24_MAP: dict[str, tuple[str, ...]] = {
    "Head": ("nose",),
    "Neck": ("nose",),
    "LeftShoulder": ("left_shoulder",),
    "LeftArm": ("left_elbow",),
    "LeftForeArm": ("left_wrist",),
    "LeftHand": ("left_index",),
    "LeftHand_end": ("left_index",),
    "RightShoulder": ("right_shoulder",),
    "RightArm": ("right_elbow",),
    "RightForeArm": ("right_wrist",),
    "RightHand": ("right_index",),
    "RightHand_end": ("right_index",),
    "LeftUpLeg": ("left_hip",),
    "LeftLeg": ("left_knee",),
    "LeftFoot": ("left_ankle",),
    "LeftToeBase": ("left_foot_index",),
    "LeftToeBase_end": ("left_foot_index",),
    "RightUpLeg": ("right_hip",),
    "RightLeg": ("right_knee",),
    "RightFoot": ("right_ankle",),
    "RightToeBase": ("right_foot_index",),
    "RightToeBase_end": ("right_foot_index",),
}

REHAB24_FAMILY = {
    1: FAMILY_TO_ID["reach"],
    2: FAMILY_TO_ID["reach"],
    3: FAMILY_TO_ID["other"],
    4: FAMILY_TO_ID["hip_flexion"],
    5: FAMILY_TO_ID["knee_extension"],
    6: FAMILY_TO_ID["other"],
}

INTELLI_SOURCE_NAMES = (
    "SpineBase",
    "SpineMid",
    "Neck",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
    "SpineShoulder",
    "HandTipLeft",
    "ThumbLeft",
    "HandTipRight",
    "ThumbRight",
)

INTELLI_MAP: dict[str, tuple[str, ...]] = {
    "Neck": ("nose",),
    "Head": ("nose",),
    "ShoulderLeft": ("left_shoulder",),
    "ElbowLeft": ("left_elbow",),
    "WristLeft": ("left_wrist",),
    "HandLeft": ("left_index",),
    "ShoulderRight": ("right_shoulder",),
    "ElbowRight": ("right_elbow",),
    "WristRight": ("right_wrist",),
    "HandRight": ("right_index",),
    "HipLeft": ("left_hip",),
    "KneeLeft": ("left_knee",),
    "AnkleLeft": ("left_ankle",),
    "FootLeft": ("left_foot_index",),
    "HipRight": ("right_hip",),
    "KneeRight": ("right_knee",),
    "AnkleRight": ("right_ankle",),
    "FootRight": ("right_foot_index",),
    "SpineShoulder": ("nose",),
    "HandTipLeft": ("left_pinky",),
    "ThumbLeft": ("left_thumb",),
    "HandTipRight": ("right_pinky",),
    "ThumbRight": ("right_thumb",),
}

INTELLI_FAMILY = {
    0: FAMILY_TO_ID["arm_flexion"],
    1: FAMILY_TO_ID["arm_flexion"],
    2: FAMILY_TO_ID["arm_flexion"],
    3: FAMILY_TO_ID["arm_flexion"],
    4: FAMILY_TO_ID["reach"],
    5: FAMILY_TO_ID["reach"],
    6: FAMILY_TO_ID["reach"],
    7: FAMILY_TO_ID["hip_flexion"],
    8: FAMILY_TO_ID["hip_flexion"],
}

# MM-Fit publishes 2D COCO-style pose estimates and explicit exercise-set
# annotations. Its fine-grained exercise names are mapped into AdaptFit's
# first-release movement families and retained in metadata for auditability.
MMFIT_SOURCE_NAMES = (
    "Nose",
    "Neck",
    "Right Shoulder",
    "Right Elbow",
    "Right Wrist",
    "Left Shoulder",
    "Left Elbow",
    "Left Wrist",
    "Right Hip",
    "Right Knee",
    "Right Ankle",
    "Left Hip",
    "Left Knee",
    "Left Ankle",
    "Right Eye",
    "Left Eye",
    "Right Ear",
    "Left Ear",
)

MMFIT_MAP: dict[str, tuple[str, ...]] = {
    "Nose": ("nose",),
    "Right Shoulder": ("right_shoulder",),
    "Right Elbow": ("right_elbow",),
    "Right Wrist": ("right_wrist",),
    "Left Shoulder": ("left_shoulder",),
    "Left Elbow": ("left_elbow",),
    "Left Wrist": ("left_wrist",),
    "Right Hip": ("right_hip",),
    "Right Knee": ("right_knee",),
    "Right Ankle": ("right_ankle",),
    "Left Hip": ("left_hip",),
    "Left Knee": ("left_knee",),
    "Left Ankle": ("left_ankle",),
    "Right Eye": ("right_eye",),
    "Left Eye": ("left_eye",),
    "Right Ear": ("right_ear",),
    "Left Ear": ("left_ear",),
}

MMFIT_FAMILY = {
    "bicep_curls": FAMILY_TO_ID["arm_flexion"],
    "tricep_extensions": FAMILY_TO_ID["arm_flexion"],
    "dumbbell_shoulder_press": FAMILY_TO_ID["arm_flexion"],
    "lateral_shoulder_raises": FAMILY_TO_ID["arm_flexion"],
    "dumbbell_rows": FAMILY_TO_ID["row_pull"],
    "squats": FAMILY_TO_ID["knee_extension"],
    "lunges": FAMILY_TO_ID["knee_extension"],
    "situps": FAMILY_TO_ID["hip_flexion"],
    # These do not have an honest first-release family equivalent.
    "pushups": FAMILY_TO_ID["other"],
    "jumping_jacks": FAMILY_TO_ID["other"],
    "non_activity": FAMILY_TO_ID["other"],
}

MMFIT_ALIASES = {
    "bicep_curl": "bicep_curls",
    "bicep_curls": "bicep_curls",
    "dumbbell_shoulder_press": "dumbbell_shoulder_press",
    "dumbbell_shoulder_presses": "dumbbell_shoulder_press",
    "dumbbell_tricep_extensions": "tricep_extensions",
    "tricep_extensions": "tricep_extensions",
    "dumbbell_rows": "dumbbell_rows",
    "dumbbell_row": "dumbbell_rows",
    "lateral_shoulder_raises": "lateral_shoulder_raises",
    "lateral_shoulder_raise": "lateral_shoulder_raises",
    "squats": "squats",
    "squat": "squats",
    "lunges": "lunges",
    "lunge": "lunges",
    "situps": "situps",
    "sit_ups": "situps",
    "pushups": "pushups",
    "push_ups": "pushups",
    "jumping_jacks": "jumping_jacks",
    "jumping_jack": "jumping_jacks",
    "non_activity": "non_activity",
}

# The official MM-Fit starter materials publish this mapping because a
# participant can have more than one workout session. Keeping it here is
# important: using the workout ID as participant_id would leak the same
# person's sessions across the participant-level train/validation/test split.
MMFIT_PARTICIPANT_BY_WORKOUT = {
    "w00": 2,
    "w01": 0,
    "w02": 1,
    "w03": 0,
    "w04": 1,
    "w05": 2,
    "w06": 0,
    "w07": 1,
    "w08": 0,
    "w09": 1,
    "w10": 0,
    "w11": 1,
    "w12": 3,
    "w13": 4,
    "w14": 0,
    "w15": 1,
    "w16": 5,
    "w17": 6,
    "w18": 7,
    "w19": 8,
    "w20": 9,
}

# UL-RED publishes marker-less motion as positional AMC frames. The marker-less
# skeleton is a compact 19-joint hierarchy rooted at Waist; the mapping keeps
# that source's observed/missing coordinates explicit when projecting to the
# 33-joint AdaptFit layout.
ULRED_SOURCE_NAMES = (
    "Head",
    "Neck",
    "LeftCollar",
    "Torso",
    "Waist",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "LeftHand",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
    "RightHand",
    "LeftHip",
    "LeftKnee",
    "LeftAnkle",
    "RightHip",
    "RightKnee",
    "RightAnkle",
)

ULRED_MAP: dict[str, tuple[str, ...]] = {
    "Head": ("nose",),
    "Neck": ("nose",),
    "LeftShoulder": ("left_shoulder",),
    "LeftElbow": ("left_elbow",),
    "LeftWrist": ("left_wrist",),
    "LeftHand": ("left_index",),
    "RightShoulder": ("right_shoulder",),
    "RightElbow": ("right_elbow",),
    "RightWrist": ("right_wrist",),
    "RightHand": ("right_index",),
    "LeftHip": ("left_hip",),
    "LeftKnee": ("left_knee",),
    "LeftAnkle": ("left_ankle",),
    "RightHip": ("right_hip",),
    "RightKnee": ("right_knee",),
    "RightAnkle": ("right_ankle",),
}

_ULRED_FILENAME = re.compile(
    r"^(?P<exercise>.+)R(?P<recording_repetitions>[13])(?P<subject>S\d+)$",
    re.IGNORECASE,
)


def _project_named_joints(
    source_coordinates: np.ndarray,
    source_names: Iterable[str],
    source_map: dict[str, tuple[str, ...]],
    treat_zero_as_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project named source joints into the 33-joint canonical layout."""

    coordinates = np.asarray(source_coordinates, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[-1] < 2:
        raise ValueError(f"Expected [T, joints, coordinates], got {coordinates.shape}")
    xy = coordinates[..., :2]
    finite = np.isfinite(xy).all(axis=-1)
    nonzero = np.abs(xy).sum(axis=-1) > 1e-8
    source_observed = finite & (nonzero if treat_zero_as_missing else True)
    output = np.full((len(xy), len(CANONICAL_INDEX), 2), np.nan, dtype=np.float32)
    observed = np.zeros((len(xy), len(CANONICAL_INDEX)), dtype=bool)

    for source_index, source_name in enumerate(source_names):
        canonical_names = source_map.get(source_name, ())
        for canonical_name in canonical_names:
            target_index = CANONICAL_INDEX[canonical_name]
            valid = source_observed[:, source_index]
            output[valid, target_index] = xy[valid, source_index]
            observed[valid, target_index] = True

    output[~observed] = np.nan
    confidence = observed.astype(np.float32)
    return output, confidence, observed


def _make_sequence(
    *,
    joints: np.ndarray,
    confidence: np.ndarray,
    observed: np.ndarray,
    position: str,
    participant_id: str,
    session_id: str,
    source_dataset: str,
    family: int,
    quality: np.ndarray,
    quality_mask: np.ndarray,
    label_provenance: str,
    phase_label_source: str,
    boundary_label_source: str,
    quality_label_source: str,
    metadata: dict,
    rep_boundary: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    phase: np.ndarray | None = None,
    expert_quality: int = -1,
    expert_quality_mask: bool = False,
    capability_states: dict[str, str] | None = None,
) -> CanonicalSequence:
    """Build one canonical sequence without manufacturing unavailable labels.

    Callers must provide boundary labels explicitly when the source supplies
    them.  ``CanonicalSequence`` turns an omitted boundary array into an
    all-masked target, which is the correct default for sources without
    frame-level repetition segmentation.
    """

    frames = len(joints)
    metadata = dict(metadata)
    metadata.setdefault("label_provenance", label_provenance)
    metadata.setdefault("phase_label_source", phase_label_source)
    metadata.setdefault("boundary_label_source", boundary_label_source)
    metadata.setdefault("quality_label_source", quality_label_source)
    # Some public sources intentionally expose only a partial canonical pose
    # (for example, UCO records three joints for a unilateral leg exercise).
    # Preserve that structural observability contract so the tracking target
    # does not treat unlisted joints as failed camera observations.
    metadata.setdefault(
        "tracking_expected_joint_names",
        [
            name
            for name, index in CANONICAL_INDEX.items()
            if bool(np.any(observed[:, index]))
        ],
    )
    return CanonicalSequence(
        joints=joints,
        pose_confidence=confidence,
        observed_mask=observed,
        timestamps=(
            np.arange(frames, dtype=np.float32) / 30.0
            if timestamps is None
            else np.asarray(timestamps, dtype=np.float32)
        ),
        position=position,
        participant_id=str(participant_id),
        session_id=str(session_id),
        source_dataset=source_dataset,
        family=family,
        phase=derive_phase_labels(joints) if phase is None else phase,
        rep_boundary=rep_boundary,
        quality=quality,
        quality_mask=quality_mask,
        expert_quality=expert_quality,
        expert_quality_mask=expert_quality_mask,
        capability_states=capability_states or {},
        metadata=metadata,
    )


def _read_rehab24_segmentation(path: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            groups[row["video_id"]].append(row)
    return groups


def _find_rehab24_member(names: set[str], exercise_id: int, video_id: str) -> str | None:
    preferred = f"Ex{exercise_id}/{video_id}-c17-30fps.npy"
    if preferred in names:
        return preferred
    alternate = f"Ex{exercise_id}/{video_id}-c18-30fps.npy"
    if alternate in names:
        return alternate
    return None


def load_rehab24_6(root: str | Path) -> list[CanonicalSequence]:
    """Load REHAB24-6 2D 30 FPS joints and repetition segmentation."""

    root = Path(root)
    archive = root / "2d_joints.zip"
    segmentation = root / "Segmentation.csv"
    if not archive.exists() or not segmentation.exists():
        return []

    grouped = _read_rehab24_segmentation(segmentation)
    sequences: list[CanonicalSequence] = []
    with ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        for video_id, rows in grouped.items():
            arrays: dict[int, np.ndarray] = {}
            for row in rows:
                exercise_id = int(row["exercise_id"])
                member = _find_rehab24_member(names, exercise_id, video_id)
                if member is None:
                    continue
                if exercise_id not in arrays:
                    arrays[exercise_id] = np.load(BytesIO(zipped.read(member)))
                full = arrays[exercise_id]
                start = max(0, int(row["first_frame"]))
                end = min(len(full), int(row["last_frame"]) + 1)
                if end - start < 4:
                    continue
                joints, confidence, observed = _project_named_joints(
                    full[start:end], REHAB24_SOURCE_NAMES, {**REHAB24_MAP}
                )
                correctness = int(row.get("correctness", "-1"))
                quality, quality_mask = empty_quality_labels()
                sequences.append(
                    _make_sequence(
                        joints=joints,
                        confidence=confidence,
                        observed=observed,
                        position="standing",
                        participant_id=row["person_id"],
                        session_id=video_id,
                        source_dataset="rehab24_6",
                        family=REHAB24_FAMILY.get(exercise_id, FAMILY_TO_ID["other"]),
                        quality=quality,
                        quality_mask=quality_mask,
                        label_provenance="weak",
                        phase_label_source="weak_displacement",
                        boundary_label_source="segmentation",
                        quality_label_source="unlabeled",
                        metadata={
                            "video_id": video_id,
                            "exercise_id": exercise_id,
                            "camera": "c17",
                            "orientation": row.get("cam17_orientation", "unknown"),
                            "exercise_subtype": row.get("exercise_subtype", "unknown"),
                            "correctness": correctness,
                            "source_frame_start": start,
                            "source_frame_end": end - 1,
                            "labels_are_weak_phase": True,
                            "labels_are_weak_boundary": False,
                        },
                        rep_boundary=rep_boundary_labels(len(joints)),
                    )
                )
    return sequences


_INTELLI_FILENAME = re.compile(
    r"^(?P<subject>[^_]+)_(?P<date>[^_]+)_(?P<gesture>\d+)_(?P<rep>\d+)_(?P<correct>\d+)_(?P<position>.+)\.txt$",
    re.IGNORECASE,
)


def _intelli_position(raw: str) -> str:
    value = raw.lower()
    if "wheelchair" in value:
        return "wheelchair"
    if "chair" in value or value.startswith("sit"):
        return "seated"
    if "stand" in value:
        return "standing"
    return "unknown"


def _load_intelli_text(handle) -> np.ndarray:
    rows: list[list[float]] = []
    for raw_line in handle:
        line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else raw_line
        values = [value for value in line.strip().split(",") if value]
        if not values:
            continue
        try:
            rows.append([float(value) for value in values])
        except ValueError:
            continue
    if not rows:
        return np.empty((0, 25, 3), dtype=np.float32)
    matrix = np.asarray(rows, dtype=np.float32)
    usable = min(matrix.shape[1], 75)
    matrix = matrix[:, :usable]
    if usable < 75:
        padded = np.full((len(matrix), 75), np.nan, dtype=np.float32)
        padded[:, :usable] = matrix
        matrix = padded
    return matrix.reshape(len(matrix), 25, 3)


def load_intellirehabds(
    root: str | Path,
    allow_single_clip_boundaries: bool = False,
) -> list[CanonicalSequence]:
    """Load IntelliRehabDS skeleton clips with optional clip-boundary labels.

    The source filename identifies a clip/repetition, but does not provide
    frame-level segmentation. Boundary targets are therefore masked by
    default; enabling them is reserved for explicit ablation experiments.
    """

    root = Path(root)
    archive = root / "SkeletonData.zip"
    sequences: list[CanonicalSequence] = []

    def consume(name: str, handle) -> None:
        match = _INTELLI_FILENAME.match(Path(name).name)
        if match is None:
            return
        correct = int(match.group("correct"))
        if correct == 3:
            return
        gesture = int(match.group("gesture"))
        source = _load_intelli_text(handle)
        if len(source) < 4:
            return
        joints, confidence, observed = _project_named_joints(
            source, INTELLI_SOURCE_NAMES, INTELLI_MAP
        )
        quality, quality_mask = empty_quality_labels()
        boundary_source = "single_clip_assumption" if allow_single_clip_boundaries else "unlabeled"
        boundaries = (
            rep_boundary_labels(len(source))
            if allow_single_clip_boundaries
            else np.full((len(source), 2), -1.0, dtype=np.float32)
        )
        sequences.append(
            _make_sequence(
                joints=joints,
                confidence=confidence,
                observed=observed,
                position=_intelli_position(match.group("position")),
                participant_id=match.group("subject"),
                session_id=f"{match.group('subject')}_{match.group('date')}",
                source_dataset="intellirehabds",
                family=INTELLI_FAMILY.get(gesture, FAMILY_TO_ID["other"]),
                quality=quality,
                quality_mask=quality_mask,
                label_provenance="weak",
                phase_label_source="weak_displacement",
                boundary_label_source=boundary_source,
                quality_label_source="unlabeled",
                metadata={
                    "source_member": Path(name).name,
                    "gesture_id": gesture,
                    "repetition_number": int(match.group("rep")),
                    "correctness": correct,
                    "position_label": match.group("position"),
                    "labels_are_weak_phase": True,
                    "labels_are_weak_boundary": bool(allow_single_clip_boundaries),
                    "single_clip_boundary_enabled": bool(allow_single_clip_boundaries),
                },
                rep_boundary=boundaries,
            )
        )

    if archive.exists():
        with ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if not name.lower().endswith(".txt"):
                    continue
                with zipped.open(name) as handle:
                    consume(name, handle)
    else:
        for path in sorted(root.rglob("*.txt")):
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                consume(path.name, handle)
    return sequences


def _normalise_mmfit_activity(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return MMFIT_ALIASES.get(normalized, normalized)


def _read_mmfit_labels(handle) -> list[tuple[int, int, int, str]]:
    """Read MM-Fit's start/end/repetition/activity set annotations."""

    rows: list[tuple[int, int, int, str]] = []
    text = handle.read() if hasattr(handle, "read") else handle
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    reader = csv.reader(str(text).splitlines())
    for row in reader:
        if len(row) < 4:
            continue
        try:
            start = int(float(row[0]))
            end = int(float(row[1]))
            repetitions = int(float(row[2]))
        except (TypeError, ValueError):
            # This also skips a possible header without assuming one exists.
            continue
        if end < start:
            continue
        rows.append((start, end, max(0, repetitions), _normalise_mmfit_activity(row[3])))
    return rows


def _load_mmfit_pose_2d(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return MM-Fit pose as [T, 18, 2] plus its source frame IDs."""

    raw = np.asarray(np.load(BytesIO(payload), allow_pickle=False))
    if raw.ndim != 3:
        raise ValueError(f"MM-Fit pose_2d must be rank 3, got {raw.shape}")

    # The released layout is [x/y, frame, frame_id + 18 joints]. Some mirrors
    # transpose it, so accept the equivalent [frame, joints, x/y] layout too.
    if raw.shape[0] in (2, 3) and raw.shape[2] >= len(MMFIT_SOURCE_NAMES) + 1:
        frame_ids = raw[0, :, 0].astype(np.float32)
        coordinates = np.moveaxis(raw[:2, :, 1 : len(MMFIT_SOURCE_NAMES) + 1], 0, -1)
    elif raw.shape[-1] in (2, 3) and raw.shape[1] >= len(MMFIT_SOURCE_NAMES) + 1:
        frame_ids = raw[:, 0, 0].astype(np.float32)
        coordinates = raw[:, 1 : len(MMFIT_SOURCE_NAMES) + 1, :2].astype(np.float32)
    else:
        raise ValueError(f"Unrecognized MM-Fit pose_2d layout: {raw.shape}")

    if coordinates.shape != (len(frame_ids), len(MMFIT_SOURCE_NAMES), 2):
        raise ValueError(f"MM-Fit pose_2d normalized to unexpected shape {coordinates.shape}")
    finite_frames = np.isfinite(frame_ids)
    if not finite_frames.any():
        raise ValueError("MM-Fit pose_2d contains no finite frame IDs")
    frame_ids = frame_ids[finite_frames]
    coordinates = coordinates[finite_frames]
    return coordinates, frame_ids


def _mmfit_member_by_basename(names: set[str], basename: str) -> str | None:
    matches = sorted(name for name in names if Path(name).name == basename)
    return matches[0] if matches else None


def _mmfit_position(activity: str) -> str:
    if activity in {"dumbbell_shoulder_press", "lateral_shoulder_raises"}:
        return "seated"
    if activity in {
        "squats",
        "lunges",
        "bicep_curls",
        "tricep_extensions",
        "dumbbell_rows",
        "jumping_jacks",
    }:
        return "standing"
    return "unknown"


def _mmfit_participant_id(workout_id: str) -> str:
    participant = MMFIT_PARTICIPANT_BY_WORKOUT.get(workout_id)
    if participant is None:
        # Preserve the no-leakage invariant for an unrecognized future workout
        # by keeping it isolated until its participant mapping is confirmed.
        return f"mmfit_workout_{workout_id}"
    return f"mmfit_participant_{participant}"


def _mmfit_sequences_from_workout(
    workout_id: str,
    labels: list[tuple[int, int, int, str]],
    pose_payload: bytes,
) -> list[CanonicalSequence]:
    coordinates, frame_ids = _load_mmfit_pose_2d(pose_payload)
    sequences: list[CanonicalSequence] = []
    for set_index, (start, end, repetitions, activity) in enumerate(labels):
        if activity not in MMFIT_FAMILY:
            LOGGER.warning("Skipping unknown MM-Fit activity %r in %s", activity, workout_id)
            continue
        selected = (frame_ids >= start) & (frame_ids <= end)
        if int(selected.sum()) < 4:
            continue
        joints, confidence, observed = _project_named_joints(
            coordinates[selected], MMFIT_SOURCE_NAMES, MMFIT_MAP
        )
        selected_frames = frame_ids[selected]
        # MM-Fit's labels identify exercise-set spans and repetition counts, not
        # individual repetition boundaries. Keep boundary/phase supervision
        # masked instead of manufacturing exact per-rep timestamps.
        phase = np.full(len(selected_frames), -1, dtype=np.int64)
        boundaries = np.full((len(selected_frames), 2), -1.0, dtype=np.float32)
        quality, quality_mask = empty_quality_labels()
        sequences.append(
            CanonicalSequence(
                joints=joints,
                pose_confidence=confidence,
                observed_mask=observed,
                timestamps=(selected_frames - selected_frames[0]) / 30.0,
                position=_mmfit_position(activity),
                participant_id=_mmfit_participant_id(workout_id),
                session_id=workout_id,
                source_dataset="mmfit",
                family=MMFIT_FAMILY[activity],
                phase=phase,
                rep_boundary=boundaries,
                quality=quality,
                quality_mask=quality_mask,
                metadata={
                    "workout_id": workout_id,
                    "participant_source_id": MMFIT_PARTICIPANT_BY_WORKOUT.get(workout_id),
                    "set_index": set_index,
                    "activity": activity,
                    "repetition_count": repetitions,
                    "repetition_count_source": "mmfit_set_annotation",
                    "set_start_frame": int(selected_frames[0]),
                    "set_end_frame": int(selected_frames[-1]),
                    "labels_are_explicit_activity": True,
                    "labels_are_set_level": True,
                    "label_provenance": "weak",
                    "phase_label_source": "unknown",
                    "boundary_label_source": "unknown",
                    "quality_label_source": "unlabeled",
                },
            )
        )
    return sequences


def load_mmfit(root: str | Path) -> list[CanonicalSequence]:
    """Load MM-Fit's pose_2d arrays and exercise-set annotations.

    Both the official ``mm-fit.zip`` archive and an extracted workout-folder
    layout are supported. Only pose arrays are read; RGB/depth and wearable
    streams are intentionally left out of this pose-only training pipeline.
    """

    root = Path(root)
    archive = root / "mm-fit.zip"
    sequences: list[CanonicalSequence] = []

    if archive.exists():
        with ZipFile(archive) as zipped:
            names = set(zipped.namelist())
            label_members = sorted(
                name for name in names if Path(name).name.lower().endswith("_labels.csv")
            )
            for label_member in label_members:
                label_name = Path(label_member).name
                workout_id = label_name[: -len("_labels.csv")]
                pose_member = _mmfit_member_by_basename(names, f"{workout_id}_pose_2d.npy")
                if pose_member is None:
                    LOGGER.warning("MM-Fit workout %s has labels but no pose_2d array", workout_id)
                    continue
                labels = _read_mmfit_labels(zipped.read(label_member))
                try:
                    pose_payload = zipped.read(pose_member)
                    sequences.extend(_mmfit_sequences_from_workout(workout_id, labels, pose_payload))
                except (OSError, ValueError) as error:
                    LOGGER.warning("Skipping invalid MM-Fit workout %s: %s", workout_id, error)
        return sequences

    for label_path in sorted(root.rglob("*_labels.csv")):
        workout_id = label_path.name[: -len("_labels.csv")]
        pose_path = label_path.with_name(f"{workout_id}_pose_2d.npy")
        if not pose_path.exists():
            LOGGER.warning("MM-Fit workout %s has labels but no pose_2d array", workout_id)
            continue
        labels = _read_mmfit_labels(label_path.read_text(encoding="utf-8", errors="ignore"))
        try:
            sequences.extend(
                _mmfit_sequences_from_workout(workout_id, labels, pose_path.read_bytes())
            )
        except (OSError, ValueError) as error:
            LOGGER.warning("Skipping invalid MM-Fit workout %s: %s", workout_id, error)
    return sequences


def _load_ulred_amc(handle) -> tuple[np.ndarray, np.ndarray]:
    """Parse UL-RED positional AMC frames into [T, joints, 3] coordinates."""

    raw = handle.read() if hasattr(handle, "read") else handle
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")

    frames: list[dict[str, np.ndarray]] = []
    frame_ids: list[int] = []
    current: dict[str, np.ndarray] | None = None
    current_id: int | None = None

    def flush() -> None:
        if current is not None and current_id is not None and current:
            frames.append(current.copy())
            frame_ids.append(current_id)

    for line in str(raw).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        fields = stripped.split()
        if len(fields) == 1 and re.fullmatch(r"\d+", fields[0]):
            flush()
            current = {}
            current_id = int(fields[0])
            continue
        if current is None or len(fields) < 4:
            continue
        try:
            point = np.asarray([float(value) for value in fields[1:4]], dtype=np.float32)
        except ValueError:
            continue
        if np.isfinite(point).all():
            current[fields[0]] = point
    flush()

    if not frames:
        return np.empty((0, len(ULRED_SOURCE_NAMES), 3), dtype=np.float32), np.empty(0)

    coordinates = np.full(
        (len(frames), len(ULRED_SOURCE_NAMES), 3), np.nan, dtype=np.float32
    )
    for frame_index, frame in enumerate(frames):
        for joint_index, joint_name in enumerate(ULRED_SOURCE_NAMES):
            point = frame.get(joint_name)
            if point is not None:
                coordinates[frame_index, joint_index] = point
    return coordinates, np.asarray(frame_ids, dtype=np.float32)


def _normalise_ulred_exercise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _ulred_family(exercise: str) -> int:
    normalized = _normalise_ulred_exercise(exercise)
    if "bicepcurl" in normalized or "chestpress" in normalized:
        return FAMILY_TO_ID["arm_flexion"]
    if any(token in normalized for token in ("armraise", "forwardarmraise")):
        return FAMILY_TO_ID["reach"]
    if any(
        token in normalized
        for token in (
        "legextension",
            "minisquat",
            "calfraise",
            "sidewaysstep",
            "sittostand",
        )
    ):
        return FAMILY_TO_ID["knee_extension"]
    if any(
        token in normalized
        for token in ("kneeraise", "seatedhipmarch", "sidewaysleglift", "hiphoop")
    ):
        return FAMILY_TO_ID["hip_flexion"]
    return FAMILY_TO_ID["other"]


def _ulred_position(exercise: str) -> str:
    normalized = _normalise_ulred_exercise(exercise)
    if normalized.startswith("seated") or normalized == "sittostand":
        return "seated"
    return "standing"


def _ulred_metadata_from_member(member: str) -> tuple[str, int, str] | None:
    match = _ULRED_FILENAME.match(Path(member).stem)
    if match is None:
        return None
    exercise = match.group("exercise")
    repetitions = int(match.group("recording_repetitions"))
    subject = match.group("subject").upper()
    return exercise, repetitions, subject


_ULRED_BOUNDARY_KEYS = (
    "r1start",
    "r1end",
    "r2start",
    "r2end",
    "r3start",
    "r3end",
)


def _read_ulred_boundary_rows(raw) -> dict[str, dict[str, int]]:
    """Read explicit markerless three-repetition spans from a UL-RED CSV."""

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig", errors="replace")
    rows: dict[str, dict[str, int]] = {}
    for row in csv.DictReader(StringIO(str(raw))):
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        try:
            values = {key: int(row[key]) for key in _ULRED_BOUNDARY_KEYS}
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("Skipping malformed UL-RED boundary row %r", row)
            continue
        if any(value < 0 for value in values.values()):
            LOGGER.warning("Skipping negative UL-RED boundary row %r", row)
            continue
        if any(values[start] > values[end] for start, end in (("r1start", "r1end"), ("r2start", "r2end"), ("r3start", "r3end"))):
            LOGGER.warning("Skipping reversed UL-RED boundary row %r", row)
            continue
        rows[name] = values
    return rows


def _ulred_boundary_labels(
    frame_ids: np.ndarray,
    row: dict[str, int] | None,
) -> np.ndarray | None:
    """Convert zero-based UL-RED markerless spans to AMC frame labels."""

    if row is None:
        return None
    frame_to_index = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    labels = np.zeros((len(frame_ids), 2), dtype=np.float32)
    for column, keys in ((0, ("r1start", "r2start", "r3start")), (1, ("r1end", "r2end", "r3end"))):
        for key in keys:
            # UL-RED's 3Rep CSV uses zero-based exported-frame indices while
            # the positional AMC parser exposes one-based frame IDs.
            frame_id = int(row[key]) + 1
            index = frame_to_index.get(frame_id)
            if index is None:
                return None
            labels[index, column] = 1.0
    return labels


def _make_ulred_sequence(
    *,
    member: str,
    payload,
    source_archive: str | None = None,
    boundary_row: dict[str, int] | None = None,
    boundary_source_member: str | None = None,
) -> CanonicalSequence | None:
    parsed = _ulred_metadata_from_member(member)
    if parsed is None:
        LOGGER.warning("Skipping UL-RED file with unrecognized name: %s", member)
        return None
    exercise, recording_repetitions, subject = parsed
    coordinates, frame_ids = _load_ulred_amc(payload)
    if len(coordinates) < 4:
        return None
    joints, confidence, observed = _project_named_joints(
        coordinates, ULRED_SOURCE_NAMES, ULRED_MAP
    )
    if len(frame_ids) != len(joints) or len(frame_ids) < 2 or not np.all(np.diff(frame_ids) > 0):
        frame_ids = np.arange(len(joints), dtype=np.float32) + 1.0
    timestamps = (frame_ids - frame_ids[0]) / 30.0
    quality, quality_mask = empty_quality_labels()
    is_multi_repetition = recording_repetitions > 1
    phase = np.full(len(joints), -1, dtype=np.int64) if is_multi_repetition else None
    effective_phase_source = "unlabeled" if is_multi_repetition else "weak_displacement"
    boundaries = _ulred_boundary_labels(frame_ids, boundary_row)
    has_explicit_boundaries = boundaries is not None
    return _make_sequence(
        joints=joints,
        confidence=confidence,
        observed=observed,
        position=_ulred_position(exercise),
        participant_id=subject,
        session_id=Path(member).stem,
        source_dataset="ul_red",
        family=_ulred_family(exercise),
        quality=quality,
        quality_mask=quality_mask,
        label_provenance="weak",
        phase_label_source=effective_phase_source,
        boundary_label_source="strong" if has_explicit_boundaries else "unknown",
        quality_label_source="unlabeled",
        metadata={
            "exercise_name": exercise,
            "recording_repetitions": recording_repetitions,
            "repetition_count_source": "ul_red_recording_protocol",
            "pace_protocol": "normal" if recording_repetitions == 1 else "normal_fast_slow",
            "marker_type": "marker-less",
            "source_frame_rate": 30.0,
            "source_member": member,
            "source_archive": source_archive,
            "boundary_source_member": boundary_source_member,
            "labels_are_explicit_boundaries": has_explicit_boundaries,
            "boundary_label_status": (
                "explicit_markerless_three_repetition_spans"
                if has_explicit_boundaries
                else "unavailable"
            ),
            "boundary_frame_indexing": "zero_based_csv_to_one_based_amc"
            if has_explicit_boundaries
            else None,
            "labels_are_weak_phase": not is_multi_repetition,
            "labels_are_recording_level": True,
            "phase_label_status": (
                "unavailable_multi_repetition_recording"
                if is_multi_repetition
                else "weak_single_repetition_recording"
            ),
            "phase_unavailable_reason": (
                "recording_contains_multiple_repetitions" if is_multi_repetition else None
            ),
        },
        timestamps=timestamps,
        phase=phase,
        rep_boundary=boundaries,
    )


def _ulred_amc_members(names: Iterable[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if name.lower().endswith(".amc")
        and "marker-less/clean/" in name.lower().replace("\\", "/")
    )


def load_ul_red(root: str | Path) -> list[CanonicalSequence]:
    """Load UL-RED marker-less clean AMC recordings from subject archives."""

    root = Path(root)
    sequences: list[CanonicalSequence] = []
    archives = sorted(root.glob("S*.zip"))
    for archive_path in archives:
        try:
            with ZipFile(archive_path) as zipped:
                names = zipped.namelist()
                boundary_member = next(
                    (
                        name
                        for name in names
                        if re.search(r"/marker-less/3Rep_.*\.csv$", name, re.IGNORECASE)
                    ),
                    None,
                )
                boundary_rows = (
                    _read_ulred_boundary_rows(zipped.read(boundary_member))
                    if boundary_member is not None
                    else {}
                )
                for member in _ulred_amc_members(zipped.namelist()):
                    try:
                        with zipped.open(member) as payload:
                            sequence = _make_ulred_sequence(
                                member=member,
                                payload=payload,
                                source_archive=archive_path.name,
                                boundary_row=boundary_rows.get(Path(member).stem),
                                boundary_source_member=boundary_member,
                            )
                    except (OSError, ValueError) as error:
                        LOGGER.warning("Skipping UL-RED member %s: %s", member, error)
                        continue
                    if sequence is not None:
                        sequences.append(sequence)
        except (OSError, ValueError) as error:
            LOGGER.warning("Skipping UL-RED archive %s: %s", archive_path, error)

    # Prefer archives when they are present. This avoids duplicate sequences
    # when a user extracts an archive beside its original ZIP.
    if archives and sequences:
        return sequences

    for member_path in sorted(root.rglob("*.amc")):
        normalized_parts = {part.lower().replace("_", "-") for part in member_path.parts}
        if "marker-less" not in normalized_parts or "clean" not in normalized_parts:
            continue
        if any(sequence.metadata.get("source_member") == str(member_path) for sequence in sequences):
            continue
        boundary_root = member_path.parent.parent
        subject_root = boundary_root.parent
        boundary_path = boundary_root / f"3Rep_{subject_root.name}.csv"
        boundary_rows = {}
        if boundary_path.exists():
            try:
                boundary_rows = _read_ulred_boundary_rows(boundary_path.read_bytes())
            except OSError as error:
                LOGGER.warning("Unable to read UL-RED boundary file %s: %s", boundary_path, error)
        try:
            with member_path.open("rb") as payload:
                sequence = _make_ulred_sequence(
                    member=str(member_path),
                    payload=payload,
                    boundary_row=boundary_rows.get(member_path.stem),
                    boundary_source_member=str(boundary_path) if boundary_path.exists() else None,
                )
        except (OSError, ValueError) as error:
            LOGGER.warning("Skipping UL-RED file %s: %s", member_path, error)
            continue
        if sequence is not None:
            sequences.append(sequence)
    return sequences


# UCOPhyRehab++ publishes the 3D coordinates for the three landmarks used by
# its exercise-specific angle target.  The pose file is intentionally treated
# as a partial pose source: unlisted landmarks remain unobserved rather than
# being filled with a plausible-looking full body.
UCO_SOURCE_NAMES = (
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
)

UCO_MAP: dict[str, tuple[str, ...]] = {
    "l_hip": ("left_hip",),
    "l_knee": ("left_knee",),
    "l_ankle": ("left_ankle",),
    "r_hip": ("right_hip",),
    "r_knee": ("right_knee",),
    "r_ankle": ("right_ankle",),
    "l_shoulder": ("left_shoulder",),
    "l_elbow": ("left_elbow",),
    "l_wrist": ("left_wrist",),
    "r_shoulder": ("right_shoulder",),
    "r_elbow": ("right_elbow",),
    "r_wrist": ("right_wrist",),
}

UCO_EXERCISE_NAMES = {
    "01": "bending_knee_without_support_sitting",
    "02": "bending_knee_with_support_sitting",
    "03": "lift_extended_leg",
    "04": "bending_knee_with_bed_support",
    "05": "bending_knee_without_support_sitting",
    "06": "bending_knee_with_support_sitting",
    "07": "lift_extended_leg",
    "08": "bending_knee_with_bed_support",
    "09": "shoulder_flexion",
    "10": "horizontal_weighted_openings",
    "11": "external_rotation_shoulders_elastic_band",
    "12": "circular_pendulum",
    "13": "shoulder_flexion",
    "14": "horizontal_weighted_openings",
    "15": "external_rotation_shoulders_elastic_band",
    "16": "circular_pendulum",
}

# These are deliberately coarse mappings into AdaptFit's first-release
# families.  The original exercise ID and name are retained in metadata, so a
# future exercise-specific head can replace this mapping without losing the
# source label.  UCO's strongest value for v2 is its exact repetition and
# expert-score supervision, not its use as a broad fitness taxonomy source.
UCO_EXERCISE_FAMILY = {
    "01": FAMILY_TO_ID["knee_extension"],
    "02": FAMILY_TO_ID["knee_extension"],
    "03": FAMILY_TO_ID["hip_flexion"],
    "04": FAMILY_TO_ID["knee_extension"],
    "05": FAMILY_TO_ID["knee_extension"],
    "06": FAMILY_TO_ID["knee_extension"],
    "07": FAMILY_TO_ID["hip_flexion"],
    "08": FAMILY_TO_ID["knee_extension"],
    "09": FAMILY_TO_ID["arm_flexion"],
    "10": FAMILY_TO_ID["reach"],
    "11": FAMILY_TO_ID["arm_flexion"],
    "12": FAMILY_TO_ID["reach"],
    "13": FAMILY_TO_ID["arm_flexion"],
    "14": FAMILY_TO_ID["reach"],
    "15": FAMILY_TO_ID["arm_flexion"],
    "16": FAMILY_TO_ID["reach"],
}


def _uco_subject_id(value: object) -> str:
    raw = str(value).strip()
    if raw.isdigit():
        return str(int(raw))
    return raw


def _uco_exercise_id(value: object) -> str:
    raw = str(value).strip()
    return raw.zfill(2) if raw.isdigit() else raw


def _uco_position(raw: object) -> str:
    value = str(raw).strip().lower()
    if value == "seated":
        return "seated"
    if value == "standing":
        return "standing"
    # The canonical v1 feature schema has no supine class.  Keep the original
    # value in metadata and use unknown rather than mislabeling it as seated.
    return "unknown"


def _uco_capability_states(side: object, body: object) -> dict[str, str]:
    states = {limb: "unknown" for limb in LIMB_NAMES}
    side_name = str(side).strip().lower()
    body_name = str(body).strip().lower()
    limb = f"{side_name}_arm" if body_name == "upper" else f"{side_name}_leg"
    if limb in states:
        states[limb] = "available"
    return states


def _uco_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _uco_score_values(value: object) -> list[float]:
    raw_values = value if isinstance(value, (list, tuple)) else [value]
    scores: list[float] = []
    for raw in raw_values:
        parsed = _uco_float(raw)
        if parsed is not None and 1.0 <= parsed <= 5.0:
            scores.append(parsed)
    return scores


def _uco_score_class(value: object) -> tuple[int, float, int] | None:
    scores = _uco_score_values(value)
    if not scores:
        return None
    mean_score = float(np.mean(scores))
    # Internal classes are zero-based; metadata retains the raw 1..5 scale.
    return int(np.clip(np.rint(mean_score), 1, 5)) - 1, mean_score, len(scores)


def _uco_frame_bounds(
    frame_ids: np.ndarray,
    init_frame: object,
    final_frame: object,
) -> tuple[int, int] | None:
    try:
        start_value = int(init_frame)
        final_value = int(final_frame)
    except (TypeError, ValueError):
        return None
    if final_value < start_value or len(frame_ids) == 0:
        return None
    if final_value < int(frame_ids[0]) or start_value > int(frame_ids[-1]):
        return None
    start = int(np.searchsorted(frame_ids, start_value, side="left"))
    end = int(np.searchsorted(frame_ids, final_value, side="right")) - 1
    if start >= len(frame_ids) or end < 0:
        return None
    start = max(0, min(start, len(frame_ids) - 1))
    end = max(0, min(end, len(frame_ids) - 1))
    if end < start:
        return None
    return start, end


def _uco_coordinates(record: dict) -> tuple[np.ndarray, np.ndarray] | None:
    frames = record.get("frames")
    if not isinstance(frames, list) or not frames:
        return None
    frame_ids: list[int] = []
    coordinates = np.full((len(frames), len(UCO_SOURCE_NAMES), 3), np.nan, dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        try:
            frame_ids.append(int(frame.get("id", frame_index)))
        except (TypeError, ValueError):
            frame_ids.append(frame_index)
        joints = frame.get("joints", {})
        if not isinstance(joints, dict):
            continue
        for source_index, source_name in enumerate(UCO_SOURCE_NAMES):
            point = joints.get(source_name)
            if isinstance(point, dict):
                values = [point.get(axis) for axis in ("x", "y", "z")]
            elif isinstance(point, (list, tuple)):
                values = list(point[:3])
            else:
                continue
            if len(values) < 3:
                continue
            parsed = [_uco_float(value) for value in values[:3]]
            if all(value is not None for value in parsed):
                coordinates[frame_index, source_index] = np.asarray(parsed, dtype=np.float32)
    if not frame_ids:
        return None
    order = np.argsort(np.asarray(frame_ids, dtype=np.int64), kind="stable")
    ids = np.asarray(frame_ids, dtype=np.int64)[order]
    return coordinates[order], ids


def _uco_boundary_labels(
    frames: int,
    frame_ids: np.ndarray,
    score_rows: list[dict],
) -> tuple[np.ndarray, int]:
    labels = np.full((frames, 2), -1.0, dtype=np.float32)
    valid_rows = 0
    for row in score_rows:
        if not isinstance(row, dict):
            continue
        bounds = _uco_frame_bounds(frame_ids, row.get("init_frame"), row.get("final_frame"))
        if bounds is None:
            continue
        if valid_rows == 0:
            labels[:, :] = 0.0
        start, end = bounds
        labels[start, 0] = 1.0
        labels[end, 1] = 1.0
        valid_rows += 1
    return labels, valid_rows


def load_ucophyrehabpp(
    root: str | Path,
    allow_composite_quality: bool = True,
) -> list[CanonicalSequence]:
    """Load UCO 3D partial poses, exact repetition spans, and expert scores.

    The adapter emits one recording-level sequence for strong boundary labels
    and one repetition-level sequence per scored span for ordinal expert-quality
    supervision when ``allow_composite_quality`` is true.  Dimension-specific
    ROM/tempo/smoothness/trunk labels remain masked because the released score
    is a composite assessment.
    """

    root = Path(root)
    metadata_path = root / "ucophyrehab2_data.jsonl"
    pose_path = root / "dataset_3d_with_angles.json"
    if not metadata_path.exists() or not pose_path.exists():
        return []

    metadata_by_key: dict[tuple[str, str], dict] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                LOGGER.warning("Skipping invalid UCO metadata line %s: %s", line_number, error)
                continue
            if not isinstance(row, dict):
                continue
            key = (_uco_subject_id(row.get("subject_id", "unknown")), _uco_exercise_id(row.get("exercise_id", "unknown")))
            metadata_by_key[key] = row

    try:
        payload = json.loads(pose_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Unable to read UCO pose file %s: %s", pose_path, error)
        return []
    records = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        LOGGER.warning("UCO pose file has no list-valued data field")
        return []

    sequences: list[CanonicalSequence] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        subject = _uco_subject_id(record.get("folder", "unknown"))
        exercise_id = _uco_exercise_id(record.get("exercise", "unknown"))
        source_meta = metadata_by_key.get((subject, exercise_id))
        if source_meta is None:
            LOGGER.warning("No UCO score metadata for subject=%s exercise=%s", subject, exercise_id)
            continue
        coordinate_result = _uco_coordinates(record)
        if coordinate_result is None:
            continue
        coordinates, frame_ids = coordinate_result
        joints, confidence, observed = _project_named_joints(
            coordinates,
            UCO_SOURCE_NAMES,
            UCO_MAP,
            treat_zero_as_missing=False,
        )
        if not observed.any():
            continue
        score_rows = source_meta.get("scores", [])
        if not isinstance(score_rows, list):
            score_rows = []
        boundaries, boundary_count = _uco_boundary_labels(len(frame_ids), frame_ids, score_rows)
        position_label = str(record.get("position", "unknown"))
        side = str(record.get("side", "unknown"))
        body = str(record.get("body", "unknown"))
        family = UCO_EXERCISE_FAMILY.get(exercise_id, FAMILY_TO_ID["other"])
        exercise_name = UCO_EXERCISE_NAMES.get(exercise_id, f"exercise_{exercise_id}")
        common_metadata = {
            "source_member": f"{subject}/exercise_{exercise_id}/dataset_3d_with_angles.json",
            "exercise_id": exercise_id,
            "exercise_name": exercise_name,
            "position_label": position_label,
            "source_position": position_label,
            "side": side,
            "body": body,
            "source_frame_start": int(frame_ids[0]),
            "source_frame_end": int(frame_ids[-1]),
            "source_coordinate_dim": 3,
            "model_coordinate_dim": 2,
            "coordinate_projection": "xy_from_3d",
            "depth_discarded": True,
            "projection_reason": "mobile_2d_contract_without_calibration",
            "pose_source": "optitrack_ground_truth_partial_landmarks",
            "repetition_count": boundary_count,
            "repetition_metadata_source": "ucophyrehab2_data.jsonl",
            "labels_are_exact_repetition_boundaries": bool(boundary_count),
            "family_mapping": "uco_exercise_id_to_adaptfit_coarse_family_v1",
        }
        sequences.append(
            _make_sequence(
                joints=joints,
                confidence=confidence,
                observed=observed,
                position=_uco_position(position_label),
                participant_id=subject,
                session_id=f"exercise_{exercise_id}",
                source_dataset="ucophyrehabpp",
                family=family,
                quality=empty_quality_labels()[0],
                quality_mask=empty_quality_labels()[1],
                label_provenance="weak",
                phase_label_source="unlabeled",
                boundary_label_source="strong" if boundary_count else "unlabeled",
                quality_label_source="unlabeled",
                metadata={
                    **common_metadata,
                    "recording_unit": "recording",
                    "expert_quality_label_status": (
                        "repetition_only" if allow_composite_quality else "disabled_by_policy"
                    ),
                },
                rep_boundary=boundaries,
                phase=np.full(len(frame_ids), -1, dtype=np.int64),
                timestamps=(frame_ids - frame_ids[0]).astype(np.float32) / 30.0,
                capability_states=_uco_capability_states(side, body),
            )
        )

        for repetition_index, row in enumerate(score_rows):
            if not isinstance(row, dict):
                continue
            bounds = _uco_frame_bounds(frame_ids, row.get("init_frame"), row.get("final_frame"))
            score_result = _uco_score_class(row.get("score"))
            if bounds is None or score_result is None:
                continue
            start, end = bounds
            if end - start + 1 < 4:
                continue
            expert_class, mean_score, rater_count = score_result
            repetition_metadata = {
                **common_metadata,
                "recording_unit": "repetition",
                "repetition_index": repetition_index,
                "source_frame_start": int(frame_ids[start]),
                "source_frame_end": int(frame_ids[end]),
                "expert_quality_score": mean_score,
                "expert_quality_score_raw": row.get("score"),
                "expert_quality_scale": "1-5",
                "expert_quality_rater_count": rater_count,
                "expert_quality_label_source": (
                    "uco_physiotherapist" if allow_composite_quality else "disabled_by_policy"
                ),
                "expert_quality_label_status": (
                    "weak_composite" if allow_composite_quality else "disabled_by_policy"
                ),
                "labels_are_exact_repetition_boundaries": True,
            }
            sequences.append(
                _make_sequence(
                    joints=joints[start : end + 1],
                    confidence=confidence[start : end + 1],
                    observed=observed[start : end + 1],
                    position=_uco_position(position_label),
                    participant_id=subject,
                    session_id=f"exercise_{exercise_id}",
                    source_dataset="ucophyrehabpp",
                    family=family,
                    quality=empty_quality_labels()[0],
                    quality_mask=empty_quality_labels()[1],
                    label_provenance="weak",
                    phase_label_source="weak_displacement",
                    boundary_label_source="unlabeled",
                    quality_label_source="unlabeled",
                    metadata=repetition_metadata,
                    rep_boundary=np.full((end - start + 1, 2), -1.0, dtype=np.float32),
                    timestamps=(frame_ids[start : end + 1] - frame_ids[start]).astype(np.float32) / 30.0,
                    expert_quality=expert_class if allow_composite_quality else -1,
                    expert_quality_mask=bool(allow_composite_quality),
                    capability_states=_uco_capability_states(side, body),
                )
            )
    return sequences


def load_canonical_npz_files(root: str | Path) -> list[CanonicalSequence]:
    sequences: list[CanonicalSequence] = []
    for path in sorted(Path(root).rglob("*.npz")):
        try:
            sequences.append(CanonicalSequence.from_npz(path))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            LOGGER.warning("Skipping invalid canonical NPZ %s: %s", path, error)
            continue
    return sequences


def load_all_sources(config: Config, project_root: Path) -> list[CanonicalSequence]:
    """Load enabled source adapters and canonical NPZ files."""

    sequences: list[CanonicalSequence] = []
    strict_source_paths = bool(config["data"].get("strict_source_paths", False))
    for source in config["data"]["sources"]:
        if not source.get("enabled", True):
            continue
        root = project_root / source["path"]
        name = source["name"]
        if strict_source_paths and not root.exists():
            raise FileNotFoundError(f"enabled source path does not exist: {root}")
        before = len(sequences)
        if name == "rehab24_6":
            sequences.extend(load_rehab24_6(root))
        elif name == "intellirehabds":
            policy = config["data"].get("label_policy", {})
            allow_boundaries = bool(
                source.get(
                    "allow_single_clip_boundaries",
                    policy.get("allow_single_clip_boundaries", False),
                )
            )
            sequences.extend(
                load_intellirehabds(root, allow_single_clip_boundaries=allow_boundaries)
            )
        elif name == "mmfit":
            sequences.extend(load_mmfit(root))
        elif name == "ul_red":
            sequences.extend(load_ul_red(root))
        elif name == "ucophyrehabpp":
            policy = config["data"].get("label_policy", {})
            sequences.extend(
                load_ucophyrehabpp(
                    root,
                    allow_composite_quality=bool(
                        policy.get("allow_uco_composite_quality", True)
                    ),
                )
            )
        elif root.exists():
            sequences.extend(load_canonical_npz_files(root))
        if strict_source_paths and len(sequences) == before:
            raise RuntimeError(f"enabled source produced no canonical sequences: {name} ({root})")
    return sequences
