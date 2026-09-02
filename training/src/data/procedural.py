"""Small procedural movement seed set for exact v1 exercise families.

This is not a substitute for human recordings. It exists so every v1 movement
family has a labeled example while public datasets are sparse or use different
exercise vocabularies.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .schema import (
    CANONICAL_INDEX,
    FAMILY_NAMES,
    CanonicalSequence,
    PHASE_TO_ID,
    empty_quality_labels,
    rep_boundary_labels,
)


def _base_skeleton() -> np.ndarray:
    points = np.zeros((33, 2), dtype=np.float32)
    values = {
        "nose": (0.0, -1.05),
        "left_shoulder": (-0.28, -0.72),
        "right_shoulder": (0.28, -0.72),
        "left_elbow": (-0.48, -0.40),
        "right_elbow": (0.48, -0.40),
        "left_wrist": (-0.60, -0.08),
        "right_wrist": (0.60, -0.08),
        "left_pinky": (-0.65, -0.02),
        "right_pinky": (0.65, -0.02),
        "left_index": (-0.64, -0.08),
        "right_index": (0.64, -0.08),
        "left_thumb": (-0.62, -0.04),
        "right_thumb": (0.62, -0.04),
        "left_hip": (-0.18, 0.0),
        "right_hip": (0.18, 0.0),
        "left_knee": (-0.20, 0.48),
        "right_knee": (0.20, 0.48),
        "left_ankle": (-0.22, 0.92),
        "right_ankle": (0.22, 0.92),
        "left_heel": (-0.25, 0.98),
        "right_heel": (0.25, 0.98),
        "left_foot_index": (-0.18, 1.02),
        "right_foot_index": (0.18, 1.02),
        "left_ear": (-0.10, -1.05),
        "right_ear": (0.10, -1.05),
    }
    for name, value in values.items():
        points[CANONICAL_INDEX[name]] = value
    for name in ("left_eye_inner", "left_eye", "left_eye_outer", "mouth_left"):
        points[CANONICAL_INDEX[name]] = (-0.04, -1.08)
    for name in ("right_eye_inner", "right_eye", "right_eye_outer", "mouth_right"):
        points[CANONICAL_INDEX[name]] = (0.04, -1.08)
    return points


def _phase_labels(frames: int) -> np.ndarray:
    phase = np.full(frames, PHASE_TO_ID["unknown"], dtype=np.int64)
    phase[: max(1, int(frames * 0.06))] = PHASE_TO_ID["rest"]
    turn = int(frames * 0.50)
    phase[int(frames * 0.06) : turn] = PHASE_TO_ID["concentric"]
    phase[turn : min(frames, turn + max(1, int(frames * 0.05)))] = PHASE_TO_ID["hold"]
    phase[min(frames, turn + max(1, int(frames * 0.05))) : max(1, int(frames * 0.94))] = PHASE_TO_ID["eccentric"]
    phase[max(1, int(frames * 0.94)) :] = PHASE_TO_ID["rest"]
    return phase


def _apply_motion(points: np.ndarray, family: str, side: str, cycle: float, amplitude: float) -> None:
    sign = -1.0 if side == "left" else 1.0
    if family == "arm_flexion":
        elbow = CANONICAL_INDEX[f"{side}_elbow"]
        wrist = CANONICAL_INDEX[f"{side}_wrist"]
        index = CANONICAL_INDEX[f"{side}_index"]
        pinky = CANONICAL_INDEX[f"{side}_pinky"]
        thumb = CANONICAL_INDEX[f"{side}_thumb"]
        points[elbow] += np.asarray((-sign * amplitude * 0.15 * cycle, -amplitude * 0.05 * cycle))
        points[wrist] += np.asarray((-sign * amplitude * 0.40 * cycle, -amplitude * cycle))
        points[index] = points[wrist] + np.asarray((sign * 0.05, -0.02))
        points[pinky] = points[wrist] + np.asarray((sign * 0.03, 0.03))
        points[thumb] = points[wrist] + np.asarray((-sign * 0.02, 0.02))
    elif family == "row_pull":
        elbow = CANONICAL_INDEX[f"{side}_elbow"]
        wrist = CANONICAL_INDEX[f"{side}_wrist"]
        index = CANONICAL_INDEX[f"{side}_index"]
        points[elbow] += np.asarray((-sign * amplitude * 0.50 * cycle, amplitude * 0.05 * cycle))
        points[wrist] += np.asarray((-sign * amplitude * 0.75 * cycle, amplitude * 0.03 * cycle))
        points[index] = points[wrist]
    elif family == "knee_extension":
        knee = CANONICAL_INDEX[f"{side}_knee"]
        ankle = CANONICAL_INDEX[f"{side}_ankle"]
        heel = CANONICAL_INDEX[f"{side}_heel"]
        foot = CANONICAL_INDEX[f"{side}_foot_index"]
        points[knee] += np.asarray((sign * amplitude * 0.10 * cycle, -amplitude * 0.03 * cycle))
        points[ankle] += np.asarray((sign * amplitude * 0.45 * cycle, -amplitude * 0.08 * cycle))
        points[heel] = points[ankle] + np.asarray((-sign * 0.03, 0.06))
        points[foot] = points[ankle] + np.asarray((sign * 0.05, 0.06))
    elif family == "hip_flexion":
        knee = CANONICAL_INDEX[f"{side}_knee"]
        ankle = CANONICAL_INDEX[f"{side}_ankle"]
        heel = CANONICAL_INDEX[f"{side}_heel"]
        foot = CANONICAL_INDEX[f"{side}_foot_index"]
        points[knee] += np.asarray((sign * amplitude * 0.25 * cycle, -amplitude * 0.55 * cycle))
        points[ankle] += np.asarray((sign * amplitude * 0.40 * cycle, -amplitude * 0.65 * cycle))
        points[heel] = points[ankle] + np.asarray((-sign * 0.03, 0.06))
        points[foot] = points[ankle] + np.asarray((sign * 0.05, 0.06))
    elif family == "reach":
        elbow = CANONICAL_INDEX[f"{side}_elbow"]
        wrist = CANONICAL_INDEX[f"{side}_wrist"]
        index = CANONICAL_INDEX[f"{side}_index"]
        points[elbow] += np.asarray((sign * amplitude * 0.20 * cycle, -amplitude * 0.10 * cycle))
        points[wrist] += np.asarray((sign * amplitude * 0.75 * cycle, -amplitude * 0.35 * cycle))
        points[index] = points[wrist] + np.asarray((sign * 0.04, -0.02))


def make_procedural_sequences(config: Config) -> list[CanonicalSequence]:
    settings = config["data"].get("synthetic_seed", {})
    if not settings.get("enabled", True):
        return []
    rng = np.random.default_rng(int(settings.get("random_seed", 4242)))
    participants = int(settings.get("participant_count", 12))
    repetitions = int(settings.get("exercises_per_participant", 5))
    label_policy = config["data"].get("label_policy", {})
    allow_procedural_quality = bool(label_policy.get("allow_procedural_quality", False))
    families = tuple(FAMILY_NAMES[1:])
    sequences: list[CanonicalSequence] = []
    for participant in range(participants):
        position = "wheelchair" if participant % 3 == 0 else "seated"
        for family in families:
            for repetition in range(max(1, repetitions)):
                frames = int(rng.integers(96, 161))
                side = "left" if (participant + repetition + len(family)) % 2 == 0 else "right"
                amplitude = float(rng.uniform(0.12, 0.28))
                base = _base_skeleton()
                joints = np.repeat(base[None, :, :], frames, axis=0)
                for frame in range(frames):
                    progress = frame / max(1, frames - 1)
                    cycle = 0.5 - 0.5 * np.cos(2.0 * np.pi * progress)
                    points = base.copy()
                    _apply_motion(points, family, side, float(cycle), amplitude)
                    jitter = rng.normal(0.0, 0.004, points.shape).astype(np.float32)
                    joints[frame] = points + jitter
                observed = np.ones((frames, 33), dtype=bool)
                confidence = np.ones((frames, 33), dtype=np.float32)
                if allow_procedural_quality:
                    quality = np.ones(4, dtype=np.float32)
                    quality[3] = 0.0
                    quality_mask = np.ones(4, dtype=bool)
                    quality_label_source = "procedural_template"
                else:
                    quality, quality_mask = empty_quality_labels()
                    quality_label_source = "unlabeled"
                capability_states = {
                    "left_arm": "available",
                    "right_arm": "available",
                    "left_leg": "available",
                    "right_leg": "available",
                }
                if family in {"arm_flexion", "row_pull", "reach"}:
                    capability_states[f"{'right' if side == 'left' else 'left'}_arm"] = "absent"
                else:
                    capability_states[f"{'right' if side == 'left' else 'left'}_leg"] = "absent"
                sequences.append(
                    CanonicalSequence(
                        joints=joints,
                        pose_confidence=confidence,
                        observed_mask=observed,
                        timestamps=np.arange(frames, dtype=np.float32) / 30.0,
                        position=position,
                        participant_id=f"procedural_{participant:03d}",
                        session_id=f"procedural_{participant:03d}_{family}_{repetition:02d}",
                        source_dataset="procedural_seed",
                        family=FAMILY_NAMES.index(family),
                        phase=_phase_labels(frames),
                        rep_boundary=rep_boundary_labels(frames),
                        quality=quality,
                        quality_mask=quality_mask,
                        capability_states=capability_states,
                        metadata={
                            "procedural": True,
                            "active_side": side,
                            "family_name": family,
                            "repetition_number": repetition,
                            "labels_are_procedural": True,
                            "label_provenance": "procedural",
                            "phase_label_source": "procedural_template",
                            "boundary_label_source": "procedural_template",
                            "quality_label_source": quality_label_source,
                            "procedural_quality_enabled": allow_procedural_quality,
                        },
                    )
                )
    return sequences
