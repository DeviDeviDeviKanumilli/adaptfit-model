"""Deterministic repetition decoder for the movement model outputs.

The neural network produces per-frame logits.  This module owns the
stateful, product-facing interpretation of those logits: ordering, temporal
debouncing, tracking abstention, reset handling, and ``WorkoutEventV1``
serialization.  It deliberately contains no learned parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Iterable, Sequence

import numpy as np

from .data.schema import PHASE_TO_ID


DECODER_VERSION = "decoder.v1"
FEATURE_SCHEMA_VERSION = "adaptfit.features.v1"


class DecoderState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    REFRACTORY = "refractory"


@dataclass(frozen=True)
class DecoderConfig:
    """Versioned deterministic decoder thresholds and temporal constraints."""

    version: str = DECODER_VERSION
    start_threshold: float = 0.65
    end_threshold: float = 0.65
    phase_threshold: float = 0.45
    tracking_floor: float = 0.60
    low_tracking_frames: int = 30
    max_timestamp_gap_ms: int = 500
    min_rep_duration_ms: int = 300
    max_rep_duration_ms: int = 15_000
    refractory_ms: int = 300
    min_phase_frames: int = 2
    require_apex: bool = True
    apex_phase_ids: tuple[int, ...] = (PHASE_TO_ID["hold"],)

    def __post_init__(self) -> None:
        if self.version != DECODER_VERSION:
            raise ValueError(f"unsupported decoder version: {self.version!r}")
        for name in ("start_threshold", "end_threshold", "phase_threshold", "tracking_floor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and strictly between 0 and 1")
        for name in (
            "low_tracking_frames",
            "max_timestamp_gap_ms",
            "min_rep_duration_ms",
            "max_rep_duration_ms",
            "refractory_ms",
            "min_phase_frames",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_rep_duration_ms >= self.max_rep_duration_ms:
            raise ValueError("min_rep_duration_ms must be smaller than max_rep_duration_ms")
        if not self.apex_phase_ids:
            raise ValueError("apex_phase_ids must contain at least one phase")
        if any(int(value) < 0 or int(value) >= 5 for value in self.apex_phase_ids):
            raise ValueError("apex_phase_ids must contain valid phase IDs")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "DecoderConfig":
        if values is None:
            return cls()
        known = {field for field in cls.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown decoder settings: {sorted(unknown)}")
        normalized = dict(values)
        if "apex_phase_ids" in normalized:
            normalized["apex_phase_ids"] = tuple(int(value) for value in normalized["apex_phase_ids"])
        return cls(**normalized)

    def to_mapping(self) -> dict[str, Any]:
        values = asdict(self)
        values["apex_phase_ids"] = list(self.apex_phase_ids)
        return values


@dataclass(frozen=True)
class DecoderFrame:
    """One model prediction at a monotonically increasing timestamp."""

    timestamp_ms: int
    start_probability: float
    end_probability: float
    phase_probabilities: tuple[float, ...]
    tracking_confidence: float

    @property
    def phase_id(self) -> int:
        return int(np.argmax(np.asarray(self.phase_probabilities, dtype=np.float64)))

    @property
    def phase_confidence(self) -> float:
        return float(self.phase_probabilities[self.phase_id])


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return value


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    logits = np.asarray(values, dtype=np.float64)
    if logits.shape != (5,) or not np.isfinite(logits).all():
        raise ValueError("phase logits must contain five finite values")
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities)
    return tuple(float(value) for value in probabilities)


def _event(
    *,
    session_id: str,
    exercise_id: str,
    variant_id: str,
    side: str,
    start_timestamp_ms: int,
    end_timestamp_ms: int,
    count_delta: int,
    event_confidence: float,
    tracking_confidence: float,
    abstention: bool,
    reason_code: str,
    model_version: str,
    feature_schema_version: str,
    decoder_version: str,
) -> dict[str, Any]:
    if start_timestamp_ms < 0 or end_timestamp_ms < start_timestamp_ms:
        raise ValueError("event timestamps must be non-negative and ordered")
    if count_delta < 0:
        raise ValueError("count_delta cannot be negative")
    return {
        "schema_version": "workout-event.v1",
        "session_id": str(session_id),
        "exercise_id": str(exercise_id),
        "variant_id": str(variant_id),
        "side": str(side),
        "start_timestamp_ms": int(start_timestamp_ms),
        "end_timestamp_ms": int(end_timestamp_ms),
        "count_delta": int(count_delta),
        "event_confidence": _probability(event_confidence, "event_confidence"),
        "tracking_confidence": _probability(tracking_confidence, "tracking_confidence"),
        "abstention": bool(abstention),
        "reason_code": str(reason_code),
        "model_version": str(model_version),
        "feature_schema_version": str(feature_schema_version),
        "decoder_version": str(decoder_version),
    }


class WorkoutEventDecoder:
    """Decode one chronological prediction stream into ``WorkoutEventV1`` events."""

    def __init__(
        self,
        config: DecoderConfig | None = None,
        *,
        model_version: str = "movement-tcn-v1",
        feature_schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> None:
        self.config = config or DecoderConfig()
        self.model_version = str(model_version)
        self.feature_schema_version = str(feature_schema_version)
        self._session_id: str | None = None
        self._exercise_id: str | None = None
        self._variant_id: str | None = None
        self._side = "unknown"
        self._state = DecoderState.IDLE
        self._start: DecoderFrame | None = None
        self._apex_frames = 0
        self._low_tracking_run = 0
        self._last_timestamp_ms: int | None = None
        self._last_event_end_ms: int | None = None

    @property
    def state(self) -> DecoderState:
        return self._state

    def reset(self) -> None:
        self._state = DecoderState.IDLE
        self._start = None
        self._apex_frames = 0
        self._low_tracking_run = 0
        self._last_timestamp_ms = None
        self._last_event_end_ms = None

    def _context_changed(self, session_id: str, exercise_id: str, variant_id: str) -> bool:
        return (
            self._session_id is not None
            and session_id != self._session_id
        ) or (
            self._exercise_id is not None
            and exercise_id != self._exercise_id
        ) or (
            self._variant_id is not None
            and variant_id != self._variant_id
        )

    def _set_context(self, session_id: str, exercise_id: str, variant_id: str, side: str) -> None:
        self._session_id = str(session_id)
        self._exercise_id = str(exercise_id)
        self._variant_id = str(variant_id)
        self._side = str(side)

    def _abstention(self, timestamp_ms: int, reason: str, tracking: float) -> dict[str, Any]:
        return _event(
            session_id=self._session_id or "unknown",
            exercise_id=self._exercise_id or "unknown",
            variant_id=self._variant_id or "unknown",
            side=self._side,
            start_timestamp_ms=timestamp_ms,
            end_timestamp_ms=timestamp_ms,
            count_delta=0,
            event_confidence=0.0,
            tracking_confidence=tracking,
            abstention=True,
            reason_code=reason,
            model_version=self.model_version,
            feature_schema_version=self.feature_schema_version,
            decoder_version=self.config.version,
        )

    def _finish_active(self, frame: DecoderFrame, reason: str, count: int) -> dict[str, Any]:
        if self._start is None:
            raise RuntimeError("cannot finish a decoder state without a start")
        start = self._start
        duration = frame.timestamp_ms - start.timestamp_ms
        confidence = float(
            np.mean(
                [
                    start.start_probability,
                    frame.end_probability,
                    max(start.tracking_confidence, frame.tracking_confidence),
                ]
            )
        )
        event = _event(
            session_id=self._session_id or "unknown",
            exercise_id=self._exercise_id or "unknown",
            variant_id=self._variant_id or "unknown",
            side=self._side,
            start_timestamp_ms=start.timestamp_ms,
            end_timestamp_ms=frame.timestamp_ms,
            count_delta=count,
            event_confidence=confidence if count else 0.0,
            tracking_confidence=min(start.tracking_confidence, frame.tracking_confidence),
            abstention=count == 0,
            reason_code=reason,
            model_version=self.model_version,
            feature_schema_version=self.feature_schema_version,
            decoder_version=self.config.version,
        )
        self._last_event_end_ms = frame.timestamp_ms
        self._state = DecoderState.REFRACTORY
        self._start = None
        self._apex_frames = 0
        self._low_tracking_run = 0
        return event

    def _frame_from_values(
        self,
        timestamp_ms: int,
        start_probability: float,
        end_probability: float,
        phase: Sequence[float],
        tracking_confidence: float,
        *,
        phase_is_logits: bool,
    ) -> DecoderFrame:
        timestamp_ms = int(timestamp_ms)
        if timestamp_ms < 0:
            raise ValueError("timestamps must be non-negative")
        if phase_is_logits:
            phase_probabilities = _softmax(phase)
        else:
            phase_probabilities = tuple(_probability(value, "phase_probability") for value in phase)
            if len(phase_probabilities) != 5 or not math.isclose(sum(phase_probabilities), 1.0, abs_tol=1e-4):
                raise ValueError("phase probabilities must contain five values summing to one")
        return DecoderFrame(
            timestamp_ms=timestamp_ms,
            start_probability=_probability(start_probability, "start_probability"),
            end_probability=_probability(end_probability, "end_probability"),
            phase_probabilities=phase_probabilities,
            tracking_confidence=_probability(tracking_confidence, "tracking_confidence"),
        )

    def step(
        self,
        *,
        timestamp_ms: int,
        start_probability: float,
        end_probability: float,
        phase: Sequence[float],
        tracking_confidence: float,
        session_id: str,
        exercise_id: str,
        variant_id: str = "default",
        side: str = "unknown",
        phase_is_logits: bool = False,
    ) -> list[dict[str, Any]]:
        """Consume one prediction and return zero or more product events."""

        frame = self._frame_from_values(
            timestamp_ms,
            start_probability,
            end_probability,
            phase,
            tracking_confidence,
            phase_is_logits=phase_is_logits,
        )
        events: list[dict[str, Any]] = []
        context_changed = self._context_changed(str(session_id), str(exercise_id), str(variant_id))
        if context_changed:
            if self._state == DecoderState.ACTIVE and self._start is not None:
                events.append(self._finish_active(frame, "context_reset", 0))
            self.reset()
        self._set_context(str(session_id), str(exercise_id), str(variant_id), str(side))

        if self._last_timestamp_ms is not None:
            delta = frame.timestamp_ms - self._last_timestamp_ms
            if delta <= 0:
                # Repeated overlapping-window predictions are ignored rather
                # than creating a second event or silently reordering state.
                return events
            if delta > self.config.max_timestamp_gap_ms:
                if self._state == DecoderState.ACTIVE and self._start is not None:
                    events.append(self._finish_active(frame, "timestamp_gap", 0))
                self.reset()
                self._set_context(str(session_id), str(exercise_id), str(variant_id), str(side))
                events.append(self._abstention(frame.timestamp_ms, "timestamp_gap", frame.tracking_confidence))
        self._last_timestamp_ms = frame.timestamp_ms

        if frame.tracking_confidence < self.config.tracking_floor:
            self._low_tracking_run += 1
        else:
            self._low_tracking_run = 0
            if self._state == DecoderState.PAUSED:
                self._state = DecoderState.ACTIVE if self._start is not None else DecoderState.IDLE

        if self._low_tracking_run >= self.config.low_tracking_frames:
            if self._state in {DecoderState.ACTIVE, DecoderState.REFRACTORY}:
                if self._state == DecoderState.ACTIVE and self._start is not None:
                    events.append(self._finish_active(frame, "low_tracking", 0))
                self._state = DecoderState.PAUSED
                events.append(self._abstention(frame.timestamp_ms, "low_tracking", frame.tracking_confidence))
            elif self._state == DecoderState.IDLE:
                self._state = DecoderState.PAUSED
                events.append(self._abstention(frame.timestamp_ms, "low_tracking", frame.tracking_confidence))
            return events

        if self._state == DecoderState.PAUSED:
            return events

        if self._state == DecoderState.REFRACTORY:
            if self._last_event_end_ms is None or frame.timestamp_ms - self._last_event_end_ms >= self.config.refractory_ms:
                self._state = DecoderState.IDLE
            else:
                return events

        if self._state == DecoderState.IDLE:
            if (
                frame.start_probability >= self.config.start_threshold
                and frame.phase_id != PHASE_TO_ID["unknown"]
                and frame.phase_confidence >= self.config.phase_threshold
                and frame.tracking_confidence >= self.config.tracking_floor
            ):
                self._start = frame
                self._apex_frames = 0
                self._state = DecoderState.ACTIVE
            return events

        if self._state != DecoderState.ACTIVE or self._start is None:
            return events

        if frame.phase_id in self.config.apex_phase_ids and frame.phase_confidence >= self.config.phase_threshold:
            self._apex_frames += 1

        duration_ms = frame.timestamp_ms - self._start.timestamp_ms
        if duration_ms > self.config.max_rep_duration_ms:
            events.append(self._finish_active(frame, "partial_timeout", 0))
            return events

        if frame.end_probability < self.config.end_threshold:
            return events
        if duration_ms < self.config.min_rep_duration_ms:
            events.append(self._finish_active(frame, "partial_too_short", 0))
            return events
        if self.config.require_apex and self._apex_frames < self.config.min_phase_frames:
            events.append(self._finish_active(frame, "partial_without_apex", 0))
            return events
        events.append(self._finish_active(frame, "decoded_rep", 1))
        return events

    def decode_arrays(
        self,
        *,
        timestamps_ms: Iterable[int | float],
        start_probabilities: Sequence[float],
        end_probabilities: Sequence[float],
        phase: Sequence[Sequence[float]],
        tracking_confidences: Sequence[float],
        session_id: str,
        exercise_id: str,
        variant_id: str = "default",
        side: str = "unknown",
        phase_is_logits: bool = False,
    ) -> list[dict[str, Any]]:
        """Decode aligned arrays, rejecting shape mismatches before mutation."""

        timestamps = list(timestamps_ms)
        starts = list(start_probabilities)
        ends = list(end_probabilities)
        phases = list(phase)
        trackings = list(tracking_confidences)
        lengths = {len(timestamps), len(starts), len(ends), len(phases), len(trackings)}
        if len(lengths) != 1:
            raise ValueError("decoder arrays must have equal lengths")
        events: list[dict[str, Any]] = []
        for values in zip(timestamps, starts, ends, phases, trackings):
            events.extend(
                self.step(
                    timestamp_ms=int(round(float(values[0]))),
                    start_probability=float(values[1]),
                    end_probability=float(values[2]),
                    phase=values[3],
                    tracking_confidence=float(values[4]),
                    session_id=session_id,
                    exercise_id=exercise_id,
                    variant_id=variant_id,
                    side=side,
                    phase_is_logits=phase_is_logits,
                )
            )
        return events


def boundary_events_from_targets(
    boundary: np.ndarray,
    timestamps_ms: Sequence[int | float],
) -> list[tuple[int, int]]:
    """Return paired start/end timestamps from binary boundary targets."""

    values = np.asarray(boundary)
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2 or len(timestamps) != values.shape[0]:
        raise ValueError("boundary targets must have shape [time, 2] aligned to timestamps")
    starts = list(np.flatnonzero(values[:, 0] >= 0.5))
    ends = list(np.flatnonzero(values[:, 1] >= 0.5))
    pairs: list[tuple[int, int]] = []
    end_index = 0
    for start in starts:
        while end_index < len(ends) and ends[end_index] <= start:
            end_index += 1
        if end_index >= len(ends):
            break
        pairs.append((int(timestamps[start]), int(timestamps[ends[end_index]])))
        end_index += 1
    return pairs


def match_event_boundaries(
    predicted: Sequence[dict[str, Any]],
    target: Sequence[tuple[int, int]],
    tolerance_ms: int = 250,
) -> dict[str, float]:
    """One-to-one match decoded events against target start/end pairs."""

    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be non-negative")
    predicted_pairs = [
        (int(item["start_timestamp_ms"]), int(item["end_timestamp_ms"]))
        for item in predicted
        if int(item.get("count_delta", 0)) > 0 and not item.get("abstention", False)
    ]
    used: set[int] = set()
    matched = 0
    absolute_latency: list[float] = []
    for start, end in predicted_pairs:
        candidates = [
            (index, target_start, target_end)
            for index, (target_start, target_end) in enumerate(target)
            if index not in used
            and abs(start - target_start) <= tolerance_ms
            and abs(end - target_end) <= tolerance_ms
        ]
        if not candidates:
            continue
        index, target_start, target_end = min(
            candidates,
            key=lambda item: abs(start - item[1]) + abs(end - item[2]),
        )
        used.add(index)
        matched += 1
        absolute_latency.append(float(end - target_end))
    predicted_count = len(predicted_pairs)
    target_count = len(target)
    false_events = predicted_count - matched
    missed = target_count - matched
    precision = matched / predicted_count if predicted_count else 0.0
    recall = matched / target_count if target_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted_count": float(predicted_count),
        "target_count": float(target_count),
        "matched_count": float(matched),
        "false_event_count": float(false_events),
        "missed_event_count": float(missed),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_end_latency_ms": float(np.mean(absolute_latency)) if absolute_latency else float("nan"),
    }


__all__ = [
    "DECODER_VERSION",
    "DecoderConfig",
    "DecoderFrame",
    "DecoderState",
    "WorkoutEventDecoder",
    "boundary_events_from_targets",
    "match_event_boundaries",
]
