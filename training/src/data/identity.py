"""Versioned logical sequence identities shared by preparation and evaluation.

Window IDs are deliberately derived from source provenance rather than from a
small subset of labels.  This prevents repetitions, gestures, or sets that
share a participant/session/family label from being merged during splitting
or evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


SEQUENCE_IDENTITY_VERSION = "adaptfit.sequence.v2"


_SOURCE_FIELDS: dict[str, tuple[str, ...]] = {
    "rehab24_6": (
        "video_id",
        "exercise_id",
        "source_member",
        "camera",
        "orientation",
        "source_frame_start",
        "source_frame_end",
    ),
    "intellirehabds": (
        "source_member",
        "gesture_id",
        "repetition_number",
        "position_label",
        "correctness",
    ),
    "mmfit": (
        "workout_id",
        "activity",
        "set_index",
        "set_start_frame",
        "set_end_frame",
    ),
    "ul_red": (
        "source_archive",
        "source_member",
        "exercise_name",
        "recording_repetitions",
        "pace_protocol",
    ),
    "ucophyrehabpp": (
        "recording_unit",
        "source_member",
        "exercise_id",
        "source_frame_start",
        "source_frame_end",
        "repetition_index",
        "side",
        "position_label",
    ),
    "procedural_seed": (
        "procedural",
        "family_name",
        "repetition_number",
        "active_side",
    ),
}

_GENERIC_FIELDS: tuple[str, ...] = (
    "source_member",
    "source_file",
    "video_id",
    "recording_id",
    "trial_id",
    "workout_id",
    "exercise_id",
    "gesture_id",
    "activity",
    "set_index",
    "repetition_number",
    "source_frame_start",
    "source_frame_end",
    "set_start_frame",
    "set_end_frame",
    "position_label",
    "position",
    "recording_unit",
    "repetition_index",
    "side",
    "source_position",
)


def _normalise(value: Any) -> Any:
    """Convert metadata values to deterministic JSON-compatible values."""

    if isinstance(value, Mapping):
        return {
            str(key): _normalise(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value)
    return str(value)


def _selected_fields(metadata: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for field in fields:
        value = metadata.get(field)
        if value is None or value == "":
            continue
        selected[field] = _normalise(value)
    return selected


def sequence_identity_payload(
    *,
    source_dataset: Any,
    participant_id: Any,
    session_id: Any,
    source_metadata: Mapping[str, Any] | None = None,
    position: Any = None,
    legacy_sequence_id: Any = None,
) -> dict[str, Any]:
    """Build the canonical, inspectable identity payload for one sequence.

    ``source_metadata`` is optional so the evaluator can still read older
    prepared metadata.  Legacy ``sequence_id`` is used only as a last-resort
    fallback when no source-specific identity fields are available.
    """

    metadata = source_metadata if isinstance(source_metadata, Mapping) else {}
    source = str(source_dataset or metadata.get("source_dataset") or "unknown")
    participant = str(participant_id or metadata.get("participant_id") or "unknown")
    session = str(session_id or metadata.get("session_id") or "unknown")
    selected = _selected_fields(metadata, _SOURCE_FIELDS.get(source, _GENERIC_FIELDS))

    if metadata.get("synthetic"):
        variant_index = metadata.get("synthetic_variant_index")
        if variant_index is not None:
            selected["synthetic_variant_index"] = _normalise(variant_index)

    if not selected and legacy_sequence_id not in (None, ""):
        selected["legacy_sequence_id"] = str(legacy_sequence_id)

    payload: dict[str, Any] = {
        "identity_version": SEQUENCE_IDENTITY_VERSION,
        "source_dataset": source,
        "participant_id": participant,
        "session_id": session,
        "source_identity": selected,
    }
    if position not in (None, ""):
        payload["position"] = str(position)
    return payload


def sequence_identity_id(
    *,
    source_dataset: Any,
    participant_id: Any,
    session_id: Any,
    source_metadata: Mapping[str, Any] | None = None,
    position: Any = None,
    legacy_sequence_id: Any = None,
) -> str:
    """Return a stable string ID suitable for JSONL metadata and grouping."""

    return json.dumps(
        sequence_identity_payload(
            source_dataset=source_dataset,
            participant_id=participant_id,
            session_id=session_id,
            source_metadata=source_metadata,
            position=position,
            legacy_sequence_id=legacy_sequence_id,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def sequence_identity_for_metadata(item: Mapping[str, Any]) -> str:
    """Reconstruct v2 identity from a prepared metadata row.

    Recomputing from ``source_metadata`` is intentional: old rows can contain
    a legacy ``sequence_id`` while still carrying enough source metadata to
    recover the corrected logical grouping.
    """

    source_metadata = item.get("source_metadata")
    if not isinstance(source_metadata, Mapping):
        source_metadata = item
    return sequence_identity_id(
        source_dataset=item.get("source_dataset", source_metadata.get("source_dataset")),
        participant_id=item.get("participant_id", source_metadata.get("participant_id")),
        session_id=item.get("session_id", source_metadata.get("session_id")),
        source_metadata=source_metadata,
        position=item.get("position", source_metadata.get("position")),
        legacy_sequence_id=item.get("sequence_id"),
    )


def sequence_identity_for_sequence(sequence: Any) -> str:
    """Build v2 identity from a ``CanonicalSequence``-like object."""

    return sequence_identity_id(
        source_dataset=getattr(sequence, "source_dataset", None),
        participant_id=getattr(sequence, "participant_id", None),
        session_id=getattr(sequence, "session_id", None),
        source_metadata=getattr(sequence, "metadata", None),
        position=getattr(sequence, "position", None),
    )


def identity_source_signature(item: Mapping[str, Any]) -> str:
    """Return a full metadata signature for collision diagnostics."""

    source_metadata = item.get("source_metadata")
    if not isinstance(source_metadata, Mapping):
        source_metadata = item
    return json.dumps(_normalise(dict(source_metadata)), sort_keys=True, separators=(",", ":"))
