"""Auditable source-projection and effective-label provenance helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _source_metadata(item: dict[str, Any]) -> dict[str, Any]:
    nested = item.get("source_metadata")
    return nested if isinstance(nested, dict) else item


def _as_dimension(value: Any, default: int | None = None) -> int | None:
    """Parse a coordinate dimension without allowing malformed metadata through."""

    if value is None:
        return default
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return None
    return dimension if dimension > 0 else None


def projection_summary(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize coordinate dimensions and validate projection provenance.

    Prepared v2 data created before the full projection contract was added may
    contain the explicit ``source_coordinate_dim=3`` and
    ``coordinate_projection=xy_from_3d`` markers, but not the newer explanatory
    fields.  That combination is safe to reconstruct for the known UCO source,
    so it is reported as ``legacy_inferred`` rather than being mistaken for a
    fully explicit record.  Any other incomplete 3D provenance remains an
    error.
    """

    projections: Counter[str] = Counter()
    source_dimensions: Counter[str] = Counter()
    projection_status: Counter[str] = Counter()
    legacy_inferred: list[int] = []
    undocumented: list[int] = []
    invalid_dimensions: list[int] = []
    for index, item in enumerate(metadata):
        source = _source_metadata(item)
        source_dim = _as_dimension(source.get("source_coordinate_dim"), 2)
        model_dim = _as_dimension(source.get("model_coordinate_dim"), 2)
        projection = str(source.get("coordinate_projection", "identity_2d"))
        if source_dim is None or model_dim is None:
            projection_status["invalid"] += 1
            invalid_dimensions.append(index)
            continue
        source_dimensions[str(source_dim)] += 1
        projections[projection] += 1
        if source_dim <= 2:
            projection_status["explicit"] += 1
            continue
        complete = (
            model_dim == 2
            and projection == "xy_from_3d"
            and source.get("depth_discarded") is True
            and source.get("projection_reason") == "mobile_2d_contract_without_calibration"
        )
        if complete:
            projection_status["explicit"] += 1
            continue

        # The existing v2 prepared tree predates the expanded metadata fields,
        # but its UCO rows retain the source dimension and projection marker.
        # Keep that tree read-only and make the compatibility decision visible.
        source_dataset = str(item.get("source_dataset", source.get("source_dataset", "")))
        known_legacy_uco = (
            source_dim == 3
            and model_dim == 2
            and projection == "xy_from_3d"
            and source_dataset == "ucophyrehabpp"
        )
        if known_legacy_uco:
            projection_status["legacy_inferred"] += 1
            legacy_inferred.append(index)
        else:
            projection_status["invalid"] += 1
            undocumented.append(index)
    if invalid_dimensions or undocumented:
        raise ValueError(
            "3D source rows require explicit xy_from_3d projection metadata; "
            f"invalid rows: {(invalid_dimensions + undocumented)[:10]}"
        )
    return {
        "source_coordinate_dimensions": dict(source_dimensions),
        "coordinate_projections": dict(projections),
        "projection_metadata_status": dict(projection_status),
        "legacy_inferred_projection_rows": legacy_inferred[:10],
        "legacy_inferred_projection_row_count": len(legacy_inferred),
        "projection_metadata_complete": not legacy_inferred,
        "depth_discarded_source_rows": sum(
            1
            for item in metadata
            if (
                _source_metadata(item).get("depth_discarded") is True
                or (
                    _as_dimension(_source_metadata(item).get("source_coordinate_dim"), 2) == 3
                    and str(_source_metadata(item).get("coordinate_projection", ""))
                    == "xy_from_3d"
                    and str(item.get("source_dataset", "")) == "ucophyrehabpp"
                )
            )
        ),
        "model_coordinate_dimension": 2,
    }
