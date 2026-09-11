"""Regenerate an isolated prepared root after adding UL-RED 3Rep boundaries.

The feature arrays in a prepared root are immutable for this correction.  To
avoid requiring another full 15 GB extraction on a nearly-full development
volume, this command hard-links those unchanged arrays into a new root and
rewrites only the boundary arrays, per-window metadata, and provenance index.
The parent root remains untouched because every rewritten file is unlinked
from the new root before it is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .src.data.adapters import load_ul_red


SPLITS = ("train", "validation", "test")
BOUNDARY_METADATA_FIELDS = (
    "boundary_label_source",
    "boundary_source_member",
    "labels_are_explicit_boundaries",
    "boundary_label_status",
    "boundary_frame_indexing",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clone_with_hardlinks(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=os.link)


def _replace_json(path: Path, value: Any) -> None:
    path.unlink()
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _boundary_lookup(source_root: Path) -> dict[tuple[str, str], Any]:
    lookup: dict[tuple[str, str], Any] = {}
    for sequence in load_ul_red(source_root):
        if sequence.metadata.get("boundary_label_source") != "strong":
            continue
        key = (
            str(sequence.metadata.get("source_archive", "")),
            str(sequence.metadata.get("source_member", "")),
        )
        if not all(key):
            continue
        if key in lookup:
            raise ValueError(f"Duplicate explicit UL-RED boundary source: {key}")
        lookup[key] = sequence
    return lookup


def _rewrite_split(
    parent_root: Path,
    output_root: Path,
    split: str,
    boundary_lookup: dict[tuple[str, str], Any],
) -> tuple[int, int, set[str]]:
    parent_boundary = np.load(parent_root / split / "boundary.npy", mmap_mode="r")
    output_boundary_path = output_root / split / "boundary.npy"
    output_boundary_path.unlink()
    output_boundary = np.lib.format.open_memmap(
        output_boundary_path,
        mode="w+",
        dtype=np.float32,
        shape=parent_boundary.shape,
    )
    output_boundary[:] = parent_boundary[:]

    parent_metadata_path = parent_root / f"{split}.jsonl"
    output_metadata_path = output_root / f"{split}.jsonl"
    output_metadata_path.unlink()
    explicit_windows = 0
    explicit_events = 0
    boundary_sequence_ids: set[str] = set()
    with parent_metadata_path.open("r", encoding="utf-8") as source, output_metadata_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for row_index, line in enumerate(source):
            item = json.loads(line)
            if np.any(np.asarray(parent_boundary[row_index]) >= 0.0):
                boundary_sequence_ids.add(str(item["sequence_id"]))
            if item.get("source_dataset") == "ul_red":
                source_metadata = item.get("source_metadata", {})
                key = (
                    str(source_metadata.get("source_archive", "")),
                    str(source_metadata.get("source_member", "")),
                )
                sequence = boundary_lookup.get(key)
                if sequence is not None:
                    start = int(item["window_start"])
                    end = int(item["window_end"])
                    labels = np.asarray(sequence.rep_boundary, dtype=np.float32)
                    if start < 0 or end <= start or end > len(labels):
                        raise ValueError(
                            f"Window {row_index} has invalid bounds {start}:{end} for {key}"
                        )
                    local = labels[start:end]
                    if not np.all((local >= 0.0) & (local <= 1.0)):
                        raise ValueError(f"Invalid UL-RED boundary labels for {key}")
                    output_boundary[row_index, : end - start] = local
                    explicit_windows += 1
                    explicit_events += int(np.count_nonzero(local > 0.0))
                    boundary_sequence_ids.add(str(item["sequence_id"]))
                    item["boundary_label_source"] = "strong"
                    updated_source_metadata = dict(source_metadata)
                    for field in BOUNDARY_METADATA_FIELDS:
                        if field in sequence.metadata:
                            updated_source_metadata[field] = sequence.metadata[field]
                    item["source_metadata"] = updated_source_metadata
            destination.write(json.dumps(item, sort_keys=True) + "\n")
    output_boundary.flush()
    del output_boundary
    return explicit_windows, explicit_events, boundary_sequence_ids


def regenerate(
    *,
    source_root: Path,
    parent_root: Path,
    output_root: Path,
    parent_artifacts_root: Path,
    output_artifacts_root: Path,
) -> dict[str, Any]:
    if not source_root.exists() or not parent_root.exists():
        raise FileNotFoundError("UL-RED source and parent prepared root must exist")
    _clone_with_hardlinks(parent_root, output_root)
    boundary_lookup = _boundary_lookup(source_root)
    if not boundary_lookup:
        raise RuntimeError("No explicit UL-RED 3Rep boundaries were found")

    explicit_windows = 0
    explicit_events = 0
    boundary_sequence_ids: set[str] = set()
    for split in SPLITS:
        windows, events, sequence_ids = _rewrite_split(
            parent_root, output_root, split, boundary_lookup
        )
        explicit_windows += windows
        explicit_events += events
        boundary_sequence_ids.update(sequence_ids)

    index_path = output_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["label_coverage_by_sequence"]["boundary"] = len(boundary_sequence_ids)
    index["prepared_root_parent"] = str(parent_root)
    index["prepared_root_regeneration"] = "hardlink_immutable_arrays_plus_ulred_boundary_overlay"
    index["ulred_explicit_boundary_sequence_count"] = len(boundary_lookup)
    index["ulred_explicit_boundary_window_count"] = explicit_windows
    index["ulred_explicit_boundary_event_count"] = explicit_events
    index["ulred_boundary_source_revision"] = "markerless/3Rep_csv_zero_based_to_amc_one_based"
    _replace_json(index_path, index)

    output_artifacts_root.mkdir(parents=True, exist_ok=False)
    for name in ("feature_schema.json", "normalization_stats.npz"):
        source = parent_artifacts_root / name
        if not source.exists():
            raise FileNotFoundError(f"Parent artifact contract is missing: {source}")
        shutil.copy2(source, output_artifacts_root / name)
    provenance = {
        "schema_version": "adaptfit.ulred-boundary-overlay.v1",
        "source_root": str(source_root),
        "parent_prepared_root": str(parent_root),
        "output_prepared_root": str(output_root),
        "parent_index_sha256": _sha256(parent_root / "index.json"),
        "parent_artifacts_root": str(parent_artifacts_root),
        "output_artifacts_root": str(output_artifacts_root),
        "explicit_sequence_count": len(boundary_lookup),
        "explicit_window_count": explicit_windows,
        "explicit_event_count": explicit_events,
        "hardlinked_unchanged_arrays": True,
    }
    (output_root / "overlay_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_artifacts_root / "prepared_overlay_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-artifacts-root", type=Path, required=True)
    parser.add_argument("--output-artifacts-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            regenerate(
                source_root=args.source_root,
                parent_root=args.parent_root,
                output_root=args.output_root,
                parent_artifacts_root=args.parent_artifacts_root,
                output_artifacts_root=args.output_artifacts_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
