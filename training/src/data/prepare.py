"""Prepare public skeleton sources into normalized fixed-length training windows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import yaml

from .adapters import load_all_sources
from .dataset import PREPARED_ARRAY_NAMES
from .identity import SEQUENCE_IDENTITY_VERSION, sequence_identity_for_sequence
from .procedural import make_procedural_sequences
from .schema import (
    CanonicalSequence,
    PHASE_TO_ID,
    expert_quality_loss_weight,
    label_loss_weights,
)
from .synthetic import generate_augmentations
from .temporal import resample_sequence
from ..config import Config
from ..features.anatomy import (
    continuous_feature_indices,
    extract_features,
    feature_dimensions,
    feature_schema,
    tracking_confidence_target,
)


@dataclass
class SequenceEntry:
    sequence: CanonicalSequence
    split: str
    synthetic: bool = False


@dataclass
class WindowRecord:
    features: np.ndarray
    phase: np.ndarray
    boundary: np.ndarray
    family: int
    quality: np.ndarray
    quality_mask: np.ndarray
    label_weights: np.ndarray
    expert_quality: int
    expert_quality_mask: bool
    expert_label_weight: float
    tracking_target: np.ndarray
    time_mask: np.ndarray
    metadata: dict


def _sequence_id(sequence: CanonicalSequence) -> str:
    """Return the versioned source-aware logical sequence ID."""

    return sequence_identity_for_sequence(sequence)


def _group_id(sequence: CanonicalSequence) -> str:
    return f"{sequence.source_dataset}:{sequence.participant_id}"


def assign_splits(
    sequences: list[CanonicalSequence],
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    required_test_sources: Iterable[str] | None = None,
    required_test_positions: Iterable[str] | None = None,
    enforce_required_test_coverage: bool = False,
) -> list[SequenceEntry]:
    """Assign every participant group to exactly one split.

    Optional coverage constraints reserve deterministic participant groups in
    the test split before the remaining groups are allocated.  The defaults
    preserve the original unconstrained split behavior for legacy configs.
    """

    groups = sorted({_group_id(sequence) for sequence in sequences})
    if len(groups) < 3:
        raise ValueError("At least three participant groups are required for a group split")
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(groups) * (1.0 - train_fraction - validation_fraction))))
    validation_count = max(1, int(round(len(groups) * validation_fraction)))
    if test_count + validation_count >= len(groups):
        test_count = 1
        validation_count = 1

    required_sources = {
        str(value).strip() for value in (required_test_sources or ()) if str(value).strip()
    }
    required_positions = {
        str(value).strip() for value in (required_test_positions or ()) if str(value).strip()
    }
    group_sequences: dict[str, list[CanonicalSequence]] = {group: [] for group in groups}
    for sequence in sequences:
        group_sequences[_group_id(sequence)].append(sequence)

    reserved: list[str] = []

    def reserve_group(candidates: Iterable[str], requirement: str) -> None:
        for group in shuffled:
            if group in reserved or group not in candidates:
                continue
            reserved.append(group)
            return
        if enforce_required_test_coverage:
            raise ValueError(f"Required test coverage is unavailable for {requirement!r}")

    if required_sources:
        for source in sorted(required_sources):
            candidates = [
                group
                for group, group_items in group_sequences.items()
                if any(item.source_dataset == source for item in group_items)
            ]
            reserve_group(candidates, f"source={source}")
    if required_positions:
        for position in sorted(required_positions):
            # Prefer real source groups when procedural examples happen to
            # satisfy the same position requirement.  This keeps the report
            # honest about whether coverage is public-data or synthetic.
            candidates = [
                group
                for group, group_items in group_sequences.items()
                if any(item.position == position for item in group_items)
            ]
            real_candidates = [
                group
                for group in candidates
                if any(item.source_dataset != "procedural_seed" for item in group_sequences[group])
            ]
            reserve_group(real_candidates or candidates, f"position={position}")

    if len(reserved) > test_count:
        if enforce_required_test_coverage:
            raise ValueError(
                "Required test coverage needs more participant groups than the configured test split"
            )
        reserved = reserved[:test_count]

    test_groups = set(reserved)
    for group in shuffled:
        if len(test_groups) >= test_count:
            break
        test_groups.add(group)
    remaining = [group for group in shuffled if group not in test_groups]
    validation_groups = set(remaining[:validation_count])

    entries: list[SequenceEntry] = []
    for sequence in sequences:
        group = _group_id(sequence)
        split = "test" if group in test_groups else "validation" if group in validation_groups else "train"
        entries.append(SequenceEntry(sequence=sequence, split=split, synthetic=False))
    return entries


def split_summary(entries: Iterable[SequenceEntry]) -> dict[str, dict[str, object]]:
    """Summarize participant, source, and position coverage for each split."""

    summary: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        selected = [entry for entry in entries if entry.split == split]
        summary[split] = {
            "sequence_count": sum(not entry.synthetic for entry in selected),
            "entry_count": len(selected),
            "synthetic_entry_count": sum(entry.synthetic for entry in selected),
            "participant_group_count": len({_group_id(entry.sequence) for entry in selected}),
            "participant_groups": sorted({_group_id(entry.sequence) for entry in selected}),
            "source_counts": dict(Counter(entry.sequence.source_dataset for entry in selected)),
            "position_counts": dict(Counter(entry.sequence.position for entry in selected)),
            "identity_versions": sorted(
                {SEQUENCE_IDENTITY_VERSION for entry in selected if not entry.synthetic or entry.synthetic}
            ),
        }
    return summary


def add_synthetic_entries(entries: list[SequenceEntry], config: Config) -> list[SequenceEntry]:
    settings = config["data"].get("synthetic", {})
    if not settings.get("enabled", True):
        return entries
    rng = np.random.default_rng(int(settings.get("random_seed", 42)))
    augmented: list[SequenceEntry] = []
    for entry in entries:
        if entry.split != "train":
            continue
        variants = generate_augmentations(
            entry.sequence,
            rng=rng,
            copies=int(settings.get("copies_per_sequence", 2)),
            occlusion_probability=float(settings.get("occlusion_probability", 0.2)),
            noise_std=float(settings.get("noise_std", 0.008)),
            reduced_rom_scale=float(settings.get("reduced_rom_scale", 0.75)),
        )
        augmented.extend(SequenceEntry(sequence=variant, split="train", synthetic=True) for variant in variants)
    return entries + augmented


def _update_stats(
    mean_sum: np.ndarray,
    square_sum: np.ndarray,
    count: np.ndarray,
    values: np.ndarray,
    continuous_indices: np.ndarray,
) -> np.ndarray:
    continuous = values[:, continuous_indices]
    finite = np.isfinite(continuous)
    safe = np.where(finite, continuous, 0.0).astype(np.float64)
    mean_sum += safe.sum(axis=0)
    square_sum += (safe * safe).sum(axis=0)
    count += finite.sum(axis=0)
    return count


def _finalize_normalization(
    mean_sum: np.ndarray,
    square_sum: np.ndarray,
    count: np.ndarray,
    feature_dim: int,
    continuous_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(count > 0):
        raise ValueError("No training frames were available for normalization")
    safe_count = np.maximum(count, 1.0)
    mean = mean_sum / safe_count
    variance = np.maximum(square_sum / safe_count - mean * mean, 1e-8)
    std = np.sqrt(variance)
    full_mean = np.zeros(feature_dim, dtype=np.float32)
    full_std = np.ones(feature_dim, dtype=np.float32)
    full_mean[continuous_indices] = mean.astype(np.float32)
    full_std[continuous_indices] = std.astype(np.float32)
    return full_mean, full_std


def fit_normalization(
    entries: Iterable[SequenceEntry],
    include_diagnostics: bool = False,
    fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    feature_dim, continuous_dim = feature_dimensions(include_diagnostics)
    continuous_indices = continuous_feature_indices(include_diagnostics)
    mean_sum = np.zeros(continuous_dim, dtype=np.float64)
    square_sum = np.zeros(continuous_dim, dtype=np.float64)
    count = np.zeros(continuous_dim, dtype=np.float64)
    for entry in entries:
        if entry.split != "train":
            continue
        features = extract_features(
            entry.sequence,
            fps=fps,
            include_diagnostics=include_diagnostics,
        )
        count = _update_stats(mean_sum, square_sum, count, features, continuous_indices)
    return _finalize_normalization(
        mean_sum,
        square_sum,
        count,
        feature_dim,
        continuous_indices,
    )


def _window_starts(frames: int, window: int, stride: int) -> list[int]:
    if frames <= window:
        return [0]
    starts = list(range(0, frames - window + 1, max(1, stride)))
    final_start = frames - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def sequence_windows(
    sequence: CanonicalSequence,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    window_frames: int,
    window_stride: int,
    include_diagnostics: bool = False,
    fps: float = 30.0,
    raw_features: np.ndarray | None = None,
) -> list[WindowRecord]:
    feature_dim, _ = feature_dimensions(include_diagnostics)
    continuous_indices = continuous_feature_indices(include_diagnostics)
    if normalization_mean.shape != (feature_dim,) or normalization_std.shape != (feature_dim,):
        raise ValueError(
            f"normalization arrays must have shape [{feature_dim}] for the selected feature schema"
        )
    if raw_features is None:
        raw_features = extract_features(
            sequence,
            fps=fps,
            include_diagnostics=include_diagnostics,
        )
    features = (raw_features - normalization_mean) / normalization_std
    noncontinuous_indices = np.setdiff1d(
        np.arange(feature_dim, dtype=np.int64),
        continuous_indices,
        assume_unique=True,
    )
    features[:, noncontinuous_indices] = raw_features[:, noncontinuous_indices]
    frames = len(features)
    records: list[WindowRecord] = []
    raw_tracking_target = tracking_confidence_target(sequence)
    for start in _window_starts(frames, window_frames, window_stride):
        end = min(frames, start + window_frames)
        length = end - start
        padded_features = np.zeros((window_frames, feature_dim), dtype=np.float32)
        padded_phase = np.full(window_frames, -1, dtype=np.int64)
        padded_boundary = np.full((window_frames, 2), -1.0, dtype=np.float32)
        padded_expert_quality = int(sequence.expert_quality) if sequence.expert_quality_mask else -1
        padded_tracking = np.full(window_frames, -1.0, dtype=np.float32)
        time_mask = np.zeros(window_frames, dtype=bool)
        padded_features[:length] = features[start:end]
        padded_phase[:length] = sequence.phase[start:end]
        padded_boundary[:length] = sequence.rep_boundary[start:end]
        padded_tracking[:length] = raw_tracking_target[start:end]
        time_mask[:length] = True
        records.append(
            WindowRecord(
                features=padded_features,
                phase=padded_phase,
                boundary=padded_boundary,
                family=sequence.family,
                quality=sequence.quality.copy(),
                quality_mask=sequence.quality_mask.copy(),
                label_weights=label_loss_weights(sequence),
                expert_quality=padded_expert_quality,
                expert_quality_mask=bool(sequence.expert_quality_mask),
                expert_label_weight=expert_quality_loss_weight(sequence),
                tracking_target=padded_tracking,
                time_mask=time_mask,
                metadata={
                    "source_dataset": sequence.source_dataset,
                    "participant_id": sequence.participant_id,
                    "session_id": sequence.session_id,
                    "sequence_id": _sequence_id(sequence),
                    "sequence_identity_version": SEQUENCE_IDENTITY_VERSION,
                    "position": sequence.position,
                    "family": sequence.family,
                    "synthetic": bool(sequence.metadata.get("synthetic", False)),
                    "label_provenance": sequence.metadata.get("label_provenance", "unknown"),
                    "phase_label_source": sequence.metadata.get("phase_label_source", "unknown"),
                    "boundary_label_source": sequence.metadata.get("boundary_label_source", "unknown"),
                    "quality_label_source": sequence.metadata.get("quality_label_source", "unknown"),
                    "expert_quality_label_source": sequence.metadata.get(
                        "expert_quality_label_source", "unlabeled"
                    ),
                    "expert_quality_label_status": sequence.metadata.get(
                        "expert_quality_label_status", "unavailable"
                    ),
                    "expert_quality_scale": sequence.metadata.get("expert_quality_scale", "1-5"),
                    "expert_quality_rater_count": sequence.metadata.get(
                        "expert_quality_rater_count", 0
                    ),
                    "tracking_target_source": "self_supervised_observability",
                    "window_start": start,
                    "window_end": end,
                    "source_metadata": sequence.metadata,
                },
            )
        )
    return records


def _write_split(
    entries: list[SequenceEntry],
    output_root: Path,
    split: str,
    normalization_mean: np.ndarray | None,
    normalization_std: np.ndarray | None,
    window_frames: int,
    window_stride: int,
    include_diagnostics: bool = False,
    fps: float = 30.0,
    feature_storage_dtype: str = "float32",
    fit_stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> int:
    feature_dim, _ = feature_dimensions(include_diagnostics)
    record_count = sum(
        len(_window_starts(entry.sequence.frames, window_frames, window_stride))
        for entry in entries
    )
    if record_count == 0:
        raise ValueError(f"No windows produced for {split} split")
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    feature_dtype = {
        "float16": np.float16,
        "float32": np.float32,
    }.get(str(feature_storage_dtype))
    if feature_dtype is None:
        raise ValueError("feature_storage_dtype must be 'float16' or 'float32'")
    shapes_and_dtypes = {
        "features": ((record_count, window_frames, feature_dim), feature_dtype),
        "phase": ((record_count, window_frames), np.int64),
        "boundary": ((record_count, window_frames, 2), np.float32),
        "family": ((record_count,), np.int64),
        "quality": ((record_count, 4), np.float32),
        "quality_mask": ((record_count, 4), bool),
        "label_weights": ((record_count, 4), np.float32),
        "expert_quality": ((record_count,), np.int64),
        "expert_quality_mask": ((record_count,), bool),
        "expert_label_weight": ((record_count,), np.float32),
        "tracking_target": ((record_count, window_frames), np.float32),
        "time_mask": ((record_count, window_frames), bool),
    }
    arrays = {
        name: np.lib.format.open_memmap(
            split_root / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        for name, (shape, dtype) in shapes_and_dtypes.items()
    }
    offset = 0
    if normalization_mean is None:
        normalization_mean = np.zeros(feature_dim, dtype=np.float32)
    if normalization_std is None:
        normalization_std = np.ones(feature_dim, dtype=np.float32)
    with (output_root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            raw_features = extract_features(
                entry.sequence,
                fps=fps,
                include_diagnostics=include_diagnostics,
            )
            if fit_stats is not None and split == "train":
                _update_stats(
                    fit_stats[0],
                    fit_stats[1],
                    fit_stats[2],
                    raw_features,
                    fit_stats[3],
                )
            records = sequence_windows(
                entry.sequence,
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
                window_frames=window_frames,
                window_stride=window_stride,
                include_diagnostics=include_diagnostics,
                fps=fps,
                raw_features=raw_features,
            )
            for record in records:
                arrays["features"][offset] = record.features
                arrays["phase"][offset] = record.phase
                arrays["boundary"][offset] = record.boundary
                arrays["family"][offset] = record.family
                arrays["quality"][offset] = record.quality
                arrays["quality_mask"][offset] = record.quality_mask
                arrays["label_weights"][offset] = record.label_weights
                arrays["expert_quality"][offset] = record.expert_quality
                arrays["expert_quality_mask"][offset] = record.expert_quality_mask
                arrays["expert_label_weight"][offset] = record.expert_label_weight
                arrays["tracking_target"][offset] = record.tracking_target
                arrays["time_mask"][offset] = record.time_mask
                handle.write(json.dumps(record.metadata, sort_keys=True) + "\n")
                offset += 1
    for array in arrays.values():
        array.flush()
    if offset != record_count:
        raise RuntimeError(f"Wrote {offset} records for {split}, expected {record_count}")
    return record_count


def _normalize_feature_memmaps(
    processed_root: Path,
    splits: Iterable[str],
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    continuous_indices: np.ndarray,
    chunk_rows: int = 2048,
) -> None:
    """Normalize feature storage in bounded chunks after the single write pass."""

    chunk_rows = max(1, int(chunk_rows))
    for split in splits:
        feature_path = processed_root / split / "features.npy"
        features = np.load(feature_path, mmap_mode="r+", allow_pickle=False)
        for start in range(0, len(features), chunk_rows):
            end = min(len(features), start + chunk_rows)
            block = np.asarray(features[start:end], dtype=np.float32).copy()
            block[..., continuous_indices] = (
                block[..., continuous_indices] - normalization_mean[continuous_indices]
            ) / normalization_std[continuous_indices]
            features[start:end] = block.astype(features.dtype, copy=False)
        features.flush()


def _has_material_files(path: Path) -> bool:
    return path.exists() and any(item.name != ".gitkeep" for item in path.iterdir())


def prepare_dataset(
    config: Config,
    project_root: Path,
    *,
    replace_existing: bool = False,
) -> dict:
    include_diagnostics = bool(config["features"].get("include_diagnostics", False))
    feature_dim, continuous_dim = feature_dimensions(include_diagnostics)
    data_config = config["data"]
    target_fps = float(data_config.get("target_fps", 30.0))
    max_interpolation_gap = int(data_config.get("max_interpolation_gap", 5))
    feature_storage_dtype = str(data_config.get("feature_storage_dtype", "float32"))
    sequences = load_all_sources(config, project_root)
    sequences.extend(make_procedural_sequences(config))
    sequences = [
        resample_sequence(
            sequence,
            target_fps=target_fps,
            max_interpolation_gap=max_interpolation_gap,
        )
        for sequence in sequences
    ]
    if not sequences:
        raise RuntimeError(
            "No sequences were loaded. Stage the configured public datasets under data/raw or add canonical NPZ files."
        )
    entries = assign_splits(
        sequences,
        seed=int(config["project"].get("seed", 42)),
        train_fraction=float(config["data"]["split"]["train"]),
        validation_fraction=float(config["data"]["split"]["validation"]),
        required_test_sources=data_config["split"].get("required_test_sources", []),
        required_test_positions=data_config["split"].get("required_test_positions", []),
        enforce_required_test_coverage=bool(
            data_config["split"].get("enforce_required_test_coverage", False)
        ),
    )
    entries = add_synthetic_entries(entries, config)
    processed_root = project_root / data_config.get("processed_root", "data/processed")
    artifacts_root = project_root / config["project"].get("artifacts_root", "artifacts")
    if _has_material_files(processed_root) and not replace_existing:
        raise FileExistsError(
            f"Prepared dataset already exists at {processed_root}; "
            "use an isolated processed_root or pass replace_existing=True explicitly"
        )
    staging_root = processed_root.with_name(f".{processed_root.name}.staging")
    if staging_root.exists():
        raise FileExistsError(
            f"Preparation staging directory already exists at {staging_root}; "
            "inspect or remove it before retrying"
        )
    processed_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=False)
    placeholder = processed_root / ".gitkeep"
    staged_placeholder = staging_root / ".gitkeep"
    if placeholder.exists() and not _has_material_files(processed_root):
        placeholder.replace(staged_placeholder)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    entries_by_split = {
        split: [entry for entry in entries if entry.split == split]
        for split in ("train", "validation", "test")
    }
    continuous_indices = continuous_feature_indices(include_diagnostics)
    mean_sum = np.zeros(continuous_dim, dtype=np.float64)
    square_sum = np.zeros(continuous_dim, dtype=np.float64)
    normalization_count = np.zeros(continuous_dim, dtype=np.float64)
    counts = {}
    for split, split_entries in entries_by_split.items():
        counts[split] = _write_split(
            split_entries,
            staging_root,
            split,
            normalization_mean=None,
            normalization_std=None,
            window_frames=int(data_config["window_frames"]),
            window_stride=int(data_config["window_stride"]),
            include_diagnostics=include_diagnostics,
            fps=target_fps,
            feature_storage_dtype=feature_storage_dtype,
            fit_stats=(mean_sum, square_sum, normalization_count, continuous_indices),
        )
    mean, std = _finalize_normalization(
        mean_sum,
        square_sum,
        normalization_count,
        feature_dim,
        continuous_indices,
    )
    _normalize_feature_memmaps(
        staging_root,
        ("train", "validation", "test"),
        mean,
        std,
        continuous_indices,
    )
    np.savez_compressed(artifacts_root / "normalization_stats.npz", mean=mean, std=std)
    (artifacts_root / "feature_schema.json").write_text(
        json.dumps(
            feature_schema(include_diagnostics=include_diagnostics, fps=target_fps),
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (artifacts_root / "training_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "sequence_count": len(sequences),
        "entry_count_with_synthetic": len(entries),
        "window_counts": counts,
        "feature_dimension": feature_dim,
        "continuous_dimension": continuous_dim,
        "feature_schema_version": feature_schema(include_diagnostics)["version"],
        "target_fps": target_fps,
        "max_interpolation_gap": max_interpolation_gap,
        "window_frames": int(data_config["window_frames"]),
        "window_stride": int(data_config["window_stride"]),
        "storage_format": "npy_memmap",
        "feature_storage_dtype": feature_storage_dtype,
        "normalization_pass": "single_extraction_chunked_in_place",
        "prepared_arrays": list(PREPARED_ARRAY_NAMES),
        "label_policy": dict(data_config.get("label_policy", {})),
        "sequence_identity_version": SEQUENCE_IDENTITY_VERSION,
        "target_population_coverage": {
            "amputee_participants": False,
            "limb_difference_participants": False,
            "wheelchair_user_participants": False,
            "wheelchair_position_public_proxy": bool(
                any(sequence.position == "wheelchair" for sequence in sequences)
            ),
            "interpretation": "Public-data position/proxy coverage is not target-population validation.",
        },
        "split_summary": split_summary(entries),
        "sources": sorted({sequence.source_dataset for sequence in sequences}),
        "positions": sorted({sequence.position for sequence in sequences}),
        "sequence_counts_by_source": dict(
            Counter(sequence.source_dataset for sequence in sequences)
        ),
        "sequence_counts_by_position": dict(
            Counter(sequence.position for sequence in sequences)
        ),
        "label_coverage_by_sequence": {
            "family": sum(sequence.family >= 0 for sequence in sequences),
            "phase": sum(bool(np.any(sequence.phase >= 0)) for sequence in sequences),
            "boundary": sum(bool(np.any(sequence.rep_boundary >= 0)) for sequence in sequences),
            "quality": sum(bool(sequence.quality_mask.any()) for sequence in sequences),
            "expert_quality": sum(
                bool(sequence.expert_quality_mask) for sequence in sequences
            ),
        },
    }
    (staging_root / "index.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if processed_root.exists():
        if _has_material_files(processed_root):
            if not replace_existing:
                raise FileExistsError(f"Prepared dataset appeared during staging: {processed_root}")
            backup_root = processed_root.with_name(
                f"{processed_root.name}.backup-{time.time_ns()}"
            )
            processed_root.replace(backup_root)
        else:
            processed_root.rmdir()
    staging_root.replace(processed_root)
    return summary
