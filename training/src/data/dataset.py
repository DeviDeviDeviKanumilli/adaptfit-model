"""PyTorch datasets for prepared fixed-length windows."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


PREPARED_ARRAY_NAMES = (
    "features",
    "phase",
    "boundary",
    "family",
    "quality",
    "quality_mask",
    "label_weights",
    "expert_quality",
    "expert_quality_mask",
    "expert_label_weight",
    "tracking_target",
    "time_mask",
)

def _metadata_phase_policy(item: dict[str, Any]) -> tuple[bool, str]:
    """Return whether weak phase labels are trustworthy for one window."""

    nested = item.get("source_metadata")
    source = nested if isinstance(nested, dict) else item
    phase_source = str(item.get("phase_label_source", source.get("phase_label_source", "unknown")))
    weak = phase_source == "weak_displacement" or bool(source.get("labels_are_weak_phase", False))
    repetitions = source.get("recording_repetitions", item.get("recording_repetitions", 1))
    try:
        repetitions = int(repetitions)
    except (TypeError, ValueError):
        repetitions = 1
    recording_level = bool(
        source.get("labels_are_recording_level", item.get("labels_are_recording_level", False))
    )
    if weak and recording_level and repetitions > 1:
        return False, "multi_rep_weak_phase"
    return True, "available"


def effective_phase_label_status(item: dict[str, Any]) -> tuple[bool, str]:
    """Public wrapper used by reports and prepared-data policy checks."""

    return _metadata_phase_policy(item)


def phase_policy_from_metadata(
    metadata_path: str | Path,
    expected_count: int | None = None,
    allow_weak_phase_on_multi_rep_recordings: bool = False,
) -> tuple[np.ndarray, Counter[str]]:
    """Build an effective phase-label mask without loading feature arrays."""

    path = Path(metadata_path)
    available: list[bool] = []
    reasons: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
            if not isinstance(item, dict):
                raise ValueError(f"Metadata row {line_number} in {path} must be an object")
            usable, reason = effective_phase_label_status(item)
            if allow_weak_phase_on_multi_rep_recordings and reason == "multi_rep_weak_phase":
                usable, reason = True, "allowed_by_policy"
            available.append(usable)
            reasons[reason] += 1
    if expected_count is not None and len(available) != int(expected_count):
        raise ValueError(
            f"Metadata count ({len(available)}) does not match prepared window count "
            f"({int(expected_count)}) for {path.name}"
        )
    return np.asarray(available, dtype=bool), reasons


def prepared_split_path(processed_root: str | Path, split: str) -> Path:
    """Resolve the current memmap layout, falling back to legacy NPZ output."""

    root = Path(processed_root)
    directory = root / split
    if directory.is_dir():
        return directory
    return root / f"{split}.npz"


def prepared_metadata_path(processed_root: str | Path, split: str) -> Path:
    """Return the JSONL metadata path paired with a prepared split."""

    return Path(processed_root) / f"{split}.jsonl"


def load_prepared_metadata(
    processed_root: str | Path,
    split: str,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Load and validate one metadata row per prepared window."""

    path = prepared_metadata_path(processed_root, split)
    if not path.exists():
        raise FileNotFoundError(f"Prepared metadata is missing: {path}")
    metadata: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
            if not isinstance(item, dict):
                raise ValueError(f"Metadata row {line_number} in {path} must be an object")
            metadata.append(item)
    if expected_count is not None and len(metadata) != int(expected_count):
        raise ValueError(
            f"Metadata count ({len(metadata)}) does not match prepared window count "
            f"({int(expected_count)}) for {split}"
        )
    return metadata


class PreparedWindowDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        validate_features: bool = True,
        return_numpy: bool = False,
        metadata_path: str | Path | None = None,
        allow_weak_phase_on_multi_rep_recordings: bool = False,
    ):
        path = Path(path)
        self.path = path
        self.validate_features_on_open = bool(validate_features)
        self.return_numpy = bool(return_numpy)
        self.storage_format = "npy_memmap" if path.is_dir() else "npz_legacy"
        if path.is_dir():
            self.storage_format = "npy_memmap"
            arrays = {}
            optional_arrays = {
                "tracking_target",
                "expert_quality",
                "expert_quality_mask",
                "expert_label_weight",
            }
            for name in PREPARED_ARRAY_NAMES:
                array_path = path / f"{name}.npy"
                if name in optional_arrays and not array_path.exists():
                    continue
                arrays[name] = np.load(array_path, mmap_mode="r", allow_pickle=False)
            count = int(arrays["family"].shape[0])
            if "expert_quality" not in arrays:
                arrays["expert_quality"] = np.full(count, -1, dtype=np.int64)
            if "expert_quality_mask" not in arrays:
                arrays["expert_quality_mask"] = np.zeros(count, dtype=bool)
            if "expert_label_weight" not in arrays:
                arrays["expert_label_weight"] = np.zeros(count, dtype=np.float32)
            if "tracking_target" not in arrays and "time_mask" in arrays:
                arrays["tracking_target"] = np.where(arrays["time_mask"], 1.0, -1.0).astype(np.float32)
        else:
            with np.load(path, allow_pickle=False) as data:
                arrays = {
                    name: data[name].copy()
                    for name in PREPARED_ARRAY_NAMES
                    if name in data
                }
                if "label_weights" not in arrays:
                    arrays["label_weights"] = np.ones((data["family"].shape[0], 4), dtype=np.float32)
                count = int(data["family"].shape[0])
                if "expert_quality" not in arrays:
                    arrays["expert_quality"] = np.full(count, -1, dtype=np.int64)
                if "expert_quality_mask" not in arrays:
                    arrays["expert_quality_mask"] = np.zeros(count, dtype=bool)
                if "expert_label_weight" not in arrays:
                    arrays["expert_label_weight"] = np.zeros(count, dtype=np.float32)
                if "tracking_target" not in arrays:
                    arrays["tracking_target"] = np.where(data["time_mask"], 1.0, -1.0).astype(np.float32)
        self.features = arrays["features"]
        self.phase = arrays["phase"]
        self.boundary = arrays["boundary"]
        self.family = arrays["family"]
        self.quality = arrays["quality"]
        self.quality_mask = arrays["quality_mask"]
        self.label_weights = arrays["label_weights"]
        self.expert_quality = arrays["expert_quality"]
        self.expert_quality_mask = arrays["expert_quality_mask"]
        self.expert_label_weight = arrays["expert_label_weight"]
        self.tracking_target = arrays["tracking_target"]
        self.time_mask = arrays["time_mask"]
        count = int(self.features.shape[0])
        self.phase_available = np.ones(count, dtype=bool)
        self.phase_policy_reasons: Counter[str] = Counter()
        if metadata_path is None:
            metadata_path = (
                path.parent / f"{path.name}.jsonl" if path.is_file()
                else path.parent / f"{path.name}.jsonl"
            )
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.exists():
            self.phase_available, self.phase_policy_reasons = phase_policy_from_metadata(
                self.metadata_path,
                expected_count=count,
                allow_weak_phase_on_multi_rep_recordings=(
                    allow_weak_phase_on_multi_rep_recordings
                ),
            )
        else:
            self.phase_policy_reasons["metadata_missing"] = count
        self.valid_lengths = np.asarray(self.time_mask, dtype=bool).sum(axis=1).astype(np.int64)
        self.validate(check_features_finite=self.validate_features_on_open)

    def validate(self, *, check_features_finite: bool = True) -> None:
        if self.features.ndim != 3:
            raise ValueError(f"features must have shape [count, frames, features], got {self.features.shape}")
        count, frames, features = self.features.shape
        if count == 0 or frames == 0 or features == 0:
            raise ValueError("prepared dataset cannot have an empty count, time axis, or feature axis")
        if check_features_finite and not np.isfinite(self.features).all():
            raise ValueError("prepared features must be finite")
        if self.phase.shape != (count, frames):
            raise ValueError("phase shape does not match features")
        if self.phase_available.shape != (count,):
            raise ValueError("phase availability shape does not match features")
        if self.boundary.shape != (count, frames, 2):
            raise ValueError("boundary shape does not match features")
        if self.family.shape != (count,):
            raise ValueError("family shape does not match features")
        if self.quality.shape != (count, 4) or self.quality_mask.shape != (count, 4):
            raise ValueError("quality shape does not match features")
        if self.label_weights.shape != (count, 4):
            raise ValueError("label_weights shape does not match features")
        if not np.isfinite(self.label_weights).all() or (self.label_weights < 0).any():
            raise ValueError("label_weights must be finite and non-negative")
        if self.time_mask.shape != (count, frames):
            raise ValueError("time_mask shape does not match features")
        if self.tracking_target.shape != (count, frames):
            raise ValueError("tracking_target shape does not match features")
        in_time = self.time_mask.astype(bool)
        if not in_time.any(axis=1).all():
            raise ValueError("each prepared window must contain at least one valid frame")
        if in_time.shape[1] > 1 and np.any(in_time[:, 1:] & ~in_time[:, :-1]):
            raise ValueError("prepared time_mask must be right-padded")
        if np.any(in_time & ~np.isfinite(self.tracking_target)):
            raise ValueError("tracking targets in valid frames must be finite")
        if np.any(in_time & (self.tracking_target < -1.0)):
            raise ValueError("tracking targets must be -1 or within [0, 1]")
        tracking_valid = in_time & (self.tracking_target >= 0.0)
        if np.any(tracking_valid & (self.tracking_target > 1.0)):
            raise ValueError("tracking targets must be within [0, 1]")
        if np.any((~self.time_mask.astype(bool)) & (self.tracking_target != -1.0)):
            raise ValueError("padded frames must not contain tracking targets")
        if np.any((self.family < -1) | (self.family >= 6)):
            raise ValueError("family labels must be -1 or within the v1 class range")
        if np.any((self.phase < -1) | (self.phase >= 5)):
            raise ValueError("phase labels must be -1 or within the v1 class range")
        if np.any(~np.isin(self.boundary, (-1.0, 0.0, 1.0))):
            raise ValueError("boundary labels must be -1, 0, or 1")
        quality_finite = np.isfinite(self.quality)
        if np.any(self.quality_mask & ~quality_finite):
            raise ValueError("masked quality labels must be finite")
        if np.any(quality_finite & ((self.quality < 0.0) | (self.quality > 1.0))):
            raise ValueError("quality labels must be within [0, 1]")
        if self.expert_quality.shape != (count,) or self.expert_quality_mask.shape != (count,):
            raise ValueError("expert quality targets and masks must have shape [count]")
        if self.expert_label_weight.shape != (count,):
            raise ValueError("expert_label_weight must have shape [count]")
        if not np.isfinite(self.expert_label_weight).all() or (self.expert_label_weight < 0).any():
            raise ValueError("expert_label_weight must be finite and non-negative")
        if np.any((self.expert_quality < -1) | (self.expert_quality >= 5)):
            raise ValueError("expert_quality targets must be -1 or within the 0..4 class range")
        if np.any(self.expert_quality_mask & (self.expert_quality < 0)):
            raise ValueError("expert quality masks cannot be true for unavailable targets")
        if np.any((~self.expert_quality_mask) & (self.expert_quality != -1)):
            raise ValueError("unmasked expert quality targets must be -1")
        padding = ~self.time_mask.astype(bool)
        if np.any((self.phase != -1) & padding):
            raise ValueError("padded frames must not contain phase labels")
        if np.any((self.boundary != -1.0) & padding[..., None]):
            raise ValueError("padded frames must not contain boundary labels")

    def validate_feature_storage(self, chunk_rows: int = 4096) -> dict[str, int]:
        """Scan feature storage in bounded chunks for an explicit integrity check."""

        chunk_rows = max(1, int(chunk_rows))
        checked = 0
        for start in range(0, len(self), chunk_rows):
            end = min(len(self), start + chunk_rows)
            values = np.asarray(self.features[start:end])
            if not np.isfinite(values).all():
                raise ValueError(f"prepared features must be finite in rows [{start}, {end})")
            checked += end - start
        return {"rows_checked": checked, "chunks": (checked + chunk_rows - 1) // chunk_rows}

    def __getstate__(self) -> dict[str, Any]:
        """Drop open memmap handles before a macOS worker process is spawned."""

        state = self.__dict__.copy()
        if self.storage_format == "npy_memmap":
            for name in PREPARED_ARRAY_NAMES:
                value = state.get(name)
                if isinstance(value, np.memmap):
                    state[name] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.storage_format != "npy_memmap":
            return
        directory = Path(self.path)
        optional_arrays = {
            "tracking_target",
            "expert_quality",
            "expert_quality_mask",
            "expert_label_weight",
        }
        for name in PREPARED_ARRAY_NAMES:
            array_path = directory / f"{name}.npy"
            if name in optional_arrays and not array_path.exists():
                continue
            setattr(self, name, np.load(array_path, mmap_mode="r", allow_pickle=False))

    def __len__(self) -> int:
        return int(len(self.features))

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.return_numpy:
            phase = np.asarray(self.phase[index], dtype=np.int64)
            if not self.phase_available[index]:
                phase = phase.copy()
                phase[:] = -1
            return {
                "features": np.asarray(self.features[index]),
                "phase": phase,
                "boundary": np.asarray(self.boundary[index]),
                "family": np.asarray(self.family[index], dtype=np.int64),
                "quality": np.asarray(self.quality[index]),
                "quality_mask": np.asarray(self.quality_mask[index]),
                "label_weights": np.asarray(self.label_weights[index]),
                "expert_quality": np.asarray(self.expert_quality[index], dtype=np.int64),
                "expert_quality_mask": np.asarray(self.expert_quality_mask[index]),
                "expert_label_weight": np.asarray(self.expert_label_weight[index]),
                "tracking_target": np.asarray(self.tracking_target[index]),
                "time_mask": np.asarray(self.time_mask[index]),
                "sample_index": np.asarray(index, dtype=np.int64),
            }

        phase = np.asarray(self.phase[index], dtype=np.int64).copy()
        if not self.phase_available[index]:
            phase[:] = -1
        return {
            # The compatibility path intentionally returns independent tensors.
            # Training uses return_numpy=True and materializes one batch instead.
            "features": torch.from_numpy(np.asarray(self.features[index], dtype=np.float32).copy()),
            "phase": torch.from_numpy(phase),
            "boundary": torch.from_numpy(np.asarray(self.boundary[index], dtype=np.float32).copy()),
            "family": torch.tensor(self.family[index], dtype=torch.long),
            "quality": torch.from_numpy(np.asarray(self.quality[index], dtype=np.float32).copy()),
            "quality_mask": torch.from_numpy(np.asarray(self.quality_mask[index], dtype=bool).copy()),
            "label_weights": torch.from_numpy(np.asarray(self.label_weights[index], dtype=np.float32).copy()),
            "expert_quality": torch.tensor(self.expert_quality[index], dtype=torch.long),
            "expert_quality_mask": torch.tensor(self.expert_quality_mask[index], dtype=torch.bool),
            "expert_label_weight": torch.tensor(self.expert_label_weight[index], dtype=torch.float32),
            "tracking_target": torch.from_numpy(np.asarray(self.tracking_target[index], dtype=np.float32).copy()),
            "time_mask": torch.from_numpy(np.asarray(self.time_mask[index], dtype=bool).copy()),
        }


def prepared_collate(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Materialize one batch and trim right-padding before model execution."""

    if not samples:
        raise ValueError("Cannot collate an empty prepared batch")
    names = (
        "features",
        "phase",
        "boundary",
        "family",
        "quality",
        "quality_mask",
        "label_weights",
        "expert_quality",
        "expert_quality_mask",
        "expert_label_weight",
        "tracking_target",
        "time_mask",
        "sample_index",
    )
    arrays = {name: np.stack([np.asarray(item[name]) for item in samples], axis=0) for name in names}
    valid_lengths = arrays["time_mask"].astype(bool).sum(axis=1)
    max_length = int(valid_lengths.max())
    for name in ("features", "phase", "boundary", "tracking_target", "time_mask"):
        arrays[name] = arrays[name][:, :max_length]
    return {
        name: torch.from_numpy(values)
        for name, values in arrays.items()
    }


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batch windows with identical valid lengths to avoid TCN padding work."""

    def __init__(
        self,
        dataset: PreparedWindowDataset,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int = 42,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        buckets: defaultdict[int, list[int]] = defaultdict(list)
        for index, length in enumerate(dataset.valid_lengths.tolist()):
            if int(length) <= 0:
                raise ValueError("Prepared windows must have at least one valid frame")
            buckets[int(length)].append(index)
        self.buckets = {length: tuple(indices) for length, indices in buckets.items()}

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.buckets.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.buckets.values():
            selected = list(indices)
            if self.shuffle:
                rng.shuffle(selected)
            batches.extend(
                [selected[start : start + self.batch_size] for start in range(0, len(selected), self.batch_size)]
            )
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches
