from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.src.config import load_config
from training.src.data.dataset import (
    LengthBucketBatchSampler,
    PREPARED_ARRAY_NAMES,
    PreparedWindowDataset,
    load_prepared_metadata,
    prepared_collate,
)
from training.src.data.prepare import (
    SequenceEntry,
    _window_starts,
    fit_normalization,
    prepare_dataset,
    sequence_windows,
)
from training.src.data.procedural import make_procedural_sequences
from training.src.data.schema import CanonicalSequence
from training.src.data.synthetic import generate_augmentations
from training.src.features.anatomy import FEATURE_DIM, extract_features
from training.src.features.anatomy import continuous_feature_indices
from training.src.runner import collect_predictions
from training.tests.support import CONFIG_PATH


class CanonicalIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)
        cls.sequence = make_procedural_sequences(cls.config)[0]

    def test_empty_sequence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            CanonicalSequence(
                joints=np.zeros((0, 33, 2), dtype=np.float32),
                pose_confidence=np.zeros((0, 33), dtype=np.float32),
                observed_mask=np.zeros((0, 33), dtype=bool),
                timestamps=np.zeros(0, dtype=np.float32),
            )

    def test_timestamps_must_be_finite_and_strictly_increasing(self) -> None:
        duplicate = self.sequence.timestamps.copy()
        duplicate[1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.sequence.clone(timestamps=duplicate)

        nonfinite = self.sequence.timestamps.copy()
        nonfinite[1] = np.nan
        with self.assertRaisesRegex(ValueError, "timestamps must be finite"):
            self.sequence.clone(timestamps=nonfinite)

    def test_observed_landmarks_must_be_finite(self) -> None:
        joints = self.sequence.joints.copy()
        joints[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "observed joints"):
            self.sequence.clone(joints=joints)

    def test_unobserved_landmarks_may_be_missing(self) -> None:
        joint_index = 15
        joints = self.sequence.joints.copy()
        confidence = self.sequence.pose_confidence.copy()
        observed = self.sequence.observed_mask.copy()
        joints[:, joint_index] = np.nan
        confidence[:, joint_index] = 0.0
        observed[:, joint_index] = False

        restored = self.sequence.clone(
            joints=joints,
            pose_confidence=confidence,
            observed_mask=observed,
        )
        self.assertFalse(restored.observed_mask[:, joint_index].any())
        self.assertTrue(np.isnan(restored.joints[:, joint_index]).all())

    def test_pose_confidence_must_be_finite_and_bounded(self) -> None:
        confidence = self.sequence.pose_confidence.copy()
        confidence[0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "pose_confidence"):
            self.sequence.clone(pose_confidence=confidence)

        confidence[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "pose_confidence"):
            self.sequence.clone(pose_confidence=confidence)

    def test_boundary_and_quality_labels_have_explicit_ranges(self) -> None:
        boundary = self.sequence.rep_boundary.copy()
        boundary[0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "rep_boundary"):
            self.sequence.clone(rep_boundary=boundary)

        quality = self.sequence.quality.copy()
        quality[0] = 1.5
        with self.assertRaisesRegex(ValueError, "quality labels"):
            self.sequence.clone(quality=quality)

        quality[0] = np.nan
        quality_mask = self.sequence.quality_mask.copy()
        quality_mask[0] = True
        with self.assertRaisesRegex(ValueError, "masked quality"):
            self.sequence.clone(quality=quality, quality_mask=quality_mask)

    def test_normalization_uses_training_entries_only(self) -> None:
        train = self.sequence.clone()
        test_joints = self.sequence.joints.copy() * 1000.0
        test = self.sequence.clone(
            joints=test_joints,
            participant_id="held_out",
        )
        entries = [
            SequenceEntry(sequence=train, split="train"),
            SequenceEntry(sequence=test, split="test"),
        ]
        mean, std = fit_normalization(entries)
        train_features = extract_features(train)
        finite = np.isfinite(train_features[:, :193])
        expected_mean = np.sum(np.where(finite, train_features[:, :193], 0.0), axis=0) / finite.sum(axis=0)
        np.testing.assert_allclose(mean[:193], expected_mean, rtol=1e-5, atol=1e-6)
        self.assertTrue(np.isfinite(std).all())
        self.assertTrue(np.all(std > 0.0))

    def test_diagnostic_windows_normalize_only_declared_continuous_columns(self) -> None:
        mean, std = fit_normalization(
            [SequenceEntry(sequence=self.sequence, split="train")],
            include_diagnostics=True,
        )
        raw = extract_features(self.sequence, include_diagnostics=True)
        records = sequence_windows(
            self.sequence,
            normalization_mean=mean,
            normalization_std=std,
            window_frames=self.sequence.frames,
            window_stride=1,
            include_diagnostics=True,
        )
        prepared = records[0].features[: self.sequence.frames]
        continuous = continuous_feature_indices(include_diagnostics=True)
        noncontinuous = np.setdiff1d(np.arange(378), continuous, assume_unique=True)
        np.testing.assert_array_equal(prepared[:, noncontinuous], raw[:, noncontinuous])
        np.testing.assert_allclose(
            prepared[:, continuous],
            (raw[:, continuous] - mean[continuous]) / std[continuous],
            rtol=1e-5,
            atol=1e-6,
        )


class WindowAndDatasetIntegrityTests(unittest.TestCase):
    @staticmethod
    def _prepared_arrays(count: int = 2, frames: int = 4, features: int = 3) -> dict[str, np.ndarray]:
        time_mask = np.zeros((count, frames), dtype=bool)
        time_mask[:, :2] = True
        phase = np.full((count, frames), -1, dtype=np.int64)
        phase[:, :2] = 0
        boundary = np.full((count, frames, 2), -1.0, dtype=np.float32)
        boundary[:, :2] = 0.0
        quality = np.full((count, 4), np.nan, dtype=np.float32)
        return {
            "features": np.zeros((count, frames, features), dtype=np.float32),
            "phase": phase,
            "boundary": boundary,
            "family": np.full(count, -1, dtype=np.int64),
            "quality": quality,
            "quality_mask": np.zeros((count, 4), dtype=bool),
            "label_weights": np.ones((count, 4), dtype=np.float32),
            "time_mask": time_mask,
        }

    def test_window_starts_cover_the_tail_without_duplicates(self) -> None:
        for frames in range(1, 80):
            starts = _window_starts(frames, window=16, stride=7)
            self.assertEqual(starts, sorted(set(starts)))
            self.assertTrue(starts)
            if frames <= 16:
                self.assertEqual(starts, [0])
            else:
                self.assertEqual(starts[-1], frames - 16)
                self.assertTrue(all(0 <= start <= frames - 16 for start in starts))

    def test_prepared_dataset_rejects_nonfinite_features(self) -> None:
        arrays = self._prepared_arrays()
        arrays["features"][0, 0, 0] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npz"
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "features must be finite"):
                PreparedWindowDataset(path)

    def test_prepared_dataset_rejects_invalid_label_ranges(self) -> None:
        cases = {
            "family": (np.asarray([6, -1], dtype=np.int64), "family labels"),
            "phase": (np.asarray([[0, 0, -1, -1], [5, 0, -1, -1]], dtype=np.int64), "phase labels"),
            "boundary": (
                np.asarray(
                    [[[0.0, 0.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, -1.0]],
                     [[0.25, 0.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, -1.0]]],
                    dtype=np.float32,
                ),
                "boundary labels",
            ),
        }
        for name, (value, message) in cases.items():
            with self.subTest(name=name):
                arrays = self._prepared_arrays()
                arrays[name] = value
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "bad.npz"
                    np.savez(path, **arrays)
                    with self.assertRaisesRegex(ValueError, message):
                        PreparedWindowDataset(path)

    def test_prepared_dataset_rejects_labels_in_padding(self) -> None:
        arrays = self._prepared_arrays()
        arrays["phase"][0, 3] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npz"
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "padded frames.*phase"):
                PreparedWindowDataset(path)

    def test_prepared_dataset_rejects_holey_or_empty_time_masks(self) -> None:
        for kind in ("holey", "empty"):
            arrays = self._prepared_arrays()
            if kind == "holey":
                arrays["time_mask"][0] = np.asarray([True, False, True, False])
            else:
                arrays["time_mask"][0] = False
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "bad.npz"
                np.savez(path, **arrays)
                with self.assertRaisesRegex(ValueError, "time_mask|valid frame"):
                    PreparedWindowDataset(path)

    def test_prepared_dataset_rejects_masked_nan_quality(self) -> None:
        arrays = self._prepared_arrays()
        arrays["quality_mask"][0, 0] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npz"
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "masked quality"):
                PreparedWindowDataset(path)

    def test_prepared_dataset_rejects_invalid_tracking_targets(self) -> None:
        cases = {
            "nonfinite": np.nan,
            "too_low": -2.0,
            "too_high": 1.1,
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                arrays = self._prepared_arrays()
                arrays["tracking_target"] = np.ones((2, 4), dtype=np.float32)
                arrays["tracking_target"][0, 0] = value
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "bad.npz"
                    np.savez(path, **arrays)
                    with self.assertRaisesRegex(ValueError, "tracking"):
                        PreparedWindowDataset(path)

    def test_prepared_dataset_returns_tracking_target_and_legacy_fallback(self) -> None:
        arrays = self._prepared_arrays()
        arrays["tracking_target"] = np.full((2, 4), -1.0, dtype=np.float32)
        arrays["tracking_target"][:, :2] = 0.75
        with tempfile.TemporaryDirectory() as temporary:
            explicit_path = Path(temporary) / "explicit.npz"
            np.savez(explicit_path, **arrays)
            explicit = PreparedWindowDataset(explicit_path)
            self.assertIn("tracking_target", explicit[0])
            np.testing.assert_allclose(
                explicit[0]["tracking_target"].numpy(), arrays["tracking_target"][0]
            )

            legacy = {key: value for key, value in arrays.items() if key != "tracking_target"}
            legacy_path = Path(temporary) / "legacy.npz"
            np.savez(legacy_path, **legacy)
            restored = PreparedWindowDataset(legacy_path)
            np.testing.assert_allclose(
                restored[0]["tracking_target"].numpy(),
                np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float32),
            )

    def test_memmap_runtime_policy_masks_multi_rep_phase_without_feature_scan(self) -> None:
        arrays = self._prepared_arrays(count=2, frames=4, features=3)
        arrays["features"] = arrays["features"].astype(np.float16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "prepared"
            root.mkdir()
            for name, values in arrays.items():
                np.save(root / f"{name}.npy", values)
            (Path(temporary) / "prepared.jsonl").write_text(
                json.dumps(
                    {
                        "source_dataset": "ul_red",
                        "phase_label_source": "weak_displacement",
                        "source_metadata": {
                            "recording_repetitions": 3,
                            "labels_are_recording_level": True,
                            "labels_are_weak_phase": True,
                        },
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "source_dataset": "ul_red",
                        "phase_label_source": "weak_displacement",
                        "source_metadata": {
                            "recording_repetitions": 1,
                            "labels_are_recording_level": True,
                            "labels_are_weak_phase": True,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = PreparedWindowDataset(
                root,
                validate_features=False,
                return_numpy=True,
            )
            self.assertEqual(dataset.phase_policy_reasons["multi_rep_weak_phase"], 1)
            self.assertFalse(dataset.phase_available[0])
            self.assertTrue(dataset.phase_available[1])
            batch = prepared_collate([dataset[0], dataset[1]])
            self.assertEqual(batch["features"].dtype, torch.float16)
            self.assertEqual(batch["features"].shape, (2, 2, 3))
            self.assertTrue(torch.all(batch["phase"][0] == -1))
            self.assertTrue(torch.all(batch["phase"][1] == 0))

            sampler = LengthBucketBatchSampler(dataset, 1, shuffle=False)
            self.assertEqual(list(sampler), [[0], [1]])
            self.assertEqual(dataset.validate_feature_storage(chunk_rows=1)["rows_checked"], 2)
            stored = np.load(root / "features.npy", mmap_mode="r+")
            stored[0, 0, 0] = np.nan
            stored.flush()
            with self.assertRaisesRegex(ValueError, "features must be finite"):
                dataset.validate_feature_storage(chunk_rows=1)

    def test_prepared_metadata_loader_reports_json_and_count_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "validation.jsonl").write_text('{"sequence_id": "a"}\n', encoding="utf-8")
            self.assertEqual(
                load_prepared_metadata(root, "validation", expected_count=1),
                [{"sequence_id": "a"}],
            )
            with self.assertRaisesRegex(ValueError, "Metadata count"):
                load_prepared_metadata(root, "validation", expected_count=2)

            (root / "validation.jsonl").write_text('{not-json}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                load_prepared_metadata(root, "validation")

            (root / "validation.jsonl").write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                load_prepared_metadata(root, "validation")

    def test_bucketed_loader_trims_padding_and_restores_sample_order(self) -> None:
        arrays = self._prepared_arrays(count=3)
        arrays["features"][:, 0, 0] = np.arange(3, dtype=np.float32)
        arrays["time_mask"][:] = False
        arrays["time_mask"][0, :1] = True
        arrays["time_mask"][1, :2] = True
        arrays["time_mask"][2, :1] = True
        arrays["phase"][:] = -1
        arrays["phase"][1, :2] = 0
        arrays["boundary"][:] = -1.0
        arrays["boundary"][1, :2] = 0.0

        class EchoModel(torch.nn.Module):
            def forward(self, values: torch.Tensor, time_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
                batch_size, frames, _ = values.shape
                family = torch.zeros((batch_size, 6), dtype=torch.float32, device=values.device)
                family[:, 0] = values[:, 0, 0]
                return {
                    "family_logits": family,
                    "phase_logits": torch.zeros((batch_size, frames, 5), device=values.device),
                    "boundary_logits": torch.zeros((batch_size, frames, 2), device=values.device),
                    "quality_logits": torch.zeros((batch_size, 4), device=values.device),
                    "tracking_logits": torch.zeros((batch_size, frames), device=values.device),
                }

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prepared.npz"
            np.savez(path, **arrays)
            dataset = PreparedWindowDataset(path, validate_features=False, return_numpy=True)
            sampler = LengthBucketBatchSampler(dataset, 2, shuffle=False)
            loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=prepared_collate)
            outputs, targets = collect_predictions(EchoModel(), loader, torch.device("cpu"))

        np.testing.assert_array_equal(outputs["family_logits"][:, 0], np.arange(3, dtype=np.float32))
        self.assertEqual(outputs["phase_logits"].shape, (3, 4, 5))
        self.assertEqual(targets["phase"].shape, (3, 4))
        self.assertFalse(targets["time_mask"][0, 1])

    def test_prepare_extracts_features_once_per_prepared_entry(self) -> None:
        config = load_config(CONFIG_PATH)
        config = deepcopy(config)
        config["data"]["sources"] = []
        config["data"]["required_roles"] = []
        config["data"]["synthetic"]["enabled"] = False
        config["data"]["synthetic_seed"]["participant_count"] = 3
        config["data"]["synthetic_seed"]["exercises_per_participant"] = 1

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "training.src.data.prepare.extract_features",
                wraps=extract_features,
            ) as extract_mock:
                summary = prepare_dataset(config, Path(temporary))

        self.assertEqual(extract_mock.call_count, summary["entry_count_with_synthetic"])
        self.assertEqual(summary["normalization_pass"], "single_extraction_chunked_in_place")

    def test_prepare_writes_a_self_consistent_artifact_index(self) -> None:
        config = load_config(CONFIG_PATH)
        config = deepcopy(config)
        config["data"]["sources"] = []
        config["data"]["required_roles"] = []
        config["data"]["synthetic"]["enabled"] = False
        config["data"]["synthetic_seed"]["participant_count"] = 3
        config["data"]["synthetic_seed"]["exercises_per_participant"] = 1

        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            summary = prepare_dataset(config, project_root)
            processed_root = project_root / "data/processed"
            stored_summary = json.loads((processed_root / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(stored_summary, summary)
            self.assertEqual(summary["sequence_counts_by_source"], {"procedural_seed": 15})
            self.assertEqual(summary["label_coverage_by_sequence"]["quality"], 0)
            self.assertEqual(summary["target_fps"], 30.0)
            self.assertEqual(summary["max_interpolation_gap"], 5)
            self.assertEqual(summary["prepared_arrays"], list(PREPARED_ARRAY_NAMES))
            feature_schema = json.loads(
                (project_root / "artifacts/feature_schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(feature_schema["nominal_fps"], 30.0)

            for split, expected_count in summary["window_counts"].items():
                metadata_path = processed_root / f"{split}.jsonl"
                metadata_count = len(metadata_path.read_text(encoding="utf-8").splitlines())
                self.assertEqual(metadata_count, expected_count)
                split_root = processed_root / split
                for name in PREPARED_ARRAY_NAMES:
                    self.assertTrue((split_root / f"{name}.npy").exists())

            with np.load(project_root / "artifacts/normalization_stats.npz") as stats:
                self.assertEqual(stats["mean"].shape, (FEATURE_DIM,))
                self.assertEqual(stats["std"].shape, (FEATURE_DIM,))
                self.assertTrue(np.isfinite(stats["mean"]).all())
                self.assertTrue(np.isfinite(stats["std"]).all())

    def test_prepare_supports_the_optional_diagnostic_feature_schema(self) -> None:
        config = load_config(CONFIG_PATH)
        config = deepcopy(config)
        config["data"]["sources"] = []
        config["data"]["required_roles"] = []
        config["data"]["synthetic"]["enabled"] = False
        config["data"]["synthetic_seed"]["participant_count"] = 3
        config["data"]["synthetic_seed"]["exercises_per_participant"] = 1
        config["features"]["include_diagnostics"] = True
        config["features"]["input_dim"] = 378
        config["features"]["continuous_dim"] = 288

        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            summary = prepare_dataset(config, project_root)
            dataset = PreparedWindowDataset(project_root / "data/processed/train")
            schema = json.loads(
                (project_root / "artifacts/feature_schema.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["feature_dimension"], 378)
        self.assertEqual(summary["continuous_dimension"], 288)
        self.assertEqual(summary["feature_schema_version"], "adaptfit.features.v2.diagnostics")
        self.assertEqual(tuple(dataset.features.shape[1:]), (128, 378))
        self.assertEqual(schema["dimension"], 378)


class AugmentationIntegrityTests(unittest.TestCase):
    def test_augmentations_are_deterministic_and_do_not_mutate_parent(self) -> None:
        config = load_config(CONFIG_PATH)
        parent = next(
            sequence
            for sequence in make_procedural_sequences(config)
            if sequence.metadata.get("family_name") == "arm_flexion"
        )
        original_joints = parent.joints.copy()
        original_metadata = dict(parent.metadata)
        first = generate_augmentations(
            parent,
            rng=np.random.default_rng(123),
            copies=4,
            occlusion_probability=1.0,
        )
        second = generate_augmentations(
            parent,
            rng=np.random.default_rng(123),
            copies=4,
            occlusion_probability=1.0,
        )

        np.testing.assert_array_equal(parent.joints, original_joints)
        self.assertEqual(parent.metadata, original_metadata)
        for left, right in zip(first, second):
            np.testing.assert_allclose(left.joints, right.joints, equal_nan=True)
            np.testing.assert_array_equal(left.observed_mask, right.observed_mask)
            self.assertEqual(left.metadata, right.metadata)


if __name__ == "__main__":
    unittest.main()
