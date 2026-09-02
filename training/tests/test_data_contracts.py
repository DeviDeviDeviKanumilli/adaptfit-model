from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from training.src.config import load_config
from training.src.data.adapters import (
    INTELLI_SOURCE_NAMES,
    MMFIT_SOURCE_NAMES,
    REHAB24_SOURCE_NAMES,
    ULRED_SOURCE_NAMES,
    load_intellirehabds,
    load_mmfit,
    load_rehab24_6,
    load_ul_red,
)
from training.src.data.dataset import PreparedWindowDataset
from training.src.data.prepare import (
    add_synthetic_entries,
    assign_splits,
    sequence_windows,
)
from training.src.data.procedural import make_procedural_sequences
from training.src.data.schema import (
    CANONICAL_INDEX,
    FAMILY_TO_ID,
    CanonicalSequence,
    label_loss_weights,
    quality_from_correctness,
)
from training.src.data.synthetic import generate_augmentations
from training.src.features.anatomy import FEATURE_DIM, LIMB_JOINTS
from training.tests.support import CONFIG_PATH


class DataContractTests(unittest.TestCase):
    @staticmethod
    def config() -> dict:
        return deepcopy(load_config(CONFIG_PATH))

    def test_canonical_sequence_round_trip_preserves_profile_and_labels(self) -> None:
        sequence = make_procedural_sequences(self.config())[0]
        sequence.capability_states["right_arm"] = "absent"
        sequence.metadata["round_trip_test"] = True

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sequence.npz"
            sequence.to_npz(path)
            restored = CanonicalSequence.from_npz(path)

        np.testing.assert_allclose(restored.joints, sequence.joints, equal_nan=True)
        np.testing.assert_array_equal(restored.observed_mask, sequence.observed_mask)
        np.testing.assert_array_equal(restored.phase, sequence.phase)
        np.testing.assert_array_equal(restored.rep_boundary, sequence.rep_boundary)
        self.assertEqual(restored.capability_states, sequence.capability_states)
        self.assertEqual(restored.metadata, sequence.metadata)

    def test_invalid_canonical_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CanonicalSequence(
                joints=np.zeros((4, 32, 2), dtype=np.float32),
                pose_confidence=np.ones((4, 33), dtype=np.float32),
                observed_mask=np.ones((4, 33), dtype=bool),
                timestamps=np.arange(4, dtype=np.float32),
            )

    def test_unavailable_quality_labels_are_masked(self) -> None:
        quality, mask = quality_from_correctness(None)
        self.assertTrue(np.isnan(quality).all())
        self.assertFalse(mask.any())

        quality, mask = quality_from_correctness(1)
        self.assertTrue(np.isnan(quality).all())
        self.assertFalse(mask.any())

    def test_missing_family_and_phase_labels_are_not_valid_classes(self) -> None:
        sequence = CanonicalSequence(
            joints=np.zeros((4, 33, 2), dtype=np.float32),
            pose_confidence=np.ones((4, 33), dtype=np.float32),
            observed_mask=np.ones((4, 33), dtype=bool),
            timestamps=np.arange(4, dtype=np.float32) / 30.0,
        )
        self.assertEqual(sequence.family, -1)
        self.assertTrue(np.all(sequence.phase == -1))

    def test_procedural_seed_covers_the_first_release_movement_families(self) -> None:
        sequences = make_procedural_sequences(self.config())
        families = {sequence.family for sequence in sequences}
        self.assertTrue({FAMILY_TO_ID[name] for name in ("arm_flexion", "row_pull", "knee_extension", "hip_flexion", "reach")} <= families)
        self.assertIn("seated", {sequence.position for sequence in sequences})
        self.assertIn("wheelchair", {sequence.position for sequence in sequences})

    def test_label_weights_downweight_weak_and_procedural_supervision(self) -> None:
        config = self.config()
        config["data"]["label_policy"]["allow_procedural_quality"] = True
        procedural = make_procedural_sequences(config)[0]
        procedural_weights = label_loss_weights(procedural)
        self.assertTrue(np.allclose(procedural_weights, [0.25, 0.25, 0.25, 0.25]))

        weak = procedural.clone(
            metadata={
                "label_provenance": "weak",
                "phase_label_source": "weak_displacement",
                "boundary_label_source": "segmentation",
                "quality_label_source": "unlabeled",
            }
        )
        weak_weights = label_loss_weights(weak)
        self.assertTrue(np.allclose(weak_weights, [0.60, 0.35, 1.0, 0.0]))

    def test_procedural_quality_is_masked_by_default(self) -> None:
        sequence = make_procedural_sequences(self.config())[0]
        self.assertFalse(sequence.quality_mask.any())
        self.assertEqual(sequence.metadata["quality_label_source"], "unlabeled")
        self.assertFalse(sequence.metadata["procedural_quality_enabled"])

        config = self.config()
        config["data"]["label_policy"]["allow_procedural_quality"] = True
        enabled = make_procedural_sequences(config)[0]
        self.assertTrue(enabled.quality_mask.all())
        self.assertEqual(enabled.metadata["quality_label_source"], "procedural_template")

    def test_public_rehab24_adapter_emits_canonical_sequence(self) -> None:
        frames = 8
        source = np.ones((frames, len(REHAB24_SOURCE_NAMES), 2), dtype=np.float32)
        source[:, REHAB24_SOURCE_NAMES.index("LeftShoulder")] = (-1.0, 1.0)
        source[:, REHAB24_SOURCE_NAMES.index("RightShoulder")] = (1.0, 1.0)
        source[:, REHAB24_SOURCE_NAMES.index("LeftUpLeg")] = (-0.5, 2.0)
        source[:, REHAB24_SOURCE_NAMES.index("RightUpLeg")] = (0.5, 2.0)
        source_buffer = BytesIO()
        np.save(source_buffer, source)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ZipFile(root / "2d_joints.zip", "w") as archive:
                archive.writestr("Ex1/vid-c17-30fps.npy", source_buffer.getvalue())
            (root / "Segmentation.csv").write_text(
                "video_id;repetition_number;exercise_id;person_id;first_frame;last_frame;cam17_orientation;correctness\n"
                "vid;1;1;person-a;0;7;front;1\n",
                encoding="utf-8",
            )

            sequences = load_rehab24_6(root)

        self.assertEqual(len(sequences), 1)
        sequence = sequences[0]
        self.assertEqual(sequence.joints.shape, (frames, 33, 2))
        self.assertEqual(sequence.family, FAMILY_TO_ID["reach"])
        self.assertEqual(sequence.participant_id, "person-a")
        self.assertFalse(sequence.quality_mask.any())
        self.assertEqual(sequence.metadata["quality_label_source"], "unlabeled")
        self.assertEqual(sequence.metadata["phase_label_source"], "weak_displacement")
        self.assertEqual(sequence.metadata["boundary_label_source"], "segmentation")

    def test_public_intelli_adapter_emits_wheelchair_metadata(self) -> None:
        rows = []
        for frame in range(5):
            values = np.full((len(INTELLI_SOURCE_NAMES), 3), 0.5 + frame, dtype=np.float32)
            rows.append(",".join(str(value) for value in values.reshape(-1)))
        payload = ("\n".join(rows) + "\n").encode("utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ZipFile(root / "SkeletonData.zip", "w") as archive:
                archive.writestr("001_20200101_4_1_1_Wheelchair.txt", payload)

            sequences = load_intellirehabds(root)
            enabled = load_intellirehabds(root, allow_single_clip_boundaries=True)

        self.assertEqual(len(sequences), 1)
        sequence = sequences[0]
        self.assertEqual(sequence.position, "wheelchair")
        self.assertEqual(sequence.family, FAMILY_TO_ID["reach"])
        self.assertEqual(sequence.joints.shape, (5, 33, 2))
        self.assertFalse(sequence.quality_mask.any())
        self.assertEqual(sequence.metadata["quality_label_source"], "unlabeled")
        self.assertEqual(sequence.metadata["boundary_label_source"], "unlabeled")
        self.assertTrue(np.all(sequence.rep_boundary == -1.0))
        self.assertFalse(sequence.metadata["labels_are_weak_boundary"])

        self.assertTrue(enabled[0].rep_boundary.any())
        self.assertEqual(enabled[0].metadata["boundary_label_source"], "single_clip_assumption")

    def test_public_mmfit_adapter_emits_set_level_pose_sequences(self) -> None:
        frames = 8
        raw = np.zeros((2, frames, len(MMFIT_SOURCE_NAMES) + 1), dtype=np.float32)
        raw[0, :, 0] = np.arange(frames, dtype=np.float32)
        raw[1, :, 0] = np.arange(frames, dtype=np.float32)
        for joint in range(len(MMFIT_SOURCE_NAMES)):
            raw[0, :, joint + 1] = 100.0 + joint
            raw[1, :, joint + 1] = 200.0 + joint
        pose_buffer = BytesIO()
        np.save(pose_buffer, raw)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ZipFile(root / "mm-fit.zip", "w") as archive:
                archive.writestr("w00/w00_pose_2d.npy", pose_buffer.getvalue())
                archive.writestr("w00/w00_labels.csv", "1,6,3,bicep_curls\n")

            sequences = load_mmfit(root)

        self.assertEqual(len(sequences), 1)
        sequence = sequences[0]
        self.assertEqual(sequence.source_dataset, "mmfit")
        self.assertEqual(sequence.participant_id, "mmfit_participant_2")
        self.assertEqual(sequence.family, FAMILY_TO_ID["arm_flexion"])
        self.assertEqual(sequence.position, "standing")
        self.assertEqual(sequence.frames, 6)
        self.assertEqual(sequence.metadata["repetition_count"], 3)
        self.assertEqual(sequence.metadata["phase_label_source"], "unknown")
        self.assertFalse(np.any(sequence.phase >= 0))
        self.assertFalse(np.any(sequence.rep_boundary >= 0))

    def test_mmfit_workout_mapping_prevents_participant_split_leakage(self) -> None:
        frames = 6
        raw = np.zeros((2, frames, len(MMFIT_SOURCE_NAMES) + 1), dtype=np.float32)
        raw[:, :, 0] = np.arange(frames, dtype=np.float32)
        raw[:, :, 1:] = 1.0
        pose_buffer = BytesIO()
        np.save(pose_buffer, raw)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ZipFile(root / "mm-fit.zip", "w") as archive:
                for workout in ("w00", "w05"):
                    archive.writestr(f"mm-fit/{workout}/{workout}_pose_2d.npy", pose_buffer.getvalue())
                    archive.writestr(f"mm-fit/{workout}/{workout}_labels.csv", "0,5,2,bicep_curls\n")

            sequences = load_mmfit(root)

        self.assertEqual(len(sequences), 2)
        self.assertEqual({sequence.participant_id for sequence in sequences}, {"mmfit_participant_2"})
        self.assertEqual({sequence.session_id for sequence in sequences}, {"w00", "w05"})

    def test_public_ul_red_adapter_reads_markerless_amc_and_preserves_protocol(self) -> None:
        def amc_payload(missing_joint: str | None = None) -> str:
            lines = [":FULLY-SPECIFIED", ":DEGREES"]
            for frame in range(1, 7):
                lines.append(str(frame))
                for joint_index, joint_name in enumerate(ULRED_SOURCE_NAMES):
                    if joint_name == missing_joint:
                        continue
                    lines.append(
                        f"{joint_name}\t{100 + joint_index + frame * 0.1:.3f}\t"
                        f"{200 + joint_index + frame * 0.2:.3f}\t"
                        f"{300 + joint_index + frame * 0.3:.3f}"
                    )
            return "\n".join(lines) + "\n"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ZipFile(root / "S01.zip", "w") as archive:
                archive.writestr(
                    "S01/marker-less/clean/ArmRaiseR1S01.amc",
                    amc_payload(missing_joint="LeftHand"),
                )
                archive.writestr(
                    "S01/marker-less/clean/SeatedHipMarchR3S01.amc",
                    amc_payload(),
                )

            sequences = load_ul_red(root)

        self.assertEqual(len(sequences), 2)
        arm_raise = next(sequence for sequence in sequences if "ArmRaise" in sequence.session_id)
        seated_march = next(sequence for sequence in sequences if "SeatedHipMarch" in sequence.session_id)
        self.assertEqual(arm_raise.source_dataset, "ul_red")
        self.assertEqual(arm_raise.participant_id, "S01")
        self.assertEqual(arm_raise.family, FAMILY_TO_ID["reach"])
        self.assertEqual(arm_raise.position, "standing")
        self.assertEqual(arm_raise.metadata["recording_repetitions"], 1)
        self.assertEqual(arm_raise.metadata["pace_protocol"], "normal")
        self.assertFalse(arm_raise.observed_mask[:, CANONICAL_INDEX["left_index"]].any())
        self.assertEqual(seated_march.position, "seated")
        self.assertEqual(seated_march.family, FAMILY_TO_ID["hip_flexion"])
        self.assertEqual(seated_march.metadata["recording_repetitions"], 3)
        self.assertEqual(seated_march.metadata["pace_protocol"], "normal_fast_slow")
        self.assertFalse(np.any(seated_march.phase >= 0))
        self.assertEqual(
            seated_march.metadata["phase_label_status"],
            "unavailable_multi_repetition_recording",
        )
        self.assertTrue(np.all(np.diff(seated_march.timestamps) > 0.0))

    def test_ul_red_archives_keep_subject_ids_separate(self) -> None:
        payload = (
            ":FULLY-SPECIFIED\n:DEGREES\n"
            "1\nWaist\t100\t200\t300\n"
            "2\nWaist\t101\t201\t301\n"
            "3\nWaist\t102\t202\t302\n"
            "4\nWaist\t103\t203\t303\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for subject in ("S01", "S02"):
                with ZipFile(root / f"{subject}.zip", "w") as archive:
                    archive.writestr(
                        f"{subject}/marker-less/clean/ArmRaiseR1{subject}.amc",
                        payload,
                    )

            sequences = load_ul_red(root)

        self.assertEqual({sequence.participant_id for sequence in sequences}, {"S01", "S02"})

    def test_augmentations_mark_occlusions_and_absent_capabilities(self) -> None:
        sequence = next(
            item
            for item in make_procedural_sequences(self.config())
            if item.family == FAMILY_TO_ID["arm_flexion"]
        )
        variants = generate_augmentations(
            sequence,
            rng=np.random.default_rng(7),
            copies=3,
            occlusion_probability=1.0,
            noise_std=0.0,
        )

        self.assertEqual(len(variants), 3)
        self.assertTrue(all(item.metadata.get("synthetic") for item in variants))
        occluded = [item for item in variants if "synthetic_occlusion" in item.metadata]
        self.assertGreaterEqual(len(occluded), 2)
        for item in occluded:
            limb = item.metadata["synthetic_occlusion"]
            indices = [CANONICAL_INDEX[name] for name in LIMB_JOINTS[limb]]
            self.assertTrue(np.isnan(item.joints[:, indices]).all())
            self.assertFalse(item.observed_mask[:, indices].any())
            self.assertEqual(float(item.pose_confidence[:, indices].sum()), 0.0)
            self.assertFalse(item.quality_mask.any())
        self.assertTrue(any("absent" in item.capability_states.values() for item in variants))

    def test_synthetic_variants_are_added_only_to_training_entries(self) -> None:
        config = self.config()
        config["data"]["synthetic"]["copies_per_sequence"] = 1
        config["data"]["synthetic_seed"]["participant_count"] = 3
        config["data"]["synthetic_seed"]["exercises_per_participant"] = 1
        entries = assign_splits(make_procedural_sequences(config), seed=42)

        augmented = add_synthetic_entries(entries, config)
        synthetic = [entry for entry in augmented if entry.synthetic]
        self.assertTrue(synthetic)
        self.assertTrue(all(entry.split == "train" for entry in synthetic))
        self.assertFalse(any(entry.synthetic for entry in augmented if entry.split != "train"))

    def test_short_sequence_windows_are_padded_and_masked(self) -> None:
        sequence = make_procedural_sequences(self.config())[0]
        length = 24
        short = sequence.clone(
            joints=sequence.joints[:length],
            pose_confidence=sequence.pose_confidence[:length],
            observed_mask=sequence.observed_mask[:length],
            timestamps=sequence.timestamps[:length],
            phase=sequence.phase[:length],
            rep_boundary=sequence.rep_boundary[:length],
        )
        records = sequence_windows(
            short,
            normalization_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
            normalization_std=np.ones(FEATURE_DIM, dtype=np.float32),
            window_frames=32,
            window_stride=8,
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(int(record.time_mask.sum()), length)
        self.assertTrue(np.all(record.phase[length:] == -1))
        self.assertTrue(np.all(record.boundary[length:] == -1.0))
        self.assertTrue(np.all(record.tracking_target[:length] >= 0.0))
        self.assertTrue(np.all(record.tracking_target[length:] == -1.0))
        np.testing.assert_array_equal(record.features[length:], np.zeros((8, FEATURE_DIM), dtype=np.float32))

    def test_prepared_dataset_rejects_mismatched_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npz"
            np.savez(
                path,
                features=np.zeros((2, 4, 3), dtype=np.float32),
                phase=np.zeros((2, 4), dtype=np.int64),
                boundary=np.zeros((2, 3, 2), dtype=np.float32),
                family=np.zeros(2, dtype=np.int64),
                quality=np.zeros((2, 4), dtype=np.float32),
                quality_mask=np.zeros((2, 4), dtype=bool),
                time_mask=np.ones((2, 4), dtype=bool),
            )
            with self.assertRaises(ValueError):
                PreparedWindowDataset(path)

    def test_legacy_npz_storage_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.npz"
            np.savez(
                path,
                features=np.zeros((1, 4, 3), dtype=np.float32),
                phase=np.zeros((1, 4), dtype=np.int64),
                boundary=np.zeros((1, 4, 2), dtype=np.float32),
                family=np.zeros(1, dtype=np.int64),
                quality=np.zeros((1, 4), dtype=np.float32),
                quality_mask=np.ones((1, 4), dtype=bool),
                time_mask=np.ones((1, 4), dtype=bool),
            )
            dataset = PreparedWindowDataset(path)
            sample = dataset[0]

        self.assertEqual(dataset.storage_format, "npz_legacy")
        self.assertEqual(tuple(sample["features"].shape), (4, 3))


if __name__ == "__main__":
    unittest.main()
