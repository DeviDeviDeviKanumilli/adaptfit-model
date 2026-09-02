from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from training.src.config import load_config
from training.src.data.adapters import load_ucophyrehabpp
from training.src.data.identity import sequence_identity_for_sequence
from training.src.data.prepare import assign_splits
from training.src.data.provenance import projection_summary
from training.src.data.schema import CanonicalSequence, FAMILY_TO_ID
from training.src.evaluation import aggregate_sequence_predictions
from training.src.losses import LossWeights, compute_loss
from training.src.metrics import compute_metrics
from training.src.models import build_model
from training.tests.support import PROJECT_ROOT


V2_CONFIG = PROJECT_ROOT / "training" / "configs" / "v2_quality.yaml"


def _uco_frame(frame_id: int) -> dict:
    return {
        "id": frame_id,
        "joints": {
            "l_hip": {"x": str(1.0 + frame_id), "y": "2.0", "z": "3.0"},
            "l_knee": {"x": str(1.5 + frame_id), "y": "1.0", "z": "2.0"},
            "l_ankle": {"x": str(2.0 + frame_id), "y": "0.0", "z": "1.0"},
        },
    }


class V2QualityPipelineTests(unittest.TestCase):
    def test_projection_summary_flags_known_legacy_uco_projection(self) -> None:
        summary = projection_summary(
            [
                {
                    "source_dataset": "ucophyrehabpp",
                    "source_metadata": {
                        "source_coordinate_dim": 3,
                        "coordinate_projection": "xy_from_3d",
                    },
                }
            ]
        )
        self.assertEqual(summary["projection_metadata_status"], {"legacy_inferred": 1})
        self.assertEqual(summary["legacy_inferred_projection_row_count"], 1)
        self.assertFalse(summary["projection_metadata_complete"])
        self.assertEqual(summary["depth_discarded_source_rows"], 1)

    def test_projection_summary_rejects_unknown_incomplete_3d_projection(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit xy_from_3d"):
            projection_summary(
                [
                    {
                        "source_dataset": "unknown_source",
                        "source_metadata": {
                            "source_coordinate_dim": 3,
                            "coordinate_projection": "xy_from_3d",
                        },
                    }
                ]
            )

    def test_uco_adapter_emits_recording_boundaries_and_scored_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ucophyrehab2_data.jsonl").write_text(
                json.dumps(
                    {
                        "subject_id": "00",
                        "exercise_id": "03",
                        "sex": "F",
                        "age": 30,
                        "scores": [
                            {"init_frame": 5, "final_frame": 8, "score": [3.0, 4.0]},
                            {"init_frame": 9, "final_frame": 12, "score": 2.0},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "dataset_3d_with_angles.json").write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "folder": "00",
                                "exercise": "03",
                                "position": "supine",
                                "side": "left",
                                "body": "lower",
                                "n_frames": 8,
                                "n_valid_frames": 8,
                                "frames": [_uco_frame(frame) for frame in range(5, 13)],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            sequences = load_ucophyrehabpp(root)
            disabled_sequences = load_ucophyrehabpp(
                root,
                allow_composite_quality=False,
            )

        recordings = [item for item in sequences if item.metadata["recording_unit"] == "recording"]
        repetitions = [item for item in sequences if item.metadata["recording_unit"] == "repetition"]
        self.assertEqual(len(recordings), 1)
        self.assertEqual(len(repetitions), 2)
        recording = recordings[0]
        self.assertEqual(recording.position, "unknown")
        self.assertEqual(recording.metadata["source_position"], "supine")
        self.assertEqual(recording.metadata["source_coordinate_dim"], 3)
        self.assertEqual(recording.metadata["model_coordinate_dim"], 2)
        self.assertEqual(recording.metadata["coordinate_projection"], "xy_from_3d")
        self.assertTrue(recording.metadata["depth_discarded"])
        self.assertEqual(recording.family, FAMILY_TO_ID["hip_flexion"])
        self.assertEqual(recording.metadata["boundary_label_source"], "strong")
        self.assertEqual(recording.rep_boundary[0, 0], 1.0)
        self.assertEqual(recording.rep_boundary[3, 1], 1.0)
        self.assertEqual(recording.rep_boundary[4, 0], 1.0)
        self.assertEqual(recording.rep_boundary[7, 1], 1.0)
        self.assertEqual(recording.capability_states["left_leg"], "available")
        self.assertTrue(np.isnan(recording.quality).all())
        self.assertFalse(recording.quality_mask.any())

        self.assertEqual(repetitions[0].expert_quality, 3)
        self.assertTrue(repetitions[0].expert_quality_mask)
        self.assertEqual(repetitions[0].metadata["expert_quality_rater_count"], 2)
        self.assertEqual(repetitions[1].expert_quality, 1)
        self.assertEqual(repetitions[1].metadata["expert_quality_rater_count"], 1)
        self.assertEqual(repetitions[0].rep_boundary[0, 0], -1.0)
        self.assertEqual(repetitions[0].metadata["quality_label_source"], "unlabeled")
        self.assertNotEqual(
            sequence_identity_for_sequence(recording),
            sequence_identity_for_sequence(repetitions[0]),
        )
        disabled_repetition = next(
            item
            for item in disabled_sequences
            if item.metadata["recording_unit"] == "repetition"
        )
        self.assertEqual(disabled_repetition.expert_quality, -1)
        self.assertFalse(disabled_repetition.expert_quality_mask)
        self.assertEqual(
            disabled_repetition.metadata["expert_quality_label_status"],
            "disabled_by_policy",
        )

    def test_v2_models_expose_optional_expert_quality_head_and_finite_loss(self) -> None:
        config = load_config(V2_CONFIG)
        features = torch.randn(2, 16, 283)
        time_mask = torch.ones(2, 16, dtype=torch.bool)
        for model_name in ("tcn", "gru"):
            model = build_model(model_name, config)
            outputs = model(features, time_mask=time_mask)
            self.assertEqual(outputs["expert_quality_logits"].shape, (2, 5))
            batch = {
                "family": torch.tensor([1, 2]),
                "phase": torch.zeros(2, 16, dtype=torch.long),
                "boundary": torch.zeros(2, 16, 2),
                "quality": torch.full((2, 4), float("nan")),
                "quality_mask": torch.zeros(2, 4, dtype=torch.bool),
                "expert_quality": torch.tensor([0, 4]),
                "expert_quality_mask": torch.ones(2, dtype=torch.bool),
                "expert_label_weight": torch.ones(2),
                "tracking_target": torch.ones(2, 16),
                "time_mask": time_mask,
            }
            loss, values = compute_loss(
                outputs,
                batch,
                LossWeights(expert_quality=1.0),
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(np.isfinite(values["expert_quality"]))

    def test_v2_checkpoint_state_round_trip_preserves_expert_logits(self) -> None:
        config = load_config(V2_CONFIG)
        model = build_model("tcn", config).eval()
        restored = build_model("tcn", config).eval()
        restored.load_state_dict(model.state_dict())
        values = torch.randn(1, 16, 283)
        with torch.no_grad():
            original = model(values)
            reloaded = restored(values)
        torch.testing.assert_close(
            original["expert_quality_logits"], reloaded["expert_quality_logits"]
        )

    def test_expert_metrics_and_sequence_aggregation_preserve_ordinal_targets(self) -> None:
        outputs = {
            "family_logits": np.asarray([[0.0, 4.0, 0.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
            "phase_logits": np.zeros((2, 4, 5), dtype=np.float32),
            "boundary_logits": np.zeros((2, 4, 2), dtype=np.float32),
            "quality_logits": np.zeros((2, 4), dtype=np.float32),
            "expert_quality_logits": np.asarray(
                [[5.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 5.0]],
                dtype=np.float32,
            ),
        }
        targets = {
            "family": np.asarray([1, 1], dtype=np.int64),
            "phase": np.zeros((2, 4), dtype=np.int64),
            "boundary": np.zeros((2, 4, 2), dtype=np.float32),
            "quality": np.full((2, 4), np.nan, dtype=np.float32),
            "quality_mask": np.zeros((2, 4), dtype=bool),
            "expert_quality": np.asarray([0, 4], dtype=np.int64),
            "expert_quality_mask": np.ones(2, dtype=bool),
            "time_mask": np.ones((2, 4), dtype=bool),
        }
        metrics = compute_metrics(outputs, targets)
        self.assertEqual(metrics["expert_quality_accuracy"], 1.0)
        self.assertEqual(metrics["expert_quality_mae"], 0.0)
        self.assertEqual(metrics["expert_quality_support"], 2.0)

        source_metadata = {
            "recording_unit": "repetition",
            "source_member": "0/exercise_03/dataset_3d_with_angles.json",
            "exercise_id": "03",
            "source_frame_start": 5,
            "source_frame_end": 8,
            "repetition_index": 0,
            "side": "left",
            "position_label": "supine",
        }
        metadata = [
            {
                "source_dataset": "ucophyrehabpp",
                "participant_id": "0",
                "session_id": "exercise_03",
                "position": "unknown",
                "window_start": 0,
                "window_end": 4,
                "source_metadata": source_metadata,
            },
            {
                "source_dataset": "ucophyrehabpp",
                "participant_id": "0",
                "session_id": "exercise_03",
                "position": "unknown",
                "window_start": 2,
                "window_end": 6,
                "source_metadata": source_metadata,
            },
        ]
        sequence_outputs, sequence_targets, sequence_metadata = aggregate_sequence_predictions(
            outputs, targets, metadata
        )
        self.assertEqual(len(sequence_metadata), 1)
        self.assertEqual(sequence_targets["expert_quality"].tolist(), [0])
        self.assertTrue(sequence_targets["expert_quality_mask"].all())
        self.assertEqual(
            compute_metrics(sequence_outputs, sequence_targets)["expert_quality_support"],
            1.0,
        )

    def test_v2_required_source_split_is_participant_isolated(self) -> None:
        def sequence(source: str, participant: str, position: str = "standing") -> CanonicalSequence:
            joints = np.ones((6, 33, 2), dtype=np.float32)
            return CanonicalSequence(
                joints=joints,
                pose_confidence=np.ones((6, 33), dtype=np.float32),
                observed_mask=np.ones((6, 33), dtype=bool),
                timestamps=np.arange(6, dtype=np.float32) / 30.0,
                position=position,
                participant_id=participant,
                session_id="session",
                source_dataset=source,
                family=1,
            )

        sequences = [sequence("ucophyrehabpp", "uco-0"), sequence("intellirehabds", "chair-0", "wheelchair")]
        sequences.extend(sequence("fixture", f"p-{index}") for index in range(8))
        entries = assign_splits(
            sequences,
            seed=42,
            required_test_sources=["ucophyrehabpp"],
            required_test_positions=["wheelchair"],
            enforce_required_test_coverage=True,
        )
        self.assertTrue(any(entry.split == "test" and entry.sequence.source_dataset == "ucophyrehabpp" for entry in entries))
        self.assertTrue(any(entry.split == "test" and entry.sequence.position == "wheelchair" for entry in entries))
        groups: dict[str, set[str]] = {}
        for entry in entries:
            groups.setdefault(entry.sequence.participant_id, set()).add(entry.split)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))


if __name__ == "__main__":
    unittest.main()
