from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.evaluate import _update_combined_report
from training.src.data.identity import (
    SEQUENCE_IDENTITY_VERSION,
    sequence_identity_for_metadata,
    sequence_identity_id,
)
from training.src.data.prepare import assign_splits
from training.src.data.schema import CanonicalSequence
from training.src.evaluation import aggregate_sequence_predictions
from training.src.metrics import compute_metrics
from training.src.reporting import RUN_METRICS_SCHEMA_VERSION


def _sequence(
    source: str,
    participant: str,
    position: str = "standing",
    session: str = "session",
    metadata: dict | None = None,
) -> CanonicalSequence:
    frames = 6
    joints = np.zeros((frames, 33, 2), dtype=np.float32)
    joints[:, :, 0] = np.linspace(0.0, 1.0, frames, dtype=np.float32)[:, None]
    return CanonicalSequence(
        joints=joints,
        pose_confidence=np.ones((frames, 33), dtype=np.float32),
        observed_mask=np.ones((frames, 33), dtype=bool),
        timestamps=np.arange(frames, dtype=np.float32) / 30.0,
        position=position,
        participant_id=participant,
        session_id=session,
        source_dataset=source,
        family=1,
        phase=np.full(frames, 2, dtype=np.int64),
        rep_boundary=np.zeros((frames, 2), dtype=np.float32),
        metadata=metadata or {},
    )


def _prediction_fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    phase = np.asarray([[1, 2, 3, 4], [3, 4, 1, 2]], dtype=np.int64)
    phase_logits = np.full((2, 4, 5), -5.0, dtype=np.float32)
    for window in range(2):
        for frame in range(4):
            phase_logits[window, frame, phase[window, frame]] = 5.0
    boundary = np.zeros((2, 4, 2), dtype=np.float32)
    boundary[0, 0, 0] = 1.0
    boundary[1, 3, 1] = 1.0
    outputs = {
        "family_logits": np.asarray([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
        "phase_logits": phase_logits,
        "boundary_logits": np.where(boundary > 0, 5.0, -5.0).astype(np.float32),
        "quality_logits": np.zeros((2, 4), dtype=np.float32),
    }
    targets = {
        "family": np.asarray([1, 1], dtype=np.int64),
        "phase": phase,
        "boundary": boundary,
        "quality": np.full((2, 4), np.nan, dtype=np.float32),
        "quality_mask": np.zeros((2, 4), dtype=bool),
        "time_mask": np.ones((2, 4), dtype=bool),
    }
    return outputs, targets


class CorrectedBenchmarkTests(unittest.TestCase):
    def test_intelli_gesture_identity_does_not_merge_same_session_rep(self) -> None:
        common = {
            "source_member": "101_20200101_1_1_1_chair.txt",
            "repetition_number": 1,
            "position_label": "chair",
            "correctness": 1,
        }
        first = sequence_identity_id(
            source_dataset="intellirehabds",
            participant_id="101",
            session_id="101_20200101",
            source_metadata={**common, "gesture_id": 1},
        )
        second = sequence_identity_id(
            source_dataset="intellirehabds",
            participant_id="101",
            session_id="101_20200101",
            source_metadata={**common, "gesture_id": 2},
        )
        self.assertNotEqual(first, second)

    def test_mmfit_set_identity_does_not_merge_same_workout(self) -> None:
        first = sequence_identity_id(
            source_dataset="mmfit",
            participant_id="p1",
            session_id="workout-1",
            source_metadata={
                "workout_id": "workout-1",
                "activity": "squat",
                "set_index": 0,
                "set_start_frame": 10,
                "set_end_frame": 100,
            },
        )
        second = sequence_identity_id(
            source_dataset="mmfit",
            participant_id="p1",
            session_id="workout-1",
            source_metadata={
                "workout_id": "workout-1",
                "activity": "squat",
                "set_index": 1,
                "set_start_frame": 110,
                "set_end_frame": 200,
            },
        )
        self.assertNotEqual(first, second)

    def test_legacy_rows_reconstruct_the_same_corrected_identity(self) -> None:
        source = {
            "gesture_id": 4,
            "repetition_number": 2,
            "position_label": "chair",
            "correctness": 0,
        }
        first = {
            "sequence_id": "legacy-short-id-a",
            "source_dataset": "intellirehabds",
            "participant_id": "101",
            "session_id": "101_20200101",
            "position": "seated",
            "window_start": 0,
            "source_metadata": source,
        }
        second = {**first, "sequence_id": "legacy-short-id-b", "window_start": 8}
        self.assertEqual(sequence_identity_for_metadata(first), sequence_identity_for_metadata(second))

    def test_overlapping_windows_merge_by_corrected_identity(self) -> None:
        outputs, targets = _prediction_fixture()
        source_metadata = {
            "gesture_id": 1,
            "repetition_number": 1,
            "position_label": "chair",
            "correctness": 1,
        }
        metadata = [
            {
                "sequence_id": "legacy-a",
                "source_dataset": "intellirehabds",
                "participant_id": "101",
                "session_id": "101_session",
                "position": "seated",
                "window_start": 0,
                "window_end": 4,
                "source_metadata": source_metadata,
            },
            {
                "sequence_id": "legacy-b",
                "source_dataset": "intellirehabds",
                "participant_id": "101",
                "session_id": "101_session",
                "position": "seated",
                "window_start": 2,
                "window_end": 6,
                "source_metadata": source_metadata,
            },
        ]
        sequence_outputs, sequence_targets, sequence_metadata = aggregate_sequence_predictions(
            outputs, targets, metadata
        )
        self.assertEqual(len(sequence_metadata), 1)
        self.assertEqual(sequence_metadata[0]["sequence_identity_version"], SEQUENCE_IDENTITY_VERSION)
        self.assertEqual(compute_metrics(sequence_outputs, sequence_targets)["family_accuracy"], 1.0)

    def test_required_test_coverage_is_deterministic_and_group_isolated(self) -> None:
        sequences = [
            _sequence("ul_red", "u0", metadata={"source_member": "u0.amc", "exercise_name": "curl"}),
            _sequence("intellirehabds", "i0", position="wheelchair", metadata={"gesture_id": 1, "repetition_number": 1}),
        ]
        for index in range(8):
            sequences.append(_sequence("fixture", f"p{index}", metadata={"trial_id": index}))
        first = assign_splits(
            sequences,
            seed=42,
            required_test_sources=["ul_red"],
            required_test_positions=["wheelchair"],
            enforce_required_test_coverage=True,
        )
        second = assign_splits(
            sequences,
            seed=42,
            required_test_sources=["ul_red"],
            required_test_positions=["wheelchair"],
            enforce_required_test_coverage=True,
        )
        first_map = {_group(entry.sequence): entry.split for entry in first}
        second_map = {_group(entry.sequence): entry.split for entry in second}
        self.assertEqual(first_map, second_map)
        test_entries = [entry for entry in first if entry.split == "test"]
        self.assertTrue(any(entry.sequence.source_dataset == "ul_red" for entry in test_entries))
        self.assertTrue(any(entry.sequence.position == "wheelchair" for entry in test_entries))
        by_group: dict[str, set[str]] = {}
        for entry in first:
            by_group.setdefault(_group(entry.sequence), set()).add(entry.split)
        self.assertTrue(all(len(splits) == 1 for splits in by_group.values()))

    def test_metrics_report_accuracy_and_every_phase_support(self) -> None:
        outputs, targets = _prediction_fixture()
        metrics = compute_metrics(outputs, targets)
        self.assertEqual(metrics["family_accuracy"], 1.0)
        self.assertEqual(metrics["phase_frame_accuracy"], 1.0)
        self.assertEqual(metrics["phase_labeled_frame_count"], 8.0)
        self.assertEqual(metrics["phase_support_hold"], 2.0)
        self.assertEqual(metrics["phase_f1_hold"], 1.0)
        self.assertTrue(all(f"phase_f1_{name}" in metrics for name in ("unknown", "rest", "concentric", "hold", "eccentric")))

    def test_combined_report_preserves_both_model_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metrics.json").write_text(
                '{"models": {"tcn": {"parameter_count": 1}}}\n', encoding="utf-8"
            )
            _update_combined_report(root, "tcn", {"model_name": "tcn", "identity": {"identity_version": "v2"}})
            _update_combined_report(root, "gru", {"model_name": "gru", "identity": {"identity_version": "v2"}})
            report = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(report["evaluations"]), {"tcn", "gru"})
            self.assertEqual(report["models"]["tcn"]["parameter_count"], 1)
            self.assertEqual(report["schema_version"], RUN_METRICS_SCHEMA_VERSION)
            self.assertTrue(
                all(
                    key in report
                    for key in ("config", "seed", "data_summary", "run", "models", "evaluations", "artifacts")
                )
            )


def _group(sequence: CanonicalSequence) -> str:
    return f"{sequence.source_dataset}:{sequence.participant_id}"


if __name__ == "__main__":
    unittest.main()
