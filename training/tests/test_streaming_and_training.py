from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.evaluate import _label_summary, _quality_interpretation, aggregate_sequence_predictions
from training.src.config import load_config
from training.src.data.dataset import PreparedWindowDataset, prepared_split_path
from training.src.data.prepare import prepare_dataset
from training.src.losses import LossWeights, compute_loss
from training.src.metrics import composite_score, compute_metrics
from training.src.models import build_model, parameter_count
from training.src.runner import collect_predictions, train_model
from training.tests.support import CONFIG_PATH


class StreamingAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)

    def test_models_fit_the_intended_small_parameter_budget(self) -> None:
        tcn = build_model("tcn", self.config)
        gru = build_model("gru", self.config)
        self.assertLess(parameter_count(tcn), 500_000)
        self.assertLess(parameter_count(gru), 150_000)

    def test_tcn_and_gru_do_not_use_future_frames(self) -> None:
        torch.manual_seed(11)
        values = torch.randn(1, 32, 283)
        cut = 16
        changed_future = values.clone()
        changed_future[:, cut:] += 25.0
        time_mask = torch.zeros(1, 32, dtype=torch.bool)
        time_mask[:, :cut] = True

        for name in ("tcn", "gru"):
            model = build_model(name, self.config).eval()
            with torch.no_grad():
                original = model(values, time_mask=time_mask)
                changed = model(changed_future, time_mask=time_mask)
            torch.testing.assert_close(
                original["phase_logits"][:, :cut], changed["phase_logits"][:, :cut]
            )
            torch.testing.assert_close(
                original["boundary_logits"][:, :cut], changed["boundary_logits"][:, :cut]
            )
            torch.testing.assert_close(
                original["tracking_logits"][:, :cut], changed["tracking_logits"][:, :cut]
            )
            torch.testing.assert_close(original["family_logits"], changed["family_logits"])
            torch.testing.assert_close(original["quality_logits"], changed["quality_logits"])

    def test_models_are_stateless_across_session_boundaries(self) -> None:
        torch.manual_seed(23)
        first_session = torch.randn(1, 20, 283)
        second_session = torch.randn(1, 20, 283)
        for name in ("tcn", "gru"):
            model = build_model(name, self.config).eval()
            with torch.no_grad():
                expected = model(first_session)
                _ = model(second_session)
                actual = model(first_session)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key])

    def test_models_reject_non_temporal_input(self) -> None:
        for name in ("tcn", "gru"):
            model = build_model(name, self.config)
            with self.assertRaises(ValueError):
                model(torch.randn(2, 283))

    def test_models_reject_feature_schema_width_drift(self) -> None:
        for name in ("tcn", "gru"):
            model = build_model(name, self.config)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "input features"):
                    model(torch.randn(2, 8, 282))

    def test_missing_labels_produce_zero_finite_loss(self) -> None:
        model = build_model("gru", self.config)
        outputs = model(torch.randn(2, 16, 283))
        batch = {
            "family": torch.full((2,), -1, dtype=torch.long),
            "phase": torch.full((2, 16), -1, dtype=torch.long),
            "boundary": torch.full((2, 16, 2), -1.0),
            "quality": torch.full((2, 4), float("nan")),
            "quality_mask": torch.zeros(2, 4, dtype=torch.bool),
            "time_mask": torch.ones(2, 16, dtype=torch.bool),
        }
        loss, values = compute_loss(outputs, batch, LossWeights())
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        self.assertEqual(values["total"], 0.0)

    def test_metrics_ignore_padding_and_unlabeled_quality(self) -> None:
        family_logits = np.asarray([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        phase_logits = np.zeros((1, 4, 5), dtype=np.float32)
        phase_logits[0, 0, 2] = 5.0
        boundary_logits = np.full((1, 4, 2), 8.0, dtype=np.float32)
        quality_logits = np.asarray([[8.0, -8.0, -8.0, -8.0]], dtype=np.float32)
        targets = {
            "family": np.asarray([0], dtype=np.int64),
            "phase": np.asarray([[2, -1, -1, -1]], dtype=np.int64),
            "boundary": np.asarray([[[1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]]),
            "quality": np.asarray([[1.0, np.nan, np.nan, np.nan]], dtype=np.float32),
            "quality_mask": np.asarray([[True, False, False, False]], dtype=bool),
            "tracking_target": np.asarray([[1.0, -1.0, -1.0, -1.0]], dtype=np.float32),
            "time_mask": np.asarray([[True, False, False, False]], dtype=bool),
        }
        metrics = compute_metrics(
            {
                "family_logits": family_logits,
                "phase_logits": phase_logits,
                "boundary_logits": boundary_logits,
                "quality_logits": quality_logits,
                "tracking_logits": np.asarray([[8.0, -8.0, -8.0, -8.0]], dtype=np.float32),
            },
            targets,
        )
        self.assertEqual(metrics["family_macro_f1"], 1.0)
        self.assertEqual(metrics["phase_macro_f1"], 1.0)
        self.assertEqual(metrics["boundary_f1"], 1.0)
        self.assertEqual(metrics["rep_count_mae"], 0.0)
        self.assertEqual(metrics["quality_macro_f1"], 1.0)
        self.assertEqual(metrics["quality_label_coverage"], 0.25)
        self.assertAlmostEqual(metrics["tracking_confidence_mae"], 1.0 - 1.0 / (1.0 + np.exp(-8.0)), places=5)
        self.assertEqual(metrics["tracking_target_coverage"], 1.0)

    def test_tracking_metrics_require_matching_shapes(self) -> None:
        outputs = {
            "family_logits": np.zeros((1, 6), dtype=np.float32),
            "phase_logits": np.zeros((1, 2, 5), dtype=np.float32),
            "boundary_logits": np.zeros((1, 2, 2), dtype=np.float32),
            "quality_logits": np.zeros((1, 4), dtype=np.float32),
            "tracking_logits": np.zeros((1, 2), dtype=np.float32),
        }
        targets = {
            "family": np.asarray([0], dtype=np.int64),
            "phase": np.zeros((1, 2), dtype=np.int64),
            "boundary": np.zeros((1, 2, 2), dtype=np.float32),
            "quality": np.zeros((1, 4), dtype=np.float32),
            "quality_mask": np.ones((1, 4), dtype=bool),
            "tracking_target": np.zeros((1, 3), dtype=np.float32),
            "time_mask": np.ones((1, 2), dtype=bool),
        }
        with self.assertRaises(ValueError):
            compute_metrics(outputs, targets)

    def test_metrics_reject_malformed_or_nonfinite_prediction_contracts(self) -> None:
        outputs = {
            "family_logits": np.zeros((1, 6), dtype=np.float32),
            "phase_logits": np.zeros((1, 2, 5), dtype=np.float32),
            "boundary_logits": np.zeros((1, 2, 2), dtype=np.float32),
            "quality_logits": np.zeros((1, 4), dtype=np.float32),
        }
        targets = {
            "family": np.asarray([0], dtype=np.int64),
            "phase": np.zeros((1, 2), dtype=np.int64),
            "boundary": np.zeros((1, 2, 2), dtype=np.float32),
            "quality": np.zeros((1, 4), dtype=np.float32),
            "quality_mask": np.ones((1, 4), dtype=bool),
            "time_mask": np.ones((1, 2), dtype=bool),
        }
        malformed = dict(outputs)
        malformed["boundary_logits"] = np.zeros((1, 2, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "boundary_logits"):
            compute_metrics(malformed, targets)

        nonfinite = dict(outputs)
        nonfinite["family_logits"] = outputs["family_logits"].copy()
        nonfinite["family_logits"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "family_logits must be finite"):
            compute_metrics(nonfinite, targets)

        invalid_target = dict(targets)
        invalid_target["phase"] = np.asarray([[0, 5]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "phase targets"):
            compute_metrics(outputs, invalid_target)

    def test_sequence_aggregation_merges_overlapping_windows(self) -> None:
        phase_targets = np.asarray([[1, 2, 3, 4], [3, 4, 1, 2]], dtype=np.int64)
        boundary_targets = np.zeros((2, 4, 2), dtype=np.float32)
        boundary_targets[0, 0, 0] = 1.0
        boundary_targets[1, 3, 1] = 1.0
        phase_logits = np.full((2, 4, 5), -8.0, dtype=np.float32)
        for window in range(2):
            for frame in range(4):
                phase_logits[window, frame, phase_targets[window, frame]] = 8.0
        boundary_logits = np.full((2, 4, 2), -8.0, dtype=np.float32)
        boundary_logits[0, 0, 0] = 8.0
        boundary_logits[1, 3, 1] = 8.0
        outputs = {
            "family_logits": np.asarray([[0.0, 8.0, 0.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
            "phase_logits": phase_logits,
            "boundary_logits": boundary_logits,
            "quality_logits": np.asarray([[8.0, -8.0, -8.0, -8.0]] * 2, dtype=np.float32),
        }
        targets = {
            "family": np.asarray([1, 1], dtype=np.int64),
            "phase": phase_targets,
            "boundary": boundary_targets,
            "quality": np.asarray([[1.0, np.nan, np.nan, np.nan]] * 2, dtype=np.float32),
            "quality_mask": np.asarray([[True, False, False, False]] * 2, dtype=bool),
            "tracking_target": np.asarray([[1.0, 0.0, 1.0, 0.0]] * 2, dtype=np.float32),
            "time_mask": np.ones((2, 4), dtype=bool),
        }
        outputs["tracking_logits"] = np.full((2, 4), -8.0, dtype=np.float32)
        outputs["tracking_logits"][0, :2] = 8.0
        outputs["tracking_logits"][1, 2:] = 8.0
        metadata = [
            {"sequence_id": "sequence-a", "window_start": 0, "source_dataset": "fixture", "position": "seated"},
            {"sequence_id": "sequence-a", "window_start": 2, "source_dataset": "fixture", "position": "seated"},
        ]

        sequence_outputs, sequence_targets, sequence_metadata = aggregate_sequence_predictions(
            outputs, targets, metadata
        )
        self.assertEqual(len(sequence_metadata), 1)
        self.assertEqual(int(sequence_targets["time_mask"].sum()), 6)
        self.assertEqual(sequence_outputs["tracking_logits"].shape, (1, 6))
        np.testing.assert_array_equal(
            sequence_targets["tracking_target"][0],
            np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        )
        metrics = compute_metrics(sequence_outputs, sequence_targets)
        self.assertEqual(metrics["family_macro_f1"], 1.0)
        self.assertEqual(metrics["phase_macro_f1"], 1.0)
        self.assertEqual(metrics["boundary_f1"], 1.0)
        self.assertEqual(metrics["rep_count_mae"], 0.0)
        self.assertIn("tracking_confidence_mae", metrics)

    def test_sequence_aggregation_rejects_untrustworthy_window_metadata(self) -> None:
        outputs = {
            "family_logits": np.zeros((1, 6), dtype=np.float32),
            "phase_logits": np.zeros((1, 2, 5), dtype=np.float32),
            "boundary_logits": np.zeros((1, 2, 2), dtype=np.float32),
            "quality_logits": np.zeros((1, 4), dtype=np.float32),
        }
        targets = {
            "family": np.asarray([0], dtype=np.int64),
            "phase": np.zeros((1, 2), dtype=np.int64),
            "boundary": np.zeros((1, 2, 2), dtype=np.float32),
            "quality": np.zeros((1, 4), dtype=np.float32),
            "quality_mask": np.ones((1, 4), dtype=bool),
            "time_mask": np.ones((1, 2), dtype=bool),
        }
        with self.assertRaisesRegex(ValueError, "window_start"):
            aggregate_sequence_predictions(outputs, targets, [{"sequence_id": "s"}])

        holey_targets = dict(targets)
        holey_targets["time_mask"] = np.asarray([[True, False]], dtype=bool)
        with self.assertRaisesRegex(ValueError, "window_end"):
            aggregate_sequence_predictions(
                outputs,
                holey_targets,
                [{"sequence_id": "s", "window_start": 0, "window_end": 2}],
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            nonfinite = dict(outputs)
            nonfinite["phase_logits"] = outputs["phase_logits"].copy()
            nonfinite["phase_logits"][0, 0, 0] = np.nan
            aggregate_sequence_predictions(
                nonfinite,
                targets,
                [{"sequence_id": "s", "window_start": 0, "window_end": 2}],
            )

    def test_label_summary_keeps_weak_and_procedural_sources_visible(self) -> None:
        summary = _label_summary(
            [
                {
                    "label_provenance": "weak",
                    "phase_label_source": "weak_displacement",
                    "boundary_label_source": "segmentation",
                    "quality_label_source": "source_correctness",
                    "synthetic": False,
                },
                {
                    "label_provenance": "procedural",
                    "phase_label_source": "procedural_template",
                    "boundary_label_source": "procedural_template",
                    "quality_label_source": "procedural_template",
                    "synthetic": True,
                },
            ]
        )
        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["synthetic_window_count"], 1)
        self.assertEqual(summary["by_provenance"], {"weak": 1, "procedural": 1})
        self.assertEqual(summary["by_boundary_source"]["segmentation"], 1)
        self.assertEqual(summary["by_tracking_target_source"], {"unknown": 2})

    def test_quality_interpretation_does_not_overclaim_unlabeled_metrics(self) -> None:
        summary = _label_summary(
            [{"quality_label_source": "unlabeled"}, {"quality_label_source": "unlabeled"}]
        )
        interpretation = _quality_interpretation(summary, 0.0)
        self.assertEqual(interpretation["status"], "unavailable")
        self.assertEqual(interpretation["label_coverage"], 0.0)

        procedural = _label_summary([{"quality_label_source": "procedural_template"}])
        interpretation = _quality_interpretation(procedural, 1.0)
        self.assertEqual(interpretation["status"], "procedural_only")

    def test_metrics_handle_extreme_logits_and_empty_labels_without_runtime_warning(self) -> None:
        outputs = {
            "family_logits": np.full((1, 6), 1e9, dtype=np.float32),
            "phase_logits": np.full((1, 4, 5), -1e9, dtype=np.float32),
            "boundary_logits": np.full((1, 4, 2), 1e9, dtype=np.float32),
            "quality_logits": np.full((1, 4), -1e9, dtype=np.float32),
        }
        targets = {
            "family": np.asarray([-1], dtype=np.int64),
            "phase": np.full((1, 4), -1, dtype=np.int64),
            "boundary": np.full((1, 4, 2), -1.0, dtype=np.float32),
            "quality": np.full((1, 4), np.nan, dtype=np.float32),
            "quality_mask": np.zeros((1, 4), dtype=bool),
            "time_mask": np.ones((1, 4), dtype=bool),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = compute_metrics(outputs, targets)
        self.assertFalse(any(issubclass(item.category, RuntimeWarning) for item in caught))
        self.assertTrue(all(not np.isinf(value) for value in metrics.values()))

    def test_metrics_validate_abstention_threshold_and_composite_score(self) -> None:
        outputs = {
            "family_logits": np.zeros((1, 6), dtype=np.float32),
            "phase_logits": np.zeros((1, 2, 5), dtype=np.float32),
            "boundary_logits": np.zeros((1, 2, 2), dtype=np.float32),
            "quality_logits": np.zeros((1, 4), dtype=np.float32),
        }
        targets = {
            "family": np.asarray([0], dtype=np.int64),
            "phase": np.asarray([[0, 0]], dtype=np.int64),
            "boundary": np.zeros((1, 2, 2), dtype=np.float32),
            "quality": np.zeros((1, 4), dtype=np.float32),
            "quality_mask": np.ones((1, 4), dtype=bool),
            "time_mask": np.ones((1, 2), dtype=bool),
        }
        with self.assertRaises(ValueError):
            compute_metrics(outputs, targets, abstention_threshold=0.0)
        with self.assertRaises(ValueError):
            compute_metrics(outputs, targets, abstention_threshold=1.0)
        metrics = compute_metrics(outputs, targets)
        self.assertGreaterEqual(metrics["family_abstention_rate"], 0.0)
        self.assertLessEqual(metrics["family_abstention_rate"], 1.0)
        self.assertTrue(np.isfinite(composite_score(metrics)))
        self.assertTrue(np.isneginf(composite_score({"family_macro_f1": float("nan")})))

    def test_checkpoint_state_round_trip_preserves_outputs(self) -> None:
        torch.manual_seed(3)
        model = build_model("tcn", self.config).eval()
        values = torch.randn(1, 16, 283)
        with torch.no_grad():
            expected = model(values)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = build_model("tcn", self.config).eval()
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            with torch.no_grad():
                actual = restored(values)

        for key in expected:
            torch.testing.assert_close(expected[key], actual[key])

    def test_one_epoch_training_saves_and_reloads_gru_baseline(self) -> None:
        config = load_config(CONFIG_PATH)
        config["data"]["sources"] = []
        config["data"]["synthetic"]["enabled"] = False
        config["data"]["synthetic_seed"]["participant_count"] = 3
        config["data"]["synthetic_seed"]["exercises_per_participant"] = 1
        config["training"]["max_epochs"] = 1
        config["training"]["batch_size"] = 64

        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            summary = prepare_dataset(config, project_root)
            result = train_model("gru", config, project_root, seed=42, requested_device="cpu")
            checkpoint_path = project_root / "artifacts/checkpoints/gru_baseline.pt"
            latest_checkpoint_path = project_root / "artifacts/checkpoints/gru_latest.pt"
            history_path = project_root / "artifacts/metrics/gru_history.json"
            self.assertGreater(summary["window_counts"]["test"], 0)
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(latest_checkpoint_path.exists())
            self.assertTrue(history_path.exists())
            self.assertGreater(result["test_metrics"]["samples"], 0)
            self.assertIn("validation_sequence_metrics", result["history"][0])
            self.assertIn("validation_sequence_score", result["history"][0])
            self.assertIn("validation_window_score", result["history"][0])
            self.assertIn("validation_sequence_metrics", torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            ))

            dataset = PreparedWindowDataset(
                prepared_split_path(project_root / "data/processed", "test")
            )
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["checkpoint_schema_version"], "adaptfit.checkpoint.v2")
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("rng_state_torch", checkpoint)
            self.assertIn("sampler_epoch", checkpoint)
            restored = build_model("gru", config)
            restored.load_state_dict(checkpoint["model_state_dict"])
            outputs, targets = collect_predictions(restored, loader, torch.device("cpu"))

        self.assertEqual(outputs["family_logits"].shape[0], targets["family"].shape[0])
        self.assertEqual(outputs["phase_logits"].shape[1:], (128, 5))
        self.assertEqual(outputs["tracking_logits"].shape[1:], (128,))
        self.assertEqual(targets["tracking_target"].shape[1:], (128,))

    def test_training_explicitly_falls_back_when_legacy_metadata_is_missing(self) -> None:
        config = load_config(CONFIG_PATH)
        config["data"]["sources"] = []
        config["data"]["synthetic"]["enabled"] = False
        config["data"]["synthetic_seed"]["participant_count"] = 3
        config["data"]["synthetic_seed"]["exercises_per_participant"] = 1
        config["training"]["max_epochs"] = 1
        config["training"]["batch_size"] = 64

        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            prepare_dataset(config, project_root)
            (project_root / "data/processed/validation.jsonl").unlink()
            output = StringIO()
            with redirect_stdout(output):
                result = train_model("gru", config, project_root, seed=42, requested_device="cpu")

        self.assertIsNone(result["history"][0]["validation_sequence_metrics"])
        self.assertTrue(np.isnan(result["history"][0]["validation_sequence_score"]))
        self.assertTrue(np.isfinite(result["history"][0]["validation_window_score"]))
        self.assertIn("window-level validation scoring", output.getvalue())


if __name__ == "__main__":
    unittest.main()
