from __future__ import annotations

import unittest

import numpy as np
import torch

from training.src.config import load_config
from training.src.losses import (
    LossWeights,
    compute_loss,
    masked_binary_cross_entropy,
    masked_cross_entropy,
    positive_weight_from_binary,
)
from training.src.models.heads import MultiTaskHeads
from training.src.runner import _class_weights
from training.tests.support import CONFIG_PATH
from training.src.models import build_model, parameter_count


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)

    def test_models_share_output_contract(self) -> None:
        features = torch.randn(2, 128, 283)
        time_mask = torch.ones(2, 128, dtype=torch.bool)
        for name in ("tcn", "gru"):
            model = build_model(name, self.config)
            outputs = model(features, time_mask=time_mask)
            self.assertEqual(outputs["family_logits"].shape, (2, 6))
            self.assertEqual(outputs["phase_logits"].shape, (2, 128, 5))
            self.assertEqual(outputs["boundary_logits"].shape, (2, 128, 2))
            self.assertEqual(outputs["quality_logits"].shape, (2, 4))
            self.assertEqual(outputs["tracking_logits"].shape, (2, 128))
            self.assertGreater(parameter_count(model), 0)

    def test_masked_multitask_loss_is_finite(self) -> None:
        model = build_model("tcn", self.config)
        features = torch.randn(2, 128, 283)
        outputs = model(features)
        batch = {
            "family": torch.tensor([1, 2]),
            "phase": torch.randint(0, 5, (2, 128)),
            "boundary": torch.zeros(2, 128, 2),
            "quality": torch.ones(2, 4),
            "quality_mask": torch.ones(2, 4, dtype=torch.bool),
            "tracking_target": torch.ones(2, 128),
            "time_mask": torch.ones(2, 128, dtype=torch.bool),
        }
        loss, values = compute_loss(outputs, batch, LossWeights())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.tensor(values["tracking"])))
        self.assertIn("total", values)

    def test_tracking_loss_ignores_padded_frames(self) -> None:
        model = build_model("gru", self.config)
        outputs = model(torch.randn(1, 6, 283))
        batch = {
            "family": torch.tensor([1]),
            "phase": torch.zeros(1, 6, dtype=torch.long),
            "boundary": torch.zeros(1, 6, 2),
            "quality": torch.full((1, 4), float("nan")),
            "quality_mask": torch.zeros(1, 4, dtype=torch.bool),
            "tracking_target": torch.tensor([[1.0, 0.0, 1.0, -1.0, -1.0, -1.0]]),
            "time_mask": torch.tensor([[True, True, True, False, False, False]]),
        }
        loss, values = compute_loss(outputs, batch, LossWeights())
        self.assertTrue(torch.isfinite(loss))
        expected = masked_binary_cross_entropy(
            outputs["tracking_logits"][..., :3],
            batch["tracking_target"][..., :3],
            batch["time_mask"][..., :3],
        )
        self.assertAlmostEqual(values["tracking"], float(expected.detach()), places=6)

    def test_masked_mean_pooling_uses_only_observed_context(self) -> None:
        heads = MultiTaskHeads(
            channels=2,
            family_classes=2,
            phase_classes=2,
            quality_outputs=1,
            pooling="masked_mean",
        )
        sequence = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        time_mask = torch.tensor([[True, True, False]])
        outputs = heads(sequence, time_mask=time_mask)
        expected = heads.family(sequence[:, :2].mean(dim=1))
        torch.testing.assert_close(outputs["family_logits"], expected)

    def test_heads_reject_unknown_pooling_mode(self) -> None:
        with self.assertRaises(ValueError):
            MultiTaskHeads(2, 2, 2, 1, pooling="attention")

    def test_heads_reject_wrong_time_mask_shape(self) -> None:
        heads = MultiTaskHeads(2, 2, 2, 1, pooling="masked_mean")
        sequence = torch.randn(2, 4, 2)
        with self.assertRaises(ValueError):
            heads(sequence, time_mask=torch.ones(2, 3, dtype=torch.bool))

    def test_heads_reject_holey_or_empty_time_masks(self) -> None:
        heads = MultiTaskHeads(2, 2, 2, 1, pooling="masked_mean")
        sequence = torch.randn(2, 4, 2)
        with self.assertRaisesRegex(ValueError, "right-padded"):
            heads(sequence, time_mask=torch.tensor([[True, False, True, False], [True] * 4]))
        with self.assertRaisesRegex(ValueError, "at least one valid frame"):
            heads(sequence, time_mask=torch.zeros(2, 4, dtype=torch.bool))

    def test_gru_rejects_holey_or_empty_masks(self) -> None:
        model = build_model("gru", self.config)
        values = torch.randn(2, 8, 283)
        holey = torch.tensor(
            [[True, True, False, True, False, False, False, False], [True] * 8],
            dtype=torch.bool,
        )
        with self.assertRaisesRegex(ValueError, "right-padded"):
            model(values, time_mask=holey)

        empty = torch.zeros(2, 8, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "at least one valid frame"):
            model(values, time_mask=empty)

    def test_sample_weights_remove_weak_examples_from_a_loss(self) -> None:
        logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
        target = torch.tensor([0, 1])
        mask = torch.ones(2, dtype=torch.bool)
        weighted = masked_cross_entropy(
            logits,
            target,
            mask,
            sample_weight=torch.tensor([1.0, 0.0]),
        )
        expected = masked_cross_entropy(logits[:1], target[:1], mask[:1])
        torch.testing.assert_close(weighted, expected)

    def test_sample_weights_remove_weak_examples_from_binary_losses(self) -> None:
        logits = torch.tensor([-8.0, 8.0])
        target = torch.tensor([0.0, 1.0])
        mask = torch.ones(2, dtype=torch.bool)
        weighted = masked_binary_cross_entropy(
            logits,
            target,
            mask,
            sample_weight=torch.tensor([1.0, 0.0]),
        )
        expected = masked_binary_cross_entropy(logits[:1], target[:1], mask[:1])
        torch.testing.assert_close(weighted, expected)

    def test_positive_weight_is_safe_for_empty_and_single_class_labels(self) -> None:
        empty = positive_weight_from_binary(
            torch.tensor([-1.0, -1.0]),
            torch.tensor([True, True]),
        )
        self.assertEqual(float(empty), 1.0)
        all_negative = positive_weight_from_binary(
            torch.zeros(4),
            torch.ones(4, dtype=torch.bool),
        )
        self.assertEqual(float(all_negative), 4.0)
        all_positive = positive_weight_from_binary(
            torch.ones(4),
            torch.ones(4, dtype=torch.bool),
        )
        self.assertEqual(float(all_positive), 1.0)
        capped = positive_weight_from_binary(
            torch.tensor([0.0] * 100 + [1.0]),
            torch.ones(101, dtype=torch.bool),
            max_weight=32.0,
        )
        self.assertEqual(float(capped), 32.0)

    def test_class_weights_ignore_absent_classes(self) -> None:
        weights = _class_weights(np.asarray([1, 1, 2, 2, 2]), classes=4)

        self.assertEqual(float(weights[0]), 0.0)
        self.assertGreater(float(weights[1]), float(weights[2]))
        self.assertGreater(float(weights[1]), 0.25)

    def test_zero_label_weights_skip_valid_targets(self) -> None:
        model = build_model("gru", self.config)
        outputs = model(torch.randn(2, 16, 283))
        batch = {
            "family": torch.tensor([1, 2]),
            "phase": torch.zeros(2, 16, dtype=torch.long),
            "boundary": torch.zeros(2, 16, 2),
            "quality": torch.ones(2, 4),
            "quality_mask": torch.ones(2, 4, dtype=torch.bool),
            "label_weights": torch.zeros(2, 4),
            "time_mask": torch.ones(2, 16, dtype=torch.bool),
        }
        loss, values = compute_loss(outputs, batch, LossWeights())
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(all(value == 0.0 for value in values.values()))


if __name__ == "__main__":
    unittest.main()
