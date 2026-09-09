from __future__ import annotations

import unittest

from training.src.config import load_config
from training.src.models import build_model
from training.src.runner import _optimizer_for_model, configure_trainable_parameters
from training.tests.support import CONFIG_PATH


class StagedTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_head_only_and_partial_backbone_selection_are_explicit(self) -> None:
        model = build_model("tcn", self.config)
        self.config["training"]["freeze_backbone"] = True
        self.config["training"]["unfreeze_last_blocks"] = 0
        head_only = configure_trainable_parameters(model, self.config)
        self.assertTrue(head_only)
        self.assertTrue(all(name.startswith("heads.") for name in head_only))

        model = build_model("tcn", self.config)
        self.config["training"]["unfreeze_last_blocks"] = 1
        partial = configure_trainable_parameters(model, self.config)
        self.assertTrue(any(name.startswith("blocks.4.") for name in partial))
        self.assertFalse(any(name.startswith("blocks.3.") for name in partial))
        self.assertTrue(any(name.startswith("heads.") for name in partial))

    def test_optimizer_records_separate_head_and_backbone_rates(self) -> None:
        model = build_model("tcn", self.config)
        self.config["training"]["freeze_backbone"] = True
        self.config["training"]["head_learning_rate"] = 3e-4
        self.config["training"]["backbone_learning_rate"] = 3e-5
        configure_trainable_parameters(model, self.config)
        optimizer = _optimizer_for_model(model, self.config)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 3e-4)

        model = build_model("tcn", self.config)
        self.config["training"]["unfreeze_last_blocks"] = 1
        configure_trainable_parameters(model, self.config)
        optimizer = _optimizer_for_model(model, self.config)
        self.assertEqual(
            sorted(group["lr"] for group in optimizer.param_groups),
            [3e-5, 3e-4],
        )


if __name__ == "__main__":
    unittest.main()
