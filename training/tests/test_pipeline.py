from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.src.config import load_config
from training.src.data.dataset import PreparedWindowDataset, prepared_split_path
from training.src.data.prepare import assign_splits, prepare_dataset
from training.src.data.procedural import make_procedural_sequences
from training.tests.support import CONFIG_PATH


class PipelineTests(unittest.TestCase):
    def test_group_split_has_no_participant_leakage(self) -> None:
        config = load_config(CONFIG_PATH)
        config["data"]["synthetic_seed"]["participant_count"] = 3
        config["data"]["synthetic_seed"]["exercises_per_participant"] = 1
        sequences = make_procedural_sequences(config)
        entries = assign_splits(sequences, seed=42)
        by_group: dict[str, set[str]] = {}
        for entry in entries:
            group = f"{entry.sequence.source_dataset}:{entry.sequence.participant_id}"
            by_group.setdefault(group, set()).add(entry.split)
        self.assertTrue(all(len(splits) == 1 for splits in by_group.values()))

    def test_prepare_dataset_writes_fixed_windows(self) -> None:
        config = load_config(CONFIG_PATH)
        config = deepcopy(config)
        config["data"]["sources"] = []
        config["data"]["synthetic_seed"]["participant_count"] = 3
        config["data"]["synthetic_seed"]["exercises_per_participant"] = 1
        config["data"]["synthetic"]["copies_per_sequence"] = 0
        with tempfile.TemporaryDirectory() as temporary:
            summary = prepare_dataset(config, Path(temporary))
            self.assertEqual(summary["feature_dimension"], 283)
            self.assertEqual(summary["storage_format"], "npy_memmap")
            for split in ("train", "validation", "test"):
                dataset = PreparedWindowDataset(
                    prepared_split_path(Path(temporary) / "data/processed", split)
                )
                self.assertEqual(dataset.features.shape[1:], (128, 283))
                self.assertEqual(dataset.boundary.shape[-1], 2)
                self.assertEqual(dataset.storage_format, "npy_memmap")
                self.assertIsInstance(dataset.features, np.memmap)
                self.assertEqual(dataset.label_weights.shape, (len(dataset), 4))


if __name__ == "__main__":
    unittest.main()
