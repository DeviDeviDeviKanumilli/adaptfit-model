from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from training.preflight import run_preflight
from training.src.config import (
    ConfigError,
    load_config,
    resolve_project_root,
    validate_checkpoint_compatibility,
    validate_config,
)
from training.src.data.adapters import MMFIT_SOURCE_NAMES, load_canonical_npz_files
from training.src.data.procedural import make_procedural_sequences
from training.tests.support import CONFIG_PATH, PROJECT_ROOT

V2_CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "v2_diagnostics.yaml"


class ConfigAndPreflightTests(unittest.TestCase):
    def test_config_validation_rejects_contract_drift(self) -> None:
        config = load_config(CONFIG_PATH)
        config["features"]["input_dim"] = 282
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_config_validation_rejects_non_boolean_label_policy(self) -> None:
        config = deepcopy(load_config(CONFIG_PATH))
        config["data"]["label_policy"]["allow_procedural_quality"] = "yes"
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_config_validation_rejects_non_float32_training(self) -> None:
        config = deepcopy(load_config(CONFIG_PATH))
        config["training"]["float32"] = False
        with self.assertRaisesRegex(ConfigError, "float32"):
            validate_config(config)
        config["training"]["float32"] = "yes"
        with self.assertRaisesRegex(ConfigError, "float32"):
            validate_config(config)

    def test_config_validation_rejects_invalid_loss_weights(self) -> None:
        for value in (-0.1, float("nan"), True, "not-a-number"):
            with self.subTest(value=value):
                config = deepcopy(load_config(CONFIG_PATH))
                config["training"]["loss_weights"]["tracking"] = value
                with self.assertRaises(ConfigError):
                    validate_config(config)

        config = deepcopy(load_config(CONFIG_PATH))
        config["training"]["loss_weights"]["unknown"] = 1.0
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_checkpoint_compatibility_rejects_feature_and_architecture_drift(self) -> None:
        config = load_config(CONFIG_PATH)
        checkpoint = {
            "model_name": "tcn",
            "config": deepcopy(config),
        }
        validate_checkpoint_compatibility(checkpoint, config)

        changed = deepcopy(config)
        changed["features"]["input_dim"] = 378
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_checkpoint_compatibility(checkpoint, changed)

        changed = deepcopy(config)
        changed["model"]["tcn"]["channels"] += 1
        with self.assertRaisesRegex(ValueError, "model.tcn.channels"):
            validate_checkpoint_compatibility(checkpoint, changed)

    def test_checkpoint_compatibility_requires_effective_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "effective training config"):
            validate_checkpoint_compatibility({"model_name": "tcn"}, load_config(CONFIG_PATH))

    def test_train_cli_validates_overrides_before_starting(self) -> None:
        from training.train import main as train_main

        with self.assertRaises(ConfigError):
            train_main(
                ["--config", str(CONFIG_PATH), "--max-epochs", "0", "--models", ""]
            )

    def test_diagnostics_config_validates_against_its_expanded_schema(self) -> None:
        config = load_config(V2_CONFIG_PATH)
        self.assertTrue(config["features"]["include_diagnostics"])
        self.assertEqual(config["features"]["input_dim"], 378)
        self.assertEqual(config["features"]["continuous_dim"], 288)

    def test_project_root_is_inferred_from_config_path(self) -> None:
        self.assertEqual(resolve_project_root(CONFIG_PATH), PROJECT_ROOT)

    def test_preflight_requires_both_bootstrap_roles(self) -> None:
        config = deepcopy(load_config(CONFIG_PATH))
        config["data"]["sources"] = []
        with tempfile.TemporaryDirectory() as temporary:
            result = run_preflight(config, Path(temporary))
        self.assertFalse(result["ok"])
        self.assertTrue(any("rep-labeled" in error for error in result["errors"]))
        self.assertTrue(any("seated/wheelchair" in error for error in result["errors"]))

    def test_invalid_canonical_npz_is_reported_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "broken.npz").write_text("not a numpy archive", encoding="utf-8")
            with self.assertLogs("training.src.data.adapters", level="WARNING") as logs:
                sequences = load_canonical_npz_files(root)
        self.assertEqual(sequences, [])
        self.assertIn("Skipping invalid canonical NPZ", "\n".join(logs.output))

    def test_preflight_detects_canonical_sources_for_both_required_roles(self) -> None:
        config = deepcopy(load_config(CONFIG_PATH))
        config["data"]["sources"] = [
            {
                "name": "rep_fixture",
                "path": "data/raw/rep_fixture",
                "role": "rep_labeled",
                "enabled": True,
            },
            {
                "name": "seated_fixture",
                "path": "data/raw/seated_fixture",
                "role": "seated_or_wheelchair",
                "enabled": True,
            },
        ]
        sequence = make_procedural_sequences(config)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence.to_npz(root / "data/raw/rep_fixture/rep.npz")
            sequence.to_npz(root / "data/raw/seated_fixture/seated.npz")
            result = run_preflight(config, root)

        self.assertTrue(result["ok"])
        self.assertEqual(
            {item["name"] for item in result["sources"] if item["available"]},
            {"rep_fixture", "seated_fixture"},
        )
        self.assertTrue(
            all(item["sequence_count"] == 1 for item in result["sources"] if item["available"])
        )

    def test_preflight_does_not_treat_invalid_canonical_files_as_available(self) -> None:
        config = deepcopy(load_config(CONFIG_PATH))
        config["data"]["sources"] = [
            {
                "name": "broken_rep",
                "path": "data/raw/broken_rep",
                "role": "rep_labeled",
                "enabled": True,
            },
            {
                "name": "broken_seated",
                "path": "data/raw/broken_seated",
                "role": "seated_or_wheelchair",
                "enabled": True,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("data/raw/broken_rep/bad.npz", "data/raw/broken_seated/bad.npz"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("not a canonical archive", encoding="utf-8")
            with self.assertLogs("training.src.data.adapters", level="WARNING"):
                result = run_preflight(config, root)

        self.assertFalse(result["ok"])
        self.assertTrue(all(not item["available"] for item in result["sources"]))
        self.assertTrue(all(item["sequence_count"] == 0 for item in result["sources"]))
        self.assertTrue(any("No rep-labeled source" in error for error in result["errors"]))

    def test_preflight_inspects_mmfit_archive_contents(self) -> None:
        config = deepcopy(load_config(CONFIG_PATH))
        config["data"]["sources"] = [
            {
                "name": "mmfit",
                "path": "data/raw/mmfit",
                "role": "rep_labeled",
                "enabled": True,
            },
            {
                "name": "seated_fixture",
                "path": "data/raw/seated_fixture",
                "role": "seated_or_wheelchair",
                "enabled": True,
            },
        ]
        frames = 6
        raw = np.zeros((2, frames, len(MMFIT_SOURCE_NAMES) + 1), dtype=np.float32)
        raw[0, :, 0] = np.arange(frames, dtype=np.float32)
        raw[1, :, 0] = np.arange(frames, dtype=np.float32)
        raw[:, :, 1:] = 1.0
        pose_buffer = BytesIO()
        np.save(pose_buffer, raw)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mmfit_root = root / "data/raw/mmfit"
            mmfit_root.mkdir(parents=True)
            with ZipFile(mmfit_root / "mm-fit.zip", "w") as archive:
                archive.writestr("w00/w00_pose_2d.npy", pose_buffer.getvalue())
                archive.writestr("w00/w00_labels.csv", "0,5,2,bicep_curls\n")
            sequence = make_procedural_sequences(config)[0]
            sequence.to_npz(root / "data/raw/seated_fixture/seated.npz")

            result = run_preflight(config, root)

        self.assertTrue(result["ok"])
        mmfit_status = next(item for item in result["sources"] if item["name"] == "mmfit")
        self.assertTrue(mmfit_status["file_present"])
        self.assertTrue(mmfit_status["available"])
        self.assertEqual(mmfit_status["sequence_count"], 1)

    def test_file_only_preflight_defers_source_decoding(self) -> None:
        config = deepcopy(load_config(CONFIG_PATH))
        config["data"]["sources"] = [
            {
                "name": "rep_fixture",
                "path": "data/raw/rep_fixture",
                "role": "rep_labeled",
                "enabled": True,
            },
            {
                "name": "seated_fixture",
                "path": "data/raw/seated_fixture",
                "role": "seated_or_wheelchair",
                "enabled": True,
            },
        ]
        sequence = make_procedural_sequences(config)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence.to_npz(root / "data/raw/rep_fixture/rep.npz")
            sequence.to_npz(root / "data/raw/seated_fixture/seated.npz")
            result = run_preflight(config, root, inspect_sequences=False)

        self.assertTrue(result["ok"])
        self.assertEqual(result["inspection_mode"], "files")
        self.assertTrue(all(item["available"] for item in result["sources"]))
        self.assertTrue(all(item["sequence_count"] is None for item in result["sources"]))


if __name__ == "__main__":
    unittest.main()
