from __future__ import annotations

import unittest

import numpy as np

from training.src.data.procedural import make_procedural_sequences
from training.src.data.schema import CANONICAL_INDEX
from training.src.features.anatomy import (
    FEATURE_DIM,
    LIMB_JOINTS,
    _rolling_range,
    continuous_feature_indices,
    extract_features,
    feature_dimensions,
    feature_schema,
    tracking_confidence_target,
)
from training.src.features.anatomy import CAPABILITY_WEIGHTS
from training.src.features.diagnostics import extract_observable_diagnostics
from training.src.config import load_config
from training.tests.support import CONFIG_PATH


class FeatureTests(unittest.TestCase):
    def test_vectorized_rolling_range_matches_causal_reference(self) -> None:
        rng = np.random.default_rng(11)
        values = rng.normal(size=(23, 14)).astype(np.float32)
        valid = rng.random((23, 14)) > 0.25
        actual = _rolling_range(values, valid, frames=7)
        expected = np.zeros_like(values)
        for index in range(len(values)):
            start = max(0, index - 7 + 1)
            for column in range(values.shape[1]):
                selected = values[start : index + 1, column][valid[start : index + 1, column]]
                if selected.size:
                    expected[index, column] = selected.max() - selected.min()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_feature_shape_and_finiteness(self) -> None:
        config = load_config(CONFIG_PATH)
        sequence = make_procedural_sequences(config)[0]
        features = extract_features(sequence)
        self.assertEqual(features.shape[1], FEATURE_DIM)
        self.assertEqual(features.shape[1], 283)
        self.assertTrue(np.isfinite(features).all())

    def test_absent_limb_is_masked_without_nan(self) -> None:
        config = load_config(CONFIG_PATH)
        sequence = make_procedural_sequences(config)[0]
        sequence.capability_states["right_arm"] = "absent"
        features = extract_features(sequence)
        schema = feature_schema()
        capability_group = next(group for group in schema["groups"] if group["name"] == "capability_mask")
        values = features[:, capability_group["start"] : capability_group["end"]]
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(float(values[:, 12:23].sum()), float(values.size))

    def test_schema_groups_cover_feature_vector(self) -> None:
        schema = feature_schema()
        self.assertEqual(schema["dimension"], 283)
        self.assertEqual(schema["groups"][-1]["end"], 283)
        self.assertEqual(schema["nominal_fps"], 30.0)
        with self.assertRaises(ValueError):
            feature_schema(fps=0.0)

    def test_limited_and_assisted_states_change_joint_capability_values(self) -> None:
        config = load_config(CONFIG_PATH)
        sequence = make_procedural_sequences(config)[0]
        sequence.capability_states["left_arm"] = "limited"
        sequence.capability_states["right_leg"] = "assisted"
        features = extract_features(sequence)
        schema = feature_schema()
        group = next(item for item in schema["groups"] if item["name"] == "capability_mask")
        values = features[:, group["start"] : group["end"]]
        self.assertTrue(
            np.all(values[:, CANONICAL_INDEX["left_wrist"]] == CAPABILITY_WEIGHTS["limited"])
        )
        self.assertTrue(
            np.all(values[:, CANONICAL_INDEX["right_ankle"]] == CAPABILITY_WEIGHTS["assisted"])
        )

    def test_observable_diagnostics_are_finite_and_frame_aligned(self) -> None:
        config = load_config(CONFIG_PATH)
        sequence = make_procedural_sequences(config)[0]
        diagnostics = extract_observable_diagnostics(sequence, rolling_frames=9)
        self.assertEqual(diagnostics["acceleration_xy"].shape, (sequence.frames, 66))
        self.assertEqual(diagnostics["angular_acceleration"].shape, (sequence.frames, 14))
        self.assertEqual(diagnostics["rolling_rom"].shape, (sequence.frames, 14))
        for values in diagnostics.values():
            self.assertEqual(values.shape[0], sequence.frames)
            self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all((diagnostics["smoothness"] > 0.0) & (diagnostics["smoothness"] <= 1.0)))

    def test_diagnostic_feature_variant_preserves_base_order_and_width(self) -> None:
        config = load_config(CONFIG_PATH)
        sequence = make_procedural_sequences(config)[0]
        base = extract_features(sequence)
        diagnostic = extract_features(sequence, include_diagnostics=True)
        diagnostic_dimension, continuous_dimension = feature_dimensions(True)
        schema = feature_schema(include_diagnostics=True)

        self.assertEqual(diagnostic.shape, (sequence.frames, diagnostic_dimension))
        self.assertEqual(diagnostic_dimension, 378)
        self.assertEqual(continuous_dimension, 288)
        self.assertEqual(schema["version"], "adaptfit.features.v2.diagnostics")
        self.assertEqual(schema["dimension"], diagnostic_dimension)
        self.assertEqual(schema["continuous_dimension"], continuous_dimension)
        self.assertEqual(len(schema["continuous_indices"]), continuous_dimension)
        self.assertEqual(schema["groups"][-1]["end"], diagnostic_dimension)
        np.testing.assert_array_equal(diagnostic[:, :FEATURE_DIM], base)
        self.assertTrue(np.isfinite(diagnostic).all())

    def test_diagnostic_features_match_observable_diagnostics_api(self) -> None:
        config = load_config(CONFIG_PATH)
        sequence = make_procedural_sequences(config)[0]
        features = extract_features(sequence, include_diagnostics=True)
        schema = feature_schema(include_diagnostics=True)
        diagnostics = extract_observable_diagnostics(sequence)
        for name in ("acceleration_xy", "angular_acceleration", "rolling_rom", "smoothness"):
            group = next(item for item in schema["groups"] if item["name"] == name)
            values = features[:, group["start"] : group["end"]]
            expected = diagnostics[name]
            if expected.ndim == 1:
                expected = expected[:, None]
            np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-6)

    def test_diagnostic_normalization_indices_skip_masks_and_context(self) -> None:
        base_indices = continuous_feature_indices()
        diagnostic_indices = continuous_feature_indices(include_diagnostics=True)
        self.assertEqual(len(base_indices), 193)
        self.assertEqual(len(diagnostic_indices), 288)
        np.testing.assert_array_equal(base_indices, np.arange(193))
        np.testing.assert_array_equal(diagnostic_indices[:193], np.arange(193))
        self.assertTrue(np.all(diagnostic_indices[193:] >= FEATURE_DIM))

    def test_tracking_target_respects_capability_profile_and_observation(self) -> None:
        config = load_config(CONFIG_PATH)
        sequence = next(
            item
            for item in make_procedural_sequences(config)
            if item.metadata.get("family_name") == "arm_flexion"
        )
        baseline = tracking_confidence_target(sequence)
        np.testing.assert_allclose(baseline, 1.0, rtol=0.0, atol=1e-6)

        absent_limb = next(
            limb for limb, state in sequence.capability_states.items() if state == "absent"
        )
        absent_indices = [CANONICAL_INDEX[name] for name in LIMB_JOINTS[absent_limb]]
        absent_joints = sequence.joints.copy()
        absent_confidence = sequence.pose_confidence.copy()
        absent_observed = sequence.observed_mask.copy()
        absent_joints[:, absent_indices] = np.nan
        absent_confidence[:, absent_indices] = 0.0
        absent_observed[:, absent_indices] = False
        absent_only = sequence.clone(
            joints=absent_joints,
            pose_confidence=absent_confidence,
            observed_mask=absent_observed,
        )
        np.testing.assert_allclose(tracking_confidence_target(absent_only), baseline)

        expected_index = CANONICAL_INDEX[
            "left_wrist" if absent_limb == "right_arm" else "right_wrist"
        ]
        expected_joints = sequence.joints.copy()
        expected_confidence = sequence.pose_confidence.copy()
        expected_observed = sequence.observed_mask.copy()
        expected_joints[:, expected_index] = np.nan
        expected_confidence[:, expected_index] = 0.0
        expected_observed[:, expected_index] = False
        expected_missing = sequence.clone(
            joints=expected_joints,
            pose_confidence=expected_confidence,
            observed_mask=expected_observed,
        )
        self.assertTrue(np.all(tracking_confidence_target(expected_missing) < baseline))

    def test_tracking_target_uses_partial_confidence_and_handles_all_absent_profile(self) -> None:
        config = load_config(CONFIG_PATH)
        sequence = next(
            item
            for item in make_procedural_sequences(config)
            if item.metadata.get("family_name") == "reach"
        )
        confidence = sequence.pose_confidence.copy()
        absent_limb = next(
            limb for limb, state in sequence.capability_states.items() if state == "absent"
        )
        absent_indices = {CANONICAL_INDEX[name] for name in LIMB_JOINTS[absent_limb]}
        expected_indices = sorted(set(range(33)) - absent_indices)
        confidence[:, expected_indices] = 0.25
        partial = sequence.clone(pose_confidence=confidence)
        np.testing.assert_allclose(tracking_confidence_target(partial), 0.25, atol=1e-6)

        all_absent = sequence.clone(
            capability_states={limb: "absent" for limb in sequence.capability_states}
        )
        # The profile removes limb joints from the expected set, but the
        # camera may still observe the head and trunk.
        np.testing.assert_allclose(tracking_confidence_target(all_absent), 1.0, atol=1e-6)

    def test_dropped_landmarks_keep_features_finite_and_lower_observability(self) -> None:
        config = load_config(CONFIG_PATH)
        sequence = make_procedural_sequences(config)[0]
        start, end = 10, 15
        active_arm = "left_arm" if sequence.capability_states["left_arm"] != "absent" else "right_arm"
        index = CANONICAL_INDEX[f"{active_arm.removesuffix('_arm')}_wrist"]
        joints = sequence.joints.copy()
        confidence = sequence.pose_confidence.copy()
        observed = sequence.observed_mask.copy()
        joints[start:end, index] = np.nan
        confidence[start:end, index] = 0.0
        observed[start:end, index] = False
        dropped = sequence.clone(
            joints=joints,
            pose_confidence=confidence,
            observed_mask=observed,
        )
        features = extract_features(dropped)
        target = tracking_confidence_target(dropped)
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(target[start:end] < target[:start].min()))


if __name__ == "__main__":
    unittest.main()
