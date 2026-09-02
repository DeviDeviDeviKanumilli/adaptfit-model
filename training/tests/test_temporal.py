from __future__ import annotations

import unittest

import numpy as np

from training.src.config import ConfigError, load_config, validate_config
from training.src.data.schema import CanonicalSequence
from training.src.data.temporal import resample_sequence
from training.tests.support import CONFIG_PATH


class TemporalPreprocessingTests(unittest.TestCase):
    @staticmethod
    def _sequence() -> CanonicalSequence:
        frames = 6
        joints = np.zeros((frames, 33, 2), dtype=np.float32)
        joints[:, 15, 0] = np.arange(frames, dtype=np.float32)
        joints[:, 16, 0] = np.arange(frames, dtype=np.float32) * 10.0
        confidence = np.ones((frames, 33), dtype=np.float32)
        observed = np.ones((frames, 33), dtype=bool)
        # One missing frame is short enough to interpolate.
        joints[2, 15] = np.nan
        confidence[2, 15] = 0.0
        observed[2, 15] = False
        # Two missing frames remain masked when max_interpolation_gap=1.
        joints[2:4, 16] = np.nan
        confidence[2:4, 16] = 0.0
        observed[2:4, 16] = False
        # No landmarks at all in this long gap means labels are masked too.
        joints[2:4] = np.nan
        confidence[2:4] = 0.0
        observed[2:4] = False
        return CanonicalSequence(
            joints=joints,
            pose_confidence=confidence,
            observed_mask=observed,
            timestamps=np.arange(frames, dtype=np.float32) / 5.0,
            phase=np.zeros(frames, dtype=np.int64),
            rep_boundary=np.zeros((frames, 2), dtype=np.float32),
        )

    def test_short_gaps_are_interpolated_and_long_gaps_remain_masked(self) -> None:
        sequence = self._sequence()
        restored = resample_sequence(sequence, target_fps=5.0, max_interpolation_gap=1)

        self.assertEqual(restored.frames, sequence.frames)
        np.testing.assert_allclose(restored.timestamps, sequence.timestamps)
        # The all-landmark gap is intentionally long and is not fabricated.
        self.assertFalse(restored.observed_mask[2].any())
        self.assertFalse(restored.observed_mask[3].any())
        self.assertTrue(np.isnan(restored.joints[2:4]).all())
        self.assertTrue(np.all(restored.phase[2:4] == -1))
        self.assertTrue(np.all(restored.rep_boundary[2:4] == -1.0))

    def test_one_landmark_gap_interpolates_when_other_landmarks_are_present(self) -> None:
        sequence = self._sequence()
        sequence.joints[2, 15] = np.nan
        sequence.pose_confidence[2, 15] = 0.0
        sequence.observed_mask[2, 15] = False
        sequence.joints[3, 15] = np.asarray([3.0, 0.0], dtype=np.float32)
        sequence.pose_confidence[3, 15] = 1.0
        sequence.observed_mask[3, 15] = True
        # Keep the rest of the body visible so frame-level labels remain usable.
        sequence.joints[2, 16] = sequence.joints[1, 16]
        sequence.pose_confidence[2, 16] = 1.0
        sequence.observed_mask[2, 16] = True
        sequence.joints[3, 16] = sequence.joints[4, 16]
        sequence.pose_confidence[3, 16] = 1.0
        sequence.observed_mask[3, 16] = True

        restored = resample_sequence(sequence, target_fps=5.0, max_interpolation_gap=1)
        self.assertTrue(restored.observed_mask[2, 15])
        self.assertAlmostEqual(float(restored.joints[2, 15, 0]), 2.0, places=5)
        self.assertGreater(float(restored.pose_confidence[2, 15]), 0.0)

    def test_capability_absence_is_not_created_by_resampling(self) -> None:
        sequence = self._sequence()
        sequence.capability_states["right_arm"] = "absent"
        right_arm = [12, 14, 16, 18, 20, 22]
        sequence.joints[:, right_arm] = np.nan
        sequence.pose_confidence[:, right_arm] = 0.0
        sequence.observed_mask[:, right_arm] = False

        restored = resample_sequence(sequence, target_fps=5.0, max_interpolation_gap=5)
        self.assertEqual(restored.capability_states["right_arm"], "absent")
        self.assertFalse(restored.observed_mask[:, right_arm].any())
        self.assertTrue(np.isnan(restored.joints[:, right_arm]).all())

    def test_irregular_clip_end_is_preserved(self) -> None:
        sequence = self._sequence().clone(
            timestamps=np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.1], dtype=np.float32)
        )
        restored = resample_sequence(sequence, target_fps=5.0, max_interpolation_gap=1)
        self.assertEqual(restored.frames, 7)
        self.assertAlmostEqual(float(restored.timestamps[-1]), 1.1, places=5)
        self.assertTrue(np.all(np.diff(restored.timestamps) > 0.0))

    def test_temporal_config_rejects_invalid_sampling_settings(self) -> None:
        config = load_config(CONFIG_PATH)
        config["data"]["target_fps"] = 0
        with self.assertRaises(ConfigError):
            validate_config(config)

        config = load_config(CONFIG_PATH)
        config["data"]["max_interpolation_gap"] = -1
        with self.assertRaises(ConfigError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
