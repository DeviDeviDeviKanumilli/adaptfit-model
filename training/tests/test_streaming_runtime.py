from __future__ import annotations

import unittest

import torch

from training.src.config import load_config
from training.src.models import CausalStreamingRuntime, build_model
from training.tests.support import CONFIG_PATH


class StreamingRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)

    def test_chunked_runtime_matches_full_causal_temporal_outputs(self) -> None:
        torch.manual_seed(31)
        values = torch.randn(18, 283)
        timestamps = torch.arange(18, dtype=torch.float32) / 30.0
        mask = torch.ones(1, len(values), dtype=torch.bool)

        for name in ("tcn", "gru"):
            model = build_model(name, self.config).eval()
            with torch.no_grad():
                full = model(values.unsqueeze(0), time_mask=mask)
            runtime = CausalStreamingRuntime(model)
            results = [
                runtime.step(
                    values[start:end],
                    timestamps=timestamps[start:end],
                    session_id="session-a",
                    exercise_id="curl",
                )
                for start, end in ((0, 4), (4, 9), (9, 18))
            ]
            for key in ("phase_logits", "boundary_logits", "tracking_logits"):
                chunked = torch.cat([result.outputs[key] for result in results], dim=0)
                torch.testing.assert_close(chunked, full[key].squeeze(0), atol=1e-5, rtol=1e-5)
            self.assertIsNone(results[0].reset_reason)
            self.assertIsNone(results[1].reset_reason)
            self.assertLessEqual(runtime.buffered_context_frames, runtime.retained_context_frames)

    def test_runtime_resets_on_exercise_change_and_accepts_timestamp_gap(self) -> None:
        torch.manual_seed(32)
        values = torch.randn(6, 283)
        timestamps = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.6, 1.7])
        for name in ("tcn", "gru"):
            model = build_model(name, self.config).eval()
            runtime = CausalStreamingRuntime(model)
            runtime.step(
                values[:2], timestamps=timestamps[:2], session_id="s", exercise_id="curl"
            )
            runtime.step(
                values[2:4], timestamps=timestamps[2:4], session_id="s", exercise_id="curl"
            )
            changed = runtime.step(
                values[4:], timestamps=timestamps[4:], session_id="s", exercise_id="row"
            )
            fresh_model = build_model(name, self.config).eval()
            fresh_model.load_state_dict(model.state_dict())
            fresh = CausalStreamingRuntime(fresh_model)
            expected = fresh.step(
                values[4:], timestamps=timestamps[4:], session_id="s", exercise_id="row"
            )
            self.assertEqual(changed.reset_reason, "exercise_changed")
            for key in changed.outputs:
                torch.testing.assert_close(changed.outputs[key], expected.outputs[key])

    def test_runtime_reset_clears_recurrent_and_tcn_context(self) -> None:
        torch.manual_seed(33)
        values = torch.randn(5, 283)
        timestamps = torch.arange(5, dtype=torch.float32)
        for name in ("tcn", "gru"):
            model = build_model(name, self.config).eval()
            runtime = CausalStreamingRuntime(model)
            runtime.step(values, timestamps=timestamps, session_id="s", exercise_id="curl")
            runtime.reset()
            actual = runtime.step(values, timestamps=timestamps, session_id="s", exercise_id="curl")
            fresh_model = build_model(name, self.config).eval()
            fresh_model.load_state_dict(model.state_dict())
            fresh = CausalStreamingRuntime(fresh_model)
            expected = fresh.step(values, timestamps=timestamps, session_id="s", exercise_id="curl")
            for key in actual.outputs:
                torch.testing.assert_close(actual.outputs[key], expected.outputs[key])

    def test_runtime_rejects_bad_chunks_and_nonchronological_timestamps(self) -> None:
        runtime = CausalStreamingRuntime(build_model("gru", self.config).eval())
        with self.assertRaisesRegex(ValueError, "non-empty"):
            runtime.step(torch.zeros(0, 283))
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = torch.zeros(2, 283)
            bad[0, 0] = float("nan")
            runtime.step(bad)
        runtime.step(torch.zeros(2, 283), timestamps=torch.tensor([1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "continue"):
            runtime.step(torch.zeros(1, 283), timestamps=torch.tensor([2.0]))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            runtime.step(torch.zeros(2, 283), timestamps=torch.tensor([3.0, 3.0]))

    def test_runtime_preserves_expert_quality_logits_for_tcn_and_gru(self) -> None:
        torch.manual_seed(34)
        values = torch.randn(8, 283)
        timestamps = torch.arange(8, dtype=torch.float32) / 30.0
        v2_config = dict(self.config)
        v2_config["model"] = dict(self.config["model"])
        v2_config["model"]["expert_quality_classes"] = 5

        for name in ("tcn", "gru"):
            model = build_model(name, v2_config).eval()
            runtime = CausalStreamingRuntime(model)
            result = runtime.step(values, timestamps=timestamps, session_id="s1", exercise_id="curl")
            self.assertIn("expert_quality_logits", result.outputs)
            self.assertEqual(tuple(result.outputs["expert_quality_logits"].shape), (5,))


if __name__ == "__main__":
    unittest.main()
