from __future__ import annotations

import unittest

import numpy as np

from training.src.decoder import (
    DecoderConfig,
    DecoderState,
    WorkoutEventDecoder,
    boundary_events_from_targets,
    match_event_boundaries,
)


def _phase(name: str) -> tuple[float, ...]:
    values = [0.0] * 5
    values[{"unknown": 0, "rest": 1, "concentric": 2, "hold": 3, "eccentric": 4}[name]] = 1.0
    return tuple(values)


def _decoder(**overrides: object) -> WorkoutEventDecoder:
    values = {
        "min_rep_duration_ms": 200,
        "max_rep_duration_ms": 2_000,
        "refractory_ms": 100,
        "low_tracking_frames": 2,
        "max_timestamp_gap_ms": 500,
        "min_phase_frames": 2,
    }
    values.update(overrides)
    return WorkoutEventDecoder(DecoderConfig(**values))


class DecoderTests(unittest.TestCase):
    def test_clean_rep_requires_apex_and_emits_workout_event(self) -> None:
        decoder = _decoder()
        events = []
        events += decoder.step(
            timestamp_ms=0,
            start_probability=0.9,
            end_probability=0.01,
            phase=_phase("concentric"),
            tracking_confidence=0.95,
            session_id="s",
            exercise_id="curl",
            variant_id="left",
        )
        events += decoder.step(
            timestamp_ms=100,
            start_probability=0.01,
            end_probability=0.01,
            phase=_phase("hold"),
            tracking_confidence=0.95,
            session_id="s",
            exercise_id="curl",
            variant_id="left",
        )
        events += decoder.step(
            timestamp_ms=200,
            start_probability=0.01,
            end_probability=0.01,
            phase=_phase("hold"),
            tracking_confidence=0.95,
            session_id="s",
            exercise_id="curl",
            variant_id="left",
        )
        events += decoder.step(
            timestamp_ms=300,
            start_probability=0.01,
            end_probability=0.9,
            phase=_phase("eccentric"),
            tracking_confidence=0.90,
            session_id="s",
            exercise_id="curl",
            variant_id="left",
            side="left",
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["schema_version"], "workout-event.v1")
        self.assertEqual(events[0]["count_delta"], 1)
        self.assertEqual(events[0]["reason_code"], "decoded_rep")
        self.assertFalse(events[0]["abstention"])
        self.assertEqual(decoder.state, DecoderState.REFRACTORY)

    def test_partial_rep_without_apex_is_abstained(self) -> None:
        decoder = _decoder()
        decoder.step(
            timestamp_ms=0,
            start_probability=0.9,
            end_probability=0.0,
            phase=_phase("concentric"),
            tracking_confidence=0.9,
            session_id="s",
            exercise_id="curl",
        )
        event = decoder.step(
            timestamp_ms=300,
            start_probability=0.0,
            end_probability=0.9,
            phase=_phase("eccentric"),
            tracking_confidence=0.9,
            session_id="s",
            exercise_id="curl",
        )[0]
        self.assertEqual(event["count_delta"], 0)
        self.assertTrue(event["abstention"])
        self.assertEqual(event["reason_code"], "partial_without_apex")

    def test_low_tracking_pause_and_timestamp_gap_do_not_count(self) -> None:
        decoder = _decoder()
        decoder.step(
            timestamp_ms=0,
            start_probability=0.9,
            end_probability=0.0,
            phase=_phase("concentric"),
            tracking_confidence=0.9,
            session_id="s",
            exercise_id="curl",
        )
        decoder.step(
            timestamp_ms=100,
            start_probability=0.0,
            end_probability=0.0,
            phase=_phase("hold"),
            tracking_confidence=0.1,
            session_id="s",
            exercise_id="curl",
        )
        events = decoder.step(
            timestamp_ms=200,
            start_probability=0.0,
            end_probability=0.0,
            phase=_phase("hold"),
            tracking_confidence=0.1,
            session_id="s",
            exercise_id="curl",
        )
        self.assertTrue(any(item["reason_code"] == "low_tracking" for item in events))
        self.assertEqual(decoder.state, DecoderState.PAUSED)

        decoder.reset()
        decoder.step(
            timestamp_ms=0,
            start_probability=0.9,
            end_probability=0.0,
            phase=_phase("concentric"),
            tracking_confidence=0.9,
            session_id="s",
            exercise_id="curl",
        )
        events = decoder.step(
            timestamp_ms=700,
            start_probability=0.0,
            end_probability=0.0,
            phase=_phase("rest"),
            tracking_confidence=0.9,
            session_id="s",
            exercise_id="curl",
        )
        self.assertTrue(any(item["reason_code"] == "timestamp_gap" for item in events))

    def test_context_reset_and_duplicate_timestamp_are_safe(self) -> None:
        decoder = _decoder()
        decoder.step(
            timestamp_ms=0,
            start_probability=0.9,
            end_probability=0.0,
            phase=_phase("concentric"),
            tracking_confidence=0.9,
            session_id="s",
            exercise_id="curl",
        )
        events = decoder.step(
            timestamp_ms=100,
            start_probability=0.0,
            end_probability=0.0,
            phase=_phase("rest"),
            tracking_confidence=0.9,
            session_id="s",
            exercise_id="row",
        )
        self.assertTrue(any(item["reason_code"] == "context_reset" for item in events))
        duplicate = decoder.step(
            timestamp_ms=100,
            start_probability=0.9,
            end_probability=0.9,
            phase=_phase("hold"),
            tracking_confidence=0.9,
            session_id="s",
            exercise_id="row",
        )
        self.assertEqual(duplicate, [])

    def test_validation_helpers_pair_boundaries_and_match_once(self) -> None:
        targets = np.zeros((8, 2), dtype=np.float32)
        targets[1, 0] = 1.0
        targets[5, 1] = 1.0
        target_pairs = boundary_events_from_targets(targets, np.arange(8) * 100)
        self.assertEqual(target_pairs, [(100, 500)])
        event = {
            "start_timestamp_ms": 110,
            "end_timestamp_ms": 510,
            "count_delta": 1,
            "abstention": False,
        }
        metrics = match_event_boundaries([event, event], target_pairs, tolerance_ms=20)
        self.assertEqual(metrics["matched_count"], 1.0)
        self.assertEqual(metrics["false_event_count"], 1.0)
        self.assertEqual(metrics["f1"], 2.0 / 3.0)

    def test_decoder_rejects_invalid_probability_contract(self) -> None:
        decoder = _decoder()
        with self.assertRaises(ValueError):
            decoder.step(
                timestamp_ms=0,
                start_probability=1.5,
                end_probability=0.0,
                phase=_phase("rest"),
                tracking_confidence=0.9,
                session_id="s",
                exercise_id="curl",
            )


if __name__ == "__main__":
    unittest.main()
