"""Causal streaming inference for the compact temporal models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gru import CausalGRU
from .tcn import CausalTCN


@dataclass(frozen=True)
class StreamingResult:
    """Predictions for one feature chunk and its runtime reset event."""

    outputs: dict[str, torch.Tensor]
    timestamps: torch.Tensor | None
    reset_reason: str | None


class CausalStreamingRuntime:
    """Maintain the minimum state needed to run a model over live chunks.

    TCN inference retains only the model's causal receptive-field history.
    GRU inference retains the recurrent hidden state. A session or exercise ID
    change resets either state before processing the new chunk.
    """

    def __init__(self, model: CausalTCN | CausalGRU):
        if not isinstance(model, (CausalTCN, CausalGRU)):
            raise TypeError("model must be a CausalTCN or CausalGRU")
        self.model = model
        self.model.eval()
        self._feature_buffer: torch.Tensor | None = None
        self._hidden: torch.Tensor | None = None
        self._last_timestamp: float | None = None
        self._session_id: str | None = None
        self._exercise_id: str | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def input_dim(self) -> int:
        if isinstance(self.model, CausalTCN):
            return int(self.model.input_projection.in_channels)
        return int(self.model.gru.input_size)

    @property
    def retained_context_frames(self) -> int:
        if isinstance(self.model, CausalTCN):
            return max(0, self.model.receptive_field - 1)
        return 0

    @property
    def buffered_context_frames(self) -> int:
        """Number of cached TCN feature frames currently retained."""

        return 0 if self._feature_buffer is None else int(self._feature_buffer.shape[0])

    def reset(
        self,
        *,
        session_id: str | None = None,
        exercise_id: str | None = None,
    ) -> None:
        """Clear temporal state and optionally establish new runtime IDs."""

        self._feature_buffer = None
        self._hidden = None
        self._last_timestamp = None
        self._session_id = session_id
        self._exercise_id = exercise_id

    def _validate_chunk(
        self,
        features: torch.Tensor,
        timestamps: torch.Tensor | None,
        *,
        check_timestamp_continuity: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if features.ndim != 2:
            raise ValueError(f"Expected features with shape [time, {self.input_dim}], got {tuple(features.shape)}")
        if features.shape[0] == 0 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected a non-empty [time, {self.input_dim}] feature chunk, got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError("streaming features must be finite")
        model_dtype = next(self.model.parameters()).dtype
        features = features.to(device=self.device, dtype=model_dtype)
        if timestamps is None:
            return features, None
        if timestamps.ndim != 1 or timestamps.shape[0] != features.shape[0]:
            raise ValueError("timestamps must have shape [time] matching features")
        timestamps = timestamps.to(device=features.device, dtype=torch.float32)
        if not torch.isfinite(timestamps).all():
            raise ValueError("streaming timestamps must be finite")
        if timestamps.shape[0] > 1 and not torch.all(timestamps[1:] > timestamps[:-1]):
            raise ValueError("streaming timestamps must be strictly increasing")
        if (
            check_timestamp_continuity
            and self._last_timestamp is not None
            and float(timestamps[0].item()) <= self._last_timestamp
        ):
            raise ValueError("streaming timestamps must continue after the previous chunk")
        return features, timestamps

    def _reset_for_id_change(
        self,
        session_id: str | None,
        exercise_id: str | None,
    ) -> str | None:
        if self._session_id is not None and session_id is not None and session_id != self._session_id:
            self.reset(session_id=session_id, exercise_id=exercise_id)
            return "session_changed"
        if self._exercise_id is not None and exercise_id is not None and exercise_id != self._exercise_id:
            self.reset(session_id=session_id or self._session_id, exercise_id=exercise_id)
            return "exercise_changed"
        if self._session_id is None and session_id is not None:
            self._session_id = session_id
        if self._exercise_id is None and exercise_id is not None:
            self._exercise_id = exercise_id
        return None

    @torch.no_grad()
    def step(
        self,
        features: torch.Tensor,
        *,
        timestamps: torch.Tensor | None = None,
        session_id: str | None = None,
        exercise_id: str | None = None,
    ) -> StreamingResult:
        """Process one non-empty chronological feature chunk."""

        features, timestamps = self._validate_chunk(
            features,
            timestamps,
            check_timestamp_continuity=False,
        )
        reset_reason = self._reset_for_id_change(session_id, exercise_id)
        if (
            reset_reason is None
            and timestamps is not None
            and self._last_timestamp is not None
            and float(timestamps[0].item()) <= self._last_timestamp
        ):
            raise ValueError("streaming timestamps must continue after the previous chunk")
        if isinstance(self.model, CausalTCN):
            context = self._feature_buffer
            if context is None:
                context = features.new_empty((0, self.input_dim))
            model_input = torch.cat((context, features), dim=0).unsqueeze(0)
            time_mask = torch.ones(model_input.shape[:2], dtype=torch.bool, device=model_input.device)
            model_outputs = self.model(model_input, time_mask=time_mask)
            context_length = context.shape[0]
            outputs = {
                "family_logits": model_outputs["family_logits"].squeeze(0),
                "phase_logits": model_outputs["phase_logits"].squeeze(0)[context_length:],
                "boundary_logits": model_outputs["boundary_logits"].squeeze(0)[context_length:],
                "quality_logits": model_outputs["quality_logits"].squeeze(0),
                "tracking_logits": model_outputs["tracking_logits"].squeeze(0)[context_length:],
            }
            if "expert_quality_logits" in model_outputs:
                outputs["expert_quality_logits"] = model_outputs["expert_quality_logits"].squeeze(0)
            retained = torch.cat((context, features), dim=0)
            if self.retained_context_frames:
                self._feature_buffer = retained[-self.retained_context_frames :].detach()
            else:
                self._feature_buffer = retained.new_empty((0, self.input_dim))
        else:
            model_output, self._hidden = self.model.gru(
                features.unsqueeze(0), self._hidden
            )
            time_mask = torch.ones(model_output.shape[:2], dtype=torch.bool, device=model_output.device)
            model_outputs = self.model.heads(model_output, time_mask=time_mask)
            outputs = {key: value.squeeze(0) for key, value in model_outputs.items()}

        if timestamps is not None:
            self._last_timestamp = float(timestamps[-1].item())
        if session_id is not None:
            self._session_id = session_id
        if exercise_id is not None:
            self._exercise_id = exercise_id
        return StreamingResult(outputs=outputs, timestamps=timestamps, reset_reason=reset_reason)
