"""Small causal GRU baseline with the same multi-task output contract."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .heads import MultiTaskHeads


class CausalGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        pooling: str = "last_valid",
        family_classes: int = 6,
        phase_classes: int = 5,
        quality_outputs: int = 4,
        expert_quality_classes: int = 0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.heads = MultiTaskHeads(
            hidden_size,
            family_classes,
            phase_classes,
            quality_outputs,
            expert_quality_classes,
            pooling=pooling,
        )

    def forward(self, values: torch.Tensor, time_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if values.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(values.shape)}")
        if values.shape[-1] != self.gru.input_size:
            raise ValueError(
                f"Expected {self.gru.input_size} input features, got {values.shape[-1]}"
            )
        if time_mask is None:
            output, _ = self.gru(values)
        else:
            if time_mask.shape != values.shape[:2]:
                raise ValueError(
                    f"Expected time_mask shape {tuple(values.shape[:2])}, got {tuple(time_mask.shape)}"
                )
            # Windows are right-padded. Packing prevents padded frames from
            # consuming recurrent compute or changing hidden-state semantics.
            valid_mask = time_mask.bool()
            if not valid_mask.any(dim=1).all():
                raise ValueError("Each sequence must contain at least one valid frame")
            if valid_mask.shape[1] > 1 and torch.any(valid_mask[:, 1:] & ~valid_mask[:, :-1]):
                raise ValueError("GRU time_mask must be right-padded")
            lengths = valid_mask.sum(dim=1).to("cpu")
            packed = pack_padded_sequence(
                values,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.gru(packed)
            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=values.shape[1],
            )
        return self.heads(output, time_mask=time_mask)
