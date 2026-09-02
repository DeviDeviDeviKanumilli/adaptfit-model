"""Shared output heads for temporal movement models."""

from __future__ import annotations

import torch
from torch import nn


class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        channels: int,
        family_classes: int,
        phase_classes: int,
        quality_outputs: int,
        expert_quality_classes: int = 0,
        pooling: str = "last_valid",
    ):
        super().__init__()
        if pooling not in {"last_valid", "masked_mean"}:
            raise ValueError("pooling must be 'last_valid' or 'masked_mean'")
        self.family = nn.Linear(channels, family_classes)
        self.phase = nn.Linear(channels, phase_classes)
        self.boundary = nn.Linear(channels, 2)
        self.quality = nn.Linear(channels, quality_outputs)
        if expert_quality_classes < 0:
            raise ValueError("expert_quality_classes must be non-negative")
        self.expert_quality = (
            nn.Linear(channels, expert_quality_classes)
            if expert_quality_classes
            else None
        )
        # This auxiliary head estimates observable landmark tracking quality.
        # It is intentionally separate from movement-quality labels and uses a
        # self-supervised target derived from confidence and visibility masks.
        self.tracking = nn.Linear(channels, 1)
        self.pooling = pooling

    def forward(self, sequence: torch.Tensor, time_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # sequence is [batch, time, channels]; temporal heads remain causal.
        if time_mask is None:
            valid_mask = torch.ones(sequence.shape[:2], dtype=torch.bool, device=sequence.device)
        else:
            if time_mask.shape != sequence.shape[:2]:
                raise ValueError(
                    f"Expected time_mask shape {tuple(sequence.shape[:2])}, got {tuple(time_mask.shape)}"
                )
            valid_mask = time_mask.bool()
        if not valid_mask.any(dim=1).all():
            raise ValueError("Each sequence must contain at least one valid frame")
        if valid_mask.shape[1] > 1 and torch.any(valid_mask[:, 1:] & ~valid_mask[:, :-1]):
            raise ValueError("time_mask must be right-padded")
        if self.pooling == "masked_mean":
            weights = valid_mask.to(sequence.dtype).unsqueeze(-1)
            pooled = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        else:
            last_indices = valid_mask.long().sum(dim=1).clamp(min=1) - 1
            batch_indices = torch.arange(sequence.shape[0], device=sequence.device)
            pooled = sequence[batch_indices, last_indices]
        outputs = {
            "family_logits": self.family(pooled),
            "phase_logits": self.phase(sequence),
            "boundary_logits": self.boundary(sequence),
            "quality_logits": self.quality(pooled),
            "tracking_logits": self.tracking(sequence).squeeze(-1),
        }
        if self.expert_quality is not None:
            outputs["expert_quality_logits"] = self.expert_quality(pooled)
        return outputs
