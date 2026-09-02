"""Masked multi-task losses for partially labeled movement sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    family: float = 0.5
    phase: float = 1.0
    boundary: float = 2.0
    quality: float = 1.0
    expert_quality: float = 0.0
    tracking: float = 0.2


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    return values.mean() if values.numel() else values.new_zeros(())


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = mask & (target >= 0)
    if sample_weight is not None:
        valid = valid & (sample_weight > 0)
    # Keep the tensor shape fixed instead of boolean-indexing valid elements.
    # The latter creates dynamic MPS graphs and `valid.any()` synchronizes the
    # accelerator on every batch. Invalid targets are clamped only for the
    # reduction and receive zero weight below.
    target_safe = target.to(dtype=torch.long).clamp(min=0, max=logits.shape[-1] - 1)
    per_item = F.cross_entropy(
        logits,
        target_safe,
        weight=class_weights,
        reduction="none",
    )
    weights = valid.to(dtype=per_item.dtype)
    if sample_weight is not None:
        weights = weights * sample_weight.to(dtype=per_item.dtype)
    return (per_item * weights).sum() / weights.sum().clamp(min=1e-8)


def masked_binary_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = mask & (target >= 0)
    if sample_weight is not None:
        valid = valid & (sample_weight > 0)
    # As above, use a dense fixed-shape reduction to avoid per-batch MPS
    # synchronization and dynamic indexing. NaN or sentinel targets are
    # replaced before BCE and contribute zero through the validity weights.
    selected_target = torch.where(valid, target, torch.zeros_like(target)).to(
        dtype=logits.dtype
    ).clamp(min=0.0, max=1.0)
    if positive_weight is None:
        class_weights = torch.ones_like(selected_target)
    else:
        class_weights = torch.where(
            selected_target > 0.5,
            positive_weight.to(device=logits.device, dtype=logits.dtype),
            torch.ones_like(selected_target),
        )
    weights = valid.to(dtype=logits.dtype) * class_weights
    if sample_weight is not None:
        weights = weights * sample_weight.to(dtype=logits.dtype)
    losses = F.binary_cross_entropy_with_logits(
        logits,
        selected_target,
        reduction="none",
    )
    return (losses * weights).sum() / weights.sum().clamp(min=1e-8)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: LossWeights,
    family_class_weights: torch.Tensor | None = None,
    phase_class_weights: torch.Tensor | None = None,
    boundary_positive_weight: torch.Tensor | None = None,
    quality_positive_weights: torch.Tensor | None = None,
    expert_quality_class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    time_mask = batch["time_mask"].bool()
    label_weights = batch.get(
        "label_weights",
        torch.ones(
            (batch["family"].shape[0], 4),
            dtype=outputs["family_logits"].dtype,
            device=outputs["family_logits"].device,
        ),
    )
    family_loss = masked_cross_entropy(
        outputs["family_logits"],
        batch["family"],
        torch.ones_like(batch["family"], dtype=torch.bool),
        family_class_weights,
        sample_weight=label_weights[:, 0],
    )
    phase_loss = masked_cross_entropy(
        outputs["phase_logits"].reshape(-1, outputs["phase_logits"].shape[-1]),
        batch["phase"].reshape(-1),
        time_mask.reshape(-1),
        phase_class_weights,
        sample_weight=label_weights[:, 1, None].expand_as(batch["phase"]).reshape(-1),
    )
    boundary_loss = masked_binary_cross_entropy(
        outputs["boundary_logits"],
        batch["boundary"],
        time_mask[..., None].expand_as(batch["boundary"]),
        boundary_positive_weight,
        sample_weight=label_weights[:, 2, None, None].expand_as(batch["boundary"]),
    )

    quality_logits = outputs["quality_logits"]
    quality_target = batch["quality"]
    quality_mask = batch["quality_mask"].bool()
    quality_losses = []
    for index in range(quality_logits.shape[-1]):
        positive_weight = None
        if quality_positive_weights is not None:
            positive_weight = quality_positive_weights[index]
        quality_losses.append(
            masked_binary_cross_entropy(
                quality_logits[:, index],
                quality_target[:, index],
                quality_mask[:, index],
                positive_weight,
                sample_weight=label_weights[:, 3],
            )
        )
    quality_loss = _safe_mean(torch.stack(quality_losses)) if quality_losses else quality_logits.sum() * 0.0

    tracking_logits = outputs.get("tracking_logits")
    tracking_target = batch.get("tracking_target")
    if tracking_logits is not None and tracking_target is not None:
        tracking_loss = masked_binary_cross_entropy(
            tracking_logits,
            tracking_target,
            time_mask,
        )
    elif tracking_logits is not None:
        tracking_loss = tracking_logits.sum() * 0.0
    else:
        tracking_loss = quality_logits.sum() * 0.0

    expert_quality_logits = outputs.get("expert_quality_logits")
    expert_quality_target = batch.get("expert_quality")
    expert_quality_mask = batch.get("expert_quality_mask")
    expert_label_weight = batch.get("expert_label_weight")
    if (
        expert_quality_logits is not None
        and expert_quality_target is not None
        and expert_quality_mask is not None
    ):
        if expert_label_weight is None:
            expert_label_weight = torch.ones_like(
                expert_quality_target,
                dtype=expert_quality_logits.dtype,
            )
        expert_quality_loss = masked_cross_entropy(
            expert_quality_logits,
            expert_quality_target,
            expert_quality_mask.bool(),
            expert_quality_class_weights,
            sample_weight=expert_label_weight,
        )
    elif expert_quality_logits is not None:
        expert_quality_loss = expert_quality_logits.sum() * 0.0
    else:
        expert_quality_loss = quality_logits.sum() * 0.0

    total = (
        weights.family * family_loss
        + weights.phase * phase_loss
        + weights.boundary * boundary_loss
        + weights.quality * quality_loss
        + weights.expert_quality * expert_quality_loss
        + weights.tracking * tracking_loss
    )
    values = {
        "family": float(family_loss.detach().cpu()),
        "phase": float(phase_loss.detach().cpu()),
        "boundary": float(boundary_loss.detach().cpu()),
        "quality": float(quality_loss.detach().cpu()),
        "expert_quality": float(expert_quality_loss.detach().cpu()),
        "tracking": float(tracking_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, values


def positive_weight_from_binary(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask & (values >= 0)]
    if selected.numel() == 0:
        return values.new_tensor(1.0)
    positives = (selected > 0.5).sum().float()
    negatives = (selected <= 0.5).sum().float()
    return torch.clamp(negatives / torch.clamp(positives, min=1.0), min=1.0, max=10.0)
