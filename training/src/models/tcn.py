"""Compact causal dilated temporal convolutional network."""

from __future__ import annotations

import torch
from torch import nn

from .heads import MultiTaskHeads


class CausalConv1d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.conv(values)
        if self.padding:
            output = output[..., :-self.padding]
        return output


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.first = CausalConv1d(channels, channels, kernel_size, dilation)
        self.second = CausalConv1d(channels, channels, kernel_size, dilation)
        # Normalize only across channels at each timestep. GroupNorm over a
        # [channels, time] tensor would leak future frames into a causal model.
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        output = self.first(values)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.second(output)
        output = self.norm(output.transpose(1, 2)).transpose(1, 2)
        output = self.activation(output)
        output = self.dropout(output)
        return output + residual


class CausalTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int = 96,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        pooling: str = "last_valid",
        family_classes: int = 6,
        phase_classes: int = 5,
        quality_outputs: int = 4,
        expert_quality_classes: int = 0,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(channels, kernel_size, dilation, dropout)
                for dilation in dilations
            ]
        )
        self.heads = MultiTaskHeads(
            channels,
            family_classes,
            phase_classes,
            quality_outputs,
            expert_quality_classes,
            pooling=pooling,
        )

    @property
    def receptive_field(self) -> int:
        """Number of input frames that can influence one output frame."""

        kernel_size = self.blocks[0].first.conv.kernel_size[0] if len(self.blocks) else 1
        dilations = [block.first.conv.dilation[0] for block in self.blocks]
        return 1 + 2 * (kernel_size - 1) * sum(dilations)

    def forward(self, values: torch.Tensor, time_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if values.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(values.shape)}")
        if values.shape[-1] != self.input_projection.in_channels:
            raise ValueError(
                f"Expected {self.input_projection.in_channels} input features, got {values.shape[-1]}"
            )
        output = values.transpose(1, 2)
        output = self.input_projection(output)
        output = self.blocks(output)
        return self.heads(output.transpose(1, 2), time_mask=time_mask)
