"""Inference-time target-side boundaries for the coupled CNNs."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

BoundaryKind = Literal["self_generated", "null", "learned"]


class SelfGeneratedTargetBoundary(nn.Module):
    """Lift a soft class hypothesis into the reverse lattice state space."""

    def __init__(
        self,
        *,
        num_classes: int,
        channels: int = 64,
        prototype_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or channels <= 0 or prototype_dim <= 0:
            raise ValueError("num_classes, channels, and prototype_dim must be positive")
        self.num_classes = int(num_classes)
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, prototype_dim) * 0.02
        )
        self.projector = nn.Linear(prototype_dim, channels)

    def forward(self, logits: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"expected logits [B,{self.num_classes}], got {tuple(logits.shape)}"
            )
        probabilities = logits.softmax(dim=-1)
        hypothesis = probabilities @ self.class_prototypes
        channels = self.projector(hypothesis)
        boundary = channels[:, :, None, None].expand(
            -1, -1, reference.shape[-2], reference.shape[-1]
        )
        return boundary, probabilities


class NullTargetBoundary(nn.Module):
    """Zero-information target boundary used only as a topology ablation."""

    def forward(self, logits: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        return torch.zeros_like(reference), logits.softmax(dim=-1)


class LearnedTargetBoundary(nn.Module):
    """One sample-independent learned target-side reservoir."""

    def __init__(self, *, channels: int = 64, spatial_size: int = 8) -> None:
        super().__init__()
        if channels <= 0 or spatial_size <= 0:
            raise ValueError("channels and spatial_size must be positive")
        self.state = nn.Parameter(torch.zeros(1, channels, spatial_size, spatial_size))

    def forward(self, logits: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        if self.state.shape[1:] != reference.shape[1:]:
            raise ValueError(
                "learned boundary shape does not match the lattice; "
                f"got {tuple(self.state.shape[1:])} and {tuple(reference.shape[1:])}"
            )
        return self.state.expand(reference.shape[0], -1, -1, -1), logits.softmax(dim=-1)


def build_boundary(
    kind: BoundaryKind,
    *,
    num_classes: int,
    channels: int,
    prototype_dim: int,
    spatial_size: int,
) -> nn.Module:
    if kind == "self_generated":
        return SelfGeneratedTargetBoundary(
            num_classes=num_classes,
            channels=channels,
            prototype_dim=prototype_dim,
        )
    if kind == "null":
        return NullTargetBoundary()
    if kind == "learned":
        return LearnedTargetBoundary(channels=channels, spatial_size=spatial_size)
    raise ValueError(f"unknown boundary kind {kind!r}")


__all__ = [
    "BoundaryKind",
    "LearnedTargetBoundary",
    "NullTargetBoundary",
    "SelfGeneratedTargetBoundary",
    "build_boundary",
]
