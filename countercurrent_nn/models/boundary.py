"""Inference-time target-side boundaries for the coupled CNNs."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from .stem import _check_groups

BoundaryKind = Literal[
    "self_generated",
    "input_conditional",
    "null",
    "learned",
]


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


class InputConditionalTargetBoundary(nn.Module):
    """Target boundary conditioned on both the hypothesis and the evidence.

    The plain self-generated boundary is a spatial constant built only from the
    class hypothesis, so it cannot describe *this* input.  Measured on the V4
    model it moved the logits by 0.19% relative and was indistinguishable from a
    uniform hypothesis.  This boundary adds an instance-conditioned term read off
    the forward terminal state:

        C_L = B0(z_hyp) + alpha * Proj(phi(sg[H_L]))

    ``sg`` is a stop-gradient: it keeps the information path hypothesis -> evidence
    while removing the degenerate solution in which the network collapses H_L onto
    C_L to silence the exchange flux.  ``alpha`` starts at zero so training begins
    from the previous behaviour and learns how much instance content to inject.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        channels: int = 64,
        prototype_dim: int = 64,
        num_groups: int = 8,
        initial_alpha: float = 0.1,
        target_rms: float | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or channels <= 0 or prototype_dim <= 0:
            raise ValueError("num_classes, channels, and prototype_dim must be positive")
        _check_groups(channels, num_groups)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, prototype_dim) * 0.02
        )
        self.projector = nn.Linear(prototype_dim, channels)
        self.instance_scale = nn.Parameter(torch.tensor(float(initial_alpha)))
        # The prototype/projector output is naturally tiny (rms ~0.07), while the
        # forward terminal state sits around rms 0.6.  A transport block is
        # x + beta * f(x) with beta = 0.1, so it is close to the identity in
        # magnitude and cannot make up a 7x gap: without this gain the reverse
        # field stays an order of magnitude smaller than H and the discrepancy
        # reduces to a rescaling of the forward activations.
        self.gain = nn.Parameter(torch.ones(()))
        self.instance_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        if target_rms is not None:
            self._calibrate_gain(float(target_rms))

    @property
    def alpha(self) -> Tensor:
        return self.instance_scale

    @torch.no_grad()
    def _calibrate_gain(self, target_rms: float) -> None:
        """Set the output gain so the boundary starts at ``target_rms``.

        The scale is measured on the module as built, so the calibration holds
        whatever the prototype or projector initialisation happens to be.  The
        gain stays a learnable parameter and is free to move during training.
        """
        if target_rms <= 0:
            raise ValueError("target_rms must be positive")
        logits = torch.zeros(1, self.num_classes)
        reference = torch.zeros(1, self.channels, 8, 8)
        raw = self._raw_boundary(logits, reference)
        current = raw.square().mean().sqrt()
        if float(current) > 0:
            self.gain.fill_(target_rms / float(current))

    def _raw_boundary(self, logits: Tensor, reference: Tensor) -> Tensor:
        """Boundary before the output gain; shared by forward and calibration. """
        probabilities = logits.softmax(dim=-1)
        hypothesis = probabilities @ self.class_prototypes
        channels = self.projector(hypothesis)
        boundary = channels[:, :, None, None].expand(
            -1, -1, reference.shape[-2], reference.shape[-1]
        )
        instance = self.instance_encoder(reference.detach())
        return boundary + self.instance_scale * instance

    def forward(self, logits: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"expected logits [B,{self.num_classes}], got {tuple(logits.shape)}"
            )
        if reference.ndim != 4 or reference.shape[1] != self.channels:
            raise ValueError(
                f"expected reference [B,{self.channels},H,W], got {tuple(reference.shape)}"
            )
        boundary = self._raw_boundary(logits, reference) * self.gain
        return boundary, logits.softmax(dim=-1)


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
    target_rms: float | None = None,
) -> nn.Module:
    if kind == "self_generated":
        return SelfGeneratedTargetBoundary(
            num_classes=num_classes,
            channels=channels,
            prototype_dim=prototype_dim,
        )
    if kind == "input_conditional":
        return InputConditionalTargetBoundary(
            num_classes=num_classes,
            channels=channels,
            prototype_dim=prototype_dim,
            target_rms=target_rms,
        )
    if kind == "null":
        return NullTargetBoundary()
    if kind == "learned":
        return LearnedTargetBoundary(channels=channels, spatial_size=spatial_size)
    raise ValueError(f"unknown boundary kind {kind!r}")


__all__ = [
    "BoundaryKind",
    "InputConditionalTargetBoundary",
    "LearnedTargetBoundary",
    "NullTargetBoundary",
    "SelfGeneratedTargetBoundary",
    "build_boundary",
]
