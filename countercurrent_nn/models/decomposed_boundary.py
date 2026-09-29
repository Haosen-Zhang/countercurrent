"""Decomposed fixed target boundary for V5-C."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .stem import _check_groups


class DecomposedTargetBoundary(nn.Module):
    """Combine base, centered spatial class, and stop-gradient instance terms."""

    def __init__(
        self,
        *,
        num_classes: int,
        channels: int = 64,
        spatial_size: int = 8,
        class_rank: int = 16,
        num_groups: int = 8,
        base_scale: float = 0.15,
        class_scale: float = 1.0,
        instance_scale: float = 0.1,
        target_rms: float | None = 0.55,
    ) -> None:
        super().__init__()
        if min(num_classes, channels, spatial_size, class_rank) <= 0:
            raise ValueError("boundary dimensions must be positive")
        _check_groups(channels, num_groups)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.spatial_size = int(spatial_size)
        self.class_rank = int(class_rank)

        # Give all three raw components comparable natural scale so the explicit
        # lambda values, rather than accidental initializer magnitudes, control
        # their initial importance.
        self.base_state = nn.Parameter(
            torch.randn(1, channels, spatial_size, spatial_size)
        )
        self.class_embedding = nn.Parameter(
            torch.randn(num_classes, class_rank) / math.sqrt(class_rank)
        )
        # No bias: a uniform hypothesis must produce exactly zero class forcing.
        self.class_spatial_projector = nn.Linear(
            class_rank, channels * spatial_size * spatial_size, bias=False
        )
        self.instance_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )
        self.base_scale = nn.Parameter(torch.tensor(float(base_scale)))
        self.class_scale = nn.Parameter(torch.tensor(float(class_scale)))
        self.instance_scale = nn.Parameter(torch.tensor(float(instance_scale)))
        self.log_global_gain = nn.Parameter(torch.zeros(()))
        if target_rms is not None:
            self._calibrate_gain(float(target_rms))

    @property
    def global_gain(self) -> Tensor:
        return self.log_global_gain.exp()

    def _validate(self, probabilities: Tensor, reference: Tensor) -> None:
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError(
                f"expected probabilities [B,{self.num_classes}], "
                f"got {tuple(probabilities.shape)}"
            )
        expected = (self.channels, self.spatial_size, self.spatial_size)
        if reference.ndim != 4 or tuple(reference.shape[1:]) != expected:
            raise ValueError(
                f"expected reference [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(reference.shape)}"
            )

    def components_from_probabilities(
        self, probabilities: Tensor, reference: Tensor
    ) -> dict[str, Tensor]:
        self._validate(probabilities, reference)
        batch = probabilities.shape[0]
        base = self.base_state.expand(batch, -1, -1, -1)
        centered = probabilities - 1.0 / self.num_classes
        class_latent = centered @ self.class_embedding
        class_component = self.class_spatial_projector(class_latent).reshape(
            batch, self.channels, self.spatial_size, self.spatial_size
        )
        instance = self.instance_encoder(reference.detach())
        weighted_base = self.base_scale * base
        weighted_class = self.class_scale * class_component
        weighted_instance = self.instance_scale * instance
        combined = weighted_base + weighted_class + weighted_instance
        return {
            "base": base,
            "class": class_component,
            "instance": instance,
            "weighted_base": weighted_base,
            "weighted_class": weighted_class,
            "weighted_instance": weighted_instance,
            "combined": combined,
            "centered_probabilities": centered,
        }

    @torch.no_grad()
    def _calibrate_gain(self, target_rms: float) -> None:
        if not math.isfinite(target_rms) or target_rms <= 0:
            raise ValueError("target_rms must be finite and positive")
        logits = torch.zeros(1, self.num_classes)
        logits[0, 0] = 4.0
        probabilities = logits.softmax(dim=-1)
        generator = torch.Generator().manual_seed(1729)
        reference = 0.6 * torch.randn(
            1,
            self.channels,
            self.spatial_size,
            self.spatial_size,
            generator=generator,
        )
        combined = self.components_from_probabilities(probabilities, reference)[
            "combined"
        ]
        current = combined.square().mean().sqrt().clamp_min(1e-12)
        self.log_global_gain.fill_(math.log(target_rms / float(current)))

    def forward(
        self,
        logits: Tensor,
        reference: Tensor,
        *,
        return_components: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"expected logits [B,{self.num_classes}], got {tuple(logits.shape)}"
            )
        probabilities = logits.softmax(dim=-1)
        components = self.components_from_probabilities(probabilities, reference)
        boundary = self.global_gain * components["combined"]
        if not return_components:
            return boundary, probabilities
        details = dict(components)
        details["boundary"] = boundary
        details["global_gain"] = self.global_gain
        return boundary, probabilities, details


__all__ = ["DecomposedTargetBoundary"]
