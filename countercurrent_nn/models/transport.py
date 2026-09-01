"""Residual transport operator for one lattice position."""

from __future__ import annotations

from torch import Tensor, nn

from .stem import _check_groups


class TransportBlock(nn.Module):
    """Pre-activation residual block reused across outer inference iterations."""

    def __init__(
        self,
        channels: int = 64,
        num_groups: int = 8,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        _check_groups(channels, num_groups)
        if residual_scale <= 0:
            raise ValueError(f"residual_scale must be positive, got {residual_scale}")

        self.residual_scale = float(residual_scale)
        self.residual = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.residual_scale * self.residual(x)

    def extra_repr(self) -> str:
        return f"residual_scale={self.residual_scale}"
