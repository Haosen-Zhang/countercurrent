"""Local discrepancy-driven, paired state exchange."""

from __future__ import annotations

from torch import Tensor, nn


class ChannelwiseConductance(nn.Module):
    """Compute Q = gamma * D with a positive conductance below 0.5."""

    def __init__(
        self,
        channels: int = 64,
        conductance_max: float = 0.49,
        initial_logit: float = -2.2,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if not 0.0 < conductance_max < 0.5:
            raise ValueError("conductance_max must satisfy 0 < max < 0.5")
        self.channels = int(channels)
        self.conductance_max = float(conductance_max)
        self.logit_gamma = nn.Parameter(
            Tensor(1, channels, 1, 1).fill_(float(initial_logit))
        )

    @property
    def gamma(self) -> Tensor:
        return self.conductance_max * self.logit_gamma.sigmoid()

    def forward(self, discrepancy: Tensor) -> Tensor:
        if discrepancy.ndim != 4 or discrepancy.shape[1] != self.channels:
            raise ValueError(
                f"expected discrepancy [B,{self.channels},H,W], "
                f"got {tuple(discrepancy.shape)}"
            )
        return self.gamma * discrepancy

    def paired_update(self, h_bar: Tensor, c_bar: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the conservative exchange and return (h_out, c_out, q)."""
        if h_bar.shape != c_bar.shape:
            raise ValueError("paired exchange inputs must have identical shape")
        q = self(h_bar - c_bar)
        return h_bar - q, c_bar + q, q

    def extra_repr(self) -> str:
        return f"conductance_max={self.conductance_max}"


ExchangeBlock = ChannelwiseConductance

__all__ = ["ChannelwiseConductance", "ExchangeBlock"]
