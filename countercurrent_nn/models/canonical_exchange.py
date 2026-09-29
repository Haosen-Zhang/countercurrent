"""Orthogonally aligned paired exchange for V5-C counterflow cells."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import orthogonal

from .exchange import ChannelwiseConductance


class OrthogonalChannelTransform(nn.Module):
    """Apply an exactly orthogonal channel transform at every spatial position."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        linear = nn.Linear(channels, channels, bias=False)
        with torch.no_grad():
            nn.init.eye_(linear.weight)
        self.linear = orthogonal(linear)
        self.channels = int(channels)

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    def _check(self, state: Tensor) -> None:
        if state.ndim != 4 or state.shape[1] != self.channels:
            raise ValueError(
                f"expected state [B,{self.channels},H,W], got {tuple(state.shape)}"
            )

    def encode(self, state: Tensor) -> Tensor:
        self._check(state)
        channels_last = state.permute(0, 2, 3, 1)
        return self.linear(channels_last).permute(0, 3, 1, 2)

    def decode(self, state: Tensor) -> Tensor:
        self._check(state)
        channels_last = state.permute(0, 2, 3, 1)
        decoded = F.linear(channels_last, self.weight.transpose(0, 1))
        return decoded.permute(0, 3, 1, 2)

    def forward(self, state: Tensor) -> Tensor:
        return self.encode(state)


class CanonicalExchange(nn.Module):
    """Align H/C, apply one antisymmetric flux, then decode each stream."""

    def __init__(
        self,
        channels: int = 64,
        *,
        conductance_max: float = 0.49,
        conductance_init_logit: float = -2.2,
        use_canonical: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.use_canonical = bool(use_canonical)
        if self.use_canonical:
            self.encode_h = OrthogonalChannelTransform(channels)
            self.encode_c = OrthogonalChannelTransform(channels)
        else:
            self.encode_h = nn.Identity()
            self.encode_c = nn.Identity()
        self.conductance = ChannelwiseConductance(
            channels,
            conductance_max=conductance_max,
            initial_logit=conductance_init_logit,
        )

    @property
    def gamma(self) -> Tensor:
        return self.conductance.gamma

    def _decode_h(self, state: Tensor) -> Tensor:
        if isinstance(self.encode_h, OrthogonalChannelTransform):
            return self.encode_h.decode(state)
        return state

    def _decode_c(self, state: Tensor) -> Tensor:
        if isinstance(self.encode_c, OrthogonalChannelTransform):
            return self.encode_c.decode(state)
        return state

    def forward(
        self,
        h_bar: Tensor,
        c_bar: Tensor,
        *,
        reverse_off: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if h_bar.shape != c_bar.shape:
            raise ValueError("canonical exchange inputs must have identical shape")
        u_h = self.encode_h(h_bar)
        u_c = self.encode_c(c_bar)
        discrepancy = u_h - u_c
        flux = (
            torch.zeros_like(discrepancy)
            if reverse_off
            else self.conductance(discrepancy)
        )
        u_h_new = u_h - flux
        u_c_new = u_c + flux
        h_new = self._decode_h(u_h_new)
        c_new = self._decode_c(u_c_new)
        return h_new, c_new, {
            "u_h": u_h,
            "u_c": u_c,
            "D": discrepancy,
            "Q": flux,
            "u_h_new": u_h_new,
            "u_c_new": u_c_new,
        }


__all__ = ["CanonicalExchange", "OrthogonalChannelTransform"]
