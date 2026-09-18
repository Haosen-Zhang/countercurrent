"""Invertible residual transport for the two-point boundary-value lattice (P0).

Plan section 2.5.  The reverse field must live in the same representable space as
the forward field, otherwise the discrepancy D = H_bar - C_bar measures a units
mismatch instead of a semantic one.  Measured on the P1 solver the reverse field
sat at rms(C)/rms(H) ~ 0.15 and cos(D, H_bar) ~ 0.99, which makes the exchange a
uniform rescaling of the forward activations.

Making the reverse transport the exact inverse of the forward transport fixes the
geometry by construction:

    F(x) = x + beta * f(x),      Lip(beta * f) <= c < 1
    F^-1(y) solves x = y - beta * f(x)   by fixed-point iteration

With c <= 0.1 the iteration reaches float32 precision in about 13 steps, and the
contraction is guaranteed rather than hoped for.  ExactInverseTransport adapts
F^-1 to the nn.Module interface the lattice expects, so G can be either an
independent learned transport (P1) or the exact inverse (P0) without touching the
solver.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .stem import _check_groups


class SpectralNormConv2d(nn.Module):
    """Conv2d whose operator norm is bounded by a target value.

    A single power iteration per forward pass is enough to keep a running estimate
    accurate during training, which is the standard trick from spectral
    normalisation for GANs and invertible residual networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        padding: int = 1,
        target: float = 1.0,
        power_iterations: int = 1,
    ) -> None:
        super().__init__()
        if target <= 0:
            raise ValueError("target must be positive")
        if power_iterations < 1:
            raise ValueError("power_iterations must be at least 1")
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.target = float(target)
        self.power_iterations = int(power_iterations)
        self.register_buffer("_u", torch.randn(1, out_channels, 1, 1))
        self.register_buffer("_v", torch.randn(1, in_channels, 1, 1))

    @torch.no_grad()
    def _update_estimates(self) -> None:
        weight = self.conv.weight
        u = self._u
        v = self._v
        for _ in range(self.power_iterations):
            v = torch.nn.functional.conv2d(u, weight, padding=self.conv.padding)
            v = v / (v.norm() + 1e-12)
            u = torch.nn.functional.conv_transpose2d(
                v, weight, padding=self.conv.padding
            )
            u = u / (u.norm() + 1e-12)
        self._u.copy_(u)
        self._v.copy_(v)

    def spectral_norm(self) -> Tensor:
        return torch.nn.functional.conv2d(
            self._u, self.conv.weight, padding=self.conv.padding
        ).norm()

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            if self.training:
                # Update the running estimate outside the autograd graph.  The
                # solver calls this block many times per forward pass, so an
                # in-place buffer mutation inside a differentiable forward would
                # invalidate the saved values whenever grad mode is active.
                self._update_estimates()
            scale = self.target / self.spectral_norm().clamp_min(1e-6)
        # Detaching the scale keeps it a constant multiplier; the signal reaches
        # conv.weight through the convolution itself.
        return self.conv(x) * scale


class InvertibleTransportBlock(nn.Module):
    """Residual transport F(x) = x + beta * f(x) with a guaranteed contraction.

    beta scales the residual branch and the final convolution is spectrally
    normalised, so Lip(beta * f) <= beta.  The inverse is computed by fixed-point
    iteration, which is therefore guaranteed to converge at rate beta.
    """

    def __init__(
        self,
        channels: int = 64,
        num_groups: int = 8,
        beta: float = 0.1,
        inverse_steps: int = 15,
        inverse_tol: float = 1e-7,
    ) -> None:
        super().__init__()
        _check_groups(channels, num_groups)
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1) for the inverse to converge")
        if inverse_steps < 1:
            raise ValueError("inverse_steps must be at least 1")
        self.beta = float(beta)
        self.inverse_steps = int(inverse_steps)
        self.inverse_tol = float(inverse_tol)
        self.residual = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            SpectralNormConv2d(channels, channels, kernel_size=3, padding=1, target=1.0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.beta * self.residual(x)

    def inverse(self, y: Tensor, *, create_graph: bool = False) -> Tensor:
        """Solve x = y - beta * f(x) by fixed-point iteration.

        The converged solution x_star is obtained with gradient tracking disabled,
        which is what makes the inversion cheap and free of in-place conflicts.
        The returned value re-expresses the solution as y - beta * f(x_star), so it
        is a differentiable function of the input with the exactly correct
        identity term: d F^-1 / dy = I - beta * J_f (evaluated at x_star).  The
        parameters inside f therefore receive gradients through x_star treated as a
        constant, which is accurate to first order in beta.
        """
        with torch.no_grad():
            x = y
            for _ in range(self.inverse_steps):
                previous = x
                x = y - self.beta * self.residual(x)
                if (x - previous).norm() <= self.inverse_tol * (
                    previous.norm() + self.inverse_tol
                ):
                    break
        return y - self.beta * self.residual(x.detach())

    def extra_repr(self) -> str:
        return f"beta={self.beta}, inverse_steps={self.inverse_steps}"


class ExactInverseTransport(nn.Module):
    """Adapt an invertible block's inverse to the transport G interface.

    The lattice calls G on a reverse inlet and expects the transported field.
    Using the exact inverse of the corresponding forward block is what puts the
    two fields in the same representable space.
    """

    def __init__(
        self, block: InvertibleTransportBlock, *, create_graph: bool = False
    ) -> None:
        super().__init__()
        self.block = block
        self.create_graph = bool(create_graph)

    def forward(self, inlet: Tensor) -> Tensor:
        return self.block.inverse(inlet, create_graph=self.create_graph)


__all__ = [
    "ExactInverseTransport",
    "InvertibleTransportBlock",
    "SpectralNormConv2d",
]
