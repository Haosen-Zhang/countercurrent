"""Two-sweep counter/co-current inference with endogenous target boundaries."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from .boundary import BoundaryKind, build_boundary
from .exchange import ChannelwiseConductance
from .stem import Stem
from .transport import TransportBlock

Topology = Literal["countercurrent", "cocurrent"]


def _normalized_l2(x: Tensor) -> Tensor:
    return torch.linalg.vector_norm(x.float()) / max(x.numel(), 1)


def _relative_state_change(old: list[Tensor], new: list[Tensor]) -> Tensor:
    numerator = sum(
        (new_state.float() - old_state.float()).square().sum()
        for old_state, new_state in zip(old, new, strict=True)
    ).sqrt()
    denominator = sum(state.float().square().sum() for state in old).sqrt()
    return numerator / (denominator + 1e-8)


class CoupledLatticeBase(nn.Module):
    """Closed-loop two-boundary inference using the first-stage two-sweep update."""

    def __init__(
        self,
        *,
        topology: Topology,
        in_channels: int = 3,
        channels: int = 64,
        depth: int = 4,
        refine_steps: int = 3,
        num_classes: int = 10,
        num_groups: int = 8,
        residual_scale: float = 0.1,
        boundary_type: BoundaryKind = "self_generated",
        prototype_dim: int = 64,
        lattice_size: int = 8,
        conductance_max: float = 0.49,
        conductance_init_logit: float = -2.2,
        exchange_enabled: bool = True,
    ) -> None:
        super().__init__()
        if topology not in ("countercurrent", "cocurrent"):
            raise ValueError(f"unsupported topology: {topology!r}")
        if depth <= 0 or refine_steps <= 0:
            raise ValueError("depth and refine_steps must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.topology: Topology = topology
        self.channels = int(channels)
        self.depth = int(depth)
        self.refine_steps = int(refine_steps)
        self.boundary_type: BoundaryKind = boundary_type
        self.exchange_enabled = bool(exchange_enabled)

        self.stem = Stem(in_channels, channels, num_groups)
        self.F = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale) for _ in range(depth)
        )
        self.G = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale) for _ in range(depth)
        )
        self.exchange = nn.ModuleList(
            ChannelwiseConductance(
                channels,
                conductance_max=conductance_max,
                initial_logit=conductance_init_logit,
            )
            for _ in range(depth)
        )
        self.head = nn.Linear(channels, num_classes)
        self.target_boundary = build_boundary(
            boundary_type,
            num_classes=num_classes,
            channels=channels,
            prototype_dim=prototype_dim,
            spatial_size=lattice_size,
        )

    @property
    def steps(self) -> int:
        """Readability alias: these are outer hypothesis-refinement steps."""
        return self.refine_steps

    def _classify(self, state: Tensor) -> Tensor:
        return self.head(state.mean(dim=(2, 3)))

    def _pure_forward(self, source: Tensor) -> list[Tensor]:
        states = [source]
        for block in self.F:
            states.append(block(states[-1]))
        return states

    def _reverse_proposal(self, boundary: Tensor) -> list[Tensor]:
        states: list[Tensor | None] = [None] * (self.depth + 1)
        if self.topology == "countercurrent":
            states[self.depth] = boundary
            for position in reversed(range(self.depth)):
                inlet = states[position + 1]
                if inlet is None:
                    raise RuntimeError("countercurrent reverse inlet is undefined")
                states[position] = self.G[position](inlet)
        else:
            states[0] = boundary
            for position in range(self.depth):
                inlet = states[position]
                if inlet is None:
                    raise RuntimeError("co-current reverse inlet is undefined")
                states[position + 1] = self.G[position](inlet)
        result = [state for state in states if state is not None]
        if len(result) != self.depth + 1:
            raise RuntimeError("reverse sweep left an undefined state")
        return result

    def _reverse_reference_index(self, position: int) -> int:
        return position if self.topology == "countercurrent" else position + 1

    def _run(
        self,
        x: Tensor,
        *,
        capture_diagnostics: bool,
        reverse_off: bool,
    ) -> tuple[Tensor, list[Tensor], dict[str, Tensor]]:
        # Iteration 0 is a pure feed-forward pass with no reverse exchange.
        h_states = self._pure_forward(self.stem(x))
        logits = self._classify(h_states[-1])
        iteration_logits = [logits]

        h_history: list[Tensor] = []
        c_history: list[Tensor] = []
        boundary_history: list[Tensor] = []
        probability_history: list[Tensor] = []
        h_norm_history: list[Tensor] = []
        c_norm_history: list[Tensor] = []
        q_norm_history: list[Tensor] = []
        discrepancy_history: list[Tensor] = []
        energy_history: list[Tensor] = []
        residual_history: list[Tensor] = []

        if capture_diagnostics:
            h_history.append(torch.stack([state.detach().cpu() for state in h_states]))
            h_norm_history.append(
                torch.stack([_normalized_l2(state) for state in h_states]).detach().cpu()
            )

        for _ in range(self.refine_steps):
            # Recompute and hard-clamp source evidence at every outer iteration.
            source = self.stem(x)
            boundary, probabilities = self.target_boundary(logits, source)
            c_states = self._reverse_proposal(boundary)
            corrected_c = list(c_states)
            new_h = [source]
            q_this_step: list[Tensor] = []
            d_this_step: list[Tensor] = []
            energy_this_step: list[Tensor] = []

            # Corrected forward resweep: reverse states alter every intermediate H.
            for position in range(self.depth):
                h_bar = self.F[position](new_h[position])
                c_index = self._reverse_reference_index(position)
                c_ref = c_states[c_index]
                discrepancy = h_bar - c_ref
                q = (
                    self.exchange[position](discrepancy)
                    if self.exchange_enabled and not reverse_off
                    else torch.zeros_like(discrepancy)
                )
                new_h.append(h_bar - q)
                corrected_c[c_index] = c_ref + q
                if capture_diagnostics:
                    q_this_step.append(_normalized_l2(q))
                    d_this_step.append(_normalized_l2(discrepancy))
                    energy_this_step.append(
                        0.5
                        * (
                            self.exchange[position].gamma
                            * discrepancy.float().square()
                        ).mean()
                    )

            if capture_diagnostics:
                residual_history.append(
                    _relative_state_change(h_states, new_h).detach().cpu()
                )
            h_states = new_h
            logits = self._classify(h_states[-1])
            iteration_logits.append(logits)

            if capture_diagnostics:
                h_history.append(
                    torch.stack([state.detach().cpu() for state in h_states])
                )
                c_history.append(
                    torch.stack([state.detach().cpu() for state in corrected_c])
                )
                boundary_history.append(boundary.detach().cpu())
                probability_history.append(probabilities.detach().cpu())
                h_norm_history.append(
                    torch.stack([_normalized_l2(state) for state in h_states])
                    .detach()
                    .cpu()
                )
                c_norm_history.append(
                    torch.stack([_normalized_l2(state) for state in corrected_c])
                    .detach()
                    .cpu()
                )
                q_norm_history.append(torch.stack(q_this_step).detach().cpu())
                discrepancy_history.append(torch.stack(d_this_step).detach().cpu())
                energy_history.append(torch.stack(energy_this_step).detach().cpu())

        diagnostics: dict[str, Tensor] = {}
        if capture_diagnostics:
            diagnostics = {
                "H_history": torch.stack(h_history),
                "C_history": torch.stack(c_history),
                "boundary_history": torch.stack(boundary_history),
                "boundary_probabilities": torch.stack(probability_history),
                "H_norm": torch.stack(h_norm_history),
                "C_norm": torch.stack(c_norm_history),
                "q_norm": torch.stack(q_norm_history),
                "discrepancy": torch.stack(discrepancy_history),
                "exchange_energy": torch.stack(energy_history),
                "residual": torch.stack(residual_history),
                "iter_logits": torch.stack(
                    [value.detach().cpu() for value in iteration_logits]
                ),
                "initial_logits": iteration_logits[0].detach().cpu(),
                "gamma": torch.stack(
                    [block.gamma.detach().cpu().flatten() for block in self.exchange]
                ),
            }
        return logits, iteration_logits, diagnostics

    def predict_iterations(self, x: Tensor, *, reverse_off: bool = False) -> Tensor:
        """Return logits for pure-forward t=0 and every refinement iteration."""
        _, logits, _ = self._run(
            x, capture_diagnostics=False, reverse_off=reverse_off
        )
        return torch.stack(logits)

    def forward(
        self,
        x: Tensor,
        return_diagnostics: bool = False,
        reverse_off: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        logits, _, diagnostics = self._run(
            x,
            capture_diagnostics=return_diagnostics,
            reverse_off=reverse_off,
        )
        return (logits, diagnostics) if return_diagnostics else logits

    def extra_repr(self) -> str:
        return (
            f"topology={self.topology}, boundary={self.boundary_type}, "
            f"channels={self.channels}, depth={self.depth}, "
            f"refine_steps={self.refine_steps}, exchange_enabled={self.exchange_enabled}"
        )
