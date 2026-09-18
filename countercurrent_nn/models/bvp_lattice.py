"""Two-point boundary-value counterflow lattice (plan section 2, P1).

The V2-V4 schedule rebuilt the entire state from `stem(x)» and the previous
logits on every round, so the cycle was a fixed point of the schedule after one
round and the refinement had nothing left to do.  Measured on V4 the relative
state change was 1.7e-5 at t=2 and exactly zero from t=3 onward.

This module instead solves the two-point boundary-value problem directly:

    H_bar_0 = E(x)                    source inlet, re-clamped every sweep
    C_bar_L = B(p, sg[H_L])           target inlet, endogenous

    H_bar[l+1] = F[l](H[l])           forward transport
    C_bar[l]   = G[l](C[l+1])         reverse transport (L -> 0)

    D[l] = H_bar[l+1] - C_bar[l]
    Q[l] = gamma[l] * D[l]
    H[l+1] = H_bar[l+1] - Q[l]        paired, zero-sum exchange
    C[l]   = C_bar[l]   + Q[l]

Only the source boundary is clamped; the interior persists across sweeps, so the
iteration is a genuine solver and stops on a residual criterion.  `cocurrent»
uses the same solver, boundary and exchange law with the opposite transport
direction for C, so the two topologies differ only in flow geometry.

Gradients are computed by unrolling the recorded solver trajectory (the last
`truncate_steps» sweeps plus the initial forward pass).  The adjoint-based
counterflow backward pass from plan section 3 is a later step.

The two numerical conditions that make the exchange meaningful --- C living in
the same space as H, and D not being a rescaling of H --- are NOT addressed here;
that is what the invertible transport of section 2.5 (P0) is for.  This module
deliberately keeps F and G independent so the solver and the boundary changes can
be attributed on their own.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn

from .boundary import BoundaryKind, build_boundary
from .exchange import ChannelwiseConductance
from .invertible_transport import ExactInverseTransport, InvertibleTransportBlock
from .stem import Stem
from .transport import TransportBlock

Topology = Literal["countercurrent", "cocurrent"]


def _normalized_l2(x: Tensor) -> Tensor:
    return torch.linalg.vector_norm(x.float()) / max(x.numel(), 1)


class BVPCounterflow(nn.Module):
    """Solve the two-boundary counterflow fixed point with a residual criterion."""

    def __init__(
        self,
        *,
        topology: Topology = "countercurrent",
        in_channels: int = 3,
        channels: int = 64,
        depth: int = 4,
        solve_steps: int = 8,
        num_classes: int = 10,
        num_groups: int = 8,
        residual_scale: float = 0.1,
        boundary_type: BoundaryKind = "input_conditional",
        prototype_dim: int = 64,
        lattice_size: int = 8,
        conductance_max: float = 0.49,
        conductance_init_logit: float = -2.2,
        exchange_enabled: bool = True,
        damping: float = 1.0,
        tol: float = 1e-3,
        truncate_steps: int = 4,
        diagnostic_stride: int = 1,
        boundary_initial_alpha: float = 0.1,
        flow: Literal["independent", "invertible"] = "independent",
        reverse_beta: float = 0.1,
        reverse_init_from_forward: bool = False,
        inverse_steps: int = 15,
        boundary_target_rms: float | None = None,
    ) -> None:
        super().__init__()
        if topology not in ("countercurrent", "cocurrent"):
            raise ValueError(f"unsupported topology: {topology!r}")
        if depth <= 0 or solve_steps <= 0:
            raise ValueError("depth and solve_steps must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must lie in (0, 1]")
        if tol <= 0.0:
            raise ValueError("tol must be positive")
        if truncate_steps < 1:
            raise ValueError("truncate_steps must be at least 1")
        if diagnostic_stride < 1:
            raise ValueError("diagnostic_stride must be at least 1")
        if flow not in ("independent", "invertible"):
            raise ValueError(f"unsupported flow {flow!r}")

        self.topology: Topology = topology
        self.channels = int(channels)
        self.depth = int(depth)
        self.solve_steps = int(solve_steps)
        self.exchange_enabled = bool(exchange_enabled)
        self.damping = float(damping)
        self.tol = float(tol)
        self.truncate_steps = int(truncate_steps)
        self.diagnostic_stride = int(diagnostic_stride)
        self.lattice_size = int(lattice_size)
        self.flow = flow

        self.stem = Stem(in_channels, channels, num_groups)
        if flow == "invertible":
            self.F = nn.ModuleList(
                InvertibleTransportBlock(
                    channels,
                    num_groups,
                    beta=reverse_beta,
                    inverse_steps=inverse_steps,
                )
                for _ in range(depth)
            )
            self.G = nn.ModuleList(
                ExactInverseTransport(block) for block in self.F
            )
        else:
            self.F = nn.ModuleList(
                TransportBlock(channels, num_groups, residual_scale)
                for _ in range(depth)
            )
            self.G = nn.ModuleList(
                TransportBlock(channels, num_groups, residual_scale)
                for _ in range(depth)
            )
            if reverse_init_from_forward:
                for forward_block, reverse_block in zip(self.F, self.G, strict=True):
                    reverse_block.load_state_dict(forward_block.state_dict())
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
            target_rms=boundary_target_rms,
        )
        if boundary_type == "input_conditional":
            with torch.no_grad():
                self.target_boundary.instance_scale.fill_(boundary_initial_alpha)

    def _classify(self, state: Tensor) -> Tensor:
        return self.head(state.mean(dim=(2, 3)))

    def _pure_forward(self, source: Tensor) -> list[Tensor]:
        states = [source]
        for block in self.F:
            states.append(block(states[-1]))
        return states

    def _transport_c(self, boundary: Tensor) -> list[Tensor]:
        """Transport the target inlet through every depth position."""
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
            raise RuntimeError("reverse transport left an undefined state")
        return result

    def _exchange_reference_index(self, position: int) -> int:
        """Which C state is paired with H[position + 1] in cell `position»."""
        return position if self.topology == "countercurrent" else position + 1

    def _sweep(
        self,
        source: Tensor,
        h_states: list[Tensor],
        c_states: list[Tensor],
        *,
        reverse_off: bool,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        h_bar_states = [source]
        for position in range(self.depth):
            h_bar_states.append(self.F[position](h_states[position]))

        new_h = [source]
        new_c = list(c_states)
        discrepancies: list[Tensor] = []
        fluxes: list[Tensor] = []
        for position in range(self.depth):
            c_index = self._exchange_reference_index(position)
            discrepancy = h_bar_states[position + 1] - c_states[c_index]
            discrepancies.append(discrepancy)
            if self.exchange_enabled and not reverse_off:
                h_out, c_out, q = self.exchange[position].paired_update(
                    h_bar_states[position + 1], c_states[c_index]
                )
            else:
                q = torch.zeros_like(discrepancy)
                h_out, c_out = h_bar_states[position + 1], c_states[c_index]
            new_h.append(h_out)
            new_c[c_index] = c_out
            fluxes.append(q)
        return new_h, new_c, discrepancies, fluxes

    def _record(
        self,
        h_states: list[Tensor],
        c_states: list[Tensor],
        discrepancies: list[Tensor],
        fluxes: list[Tensor],
    ) -> dict[str, list[Tensor]]:
        return {
            "H": [state.detach().cpu() for state in h_states],
            "C": [state.detach().cpu() for state in c_states],
            "D": [state.detach().cpu() for state in discrepancies],
            "Q": [state.detach().cpu() for state in fluxes],
        }

    def solve(
        self,
        x: Tensor,
        *,
        reverse_off: bool = False,
        capture_diagnostics: bool = False,
    ) -> tuple[Tensor, list[Tensor], dict[str, Any]]:
        """Run the fixed-point solve and return (logits, logits per step, trace)."""
        source = self.stem(x)
        h_states = self._pure_forward(source)
        logits = self._classify(h_states[-1])
        iteration_logits = [logits]
        c_states = self._transport_c(self._boundary_for(logits, h_states[-1]))

        history: list[dict[str, Any]] = []
        recorded: list[dict[str, list[Tensor]]] = []
        if capture_diagnostics:
            recorded.append(self._record(h_states, c_states, [], []))

        for step in range(1, self.solve_steps + 1):
            boundary = self._boundary_for(logits, h_states[-1])
            c_states = self._transport_c(boundary)
            new_h, new_c, discrepancies, fluxes = self._sweep(
                source, h_states, c_states, reverse_off=reverse_off
            )
            if self.damping < 1.0:
                new_h = [
                    old + self.damping * (new - old)
                    for old, new in zip(h_states, new_h, strict=True)
                ]
                new_c = [
                    old + self.damping * (new - old)
                    for old, new in zip(c_states, new_c, strict=True)
                ]

            # The residual measures the persistent state only.  C is
            # reconstructed from the target inlet every sweep, so its step-to-step
            # change never vanishes even at the fixed point and must not gate
            # convergence.  Its change is reported separately for diagnostics.
            numerator = sum(
                (new - old).square().sum()
                for old, new in zip(h_states, new_h, strict=True)
            ).sqrt()
            denominator = sum(state.square().sum() for state in h_states).sqrt()
            relative_change = (numerator / (denominator + 1e-12)).item()
            c_numerator = sum(
                (new - old).square().sum()
                for old, new in zip(c_states, new_c, strict=True)
            ).sqrt()
            c_denominator = sum(state.square().sum() for state in c_states).sqrt()
            c_change = (c_numerator / (c_denominator + 1e-12)).item()
            history.append(
                {
                    "step": step,
                    "relative_state_change": relative_change,
                    "relative_c_change": c_change,
                    "D": [state.detach() for state in discrepancies],
                }
            )

            h_states, c_states = new_h, new_c
            logits = self._classify(h_states[-1])
            iteration_logits.append(logits)

            if capture_diagnostics and step % self.diagnostic_stride == 0:
                recorded.append(self._record(h_states, c_states, discrepancies, fluxes))

            if relative_change < self.tol:
                break

        trace: dict[str, Any] = {"history": history, "steps": len(history)}
        if capture_diagnostics:
            trace["recorded"] = recorded
        return logits, iteration_logits, trace

    def _boundary_for(self, logits: Tensor, h_terminal: Tensor) -> Tensor:
        return self.target_boundary(logits, h_terminal)[0]

    def run_solver(self, x: Tensor) -> tuple[Tensor, list[dict[str, Any]]]:
        """Public hook used by the R1-R4 diagnostic to read the residual history."""
        logits, _, trace = self.solve(x, capture_diagnostics=False)
        return logits, trace["history"]

    def _run(
        self,
        x: Tensor,
        *,
        capture_diagnostics: bool,
        reverse_off: bool,
        oracle_boundary: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], dict[str, Tensor]]:
        if oracle_boundary is not None:
            raise NotImplementedError(
                "the oracle boundary path is analysis-only and absent from the solver"
            )
        logits, iteration_logits, trace = self.solve(
            x, reverse_off=reverse_off, capture_diagnostics=capture_diagnostics
        )
        diagnostics: dict[str, Tensor] = {}
        if capture_diagnostics:
            recorded = trace["recorded"]
            diagnostics = {
                "H_history": torch.stack([torch.stack(entry["H"]) for entry in recorded]),
                "C_history": torch.stack([torch.stack(entry["C"]) for entry in recorded]),
                "residual": torch.tensor(
                    [entry["relative_state_change"] for entry in trace["history"]]
                ),
                "solve_steps": torch.tensor(float(trace["steps"])),
                "iter_logits": torch.stack(
                    [value.detach().cpu() for value in iteration_logits]
                ),
                "initial_logits": iteration_logits[0].detach().cpu(),
                "gamma": torch.stack(
                    [block.gamma.detach().cpu().flatten() for block in self.exchange]
                ),
            }
            exchange_records = recorded[1:]
            diagnostics["D"] = torch.stack(
                [torch.stack(entry["D"]) for entry in exchange_records]
            )
            diagnostics["Q"] = torch.stack(
                [torch.stack(entry["Q"]) for entry in exchange_records]
            )
            diagnostics["proposal_D"] = diagnostics["D"]
            diagnostics["proposal_Q"] = diagnostics["Q"]
            for stage in ("proposal_", ""):
                for field, name in (("D", "discrepancy"), ("Q", "q_norm")):
                    values = diagnostics[stage + field].float().flatten(2)
                    diagnostics[stage + name] = values.norm(dim=-1) / values.shape[-1]
            probabilities = diagnostics["iter_logits"].softmax(dim=-1)
            diagnostics["confidence"] = probabilities.amax(dim=-1)
            diagnostics["predictive_entropy"] = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(-1)
        return logits, iteration_logits, diagnostics

    def predict_iterations(self, x: Tensor, *, reverse_off: bool = False) -> Tensor:
        _, logits, _ = self._run(x, capture_diagnostics=False, reverse_off=reverse_off)
        return torch.stack(logits)

    def forward(
        self,
        x: Tensor,
        *,
        return_diagnostics: bool = False,
        reverse_off: bool = False,
        return_initial_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, dict[str, Tensor]]:
        if return_diagnostics and return_initial_logits:
            raise ValueError("request diagnostics or initial logits, not both")
        logits, iterations, diagnostics = self._run(
            x, capture_diagnostics=return_diagnostics, reverse_off=reverse_off
        )
        if return_initial_logits:
            return logits, iterations[0]
        return (logits, diagnostics) if return_diagnostics else logits

    def extra_repr(self) -> str:
        boundary = type(self.target_boundary).__name__
        return (
            f"topology={self.topology}, boundary={boundary}, channels={self.channels}, "
            f"depth={self.depth}, solve_steps={self.solve_steps}, damping={self.damping}, "
            f"tol={self.tol}, truncate_steps={self.truncate_steps}"
        )


class BVPCountercurrentCNN(BVPCounterflow):
    """Counterflow BVP solver: C is injected at depth L and transported L -> 0."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("boundary_type", "input_conditional")
        kwargs["topology"] = "countercurrent"
        super().__init__(**kwargs)


class BVPCocurrentCNN(BVPCounterflow):
    """Same solver, boundary and exchange law; C is injected at depth 0."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("boundary_type", "input_conditional")
        kwargs["topology"] = "cocurrent"
        super().__init__(**kwargs)


class InvertibleBVPCountercurrentCNN(BVPCounterflow):
    """P0: reverse transport is the exact inverse of the forward transport."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("boundary_type", "input_conditional")
        kwargs["topology"] = "countercurrent"
        kwargs["flow"] = "invertible"
        super().__init__(**kwargs)


class InvertibleBVPCocurrentCNN(BVPCounterflow):
    """P0 co-current control with the same solver, boundary and exchange law."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("boundary_type", "input_conditional")
        kwargs["topology"] = "cocurrent"
        kwargs["flow"] = "invertible"
        super().__init__(**kwargs)


__all__ = [
    "BVPCocurrentCNN",
    "BVPCountercurrentCNN",
    "BVPCounterflow",
    "InvertibleBVPCocurrentCNN",
    "InvertibleBVPCountercurrentCNN",
    "Topology",
]
