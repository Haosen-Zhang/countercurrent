"""Proposal, paired reconciliation, and corrected forward resweep (POC §15).

The reconciled reverse states are live inputs to the resweep. Recording C+Q
without consuming it would reduce this operator to one-way feedback.
"""

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
        inference_version: int = 3,
    ) -> None:
        super().__init__()
        if topology not in ("countercurrent", "cocurrent"):
            raise ValueError(f"unsupported topology: {topology!r}")
        if depth <= 0 or refine_steps <= 0:
            raise ValueError("depth and refine_steps must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if inference_version not in (2, 3):
            raise ValueError("inference_version must be 2 or 3")

        self.topology: Topology = topology
        self.channels = int(channels)
        self.depth = int(depth)
        self.refine_steps = int(refine_steps)
        self.boundary_type: BoundaryKind = boundary_type
        self.exchange_enabled = bool(exchange_enabled)
        # Version 2 remains executable only so completed checkpoints retain their
        # original meaning. Fresh models default to the sequential reverse sweep.
        self.register_buffer("_inference_version", torch.tensor(inference_version))

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

    def _paired_exchange(
        self, position: int, h_bar: Tensor, c_bar: Tensor, reverse_off: bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        discrepancy = h_bar - c_bar
        if self.exchange_enabled and not reverse_off:
            h_out, c_out, q = self.exchange[position].paired_update(h_bar, c_bar)
        else:
            q = torch.zeros_like(discrepancy)
            h_out, c_out = h_bar - q, c_bar + q
        return h_out, c_out, discrepancy, q

    def _refine(
        self, source: Tensor, boundary: Tensor, *, reverse_off: bool
    ) -> tuple[list[Tensor], list[Tensor], dict[str, list[Tensor]]]:
        """One finite two-sweep update; no equilibrium solver or hidden loop.

        Version 3 transports and reconciles C one cell at a time, so each C+Q
        is the live inlet to the next G block. Version 2's independent reverse
        proposal remains available only for completed-checkpoint reproduction.
        Both versions resweep F using corrected C and a second paired flux.

        The first reconciliation's C outputs are consumed in D. The final C
        states are retained as the iteration's output; the next outer iteration
        rebuilds its boundaries from x and the revised logits, as in POC §15.
        """
        h_proposal = self._pure_forward(source)
        if int(self._inference_version.item()) == 2:
            c_proposal, reconciled_h, reconciled_c, proposal_d, proposal_q = (
                self._proposal_then_reconcile(h_proposal, boundary, reverse_off)
            )
        else:
            c_proposal, reconciled_h, reconciled_c, proposal_d, proposal_q = (
                self._sequential_reverse_reconcile(h_proposal, boundary, reverse_off)
            )

        new_h = [source]  # hard clamp, never the corrected source-side C state
        new_c = list(reconciled_c)
        resweep_h_bar, resweep_d, resweep_q = [], [], []
        for position in range(self.depth):
            h_bar = self.F[position](new_h[position])
            c_index = self._reverse_reference_index(position)
            h_out, c_out, d, q = self._paired_exchange(
                position, h_bar, reconciled_c[c_index], reverse_off
            )
            new_h.append(h_out)
            new_c[c_index] = c_out
            resweep_h_bar.append(h_bar)
            resweep_d.append(d)
            resweep_q.append(q)
        return new_h, new_c, {
            "H_proposal": h_proposal,
            "C_proposal": c_proposal,
            "H_reconciled": reconciled_h,
            "C_reconciled": reconciled_c,
            "proposal_D": proposal_d,
            "proposal_Q": proposal_q,
            "resweep_H_bar": resweep_h_bar,
            "D": resweep_d,
            "Q": resweep_q,
        }

    def _proposal_then_reconcile(
        self, h_proposal: list[Tensor], boundary: Tensor, reverse_off: bool
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """Version-2 schedule retained for exact evaluation of old checkpoints."""
        c_proposal = self._reverse_proposal(boundary)
        reconciled_h = [h_proposal[0]]
        reconciled_c = list(c_proposal)
        proposal_d, proposal_q = [], []
        for position in range(self.depth):
            c_index = self._reverse_reference_index(position)
            h_out, c_out, d, q = self._paired_exchange(
                position, h_proposal[position + 1], c_proposal[c_index], reverse_off
            )
            reconciled_h.append(h_out)
            reconciled_c[c_index] = c_out
            proposal_d.append(d)
            proposal_q.append(q)
        return c_proposal, reconciled_h, reconciled_c, proposal_d, proposal_q

    def _sequential_reverse_reconcile(
        self, h_proposal: list[Tensor], boundary: Tensor, reverse_off: bool
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """Transport and exchange each C state before it enters the next G block."""
        c_transport: list[Tensor | None] = [None] * (self.depth + 1)
        reconciled_c: list[Tensor | None] = [None] * (self.depth + 1)
        reconciled_h: list[Tensor | None] = [None] * (self.depth + 1)
        boundary_index = self.depth if self.topology == "countercurrent" else 0
        c_transport[boundary_index] = boundary
        reconciled_c[boundary_index] = boundary
        reconciled_h[0] = h_proposal[0]
        order = reversed(range(self.depth)) if self.topology == "countercurrent" else range(self.depth)
        proposal_d: list[Tensor | None] = [None] * self.depth
        proposal_q: list[Tensor | None] = [None] * self.depth
        for position in order:
            inlet_index = position + 1 if self.topology == "countercurrent" else position
            output_index = self._reverse_reference_index(position)
            inlet = reconciled_c[inlet_index]
            if inlet is None:
                raise RuntimeError("sequential reverse inlet is undefined")
            c_bar = self.G[position](inlet)
            c_transport[output_index] = c_bar
            h_out, c_out, d, q = self._paired_exchange(
                position, h_proposal[position + 1], c_bar, reverse_off
            )
            reconciled_h[position + 1] = h_out
            reconciled_c[output_index] = c_out
            proposal_d[position] = d
            proposal_q[position] = q
        values = (c_transport, reconciled_h, reconciled_c, proposal_d, proposal_q)
        if any(any(value is None for value in sequence) for sequence in values):
            raise RuntimeError("sequential reverse sweep left an undefined state")
        return (
            [value for value in c_transport if value is not None],
            [value for value in reconciled_h if value is not None],
            [value for value in reconciled_c if value is not None],
            [value for value in proposal_d if value is not None],
            [value for value in proposal_q if value is not None],
        )

    def _run(
        self,
        x: Tensor,
        *,
        capture_diagnostics: bool,
        reverse_off: bool,
        oracle_boundary: Tensor | None = None,
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
        residual_history: list[Tensor] = []
        trace_history: dict[str, list[Tensor]] = {}

        if capture_diagnostics:
            h_history.append(torch.stack([state.detach().cpu() for state in h_states]))
            h_norm_history.append(
                torch.stack([_normalized_l2(state) for state in h_states]).detach().cpu()
            )

        for _ in range(self.refine_steps):
            # Recompute and hard-clamp source evidence at every outer iteration.
            source = self.stem(x)
            if oracle_boundary is None:
                boundary, probabilities = self.target_boundary(logits, source)
            else:
                boundary = oracle_boundary[:, :, None, None].expand_as(source)
                probabilities = logits.softmax(dim=-1)
            new_h, corrected_c, trace = self._refine(
                source, boundary, reverse_off=reverse_off
            )

            if capture_diagnostics:
                for key, states in trace.items():
                    trace_history.setdefault(key, []).append(
                        torch.stack([state.detach().cpu() for state in states])
                    )
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

        diagnostics: dict[str, Tensor] = {}
        if capture_diagnostics:
            diagnostics = {
                **{key: torch.stack(values) for key, values in trace_history.items()},
                "H_history": torch.stack(h_history),
                "C_history": torch.stack(c_history),
                "boundary_history": torch.stack(boundary_history),
                "boundary_probabilities": torch.stack(probability_history),
                "H_norm": torch.stack(h_norm_history),
                "C_norm": torch.stack(c_norm_history),
                "residual": torch.stack(residual_history),
                "iter_logits": torch.stack(
                    [value.detach().cpu() for value in iteration_logits]
                ),
                "initial_logits": iteration_logits[0].detach().cpu(),
                "gamma": torch.stack(
                    [block.gamma.detach().cpu().flatten() for block in self.exchange]
                ),
                "inference_version": self._inference_version.detach().cpu().clone(),
            }
            for stage in ("proposal_", ""):
                for field, name in (("D", "discrepancy"), ("Q", "q_norm")):
                    values = diagnostics[stage + field].float().flatten(2)
                    diagnostics[stage + name] = values.norm(dim=-1) / values.shape[-1]
                d = diagnostics[stage + "D"].float()
                gamma = diagnostics["gamma"][None, :, None, :, None, None]
                diagnostics[stage + "exchange_energy"] = (0.5 * gamma * d.square()).flatten(2).mean(-1)
            p = diagnostics["iter_logits"].softmax(dim=-1)
            diagnostics["confidence"] = p.amax(dim=-1)
            diagnostics["predictive_entropy"] = -(p * p.clamp_min(1e-12).log()).sum(-1)
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
        *,
        return_diagnostics: bool = False,
        reverse_off: bool = False,
        return_initial_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, dict[str, Tensor]]:
        # Return both differentiable outputs through DDP's forward invocation.
        if return_diagnostics and return_initial_logits:
            raise ValueError("request diagnostics or initial logits, not both")
        logits, iterations, diagnostics = self._run(
            x,
            capture_diagnostics=return_diagnostics,
            reverse_off=reverse_off,
        )
        if return_initial_logits:
            return logits, iterations[0]
        return (logits, diagnostics) if return_diagnostics else logits

    def extra_repr(self) -> str:
        return (
            f"topology={self.topology}, boundary={self.boundary_type}, "
            f"channels={self.channels}, depth={self.depth}, "
            f"refine_steps={self.refine_steps}, exchange_enabled={self.exchange_enabled}, "
            f"inference_version={int(self._inference_version.item())}"
        )
