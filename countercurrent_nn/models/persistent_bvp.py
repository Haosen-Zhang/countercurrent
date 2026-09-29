"""V5-C persistent fixed-boundary counterflow solver."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .canonical_exchange import CanonicalExchange
from .decomposed_boundary import DecomposedTargetBoundary
from .stem import Stem
from .transport import TransportBlock

Topology = Literal["countercurrent", "cocurrent"]


def _state_rms(states: Tensor) -> Tensor:
    flattened = states.float().flatten(3)
    return flattened.square().mean(dim=-1).sqrt().mean(dim=2)


class PersistentCanonicalCounterflow(nn.Module):
    """Two persistent fields with fixed boundaries and canonical paired exchange."""

    def __init__(
        self,
        *,
        topology: Topology = "countercurrent",
        in_channels: int = 3,
        channels: int = 64,
        depth: int = 4,
        num_classes: int = 10,
        num_groups: int = 8,
        residual_scale: float = 0.1,
        train_inner_steps: int = 8,
        eval_max_steps: int = 24,
        num_outer_train: int = 1,
        num_outer_eval: int = 1,
        tolerance: float = 1e-3,
        damping: float = 0.5,
        conductance_max: float = 0.49,
        conductance_init_logit: float = -2.2,
        class_rank: int = 16,
        lattice_size: int = 8,
        boundary_base_scale: float = 0.15,
        boundary_class_scale: float = 1.0,
        boundary_instance_scale: float = 0.1,
        boundary_target_rms: float | None = 0.55,
        use_canonical_exchange: bool = True,
        match_canonical_compute: bool = False,
    ) -> None:
        super().__init__()
        if topology not in ("countercurrent", "cocurrent"):
            raise ValueError(f"unsupported topology {topology!r}")
        if min(channels, depth, train_inner_steps, eval_max_steps) <= 0:
            raise ValueError("channels, depth and inner step counts must be positive")
        if min(num_outer_train, num_outer_eval) <= 0:
            raise ValueError("outer step counts must be positive")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must lie in (0, 1]")
        if tolerance <= 0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be finite and positive")

        self.topology: Topology = topology
        self.channels = int(channels)
        self.depth = int(depth)
        self.train_inner_steps = int(train_inner_steps)
        self.eval_max_steps = int(eval_max_steps)
        self.num_outer_train = int(num_outer_train)
        self.num_outer_eval = int(num_outer_eval)
        self.tolerance = float(tolerance)
        self.damping = float(damping)
        self.lattice_size = int(lattice_size)
        self.exchange_enabled = True

        self.stem = Stem(in_channels, channels, num_groups)
        self.F = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale)
            for _ in range(depth)
        )
        self.G = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale)
            for _ in range(depth)
        )
        self.exchange = nn.ModuleList(
            CanonicalExchange(
                channels,
                conductance_max=conductance_max,
                conductance_init_logit=conductance_init_logit,
                use_canonical=use_canonical_exchange,
                match_canonical_compute=match_canonical_compute,
            )
            for _ in range(depth)
        )
        self.target_boundary = DecomposedTargetBoundary(
            num_classes=num_classes,
            channels=channels,
            spatial_size=lattice_size,
            class_rank=class_rank,
            num_groups=num_groups,
            base_scale=boundary_base_scale,
            class_scale=boundary_class_scale,
            instance_scale=boundary_instance_scale,
            target_rms=boundary_target_rms,
        )
        self.head = nn.Linear(channels, num_classes)
        self.register_buffer("_inference_version", torch.tensor(5, dtype=torch.int64))

    @property
    def boundary_index(self) -> int:
        return self.depth if self.topology == "countercurrent" else 0

    def _classify(self, terminal: Tensor) -> Tensor:
        return self.head(terminal.mean(dim=(2, 3)))

    def _pure_forward(self, source: Tensor) -> list[Tensor]:
        states = [source]
        for block in self.F:
            states.append(block(states[-1]))
        return states

    def _initialize_c(self, boundary: Tensor) -> list[Tensor]:
        states: list[Tensor | None] = [None] * (self.depth + 1)
        if self.topology == "countercurrent":
            states[self.depth] = boundary
            for layer in reversed(range(self.depth)):
                inlet = states[layer + 1]
                if inlet is None:
                    raise RuntimeError("undefined reverse inlet")
                states[layer] = self.G[layer](inlet)
        else:
            states[0] = boundary
            for layer in range(self.depth):
                inlet = states[layer]
                if inlet is None:
                    raise RuntimeError("undefined co-current inlet")
                states[layer + 1] = self.G[layer](inlet)
        result = [state for state in states if state is not None]
        if len(result) != self.depth + 1:
            raise RuntimeError("C initialization left an undefined state")
        return result

    def _fixed_point_map(
        self,
        h_states: list[Tensor],
        c_states: list[Tensor],
        *,
        source: Tensor,
        boundary: Tensor,
        reverse_off: bool = False,
    ) -> tuple[list[Tensor], list[Tensor], list[dict[str, Tensor]]]:
        """Jacobi map: every cell reads only the previous H/C state."""
        h_candidate = [source] + [state for state in h_states[1:]]
        c_candidate = list(c_states)
        cells: list[dict[str, Tensor]] = []
        for layer in range(self.depth):
            h_bar = self.F[layer](h_states[layer])
            if self.topology == "countercurrent":
                c_bar = self.G[layer](c_states[layer + 1])
                c_index = layer
            else:
                c_bar = self.G[layer](c_states[layer])
                c_index = layer + 1
            h_out, c_out, exchange = self.exchange[layer](
                h_bar, c_bar, reverse_off=reverse_off
            )
            h_candidate[layer + 1] = h_out
            c_candidate[c_index] = c_out
            cells.append({"h_bar": h_bar, "c_bar": c_bar, **exchange})
        h_candidate[0] = source
        c_candidate[self.boundary_index] = boundary
        return h_candidate, c_candidate, cells

    def _interior_pairs(
        self,
        h_states: list[Tensor],
        c_states: list[Tensor],
    ) -> list[Tensor]:
        c_interior = (
            c_states[: self.depth]
            if self.topology == "countercurrent"
            else c_states[1:]
        )
        return h_states[1:] + c_interior

    def _residual_per_sample(
        self,
        h_states: list[Tensor],
        c_states: list[Tensor],
        h_candidate: list[Tensor],
        c_candidate: list[Tensor],
    ) -> Tensor:
        current = self._interior_pairs(h_states, c_states)
        candidate = self._interior_pairs(h_candidate, c_candidate)
        numerator = sum(
            (new.float() - old.float()).flatten(1).square().sum(dim=1)
            for old, new in zip(current, candidate, strict=True)
        )
        denominator = sum(
            state.float().flatten(1).square().sum(dim=1) for state in current
        )
        return (numerator / (denominator + 1e-12)).sqrt()

    def _damped_update(
        self,
        h_states: list[Tensor],
        c_states: list[Tensor],
        h_candidate: list[Tensor],
        c_candidate: list[Tensor],
        *,
        source: Tensor,
        boundary: Tensor,
        update_mask: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor], Tensor]:
        def update(old: Tensor, candidate: Tensor) -> Tensor:
            damped = old + self.damping * (candidate - old)
            if update_mask is None:
                return damped
            mask = update_mask.reshape(-1, 1, 1, 1)
            return torch.where(mask, damped, old)

        new_h = [source]
        new_h.extend(
            update(old, candidate)
            for old, candidate in zip(
                h_states[1:], h_candidate[1:], strict=True
            )
        )
        new_c = list(c_states)
        interior_indices = (
            range(self.depth)
            if self.topology == "countercurrent"
            else range(1, self.depth + 1)
        )
        for index in interior_indices:
            new_c[index] = update(c_states[index], c_candidate[index])
        new_c[self.boundary_index] = boundary
        step_residual = self._residual_per_sample(
            h_states, c_states, new_h, new_c
        )
        return new_h, new_c, step_residual

    @staticmethod
    def _cpu_state(states: list[Tensor]) -> list[Tensor]:
        return [state.detach().cpu() for state in states]

    @staticmethod
    def _cpu_cells(cells: list[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
        def rms(value: Tensor) -> Tensor:
            return value.detach().float().flatten(1).square().mean(dim=1).sqrt().cpu()

        def cosine(left: Tensor, right: Tensor) -> Tensor:
            return F.cosine_similarity(
                left.detach().float().flatten(1),
                right.detach().float().flatten(1),
                dim=1,
            ).cpu()

        return [
            {
                "u_h_norm": rms(cell["u_h"]),
                "u_c_norm": rms(cell["u_c"]),
                "D_norm": rms(cell["D"]),
                "Q_norm": rms(cell["Q"]),
                "cos_u_h_u_c": cosine(cell["u_h"], cell["u_c"]),
                "cos_d_u_h": cosine(cell["D"], cell["u_h"]),
                "cos_q_u_h": cosine(cell["Q"], cell["u_h"]),
            }
            for cell in cells
        ]

    def _build_boundary(
        self, logits: Tensor, terminal: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        result = self.target_boundary(
            logits, terminal, return_components=True
        )
        if len(result) != 3:
            raise RuntimeError("decomposed boundary did not return components")
        return result

    def unroll_train(
        self,
        x: Tensor,
        *,
        reverse_off: bool = False,
        capture_diagnostics: bool = False,
    ) -> tuple[Tensor, list[Tensor], dict[str, Any]]:
        """Fixed-depth full BPTT; tolerance never changes training graph depth."""
        source = self.stem(x)
        h_states = self._pure_forward(source)
        logits = self._classify(h_states[-1])
        iteration_logits = [logits]
        c_states: list[Tensor] | None = None
        history: list[dict[str, Any]] = []
        state_records: list[dict[str, list[Tensor]]] = []
        boundaries: list[Tensor] = []
        boundary_probabilities: list[Tensor] = []
        boundary_components: list[dict[str, Tensor]] = []

        for _outer in range(self.num_outer_train):
            boundary, probabilities, components = self._build_boundary(
                logits, h_states[-1]
            )
            boundaries.append(boundary.detach().cpu())
            boundary_probabilities.append(probabilities.detach().cpu())
            boundary_components.append(
                {key: value.detach().cpu() for key, value in components.items()}
            )
            if c_states is None:
                c_states = self._initialize_c(boundary)
            else:
                c_states[self.boundary_index] = boundary
            if capture_diagnostics and not state_records:
                state_records.append(
                    {"H": self._cpu_state(h_states), "C": self._cpu_state(c_states)}
                )
            for _step in range(self.train_inner_steps):
                candidate_h, candidate_c, cells = self._fixed_point_map(
                    h_states,
                    c_states,
                    source=source,
                    boundary=boundary,
                    reverse_off=reverse_off,
                )
                equation = self._residual_per_sample(
                    h_states, c_states, candidate_h, candidate_c
                )
                h_states, c_states, step_residual = self._damped_update(
                    h_states,
                    c_states,
                    candidate_h,
                    candidate_c,
                    source=source,
                    boundary=boundary,
                )
                logits = self._classify(h_states[-1])
                iteration_logits.append(logits)
                history.append(
                    {
                        "equation_residual": equation.detach().cpu(),
                        "step_residual": step_residual.detach().cpu(),
                        "cells": self._cpu_cells(cells) if capture_diagnostics else [],
                    }
                )
                if capture_diagnostics:
                    state_records.append(
                        {"H": self._cpu_state(h_states), "C": self._cpu_state(c_states)}
                    )

        trace = {
            "history": history,
            "states": state_records,
            "boundaries": boundaries,
            "boundary_probabilities": boundary_probabilities,
            "boundary_components": boundary_components,
            "steps_per_sample": torch.full(
                (x.shape[0],),
                self.train_inner_steps * self.num_outer_train,
                dtype=torch.int64,
            ),
            "converged": torch.zeros(x.shape[0], dtype=torch.bool),
        }
        return logits, iteration_logits, trace

    def solve_eval(
        self,
        x: Tensor,
        *,
        reverse_off: bool = False,
        capture_diagnostics: bool = False,
    ) -> tuple[Tensor, list[Tensor], dict[str, Any]]:
        """Adaptive fixed-boundary solve with independent per-sample stopping."""
        source = self.stem(x)
        h_states = self._pure_forward(source)
        logits = self._classify(h_states[-1])
        iteration_logits = [logits]
        c_states: list[Tensor] | None = None
        history: list[dict[str, Any]] = []
        state_records: list[dict[str, list[Tensor]]] = []
        boundaries: list[Tensor] = []
        boundary_probabilities: list[Tensor] = []
        boundary_components: list[dict[str, Tensor]] = []
        total_steps = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        final_converged = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

        for _outer in range(self.num_outer_eval):
            boundary, probabilities, components = self._build_boundary(
                logits, h_states[-1]
            )
            boundaries.append(boundary.detach().cpu())
            boundary_probabilities.append(probabilities.detach().cpu())
            boundary_components.append(
                {key: value.detach().cpu() for key, value in components.items()}
            )
            if c_states is None:
                c_states = self._initialize_c(boundary)
            else:
                c_states[self.boundary_index] = boundary
            if capture_diagnostics and not state_records:
                state_records.append(
                    {"H": self._cpu_state(h_states), "C": self._cpu_state(c_states)}
                )

            active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            for _step in range(self.eval_max_steps):
                candidate_h, candidate_c, cells = self._fixed_point_map(
                    h_states,
                    c_states,
                    source=source,
                    boundary=boundary,
                    reverse_off=reverse_off,
                )
                equation = self._residual_per_sample(
                    h_states, c_states, candidate_h, candidate_c
                )
                total_steps += active.to(total_steps.dtype)
                converged_now = active & equation.lt(self.tolerance)
                update_mask = active & ~converged_now
                h_states, c_states, step_residual = self._damped_update(
                    h_states,
                    c_states,
                    candidate_h,
                    candidate_c,
                    source=source,
                    boundary=boundary,
                    update_mask=update_mask,
                )
                logits = self._classify(h_states[-1])
                iteration_logits.append(logits)
                history.append(
                    {
                        "equation_residual": equation.detach().cpu(),
                        "step_residual": step_residual.detach().cpu(),
                        "active": active.detach().cpu(),
                        "cells": self._cpu_cells(cells) if capture_diagnostics else [],
                    }
                )
                if capture_diagnostics:
                    state_records.append(
                        {"H": self._cpu_state(h_states), "C": self._cpu_state(c_states)}
                    )
                active = update_mask
                if not bool(active.any()):
                    break
            final_converged = ~active

        trace = {
            "history": history,
            "states": state_records,
            "boundaries": boundaries,
            "boundary_probabilities": boundary_probabilities,
            "boundary_components": boundary_components,
            "steps_per_sample": total_steps.detach().cpu(),
            "converged": final_converged.detach().cpu(),
        }
        return logits, iteration_logits, trace

    def _pad_logits(self, logits: list[Tensor], expected: int) -> list[Tensor]:
        if len(logits) < expected:
            logits = logits + [logits[-1]] * (expected - len(logits))
        return logits

    def _diagnostics(
        self, iteration_logits: list[Tensor], trace: dict[str, Any]
    ) -> dict[str, Tensor]:
        history = trace["history"]
        states = trace["states"]
        h_history = torch.stack([torch.stack(record["H"]) for record in states])
        c_history = torch.stack([torch.stack(record["C"]) for record in states])
        equation = torch.stack([record["equation_residual"] for record in history])
        step_residual = torch.stack([record["step_residual"] for record in history])
        expected = 1 + (
            self.train_inner_steps * self.num_outer_train
            if self.training
            else self.eval_max_steps * self.num_outer_eval
        )
        padded_logits = self._pad_logits(iteration_logits, expected)
        diagnostics: dict[str, Tensor] = {
            "H_history": h_history,
            "C_history": c_history,
            "H_norm": _state_rms(h_history),
            "C_norm": _state_rms(c_history),
            "equation_residual": equation,
            "residual": equation.mean(dim=1),
            "step_residual": step_residual,
            "solve_steps_per_sample": trace["steps_per_sample"],
            "converged": trace["converged"],
            "iter_logits": torch.stack(
                [value.detach().cpu() for value in padded_logits]
            ),
            "initial_logits": padded_logits[0].detach().cpu(),
            "gamma": torch.stack(
                [cell.gamma.detach().cpu().flatten() for cell in self.exchange]
            ),
            "boundary_history": torch.stack(trace["boundaries"]),
            "boundary_probabilities": torch.stack(trace["boundary_probabilities"]),
        }
        if history and history[0]["cells"]:
            for key in (
                "u_h_norm",
                "u_c_norm",
                "D_norm",
                "Q_norm",
                "cos_u_h_u_c",
                "cos_d_u_h",
                "cos_q_u_h",
            ):
                per_sample = torch.stack(
                    [
                        torch.stack([cell[key] for cell in record["cells"]])
                        for record in history
                    ]
                )
                diagnostics[key] = per_sample.mean(dim=2)
            diagnostics["discrepancy"] = diagnostics.pop("D_norm")
            diagnostics["q_norm"] = diagnostics.pop("Q_norm")
        component_norms = []
        component_ratios = []
        for details in trace["boundary_components"]:
            boundary = details["boundary"].float().flatten(1).norm(dim=1)
            gain = details["global_gain"].float()
            weighted = torch.stack(
                [
                    (gain * details[name].float()).flatten(1).norm(dim=1)
                    for name in (
                        "weighted_base",
                        "weighted_class",
                        "weighted_instance",
                    )
                ],
                dim=1,
            )
            component_norms.append(weighted.mean(dim=0))
            component_ratios.append((weighted / boundary[:, None].clamp_min(1e-12)).mean(dim=0))
        diagnostics["boundary_component_norm"] = torch.stack(component_norms)
        diagnostics["boundary_component_ratio"] = torch.stack(component_ratios)
        diagnostics["boundary_global_gain"] = torch.stack(
            [details["global_gain"].reshape(()) for details in trace["boundary_components"]]
        )
        return diagnostics

    def _run(
        self,
        x: Tensor,
        *,
        return_diagnostics: bool,
        reverse_off: bool,
    ) -> tuple[Tensor, list[Tensor], dict[str, Tensor]]:
        runner = self.unroll_train if self.training else self.solve_eval
        logits, iterations, trace = runner(
            x,
            reverse_off=reverse_off,
            capture_diagnostics=return_diagnostics,
        )
        diagnostics = self._diagnostics(iterations, trace) if return_diagnostics else {}
        return logits, iterations, diagnostics

    def run_solver(self, x: Tensor) -> tuple[Tensor, list[dict[str, Any]]]:
        logits, _, trace = self.solve_eval(x, capture_diagnostics=False)
        history = [
            {
                "step": index + 1,
                "equation_residual": record["equation_residual"],
                "relative_state_change": float(record["equation_residual"].mean()),
            }
            for index, record in enumerate(trace["history"])
        ]
        return logits, history

    def predict_iterations(self, x: Tensor, *, reverse_off: bool = False) -> Tensor:
        logits, iterations, _ = self._run(
            x, return_diagnostics=False, reverse_off=reverse_off
        )
        del logits
        expected = 1 + (
            self.train_inner_steps * self.num_outer_train
            if self.training
            else self.eval_max_steps * self.num_outer_eval
        )
        return torch.stack(self._pad_logits(iterations, expected))

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
            x,
            return_diagnostics=return_diagnostics,
            reverse_off=reverse_off,
        )
        if return_initial_logits:
            return logits, iterations[0]
        return (logits, diagnostics) if return_diagnostics else logits

    def extra_repr(self) -> str:
        return (
            f"topology={self.topology}, depth={self.depth}, "
            f"train_inner_steps={self.train_inner_steps}, "
            f"eval_max_steps={self.eval_max_steps}, damping={self.damping}, "
            f"tolerance={self.tolerance}"
        )


class PersistentCanonicalCountercurrent(PersistentCanonicalCounterflow):
    def __init__(self, **kwargs: Any) -> None:
        kwargs["topology"] = "countercurrent"
        super().__init__(**kwargs)


class PersistentCanonicalCocurrent(PersistentCanonicalCounterflow):
    def __init__(self, **kwargs: Any) -> None:
        kwargs["topology"] = "cocurrent"
        super().__init__(**kwargs)


__all__ = [
    "PersistentCanonicalCocurrent",
    "PersistentCanonicalCountercurrent",
    "PersistentCanonicalCounterflow",
    "Topology",
]
