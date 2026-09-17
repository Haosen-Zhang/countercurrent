"""Forward-only recurrent compute control for the two-sweep models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .coupled import _normalized_l2, _relative_state_change
from .stem import Stem
from .transport import TransportBlock


class ForwardOnlyRecurrentCNN(nn.Module):
    """Iteratively refine H with source re-injection and no reverse pathway."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        channels: int = 64,
        depth: int = 4,
        refine_steps: int = 3,
        num_classes: int = 10,
        num_groups: int = 8,
        residual_scale: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        if depth <= 0 or refine_steps <= 0:
            raise ValueError("depth and refine_steps must be positive")
        self.depth = int(depth)
        self.refine_steps = int(refine_steps)
        self.stem = Stem(in_channels, channels, num_groups)
        self.F = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale) for _ in range(depth)
        )
        # R supplies recurrent refinement capacity and matches the G-block compute.
        self.R = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale) for _ in range(depth)
        )
        self.head = nn.Linear(channels, num_classes)

    def _classify(self, state: Tensor) -> Tensor:
        return self.head(state.mean(dim=(2, 3)))

    def _initial_forward(self, source: Tensor) -> list[Tensor]:
        states = [source]
        for block in self.F:
            states.append(block(states[-1]))
        return states

    def _run(self, x: Tensor, capture: bool) -> tuple[Tensor, list[Tensor], dict[str, Tensor]]:
        states = self._initial_forward(self.stem(x))
        logits = self._classify(states[-1])
        iteration_logits = [logits]
        history: list[Tensor] = []
        norms: list[Tensor] = []
        residuals: list[Tensor] = []
        if capture:
            history.append(torch.stack([state.detach().cpu() for state in states]))
            norms.append(
                torch.stack([_normalized_l2(state) for state in states]).detach().cpu()
            )

        for _ in range(self.refine_steps):
            new_states = [self.stem(x)]
            for position in range(self.depth):
                proposal = self.F[position](new_states[position])
                mixed = 0.5 * (proposal + states[position + 1])
                new_states.append(self.R[position](mixed))
            if capture:
                residuals.append(_relative_state_change(states, new_states).detach().cpu())
            states = new_states
            logits = self._classify(states[-1])
            iteration_logits.append(logits)
            if capture:
                history.append(torch.stack([state.detach().cpu() for state in states]))
                norms.append(
                    torch.stack([_normalized_l2(state) for state in states])
                    .detach()
                    .cpu()
                )

        diagnostics: dict[str, Tensor] = {}
        if capture:
            diagnostics = {
                "H_history": torch.stack(history),
                "H_norm": torch.stack(norms),
                "residual": torch.stack(residuals),
                "iter_logits": torch.stack(
                    [value.detach().cpu() for value in iteration_logits]
                ),
                "initial_logits": iteration_logits[0].detach().cpu(),
            }
        return logits, iteration_logits, diagnostics

    def predict_iterations(self, x: Tensor) -> Tensor:
        _, logits, _ = self._run(x, False)
        return torch.stack(logits)

    def forward(
        self, x: Tensor, return_diagnostics: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        logits, _, diagnostics = self._run(x, return_diagnostics)
        return (logits, diagnostics) if return_diagnostics else logits


SingleStreamRecurrentCNN = ForwardOnlyRecurrentCNN

__all__ = ["ForwardOnlyRecurrentCNN", "SingleStreamRecurrentCNN"]
