"""Ordinary feedback-fusion control without discrepancy-driven paired flux."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .boundary import SelfGeneratedTargetBoundary
from .coupled import _normalized_l2, _relative_state_change
from .stem import Stem
from .transport import TransportBlock


class FeedbackFusionCNN(nn.Module):
    """Use a target hypothesis and reverse sweep, but fuse it only into H."""

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
        prototype_dim: int = 64,
        fusion_scale: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        if depth <= 0 or refine_steps <= 0 or fusion_scale <= 0:
            raise ValueError("depth, refine_steps, and fusion_scale must be positive")
        self.depth = int(depth)
        self.refine_steps = int(refine_steps)
        self.fusion_scale = float(fusion_scale)
        self.stem = Stem(in_channels, channels, num_groups)
        self.F = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale) for _ in range(depth)
        )
        self.G = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale) for _ in range(depth)
        )
        self.fusion = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=1) for _ in range(depth)
        )
        self.head = nn.Linear(channels, num_classes)
        self.target_boundary = SelfGeneratedTargetBoundary(
            num_classes=num_classes,
            channels=channels,
            prototype_dim=prototype_dim,
        )

    def _classify(self, state: Tensor) -> Tensor:
        return self.head(state.mean(dim=(2, 3)))

    def _run(self, x: Tensor, capture: bool) -> tuple[Tensor, list[Tensor], dict[str, Tensor]]:
        states = [self.stem(x)]
        for block in self.F:
            states.append(block(states[-1]))
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
            source = self.stem(x)
            boundary, _ = self.target_boundary(logits, source)
            reverse: list[Tensor | None] = [None] * (self.depth + 1)
            reverse[self.depth] = boundary
            for position in reversed(range(self.depth)):
                inlet = reverse[position + 1]
                if inlet is None:
                    raise RuntimeError("feedback reverse inlet is undefined")
                reverse[position] = self.G[position](inlet)
            new_states = [source]
            for position in range(self.depth):
                c_ref = reverse[position]
                if c_ref is None:
                    raise RuntimeError("feedback reference is undefined")
                proposal = self.F[position](new_states[position])
                new_states.append(
                    proposal + self.fusion_scale * self.fusion[position](c_ref)
                )
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


__all__ = ["FeedbackFusionCNN"]
