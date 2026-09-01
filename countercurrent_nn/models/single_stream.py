"""Classic feed-forward CNN and recurrent compute control."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .coupled import _normalized_l2
from .stem import Stem
from .transport import TransportBlock


class _SingleStreamBase(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        channels: int = 64,
        depth: int = 4,
        num_classes: int = 10,
        num_groups: int = 8,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.channels = channels
        self.depth = depth
        self.stem = Stem(in_channels, channels, num_groups)
        self.F = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale) for _ in range(depth)
        )
        self.head = nn.Linear(channels, num_classes)

    def _classify(self, state: Tensor) -> Tensor:
        return self.head(state.mean(dim=(2, 3)))


class SingleStreamFeedForwardCNN(_SingleStreamBase):
    """Classic residual CNN: Stem -> F0..FL-1 -> GAP -> Linear."""

    def __init__(self, **kwargs: object) -> None:
        # Generic experiment configs may include refinement fields.
        kwargs.pop("steps", None)
        kwargs.pop("refine_steps", None)
        kwargs.pop("prototype_dim", None)
        kwargs.pop("lattice_size", None)
        kwargs.pop("conductance_max", None)
        kwargs.pop("conductance_init_logit", None)
        kwargs.pop("boundary_type", None)
        super().__init__(**kwargs)

    def forward(
        self, x: Tensor, return_diagnostics: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        state = self.stem(x)
        states = [state]
        for block in self.F:
            state = block(state)
            states.append(state)
        logits = self._classify(state)
        if not return_diagnostics:
            return logits
        diagnostics = {
            "H_history": torch.stack(
                [torch.stack([item.detach().cpu() for item in states])]
            ),
            "H_norm": torch.stack(
                [torch.stack([_normalized_l2(item) for item in states]).detach().cpu()]
            ),
            "iter_logits": logits.detach().cpu().unsqueeze(0),
        }
        return logits, diagnostics


# Explicit name for the user's classic-CNN clone; it is also the mandated Single-FF control.
ClassicCNN = SingleStreamFeedForwardCNN
