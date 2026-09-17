"""Independent two-branch capacity sanity control."""

from __future__ import annotations

from torch import Tensor, nn

from .stem import Stem
from .transport import TransportBlock


class ParallelTwoStreamCNN(nn.Module):
    """Two independent post-stem CNN branches with averaged representations."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        channels: int = 64,
        depth: int = 4,
        num_classes: int = 10,
        num_groups: int = 8,
        residual_scale: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.stem = Stem(in_channels, channels, num_groups)
        self.branch_a = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale) for _ in range(depth)
        )
        self.branch_b = nn.ModuleList(
            TransportBlock(channels, num_groups, residual_scale) for _ in range(depth)
        )
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        stem_state = self.stem(x)
        state_a = stem_state
        state_b = stem_state
        for block_a, block_b in zip(self.branch_a, self.branch_b, strict=True):
            state_a = block_a(state_a)
            state_b = block_b(state_b)
        state = 0.5 * (state_a + state_b)
        return self.head(state.mean(dim=(2, 3)))
