"""Shared CIFAR stem used by every comparison model."""

from __future__ import annotations

from torch import Tensor, nn


def _check_groups(channels: int, groups: int) -> None:
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    if groups <= 0 or channels % groups:
        raise ValueError(
            f"num_groups must be positive and divide channels; got {groups=} and {channels=}"
        )


class Stem(nn.Module):
    """Map a 32x32 RGB image to a same-width 8x8 lattice boundary."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 64,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        _check_groups(channels, num_groups)
        layers: list[nn.Module] = []
        input_width = in_channels
        for stride in (1, 2, 2):
            layers.extend(
                [
                    nn.Conv2d(
                        input_width,
                        channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(num_groups, channels),
                    nn.SiLU(),
                ]
            )
            input_width = channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
