"""Same-direction scientific control for CountercurrentCNN."""

from __future__ import annotations

from .coupled import CoupledLatticeBase


class CocurrentCNN(CoupledLatticeBase):
    """Matched control: the identical soft hypothesis enters C at depth zero."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("boundary_type", "self_generated")
        super().__init__(topology="cocurrent", exchange_enabled=True, **kwargs)
