"""Opposing-direction coupled CNN models."""

from __future__ import annotations

from .coupled import CoupledLatticeBase


class CountercurrentCNN(CoupledLatticeBase):
    """Main model: endogenous target boundary transported against H."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("boundary_type", "self_generated")
        super().__init__(topology="countercurrent", exchange_enabled=True, **kwargs)


class CountercurrentNoExchangeCNN(CoupledLatticeBase):
    """Training-time no-exchange control; reverse-off evaluation is preferred."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("boundary_type", "self_generated")
        super().__init__(topology="countercurrent", exchange_enabled=False, **kwargs)


class NullBoundaryCountercurrentCNN(CoupledLatticeBase):
    """Counterflow topology ablation with C_L fixed to zero."""

    def __init__(self, **kwargs: object) -> None:
        kwargs["boundary_type"] = "null"
        super().__init__(topology="countercurrent", exchange_enabled=True, **kwargs)


class LearnedBoundaryCountercurrentCNN(CoupledLatticeBase):
    """Counterflow ablation with a sample-independent learned C_L."""

    def __init__(self, **kwargs: object) -> None:
        kwargs["boundary_type"] = "learned"
        super().__init__(topology="countercurrent", exchange_enabled=True, **kwargs)
