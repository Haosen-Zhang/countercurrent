"""Explicit GT oracle for analysis only; absent from the experiment registry."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from countercurrent_nn.models.boundary import SelfGeneratedTargetBoundary
from countercurrent_nn.models.countercurrent import CountercurrentCNN


class GTOracleCountercurrent(nn.Module):
    """Analyze a supplied CC model with B(one_hot(y) @ prototypes).

    This wrapper shares the supplied model's parameters. It is intentionally
    separate from forward(x), and its outputs are not ordinary test predictions.
    """

    def __init__(self, model: CountercurrentCNN) -> None:
        super().__init__()
        if not isinstance(model.target_boundary, SelfGeneratedTargetBoundary):
            raise ValueError("oracle analysis requires a self-generated boundary model")
        self.model = model

    @torch.no_grad()
    def forward(
        self, x: Tensor, labels: Tensor, *, return_diagnostics: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        boundary = self.model.target_boundary
        if labels.shape != (x.shape[0],) or labels.dtype != torch.long:
            raise ValueError("oracle labels must be int64 with shape [batch]")
        weights = F.one_hot(labels.to(x.device), boundary.num_classes).to(
            boundary.class_prototypes.dtype
        )
        channels = boundary.projector(weights @ boundary.class_prototypes)
        logits, _, diagnostics = self.model._run(
            x, capture_diagnostics=return_diagnostics,
            reverse_off=False, oracle_boundary=channels,
        )
        if return_diagnostics:
            diagnostics["oracle_analysis_only"] = torch.tensor(True)
            return logits, diagnostics
        return logits
