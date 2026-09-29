"""Model registry for all proof-of-concept controls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .boundary import (
    InputConditionalTargetBoundary,
    LearnedTargetBoundary,
    NullTargetBoundary,
    SelfGeneratedTargetBoundary,
)
from .bvp_lattice import (
    BVPCocurrentCNN,
    BVPCountercurrentCNN,
    InvertibleBVPCocurrentCNN,
    InvertibleBVPCountercurrentCNN,
)
from .cocurrent import CocurrentCNN
from .countercurrent import (
    CountercurrentCNN,
    CountercurrentNoExchangeCNN,
    LearnedBoundaryCountercurrentCNN,
    NullBoundaryCountercurrentCNN,
)
from .exchange import ChannelwiseConductance, ExchangeBlock
from .feedback_fusion import FeedbackFusionCNN
from .parallel_two_stream import ParallelTwoStreamCNN
from .persistent_bvp import (
    PersistentCanonicalCocurrent,
    PersistentCanonicalCountercurrent,
    PersistentCanonicalCounterflow,
)
from .recurrent_forward import ForwardOnlyRecurrentCNN, SingleStreamRecurrentCNN
from .single_stream import (
    ClassicCNN,
    SingleStreamFeedForwardCNN,
)
from .stem import Stem
from .transport import TransportBlock

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "classic_cnn": SingleStreamFeedForwardCNN,
    "single_feedforward": SingleStreamFeedForwardCNN,
    "forward_recurrent": ForwardOnlyRecurrentCNN,
    "single_recurrent": SingleStreamRecurrentCNN,
    "cc_no_exchange": CountercurrentNoExchangeCNN,
    "countercurrent_null": NullBoundaryCountercurrentCNN,
    "countercurrent_learnedz": LearnedBoundaryCountercurrentCNN,
    "feedback_fusion": FeedbackFusionCNN,
    "cocurrent": CocurrentCNN,
    "countercurrent": CountercurrentCNN,
    "parallel_two_stream": ParallelTwoStreamCNN,
    "bvp_countercurrent": BVPCountercurrentCNN,
    "bvp_cocurrent": BVPCocurrentCNN,
    "inv_bvp_countercurrent": InvertibleBVPCountercurrentCNN,
    "inv_bvp_cocurrent": InvertibleBVPCocurrentCNN,
    "persistent_canonical_countercurrent": PersistentCanonicalCountercurrent,
    "persistent_canonical_cocurrent": PersistentCanonicalCocurrent,
}


def build_model(config: Mapping[str, Any] | str, **overrides: Any) -> nn.Module:
    """Build a registered model from a name or a model-config mapping."""
    if isinstance(config, str):
        name = config
        kwargs: dict[str, Any] = {}
    else:
        kwargs = dict(config)
        try:
            name = str(kwargs.pop("name"))
        except KeyError as error:
            raise KeyError("model config must contain a 'name' field") from error
    kwargs.update(overrides)
    try:
        model_type = MODEL_REGISTRY[name]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model {name!r}; choose one of: {choices}") from error
    return model_type(**kwargs)


__all__ = [
    "BVPCocurrentCNN",
    "BVPCountercurrentCNN",
    "ClassicCNN",
    "InvertibleBVPCocurrentCNN",
    "InvertibleBVPCountercurrentCNN",
    "ChannelwiseConductance",
    "CocurrentCNN",
    "CountercurrentCNN",
    "CountercurrentNoExchangeCNN",
    "ExchangeBlock",
    "FeedbackFusionCNN",
    "ForwardOnlyRecurrentCNN",
    "LearnedBoundaryCountercurrentCNN",
    "InputConditionalTargetBoundary",
    "LearnedTargetBoundary",
    "MODEL_REGISTRY",
    "ParallelTwoStreamCNN",
    "PersistentCanonicalCocurrent",
    "PersistentCanonicalCountercurrent",
    "PersistentCanonicalCounterflow",
    "NullBoundaryCountercurrentCNN",
    "NullTargetBoundary",
    "SingleStreamFeedForwardCNN",
    "SingleStreamRecurrentCNN",
    "Stem",
    "SelfGeneratedTargetBoundary",
    "TransportBlock",
    "build_model",
]
