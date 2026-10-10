"""Evaluate a V5 checkpoint under target-boundary component interventions."""

from __future__ import annotations

import argparse
import json
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from countercurrent_nn.config import load_config
from countercurrent_nn.data import build_dataloaders
from countercurrent_nn.engine import evaluate
from countercurrent_nn.models import build_model
from countercurrent_nn.models.decomposed_boundary import DecomposedTargetBoundary
from countercurrent_nn.utils import resolve_device, write_json


VARIANTS = (
    "real",
    "uniform",
    "shuffled",
    "class_off",
    "instance_off",
    "base_only",
    "zero",
)


def _intervened_forward(mode: str):
    def forward(
        module: DecomposedTargetBoundary,
        logits: Tensor,
        reference: Tensor,
        *,
        return_components: bool = False,
    ):
        if mode == "uniform":
            probabilities = torch.full_like(logits, 1.0 / module.num_classes)
        elif mode == "shuffled":
            probabilities = logits.roll(1, dims=0).softmax(dim=-1)
        else:
            probabilities = logits.softmax(dim=-1)
        components = module.components_from_probabilities(probabilities, reference)
        effective = dict(components)
        if mode == "class_off":
            effective["weighted_class"] = torch.zeros_like(
                components["weighted_class"]
            )
        elif mode == "instance_off":
            effective["weighted_instance"] = torch.zeros_like(
                components["weighted_instance"]
            )
        elif mode == "base_only":
            effective["weighted_class"] = torch.zeros_like(
                components["weighted_class"]
            )
            effective["weighted_instance"] = torch.zeros_like(
                components["weighted_instance"]
            )
        elif mode == "zero":
            for key in ("weighted_base", "weighted_class", "weighted_instance"):
                effective[key] = torch.zeros_like(components[key])
        combined = (
            effective["weighted_base"]
            + effective["weighted_class"]
            + effective["weighted_instance"]
        )
        boundary = module.global_gain * combined
        if not return_components:
            return boundary, probabilities
        effective["combined"] = combined
        effective["boundary"] = boundary
        effective["global_gain"] = module.global_gain
        return boundary, probabilities, effective

    return forward


@contextmanager
def boundary_intervention(
    module: DecomposedTargetBoundary, mode: str
) -> Iterator[None]:
    """Temporarily replace the boundary forward pass with one named intervention."""
    if mode not in VARIANTS:
        raise ValueError(f"unknown boundary intervention {mode!r}")
    original = module.forward
    if mode != "real":
        module.forward = types.MethodType(_intervened_forward(mode), module)
    try:
        yield
    finally:
        module.forward = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config: dict[str, Any] | None = (
        load_config(args.config) if args.config else checkpoint.get("config")
    )
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no config; pass --config")
    data_config = dict(config["dataset"])
    data_config["batch_size"] = args.batch_size
    if args.data_root is not None:
        data_config["root"] = str(args.data_root)
    data = build_dataloaders(
        data_config,
        seed=int(config.get("runtime", {}).get("seed", 0)),
        synthetic=False,
    )
    model_config = dict(config["model"])
    model_config["num_classes"] = data.num_classes
    model = build_model(model_config).to(device).eval()
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    boundary = getattr(model, "target_boundary", None)
    if not isinstance(boundary, DecomposedTargetBoundary):
        raise TypeError("model does not use DecomposedTargetBoundary")

    criterion = nn.CrossEntropyLoss()
    results: dict[str, dict[str, float]] = {}
    for mode in args.variants:
        with boundary_intervention(boundary, mode):
            results[mode] = evaluate(
                model,
                data.test,
                criterion,
                device,
                limit_batches=args.limit_batches,
            )
        print(mode, json.dumps(results[mode]), flush=True)
    real_accuracy = results.get("real", {}).get("accuracy")
    report: dict[str, Any] = {"checkpoint": str(args.checkpoint), "results": results}
    if real_accuracy is not None:
        report["accuracy_delta_vs_real"] = {
            mode: metrics["accuracy"] - real_accuracy
            for mode, metrics in results.items()
        }
    if args.output:
        write_json(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
