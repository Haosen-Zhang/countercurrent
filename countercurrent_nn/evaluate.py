"""Evaluate a saved checkpoint and optionally export lattice diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from countercurrent_nn.config import load_config
from countercurrent_nn.data import build_dataloaders
from countercurrent_nn.engine import (
    collect_diagnostics,
    evaluate,
    evaluate_refinement,
    summarize_diagnostics,
)
from countercurrent_nn.models import build_model
from countercurrent_nn.utils import resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--diagnostics-out", type=Path, default=None)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument(
        "--reverse-off", action="store_true", help="set every paired exchange Q to zero"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = load_config(args.config) if args.config is not None else checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no config; pass --config")
    seed = int(config.get("runtime", {}).get("seed", 0))
    data_config = dict(config["dataset"])
    data_config["batch_size"] = int(config["training"].get("batch_size", 128))
    data = build_dataloaders(data_config, seed=seed, synthetic=args.synthetic_data)
    model_config = dict(config["model"])
    model_config["num_classes"] = data.num_classes
    model = build_model(model_config).to(device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    loader = data.validation if args.split == "validation" else data.test
    if loader is None:
        raise ValueError("configuration does not define a validation split")
    forward_kwargs = {"reverse_off": True} if args.reverse_off else None
    metrics = evaluate(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        limit_batches=args.limit_batches,
        forward_kwargs=forward_kwargs,
    )
    refinement = evaluate_refinement(
        model,
        loader,
        device,
        limit_batches=args.limit_batches,
        reverse_off=args.reverse_off,
    )
    print({"task": metrics, "refinement": refinement})
    if args.diagnostics_out is not None:
        diagnostics = collect_diagnostics(
            model, loader, device, reverse_off=args.reverse_off
        )
        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(diagnostics, args.diagnostics_out)
        write_json(
            summarize_diagnostics(diagnostics),
            args.diagnostics_out.with_suffix(".json"),
        )


if __name__ == "__main__":
    main()
