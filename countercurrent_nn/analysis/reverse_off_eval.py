"""Evaluate a trained coupled model with normal inference and with Q fixed to zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from countercurrent_nn.config import load_config
from countercurrent_nn.data import build_dataloaders
from countercurrent_nn.engine import evaluate, evaluate_refinement
from countercurrent_nn.models import build_model
from countercurrent_nn.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--synthetic-data", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = load_config(args.config) if args.config else checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no config; pass --config")
    data_config = dict(config["dataset"])
    data_config["batch_size"] = int(config["training"].get("batch_size", 128))
    data = build_dataloaders(
        data_config,
        seed=int(config.get("runtime", {}).get("seed", 0)),
        synthetic=args.synthetic_data,
    )
    model_config = dict(config["model"])
    model_config["num_classes"] = data.num_classes
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    criterion = nn.CrossEntropyLoss()
    full = evaluate(
        model, data.test, criterion, device, limit_batches=args.limit_batches
    )
    off = evaluate(
        model,
        data.test,
        criterion,
        device,
        limit_batches=args.limit_batches,
        forward_kwargs={"reverse_off": True},
    )
    report = {
        "full": full,
        "reverse_off": off,
        "accuracy_drop": full["accuracy"] - off["accuracy"],
        "full_refinement": evaluate_refinement(
            model, data.test, device, limit_batches=args.limit_batches
        ),
        "reverse_off_refinement": evaluate_refinement(
            model,
            data.test,
            device,
            limit_batches=args.limit_batches,
            reverse_off=True,
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
