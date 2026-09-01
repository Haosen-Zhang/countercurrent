"""Profile parameters, approximate Conv/Linear FLOPs, latency, and CUDA memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from countercurrent_nn.config import load_config
from countercurrent_nn.models import build_model
from countercurrent_nn.profiling import profile_model
from countercurrent_nn.utils import resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    model = build_model(config["model"]).to(device)
    sample = torch.zeros(args.batch_size, 3, 32, 32, device=device)
    report = profile_model(
        model,
        sample,
        warmup_runs=args.warmup_runs,
        measured_runs=args.runs,
    )
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(report, args.output)


if __name__ == "__main__":
    main()
