from __future__ import annotations

import argparse
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from countercurrent_nn.config import load_config
from countercurrent_nn.data import build_dataloaders
from countercurrent_nn.engine import build_optimizer, evaluate, train_one_epoch
from countercurrent_nn.models import build_model
from countercurrent_nn.profiling import count_macs
from countercurrent_nn.utils import parameter_counts, set_seed


class FirstForwardMacCounter:
    def __init__(self, model: nn.Module) -> None:
        self.total = 0
        self.active = True
        self.handles: list[Any] = []

        def conv(module: nn.Conv2d, _inputs: Any, output: Tensor) -> None:
            if self.active:
                kernel = module.kernel_size[0] * module.kernel_size[1]
                kernel *= module.in_channels // module.groups
                self.total += int(output.numel() * kernel)

        def linear(module: nn.Linear, _inputs: Any, output: Tensor) -> None:
            if self.active:
                self.total += int(output.numel() * module.in_features)

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(conv))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(linear))
        self.handles.append(model.register_forward_hook(self._finish))

    def _finish(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        del output
        self.active = False

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    config_path = Path("countercurrent_nn/configs") / f"cifar10_{args.arm}.yaml"
    config = load_config(config_path)
    seed = int(config["runtime"].get("seed", 0))
    set_seed(seed)
    device = torch.device(args.device)
    data_config = dict(config["dataset"])
    data_config["root"] = str(args.data_root)
    data_config["batch_size"] = int(config["training"]["batch_size"])
    data = build_dataloaders(data_config, seed=seed, synthetic=False)
    model = build_model(config["model"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config["training"])

    sample = torch.zeros(1, 3, 32, 32, device=device)
    upper_eval_macs = count_macs(model, sample)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    counter = FirstForwardMacCounter(model)
    train_metrics, _ = train_one_epoch(
        model,
        data.train,
        optimizer,
        criterion,
        device,
        global_step=0,
        total_steps=len(data.train),
        warmup_steps=int(config["training"]["warmup_epochs"]) * len(data.train),
        base_lr=float(config["training"]["lr"]),
        amp=False,
    )
    counter.close()
    train_peak = int(torch.cuda.max_memory_allocated(device))
    train_macs_per_sample = counter.total // int(config["training"]["batch_size"])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    validation_forward_kwargs = (
        {"fixed_steps": 8}
        if "fixed_steps" in inspect.signature(model.forward).parameters
        else None
    )
    validation_metrics = evaluate(
        model,
        data.validation,
        criterion,
        device,
        forward_kwargs=validation_forward_kwargs,
    )
    validation_peak = int(torch.cuda.max_memory_allocated(device))
    report = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "arm": args.arm,
        "device": torch.cuda.get_device_name(device),
        **parameter_counts(model),
        "eval_upper_macs_per_sample": upper_eval_macs,
        "train_macs_per_sample": train_macs_per_sample,
        "train": train_metrics,
        "validation": validation_metrics,
        "train_peak_memory_bytes": train_peak,
        "validation_peak_memory_bytes": validation_peak,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
