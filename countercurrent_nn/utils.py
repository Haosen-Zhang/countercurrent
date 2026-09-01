"""Small runtime utilities shared by commands."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def top1_correct(logits: Tensor, targets: Tensor) -> int:
    return int((logits.argmax(dim=1) == targets).sum().item())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot JSON-serialize {type(value).__name__}")


def write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")
    os.replace(temporary, output_path)


def append_jsonl(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, default=_json_default) + "\n")
