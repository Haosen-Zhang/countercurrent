"""Shared helpers for diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_diagnostics(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"expected a diagnostic dictionary in {path}")
    return data


def default_output(input_path: Path, stem: str) -> Path:
    return input_path.with_name(f"{stem}.png")


def require_tensor(data: dict[str, Any], key: str) -> torch.Tensor:
    value = data.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"diagnostics do not contain tensor {key!r}")
    return value.float()
