"""Dependency-free parameter, MAC, latency, and memory profiling."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from .utils import parameter_counts


def _conv_macs(module: nn.Conv2d, _inputs: tuple[Tensor, ...], output: Tensor) -> int:
    kernel_ops = module.kernel_size[0] * module.kernel_size[1]
    kernel_ops *= module.in_channels // module.groups
    return int(output.numel() * kernel_ops)


def _linear_macs(module: nn.Linear, _inputs: tuple[Tensor, ...], output: Tensor) -> int:
    return int(output.numel() * module.in_features)


def count_macs(model: nn.Module, sample: Tensor) -> int:
    """Count Conv2d/Linear MACs, including recurrent module invocations."""
    total = 0
    handles: list[Any] = []

    def add_hook(counter: Callable[[Any, Any, Any], int]) -> Callable[[Any, Any, Any], None]:
        def hook(module: Any, inputs: Any, output: Any) -> None:
            nonlocal total
            total += counter(module, inputs, output)

        return hook

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(add_hook(_conv_macs)))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(add_hook(_linear_macs)))

    was_training = model.training
    # Adaptive solvers execute data-dependent work.  Count their architectural
    # maximum so topology controls are compared under one fixed compute budget;
    # latency and reported solve-step distributions retain the adaptive cost.
    original_tolerance = getattr(model, "tolerance", None)
    if hasattr(model, "eval_max_steps") and original_tolerance is not None:
        model.tolerance = 0.0
    model.eval()
    try:
        with torch.inference_mode():
            model(sample)
    finally:
        if original_tolerance is not None:
            model.tolerance = original_tolerance
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total


def profile_model(
    model: nn.Module,
    sample: Tensor,
    *,
    warmup_runs: int = 5,
    measured_runs: int = 20,
) -> dict[str, float | int | str]:
    counts = parameter_counts(model)
    macs = count_macs(model, sample)
    device = sample.device
    was_training = model.training
    model.eval()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(max(warmup_runs, 0)):
            model(sample)
        synchronize()
        start = time.perf_counter()
        for _ in range(max(measured_runs, 1)):
            model(sample)
        synchronize()
        elapsed = time.perf_counter() - start
    model.train(was_training)

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return {
        **counts,
        "macs": macs,
        "flops_approx": 2 * macs,
        "latency_ms": 1000.0 * elapsed / max(measured_runs, 1),
        "peak_memory_bytes": peak_memory,
        "device": str(device),
        "batch_size": int(sample.shape[0]),
        "mac_scope": "Conv2d+Linear",
    }
