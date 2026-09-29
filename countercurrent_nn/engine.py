"""Training, evaluation, and diagnostic collection."""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .utils import top1_correct
from .models.decomposed_boundary import DecomposedTargetBoundary
from .models.exchange import ChannelwiseConductance


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "sgd")).lower()
    lr = float(config.get("lr", 0.1))
    weight_decay = float(config.get("weight_decay", 5e-4))
    all_parameters = list(model.parameters())
    special_groups: list[dict[str, Any]] = []
    special_ids: set[int] = set()
    if "conductance_weight_decay" in config:
        conductance_decay = float(config["conductance_weight_decay"])
        if not math.isfinite(conductance_decay) or conductance_decay < 0:
            raise ValueError("conductance_weight_decay must be finite and nonnegative")
        conductance = [module.logit_gamma for module in model.modules()
                       if isinstance(module, ChannelwiseConductance)]
        if not conductance:
            raise ValueError("conductance_weight_decay requires conductance parameters")
        special_ids.update(id(parameter) for parameter in conductance)
        special_groups.append(
            {"params": conductance, "weight_decay": conductance_decay}
        )
    if "boundary_scalar_weight_decay" in config:
        boundary_decay = float(config["boundary_scalar_weight_decay"])
        if not math.isfinite(boundary_decay) or boundary_decay < 0:
            raise ValueError(
                "boundary_scalar_weight_decay must be finite and nonnegative"
            )
        boundary_scalars = [
            parameter
            for module in model.modules()
            if isinstance(module, DecomposedTargetBoundary)
            for parameter in (
                module.base_scale,
                module.class_scale,
                module.instance_scale,
                module.log_global_gain,
            )
        ]
        if not boundary_scalars:
            raise ValueError(
                "boundary_scalar_weight_decay requires decomposed boundary scalars"
            )
        boundary_ids = {id(parameter) for parameter in boundary_scalars}
        if special_ids.intersection(boundary_ids):
            raise RuntimeError("optimizer special parameter groups overlap")
        special_ids.update(boundary_ids)
        special_groups.append(
            {"params": boundary_scalars, "weight_decay": boundary_decay}
        )
    if special_groups:
        regular = [
            parameter for parameter in all_parameters if id(parameter) not in special_ids
        ]
        parameters: Any = [
            {"params": regular, "weight_decay": weight_decay},
            *special_groups,
        ]
    else:
        parameters = all_parameters
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"optimizer must be 'sgd' or 'adamw', got {name!r}")


def cosine_warmup_lr(
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    amp: bool = False,
    initial_loss_weight: float = 0.0,
    limit_batches: int | None = None,
) -> tuple[dict[str, float], int]:
    if not math.isfinite(initial_loss_weight) or initial_loss_weight < 0:
        raise ValueError("initial_loss_weight must be finite and nonnegative")
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    if initial_loss_weight and "return_initial_logits" not in inspect.signature(unwrapped.forward).parameters:
        raise ValueError("initial supervision requires a model returning initial logits")
    model.train()
    amp_enabled = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_loss = 0.0
    total_final_loss = 0.0
    total_initial_loss = 0.0
    total_initial_correct = 0
    total_correct = 0
    total_examples = 0
    start = time.perf_counter()
    last_lr = base_lr

    for batch_index, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        last_lr = cosine_warmup_lr(base_lr, global_step, total_steps, warmup_steps)
        _set_lr(optimizer, last_lr)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            if initial_loss_weight:
                logits, initial_logits = model(images, return_initial_logits=True)
                initial_loss = criterion(initial_logits, targets)
            else:
                logits = model(images)
            final_loss = criterion(logits, targets)
            loss = final_loss + initial_loss_weight * initial_loss if initial_loss_weight else final_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_final_loss += float(final_loss.detach().item()) * batch_size
        if initial_loss_weight:
            total_initial_loss += float(initial_loss.detach().item()) * batch_size
            total_initial_correct += top1_correct(initial_logits.detach(), targets)
        total_correct += top1_correct(logits.detach(), targets)
        total_examples += batch_size
        global_step += 1

    if total_examples == 0:
        raise RuntimeError("training loader yielded no batches")
    elapsed = time.perf_counter() - start
    if isinstance(model, DistributedDataParallel):
        totals = torch.tensor(
            [total_loss, total_correct, total_examples, total_final_loss,
             total_initial_loss, total_initial_correct], dtype=torch.float64, device=device
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        (total_loss, total_correct, total_examples, total_final_loss,
         total_initial_loss, total_initial_correct) = totals.tolist()
        duration = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        elapsed = duration.item()
    metrics = {
        "loss": total_loss / total_examples,
        "final_loss": total_final_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": float(total_examples),
        "seconds": elapsed,
        "examples_per_second": total_examples / max(elapsed, 1e-12),
        "lr": last_lr,
    }
    if initial_loss_weight:
        metrics.update(initial_loss=total_initial_loss / total_examples,
                       initial_accuracy=total_initial_correct / total_examples,
                       initial_loss_weight=initial_loss_weight)
    return metrics, global_step


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    *,
    limit_batches: int | None = None,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    start = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images, **dict(forward_kwargs or {}))
        loss = criterion(logits, targets)
        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += top1_correct(logits, targets)
        total_examples += batch_size
    if total_examples == 0:
        raise RuntimeError("evaluation loader yielded no batches")
    elapsed = time.perf_counter() - start
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": float(total_examples),
        "seconds": elapsed,
        "examples_per_second": total_examples / max(elapsed, 1e-12),
    }


@torch.inference_mode()
def evaluate_refinement(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    limit_batches: int | None = None,
    error_amplification_delta: float = 0.1,
    reverse_off: bool = False,
) -> dict[str, Any]:
    """Evaluate per-iteration predictions and confirmation-bias diagnostics."""
    predictor = getattr(model, "predict_iterations", None)
    if not callable(predictor):
        return {}
    model.eval()
    correct: Tensor | None = None
    confidence: Tensor | None = None
    entropy: Tensor | None = None
    total_examples = 0
    initial_wrong = 0
    corrected = 0
    amplified = 0
    predictor_parameters = inspect.signature(predictor).parameters

    for batch_index, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        kwargs = {"reverse_off": reverse_off} if "reverse_off" in predictor_parameters else {}
        logits = predictor(images, **kwargs)
        probabilities = logits.softmax(dim=-1)
        predictions = logits.argmax(dim=-1)
        batch_correct = predictions.eq(targets.unsqueeze(0)).sum(dim=1).cpu()
        batch_confidence = probabilities.max(dim=-1).values.sum(dim=1).cpu()
        batch_entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1).sum(dim=1).cpu()
        correct = batch_correct if correct is None else correct + batch_correct
        confidence = (
            batch_confidence if confidence is None else confidence + batch_confidence
        )
        entropy = batch_entropy if entropy is None else entropy + batch_entropy

        wrong_mask = predictions[0].ne(targets)
        initial_classes = predictions[0]
        initial_class_probability = probabilities[0].gather(
            1, initial_classes[:, None]
        ).squeeze(1)
        final_initial_class_probability = probabilities[-1].gather(
            1, initial_classes[:, None]
        ).squeeze(1)
        initial_wrong += int(wrong_mask.sum().item())
        corrected += int((wrong_mask & predictions[-1].eq(targets)).sum().item())
        amplified += int(
            (
                wrong_mask
                & (
                    final_initial_class_probability
                    > initial_class_probability + error_amplification_delta
                )
            ).sum().item()
        )
        total_examples += targets.shape[0]

    if total_examples == 0 or correct is None or confidence is None or entropy is None:
        raise RuntimeError("evaluation loader yielded no batches")
    return {
        "iteration_accuracy": (correct.float() / total_examples).tolist(),
        "iteration_confidence": (confidence / total_examples).tolist(),
        "iteration_entropy": (entropy / total_examples).tolist(),
        "initial_wrong_examples": initial_wrong,
        "error_correction_rate": corrected / initial_wrong if initial_wrong else 0.0,
        "error_amplification_rate": amplified / initial_wrong if initial_wrong else 0.0,
        "error_amplification_delta": error_amplification_delta,
    }


@torch.inference_mode()
def collect_diagnostics(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    max_samples: int = 16,
    reverse_off: bool = False,
) -> dict[str, Any]:
    """Collect one small batch; full lattice histories intentionally stay opt-in."""
    if "return_diagnostics" not in inspect.signature(model.forward).parameters:
        return {}
    model.eval()
    images, targets = next(iter(loader))
    images = images[:max_samples].to(device)
    targets = targets[:max_samples]
    parameters = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {"return_diagnostics": True}
    if "reverse_off" in parameters:
        kwargs["reverse_off"] = reverse_off
    result = model(images, **kwargs)
    if not isinstance(result, tuple):
        return {}
    logits, diagnostics = result
    diagnostics = dict(diagnostics)
    diagnostics["targets"] = targets.detach().cpu()
    diagnostics["final_logits"] = logits.detach().cpu()
    return diagnostics


def summarize_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    if not diagnostics:
        return {}
    summary: dict[str, Any] = {}
    for key in (
        "H_norm",
        "C_norm",
        "q_norm",
        "discrepancy",
        "exchange_energy",
        "proposal_q_norm",
        "proposal_discrepancy",
        "proposal_exchange_energy",
        "residual",
        "equation_residual",
        "step_residual",
        "solve_steps_per_sample",
        "converged",
        "gamma",
        "u_h_norm",
        "u_c_norm",
        "cos_u_h_u_c",
        "cos_d_u_h",
        "cos_q_u_h",
        "boundary_component_norm",
        "boundary_component_ratio",
        "boundary_global_gain",
    ):
        value = diagnostics.get(key)
        if isinstance(value, Tensor):
            summary[key] = value.tolist()

    logits = diagnostics.get("iter_logits")
    targets = diagnostics.get("targets")
    if isinstance(logits, Tensor) and isinstance(targets, Tensor):
        probabilities = logits.softmax(dim=-1)
        predictions = probabilities.argmax(dim=-1)
        summary["iteration_accuracy"] = (
            predictions.eq(targets.unsqueeze(0)).float().mean(dim=1).tolist()
        )
        summary["iteration_confidence"] = probabilities.max(dim=-1).values.mean(dim=1).tolist()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        summary["iteration_entropy"] = entropy.mean(dim=1).tolist()
        initial_wrong = predictions[0].ne(targets)
        initial_classes = predictions[0]
        initial_probability = probabilities[0].gather(
            1, initial_classes[:, None]
        ).squeeze(1)
        final_probability = probabilities[-1].gather(
            1, initial_classes[:, None]
        ).squeeze(1)
        wrong_count = int(initial_wrong.sum().item())
        corrected = initial_wrong & predictions[-1].eq(targets)
        amplified = initial_wrong & (final_probability > initial_probability + 0.1)
        summary["initial_accuracy"] = summary["iteration_accuracy"][0]
        summary["final_accuracy"] = summary["iteration_accuracy"][-1]
        summary["initial_wrong_examples"] = wrong_count
        summary["error_correction_rate"] = (
            float(corrected.sum().item()) / wrong_count if wrong_count else 0.0
        )
        summary["error_amplification_rate"] = (
            float(amplified.sum().item()) / wrong_count if wrong_count else 0.0
        )
    return summary


def summarize_v5_mechanisms(diagnostics: dict[str, Any]) -> dict[str, float]:
    """Return compact V5 pilot gates suitable for one JSONL row per epoch."""
    required = (
        "equation_residual",
        "converged",
        "boundary_component_ratio",
        "cos_d_u_h",
    )
    if any(not isinstance(diagnostics.get(key), Tensor) for key in required):
        return {}
    equation = diagnostics["equation_residual"].float()
    converged = diagnostics["converged"].float()
    ratios = diagnostics["boundary_component_ratio"].float()
    cosine = diagnostics["cos_d_u_h"].float()
    if equation.ndim != 2 or equation.shape[0] == 0:
        return {}
    if ratios.ndim != 2 or ratios.shape[1] < 2:
        return {}
    if cosine.ndim != 2 or cosine.shape[0] == 0:
        return {}
    initial_equation = equation[0]
    final_equation = equation[-1]
    final_abs_cosine = cosine[-1].abs()
    return {
        "equation_residual_initial_mean": float(initial_equation.mean()),
        "equation_residual_final_mean": float(final_equation.mean()),
        "equation_residual_final_max": float(final_equation.max()),
        "equation_residual_decreased_fraction": float(
            final_equation.lt(initial_equation).float().mean()
        ),
        "convergence_rate": float(converged.mean()),
        "class_boundary_ratio": float(ratios[:, 1].mean()),
        "cos_d_u_h_abs_final_mean": float(final_abs_cosine.mean()),
        "cos_d_u_h_abs_final_max": float(final_abs_cosine.max()),
    }
