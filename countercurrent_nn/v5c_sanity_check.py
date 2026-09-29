"""One real/synthetic batch V5-C forward, backward, update, and solver check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from countercurrent_nn.config import load_config
from countercurrent_nn.data import CIFAR_STATS, SyntheticCIFAR
from countercurrent_nn.engine import build_optimizer, summarize_diagnostics
from countercurrent_nn.models import build_model
from countercurrent_nn.profiling import profile_model
from countercurrent_nn.utils import parameter_counts


def _batch(batch_size: int, seed: int, data_root: Path | None):
    if data_root is None:
        dataset = SyntheticCIFAR(batch_size, 10, seed)
        images, labels = dataset.images, dataset.targets
        mean, std = CIFAR_STATS["cifar10"]
        images = (images - torch.tensor(mean)[None, :, None, None]) / torch.tensor(
            std
        )[None, :, None, None]
        return images, labels, "synthetic CIFAR-shaped batch"
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, Normalize, ToTensor

    dataset = CIFAR10(
        str(data_root),
        train=True,
        download=False,
        transform=Compose([ToTensor(), Normalize(*CIFAR_STATS["cifar10"])]),
    )
    samples = [dataset[index] for index in range(batch_size)]
    return (
        torch.stack([image for image, _ in samples]),
        torch.tensor([label for _, label in samples]),
        "first real CIFAR-10 training samples; normalization only",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("countercurrent_nn/configs/cifar10_v5c_countercurrent.yaml"),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v5c_one_batch_sanity.json")
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    images, labels, data_description = _batch(
        args.batch_size, args.seed, args.data_root
    )
    images, labels = images.to(device), labels.to(device)
    config = load_config(args.config)
    model = build_model(config["model"]).to(device).train()
    optimizer = build_optimizer(
        model,
        {
            "optimizer": "adamw",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "conductance_weight_decay": 0.0,
            "boundary_scalar_weight_decay": 0.0,
        },
    )

    boundary_calls = 0

    def count_boundary(*_args):
        nonlocal boundary_calls
        boundary_calls += 1

    hook = model.target_boundary.register_forward_hook(count_boundary)
    try:
        logits, train_diagnostics = model(images, return_diagnostics=True)
    finally:
        hook.remove()
    if boundary_calls != model.num_outer_train:
        raise AssertionError(f"boundary called {boundary_calls} times during training")
    if train_diagnostics["H_history"].shape[0] != model.train_inner_steps + 1:
        raise AssertionError("training did not execute the fixed inner-step budget")
    source = model.stem(images).detach().cpu()
    for state in train_diagnostics["H_history"][:, 0]:
        torch.testing.assert_close(state, source)
    boundary = train_diagnostics["boundary_history"][0]
    for state in train_diagnostics["C_history"][:, model.boundary_index]:
        torch.testing.assert_close(state, boundary)

    loss_before = F.cross_entropy(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss_before.backward()
    missing_gradients = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            missing_gradients.append(name)
    if missing_gradients:
        raise AssertionError(f"missing/nonfinite gradients: {missing_gradients}")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise AssertionError("optimizer update produced a nonfinite parameter")

    model.eval()
    boundary_calls = 0
    hook = model.target_boundary.register_forward_hook(count_boundary)
    try:
        with torch.inference_mode():
            final_logits, diagnostics = model(images, return_diagnostics=True)
            reverse_off = model(images, reverse_off=True)
    finally:
        hook.remove()
    if boundary_calls != 2 * model.num_outer_eval:
        raise AssertionError("each eval solve must construct one boundary per outer step")
    if not all(torch.isfinite(value).all() for value in diagnostics.values()):
        raise AssertionError("nonfinite V5-C diagnostic")
    if not bool((diagnostics["gamma"] > 0).all() and (diagnostics["gamma"] < 0.5).all()):
        raise AssertionError("conductance left its valid interval")

    diagnostics["targets"] = labels.detach().cpu()
    diagnostics["final_logits"] = final_logits.detach().cpu()
    report = {
        "config": str(args.config),
        "seed": args.seed,
        "torch": torch.__version__,
        "device": str(device),
        "data": data_description,
        "input_shape": list(images.shape),
        "logits_shape": list(final_logits.shape),
        "loss_before_update": float(loss_before.detach()),
        "loss_after_update": float(F.cross_entropy(final_logits, labels)),
        "optimizer_steps": 1,
        "epoch_training_started": False,
        "train_fixed_steps": model.train_inner_steps,
        "eval_max_steps": model.eval_max_steps,
        "eval_steps_per_sample": diagnostics["solve_steps_per_sample"].tolist(),
        "eval_converged": diagnostics["converged"].tolist(),
        "source_clamp_verified": True,
        "target_clamp_verified": True,
        "boundary_calls_per_outer_verified": True,
        "all_states_and_gradients_finite": True,
        "reverse_off_max_logit_difference": float(
            (final_logits - reverse_off).abs().max()
        ),
        "parameters": parameter_counts(model),
        "profile_batch_one": profile_model(
            model, images[:1], warmup_runs=0, measured_runs=1
        ),
        "summary": summarize_diagnostics(diagnostics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    torch.save(diagnostics, args.output.with_suffix(".pt"))
    print(json.dumps({key: report[key] for key in (
        "data",
        "input_shape",
        "loss_before_update",
        "loss_after_update",
        "train_fixed_steps",
        "eval_steps_per_sample",
        "eval_converged",
        "reverse_off_max_logit_difference",
    )}, indent=2))


if __name__ == "__main__":
    main()
