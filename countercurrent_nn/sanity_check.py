"""One-batch architecture validation, without launching an epoch-training run.

Default input is explicitly synthetic, normalized to CIFAR-10 statistics.
Use --data-root to read one real CIFAR-10 training batch already on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from countercurrent_nn.config import load_config
from countercurrent_nn.data import CIFAR_STATS, SyntheticCIFAR
from countercurrent_nn.engine import summarize_diagnostics
from countercurrent_nn.models import CocurrentCNN, CountercurrentCNN, build_model
from countercurrent_nn.profiling import profile_model


def stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float()
    return {
        "min": values.min().item(), "mean": values.mean().item(),
        "max": values.max().item(), "rms": values.square().mean().sqrt().item(),
        "l2_per_element": values.norm().item() / values.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/one_batch_sanity.json"))
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.data_root is None:
        dataset = SyntheticCIFAR(args.batch_size, 10, args.seed)
        x, labels = dataset.images, dataset.targets
        mean, std = CIFAR_STATS["cifar10"]
        x = (x - torch.tensor(mean)[None, :, None, None]) / torch.tensor(std)[None, :, None, None]
        data_description = "synthetic uniform RGB with random labels, CIFAR-10 normalization; not CIFAR accuracy"
    else:
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import Compose, Normalize, ToTensor
        dataset = CIFAR10(
            str(args.data_root), train=True, download=False,
            transform=Compose([ToTensor(), Normalize(*CIFAR_STATS["cifar10"])]),
        )
        batch = [dataset[index] for index in range(args.batch_size)]
        x = torch.stack([sample for sample, _ in batch])
        labels = torch.tensor([target for _, target in batch])
        data_description = "first CIFAR-10 training batch, normalization only, no download"
    x, labels = x.to(device), labels.to(device)
    model = CountercurrentCNN().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    stem_sources = []
    handle = model.stem.register_forward_hook(
        lambda _m, inputs, output: stem_sources.append((inputs[0].detach(), output.detach()))
    )
    try:
        logits, diagnostics = model(x, return_diagnostics=True)
    finally:
        handle.remove()
    assert len(stem_sources) == model.refine_steps + 1
    for iteration, (stem_input, source) in enumerate(stem_sources):
        torch.testing.assert_close(stem_input, x)
        torch.testing.assert_close(source.cpu(), diagnostics["H_history"][iteration, 0])
        torch.testing.assert_close(source, stem_sources[0][1])
    for value in diagnostics.values():
        assert torch.isfinite(value).all()
    gamma = diagnostics["gamma"]
    assert (gamma > 0).all() and (gamma < 0.5).all()
    for key in ("Q", "proposal_Q"):
        assert (diagnostics[key].flatten(2).norm(dim=-1) > 0).all()

    # Labels enter only the final task loss and external diagnostic calculations.
    loss = F.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
        group = name.split(".")[0]
        gradients[group] = gradients.get(group, 0.0) + parameter.grad.norm().item()
    model.eval()
    with torch.no_grad():
        off, off_diag = model(x, reverse_off=True, return_diagnostics=True)
    torch.testing.assert_close(off.cpu(), diagnostics["initial_logits"])
    for key in ("Q", "proposal_Q"):
        assert torch.count_nonzero(off_diag[key]) == 0
    off_accuracy = (off.argmax(-1) == labels).float().mean().item()
    diagnostics["targets"] = labels.cpu()
    diagnostics["final_logits"] = logits.detach().cpu()

    layer_stats = []
    for iteration in range(model.refine_steps):
        for layer in range(model.depth):
            layer_stats.append({
                "iteration": iteration + 1, "layer": layer,
                "gamma": stats(gamma[layer]),
                **{key: stats(diagnostics[key][iteration, layer])
                   for key in ("proposal_D", "proposal_Q", "D", "Q")},
                "H": stats(diagnostics["H_history"][iteration + 1, layer + 1]),
                "C": stats(diagnostics["C_history"][iteration, layer]),
            })

    # Exactly one optimizer update on this same batch; no data/epoch loop.
    optimizer.step()
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
    with torch.no_grad():
        after_logits, after_diag = model(x, return_diagnostics=True)
        loss_after = F.cross_entropy(after_logits, labels).item()
    for value in after_diag.values():
        assert torch.isfinite(value).all()
    assert (after_diag["gamma"] > 0).all() and (after_diag["gamma"] < 0.5).all()

    profiles = {}
    config_root = Path(__file__).parent / "configs"
    for name in (
        "single", "single_parammatched", "forward_recurrent", "feedback_fusion",
        "cocurrent_selfboundary", "countercurrent_selfboundary",
        "countercurrent_null", "countercurrent_learnedz",
    ):
        candidate = build_model(load_config(config_root / f"cifar10_{name}.yaml")["model"]).to(device)
        profiles[name] = profile_model(candidate, x[:1], warmup_runs=1, measured_runs=3)
    cc, co = profiles["countercurrent_selfboundary"], profiles["cocurrent_selfboundary"]
    assert cc["params"] == co["params"] and cc["macs"] == co["macs"]
    # Check that matched state dicts are usable across the direction control.
    CocurrentCNN().load_state_dict({key: value.cpu() for key, value in model.state_dict().items()})

    report = {
        "seed": args.seed, "torch": torch.__version__, "device": str(device),
        "threads": args.threads, "data": data_description,
        "input_shape": list(x.shape), "logits_shape": list(logits.shape),
        "loss_before": loss.item(), "loss_after_one_adamw_step": loss_after,
        "optimizer_steps": 1, "epoch_training_started": False,
        "stem_calls": len(stem_sources), "source_clamp_verified": True,
        "all_states_and_gradients_finite": True,
        "parameter_group_gradient_norm_sums": gradients,
        "gamma_initial": stats(gamma), "gamma_after_step": stats(after_diag["gamma"]),
        "diagnostic_shapes": {key: list(value.shape) for key, value in diagnostics.items()},
        "summary": summarize_diagnostics(diagnostics),
        "reverse_off_accuracy": off_accuracy,
        "reverse_off_max_logit_difference": (logits.detach() - off).abs().max().item(),
        "reverse_off_matches_initial_prediction": True,
        "reverse_off_zero_flux_both_stages": True,
        "layer_initial_statistics": layer_stats,
        "profiles_batch_one": profiles,
        "flops_convention": "2 * Conv2d/Linear MACs; excludes GN, SiLU, GAP, softmax, prototype matmul and elementwise exchange",
        "norm_convention": "L2 / number of elements, plus RMS; norms are batch-size dependent",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    torch.save(diagnostics, args.output.with_suffix(".pt"))
    print(json.dumps({key: report[key] for key in (
        "data", "input_shape", "logits_shape", "loss_before",
        "loss_after_one_adamw_step", "stem_calls", "gamma_initial",
        "reverse_off_max_logit_difference",
    )}, indent=2))
    print(f"Report: {args.output}; full initial tensors: {args.output.with_suffix('.pt')}")


if __name__ == "__main__":
    main()
