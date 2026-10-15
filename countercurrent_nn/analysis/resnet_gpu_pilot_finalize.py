"""Audit completed Experiment 07 pilots without changing their checkpoints.

Runs fixed-budget paired predictions and optional frozen boundary interventions,
archives small artifacts, and writes the pre-registered Step C decision.  It
never launches training or additional seeds.
"""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from scipy.special import gammaln, logsumexp
from torch import nn

from countercurrent_nn.data import build_dataloaders
from countercurrent_nn.models import build_model
from countercurrent_nn.profiling import count_macs

ARMS = ("plain", "lattice_control", "countercurrent", "cocurrent")


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def mcnemar_log10_p(first_only: int, second_only: int) -> float:
    """Exact symmetric binomial tail, retaining values below float underflow."""
    n = first_only + second_only
    if n == 0:
        return 0.0
    k = np.arange(min(first_only, second_only) + 1)
    log_pmf = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1) - n*np.log(2)
    return min(0.0, float((np.log(2) + logsumexp(log_pmf)) / np.log(10)))


@torch.inference_mode()
def predictions(model, loader, device):
    model.eval()
    kwargs = {"fixed_steps": 8} if "fixed_steps" in inspect.signature(model.forward).parameters else {}
    predicted, targets_all = [], []
    loss_sum = 0.0
    start = time.perf_counter()
    for images, targets in loader:
        logits = model(images.to(device), **kwargs)
        loss_sum += float(nn.functional.cross_entropy(logits, targets.to(device), reduction="sum"))
        predicted.append(logits.argmax(1).cpu().numpy())
        targets_all.append(targets.numpy())
    predicted = np.concatenate(predicted)
    targets = np.concatenate(targets_all)
    return predicted, targets, {
        "accuracy": float((predicted == targets).mean()),
        "loss": loss_sum / len(targets),
        "examples": len(targets),
        "seconds": time.perf_counter() - start,
    }


class FixedBudget(nn.Module):
    def __init__(self, model, steps):
        super().__init__()
        self.model, self.steps = model, steps

    def forward(self, images):
        return self.model(images, fixed_steps=self.steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--interventions", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    summaries, rows, paired_predictions = {}, {}, {}
    audit = {"seed_count": 1, "seed": 0, "fixed_test_steps": 8, "arms": {}}
    common_targets = None
    for arm in ARMS:
        source = args.root / f"experiment07_resnet50_{arm}_seed0"
        if not (source / "summary.json").exists():
            audit["arms"][arm] = {"status": "not completed"}
            continue
        summary = summaries[arm] = read(source / "summary.json")
        rows[arm] = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()]
        target = args.output / arm
        target.mkdir(exist_ok=True)
        for name in ("config.yaml", "profile.json", "summary.json", "metrics.jsonl", "diagnostics.json", "initial_diagnostics.json"):
            if (source / name).exists():
                shutil.copy2(source / name, target / name)
        checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        dataset_config = {**config["dataset"], "root": str(args.data_root), "download": False,
                          "batch_size": 64}
        data = build_dataloaders(dataset_config, seed=0)
        model = build_model(config["model"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        predicted, targets, measured = predictions(model, data.test, device)
        if common_targets is None:
            common_targets = targets
        else:
            assert np.array_equal(targets, common_targets), "paired examples differ"
        paired_predictions[arm] = predicted
        np.savez_compressed(target / "fixed8_test_predictions.npz", predictions=predicted, targets=targets)
        arm_audit = {
            "status": "completed", "epochs": len(rows[arm]),
            "best_epoch": checkpoint["epoch"] + 1,
            "fixed8_test_recheck": measured,
            "mean_train_seconds": float(np.mean([r["train"]["seconds"] for r in rows[arm]])),
            "mean_validation_seconds": float(np.mean([r["validation"]["seconds"] for r in rows[arm]])),
        }
        if hasattr(model, "eval_max_steps"):
            arm_audit["fixed_budget_macs"] = {
                str(step): count_macs(FixedBudget(model, step), torch.zeros(1, 3, 32, 32, device=device))
                for step in (1, 2, 4, 8)
            }
        else:
            arm_audit["fixed_budget_macs"] = {str(step): summary["macs"] for step in (1, 2, 4, 8)}
        if getattr(model, "exchange_enabled", False):
            trajectory = rows[arm]
            first, last = trajectory[0], trajectory[-1]
            diagnostic = read(source / "diagnostics.json")
            q_samples = np.asarray(diagnostic["q_rms_final_per_sample"])
            arm_audit["mechanisms"] = {
                "epoch1_R_class": first["mechanisms"]["class_boundary_ratio"],
                "epoch30_R_class": last["mechanisms"]["class_boundary_ratio"],
                "class_ratio_reduction_factor": first["mechanisms"]["class_boundary_ratio"] / last["mechanisms"]["class_boundary_ratio"],
                "epoch1_effective_boundary_rms": first["diagnostics"]["weighted_boundary_component_rms"][0],
                "epoch30_effective_boundary_rms": last["diagnostics"]["weighted_boundary_component_rms"][0],
                "selected_checkpoint_test_boundary_ratios": diagnostic["boundary_component_ratio"][0],
                "selected_checkpoint_test_boundary_effective_rms": diagnostic["weighted_boundary_component_rms"][0],
                "selected_checkpoint_cos_D_uH_per_layer": diagnostic["cos_d_u_h"][-1],
                "selected_checkpoint_Q_rms_per_layer_mean": q_samples.mean(0).tolist(),
                "selected_checkpoint_Q_rms_per_layer_p10": np.quantile(q_samples, .1, axis=0).tolist(),
                "selected_checkpoint_Q_rms_per_layer_p90": np.quantile(q_samples, .9, axis=0).tolist(),
                "Q_distribution_sample_count": len(q_samples),
                "selected_checkpoint_boundary_scalars": {
                    name: float(getattr(model.target_boundary, name).detach())
                    for name in ("base_scale", "class_scale", "instance_scale", "global_gain")
                },
            }
            with torch.inference_mode():
                images, _ = next(iter(data.validation))
                source_state = model._front(images[:2].to(device))
                state = source_state
                for block in model.F:
                    state = block(state)
                for block in reversed(model.F):
                    state = block.inverse(state)
                arm_audit["trained_roundtrip_relative_error"] = float((state-source_state).norm()/source_state.norm())
            if args.interventions:
                interventions = {}
                for component in ("class", "instance", "base"):
                    def remove(module, inputs, result, name=component):
                        boundary, probabilities = result
                        reference = inputs[1]
                        parts = module.components_from_probabilities(probabilities, reference)
                        return boundary - module.global_gain * parts[f"weighted_{name}"], probabilities
                    handle = model.target_boundary.register_forward_hook(remove)
                    try:
                        _, _, result = predictions(model, data.test, device)
                        result["drop_pp"] = 100 * (measured["accuracy"] - result["accuracy"])
                        interventions[f"{component}_off"] = result
                    finally:
                        handle.remove()
                arm_audit["frozen_interventions"] = interventions
        audit["arms"][arm] = arm_audit
        del model, checkpoint, data
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(arm, measured, flush=True)
    paired = {}
    if "countercurrent" in paired_predictions:
        cc_correct = paired_predictions["countercurrent"] == common_targets
        for other in ("plain", "lattice_control", "cocurrent"):
            if other not in paired_predictions:
                continue
            other_correct = paired_predictions[other] == common_targets
            cc_only = int((cc_correct & ~other_correct).sum())
            other_only = int((~cc_correct & other_correct).sum())
            paired[other] = {
                "cc_only_correct": cc_only, "other_only_correct": other_only,
                "difference_pp": 100 * float(cc_correct.mean() - other_correct.mean()),
                "mcnemar_exact_two_sided_p": float(binomtest(cc_only, cc_only + other_only).pvalue) if cc_only + other_only else 1.0,
                "mcnemar_exact_two_sided_log10_p": mcnemar_log10_p(cc_only, other_only),
                "scope": "paired test examples within seed 0; not a multi-seed significance test",
            }
    audit["paired_tests"] = paired
    if "plain" in paired_predictions and "lattice_control" in paired_predictions:
        plain_correct = paired_predictions["plain"] == common_targets
        control_correct = paired_predictions["lattice_control"] == common_targets
        x = int((plain_correct & ~control_correct).sum())
        y = int((~plain_correct & control_correct).sum())
        audit["baseline_pair"] = {
            "plain_only_correct": x, "control_only_correct": y,
            "mcnemar_exact_two_sided_p": float(binomtest(x, x+y).pvalue) if x+y else 1.0,
            "scope": "test examples within one seed, not training-seed uncertainty",
        }
    if "countercurrent" in summaries and "lattice_control" in summaries:
        cc, control = summaries["countercurrent"], summaries["lattice_control"]
        accuracy_curve = [cc["refinement"]["steps"][str(step)]["accuracy"] for step in (1, 2, 4, 8)]
        gains = 100 * (cc["test"]["accuracy"] - control["test"]["accuracy"])
        reverse_drop = 100 * cc["reverse_accuracy_drop"]
        gates = {
            "cc_minus_control_at_least_0_5_pp": gains >= 0.5 - 1e-10,
            "reverse_off_drop_at_least_1_pp": reverse_drop >= 1.0 - 1e-10,
            "fixed_test_curve_monotonic": all(b >= a for a, b in zip(accuracy_curve, accuracy_curve[1:])),
        }
        audit["decision"] = {
            "gates": gates, "cc_minus_control_pp": gains,
            "reverse_off_drop_pp": reverse_drop,
            "test_iteration_accuracy": accuracy_curve,
            "step_d_permitted": all(gates.values()),
            "verdict": "GO for Step D only" if all(gates.values()) else "NO-GO: stop before Co or extra seeds",
        }
    write(args.output / "audit.json", audit)
    plot(args.output, rows, summaries)
    print(json.dumps(audit.get("decision", {}), indent=2), flush=True)


def plot(output, rows, summaries):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, history in rows.items():
        axes[0].plot([r["epoch"]+1 for r in history], [100*r["validation"]["accuracy"] for r in history], label=arm)
        summary = summaries[arm]
        curve = summary.get("refinement", {}).get("steps")
        # Older summaries stored the 2k validation curve in refinement.  Never
        # draw that curve on the full-test panel; a single-pass control has the
        # same full-test accuracy at every requested solver budget.
        if summary.get("refinement", {}).get("examples") != summary["test"]["examples"]:
            curve = None
        values = [100*curve[str(s)]["accuracy"] for s in (1,2,4,8)] if curve else [100*summary["test"]["accuracy"]]*4
        axes[1].plot([1,2,4,8], values, marker="o", label=arm)
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Fixed 2,000-example validation subset")
    axes[1].set(xlabel="Solver budget", ylabel="Test accuracy (%)", title="Full 10,000-example test set")
    axes[1].set_xticks([1,2,4,8])
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "accuracy_curves.png", dpi=180); plt.close(fig)
    if "countercurrent" not in rows:
        return
    history = rows["countercurrent"]
    epochs = [r["epoch"]+1 for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, name in enumerate(("base", "class", "instance")):
        values = [r["diagnostics"]["weighted_boundary_component_rms"][0][i] for r in history]
        axes[0].semilogy(epochs, values, label=name)
    axes[1].semilogy(epochs, [r["mechanisms"]["class_boundary_ratio"] for r in history])
    for layer in range(5):
        axes[2].plot(epochs, [r["diagnostics"]["cos_d_u_h"][-1][layer] for r in history], label=f"cell {layer}")
    axes[0].set(title="Effective boundary RMS", xlabel="Epoch", ylabel="RMS including scale and gain")
    axes[1].set(title="Class boundary ratio", xlabel="Epoch", ylabel="R_class")
    axes[2].set(title="Final-sweep discrepancy direction", xlabel="Epoch", ylabel="cos(D, u_H)")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[0].legend(); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "mechanism_trajectories.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
