"""Read completed CIFAR runs, plot comparisons, and probe frozen checkpoints.

This command never trains or writes to the source run directories. Optional
boundary interventions use a fixed validation subset, not the test set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from countercurrent_nn.data import build_dataloaders
from countercurrent_nn.engine import cosine_warmup_lr
from countercurrent_nn.models import build_model


RUNS = {
    "CNN": "cifar10_single_parammatched_4gpu_seed0",
    "Co-current": "cifar10_cocurrent_v2_4gpu_seed0",
    "Countercurrent": "cifar10_countercurrent_v2_4gpu_seed0",
}


def summarize_run(root: Path) -> tuple[dict, list, dict, dict]:
    summary = json.loads((root / "summary.json").read_text())
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
    checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    diag = torch.load(root / "diagnostics.pt", map_location="cpu", weights_only=True)
    assert [row["epoch"] for row in rows] == list(range(200))
    assert all(row["train"]["examples"] == 45000 and row["validation"]["examples"] == 5000 for row in rows)
    result = {
        "source": str(root), "summary": summary,
        "best_epoch": checkpoint["epoch"] + 1,
        "final_train": rows[-1]["train"], "final_validation": rows[-1]["validation"],
        "last_ten_validation_accuracy_mean": sum(r["validation"]["accuracy"] for r in rows[-10:]) / 10,
        "diagnostic_samples": len(diag["targets"]),
        "all_logged_losses_finite": all(math.isfinite(r[k]["loss"]) for r in rows for k in ("train", "validation")),
        "all_diagnostic_tensors_finite": all(torch.isfinite(v).all().item() for v in diag.values() if isinstance(v, torch.Tensor)),
    }
    if summary["refinement"]:
        ref = summary["refinement"]
        n = int(summary["test"]["examples"])
        wrong = ref["initial_wrong_examples"]
        corrected = round(wrong * ref["error_correction_rate"])
        amplified = round(wrong * ref["error_amplification_rate"])
        final_correct = round(n * summary["test"]["accuracy"])
        harmed = n - wrong + corrected - final_correct
        result["transitions_on_full_test"] = {
            "initial_wrong": wrong, "wrong_to_correct": corrected,
            "correct_to_wrong": harmed, "net_extra_correct": corrected - harmed,
            "correct_to_wrong_rate": harmed / (n - wrong),
            "initial_wrong_class_amplified_delta_0_1": amplified,
        }
        result["gamma"] = [
            {"min": g.min().item(), "mean": g.mean().item(), "max": g.max().item()}
            for g in diag["gamma"]
        ]
        for key in ("Q", "proposal_Q"):
            squared = diag[key].float().square().flatten(2).sum(-1)
            result[key + "_squared_norm_share"] = (squared / squared.sum(1, keepdim=True)).tolist()
        result["Q_to_Hbar_norm_ratio"] = (
            diag["Q"].flatten(2).norm(dim=-1) / diag["resweep_H_bar"].flatten(2).norm(dim=-1)
        ).tolist()
        result["H_rms"] = diag["H_history"].float().square().flatten(2).mean(-1).sqrt().tolist()
        result["C_rms"] = diag["C_history"].float().square().flatten(2).mean(-1).sqrt().tolist()
        # Isolate the effect of regularization on the existing logit parameterization.
        training = checkpoint["config"]["training"]
        steps_per_epoch = math.ceil(45000 / training["batch_size"])
        a, velocity = -2.2, 0.
        for step in range(checkpoint["global_step"]):
            velocity = training["momentum"] * velocity + training["weight_decay"] * a
            a -= cosine_warmup_lr(training["lr"], step, training["epochs"] * steps_per_epoch,
                                  training["warmup_epochs"] * steps_per_epoch) * velocity
        result["zero_task_gradient_decay_counterfactual"] = {"logit_gamma": a, "gamma": .49 / (1 + math.exp(-a))}
    return result, rows, checkpoint, diag


def prediction_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    p = logits.softmax(-1)
    pred = logits.argmax(-1)
    correct = pred.eq(targets[None])
    confidence = p.amax(-1)
    wrong = ~correct[0]
    initial_class_final_conf = p[-1].gather(1, pred[0, :, None]).squeeze(1)
    amplified = wrong & (initial_class_final_conf > confidence[0] + .1)
    eces = []
    for t in range(len(logits)):
        ece = 0.
        for i in range(15):
            mask = (confidence[t] > i / 15) & (confidence[t] <= (i + 1) / 15)
            if mask.any():
                ece += (confidence[t][mask].mean() - correct[t][mask].float().mean()).abs() * mask.float().mean()
        eces.append(float(ece))
    return {
        "examples": len(targets),
        "accuracy": correct.float().mean(1).tolist(),
        "cross_entropy": [F.cross_entropy(s, targets).item() for s in logits],
        "confidence": confidence.mean(1).tolist(),
        "entropy": -(p * p.clamp_min(1e-12).log()).sum(-1).mean(1),
        "ece_15_bins": eces,
        "wrong_to_correct": (wrong & correct[-1]).sum().item(),
        "correct_to_wrong": (correct[0] & ~correct[-1]).sum().item(),
        "amplified_initial_wrong_class": amplified.sum().item(),
        "amplified_and_still_wrong": (amplified & ~correct[-1]).sum().item(),
    }


@torch.inference_mode()
def probe(checkpoint: dict, loader: DataLoader, modes: list[str]) -> tuple[dict, dict]:
    model = build_model(checkpoint["config"]["model"]).eval()
    model.load_state_dict(checkpoint["model_state"])
    values, tensors = {}, {}
    for mode in modes:
        batches, labels = [], []
        for x, y in loader:
            if mode == "oracle_labels_analysis_only":
                boundary = model.target_boundary
                channels = boundary.projector(boundary.class_prototypes[y])
                _, steps, _ = model._run(x, capture_diagnostics=False, reverse_off=False, oracle_boundary=channels)
                logits = torch.stack(steps)
            elif mode == "self":
                logits = model.predict_iterations(x) if hasattr(model, "predict_iterations") else model(x)[None]
            else:
                boundary = model.target_boundary
                original = boundary.forward

                def intervention(s, reference):
                    if mode == "zero_boundary":
                        return torch.zeros_like(reference), s.softmax(-1)
                    if mode == "uniform_hypothesis":
                        return original(torch.zeros_like(s), reference)
                    if mode == "shuffled_hypothesis":
                        return original(s.roll(1, dims=0), reference)
                    if mode == "temperature_2":
                        return original(s / 2, reference)
                    raise ValueError(mode)

                with patch.object(boundary, "forward", side_effect=intervention):
                    logits = model.predict_iterations(x)
            batches.append(logits)
            labels.append(y)
        logits, targets = torch.cat(batches, dim=1), torch.cat(labels)
        values[mode] = prediction_metrics(logits, targets)
        values[mode]["entropy"] = values[mode]["entropy"].tolist()
        tensors[mode] = logits
        tensors["targets"] = targets
        print(f'{checkpoint["config"]["model"]["name"]} {mode}: accuracy={values[mode]["accuracy"][-1]:.4f}', flush=True)
    return values, tensors


def plot(results: dict, curves: dict, destination: Path) -> None:
    colors = {"CNN": "#555555", "Co-current": "#0072B2", "Countercurrent": "#D55E00"}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    for name, result in results.items():
        rows, color = curves[name], colors[name]
        epoch = [r["epoch"] + 1 for r in rows]
        axs[0, 0].plot(epoch, [100 * r["validation"]["accuracy"] for r in rows], label=name, color=color, lw=1.2)
        axs[0, 1].plot(epoch, [r["train"]["loss"] for r in rows], color=color, label=name, lw=1.2)
        axs[0, 1].plot(epoch, [r["validation"]["loss"] for r in rows], color=color, ls="--", lw=1)
        summary = result["summary"]
        ref = summary["refinement"]
        if ref:
            axs[0, 2].plot(range(4), [100 * v for v in ref["iteration_accuracy"]], marker="o", color=color, label=name)
            axs[1, 0].plot(range(4), ref["iteration_confidence"], marker="o", color=color, label=name + " confidence")
            axs[1, 0].plot(range(4), ref["iteration_accuracy"], ls="--", color=color, label=name + " accuracy")
            axs[1, 1].plot(range(4), [v["mean"] for v in result["gamma"]], marker="o", color=color, label=name)
            axs[1, 2].plot(range(4), [100 * v for v in result["Q_squared_norm_share"][-1]], marker="o", color=color, label=name)
    axs[0, 0].set(title="Validation accuracy (all 200 epochs)", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(30, 91))
    axs[0, 1].set(title="Loss: train solid / validation dashed", xlabel="Epoch", ylabel="Cross-entropy", ylim=(0, 1))
    axs[0, 2].set(title="Full test set: iteration accuracy", xlabel="Refinement iteration", ylabel="Accuracy (%)", xticks=range(4))
    axs[1, 0].set(title="Full test set: confidence vs accuracy", xlabel="Refinement iteration", ylabel="Fraction", xticks=range(4))
    axs[1, 1].axhline(.49 / (1 + math.exp(2.2)), ls=":", color="black", label="Initialization")
    axs[1, 1].axhline(.245, ls="--", color="gray", label="Weight-decay attractor")
    axs[1, 1].set(title="Conductance by layer", xlabel="Cell index", ylabel="Mean gamma", xticks=range(4), ylim=(0, .27))
    axs[1, 2].set(title="Final resweep Q squared norm (16 samples)", xlabel="Cell index", ylabel="Share across layers (%)", xticks=range(4))
    for ax in axs.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("CIFAR-10: completed seed-0 experiments; single-seed evidence", fontsize=14)
    fig.savefig(destination / "comparison.png", dpi=180)
    fig.savefig(destination / "comparison.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("countercurrent_nn/results"))
    parser.add_argument("--output", type=Path, default=Path("reports/cifar10_threeway_20260916"))
    parser.add_argument("--probe-samples", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    results, curves, checkpoints = {}, {}, {}
    for name, directory in RUNS.items():
        results[name], curves[name], checkpoints[name], _ = summarize_run(args.root / directory)
    reference = checkpoints["Countercurrent"]["config"]
    assert all(c["config"]["dataset"] == reference["dataset"] and c["config"]["training"] == reference["training"] for c in checkpoints.values())
    co_model = dict(checkpoints["Co-current"]["config"]["model"])
    cc_model = dict(reference["model"])
    co_model.pop("name"); cc_model.pop("name")
    assert co_model == cc_model
    plot(results, curves, args.output)
    report = {"runs": results, "protocol_matches": True}
    if args.probe_samples:
        config = dict(reference["dataset"], num_workers=0, batch_size=32, pin_memory=False, download=False)
        data = build_dataloaders(config, seed=0)
        assert 0 < args.probe_samples <= len(data.validation.dataset)
        loader = DataLoader(Subset(data.validation.dataset, range(args.probe_samples)), batch_size=32, shuffle=False)
        report["validation_probe"] = {"description": "First N samples of the fixed 5k validation split; frozen best checkpoints; inference interventions, no retraining.", "samples": args.probe_samples, "results": {}}
        tensors = {}
        for name, checkpoint in checkpoints.items():
            modes = ["self"] if name == "CNN" else ["self", "uniform_hypothesis", "zero_boundary", "shuffled_hypothesis", "temperature_2", "oracle_labels_analysis_only"]
            result, tensors[name] = probe(checkpoint, loader, modes)
            report["validation_probe"]["results"][name] = result
        torch.save(tensors, args.output / "validation_probe_predictions.pt")
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved analysis to {args.output}", flush=True)


if __name__ == "__main__":
    main()
