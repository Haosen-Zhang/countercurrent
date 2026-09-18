"""Measure the representation conditions R1-R4 for a coupled counterflow model.

This script exists because the V2-V4 experiments trained for many GPU hours
without ever checking whether the reverse field could influence the prediction
at all.  R1-R4 are the necessary representation conditions for counterflow
exchange to carry information rather than act as a uniform rescaling of the
forward activations:

    R1  C lives in the same representable space as H   (rms ratio in [0.8, 1.25])
    R2  D is not explained by H alone                  (cos(D, H_bar) < 0.5)
    R3  the target boundary carries instance content   (||dlogits||/||logits|| > 0.05)
    R4  the iteration approaches a fixed point         (residual decreases)

Everything runs on random initialisation, CPU, in well under a minute, so it can
be used as a regression gate before spending epochs on a training run.

Usage:
    python -m countercurrent_nn.analysis.diagnose_counterflow \
        --config countercurrent_nn/configs/cifar10_countercurrent_v4_sequential.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from countercurrent_nn.config import load_config
from countercurrent_nn.models import build_model

# Thresholds from COUNTERCURRENT_BVP_COUNTERFLOW_PLAN.md section 2.3.
RATIO_LOW, RATIO_HIGH = 0.8, 1.25
R2_MAX = 0.5
R3_MIN = 0.05


def rms(tensor: Tensor) -> float:
    return tensor.square().mean().sqrt().item()


def batch_cosine(left: Tensor, right: Tensor) -> float:
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    return F.cosine_similarity(left, right, dim=1).mean().item()


class BoundaryCapture:
    """Temporarily replace a model's boundary generator with a controlled one."""

    def __init__(self, model: nn.Module) -> None:
        self.module = getattr(model, "target_boundary", None)
        if not isinstance(self.module, nn.Module):
            raise TypeError("model does not expose a 'target_boundary' module")
        self._original = self.module.forward
        self._call = 0

    def install(self) -> None:
        self.module.forward = self._replacement  # type: ignore[method-assign]

    def restore(self) -> None:
        self.module.forward = self._original  # type: ignore[method-assign]

    def _replacement(self, logits: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class RandomHypothesisBoundary(BoundaryCapture):
    """Same construction, but the class hypothesis is drawn at random."""

    def __init__(self, model: nn.Module, generator: torch.Generator) -> None:
        super().__init__(model)
        self.generator = generator

    def _replacement(self, logits: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        random_logits = torch.randn(
            logits.shape, generator=self.generator, dtype=logits.dtype
        )
        return self._original(random_logits, reference)


class UniformHypothesisBoundary(BoundaryCapture):
    """Uniform class probabilities; keeps the trained prototypes and projector."""

    def _replacement(self, logits: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        return self._original(torch.zeros_like(logits), reference)


def _coupled_diagnostics(model: nn.Module, x: Tensor) -> dict[str, Any]:
    _, diagnostics = model(x, return_diagnostics=True)  # type: ignore[call-arg]
    return diagnostics


def measure_R1_R2(diagnostics: dict[str, Any]) -> dict[str, Any]:
    h_history = diagnostics["H_history"]  # [T+1, L+1, B, C, h, w]
    c_history = diagnostics["C_history"]  # [T,   L+1, B, C, h, w]
    discrepancy = diagnostics["D"]  # proposal-stage D, [T, L, B, C, h, w]

    final_h = h_history[-1]
    final_c = c_history[-1]
    rows = []
    for depth in range(final_h.shape[0]):
        h_l = final_h[depth]
        c_l = final_c[depth]
        rows.append(
            {
                "depth": depth,
                "rms_H": rms(h_l),
                "rms_C": rms(c_l),
                "ratio_C_over_H": rms(c_l) / max(rms(h_l), 1e-12),
                "cos_H_C": batch_cosine(h_l, c_l),
            }
        )

    r2_rows = []
    if discrepancy.ndim == 6:
        stage = discrepancy[-1]
        for depth in range(stage.shape[0]):
            h_bar = final_h[depth + 1]
            d_l = stage[depth]
            r2_rows.append(
                {
                    "depth": depth,
                    "cos_D_Hbar": batch_cosine(d_l, h_bar),
                }
            )
    return {"R1": rows, "R2": r2_rows}


def measure_R3(model: nn.Module, x: Tensor, seed: int) -> dict[str, Any]:
    """Relative logit change when the class hypothesis is replaced."""
    with torch.no_grad():
        baseline = model(x)  # type: ignore[operator]
        denominator = baseline.norm().item()
        results: dict[str, float] = {}
        for name, capture in (
            ("uniform_hypothesis", UniformHypothesisBoundary),
            ("random_hypothesis", RandomHypothesisBoundary),
        ):
            generator = torch.Generator().manual_seed(seed)
            if name == "uniform_hypothesis":
                probe = capture(model)  # type: ignore[call-arg]
            else:
                probe = capture(model, generator)  # type: ignore[call-arg]
            probe.install()
            try:
                perturbed = model(x)  # type: ignore[operator]
            finally:
                probe.restore()
            results[name] = (perturbed - baseline).norm().item() / max(denominator, 1e-12)
    return results


def measure_R4(model: nn.Module, x: Tensor, extra_iterations: int) -> dict[str, Any]:
    """Record how the model's own state evolves when it keeps iterating.

    Two schedules are possible.  A BVP solver exposes run_solver and reports its
    own residual history, which is the honest measurement: only the clamped source
    boundary is re-injected, and the interior state persists across rounds.  The
    older proposal/resweep models expose _refine, which rebuilds the whole state
    from stem(x) and the previous logits every round; for those a residual reaching
    zero after one round means the cycle is a fixed point of the schedule rather
    than an iterative solver.
    """
    run_solver = getattr(model, "run_solver", None)
    if callable(run_solver):
        with torch.no_grad():
            _, history = run_solver(x)
        return {
            "supported": True,
            "schedule": "bvp_solver",
            "history": [
                {
                    "step": entry["step"],
                    "relative_state_change": entry["relative_state_change"],
                    "mean_rms_D": sum(rms(d) for d in entry["D"]) / len(entry["D"]),
                }
                for entry in history
            ],
        }

    refine = getattr(model, "_refine", None)
    stem = getattr(model, "stem", None)
    classify = getattr(model, "_classify", None)
    pure_forward = getattr(model, "_pure_forward", None)
    if not all(callable(item) for item in (refine, stem, classify, pure_forward)):
        return {"supported": False}

    with torch.no_grad():
        source = stem(x)
        states = pure_forward(source)
        logits = classify(states[-1])
        history = []
        for step in range(1, extra_iterations + 1):
            boundary, _ = model.target_boundary(logits, source)
            new_states, _, trace = refine(source, boundary, reverse_off=False)
            numerator = sum(
                (new - old).square().sum()
                for old, new in zip(states, new_states, strict=True)
            ).sqrt()
            denominator = sum(state.square().sum() for state in states).sqrt()
            state_change = (numerator / (denominator + 1e-12)).item()
            stage = trace["proposal_D"]
            mean_discrepancy = sum(rms(d) for d in stage) / len(stage)
            history.append(
                {
                    "step": step,
                    "relative_state_change": state_change,
                    "mean_rms_D": mean_discrepancy,
                }
            )
            states = new_states
            logits = classify(states[-1])
    return {"supported": True, "schedule": "refine_cycle", "history": history}


def verdict(report: dict[str, Any]) -> dict[str, Any]:
    r1 = report["R1"]
    ratio_ok = all(RATIO_LOW <= row["ratio_C_over_H"] <= RATIO_HIGH for row in r1)
    cos_ok = all(row["cos_H_C"] > 0.0 for row in r1)
    r2 = report["R2"]
    r2_ok = bool(r2) and all(row["cos_D_Hbar"] < R2_MAX for row in r2)
    r3_ok = report["R3"]["random_hypothesis"] > R3_MIN
    history = report["R4"].get("history", [])
    r4_ok = bool(history) and history[-1]["relative_state_change"] > 1e-6

    checks = {
        "R1_same_space": {"pass": ratio_ok and cos_ok, "detail": "rms ratio in [0.8,1.25] and cos(H,C)>0 at every depth"},
        "R2_discrepancy_is_informative": {"pass": r2_ok, "detail": f"cos(D,H_bar) < {R2_MAX} at every depth"},
        "R3_boundary_carries_instance_content": {"pass": r3_ok, "detail": f"random-hypothesis logit change > {R3_MIN}"},
        "R4_iterative_solver": {"pass": r4_ok, "detail": "state keeps changing after the first round"},
    }
    checks["ALL_PASS"] = {"pass": all(item["pass"] for key, item in checks.items()), "detail": ""}
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=12, help="rounds for the R4 probe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    model = build_model(config["model"]).to(args.device).eval()

    x = torch.randn(args.batch_size, 3, 32, 32, device=args.device)
    with torch.no_grad():
        diagnostics = _coupled_diagnostics(model, x)
    report: dict[str, Any] = {"config": str(args.config), "model": repr(model)}
    report.update(measure_R1_R2(diagnostics))
    report["R3"] = measure_R3(model, x, args.seed)
    report["R4"] = measure_R4(model, x, args.iterations)
    report["verdict"] = verdict(report)

    print(f"model: {type(model).__name__}  config: {args.config.name}")
    print()
    print("R1  same representable space?")
    print(f"    {'depth':>5} {'rms(H)':>9} {'rms(C)':>9} {'ratio':>8} {'cos(H,C)':>10}")
    for row in report["R1"]:
        print(
            f"    {row['depth']:>5} {row['rms_H']:>9.4f} {row['rms_C']:>9.4f} "
            f"{row['ratio_C_over_H']:>8.3f} {row['cos_H_C']:>10.4f}"
        )
    print()
    print("R2  is the discrepancy informative, or just a rescaling of H?")
    if report["R2"]:
        print(f"    {'cell':>5} {'cos(D, H_bar)':>14}")
        for row in report["R2"]:
            print(f"    {row['depth']:>5} {row['cos_D_Hbar']:>14.4f}")
    else:
        print("    (not available: model does not expose proposal diagnostics)")
    print()
    print("R3  does the target hypothesis reach the prediction?")
    for name, value in report["R3"].items():
        print(f"    {name:<20} ||dlogits||/||logits|| = {value:.4f}")
    print()
    print("R4  does the iteration approach a fixed point?")
    if report["R4"].get("supported"):
        print(f"    {'step':>5} {'rel_state_change':>18} {'mean rms(D)':>13}")
        for row in report["R4"]["history"]:
            print(
                f"    {row['step']:>5} {row['relative_state_change']:>18.6f} "
                f"{row['mean_rms_D']:>13.4f}"
            )
    else:
        print("    (not available for this model)")
    print()
    print("verdict")
    for key, item in report["verdict"].items():
        if key == "ALL_PASS":
            continue
        print(f"    [{'PASS' if item['pass'] else 'FAIL'}] {key}: {item['detail']}")
    overall = report["verdict"]["ALL_PASS"]["pass"]
    print()
    print(f"    => {'all conditions satisfied' if overall else 'NOT all conditions satisfied'}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.output}")


if __name__ == "__main__":
    main()
