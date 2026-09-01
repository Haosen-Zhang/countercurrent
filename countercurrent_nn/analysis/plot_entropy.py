"""Plot predictive entropy for all, initially correct, and initially wrong samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ._common import default_output, load_diagnostics, require_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    data = load_diagnostics(args.diagnostics)
    logits = require_tensor(data, "iter_logits")
    targets = require_tensor(data, "targets").long()
    probabilities = logits.softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    initially_correct = logits[0].argmax(dim=-1).eq(targets)

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    iterations = range(logits.shape[0])
    axis.plot(iterations, entropy.mean(dim=1), marker="o", label="all")
    for mask, label in (
        (initially_correct, "initially correct"),
        (~initially_correct, "initially wrong"),
    ):
        if bool(mask.any()):
            axis.plot(iterations, entropy[:, mask].mean(dim=1), marker="o", label=label)
    axis.set(
        xlabel="outer inference iteration (t=0 is pure forward)",
        ylabel="predictive entropy",
        title="Hypothesis uncertainty trajectory",
    )
    axis.legend()
    output = args.output or default_output(args.diagnostics, "entropy")
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
