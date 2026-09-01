"""Plot accuracy, confidence, and entropy from pure-forward t=0 through t=T."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

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
    accuracy = logits.argmax(dim=-1).eq(targets.unsqueeze(0)).float().mean(dim=1)
    confidence = probabilities.max(dim=-1).values.mean(dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=1)
    iterations = range(logits.shape[0])
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for axis, values, title in zip(
        axes, (accuracy, confidence, entropy), ("Accuracy", "Confidence", "Entropy")
    ):
        axis.plot(iterations, values, marker="o")
        axis.set(xlabel="outer inference iteration (t=0 is pure forward)", title=title)
    output = args.output or default_output(args.diagnostics, "iteration_accuracy")
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
