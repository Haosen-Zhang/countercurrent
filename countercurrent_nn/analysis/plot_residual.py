"""Plot relative H-state change across outer refinement iterations."""

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
    residual = require_tensor(load_diagnostics(args.diagnostics), "residual")
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(range(1, len(residual) + 1), residual, marker="o")
    axis.set(
        xlabel="refinement iteration",
        ylabel="relative H-state change",
        title="Outer refinement dynamics",
    )
    output = args.output or default_output(args.diagnostics, "residual")
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
