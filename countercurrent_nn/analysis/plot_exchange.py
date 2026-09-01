"""Plot exchange magnitude by depth across iterations."""

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
    values = require_tensor(load_diagnostics(args.diagnostics), "q_norm")
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for depth in range(values.shape[1]):
        axis.plot(range(1, values.shape[0] + 1), values[:, depth], marker="o", label=f"cell {depth}")
    axis.set(xlabel="iteration", ylabel="normalized ||Q||₂", title="Local exchange magnitude")
    axis.legend()
    output = args.output or default_output(args.diagnostics, "exchange")
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
