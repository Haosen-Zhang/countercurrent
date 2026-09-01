"""Plot transported-state discrepancy across depth."""

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
    values = require_tensor(load_diagnostics(args.diagnostics), "discrepancy")
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for iteration in range(values.shape[0]):
        axis.plot(range(values.shape[1]), values[iteration], alpha=0.65, label=f"t={iteration + 1}")
    axis.set(xlabel="cell depth", ylabel="normalized discrepancy", title="Driving discrepancy")
    axis.legend(ncol=2, fontsize="small")
    output = args.output or default_output(args.diagnostics, "discrepancy")
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
