"""Plot H/C state norms over lattice depth and iteration."""

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
    h_norm = require_tensor(data, "H_norm")
    c_norm = require_tensor(data, "C_norm")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, values, title in zip(axes, (h_norm, c_norm), ("H state", "C state")):
        image = axis.imshow(values.numpy(), aspect="auto", origin="lower")
        ylabel = "iteration (H includes t=0; C starts at t=1)"
        axis.set(title=title, xlabel="depth", ylabel=ylabel)
        figure.colorbar(image, ax=axis, label="normalized L2")
    output = args.output or default_output(args.diagnostics, "state_norms")
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
