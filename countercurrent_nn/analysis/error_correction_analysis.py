"""Report error correction and error amplification from saved diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from countercurrent_nn.engine import summarize_diagnostics

from ._common import load_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    args = parser.parse_args()
    summary = summarize_diagnostics(load_diagnostics(args.diagnostics))
    keys = (
        "initial_accuracy",
        "final_accuracy",
        "initial_wrong_examples",
        "error_correction_rate",
        "error_amplification_rate",
    )
    print(json.dumps({key: summary.get(key) for key in keys}, indent=2))


if __name__ == "__main__":
    main()
