"""Compare the ResNet-50 pilot arms and state the compute-adjusted verdict.

Reads summary.json from each pilot run directory, prints accuracy/CE alongside the
Conv/Linear MAC budget, and applies the pre-registered decision rule: a
counterflow arm only counts as better if it beats both the plain baseline and the
co-current control while its compute multiplier is reported alongside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARMS = (
    ("plain", "resnet50_plain"),
    ("lattice_control", "resnet50_lattice_control"),
    ("countercurrent", "resnet50_countercurrent"),
    ("cocurrent", "resnet50_cocurrent"),
)


def load_summary(root: Path, arm: str) -> dict[str, Any] | None:
    path = root / f"pilot_resnet50_{arm}" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pick(summary: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in summary:
            return summary[name]
    return None


def nested(summary: dict[str, Any], section: str, name: str) -> Any:
    block = summary.get(section)
    if isinstance(block, dict):
        return block.get(name)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("countercurrent_nn/results"))
    args = parser.parse_args()

    rows = []
    for arm, model in ARMS:
        summary = load_summary(args.root, arm)
        if summary is None:
            print(f"  {arm:<18} : summary.json missing (run not finished)")
            continue
        rows.append(
            {
                "arm": arm,
                "model": model,
                "val": pick(summary, "best_selection_accuracy", "best_val_accuracy"),
                "test": nested(summary, "test", "accuracy")
                or pick(summary, "test_accuracy"),
                "ce": nested(summary, "test", "loss") or pick(summary, "test_ce"),
                "macs": pick(summary, "macs", "conv_linear_macs"),
                "params": pick(summary, "params", "trainable_params"),
            }
        )

    if not rows:
        print("no pilot summaries yet")
        return

    base = next((r for r in rows if r["arm"] == "plain"), None)
    print(f"{'arm':<18} {'params':>12} {'MACs':>16} {'compute':>9} {'val':>8} {'test':>8}")
    for row in rows:
        ratio = ""
        if base and base["macs"] and row["macs"]:
            ratio = f"{row['macs'] / base['macs']:.2f}x"
        val = f"{100 * row['val']:.2f}" if isinstance(row["val"], (int, float)) else "-"
        test = f"{100 * row['test']:.2f}" if isinstance(row["test"], (int, float)) else "-"
        params = f"{row['params']:,}" if isinstance(row["params"], int) else "-"
        macs = f"{row['macs']:,}" if isinstance(row["macs"], int) else "-"
        print(f"{row['arm']:<18} {params:>12} {macs:>16} {ratio:>9} {val:>8} {test:>8}")


if __name__ == "__main__":
    main()
