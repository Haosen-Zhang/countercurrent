"""YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = deepcopy(loaded)
    config["_config_path"] = str(config_path.resolve())
    return config


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)


def require_sections(config: dict[str, Any], *sections: str) -> None:
    missing = [name for name in sections if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"missing mapping section(s): {', '.join(missing)}")
