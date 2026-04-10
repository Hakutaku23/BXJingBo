from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(config_path: Path, maybe_relative: str | None) -> Path | None:
    if not maybe_relative:
        return None
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()

