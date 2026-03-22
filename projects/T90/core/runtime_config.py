from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Runtime config not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("Runtime config must be a YAML mapping.")
    return data


def get_context_sensors(config: dict[str, Any]) -> list[str]:
    sensors = config.get("context_sensors", [])
    if not isinstance(sensors, list):
        raise ValueError("`context_sensors` must be a list in runtime config.")
    return [str(item) for item in sensors]


def get_target_spec(config: dict[str, Any]) -> tuple[float, float] | None:
    spec = config.get("target_spec")
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("`target_spec` must be a mapping in runtime config.")
    center = spec.get("center")
    tolerance = spec.get("tolerance")
    if center is None or tolerance is None:
        raise ValueError("`target_spec` must contain `center` and `tolerance`.")
    center = float(center)
    tolerance = float(tolerance)
    if tolerance < 0:
        raise ValueError("`target_spec.tolerance` must be non-negative.")
    return center, tolerance


def get_target_range(config: dict[str, Any]) -> tuple[float, float] | None:
    spec = get_target_spec(config)
    if spec is not None:
        center, tolerance = spec
        return center - tolerance, center + tolerance

    target = config.get("target_range")
    if target is None:
        return None
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        raise ValueError("`target_range` must be a two-item list in runtime config.")
    return float(target[0]), float(target[1])
