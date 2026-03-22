from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from core import (
        build_t90_recommendation,
        get_context_sensors,
        get_target_range,
        load_casebase,
        load_example_bundle,
        load_runtime_config,
    )
except ModuleNotFoundError:
    from .core import (
        build_t90_recommendation,
        get_context_sensors,
        get_target_range,
        load_casebase,
        load_example_bundle,
        load_runtime_config,
    )


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_CASEBASE_PATH = PROJECT_DIR / "assets" / "t90_casebase.csv"


def _resolve_interface_target_range(
    payload: dict[str, object],
    runtime_config: dict[str, object] | None,
) -> tuple[float, float] | None:
    target_spec = payload.get("target_spec")
    if target_spec is not None:
        if not isinstance(target_spec, dict):
            raise ValueError("`target_spec` must be a mapping with `center` and `tolerance`.")
        center = float(target_spec["center"])
        tolerance = float(target_spec["tolerance"])
        if tolerance < 0:
            raise ValueError("`target_spec.tolerance` must be non-negative.")
        return center - tolerance, center + tolerance

    target_range = payload.get("target_range")
    if target_range is not None:
        if not isinstance(target_range, (list, tuple)) or len(target_range) != 2:
            raise ValueError("`target_range` must be a two-item list or tuple.")
        return float(target_range[0]), float(target_range[1])

    if runtime_config:
        return get_target_range(runtime_config)
    return None


def recommend_t90_controls(
    input_data: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Build a recommendation report for T90 in-spec control from the latest DCS window.

    Parameters
    ----------
    input_data:
        Optional dictionary-style interface payload. Supported keys:
        - dcs_window: a pandas DataFrame for the latest fixed-length DCS window.
        - dcs_window_path: path to a parquet/csv file containing the same DCS window.
        - casebase: a pandas DataFrame containing the historical casebase.
        - casebase_path: path to a csv/parquet historical casebase file.
        - config_path: optional YAML config for selected DCS sensors and defaults.
        - runtime_time: timestamp of the current recommendation call.
        - reference_calcium/reference_bromine: optional latest lab values, setpoints,
          or operator reference values. They are not required for online recommendation.
          current_calcium/current_bromine are still accepted as backward-compatible aliases.
        - target_spec: mapping such as {"center": 8.45, "tolerance": 0.25}.
        - target_range: two-item iterable [low, high], default [7.5, 8.5].
        - top_context_features, neighbor_count, local_neighbor_count,
          probability_threshold, grid_points, include_columns, include_sensors,
          skip_feature_ranking: algorithm parameters.
        - use_example_data: bool, load bundled example casebase and DCS window when no explicit data is provided.
    """
    payload = input_data or {}
    dcs_window = payload.get("dcs_window")
    dcs_window_path = payload.get("dcs_window_path")
    casebase = payload.get("casebase")
    casebase_path = payload.get("casebase_path")
    config_path = payload.get("config_path")
    runtime_config = None
    resolved_config_path = Path(str(config_path)) if config_path else DEFAULT_CONFIG_PATH
    if resolved_config_path.exists():
        runtime_config = load_runtime_config(resolved_config_path)
    use_example_data = bool(
        payload.get(
            "use_example_data",
            dcs_window is None and not dcs_window_path and casebase is None and not casebase_path,
        )
    )

    if use_example_data:
        example_bundle = load_example_bundle()
        dcs_window = example_bundle["dcs_window"]
        casebase = example_bundle["casebase"]
        payload.setdefault("runtime_time", example_bundle.get("runtime_time"))
        payload.setdefault("current_calcium", example_bundle.get("current_calcium"))
        payload.setdefault("current_bromine", example_bundle.get("current_bromine"))

    if dcs_window is None:
        if dcs_window_path:
            path = Path(str(dcs_window_path))
            if path.suffix.lower() == ".csv":
                dcs_window = pd.read_csv(path)
            else:
                dcs_window = pd.read_parquet(path)
        else:
            raise ValueError("Please provide `dcs_window`, `dcs_window_path`, or set `use_example_data=True`.")

    if casebase is None:
        if casebase_path:
            casebase = load_casebase(
                casebase_path,
                target_range=payload.get("target_range"),
            )
        elif not use_example_data and DEFAULT_CASEBASE_PATH.exists():
            casebase = load_casebase(
                DEFAULT_CASEBASE_PATH,
                target_range=payload.get("target_range") or (get_target_range(runtime_config) if runtime_config else None),
            )
        elif not use_example_data:
            raise ValueError("Please provide `casebase`, `casebase_path`, or set `use_example_data=True`.")

    if not isinstance(dcs_window, pd.DataFrame):
        raise TypeError("`dcs_window` must be a pandas DataFrame.")
    if not isinstance(casebase, pd.DataFrame):
        raise TypeError("`casebase` must be a pandas DataFrame.")

    reference_calcium = payload.get("reference_calcium", payload.get("current_calcium"))
    reference_bromine = payload.get("reference_bromine", payload.get("current_bromine"))
    target_range = _resolve_interface_target_range(payload, runtime_config)

    return build_t90_recommendation(
        dcs_window=dcs_window,
        casebase=casebase,
        runtime_time=payload.get("runtime_time"),
        reference_calcium=reference_calcium,
        reference_bromine=reference_bromine,
        target_range=target_range,
        top_context_features=int(payload.get("top_context_features", 12)),
        neighbor_count=int(payload.get("neighbor_count", 150)),
        local_neighbor_count=int(payload.get("local_neighbor_count", 80)),
        probability_threshold=float(payload.get("probability_threshold", 0.60)),
        grid_points=int(payload.get("grid_points", 31)),
        include_sensors=tuple(payload.get("include_sensors", get_context_sensors(runtime_config) if runtime_config else ())),
        include_columns=tuple(payload.get("include_columns", ())),
        skip_feature_ranking=bool(payload.get("skip_feature_ranking", False)),
    )
