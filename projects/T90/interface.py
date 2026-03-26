from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from core import (
        build_stage_aware_recommendation,
        get_context_sensors,
        get_target_range,
        load_runtime_config,
        load_stage_aware_example_bundle,
        load_stage_casebase,
        load_stage_policy,
    )
except ModuleNotFoundError:
    from .core import (
        build_stage_aware_recommendation,
        get_context_sensors,
        get_target_range,
        load_runtime_config,
        load_stage_aware_example_bundle,
        load_stage_casebase,
        load_stage_policy,
    )


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"


def _resolve_input_path(path_value: object, *, base_dir: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else (base_dir / path).resolve()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported tabular file format: {path}")


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
    Build a stage-aware T90 recommendation from the latest DCS window.

    Expected payload keys:
    - dcs_window / dcs_window_path: current 50min DCS window.
    - ph_history / ph_history_path: optional PH history. The interface must still work
      when this input is omitted, and it will automatically fall back to the DCS-only
      path for stages that cannot use PH at runtime.
    - casebase / casebase_path: optional override for the base 50min casebase.
    - ph_casebase / ph_casebase_path: optional override for the PH(120min)-augmented casebase.
    - stage_policy / stage_policy_path: optional override for the precomputed stage policy JSON.
    - config_path: optional runtime YAML config path.
    - runtime_time: optional timestamp for logging and PH lag alignment.
    - reference_calcium / reference_bromine: optional operator reference values.
    - target_spec / target_range: optional T90 target override.
    - top_context_features / neighbor_count / local_neighbor_count /
      probability_threshold / grid_points / include_columns / include_sensors /
      skip_feature_ranking: optional algorithm parameters.
    - use_example_data: load the packaged CPU-only example request when explicit inputs are omitted.
    """
    payload = input_data or {}
    config_path = payload.get("config_path")
    resolved_config_path = _resolve_input_path(config_path, base_dir=PROJECT_DIR) if config_path else DEFAULT_CONFIG_PATH
    runtime_config = load_runtime_config(resolved_config_path)
    artifacts = runtime_config.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("`artifacts` must be a mapping in runtime config.")

    dcs_window = payload.get("dcs_window")
    dcs_window_path = payload.get("dcs_window_path")
    ph_history = payload.get("ph_history")
    ph_history_path = payload.get("ph_history_path")
    base_casebase = payload.get("casebase")
    base_casebase_path = payload.get("casebase_path")
    ph_casebase = payload.get("ph_casebase")
    ph_casebase_path = payload.get("ph_casebase_path")
    stage_policy = payload.get("stage_policy")
    stage_policy_path = payload.get("stage_policy_path")

    use_example_data = bool(
        payload.get(
            "use_example_data",
            dcs_window is None and not dcs_window_path,
        )
    )
    if use_example_data:
        example_bundle = load_stage_aware_example_bundle(resolved_config_path, project_dir=PROJECT_DIR)
        dcs_window = example_bundle["dcs_window"]
        ph_history = example_bundle.get("ph_history")
        payload.setdefault("runtime_time", example_bundle.get("runtime_time"))

    if dcs_window is None:
        if dcs_window_path:
            dcs_window = _read_table(_resolve_input_path(dcs_window_path, base_dir=PROJECT_DIR))
        else:
            raise ValueError("Please provide `dcs_window`, `dcs_window_path`, or set `use_example_data=True`.")
    if ph_history is None and ph_history_path:
        ph_history = _read_table(_resolve_input_path(ph_history_path, base_dir=PROJECT_DIR))

    target_range = _resolve_interface_target_range(payload, runtime_config)
    include_sensors = tuple(payload.get("include_sensors", get_context_sensors(runtime_config)))

    if base_casebase is None:
        resolved_casebase_path = _resolve_input_path(
            base_casebase_path or artifacts.get("casebase_path"),
            base_dir=PROJECT_DIR,
        )
        base_casebase = load_stage_casebase(resolved_casebase_path, target_range=target_range)
    if ph_casebase is None and artifacts.get("ph_casebase_path"):
        resolved_ph_casebase_path = _resolve_input_path(
            ph_casebase_path or artifacts.get("ph_casebase_path"),
            base_dir=PROJECT_DIR,
        )
        ph_casebase = load_stage_casebase(resolved_ph_casebase_path, target_range=target_range)
    if stage_policy is None:
        resolved_stage_policy_path = _resolve_input_path(
            stage_policy_path or artifacts.get("stage_policy_path"),
            base_dir=PROJECT_DIR,
        )
        stage_policy = load_stage_policy(resolved_stage_policy_path)

    if not isinstance(dcs_window, pd.DataFrame):
        raise TypeError("`dcs_window` must be a pandas DataFrame.")
    if ph_history is not None and not isinstance(ph_history, pd.DataFrame):
        raise TypeError("`ph_history` must be a pandas DataFrame when provided.")

    reference_calcium = payload.get("reference_calcium", payload.get("current_calcium"))
    reference_bromine = payload.get("reference_bromine", payload.get("current_bromine"))
    ph_config = runtime_config.get("ph", {})
    if not isinstance(ph_config, dict):
        ph_config = {}

    return build_stage_aware_recommendation(
        dcs_window=dcs_window,
        base_casebase=base_casebase,
        stage_policy=stage_policy,
        ph_casebase=ph_casebase,
        ph_history=ph_history,
        runtime_time=payload.get("runtime_time"),
        reference_calcium=reference_calcium,
        reference_bromine=reference_bromine,
        target_range=target_range,
        top_context_features=int(payload.get("top_context_features", 12)),
        neighbor_count=int(payload.get("neighbor_count", 150)),
        local_neighbor_count=int(payload.get("local_neighbor_count", 80)),
        probability_threshold=float(payload.get("probability_threshold", 0.60)),
        grid_points=int(payload.get("grid_points", 31)),
        include_sensors=include_sensors,
        include_columns=tuple(payload.get("include_columns", ())),
        skip_feature_ranking=bool(payload.get("skip_feature_ranking", False)),
        ph_time_column=payload.get("ph_time_column", ph_config.get("history_time_column")),
        ph_value_column=payload.get("ph_value_column", ph_config.get("history_value_column")),
    )
