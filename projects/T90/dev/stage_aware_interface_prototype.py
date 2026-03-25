from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core import build_casebase_from_windows, get_context_sensors, get_target_range, load_runtime_config
from core.casebase import normalize_casebase_frame
from core.online_recommender import (
    _build_control_recommendation,
    _choose_context_columns,
    _find_context_neighborhood,
    _fit_local_control_model,
    _rank_context_features,
    _resolve_target_range,
    _safe_float,
)
from core.window_encoder import encode_dcs_window
from ph_augmented_window_experiment import attach_ph_features
from ph_lag_experiment import build_ph_features, load_ph_data
from ph_segmented_window_experiment import prune_empty_numeric_columns
from stage_identifier_experiment import (
    assign_stage_labels,
    build_policy_recommendation,
    build_stage_policy_rows,
    choose_stage_count,
    evaluate_stage_counts,
)
from test import build_windows_and_outcomes, load_dcs_data, load_lims_grouped


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_PH_PATH = PROJECT_DIR / "data" / "B4-AI-C53001A.PV.F_CV.xlsx"
DEFAULT_DCS_WINDOW = 50
DEFAULT_STAGE_COUNTS = (2, 3, 4, 5, 6)
DEFAULT_PH_LAGS = (120, 240, 300)
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"


def _parse_path_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    raise ValueError(f"Unsupported tabular file format: {source}")


def _resolve_target_spec(
    payload: dict[str, object],
    runtime_config: dict[str, object],
) -> tuple[float, float]:
    target_spec = payload.get("target_spec")
    if target_spec is not None:
        center = float(target_spec["center"])
        tolerance = float(target_spec["tolerance"])
        return center - tolerance, center + tolerance
    target_range = payload.get("target_range")
    if target_range is not None:
        return _resolve_target_range(target_range)
    return get_target_range(runtime_config)


def _parse_int_iterable(value: object, default: Iterable[int], *, minimum: int) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        parsed = sorted({int(item) for item in items})
    else:
        parsed = sorted({int(item) for item in value})
    if not parsed or any(item < minimum for item in parsed):
        raise ValueError(f"Expected integer values >= {minimum}.")
    return parsed


def _load_runtime_sources(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], tuple[float, float], str]:
    runtime_config = load_runtime_config(config_path)
    sensors = get_context_sensors(runtime_config)
    target_range = get_target_range(runtime_config)
    time_column = str(runtime_config.get("window", {}).get("time_column", "time"))
    data_sources = runtime_config.get("data_sources", {})
    dcs = load_dcs_data(data_sources.get("dcs_paths", []), sensors=sensors, time_column=time_column)
    lims = load_lims_grouped(PROJECT_DIR / str(data_sources["lims_path"]))
    ph = load_ph_data(DEFAULT_PH_PATH)
    return dcs, lims, ph, sensors, target_range, time_column


def _extract_current_ph_features(
    ph_history: pd.DataFrame,
    *,
    runtime_time: pd.Timestamp,
    lag_minutes: int,
    feature_window_minutes: int,
    tolerance_minutes: int,
) -> dict[str, float] | None:
    ph_features = build_ph_features(ph_history, feature_window_minutes=feature_window_minutes)
    target_frame = pd.DataFrame({"lag_target_time": [runtime_time - pd.Timedelta(minutes=lag_minutes)]})
    aligned = pd.merge_asof(
        target_frame.sort_values("lag_target_time"),
        ph_features.sort_values("time"),
        left_on="lag_target_time",
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    if aligned.empty:
        return None
    row = aligned.iloc[0]
    required = ["ph_point", "ph_mean", "ph_std", "ph_min", "ph_max", "ph_delta"]
    if any(pd.isna(row[column]) for column in required):
        return None
    return {column: float(row[column]) for column in required}


def _build_recommendation_from_context(
    *,
    current_context: dict[str, float],
    casebase: pd.DataFrame,
    runtime_time: str | None,
    target_range: tuple[float, float],
    reference_calcium: float | None,
    reference_bromine: float | None,
    top_context_features: int,
    neighbor_count: int,
    local_neighbor_count: int,
    probability_threshold: float,
    grid_points: int,
    include_columns: tuple[str, ...],
    skip_feature_ranking: bool,
) -> dict[str, object]:
    low, high = target_range
    prepared_casebase = normalize_casebase_frame(casebase, target_low=low, target_high=high)
    context_columns = _choose_context_columns(prepared_casebase, current_context, include_columns)
    if skip_feature_ranking or top_context_features <= 0 or len(context_columns) <= top_context_features:
        selected_context = context_columns
        feature_importance: dict[str, float] = {}
    else:
        selected_context, feature_importance = _rank_context_features(
            prepared_casebase,
            context_columns,
            top_n=top_context_features,
        )
    neighborhood = _find_context_neighborhood(
        prepared_casebase,
        current_context,
        selected_context,
        neighbor_count,
    )
    local, model = _fit_local_control_model(neighborhood, local_neighbor_count)
    recommendation = _build_control_recommendation(
        local,
        model,
        reference_calcium=_safe_float(reference_calcium),
        reference_bromine=_safe_float(reference_bromine),
        probability_threshold=probability_threshold,
        grid_points=grid_points,
    )
    return {
        "target_range": {"low": low, "high": high},
        "runtime_context": {
            "runtime_time": runtime_time,
            "reference_calcium": _safe_float(reference_calcium),
            "reference_bromine": _safe_float(reference_bromine),
        },
        "method": {
            "type": "stage-aware context-conditioned case-based recommendation",
            "selected_context_features": selected_context,
            "top_context_feature_importance": feature_importance,
            "feature_ranking_skipped": bool(skip_feature_ranking),
        },
        "neighborhood": {
            "context_samples": int(len(neighborhood)),
            "local_model_samples": int(len(local)),
            "local_good_samples": int(local["is_in_spec"].sum()),
            "local_bad_samples": int((1 - local["is_in_spec"]).sum()),
            "median_context_distance": float(neighborhood["context_distance"].median()),
        },
        "recommendation": recommendation,
    }


def _build_dev_online_request(
    *,
    dcs: pd.DataFrame,
    ph: pd.DataFrame,
    lims: pd.DataFrame,
    window_minutes: int,
    time_column: str,
) -> dict[str, object]:
    windows, outcomes = build_windows_and_outcomes(dcs, lims, window_minutes, time_column=time_column)
    if not windows or outcomes.empty:
        raise ValueError("Unable to build a dev online request from private data.")
    runtime_time = pd.Timestamp(outcomes.iloc[-1]["sample_time"])
    return {
        "dcs_window": windows[-1],
        "ph_history": ph.copy(),
        "runtime_time": str(runtime_time),
    }


def recommend_t90_controls_v2_dev(
    input_data: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = input_data or {}
    config_path = Path(str(payload.get("config_path", DEFAULT_CONFIG_PATH)))
    runtime_config = load_runtime_config(config_path)
    target_range = _resolve_target_spec(payload, runtime_config)
    window_minutes = int(payload.get("window_minutes", DEFAULT_DCS_WINDOW))
    stage_counts = _parse_int_iterable(payload.get("stage_counts"), DEFAULT_STAGE_COUNTS, minimum=2)
    ph_lags = _parse_int_iterable(payload.get("ph_lags"), DEFAULT_PH_LAGS, minimum=0)
    ph_feature_window = int(payload.get("ph_feature_window", 50))
    enable_threshold = float(payload.get("enable_threshold", 0.002))
    max_recommended_lag = int(payload.get("max_recommended_lag", 240))
    tolerance_minutes = int(payload.get("tolerance_minutes", 2))
    top_context_features = int(payload.get("top_context_features", 12))
    neighbor_count = int(payload.get("neighbor_count", 150))
    local_neighbor_count = int(payload.get("local_neighbor_count", 80))
    probability_threshold = float(payload.get("probability_threshold", 0.60))
    grid_points = int(payload.get("grid_points", 31))
    skip_feature_ranking = bool(payload.get("skip_feature_ranking", False))
    reference_calcium = payload.get("reference_calcium")
    reference_bromine = payload.get("reference_bromine")

    dcs, lims, ph_private, sensors, _historical_target_range, time_column = _load_runtime_sources(config_path)

    dcs_window = payload.get("dcs_window")
    if dcs_window is None and payload.get("dcs_window_path"):
        dcs_window = _parse_path_frame(payload["dcs_window_path"])
    ph_history = payload.get("ph_history")
    if ph_history is None and payload.get("ph_history_path"):
        ph_history = _parse_path_frame(payload["ph_history_path"])

    if payload.get("use_private_example", dcs_window is None):
        request = _build_dev_online_request(
            dcs=dcs,
            ph=ph_private,
            lims=lims,
            window_minutes=window_minutes,
            time_column=time_column,
        )
        dcs_window = request["dcs_window"]
        ph_history = request["ph_history"]
        payload.setdefault("runtime_time", request["runtime_time"])

    if dcs_window is None:
        raise ValueError("Please provide `dcs_window`, `dcs_window_path`, or set `use_private_example=True`.")
    if not isinstance(dcs_window, pd.DataFrame):
        raise TypeError("`dcs_window` must be a pandas DataFrame.")

    if ph_history is None:
        ph_history = ph_private.copy()
    elif not isinstance(ph_history, pd.DataFrame):
        raise TypeError("`ph_history` must be a pandas DataFrame.")
    else:
        ph_history = load_ph_data(payload["ph_history_path"]) if payload.get("ph_history_path") else ph_history.copy()

    runtime_time = pd.Timestamp(payload.get("runtime_time") or pd.to_datetime(dcs_window[time_column]).max())

    windows, outcomes = build_windows_and_outcomes(dcs, lims, window_minutes, time_column=time_column)
    base_casebase = build_casebase_from_windows(
        windows,
        outcomes,
        target_low=target_range[0],
        target_high=target_range[1],
        include_sensors=sensors,
    )
    stage_count_summary, context_columns, imputer, scaler = evaluate_stage_counts(
        base_casebase,
        stage_counts=stage_counts,
        min_stage_samples=int(payload.get("min_stage_samples", 120)),
    )
    chosen_stage_count = choose_stage_count(stage_count_summary)
    staged_casebase, stage_model = assign_stage_labels(
        base_casebase,
        stage_count=chosen_stage_count,
        context_columns=context_columns,
        imputer=imputer,
        scaler=scaler,
    )

    ph_features = build_ph_features(ph_private, feature_window_minutes=ph_feature_window)
    stage_lookup = staged_casebase[["sample_time", "stage_id", "stage_name"]].drop_duplicates()
    staged_augmented_by_lag: dict[int, pd.DataFrame] = {}
    for lag_minutes in ph_lags:
        augmented = attach_ph_features(
            base_casebase,
            lims,
            ph_features,
            lag_minutes=lag_minutes,
            tolerance_minutes=tolerance_minutes,
        )
        staged_augmented_by_lag[lag_minutes] = augmented.merge(stage_lookup, on="sample_time", how="inner")

    policy_summary = build_stage_policy_rows(
        staged_casebase,
        staged_augmented_by_lag,
        lags=ph_lags,
        limit=0,
        neighbor_count=neighbor_count,
        local_neighbor_count=local_neighbor_count,
        probability_threshold=probability_threshold,
        grid_points=grid_points,
        mae_threshold=0.05,
        p90_threshold=0.10,
        max_threshold=0.25,
        in_spec_range_ratio_threshold=0.55,
        success_ratio_threshold=1.0,
    )
    best_by_stage = (
        policy_summary.loc[policy_summary["lag_minutes"] >= 0]
        .sort_values(["stage_name", "delta_composite_score_vs_baseline"], ascending=[True, False])
        .groupby("stage_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    policy = build_policy_recommendation(
        best_by_stage,
        enable_threshold=enable_threshold,
        max_recommended_lag=max_recommended_lag,
    )
    policy_map = {item["stage_name"]: item for item in policy["recommendations"]}

    current_dcs_context = encode_dcs_window(dcs_window, include_sensors=sensors)
    current_vector = pd.DataFrame([{column: current_dcs_context.get(column, np.nan) for column in context_columns}])
    matrix = scaler.transform(imputer.transform(current_vector))
    current_stage_id = int(stage_model.predict(matrix)[0])
    current_stage_name = f"stage_{current_stage_id}"
    stage_decision = policy_map[current_stage_name]

    warnings: list[str] = []
    current_context = dict(current_dcs_context)
    selected_casebase = staged_casebase.loc[staged_casebase["stage_name"] == current_stage_name].drop(columns=["stage_id", "stage_name"]).reset_index(drop=True)
    ph_feature_payload: dict[str, float] | None = None

    if stage_decision["enable_ph"]:
        selected_lag = int(stage_decision["best_lag_minutes"])
        ph_feature_payload = _extract_current_ph_features(
            ph_history if {"time", "value"}.issubset(ph_history.columns) else load_ph_data(payload.get("ph_history_path", DEFAULT_PH_PATH)),
            runtime_time=runtime_time,
            lag_minutes=selected_lag,
            feature_window_minutes=ph_feature_window,
            tolerance_minutes=tolerance_minutes,
        )
        if ph_feature_payload is None:
            warnings.append("PH enhancement was recommended for the current stage, but runtime PH history was insufficient. Falling back to DCS-only recommendation.")
        else:
            current_context.update(ph_feature_payload)
            selected_casebase = staged_augmented_by_lag[selected_lag].loc[
                staged_augmented_by_lag[selected_lag]["stage_name"] == current_stage_name
            ].drop(columns=["stage_id", "stage_name"]).reset_index(drop=True)
    else:
        warnings.append("Current stage policy keeps PH disabled and uses the pure 50min DCS baseline.")

    selected_casebase = prune_empty_numeric_columns(selected_casebase)
    include_columns = tuple(
        payload.get(
            "include_columns",
            [column for column in selected_casebase.select_dtypes(include=["number"]).columns if column not in {"t90", "is_in_spec", "calcium", "bromine"}],
        )
    )
    recommendation = _build_recommendation_from_context(
        current_context=current_context,
        casebase=selected_casebase,
        runtime_time=str(runtime_time),
        target_range=target_range,
        reference_calcium=None if reference_calcium is None else float(reference_calcium),
        reference_bromine=None if reference_bromine is None else float(reference_bromine),
        top_context_features=top_context_features,
        neighbor_count=neighbor_count,
        local_neighbor_count=local_neighbor_count,
        probability_threshold=probability_threshold,
        grid_points=grid_points,
        include_columns=include_columns,
        skip_feature_ranking=skip_feature_ranking,
    )

    return {
        "target_range": {"low": float(target_range[0]), "high": float(target_range[1])},
        "runtime_context": {
            "runtime_time": str(runtime_time),
            "dcs_window_rows": int(len(dcs_window)),
            "ph_history_rows": int(len(ph_history)),
            "reference_calcium": _safe_float(reference_calcium),
            "reference_bromine": _safe_float(reference_bromine),
        },
        "stage_decision": {
            "chosen_stage_count": int(chosen_stage_count),
            "current_stage_name": current_stage_name,
            "ph_enabled": bool(stage_decision["enable_ph"] and ph_feature_payload is not None),
            "selected_ph_lag_minutes": None if ph_feature_payload is None else int(stage_decision["best_lag_minutes"]),
            "policy_reason": stage_decision["reason"],
            "stage_policy_gain": float(stage_decision["delta_composite_score_vs_baseline"]),
        },
        "policy_summary": policy,
        "current_ph_features": ph_feature_payload,
        "recommendation_report": recommendation,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev-only stage-aware entry function prototype for future T90 v2 interface.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--runtime-time", default=None, help="Optional runtime timestamp override.")
    parser.add_argument("--dcs-window-path", default=None, help="Optional csv/parquet DCS window path.")
    parser.add_argument("--ph-history-path", default=None, help="Optional xlsx/csv/parquet PH history path.")
    parser.add_argument("--use-private-example", action="store_true", help="Use the latest aligned private sample as the online request.")
    parser.add_argument("--output-prefix", default="", help="Optional output filename prefix.")
    args = parser.parse_args()

    payload: dict[str, object] = {
        "config_path": args.config,
        "use_private_example": bool(args.use_private_example or not args.dcs_window_path),
    }
    if args.runtime_time:
        payload["runtime_time"] = args.runtime_time
    if args.dcs_window_path:
        payload["dcs_window_path"] = args.dcs_window_path
    if args.ph_history_path:
        payload["ph_history_path"] = args.ph_history_path

    result = recommend_t90_controls_v2_dev(payload)
    output_prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    output_path = DEFAULT_RESULTS_DIR / f"{output_prefix}stage_aware_interface_prototype_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"result_saved_to={output_path}")


if __name__ == "__main__":
    main()
