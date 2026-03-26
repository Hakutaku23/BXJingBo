from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .casebase import load_casebase_dataset, normalize_casebase_frame
from .online_recommender import (
    _build_control_recommendation,
    _choose_context_columns,
    _find_context_neighborhood,
    _fit_local_control_model,
    _rank_context_features,
    _resolve_target_range,
    _safe_float,
)
from .runtime_config import get_context_sensors, get_target_range, load_runtime_config
from .window_encoder import encode_dcs_window


DEFAULT_PH_FEATURE_COLUMNS = ("ph_point", "ph_mean", "ph_std", "ph_min", "ph_max", "ph_delta")
BLOCKED_CONTEXT_COLUMNS = {"sample_time", "t90", "is_in_spec", "calcium", "bromine", "stage_id"}


def load_stage_policy(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Stage policy file not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Stage policy must be a JSON object.")
    return data


def load_stage_casebase(
    path: str | Path,
    *,
    target_range: Iterable[float] | None = None,
) -> pd.DataFrame:
    low, high = _resolve_target_range(target_range)
    return load_casebase_dataset(path, target_low=low, target_high=high)


def _drop_stage_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("stage_id", "stage_name") if column in frame.columns]
    if not columns:
        return frame.copy()
    return frame.drop(columns=columns).copy()


def _prune_empty_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    drop_columns = [column for column in numeric_columns if frame[column].notna().sum() == 0]
    if not drop_columns:
        return frame
    return frame.drop(columns=drop_columns).copy()


def _normalize_ph_history(
    ph_history: pd.DataFrame,
    *,
    time_column: str | None = None,
    value_column: str | None = None,
) -> pd.DataFrame:
    if not isinstance(ph_history, pd.DataFrame):
        raise TypeError("`ph_history` must be a pandas DataFrame.")
    if ph_history.empty:
        raise ValueError("`ph_history` is empty.")

    frame = ph_history.copy()
    candidates = {str(column).lower(): str(column) for column in frame.columns}
    resolved_time = time_column if time_column in frame.columns else candidates.get("time") or candidates.get("timestamp") or candidates.get("sample_time")
    resolved_value = value_column if value_column in frame.columns else candidates.get("value") or candidates.get("ph") or candidates.get("ph_value")

    if resolved_time is None:
        for column in frame.columns:
            series = pd.to_datetime(frame[column], errors="coerce")
            if series.notna().sum() >= max(3, len(frame) // 3):
                resolved_time = str(column)
                break
    if resolved_value is None:
        for column in frame.columns:
            if str(column) == str(resolved_time):
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            if series.notna().sum() >= max(3, len(frame) // 3):
                resolved_value = str(column)
                break

    if resolved_time is None or resolved_value is None:
        raise ValueError("`ph_history` must provide usable time and value columns.")

    normalized = pd.DataFrame()
    normalized["time"] = pd.to_datetime(frame[resolved_time], errors="coerce")
    normalized["value"] = pd.to_numeric(frame[resolved_value], errors="coerce")
    normalized = normalized.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    if normalized.empty:
        raise ValueError("`ph_history` does not contain valid time/value rows.")
    return normalized


def build_ph_features(
    ph_history: pd.DataFrame,
    *,
    feature_window_minutes: int,
    time_column: str | None = None,
    value_column: str | None = None,
) -> pd.DataFrame:
    if feature_window_minutes <= 1:
        raise ValueError("`feature_window_minutes` must be greater than 1.")

    normalized = _normalize_ph_history(
        ph_history,
        time_column=time_column,
        value_column=value_column,
    )
    rolling = normalized["value"].rolling(
        window=feature_window_minutes,
        min_periods=max(5, feature_window_minutes // 3),
    )
    features = normalized.copy()
    features["ph_point"] = features["value"]
    features["ph_mean"] = rolling.mean()
    features["ph_std"] = rolling.std()
    features["ph_min"] = rolling.min()
    features["ph_max"] = rolling.max()
    features["ph_delta"] = features["value"] - features["value"].shift(feature_window_minutes - 1)
    return features.drop(columns=["value"])


def extract_current_ph_features(
    ph_history: pd.DataFrame,
    *,
    runtime_time: str | pd.Timestamp,
    lag_minutes: int,
    feature_window_minutes: int,
    tolerance_minutes: int,
    time_column: str | None = None,
    value_column: str | None = None,
) -> dict[str, float] | None:
    runtime_ts = pd.Timestamp(runtime_time)
    features = build_ph_features(
        ph_history,
        feature_window_minutes=feature_window_minutes,
        time_column=time_column,
        value_column=value_column,
    )
    target_frame = pd.DataFrame({"lag_target_time": [runtime_ts - pd.Timedelta(minutes=lag_minutes)]})
    aligned = pd.merge_asof(
        target_frame.sort_values("lag_target_time"),
        features.sort_values("time"),
        left_on="lag_target_time",
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    if aligned.empty:
        return None
    row = aligned.iloc[0]
    if any(pd.isna(row[column]) for column in DEFAULT_PH_FEATURE_COLUMNS):
        return None
    payload = {column: float(row[column]) for column in DEFAULT_PH_FEATURE_COLUMNS}
    payload["alignment_error_minutes"] = float(abs((row["time"] - row["lag_target_time"]) / pd.Timedelta(minutes=1)))
    return payload


def assign_stage_from_policy(
    current_context: dict[str, float],
    stage_policy: dict[str, object],
) -> dict[str, object]:
    identifier = stage_policy.get("stage_identifier")
    if not isinstance(identifier, dict):
        raise ValueError("Stage policy does not contain a valid `stage_identifier` section.")

    context_columns = [str(column) for column in identifier.get("context_columns", [])]
    imputer_statistics = identifier.get("imputer_statistics", {})
    scaler_mean = identifier.get("scaler_mean", {})
    scaler_scale = identifier.get("scaler_scale", {})
    centroids = identifier.get("stage_centroids", {})
    if not context_columns or not isinstance(centroids, dict):
        raise ValueError("Stage policy does not contain the required centroid metadata.")

    scaled_vector: list[float] = []
    raw_vector: dict[str, float] = {}
    for column in context_columns:
        value = current_context.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = imputer_statistics.get(column)
        if value is None:
            value = 0.0
        value = float(value)
        raw_vector[column] = value
        mean = float(scaler_mean.get(column, 0.0))
        scale = float(scaler_scale.get(column, 1.0)) or 1.0
        scaled_vector.append((value - mean) / scale)

    vector = np.asarray(scaled_vector, dtype=float)
    distances: dict[str, float] = {}
    for stage_name, centroid_map in centroids.items():
        centroid = np.asarray([float(centroid_map.get(column, 0.0)) for column in context_columns], dtype=float)
        distances[str(stage_name)] = float(np.linalg.norm(vector - centroid))

    chosen_stage_name = min(distances, key=distances.get)
    return {
        "stage_name": chosen_stage_name,
        "stage_distance": float(distances[chosen_stage_name]),
        "stage_distances": distances,
        "context_columns": context_columns,
        "imputed_context": raw_vector,
    }


def _get_stage_recommendation(stage_policy: dict[str, object], stage_name: str) -> dict[str, object]:
    policy = stage_policy.get("policy", {})
    recommendations = policy.get("recommendations", {})
    if isinstance(recommendations, list):
        recommendations = {item["stage_name"]: item for item in recommendations}
    if not isinstance(recommendations, dict):
        raise ValueError("Stage policy recommendations are malformed.")
    stage_decision = recommendations.get(stage_name)
    if not isinstance(stage_decision, dict):
        raise ValueError(f"Stage policy does not define a recommendation for {stage_name}.")
    return stage_decision


def _select_stage_casebase(casebase: pd.DataFrame, stage_name: str) -> pd.DataFrame:
    if "stage_name" not in casebase.columns:
        return _drop_stage_columns(casebase)
    selected = casebase.loc[casebase["stage_name"] == stage_name].reset_index(drop=True)
    if selected.empty:
        return _drop_stage_columns(casebase)
    return _drop_stage_columns(selected)


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
    include_columns: Sequence[str],
    skip_feature_ranking: bool,
) -> dict[str, object]:
    low, high = target_range
    prepared_casebase = normalize_casebase_frame(casebase, target_low=low, target_high=high)
    prepared_casebase = _prune_empty_numeric_columns(prepared_casebase)
    context_columns = _choose_context_columns(prepared_casebase, current_context, tuple(include_columns))
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


def build_stage_aware_recommendation(
    *,
    dcs_window: pd.DataFrame,
    base_casebase: pd.DataFrame,
    stage_policy: dict[str, object],
    ph_casebase: pd.DataFrame | None = None,
    ph_history: pd.DataFrame | None = None,
    runtime_time: str | pd.Timestamp | None = None,
    reference_calcium: float | None = None,
    reference_bromine: float | None = None,
    target_range: Iterable[float] | None = None,
    top_context_features: int = 12,
    neighbor_count: int = 150,
    local_neighbor_count: int = 80,
    probability_threshold: float = 0.60,
    grid_points: int = 31,
    include_sensors: Iterable[str] = (),
    include_columns: Iterable[str] = (),
    skip_feature_ranking: bool = False,
    ph_time_column: str | None = None,
    ph_value_column: str | None = None,
) -> dict[str, object]:
    if not isinstance(dcs_window, pd.DataFrame):
        raise TypeError("`dcs_window` must be a pandas DataFrame.")
    if not isinstance(base_casebase, pd.DataFrame):
        raise TypeError("`base_casebase` must be a pandas DataFrame.")

    policy_target = stage_policy.get("target_range")
    resolved_target_range = target_range
    if resolved_target_range is None and isinstance(policy_target, dict):
        resolved_target_range = (policy_target.get("low"), policy_target.get("high"))
    low, high = _resolve_target_range(resolved_target_range)

    current_context = encode_dcs_window(dcs_window, include_sensors=tuple(include_sensors))
    runtime_ts = pd.Timestamp(runtime_time) if runtime_time is not None else None
    if runtime_ts is None:
        time_column = next((column for column in dcs_window.columns if str(column).lower() in {"time", "timestamp", "sample_time"}), None)
        if time_column is not None:
            runtime_ts = pd.to_datetime(dcs_window[time_column], errors="coerce").max()
    if runtime_ts is None or pd.isna(runtime_ts):
        runtime_ts = pd.Timestamp.utcnow()

    stage_assignment = assign_stage_from_policy(current_context, stage_policy)
    stage_name = str(stage_assignment["stage_name"])
    stage_decision = _get_stage_recommendation(stage_policy, stage_name)

    warnings: list[str] = []
    current_ph_features: dict[str, float] | None = None
    ph_policy_enabled = bool(stage_decision.get("enable_ph", False))
    selected_casebase = _select_stage_casebase(base_casebase, stage_name)
    selected_casebase_source = "base_casebase"

    ph_config = stage_policy.get("ph", {})
    if not isinstance(ph_config, dict):
        ph_config = {}
    if ph_policy_enabled and ph_casebase is not None and ph_history is not None:
        lag_minutes = int(stage_decision.get("best_lag_minutes", 0))
        current_ph_features = extract_current_ph_features(
            ph_history,
            runtime_time=runtime_ts,
            lag_minutes=lag_minutes,
            feature_window_minutes=int(ph_config.get("feature_window_minutes", 50)),
            tolerance_minutes=int(ph_config.get("tolerance_minutes", 2)),
            time_column=ph_time_column or ph_config.get("history_time_column"),
            value_column=ph_value_column or ph_config.get("history_value_column"),
        )
        if current_ph_features is None:
            warnings.append(
                "Current stage prefers PH enhancement, but runtime PH history was insufficient. Falling back to the DCS-only stage casebase."
            )
        else:
            selected_casebase = _select_stage_casebase(ph_casebase, stage_name)
            selected_casebase_source = "ph_casebase"
            current_context.update({column: current_ph_features[column] for column in DEFAULT_PH_FEATURE_COLUMNS if column in current_ph_features})
    elif ph_policy_enabled and ph_casebase is None:
        warnings.append("Current stage prefers PH enhancement, but the PH-enhanced casebase is unavailable. Falling back to the DCS-only stage casebase.")
    elif ph_policy_enabled and ph_history is None:
        warnings.append("Current stage prefers PH enhancement, but runtime PH history was not provided. Falling back to the DCS-only stage casebase.")
    else:
        warnings.append("Current stage uses the pure 50min DCS baseline and keeps PH disabled.")

    selected_casebase = _prune_empty_numeric_columns(selected_casebase)
    result = _build_recommendation_from_context(
        current_context=current_context,
        casebase=selected_casebase,
        runtime_time=str(runtime_ts),
        target_range=(low, high),
        reference_calcium=reference_calcium,
        reference_bromine=reference_bromine,
        top_context_features=top_context_features,
        neighbor_count=neighbor_count,
        local_neighbor_count=local_neighbor_count,
        probability_threshold=probability_threshold,
        grid_points=grid_points,
        include_columns=tuple(include_columns),
        skip_feature_ranking=skip_feature_ranking,
    )

    method = dict(result["method"])
    method["type"] = "stage-aware context-conditioned case-based recommendation"
    method["goal"] = "Use a 50min DCS window to identify the current stage, then apply stage-specific calcium and bromine recommendation logic."
    method["primary_control_priority"] = ["calcium", "bromine"]
    method["stage_casebase_source"] = selected_casebase_source
    method["ph_is_optional_input"] = True
    method["ph_policy_enabled_for_stage"] = ph_policy_enabled
    method["ph_fallback_to_dcs_only"] = bool(ph_policy_enabled and current_ph_features is None)

    return {
        "target_range": {"low": float(low), "high": float(high)},
        "runtime_context": {
            "runtime_time": str(runtime_ts),
            "window_rows": int(len(dcs_window)),
            "ph_history_rows": None if ph_history is None else int(len(ph_history)),
            "ph_history_provided": bool(ph_history is not None),
            "reference_calcium": _safe_float(reference_calcium),
            "reference_bromine": _safe_float(reference_bromine),
        },
        "method": method,
        "stage_decision": {
            "current_stage_name": stage_name,
            "current_stage_distance": float(stage_assignment["stage_distance"]),
            "stage_distances": stage_assignment["stage_distances"],
            "chosen_stage_count": int(stage_policy.get("stage_identifier", {}).get("stage_count", 0) or 0),
            "ph_optional_input": True,
            "ph_policy_enabled": ph_policy_enabled,
            "ph_history_provided": bool(ph_history is not None),
            "ph_used": bool(current_ph_features is not None),
            "ph_fallback_to_dcs_only": bool(ph_policy_enabled and current_ph_features is None),
            "selected_ph_lag_minutes": None if current_ph_features is None else int(stage_decision.get("best_lag_minutes", 0)),
            "policy_reason": str(stage_decision.get("reason", "")),
            "stage_policy_gain": float(stage_decision.get("delta_composite_score_vs_baseline", 0.0)),
            "stage_casebase_source": selected_casebase_source,
        },
        "current_ph_features": current_ph_features,
        "neighborhood": result["neighborhood"],
        "recommendation": result["recommendation"],
        "warnings": warnings,
    }


def _build_dcs_window_from_template(
    template_row: pd.Series,
    *,
    sensors: Sequence[str],
    window_minutes: int,
    runtime_time: pd.Timestamp,
) -> pd.DataFrame:
    timestamps = pd.date_range(end=runtime_time, periods=window_minutes, freq="min")
    rows: list[dict[str, object]] = []
    for index, ts in enumerate(timestamps):
        row: dict[str, object] = {"time": ts}
        offset = index - (window_minutes - 1)
        for sensor in sensors:
            last_value = _safe_float(template_row.get(f"{sensor}__last")) or 0.0
            slope_value = _safe_float(template_row.get(f"{sensor}__slope"))
            delta_value = _safe_float(template_row.get(f"{sensor}__delta"))
            if slope_value is not None:
                row[sensor] = float(last_value + slope_value * offset)
            elif delta_value is not None and window_minutes > 1:
                start_value = float(last_value - delta_value)
                row[sensor] = float(start_value + (last_value - start_value) * index / (window_minutes - 1))
            else:
                row[sensor] = float(last_value)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_ph_history_from_template(
    template_row: pd.Series,
    *,
    runtime_time: pd.Timestamp,
    lag_minutes: int,
    feature_window_minutes: int,
) -> pd.DataFrame:
    end_time = runtime_time - pd.Timedelta(minutes=lag_minutes)
    timestamps = pd.date_range(end=end_time, periods=feature_window_minutes, freq="min")

    last_value = _safe_float(template_row.get("ph_point"))
    mean_value = _safe_float(template_row.get("ph_mean"))
    delta_value = _safe_float(template_row.get("ph_delta"))
    if last_value is None:
        last_value = 7.0
    if mean_value is None and delta_value is not None:
        mean_value = last_value - delta_value / 2.0
    if mean_value is None:
        mean_value = last_value

    start_from_mean = 2.0 * mean_value - last_value
    start_from_delta = last_value - delta_value if delta_value is not None else start_from_mean
    start_value = float((start_from_mean + start_from_delta) / 2.0)
    values = np.linspace(start_value, float(last_value), feature_window_minutes)
    return pd.DataFrame({"time": timestamps, "value": values})


def load_stage_aware_example_bundle(
    config_path: str | Path,
    *,
    project_dir: str | Path | None = None,
) -> dict[str, object]:
    project_root = Path(project_dir) if project_dir else Path(config_path).resolve().parents[1]
    runtime_config = load_runtime_config(config_path)
    artifacts = runtime_config.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("`artifacts` must be a mapping in the runtime config.")

    def resolve(relative_path: str) -> Path:
        return (project_root / relative_path).resolve()

    target_range = get_target_range(runtime_config)
    sensors = get_context_sensors(runtime_config)
    window_minutes = int(runtime_config.get("window", {}).get("minutes", 50))

    base_casebase = load_stage_casebase(resolve(str(artifacts["casebase_path"])), target_range=target_range)
    ph_casebase_path = artifacts.get("ph_casebase_path")
    ph_casebase = None
    if ph_casebase_path:
        ph_casebase = load_stage_casebase(resolve(str(ph_casebase_path)), target_range=target_range)
    stage_policy = load_stage_policy(resolve(str(artifacts["stage_policy_path"])))

    policy = stage_policy.get("policy", {})
    recommendations = policy.get("recommendations", {})
    if isinstance(recommendations, dict):
        ph_enabled_stages = [
            {"stage_name": stage_name, **item}
            for stage_name, item in recommendations.items()
            if isinstance(item, dict) and item.get("enable_ph")
        ]
    else:
        ph_enabled_stages = [item for item in recommendations if item.get("enable_ph")]

    template_row = None
    selected_lag = 120
    if ph_enabled_stages and ph_casebase is not None and "stage_name" in ph_casebase.columns:
        preferred_stage = str(ph_enabled_stages[0]["stage_name"])
        selected_lag = int(ph_enabled_stages[0].get("best_lag_minutes", 120))
        candidates = ph_casebase.loc[ph_casebase["stage_name"] == preferred_stage]
        if not candidates.empty:
            template_row = candidates.iloc[-1]
    if template_row is None:
        template_row = base_casebase.iloc[-1]

    runtime_time = pd.Timestamp(template_row.get("sample_time"))
    dcs_window = _build_dcs_window_from_template(
        template_row,
        sensors=sensors,
        window_minutes=window_minutes,
        runtime_time=runtime_time,
    )

    bundle = {
        "dcs_window": dcs_window,
        "runtime_time": str(runtime_time),
    }
    if all(column in template_row.index for column in DEFAULT_PH_FEATURE_COLUMNS):
        bundle["ph_history"] = _build_ph_history_from_template(
            template_row,
            runtime_time=runtime_time,
            lag_minutes=selected_lag,
            feature_window_minutes=int(stage_policy.get("ph", {}).get("feature_window_minutes", 50)),
        )
    return bundle
