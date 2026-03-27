from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core import (
    get_context_sensors,
    load_runtime_config,
    load_stage_aware_example_bundle,
    load_stage_casebase,
    load_stage_policy,
)
from core.stage_aware_recommender import assign_stage_from_policy, extract_current_ph_features
from core.window_encoder import encode_dcs_window


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_PH_FEATURE_COLUMNS = ["ph_point", "ph_mean", "ph_std", "ph_min", "ph_max", "ph_delta"]


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _load_assets(config_path: Path) -> tuple[dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame]:
    config = load_runtime_config(config_path)
    artifacts = config.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("`artifacts` must be a mapping in runtime config.")
    base_casebase = load_stage_casebase(PROJECT_DIR / str(artifacts["casebase_path"]))
    ph_casebase = load_stage_casebase(PROJECT_DIR / str(artifacts["ph_casebase_path"]))
    stage_policy = load_stage_policy(PROJECT_DIR / str(artifacts["stage_policy_path"]))
    return config, stage_policy, base_casebase, ph_casebase


def _select_dcs_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include=["number"]).columns
        if "__" in column
    ]


def _prepare_alert_frame(base_casebase: pd.DataFrame, ph_casebase: pd.DataFrame) -> pd.DataFrame:
    base = base_casebase.copy()
    if "sample_time" in base.columns:
        base["sample_time"] = pd.to_datetime(base["sample_time"], errors="coerce")
    base = base.sort_values("sample_time").reset_index(drop=True)
    base["is_out_of_spec"] = 1 - pd.to_numeric(base["is_in_spec"], errors="coerce").fillna(0).astype(int)

    ph_columns = ["sample_time", *DEFAULT_PH_FEATURE_COLUMNS]
    ph_frame = ph_casebase.copy()
    if "sample_time" in ph_frame.columns:
        ph_frame["sample_time"] = pd.to_datetime(ph_frame["sample_time"], errors="coerce")
    ph_frame = ph_frame[[column for column in ph_columns if column in ph_frame.columns]].drop_duplicates(subset=["sample_time"])
    merged = base.merge(ph_frame, on="sample_time", how="left")
    return merged.sort_values("sample_time").reset_index(drop=True)


def _fit_binary_ensemble(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, object] | None:
    if frame.empty:
        return None
    train = frame.dropna(subset=["is_out_of_spec"]).copy()
    if train.empty or train["is_out_of_spec"].nunique() < 2:
        return None

    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=4,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                ),
            ),
        ]
    )
    logistic.fit(train[feature_columns], train["is_out_of_spec"].astype(int))
    forest.fit(train[feature_columns], train["is_out_of_spec"].astype(int))
    return {
        "feature_columns": feature_columns,
        "logistic": logistic,
        "forest": forest,
    }


def _predict_binary_ensemble(model_bundle: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    feature_columns = list(model_bundle["feature_columns"])
    X = frame[feature_columns]
    p1 = model_bundle["logistic"].predict_proba(X)[:, 1]
    p2 = model_bundle["forest"].predict_proba(X)[:, 1]
    return (p1 + p2) / 2.0


def _build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def fit_global_logistic_alert_model(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    resolved_config_path = Path(config_path)
    config, stage_policy, base_casebase, ph_casebase = _load_assets(resolved_config_path)
    frame = _prepare_alert_frame(base_casebase, ph_casebase)
    dcs_features = _select_dcs_feature_columns(frame)
    train = frame.dropna(subset=["is_out_of_spec"]).copy()
    if train.empty or train["is_out_of_spec"].nunique() < 2:
        raise ValueError("Global logistic alert model could not be fitted.")

    model = _build_logistic_pipeline()
    model.fit(train[dcs_features], train["is_out_of_spec"].astype(int))
    return {
        "config_path": str(resolved_config_path),
        "config": config,
        "stage_policy": stage_policy,
        "frame": frame,
        "dcs_features": dcs_features,
        "model_name": "logistic_balanced",
        "model": model,
    }


def predict_global_logistic_probability(model_bundle: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    feature_columns = list(model_bundle["dcs_features"])
    return model_bundle["model"].predict_proba(frame[feature_columns])[:, 1]


def fit_oos_alert_models(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    resolved_config_path = Path(config_path)
    config, stage_policy, base_casebase, ph_casebase = _load_assets(resolved_config_path)
    frame = _prepare_alert_frame(base_casebase, ph_casebase)
    dcs_features = _select_dcs_feature_columns(frame)

    global_dcs_model = _fit_binary_ensemble(frame, dcs_features)
    stage_models: dict[str, dict[str, object]] = {}
    recommendations = stage_policy["policy"]["recommendations"]
    for stage_name, stage_frame in frame.groupby("stage_name"):
        stage_bundle: dict[str, object] = {"dcs_model": None, "ph_model": None}
        stage_bundle["dcs_model"] = _fit_binary_ensemble(stage_frame, dcs_features)
        stage_decision = recommendations.get(stage_name, {})
        ph_enabled = bool(stage_decision.get("enable_ph", False))
        if ph_enabled:
            ph_stage = stage_frame.dropna(subset=DEFAULT_PH_FEATURE_COLUMNS).copy()
            if not ph_stage.empty:
                stage_bundle["ph_model"] = _fit_binary_ensemble(ph_stage, dcs_features + DEFAULT_PH_FEATURE_COLUMNS)
        stage_models[stage_name] = stage_bundle

    return {
        "config_path": str(resolved_config_path),
        "config": config,
        "stage_policy": stage_policy,
        "frame": frame,
        "dcs_features": dcs_features,
        "global_dcs_model": global_dcs_model,
        "stage_models": stage_models,
    }


def _predict_stage_aware_probability(
    current_row: pd.DataFrame,
    *,
    stage_name: str,
    stage_policy: dict[str, object],
    stage_models: dict[str, dict[str, object]],
    global_dcs_model: dict[str, object] | None,
) -> tuple[float | None, dict[str, object]]:
    stage_decision = stage_policy["policy"]["recommendations"].get(stage_name, {})
    stage_bundle = stage_models.get(stage_name, {})
    ph_model = stage_bundle.get("ph_model")
    dcs_model = stage_bundle.get("dcs_model")
    ph_available = current_row[DEFAULT_PH_FEATURE_COLUMNS].notna().all(axis=1).iloc[0]

    if bool(stage_decision.get("enable_ph", False)) and ph_available and ph_model is not None:
        probability = float(_predict_binary_ensemble(ph_model, current_row)[0])
        return probability, {
            "selected_model": "stage_ph_model",
            "ph_policy_enabled": True,
            "ph_used": True,
            "ph_fallback_to_dcs_only": False,
        }

    if dcs_model is not None:
        probability = float(_predict_binary_ensemble(dcs_model, current_row)[0])
        return probability, {
            "selected_model": "stage_dcs_model",
            "ph_policy_enabled": bool(stage_decision.get("enable_ph", False)),
            "ph_used": False,
            "ph_fallback_to_dcs_only": bool(stage_decision.get("enable_ph", False)),
        }

    if global_dcs_model is not None:
        probability = float(_predict_binary_ensemble(global_dcs_model, current_row)[0])
        return probability, {
            "selected_model": "global_dcs_model",
            "ph_policy_enabled": bool(stage_decision.get("enable_ph", False)),
            "ph_used": False,
            "ph_fallback_to_dcs_only": True,
        }

    return None, {
        "selected_model": "unavailable",
        "ph_policy_enabled": bool(stage_decision.get("enable_ph", False)),
        "ph_used": False,
        "ph_fallback_to_dcs_only": False,
    }


def predict_t90_oos_alert_dev(
    input_data: dict[str, object] | None = None,
    *,
    model_bundle: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = input_data or {}
    if model_bundle is None:
        model_bundle = fit_oos_alert_models(payload.get("config_path", DEFAULT_CONFIG_PATH))

    config = model_bundle["config"]
    stage_policy = model_bundle["stage_policy"]
    sensors = get_context_sensors(config)

    dcs_window = payload.get("dcs_window")
    ph_history = payload.get("ph_history")
    runtime_time = payload.get("runtime_time")
    if dcs_window is None:
        example_bundle = load_stage_aware_example_bundle(payload.get("config_path", DEFAULT_CONFIG_PATH), project_dir=PROJECT_DIR)
        dcs_window = example_bundle["dcs_window"]
        ph_history = example_bundle.get("ph_history") if ph_history is None else ph_history
        runtime_time = example_bundle.get("runtime_time") if runtime_time is None else runtime_time

    current_context = encode_dcs_window(dcs_window, include_sensors=sensors)
    stage_assignment = assign_stage_from_policy(current_context, stage_policy)
    stage_name = str(stage_assignment["stage_name"])

    current_row = pd.DataFrame([{feature: current_context.get(feature, np.nan) for feature in model_bundle["dcs_features"]}])
    current_row["stage_name"] = stage_name
    current_row["sample_time"] = runtime_time
    for column in DEFAULT_PH_FEATURE_COLUMNS:
        current_row[column] = np.nan

    warnings: list[str] = []
    stage_decision = stage_policy["policy"]["recommendations"].get(stage_name, {})
    if ph_history is not None and bool(stage_decision.get("enable_ph", False)):
        ph_config = config.get("ph", {})
        if not isinstance(ph_config, dict):
            ph_config = {}
        ph_features = extract_current_ph_features(
            ph_history,
            runtime_time=pd.Timestamp(runtime_time),
            lag_minutes=int(stage_decision.get("best_lag_minutes", 0)),
            feature_window_minutes=int(ph_config.get("feature_window_minutes", 50)),
            tolerance_minutes=int(ph_config.get("tolerance_minutes", 2)),
            time_column=str(ph_config.get("history_time_column", "time")),
            value_column=str(ph_config.get("history_value_column", "value")),
        )
        if ph_features is not None:
            for column in DEFAULT_PH_FEATURE_COLUMNS:
                current_row[column] = ph_features.get(column, np.nan)
        else:
            warnings.append("PH history was provided, but the runtime PH feature alignment failed. Falling back to the DCS-only alert path.")
    elif ph_history is None and bool(stage_decision.get("enable_ph", False)):
        warnings.append("PH is optional and was not provided. The alert module will use the DCS-only path for the current stage.")

    probability, selection = _predict_stage_aware_probability(
        current_row,
        stage_name=stage_name,
        stage_policy=stage_policy,
        stage_models=model_bundle["stage_models"],
        global_dcs_model=model_bundle["global_dcs_model"],
    )
    threshold = float(payload.get("alert_threshold", 0.20))
    predicted_alert = None if probability is None else bool(probability >= threshold)

    return {
        "runtime_context": {
            "runtime_time": str(runtime_time),
            "window_rows": int(len(dcs_window)),
            "ph_history_provided": bool(ph_history is not None),
        },
        "stage_decision": {
            "current_stage_name": stage_name,
            "current_stage_distance": float(stage_assignment["stage_distance"]),
            "stage_distances": stage_assignment["stage_distances"],
            "ph_optional_input": True,
            "ph_policy_enabled": bool(stage_decision.get("enable_ph", False)),
            "selected_ph_lag_minutes": int(stage_decision["best_lag_minutes"]) if bool(stage_decision.get("enable_ph", False)) else None,
        },
        "alert": {
            "target": "T90_out_of_spec",
            "definition": "1 when T90 is outside 8.45 +/- 0.25, otherwise 0",
            "probability": probability,
            "threshold": threshold,
            "predicted_out_of_spec": predicted_alert,
            "selected_model": selection["selected_model"],
            "ph_used": selection["ph_used"],
            "ph_fallback_to_dcs_only": selection["ph_fallback_to_dcs_only"],
        },
        "warnings": warnings,
    }
