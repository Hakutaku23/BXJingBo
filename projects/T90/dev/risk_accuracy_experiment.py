from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core import get_context_sensors, get_target_range, load_runtime_config, load_stage_casebase, load_stage_policy
from core.casebase import normalize_casebase_frame
from core.online_recommender import (
    _build_control_recommendation,
    _choose_context_columns,
    _find_context_neighborhood,
    _fit_local_control_model,
)
from core.stage_aware_recommender import (
    _prune_empty_numeric_columns,
    _select_stage_casebase,
    assign_stage_from_policy,
    extract_current_ph_features,
)
from core.window_encoder import encode_dcs_window
from ph_lag_experiment import load_ph_data
from test import build_windows_and_outcomes, load_dcs_data, load_lims_grouped


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
BLOCKED_COLUMNS = {"sample_time", "t90", "is_in_spec", "calcium", "bromine", "stage_id", "stage_name"}


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _load_runtime_assets(config_path: Path) -> tuple[dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame]:
    config = load_runtime_config(config_path)
    artifacts = config.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("`artifacts` must be a mapping in runtime config.")

    base_casebase = load_stage_casebase(PROJECT_DIR / str(artifacts["casebase_path"]), target_range=get_target_range(config))
    ph_casebase = load_stage_casebase(PROJECT_DIR / str(artifacts["ph_casebase_path"]), target_range=get_target_range(config))
    stage_policy = load_stage_policy(PROJECT_DIR / str(artifacts["stage_policy_path"]))
    return config, stage_policy, base_casebase, ph_casebase


def _load_private_sources(config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame], pd.DataFrame]:
    sensors = get_context_sensors(config)
    window_minutes = int(config.get("window", {}).get("minutes", 50))
    time_column = str(config.get("window", {}).get("time_column", "time"))
    data_sources = config.get("data_sources", {})
    dcs = load_dcs_data(data_sources.get("dcs_paths", []), sensors=sensors, time_column=time_column)
    lims = load_lims_grouped(PROJECT_DIR / str(data_sources["lims_path"]))
    ph = load_ph_data(PROJECT_DIR / str(data_sources["ph_path"]))
    windows, outcomes = build_windows_and_outcomes(dcs, lims, window_minutes, time_column=time_column)
    return dcs, lims, ph, windows, outcomes


def _status_label(t90: float, low: float, high: float) -> str:
    if t90 < low:
        return "below_spec"
    if t90 > high:
        return "above_spec"
    return "in_spec"


def _extract_recommendation_errors(recommendation: dict[str, object], actual_calcium: float, actual_bromine: float) -> dict[str, object]:
    best_point = recommendation.get("best_point", {})
    calcium_range = recommendation.get("recommended_calcium_range", {})
    bromine_range = recommendation.get("recommended_bromine_range", {})

    best_calcium = _safe_float(best_point.get("calcium"))
    best_bromine = _safe_float(best_point.get("bromine"))
    calcium_min = _safe_float(calcium_range.get("min"))
    calcium_max = _safe_float(calcium_range.get("max"))
    bromine_min = _safe_float(bromine_range.get("min"))
    bromine_max = _safe_float(bromine_range.get("max"))

    return {
        "recommended_best_calcium": best_calcium,
        "recommended_best_bromine": best_bromine,
        "calcium_abs_error_to_best": None if best_calcium is None else abs(best_calcium - actual_calcium),
        "bromine_abs_error_to_best": None if best_bromine is None else abs(best_bromine - actual_bromine),
        "actual_calcium_inside_range": None if calcium_min is None or calcium_max is None else calcium_min <= actual_calcium <= calcium_max,
        "actual_bromine_inside_range": None if bromine_min is None or bromine_max is None else bromine_min <= actual_bromine <= bromine_max,
        "recommended_calcium_min": calcium_min,
        "recommended_calcium_max": calcium_max,
        "recommended_bromine_min": bromine_min,
        "recommended_bromine_max": bromine_max,
    }


def _threshold_scan(actual: pd.Series, score: pd.Series, thresholds: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        predicted = (score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(actual, predicted, zero_division=0)),
                "recall": float(recall_score(actual, predicted, zero_division=0)),
                "f1": float(f1_score(actual, predicted, zero_division=0)),
                "predicted_positive_ratio": float(predicted.mean()),
            }
        )
    return rows


def _summarize_accuracy(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"samples": 0}
    calcium_error = pd.to_numeric(frame["calcium_abs_error_to_best"], errors="coerce").dropna()
    bromine_error = pd.to_numeric(frame["bromine_abs_error_to_best"], errors="coerce").dropna()
    calcium_inside = pd.to_numeric(frame["actual_calcium_inside_range"], errors="coerce").dropna()
    bromine_inside = pd.to_numeric(frame["actual_bromine_inside_range"], errors="coerce").dropna()
    return {
        "samples": int(len(frame)),
        "calcium_mae": None if calcium_error.empty else float(calcium_error.mean()),
        "calcium_p90": None if calcium_error.empty else float(calcium_error.quantile(0.90)),
        "bromine_mae": None if bromine_error.empty else float(bromine_error.mean()),
        "bromine_p90": None if bromine_error.empty else float(bromine_error.quantile(0.90)),
        "calcium_range_coverage": None if calcium_inside.empty else float(calcium_inside.mean()),
        "bromine_range_coverage": None if bromine_inside.empty else float(bromine_inside.mean()),
    }


def _run_single_sample(
    *,
    dcs_window: pd.DataFrame,
    runtime_time: pd.Timestamp,
    actual_t90: float,
    actual_calcium: float,
    actual_bromine: float,
    sensors: list[str],
    target_low: float,
    target_high: float,
    stage_policy: dict[str, object],
    base_casebase: pd.DataFrame,
    ph_casebase: pd.DataFrame,
    ph_history: pd.DataFrame,
    ph_config: dict[str, object],
    neighbor_count: int,
    local_neighbor_count: int,
    probability_threshold: float,
    grid_points: int,
) -> dict[str, object]:
    current_context = encode_dcs_window(dcs_window, include_sensors=sensors)
    stage_assignment = assign_stage_from_policy(current_context, stage_policy)
    stage_name = str(stage_assignment["stage_name"])
    stage_decision = stage_policy["policy"]["recommendations"][stage_name]
    ph_policy_enabled = bool(stage_decision["enable_ph"])
    ph_history_for_runtime = ph_history.loc[ph_history["time"] <= runtime_time].copy()
    current_ph_features = None
    selected_casebase = _select_stage_casebase(base_casebase, stage_name)
    casebase_source = "base_casebase"
    ph_fallback = False

    if ph_policy_enabled:
        current_ph_features = extract_current_ph_features(
            ph_history_for_runtime,
            runtime_time=runtime_time,
            lag_minutes=int(stage_decision["best_lag_minutes"]),
            feature_window_minutes=int(ph_config.get("feature_window_minutes", 50)),
            tolerance_minutes=int(ph_config.get("tolerance_minutes", 2)),
            time_column=str(ph_config.get("history_time_column", "time")),
            value_column=str(ph_config.get("history_value_column", "value")),
        )
        if current_ph_features is not None:
            current_context.update({key: current_ph_features[key] for key in ("ph_point", "ph_mean", "ph_std", "ph_min", "ph_max", "ph_delta")})
            selected_casebase = _select_stage_casebase(ph_casebase, stage_name)
            casebase_source = "ph_casebase"
        else:
            ph_fallback = True

    prepared_casebase = normalize_casebase_frame(selected_casebase, target_low=target_low, target_high=target_high)
    prepared_casebase = _prune_empty_numeric_columns(prepared_casebase)
    include_columns = tuple(
        column
        for column in prepared_casebase.select_dtypes(include=["number"]).columns
        if column not in BLOCKED_COLUMNS and column in current_context
    )
    context_columns = _choose_context_columns(prepared_casebase, current_context, include_columns)
    neighborhood = _find_context_neighborhood(prepared_casebase, current_context, context_columns, neighbor_count)
    local, model = _fit_local_control_model(neighborhood, local_neighbor_count)
    recommendation = _build_control_recommendation(
        local,
        model,
        reference_calcium=None,
        reference_bromine=None,
        probability_threshold=probability_threshold,
        grid_points=grid_points,
    )
    risk_neighborhood = float((pd.to_numeric(neighborhood["t90"], errors="coerce") > target_high).mean())
    risk_local = float((pd.to_numeric(local["t90"], errors="coerce") > target_high).mean())
    recommendation_error = _extract_recommendation_errors(recommendation, actual_calcium, actual_bromine)

    return {
        "sample_time": str(runtime_time),
        "stage_name": stage_name,
        "stage_distance": float(stage_assignment["stage_distance"]),
        "t90": float(actual_t90),
        "t90_status": _status_label(float(actual_t90), target_low, target_high),
        "actual_above_spec": bool(float(actual_t90) > target_high),
        "actual_calcium": float(actual_calcium),
        "actual_bromine": float(actual_bromine),
        "ph_history_provided": True,
        "ph_policy_enabled": ph_policy_enabled,
        "ph_used": bool(current_ph_features is not None),
        "ph_fallback_to_dcs_only": ph_fallback,
        "selected_ph_lag_minutes": None if current_ph_features is None else int(stage_decision["best_lag_minutes"]),
        "selected_casebase_source": casebase_source,
        "neighborhood_size": int(len(neighborhood)),
        "local_size": int(len(local)),
        "over_spec_risk_neighborhood": risk_neighborhood,
        "over_spec_risk_local": risk_local,
        "neighborhood_t90_p90": float(pd.to_numeric(neighborhood["t90"], errors="coerce").quantile(0.90)),
        "local_t90_p90": float(pd.to_numeric(local["t90"], errors="coerce").quantile(0.90)),
        "best_point_probability": _safe_float(recommendation.get("best_point", {}).get("in_spec_probability")),
        **recommendation_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v2 over-spec risk detection and condition-specific recommendation accuracy.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--neighbor-count", type=int, default=150, help="Neighborhood size used for replay.")
    parser.add_argument("--local-neighbor-count", type=int, default=80, help="Local neighborhood size used for replay.")
    parser.add_argument("--probability-threshold", type=float, default=0.60, help="Probability threshold used for recommendation feasibility.")
    parser.add_argument("--grid-points", type=int, default=31, help="Grid size used for calcium/bromine recommendation search.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on replayed samples. 0 means all samples.")
    args = parser.parse_args()

    config_path = Path(args.config)
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "risk_accuracy_experiment_results.csv"
    json_path = results_dir / "risk_accuracy_experiment_summary.json"

    config, stage_policy, base_casebase, ph_casebase = _load_runtime_assets(config_path)
    _dcs, _lims, ph, windows, outcomes = _load_private_sources(config)
    sensors = get_context_sensors(config)
    target_low, target_high = get_target_range(config)
    ph_config = config.get("ph", {})
    if not isinstance(ph_config, dict):
        ph_config = {}

    total = len(outcomes) if args.limit <= 0 else min(args.limit, len(outcomes))
    rows: list[dict[str, object]] = []
    for index in range(total):
        if index and index % 100 == 0:
            print(f"replayed {index}/{total} samples")
        row = outcomes.iloc[index]
        rows.append(
            _run_single_sample(
                dcs_window=windows[index],
                runtime_time=pd.Timestamp(row["sample_time"]),
                actual_t90=float(row["t90"]),
                actual_calcium=float(row["calcium"]),
                actual_bromine=float(row["bromine"]),
                sensors=sensors,
                target_low=target_low,
                target_high=target_high,
                stage_policy=stage_policy,
                base_casebase=base_casebase,
                ph_casebase=ph_casebase,
                ph_history=ph,
                ph_config=ph_config,
                neighbor_count=args.neighbor_count,
                local_neighbor_count=args.local_neighbor_count,
                probability_threshold=args.probability_threshold,
                grid_points=args.grid_points,
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")

    actual = results["actual_above_spec"].astype(int)
    score = pd.to_numeric(results["over_spec_risk_local"], errors="coerce").fillna(0.0)
    thresholds = [round(item, 2) for item in np.arange(0.10, 0.91, 0.10)]
    threshold_rows = _threshold_scan(actual, score, thresholds)
    best_threshold = max(threshold_rows, key=lambda item: item["f1"]) if threshold_rows else None

    overall_risk = {
        "samples": int(len(results)),
        "actual_above_spec_ratio": float(actual.mean()),
        "roc_auc_local_over_spec_risk": float(roc_auc_score(actual, score)),
        "average_precision_local_over_spec_risk": float(average_precision_score(actual, score)),
        "best_f1_threshold": None if best_threshold is None else float(best_threshold["threshold"]),
        "threshold_scan": threshold_rows,
    }

    stage_risk_rows: list[dict[str, object]] = []
    for stage_name, frame in results.groupby("stage_name"):
        actual_stage = frame["actual_above_spec"].astype(int)
        score_stage = pd.to_numeric(frame["over_spec_risk_local"], errors="coerce").fillna(0.0)
        row = {
            "stage_name": stage_name,
            "samples": int(len(frame)),
            "actual_above_spec_ratio": float(actual_stage.mean()),
            "mean_risk_on_above_spec": None,
            "mean_risk_on_non_above_spec": None,
            "roc_auc_local_over_spec_risk": None,
            "average_precision_local_over_spec_risk": None,
        }
        if (actual_stage == 1).any():
            row["mean_risk_on_above_spec"] = float(score_stage.loc[actual_stage == 1].mean())
        if (actual_stage == 0).any():
            row["mean_risk_on_non_above_spec"] = float(score_stage.loc[actual_stage == 0].mean())
        if actual_stage.nunique() > 1:
            row["roc_auc_local_over_spec_risk"] = float(roc_auc_score(actual_stage, score_stage))
            row["average_precision_local_over_spec_risk"] = float(average_precision_score(actual_stage, score_stage))
        stage_risk_rows.append(row)

    by_status = {
        status: _summarize_accuracy(frame)
        for status, frame in results.groupby("t90_status")
    }
    by_stage = {
        stage_name: _summarize_accuracy(frame)
        for stage_name, frame in results.groupby("stage_name")
    }
    ph_comparison = {
        "ph_used": _summarize_accuracy(results.loc[results["ph_used"]]),
        "ph_fallback_to_dcs_only": _summarize_accuracy(results.loc[results["ph_fallback_to_dcs_only"]]),
    }

    summary = {
        "target_range": {"low": float(target_low), "high": float(target_high)},
        "current_v2_policy": {
            "window_minutes": int(config.get("window", {}).get("minutes", 50)),
            "stage_count": int(stage_policy["stage_identifier"]["stage_count"]),
            "ph_optional_input": True,
            "ph_policy": stage_policy["policy"]["recommendations"],
        },
        "over_spec_risk_detection": {
            "question_answer": "yes_probabilistic",
            "interpretation": "The current implementation can infer whether T90 is likely to exceed the upper spec as a risk score, not as an exact T90 prediction.",
            "overall": overall_risk,
            "by_stage": stage_risk_rows,
        },
        "recommendation_accuracy": {
            "interpretation": "Accuracy is evaluated by comparing the recommended calcium/bromine best point and recommended ranges against the true LIMS calcium/bromine values in offline replay.",
            "overall": _summarize_accuracy(results),
            "by_t90_status": by_status,
            "by_stage": by_stage,
            "by_ph_runtime_path": ph_comparison,
        },
        "artifacts": {
            "results_csv": str(csv_path),
            "summary_json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
