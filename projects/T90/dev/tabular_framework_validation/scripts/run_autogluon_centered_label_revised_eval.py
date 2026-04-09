from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_stage1_lag_scale import build_label_frame
from run_autogluon_stage1_quickcheck import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    regression_metrics,
    resolve_path,
)
from run_autogluon_stage2_dynamic_morphology import build_stage2_table
from run_autogluon_stage5_quality import build_quality_table
from run_autogluon_stage6_centered_quality import build_centered_quality_table
from run_autogluon_stage7_final_selection import (
    build_current_task_frame,
    combo_specs,
    load_references,
    select_task_priors,
)
from run_autogluon_centered_desirability_outspec_eval import load_best_centered_combo


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_centered_label_revised_eval.yaml"


def centered_point_label(y_obs: np.ndarray, center: float, tolerance: float) -> np.ndarray:
    return np.maximum(0.0, 1.0 - np.abs(y_obs - center) / tolerance)


def centered_uncertain_label(
    y_obs: np.ndarray,
    center: float,
    tolerance: float,
    err_half_width: float,
    n_grid: int,
) -> np.ndarray:
    delta_grid = np.linspace(-err_half_width, err_half_width, n_grid)
    values = []
    for delta in delta_grid:
        values.append(centered_point_label(y_obs + delta, center, tolerance))
    return np.mean(np.vstack(values), axis=0)


def centered_gaussian_label(y_obs: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-((y_obs - center) ** 2) / (2.0 * sigma ** 2))


def compute_center_band_prob(y_obs: np.ndarray, low: float, high: float, err_half_width: float) -> np.ndarray:
    left = np.maximum(y_obs - err_half_width, low)
    right = np.minimum(y_obs + err_half_width, high)
    overlap = np.maximum(0.0, right - left)
    return overlap / (2.0 * err_half_width)


def compute_outspec_prob(y_obs: np.ndarray, spec_low: float, spec_high: float, err_half_width: float) -> np.ndarray:
    low_part = np.maximum(0.0, np.minimum(spec_low, y_obs + err_half_width) - (y_obs - err_half_width))
    high_part = np.maximum(0.0, (y_obs + err_half_width) - np.maximum(spec_high, y_obs - err_half_width))
    return (low_part + high_part) / (2.0 * err_half_width)


def build_label_series(
    method_name: str,
    t90_values: np.ndarray,
    center: float,
    tolerance: float,
    method_param: float | None,
    uncertain_grid_points: int,
) -> np.ndarray:
    if method_name == "point":
        return centered_point_label(t90_values, center, tolerance)
    if method_name == "uncertain":
        if method_param is None:
            raise ValueError("uncertain label requires err_half_width.")
        return centered_uncertain_label(t90_values, center, tolerance, float(method_param), uncertain_grid_points)
    if method_name == "gaussian":
        if method_param is None:
            raise ValueError("gaussian label requires sigma.")
        return centered_gaussian_label(t90_values, center, float(method_param))
    raise ValueError(f"Unknown method_name: {method_name}")


def aligned_hard_out_metrics(in_spec_flag: np.ndarray, pred_desirability: np.ndarray) -> dict[str, float]:
    hard_out_flag = 1 - in_spec_flag.astype(int)
    hard_out_risk = 1.0 - np.clip(pred_desirability.astype(float), 0.0, 1.0)
    if len(np.unique(hard_out_flag)) > 1:
        ap = float(average_precision_score(hard_out_flag, hard_out_risk))
        auc = float(roc_auc_score(hard_out_flag, hard_out_risk))
    else:
        ap = float("nan")
        auc = float("nan")
    return {
        "hard_out_ap_aligned": ap,
        "hard_out_auc_aligned": auc,
    }


def choose_candidate_from_ablation(candidate_rows: list[dict[str, Any]], primary_metric: str) -> dict[str, Any]:
    if primary_metric != "mae":
        raise ValueError("Only mae-based ablation selection is supported.")
    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            row["autogluon_mean_mae"],
            -row["autogluon_mean_hard_out_ap_aligned"],
            str(row["method_param"]),
        ),
    )
    return ranked[0]


def approximate_drop_duplicate_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    kept: list[str] = []
    seen: set[int] = set()
    for column in columns:
        series = frame[column]
        hashed = pd.util.hash_pandas_object(series.astype(str), index=False).sum()
        if int(hashed) in seen:
            continue
        seen.add(int(hashed))
        kept.append(column)
    return kept


def preclean_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    max_missing_ratio: float,
    unique_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    keep: list[str] = []
    for column in train_x.columns:
        series = train_x[column]
        if float(series.isna().mean()) > max_missing_ratio:
            continue
        if int(series.nunique(dropna=False)) <= unique_threshold:
            continue
        keep.append(column)
    keep = approximate_drop_duplicate_columns(train_x, keep)
    return train_x[keep].copy(), test_x[keep].copy(), keep


def supervised_select(train_x: pd.DataFrame, y: np.ndarray, top_k: int) -> list[str]:
    if train_x.empty:
        return []
    numeric = train_x.copy()
    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce")
        if series.notna().any():
            numeric[column] = series.fillna(series.median())
        else:
            numeric[column] = 0.0
    from sklearn.feature_selection import mutual_info_regression

    try:
        scores = mutual_info_regression(numeric, y, random_state=42)
    except Exception:
        return list(train_x.columns[: min(top_k, train_x.shape[1])])
    score_frame = pd.DataFrame({"feature": train_x.columns, "score": scores})
    score_frame = score_frame.sort_values(["score", "feature"], ascending=[False, True]).reset_index(drop=True)
    return score_frame["feature"].head(min(top_k, len(score_frame))).tolist()


def compute_cqdi_components(
    preferred_rate: float,
    center_prob_mean_preferred: float,
    outspec_prob_mean_preferred: float,
    center_prob_base: float,
    hard_out_ap_aligned: float,
    cqdi_cfg: dict[str, Any],
) -> dict[str, float]:
    center_lift = center_prob_mean_preferred / max(center_prob_base, 1e-6)
    center_score = min(1.0, center_prob_mean_preferred / float(cqdi_cfg["center_prob_target"]))
    lift_score = min(1.0, center_lift / float(cqdi_cfg["center_lift_target"]))
    safety_score = min(1.0, float(cqdi_cfg["safety_outspec_target"]) / max(outspec_prob_mean_preferred, 1e-6))
    coverage_score = float(
        np.clip(
            1.0 - abs(preferred_rate - float(cqdi_cfg["preferred_rate_target"])) / float(cqdi_cfg["preferred_rate_half_width"]),
            0.0,
            1.0,
        )
    )
    return {
        "center_lift": center_lift,
        "center_score": center_score,
        "lift_score": lift_score,
        "safety_score": safety_score,
        "coverage_score": coverage_score,
        "hard_out_ap_aligned": hard_out_ap_aligned,
    }


def threshold_candidate_rows(
    scored: pd.DataFrame,
    preferred_threshold_grid: list[float],
    cqdi_cfg: dict[str, Any],
    ap_floor: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    center_prob_base = float(scored["true_center_band_prob"].mean())
    hard_out_ap_aligned = float(average_precision_score(scored["true_outspec_obs"], scored["pred_outspec_risk_aligned"]))
    for threshold in preferred_threshold_grid:
        preferred_flag = (scored["pred_desirability_aligned"] >= float(threshold)).astype(int)
        preferred_rate = float(preferred_flag.mean())
        if preferred_flag.sum() == 0:
            center_prob_mean_preferred = 0.0
            outspec_prob_mean_preferred = 1.0
        else:
            center_prob_mean_preferred = float(scored.loc[preferred_flag == 1, "true_center_band_prob"].mean())
            outspec_prob_mean_preferred = float(scored.loc[preferred_flag == 1, "true_outspec_prob"].mean())
        components = compute_cqdi_components(
            preferred_rate=preferred_rate,
            center_prob_mean_preferred=center_prob_mean_preferred,
            outspec_prob_mean_preferred=outspec_prob_mean_preferred,
            center_prob_base=center_prob_base,
            hard_out_ap_aligned=hard_out_ap_aligned,
            cqdi_cfg=cqdi_cfg,
        )
        gate_pass = int(
            hard_out_ap_aligned >= ap_floor
            and outspec_prob_mean_preferred <= float(cqdi_cfg["gate_outspec_limit"])
            and float(cqdi_cfg["gate_preferred_rate_low"]) <= preferred_rate <= float(cqdi_cfg["gate_preferred_rate_high"])
        )
        cqdi_train_candidate = 100.0 * gate_pass * (
            0.40 * components["center_score"]
            + 0.25 * components["lift_score"]
            + 0.20 * components["safety_score"]
            + 0.15 * components["coverage_score"]
        )
        rows.append(
            {
                "inner_threshold_pref": float(threshold),
                "train_preferred_rate": preferred_rate,
                "train_center_prob_mean_preferred": center_prob_mean_preferred,
                "train_outspec_prob_mean_preferred": outspec_prob_mean_preferred,
                "train_center_prob_base": center_prob_base,
                "train_center_lift": components["center_lift"],
                "train_hard_out_ap_aligned": hard_out_ap_aligned,
                "train_gate_pass": gate_pass,
                "cqdi_train_candidate": cqdi_train_candidate,
            }
        )
    return rows


def choose_threshold(threshold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        threshold_rows,
        key=lambda row: (
            -row["cqdi_train_candidate"],
            -row["train_gate_pass"],
            -row["train_center_prob_mean_preferred"],
            row["train_outspec_prob_mean_preferred"],
            abs(row["train_preferred_rate"] - 0.20),
            row["inner_threshold_pref"],
        ),
    )
    return ranked[0]


def summarize_fold_cqdi(
    scored: pd.DataFrame,
    preferred_threshold: float,
    cqdi_cfg: dict[str, Any],
    ap_floor: float,
) -> dict[str, Any]:
    preferred_flag = (scored["pred_desirability_aligned"] >= float(preferred_threshold)).astype(int)
    preferred_rate = float(preferred_flag.mean())
    center_prob_base = float(scored["true_center_band_prob"].mean())
    if preferred_flag.sum() == 0:
        center_prob_mean_preferred = 0.0
        outspec_prob_mean_preferred = 1.0
    else:
        center_prob_mean_preferred = float(scored.loc[preferred_flag == 1, "true_center_band_prob"].mean())
        outspec_prob_mean_preferred = float(scored.loc[preferred_flag == 1, "true_outspec_prob"].mean())
    hard_out_ap_aligned = float(average_precision_score(scored["true_outspec_obs"], scored["pred_outspec_risk_aligned"]))
    hard_out_auc_aligned = float(roc_auc_score(scored["true_outspec_obs"], scored["pred_outspec_risk_aligned"]))
    components = compute_cqdi_components(
        preferred_rate=preferred_rate,
        center_prob_mean_preferred=center_prob_mean_preferred,
        outspec_prob_mean_preferred=outspec_prob_mean_preferred,
        center_prob_base=center_prob_base,
        hard_out_ap_aligned=hard_out_ap_aligned,
        cqdi_cfg=cqdi_cfg,
    )
    gate_pass = int(
        hard_out_ap_aligned >= ap_floor
        and outspec_prob_mean_preferred <= float(cqdi_cfg["gate_outspec_limit"])
        and float(cqdi_cfg["gate_preferred_rate_low"]) <= preferred_rate <= float(cqdi_cfg["gate_preferred_rate_high"])
    )
    cqdi_fold = 100.0 * gate_pass * (
        0.40 * components["center_score"]
        + 0.25 * components["lift_score"]
        + 0.20 * components["safety_score"]
        + 0.15 * components["coverage_score"]
    )
    if gate_pass == 0 or cqdi_fold < 60.0:
        status = "FAIL"
    elif cqdi_fold >= 80.0:
        status = "READY"
    else:
        status = "CANARY"
    return {
        "test_preferred_rate": preferred_rate,
        "test_center_prob_mean_preferred": center_prob_mean_preferred,
        "test_outspec_prob_mean_preferred": outspec_prob_mean_preferred,
        "test_center_prob_base": center_prob_base,
        "test_center_lift": components["center_lift"],
        "test_hard_out_ap_aligned": hard_out_ap_aligned,
        "test_hard_out_auc_aligned": hard_out_auc_aligned,
        "cqdi_fold": cqdi_fold,
        "gate_pass": gate_pass,
        "deployment_status_fold": status,
    }


def build_centered_feature_frame(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    refs = load_references(config_path, config)
    priors = select_task_priors(refs)["centered_desirability"]
    best_combo_summary = load_best_centered_combo(resolve_path(config_path.parent, config["paths"]["stage7_summary_path"]))
    combo = next(spec for spec in combo_specs("centered_desirability", priors) if spec["name"] == str(best_combo_summary["combo_name"]))

    labeled = build_label_frame(config_path, config)
    dcs = load_dcs_frame(
        resolve_path(config_path.parent, config["paths"]["dcs_main_path"]),
        resolve_path(config_path.parent, config["paths"].get("dcs_supplemental_path")),
    )
    snapshot = build_stage2_table(
        labeled_samples=labeled,
        dcs=dcs,
        tau_minutes=int(priors["tau_minutes"]),
        window_minutes=int(priors["window_minutes"]),
        stats=list(config["snapshot"]["stats"]),
        enabled_dynamic_features=[],
        min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
    )
    quality_table = None
    if priors.get("quality_package"):
        quality_table = build_quality_table(
            labeled_samples=labeled,
            dcs=dcs,
            tau_minutes=int(priors["tau_minutes"]),
            window_minutes=int(priors["window_minutes"]),
            enabled_features=list(config["quality_feature_map"][priors["quality_package"]]),
            min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
        )
    centered_table = None
    if priors.get("centered_quality_package"):
        centered_table = build_centered_quality_table(
            labeled_samples=labeled,
            dcs=dcs,
            tau_minutes=int(priors["tau_minutes"]),
            window_minutes=int(priors["window_minutes"]),
            enabled_features=list(config["centered_quality_feature_map"][priors["centered_quality_package"]]),
            min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
        )
    prepared = build_current_task_frame("centered_desirability", snapshot, priors, combo, quality_table, centered_table)
    prepared = prepared.sort_values("sample_time").reset_index(drop=True)
    return prepared, priors, best_combo_summary


def fit_outer_fold_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    top_k: int,
    config: dict[str, Any],
    model_path: Path,
) -> tuple[np.ndarray, dict[str, float], list[str], int, int, int]:
    train_x, test_x = train[feature_columns].copy(), test[feature_columns].copy()
    raw_feature_count = int(train_x.shape[1])
    train_clean, test_clean, cleaned_cols = preclean_features(
        train_x,
        test_x,
        max_missing_ratio=float(config["preclean"]["max_missing_ratio"]),
        unique_threshold=int(config["preclean"]["near_constant_unique_threshold"]),
    )
    cleaned_feature_count = int(len(cleaned_cols))
    selected_cols = supervised_select(train_clean, y_train, int(top_k))
    if not selected_cols:
        selected_cols = cleaned_cols
    selected_feature_count = int(len(selected_cols))
    train_sel = train_clean[selected_cols].copy()
    test_sel = test_clean[selected_cols].copy()

    baseline = make_regression_baseline()
    baseline.fit(train_sel, y_train)
    baseline_pred = baseline.predict(test_sel).astype(float)
    baseline_metric = regression_metrics(y_test, baseline_pred, test["is_in_spec"].to_numpy(dtype=int))

    ag_pred, _ = fit_autogluon_fold(
        train_df=pd.concat([train_sel, pd.DataFrame({"target_centered_custom": y_train})], axis=1),
        test_df=pd.concat([test_sel, pd.DataFrame({"target_centered_custom": y_test})], axis=1),
        label="target_centered_custom",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        model_path=model_path,
        ag_config=config["autogluon"],
    )
    return ag_pred.astype(float), baseline_metric, selected_cols, raw_feature_count, cleaned_feature_count, selected_feature_count


def run_ablation(
    prepared: pd.DataFrame,
    priors: dict[str, Any],
    config: dict[str, Any],
    artifact_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    center = float(config["target_spec"]["center"])
    tolerance = float(config["target_spec"]["tolerance"])
    top_k = int(config["selection"]["top_k"])
    uncertain_grid_points = int(config["label_methods"]["uncertain"]["integration_points"])
    feature_columns = [column for column in prepared.columns if "__" in column]
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["outer_n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    candidate_specs: list[tuple[str, float | None]] = [("point", None)]
    candidate_specs += [("uncertain", float(v)) for v in config["label_methods"]["uncertain"]["err_half_width_grid"]]
    candidate_specs += [("gaussian", float(v)) for v in config["label_methods"]["gaussian"]["sigma_grid"]]

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for method_name, method_param in candidate_specs:
        fold_metrics: list[dict[str, Any]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(prepared), start=1):
            train = prepared.iloc[train_idx].copy().reset_index(drop=True)
            test = prepared.iloc[test_idx].copy().reset_index(drop=True)
            y_train = build_label_series(method_name, train["t90"].to_numpy(dtype=float), center, tolerance, method_param, uncertain_grid_points)
            y_test = build_label_series(method_name, test["t90"].to_numpy(dtype=float), center, tolerance, method_param, uncertain_grid_points)
            model_path = artifact_dir / f"ag_centered_label_ablation_{method_name}_{str(method_param).replace('.', 'p')}_{run_id}_fold{fold_idx}"
            ag_pred, base_metric, _, raw_feature_count, cleaned_feature_count, selected_feature_count = fit_outer_fold_model(
                train, test, feature_columns, y_train, y_test, top_k, config, model_path
            )
            ag_metric = regression_metrics(y_test, ag_pred, test["is_in_spec"].to_numpy(dtype=int))
            aligned = aligned_hard_out_metrics(test["is_in_spec"].to_numpy(dtype=int), ag_pred)
            row = {
                "method_name": method_name,
                "method_param": method_param,
                "fold": int(fold_idx),
                "raw_feature_count": raw_feature_count,
                "cleaned_feature_count": cleaned_feature_count,
                "selected_feature_count": selected_feature_count,
                "baseline_mae": base_metric["mae"],
                "baseline_rmse": base_metric["rmse"],
                "autogluon_mae": ag_metric["mae"],
                "autogluon_rmse": ag_metric["rmse"],
                "autogluon_rank_correlation": ag_metric["rank_correlation"],
                "autogluon_in_spec_auc_from_desirability": ag_metric["in_spec_auc_from_desirability"],
                "autogluon_mean_hard_out_ap_aligned": aligned["hard_out_ap_aligned"],
                "autogluon_mean_hard_out_auc_aligned": aligned["hard_out_auc_aligned"],
            }
            rows.append(row)
            fold_metrics.append(row)
        summaries.append(
            {
                "method_name": method_name,
                "method_param": method_param,
                "top_k": top_k,
                "selected_stage1_variant": str(priors["stage1_variant"]),
                "selected_interaction_package": priors.get("interaction_package"),
                "selected_quality_package": priors.get("quality_package"),
                "selected_centered_quality_package": priors.get("centered_quality_package"),
                "tau_minutes": int(priors["tau_minutes"]),
                "window_minutes": int(priors["window_minutes"]),
                "autogluon_mean_mae": float(np.nanmean([item["autogluon_mae"] for item in fold_metrics])),
                "autogluon_mean_rmse": float(np.nanmean([item["autogluon_rmse"] for item in fold_metrics])),
                "autogluon_mean_hard_out_ap_aligned": float(np.nanmean([item["autogluon_mean_hard_out_ap_aligned"] for item in fold_metrics])),
                "autogluon_mean_hard_out_auc_aligned": float(np.nanmean([item["autogluon_mean_hard_out_auc_aligned"] for item in fold_metrics])),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def inner_oof_predictions(train_outer: pd.DataFrame, feature_columns: list[str], y_full: np.ndarray, top_k: int, config: dict[str, Any], model_prefix: Path) -> pd.DataFrame:
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["inner_n_splits"]))
    oof_parts: list[pd.DataFrame] = []
    for inner_idx, (inner_train_idx, inner_valid_idx) in enumerate(splitter.split(train_outer), start=1):
        inner_train = train_outer.iloc[inner_train_idx].copy().reset_index(drop=True)
        inner_valid = train_outer.iloc[inner_valid_idx].copy().reset_index(drop=True)
        y_inner_train = y_full[inner_train_idx]
        y_inner_valid = y_full[inner_valid_idx]
        ag_pred, _, _, _, _, _ = fit_outer_fold_model(
            inner_train,
            inner_valid,
            feature_columns,
            y_inner_train,
            y_inner_valid,
            top_k,
            config,
            Path(f"{model_prefix}_inner{inner_idx}"),
        )
        frame = inner_valid[["sample_time", "t90", "is_in_spec", "is_out_of_spec"]].copy()
        frame["pred_score_raw"] = ag_pred
        oof_parts.append(frame)
    return pd.concat(oof_parts, ignore_index=True)


def enrich_scored_rows(rows: pd.DataFrame, center: float, tolerance: float, measurement_error_scenario: float) -> pd.DataFrame:
    scored = rows.copy()
    t90_obs = scored["t90"].to_numpy(dtype=float)
    scored["measurement_error_scenario"] = float(measurement_error_scenario)
    scored["target_centered_desirability_point"] = centered_point_label(t90_obs, center, tolerance)
    scored["target_centered_desirability_uncertain"] = centered_uncertain_label(t90_obs, center, tolerance, float(measurement_error_scenario), 21)
    scored["target_centered_desirability_gaussian"] = centered_gaussian_label(t90_obs, center, 0.139)
    scored["true_center_band_obs"] = ((t90_obs >= 8.35) & (t90_obs <= 8.55)).astype(int)
    scored["true_center_band_prob"] = compute_center_band_prob(t90_obs, 8.35, 8.55, float(measurement_error_scenario))
    scored["true_outspec_obs"] = ((t90_obs < 8.20) | (t90_obs > 8.70)).astype(int)
    scored["true_outspec_prob"] = compute_outspec_prob(t90_obs, 8.20, 8.70, float(measurement_error_scenario))
    scored["pred_score_clipped"] = np.clip(scored["pred_score_raw"].to_numpy(dtype=float), 0.0, 1.0)
    scored["pred_desirability_aligned"] = scored["pred_score_clipped"]
    scored["pred_outspec_risk_aligned"] = 1.0 - scored["pred_score_clipped"]
    return scored


def run_cqdi_evaluation(
    prepared: pd.DataFrame,
    priors: dict[str, Any],
    config: dict[str, Any],
    best_methods: list[dict[str, Any]],
    artifact_dir: Path,
    ap_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    center = float(config["target_spec"]["center"])
    tolerance = float(config["target_spec"]["tolerance"])
    top_k = int(config["selection"]["top_k"])
    uncertain_grid_points = int(config["label_methods"]["uncertain"]["integration_points"])
    feature_columns = [column for column in prepared.columns if "__" in column]
    outer_splitter = TimeSeriesSplit(n_splits=int(config["validation"]["outer_n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    scored_rows_parts: list[pd.DataFrame] = []
    threshold_candidate_parts: list[dict[str, Any]] = []
    fold_summary_parts: list[dict[str, Any]] = []
    deploy_parts: list[dict[str, Any]] = []

    for scheme in best_methods:
        method_name = str(scheme["method_name"])
        method_param = None if pd.isna(scheme["method_param"]) else float(scheme["method_param"])
        scheme_name = method_name if method_param is None else f"{method_name}_{str(method_param).replace('.', 'p')}"
        scenario_fold_rows: dict[float, list[dict[str, Any]]] = {float(e): [] for e in config["measurement_error_scenarios"]}
        for fold_idx, (train_idx, test_idx) in enumerate(outer_splitter.split(prepared), start=1):
            train_outer = prepared.iloc[train_idx].copy().reset_index(drop=True)
            test_outer = prepared.iloc[test_idx].copy().reset_index(drop=True)
            y_train_outer = build_label_series(method_name, train_outer["t90"].to_numpy(dtype=float), center, tolerance, method_param, uncertain_grid_points)
            y_test_outer = build_label_series(method_name, test_outer["t90"].to_numpy(dtype=float), center, tolerance, method_param, uncertain_grid_points)
            inner_oof = inner_oof_predictions(
                train_outer, feature_columns, y_train_outer, top_k, config, artifact_dir / f"ag_centered_label_cqdi_{scheme_name}_fold{fold_idx}_{run_id}"
            )
            outer_pred, _, _, _, _, _ = fit_outer_fold_model(
                train_outer, test_outer, feature_columns, y_train_outer, y_test_outer, top_k, config, artifact_dir / f"ag_centered_label_cqdi_{scheme_name}_outer_fold{fold_idx}_{run_id}"
            )
            test_scored_base = test_outer[["sample_time", "t90", "is_in_spec", "is_out_of_spec"]].copy()
            test_scored_base["pred_score_raw"] = outer_pred
            for scenario in config["measurement_error_scenarios"]:
                scenario = float(scenario)
                inner_scored = enrich_scored_rows(inner_oof, center, tolerance, scenario)
                candidates = threshold_candidate_rows(inner_scored, list(config["preferred_threshold_grid"]), config["cqdi"], ap_floor)
                best_threshold = choose_threshold(candidates)
                for row in candidates:
                    row.update({"run_id": run_id, "scheme_name": scheme_name, "outer_fold_id": int(fold_idx), "measurement_error_scenario": scenario})
                    threshold_candidate_parts.append(row)
                test_scored = enrich_scored_rows(test_scored_base, center, tolerance, scenario)
                test_scored["run_id"] = run_id
                test_scored["scheme_name"] = scheme_name
                test_scored["label_family"] = "centered_desirability"
                test_scored["feature_recipe_name"] = "stage7_centered_best_x"
                test_scored["model_framework"] = "autogluon"
                test_scored["outer_fold_id"] = int(fold_idx)
                test_scored["preferred_threshold"] = float(best_threshold["inner_threshold_pref"])
                test_scored["preferred_flag"] = (test_scored["pred_desirability_aligned"] >= float(best_threshold["inner_threshold_pref"])).astype(int)
                test_scored["clear_threshold"] = np.nan
                test_scored["alert_threshold"] = np.nan
                test_scored["clear_flag"] = np.nan
                test_scored["retest_flag"] = np.nan
                test_scored["alert_flag"] = np.nan
                scored_rows_parts.append(test_scored)
                fold_summary = summarize_fold_cqdi(test_scored, float(best_threshold["inner_threshold_pref"]), config["cqdi"], ap_floor)
                fold_summary.update({"run_id": run_id, "scheme_name": scheme_name, "outer_fold_id": int(fold_idx), "selected_preferred_threshold": float(best_threshold["inner_threshold_pref"]), "measurement_error_scenario": scenario})
                fold_summary_parts.append(fold_summary)
                scenario_fold_rows[scenario].append(fold_summary)
        for scenario, rows_for_scenario in scenario_fold_rows.items():
            cqdi_values = [row["cqdi_fold"] for row in rows_for_scenario]
            gate_pass_rate = float(np.mean([row["gate_pass"] for row in rows_for_scenario])) if rows_for_scenario else float("nan")
            cqdi_mean = float(np.mean(cqdi_values)) if cqdi_values else float("nan")
            cqdi_median = float(np.median(cqdi_values)) if cqdi_values else float("nan")
            cqdi_p10 = float(np.percentile(cqdi_values, 10)) if cqdi_values else float("nan")
            hard_out_ap_aligned_mean = float(np.mean([row["test_hard_out_ap_aligned"] for row in rows_for_scenario])) if rows_for_scenario else float("nan")
            hard_out_auc_aligned_mean = float(np.mean([row["test_hard_out_auc_aligned"] for row in rows_for_scenario])) if rows_for_scenario else float("nan")
            if gate_pass_rate < 0.80 or cqdi_mean < 60.0:
                status = "FAIL"
            elif gate_pass_rate >= 0.80 and cqdi_mean >= 80.0 and cqdi_p10 >= 70.0:
                status = "READY"
            else:
                status = "CANARY"
            deploy_parts.append(
                {
                    "run_id": run_id,
                    "scheme_name": scheme_name,
                    "label_family": "centered_desirability",
                    "feature_recipe_name": "stage7_centered_best_x",
                    "measurement_error_scenario": scenario,
                    "cqdi_mean": cqdi_mean,
                    "cqdi_median": cqdi_median,
                    "cqdi_p10": cqdi_p10,
                    "gate_pass_rate": gate_pass_rate,
                    "hard_out_ap_aligned_mean": hard_out_ap_aligned_mean,
                    "hard_out_auc_aligned_mean": hard_out_auc_aligned_mean,
                    "recommended_status": status,
                    "summary_note": f"method={scheme_name}",
                }
            )
    return pd.concat(scored_rows_parts, ignore_index=True), pd.DataFrame(threshold_candidate_parts), pd.DataFrame(fold_summary_parts), pd.DataFrame(deploy_parts)


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Centered Desirability 标签修订对照实验",
        "",
        "## 目的",
        "",
        "- 按 `centered_desirability_label_revised_spec.md` 对当前 centered 分支补做三种标签方法的同条件对照。",
        "- 先对 `uncertain` 与 `gaussian` 做参数消融，再按 `CQDI` 规范做统一外层离线评估。",
        "",
        "## 主要结论",
        "",
        json.dumps(report, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run revised centered_desirability label-family ablation and CQDI evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    report_dir = resolve_path(config_path.parent, config["paths"]["report_dir"])
    if artifact_dir is None or report_dir is None:
        raise ValueError("artifact_dir and report_dir must be configured.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    prepared, priors, _ = build_centered_feature_frame(config_path, config)
    ablation_results, ablation_summary = run_ablation(prepared, priors, config, artifact_dir)
    point_best = ablation_summary[ablation_summary["method_name"] == "point"].iloc[0].to_dict()
    uncertain_best = choose_candidate_from_ablation(ablation_summary[ablation_summary["method_name"] == "uncertain"].to_dict(orient="records"), str(config["selection"]["ablation_primary_metric"]))
    gaussian_best = choose_candidate_from_ablation(ablation_summary[ablation_summary["method_name"] == "gaussian"].to_dict(orient="records"), str(config["selection"]["ablation_primary_metric"]))
    best_methods = [point_best, uncertain_best, gaussian_best]

    centered_outspec_eval_summary_path = resolve_path(config_path.parent, config["paths"]["centered_outspec_eval_summary_path"])
    if centered_outspec_eval_summary_path is None or not centered_outspec_eval_summary_path.exists():
        raise ValueError("centered_outspec_eval_summary_path must exist.")
    centered_outspec_summary = json.loads(centered_outspec_eval_summary_path.read_text(encoding="utf-8"))
    ap_floor = float(centered_outspec_summary["autogluon_mean_hard_out_ap_diagnostic"]) * float(config["cqdi"]["ap_floor_ratio"])

    scored_rows, threshold_candidates, fold_summary, deploy_summary = run_cqdi_evaluation(
        prepared,
        priors,
        config,
        best_methods,
        artifact_dir,
        ap_floor,
    )
    comparison_summary = {"selected_best_methods": best_methods, "deploy_summary": deploy_summary.to_dict(orient="records")}

    ablation_results.to_csv(artifact_dir / "centered_label_revised_ablation_results.csv", index=False, encoding="utf-8-sig")
    ablation_summary.to_json(artifact_dir / "centered_label_revised_ablation_summary.json", orient="records", force_ascii=False, indent=2)
    scored_rows.to_csv(artifact_dir / "offline_eval_scored_rows.csv", index=False, encoding="utf-8-sig")
    threshold_candidates.to_csv(artifact_dir / "offline_centered_threshold_candidates.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(artifact_dir / "offline_centered_fold_summary.csv", index=False, encoding="utf-8-sig")
    deploy_summary.to_csv(artifact_dir / "offline_centered_deployability_summary.csv", index=False, encoding="utf-8-sig")
    (artifact_dir / "centered_label_revised_comparison_summary.json").write_text(json.dumps(comparison_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_dir / "centered_label_revised_eval_summary.md", comparison_summary)
    print(json.dumps({"selected_best_methods": best_methods, "deploy_summary_path": str(artifact_dir / 'offline_centered_deployability_summary.csv')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
