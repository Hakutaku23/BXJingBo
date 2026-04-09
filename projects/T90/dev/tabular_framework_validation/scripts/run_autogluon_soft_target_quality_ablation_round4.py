from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_label_x_controlled_matrix_round1 import _prepare_labeled, _preclean
from run_autogluon_stage1_quickcheck import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    resolve_path,
)
from run_autogluon_stage2_dynamic_morphology import build_stage2_table
from run_autogluon_stage2_feature_engineering import select_features_fold
from run_autogluon_stage2_soft_probability import soft_probability_metrics
from run_autogluon_stage4_interactions import build_interaction_frame
from run_autogluon_stage5_quality import build_quality_table
from run_autogluon_stage7_final_selection import load_references, select_task_priors


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_soft_target_quality_ablation_round4.yaml"


def compute_outspec_prob(
    y_obs: np.ndarray,
    spec_low: float,
    spec_high: float,
    err_half_width: float,
) -> np.ndarray:
    low_part = np.maximum(0.0, np.minimum(spec_low, y_obs + err_half_width) - (y_obs - err_half_width))
    high_part = np.maximum(0.0, (y_obs + err_half_width) - np.maximum(spec_high, y_obs - err_half_width))
    return (low_part + high_part) / (2.0 * err_half_width)


def enrich_soft_scored_rows(
    rows: pd.DataFrame,
    measurement_error_scenario: float,
    spec_low: float,
    spec_high: float,
) -> pd.DataFrame:
    scored = rows.copy()
    t90_obs = scored["t90"].to_numpy(dtype=float)
    pred_score_raw = scored["pred_score_raw"].to_numpy(dtype=float)
    pred_score_clipped = np.clip(pred_score_raw, 0.0, 1.0)
    scored["measurement_error_scenario"] = float(measurement_error_scenario)
    scored["true_outspec_obs"] = ((t90_obs < spec_low) | (t90_obs > spec_high)).astype(int)
    scored["true_outspec_prob"] = compute_outspec_prob(
        t90_obs,
        spec_low=spec_low,
        spec_high=spec_high,
        err_half_width=float(measurement_error_scenario),
    )
    scored["pred_score_clipped"] = pred_score_clipped
    scored["pred_outspec_risk_aligned"] = pred_score_clipped
    scored["pred_desirability_aligned"] = 1.0 - pred_score_clipped
    return scored


def compute_radi_components(
    clear_outspec_prob_mean: float,
    alert_outspec_recall_prob: float,
    alert_outspec_precision_prob: float,
    outspec_prob_base: float,
    retest_rate: float,
    soft_brier: float,
) -> dict[str, float]:
    alert_outspec_lift = alert_outspec_precision_prob / max(outspec_prob_base, 1e-6)
    clear_score = min(1.0, 0.05 / max(clear_outspec_prob_mean, 1e-6))
    recall_score = min(1.0, alert_outspec_recall_prob / 0.80)
    lift_score = min(1.0, alert_outspec_lift / 2.00)
    retest_score = float(np.clip(1.0 - max(0.0, retest_rate - 0.25) / 0.25, 0.0, 1.0))
    brier_score = min(1.0, 0.10 / max(soft_brier, 1e-6))
    return {
        "alert_outspec_lift": alert_outspec_lift,
        "clear_score": clear_score,
        "recall_score": recall_score,
        "lift_score": lift_score,
        "retest_score": retest_score,
        "brier_score": brier_score,
    }


def soft_threshold_candidate_rows(
    scored: pd.DataFrame,
    clear_threshold_grid: list[float],
    alert_threshold_grid: list[float],
    ap_floor: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_risk = scored["pred_outspec_risk_aligned"].to_numpy(dtype=float)
    true_outspec_prob = scored["true_outspec_prob"].to_numpy(dtype=float)
    true_outspec_obs = scored["true_outspec_obs"].to_numpy(dtype=int)
    outspec_prob_base = float(np.mean(true_outspec_prob))
    soft_brier = float(np.mean((pred_risk - true_outspec_prob) ** 2))
    hard_out_ap_aligned = float(average_precision_score(true_outspec_obs, pred_risk))
    for tau_low in clear_threshold_grid:
        for tau_high in alert_threshold_grid:
            if float(tau_low) >= float(tau_high):
                continue
            clear_flag = (pred_risk <= float(tau_low)).astype(int)
            alert_flag = (pred_risk >= float(tau_high)).astype(int)
            retest_flag = 1 - clear_flag - alert_flag
            clear_rate = float(np.mean(clear_flag))
            retest_rate = float(np.mean(retest_flag))
            alert_rate = float(np.mean(alert_flag))
            clear_outspec_prob_mean = float(np.mean(true_outspec_prob[clear_flag == 1])) if clear_flag.sum() > 0 else 1.0
            alert_outspec_recall_prob = float(np.sum(true_outspec_prob * alert_flag) / max(np.sum(true_outspec_prob), 1e-6))
            alert_outspec_precision_prob = float(np.sum(true_outspec_prob * alert_flag) / max(np.sum(alert_flag), 1e-6))
            comps = compute_radi_components(
                clear_outspec_prob_mean=clear_outspec_prob_mean,
                alert_outspec_recall_prob=alert_outspec_recall_prob,
                alert_outspec_precision_prob=alert_outspec_precision_prob,
                outspec_prob_base=outspec_prob_base,
                retest_rate=retest_rate,
                soft_brier=soft_brier,
            )
            gate_pass = int(
                clear_outspec_prob_mean <= 0.08
                and alert_outspec_recall_prob >= 0.50
                and hard_out_ap_aligned >= ap_floor
            )
            radi_train_candidate = 100.0 * gate_pass * (
                0.35 * comps["clear_score"]
                + 0.30 * comps["recall_score"]
                + 0.15 * comps["lift_score"]
                + 0.10 * comps["retest_score"]
                + 0.10 * comps["brier_score"]
            )
            rows.append(
                {
                    "inner_clear_threshold": float(tau_low),
                    "inner_alert_threshold": float(tau_high),
                    "train_clear_rate": clear_rate,
                    "train_retest_rate": retest_rate,
                    "train_alert_rate": alert_rate,
                    "train_clear_outspec_prob_mean": clear_outspec_prob_mean,
                    "train_alert_outspec_recall_prob": alert_outspec_recall_prob,
                    "train_alert_outspec_precision_prob": alert_outspec_precision_prob,
                    "train_outspec_prob_base": outspec_prob_base,
                    "train_alert_outspec_lift": comps["alert_outspec_lift"],
                    "train_soft_brier": soft_brier,
                    "train_hard_out_ap_aligned": hard_out_ap_aligned,
                    "train_gate_pass": gate_pass,
                    "radi_train_candidate": radi_train_candidate,
                }
            )
    return rows


def choose_soft_thresholds(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            -row["radi_train_candidate"],
            -row["train_gate_pass"],
            row["train_clear_outspec_prob_mean"],
            -row["train_alert_outspec_recall_prob"],
            -row["train_alert_outspec_lift"],
            row["train_retest_rate"],
            row["inner_clear_threshold"],
            row["inner_alert_threshold"],
        ),
    )
    return ranked[0]


def summarize_fold_radi(
    scored: pd.DataFrame,
    clear_threshold: float,
    alert_threshold: float,
    ap_floor: float,
) -> dict[str, Any]:
    pred_risk = scored["pred_outspec_risk_aligned"].to_numpy(dtype=float)
    true_outspec_prob = scored["true_outspec_prob"].to_numpy(dtype=float)
    true_outspec_obs = scored["true_outspec_obs"].to_numpy(dtype=int)
    clear_flag = (pred_risk <= float(clear_threshold)).astype(int)
    alert_flag = (pred_risk >= float(alert_threshold)).astype(int)
    retest_flag = 1 - clear_flag - alert_flag
    clear_rate = float(np.mean(clear_flag))
    retest_rate = float(np.mean(retest_flag))
    alert_rate = float(np.mean(alert_flag))
    clear_outspec_prob_mean = float(np.mean(true_outspec_prob[clear_flag == 1])) if clear_flag.sum() > 0 else 1.0
    alert_outspec_recall_prob = float(np.sum(true_outspec_prob * alert_flag) / max(np.sum(true_outspec_prob), 1e-6))
    alert_outspec_precision_prob = float(np.sum(true_outspec_prob * alert_flag) / max(np.sum(alert_flag), 1e-6))
    outspec_prob_base = float(np.mean(true_outspec_prob))
    soft_brier = float(np.mean((pred_risk - true_outspec_prob) ** 2))
    hard_out_ap_aligned = float(average_precision_score(true_outspec_obs, pred_risk))
    hard_out_auc_aligned = float(roc_auc_score(true_outspec_obs, pred_risk))
    comps = compute_radi_components(
        clear_outspec_prob_mean=clear_outspec_prob_mean,
        alert_outspec_recall_prob=alert_outspec_recall_prob,
        alert_outspec_precision_prob=alert_outspec_precision_prob,
        outspec_prob_base=outspec_prob_base,
        retest_rate=retest_rate,
        soft_brier=soft_brier,
    )
    gate_pass = int(
        clear_outspec_prob_mean <= 0.08
        and alert_outspec_recall_prob >= 0.50
        and hard_out_ap_aligned >= ap_floor
    )
    radi_fold = 100.0 * gate_pass * (
        0.35 * comps["clear_score"]
        + 0.30 * comps["recall_score"]
        + 0.15 * comps["lift_score"]
        + 0.10 * comps["retest_score"]
        + 0.10 * comps["brier_score"]
    )
    status = "FAIL" if gate_pass == 0 or radi_fold < 60.0 else ("READY" if radi_fold >= 80.0 else "CANARY")
    return {
        "test_clear_rate": clear_rate,
        "test_retest_rate": retest_rate,
        "test_alert_rate": alert_rate,
        "test_clear_outspec_prob_mean": clear_outspec_prob_mean,
        "test_alert_outspec_recall_prob": alert_outspec_recall_prob,
        "test_alert_outspec_precision_prob": alert_outspec_precision_prob,
        "test_outspec_prob_base": outspec_prob_base,
        "test_alert_outspec_lift": comps["alert_outspec_lift"],
        "test_soft_brier": soft_brier,
        "test_hard_out_ap_aligned": hard_out_ap_aligned,
        "test_hard_out_auc_aligned": hard_out_auc_aligned,
        "radi_fold": radi_fold,
        "gate_pass": gate_pass,
        "deployment_status_fold": status,
    }


def summarize_deployability(
    fold_rows: list[dict[str, Any]],
    variant_name: str,
    measurement_error_scenario: float,
) -> dict[str, Any]:
    radi_values = [float(row["radi_fold"]) for row in fold_rows]
    gate_pass_rate = float(np.mean([float(row["gate_pass"]) for row in fold_rows])) if fold_rows else float("nan")
    radi_mean = float(np.mean(radi_values)) if fold_rows else float("nan")
    radi_median = float(np.median(radi_values)) if fold_rows else float("nan")
    radi_p10 = float(np.percentile(radi_values, 10)) if fold_rows else float("nan")
    hard_out_ap_aligned_mean = float(np.mean([float(row["test_hard_out_ap_aligned"]) for row in fold_rows])) if fold_rows else float("nan")
    hard_out_auc_aligned_mean = float(np.mean([float(row["test_hard_out_auc_aligned"]) for row in fold_rows])) if fold_rows else float("nan")
    soft_brier_mean = float(np.mean([float(row["test_soft_brier"]) for row in fold_rows])) if fold_rows else float("nan")
    if gate_pass_rate < 0.80 or radi_mean < 60.0:
        status = "FAIL"
    elif gate_pass_rate >= 0.80 and radi_mean >= 80.0 and radi_p10 >= 70.0:
        status = "READY"
    else:
        status = "CANARY"
    return {
        "scheme_name": variant_name,
        "label_family": "soft_target",
        "feature_recipe_name": variant_name,
        "measurement_error_scenario": float(measurement_error_scenario),
        "radi_mean": radi_mean,
        "radi_median": radi_median,
        "radi_p10": radi_p10,
        "gate_pass_rate": gate_pass_rate,
        "hard_out_ap_aligned_mean": hard_out_ap_aligned_mean,
        "hard_out_auc_aligned_mean": hard_out_auc_aligned_mean,
        "soft_brier_mean": soft_brier_mean,
        "recommended_status": status,
    }


def load_round3_ap_floor(config_path: Path, config: dict[str, Any]) -> float:
    summary_path = resolve_path(config_path.parent, config["paths"]["round3_summary_path"])
    if summary_path is None or not summary_path.exists():
        raise ValueError("round3_summary_path must exist for AP floor calibration.")
    summary = pd.DataFrame(json.loads(summary_path.read_text(encoding="utf-8")))
    baseline_variant = str(config["radi"]["engineering_baseline_variant"])
    baseline = summary[summary["variant_name"] == baseline_variant].copy()
    if baseline.empty:
        raise ValueError(f"Variant {baseline_variant!r} not found in round3 summary.")
    ap = float(baseline.iloc[0]["autogluon_mean_hard_out_ap_diagnostic"])
    return ap * float(config["radi"]["ap_floor_ratio"])


def build_quality_variant_frames(config_path: Path, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    labeled = _prepare_labeled(config_path, config)
    soft_label_name = str(config["label_fuzziness"]["target_name"])
    refs = load_references(config_path, config)
    priors = select_task_priors(refs)["centered_desirability"]
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
    if snapshot.empty:
        raise ValueError("Base lag snapshot is empty.")
    interaction_frame = build_interaction_frame(
        snapshot,
        int(priors["tau_minutes"]),
        int(priors["window_minutes"]),
        str(priors["interaction_package"]),
    )
    interaction_only = interaction_frame.reset_index(drop=True)
    base_feature_columns = [column for column in snapshot.columns if "__" in column and "_dyn_" not in column]
    base_meta = snapshot[
        [
            "sample_time",
            "decision_time",
            "t90",
            "is_in_spec",
            "is_out_of_spec",
            "is_above_spec",
            "is_below_spec",
        ]
        + base_feature_columns
    ].copy()
    base_meta = base_meta.merge(labeled[["sample_time", soft_label_name]], on="sample_time", how="left")
    base_meta = pd.concat([base_meta.reset_index(drop=True), interaction_only], axis=1)

    frames: dict[str, pd.DataFrame] = {}
    for variant_name, enabled_features in dict(config["quality_variants"]).items():
        quality_table = build_quality_table(
            labeled_samples=labeled,
            dcs=dcs,
            tau_minutes=int(priors["tau_minutes"]),
            window_minutes=int(priors["window_minutes"]),
            enabled_features=list(enabled_features),
            min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
        )
        quality_cols = [column for column in quality_table.columns if column.startswith("quality__")]
        quality_merged = snapshot[["decision_time"]].merge(quality_table, on="decision_time", how="left")
        quality_only = quality_merged[quality_cols].reset_index(drop=True)
        frame = pd.concat([base_meta.reset_index(drop=True), quality_only], axis=1)
        frames[str(variant_name)] = frame.sort_values("sample_time").reset_index(drop=True)
    return frames


def build_inner_oof_predictions(
    train_outer: pd.DataFrame,
    feature_columns: list[str],
    soft_label_name: str,
    config: dict[str, Any],
    artifact_dir: Path,
    run_id: str,
    variant_name: str,
    outer_fold_idx: int,
) -> pd.DataFrame:
    inner_splitter = TimeSeriesSplit(n_splits=int(config["radi"]["inner_n_splits"]))
    top_k = int(config["selection"]["shared_top_k"])
    oof_parts: list[pd.DataFrame] = []
    for inner_idx, (train_idx, valid_idx) in enumerate(inner_splitter.split(train_outer), start=1):
        inner_train = train_outer.iloc[train_idx].copy().reset_index(drop=True)
        inner_valid = train_outer.iloc[valid_idx].copy().reset_index(drop=True)
        train_clean, valid_clean, cleaned_cols = _preclean(
            inner_train[feature_columns],
            inner_valid[feature_columns],
            max_missing_ratio=float(config["preclean"]["max_missing_ratio"]),
            unique_threshold=int(config["preclean"]["near_constant_unique_threshold"]),
        )
        selected_features, _ = select_features_fold(
            train_x=train_clean,
            train_y=inner_train[soft_label_name],
            task_type="regression",
            top_k=top_k,
        )
        if not selected_features:
            selected_features = cleaned_cols
        train_sel = train_clean[selected_features].copy()
        valid_sel = valid_clean[selected_features].copy()
        model_path = artifact_dir / f"ag_soft_target_quality_round4_{variant_name}_{run_id}_outer{outer_fold_idx}_inner{inner_idx}"
        ag_pred, _ = fit_autogluon_fold(
            train_df=pd.concat([train_sel, inner_train[[soft_label_name]]], axis=1).copy(),
            test_df=pd.concat([valid_sel, inner_valid[[soft_label_name]]], axis=1).copy(),
            label=soft_label_name,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )
        part = inner_valid[["sample_time", "t90", "is_out_of_spec"]].copy()
        part["pred_score_raw"] = ag_pred.astype(float)
        oof_parts.append(part)
    return pd.concat(oof_parts, ignore_index=True)


def evaluate_variant(
    frame: pd.DataFrame,
    variant_name: str,
    config: dict[str, Any],
    artifact_dir: Path,
    run_id: str,
    ap_floor: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    soft_label_name = str(config["label_fuzziness"]["target_name"])
    top_k = int(config["selection"]["shared_top_k"])
    feature_columns = [column for column in frame.columns if "__" in column]
    spec_low = float(config["radi"]["spec_low"])
    spec_high = float(config["radi"]["spec_high"])
    measurement_error_scenarios = [float(value) for value in config["measurement_error_scenarios"]]
    clear_threshold_grid = [float(value) for value in config["radi"]["clear_threshold_grid"]]
    alert_threshold_grid = [float(value) for value in config["radi"]["alert_threshold_grid"]]

    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    scored_rows_parts: list[pd.DataFrame] = []
    threshold_candidate_parts: list[dict[str, Any]] = []
    fold_summary_parts: list[dict[str, Any]] = []
    raw_feature_count = 0
    cleaned_feature_count = 0
    selected_feature_count = 0
    selected_features_fold1: list[str] = []
    scenario_fold_rows: dict[float, list[dict[str, Any]]] = {scenario: [] for scenario in measurement_error_scenarios}

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(frame), start=1):
        train = frame.iloc[train_idx].copy().reset_index(drop=True)
        test = frame.iloc[test_idx].copy().reset_index(drop=True)
        train_clean, test_clean, cleaned_cols = _preclean(
            train[feature_columns],
            test[feature_columns],
            max_missing_ratio=float(config["preclean"]["max_missing_ratio"]),
            unique_threshold=int(config["preclean"]["near_constant_unique_threshold"]),
        )
        raw_feature_count = int(len(feature_columns))
        cleaned_feature_count = int(len(cleaned_cols))
        selected_features, _ = select_features_fold(
            train_x=train_clean,
            train_y=train[soft_label_name],
            task_type="regression",
            top_k=top_k,
        )
        if not selected_features:
            selected_features = cleaned_cols
        selected_feature_count = int(len(selected_features))
        if fold_idx == 1:
            selected_features_fold1 = list(selected_features)

        train_sel = train_clean[selected_features].copy()
        test_sel = test_clean[selected_features].copy()
        y_train = train[soft_label_name].to_numpy(dtype=float)
        y_test = test[soft_label_name].to_numpy(dtype=float)
        hard_out = test["is_out_of_spec"].to_numpy(dtype=int)

        baseline = make_regression_baseline()
        baseline.fit(train_sel, y_train)
        base_pred = baseline.predict(test_sel).astype(float)
        base_metric = soft_probability_metrics(y_test, base_pred, hard_out)

        model_path = artifact_dir / f"ag_soft_target_quality_round4_{variant_name}_{run_id}_fold{fold_idx}"
        ag_pred, model_best = fit_autogluon_fold(
            train_df=pd.concat([train_sel, train[[soft_label_name]]], axis=1).copy(),
            test_df=pd.concat([test_sel, test[[soft_label_name]]], axis=1).copy(),
            label=soft_label_name,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )
        ag_metric = soft_probability_metrics(y_test, ag_pred.astype(float), hard_out)

        rows.append(
            {
                "variant_name": variant_name,
                "framework": "simple_baseline",
                "fold": int(fold_idx),
                "raw_feature_count": raw_feature_count,
                "cleaned_feature_count": cleaned_feature_count,
                "selected_feature_count": selected_feature_count,
                **base_metric,
            }
        )
        rows.append(
            {
                "variant_name": variant_name,
                "framework": "autogluon",
                "fold": int(fold_idx),
                "raw_feature_count": raw_feature_count,
                "cleaned_feature_count": cleaned_feature_count,
                "selected_feature_count": selected_feature_count,
                "autogluon_model_best": model_best,
                **ag_metric,
            }
        )
        fold_summaries.append(
            {
                "fold": int(fold_idx),
                "selected_feature_count": selected_feature_count,
                "baseline_soft_brier": float(base_metric["soft_brier"]),
                "autogluon_soft_brier": float(ag_metric["soft_brier"]),
                "autogluon_hard_out_ap_diagnostic": float(ag_metric["hard_out_ap_diagnostic"]),
                "autogluon_hard_out_auc_diagnostic": float(ag_metric["hard_out_auc_diagnostic"]),
            }
        )

        inner_oof = build_inner_oof_predictions(
            train_outer=train,
            feature_columns=feature_columns,
            soft_label_name=soft_label_name,
            config=config,
            artifact_dir=artifact_dir,
            run_id=run_id,
            variant_name=variant_name,
            outer_fold_idx=fold_idx,
        )
        test_scored_base = test[["sample_time", "t90", "is_out_of_spec"]].copy()
        test_scored_base["pred_score_raw"] = ag_pred.astype(float)

        for scenario in measurement_error_scenarios:
            inner_scored = enrich_soft_scored_rows(inner_oof, scenario, spec_low, spec_high)
            candidates = soft_threshold_candidate_rows(
                inner_scored,
                clear_threshold_grid=clear_threshold_grid,
                alert_threshold_grid=alert_threshold_grid,
                ap_floor=ap_floor,
            )
            best_threshold = choose_soft_thresholds(candidates)
            for row in candidates:
                row.update(
                    {
                        "run_id": run_id,
                        "scheme_name": variant_name,
                        "outer_fold_id": int(fold_idx),
                        "measurement_error_scenario": scenario,
                    }
                )
                threshold_candidate_parts.append(row)

            test_scored = enrich_soft_scored_rows(test_scored_base, scenario, spec_low, spec_high)
            test_scored["run_id"] = run_id
            test_scored["scheme_name"] = variant_name
            test_scored["label_family"] = "soft_target"
            test_scored["feature_recipe_name"] = variant_name
            test_scored["model_framework"] = "autogluon"
            test_scored["outer_fold_id"] = int(fold_idx)
            test_scored["preferred_threshold"] = np.nan
            test_scored["preferred_flag"] = np.nan
            test_scored["clear_threshold"] = float(best_threshold["inner_clear_threshold"])
            test_scored["alert_threshold"] = float(best_threshold["inner_alert_threshold"])
            test_scored["clear_flag"] = (test_scored["pred_outspec_risk_aligned"] <= float(best_threshold["inner_clear_threshold"])).astype(int)
            test_scored["alert_flag"] = (test_scored["pred_outspec_risk_aligned"] >= float(best_threshold["inner_alert_threshold"])).astype(int)
            test_scored["retest_flag"] = 1 - test_scored["clear_flag"] - test_scored["alert_flag"]
            scored_rows_parts.append(test_scored)

            fold_summary = summarize_fold_radi(
                test_scored,
                clear_threshold=float(best_threshold["inner_clear_threshold"]),
                alert_threshold=float(best_threshold["inner_alert_threshold"]),
                ap_floor=ap_floor,
            )
            fold_summary.update(
                {
                    "run_id": run_id,
                    "scheme_name": variant_name,
                    "outer_fold_id": int(fold_idx),
                    "selected_clear_threshold": float(best_threshold["inner_clear_threshold"]),
                    "selected_alert_threshold": float(best_threshold["inner_alert_threshold"]),
                    "measurement_error_scenario": scenario,
                }
            )
            fold_summary_parts.append(fold_summary)
            scenario_fold_rows[scenario].append(fold_summary)

    deploy_parts = [
        {
            "run_id": run_id,
            **summarize_deployability(
                fold_rows=scenario_fold_rows[scenario],
                variant_name=variant_name,
                measurement_error_scenario=scenario,
            ),
        }
        for scenario in measurement_error_scenarios
    ]

    results = pd.DataFrame(rows)
    baseline_rows = results[results["framework"] == "simple_baseline"]
    ag_rows = results[results["framework"] == "autogluon"]
    deploy_df = pd.DataFrame(deploy_parts)
    summary = {
        "variant_name": variant_name,
        "raw_feature_count": raw_feature_count,
        "cleaned_feature_count": cleaned_feature_count,
        "selected_feature_count": selected_feature_count,
        "selected_features_fold1": selected_features_fold1,
        "baseline_mean_soft_mae": float(baseline_rows["soft_mae"].mean()),
        "autogluon_mean_soft_mae": float(ag_rows["soft_mae"].mean()),
        "baseline_mean_soft_brier": float(baseline_rows["soft_brier"].mean()),
        "autogluon_mean_soft_brier": float(ag_rows["soft_brier"].mean()),
        "baseline_mean_hard_out_ap_diagnostic": float(baseline_rows["hard_out_ap_diagnostic"].mean()),
        "autogluon_mean_hard_out_ap_diagnostic": float(ag_rows["hard_out_ap_diagnostic"].mean()),
        "baseline_mean_hard_out_auc_diagnostic": float(baseline_rows["hard_out_auc_diagnostic"].mean()),
        "autogluon_mean_hard_out_auc_diagnostic": float(ag_rows["hard_out_auc_diagnostic"].mean()),
        "radi_measurement_summaries": deploy_df.to_dict(orient="records"),
        "radi_worst_case_mean": float(deploy_df["radi_mean"].min()) if not deploy_df.empty else float("nan"),
        "radi_worst_case_gate_pass_rate": float(deploy_df["gate_pass_rate"].min()) if not deploy_df.empty else float("nan"),
        "radi_worst_case_status": (
            deploy_df.sort_values(["radi_mean", "measurement_error_scenario"], ascending=[True, True]).iloc[0]["recommended_status"]
            if not deploy_df.empty
            else "FAIL"
        ),
        "fold_summaries": fold_summaries,
    }
    return results, summary, pd.concat(scored_rows_parts, ignore_index=True), pd.DataFrame(threshold_candidate_parts), pd.DataFrame(fold_summary_parts)


def write_report(path: Path, summary_rows: list[dict[str, Any]], ap_floor: float) -> None:
    lines = [
        "# Soft Target Quality Ablation Round 4",
        "",
        "## Scope",
        "",
        "- Freeze `soft_target + lag120_win60 + full flow_balance interaction`.",
        "- Only compress the `quality` package.",
        "- Add `RADI` offline deployability evaluation with inner `TimeSeriesSplit(3)` threshold search.",
        "",
        "## AP Floor",
        "",
        f"- `ap_floor = {ap_floor:.6f}`",
        "",
        "## Summary Rows",
        "",
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run soft target quality compression ablation with RADI evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    report_dir = resolve_path(config_path.parent, config["paths"]["report_dir"])
    if artifact_dir is None or report_dir is None:
        raise ValueError("Both artifact_dir and report_dir must be configured.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    ap_floor = load_round3_ap_floor(config_path, config)
    frames = build_quality_variant_frames(config_path, config)
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    detail_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    scored_rows_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []
    fold_summary_parts: list[pd.DataFrame] = []

    for variant_name, frame in frames.items():
        results, summary, scored_rows, threshold_candidates, fold_summary = evaluate_variant(
            frame=frame,
            variant_name=variant_name,
            config=config,
            artifact_dir=artifact_dir,
            run_id=run_id,
            ap_floor=ap_floor,
        )
        detail_parts.append(results)
        summary_rows.append(summary)
        scored_rows_parts.append(scored_rows)
        threshold_parts.append(threshold_candidates)
        fold_summary_parts.append(fold_summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["radi_worst_case_mean", "autogluon_mean_soft_brier", "autogluon_mean_hard_out_ap_diagnostic", "variant_name"],
        ascending=[False, True, False, True],
    )
    details_df = pd.concat(detail_parts, ignore_index=True)
    scored_rows_df = pd.concat(scored_rows_parts, ignore_index=True)
    threshold_df = pd.concat(threshold_parts, ignore_index=True)
    fold_summary_df = pd.concat(fold_summary_parts, ignore_index=True)
    deploy_rows = [deploy for row in summary_rows for deploy in row["radi_measurement_summaries"]]
    deploy_df = pd.DataFrame(deploy_rows).sort_values(by=["scheme_name", "measurement_error_scenario"], ascending=[True, True])

    summary_path = artifact_dir / "soft_target_quality_ablation_round4_summary.json"
    results_path = artifact_dir / "soft_target_quality_ablation_round4_results.csv"
    scored_rows_path = artifact_dir / "offline_eval_scored_rows.csv"
    threshold_path = artifact_dir / "offline_soft_threshold_candidates.csv"
    fold_summary_path = artifact_dir / "offline_soft_fold_summary.csv"
    deploy_path = artifact_dir / "offline_soft_deployability_summary.csv"
    report_path = report_dir / "soft_target_quality_ablation_round4_summary.md"

    summary_path.write_text(json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    details_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    scored_rows_df.to_csv(scored_rows_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    fold_summary_df.to_csv(fold_summary_path, index=False, encoding="utf-8-sig")
    deploy_df.to_csv(deploy_path, index=False, encoding="utf-8-sig")
    write_report(report_path, summary_df.to_dict(orient="records"), ap_floor)

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "results_path": str(results_path),
                "report_path": str(report_path),
                "offline_scored_rows_path": str(scored_rows_path),
                "offline_soft_threshold_candidates_path": str(threshold_path),
                "offline_soft_fold_summary_path": str(fold_summary_path),
                "offline_soft_deployability_summary_path": str(deploy_path),
                "best_variant": summary_df.iloc[0].to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
