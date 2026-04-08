from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression

from run_autogluon_stage1_quickcheck import load_config, resolve_path
from run_autogluon_stage2_feature_engineering import make_regression_baseline, select_features_fold
from run_autogluon_stage2_soft_probability_x_enrichment import build_variant_snapshot
from run_autogluon_stage7_final_selection import (
    build_current_task_frame,
    combo_specs,
    compose_fold_features,
    load_references,
    preclean_features,
    select_task_priors,
)
from run_autogluon_centered_desirability_outspec_eval import load_best_centered_combo
from run_autogluon_stage1_lag_scale import build_label_frame
from run_autogluon_stage2_dynamic_morphology import build_stage2_table
from run_autogluon_stage5_quality import build_quality_table
from run_autogluon_stage6_centered_quality import build_centered_quality_table


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_soft_centered_fusion_mvp.yaml"


RESERVED_AUX_PREFIXES = ("quality__", "centered__", "interaction__", "state__")


def fit_predictor(train_df: pd.DataFrame, label: str, problem_type: str, eval_metric: str, model_path: Path, ag_config: dict[str, Any]) -> tuple[TabularPredictor, str]:
    if model_path.exists():
        shutil.rmtree(model_path)
    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=str(model_path),
        verbosity=int(ag_config["verbosity"]),
    )
    predictor.fit(
        train_data=train_df,
        presets=ag_config["presets"],
        time_limit=int(ag_config["time_limit_seconds"]),
        hyperparameters=ag_config.get("hyperparameters"),
    )
    return predictor, str(predictor.model_best)


def risk_metrics(y_true_out: np.ndarray, risk_score: np.ndarray, tau_low: float, tau_high: float) -> dict[str, float]:
    risk = np.clip(risk_score.astype(float), 0.0, 1.0)
    metrics: dict[str, float] = {
        "brier": float(np.mean((y_true_out - risk) ** 2)),
    }
    if len(np.unique(y_true_out)) > 1:
        metrics["hard_out_ap_diagnostic"] = float(average_precision_score(y_true_out, risk))
        metrics["hard_out_auc_diagnostic"] = float(roc_auc_score(y_true_out, risk))
    else:
        metrics["hard_out_ap_diagnostic"] = float("nan")
        metrics["hard_out_auc_diagnostic"] = float("nan")
    low_risk_mask = risk < float(tau_low)
    high_risk_mask = risk >= float(tau_high)
    metrics["low_risk_rate"] = float(np.mean(low_risk_mask))
    metrics["high_risk_rate"] = float(np.mean(high_risk_mask))
    metrics["retest_rate"] = float(1.0 - metrics["low_risk_rate"] - metrics["high_risk_rate"])
    if int(np.sum(y_true_out == 1)) > 0:
        metrics["false_clear_rate"] = float(np.sum((y_true_out == 1) & low_risk_mask) / np.sum(y_true_out == 1))
    else:
        metrics["false_clear_rate"] = float("nan")
    return metrics


def subset_risk_metrics(y_true_out: np.ndarray, risk_score: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    mask = mask.astype(bool)
    metrics: dict[str, float] = {f"{prefix}_sample_rate": float(np.mean(mask))}
    if not mask.any():
        metrics[f"{prefix}_ap_diagnostic"] = float("nan")
        metrics[f"{prefix}_auc_diagnostic"] = float("nan")
        metrics[f"{prefix}_positive_rate"] = float("nan")
        return metrics
    y = y_true_out[mask].astype(int)
    score = np.clip(risk_score[mask].astype(float), 0.0, 1.0)
    metrics[f"{prefix}_positive_rate"] = float(np.mean(y))
    if len(np.unique(y)) > 1:
        metrics[f"{prefix}_ap_diagnostic"] = float(average_precision_score(y, score))
        metrics[f"{prefix}_auc_diagnostic"] = float(roc_auc_score(y, score))
    else:
        metrics[f"{prefix}_ap_diagnostic"] = float("nan")
        metrics[f"{prefix}_auc_diagnostic"] = float("nan")
    return metrics


def choose_calibration_tail_indices(train_len: int, calibration_fraction: float, min_calibration_samples: int, min_base_train_samples: int) -> tuple[np.ndarray, np.ndarray]:
    cal_size = max(int(np.ceil(train_len * calibration_fraction)), int(min_calibration_samples))
    cal_size = min(cal_size, train_len - int(min_base_train_samples))
    if cal_size <= 0:
        raise ValueError("Not enough samples to create a calibration tail split.")
    base_end = train_len - cal_size
    base_idx = np.arange(0, base_end, dtype=int)
    cal_idx = np.arange(base_end, train_len, dtype=int)
    return base_idx, cal_idx


def prepare_soft_frame(config_path: Path, config: dict[str, Any], variant_name: str) -> pd.DataFrame:
    snapshot, _ = build_variant_snapshot(config_path, config, variant_name)
    frame = snapshot.copy()
    if "sample_time" not in frame.columns:
        raise ValueError("Soft target frame must contain sample_time for temporal alignment.")
    frame["align_time"] = pd.to_datetime(frame["sample_time"])
    if "decision_time" not in frame.columns:
        frame["decision_time"] = frame["align_time"]
    return frame.sort_values("align_time").reset_index(drop=True).copy()


def prepare_centered_frame(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    refs = load_references(config_path, config)
    priors = select_task_priors(refs)["centered_desirability"]
    best_centered = load_best_centered_combo(resolve_path(config_path.parent, config["paths"]["stage7_summary_path"]))
    best_combo_name = str(best_centered["combo_name"])
    best_top_k = int(best_centered["top_k"])
    combo = next(spec for spec in combo_specs("centered_desirability", priors) if spec["name"] == best_combo_name)

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
    if "sample_time" in prepared.columns:
        prepared["align_time"] = pd.to_datetime(prepared["sample_time"])
    elif "decision_time" in prepared.columns:
        prepared["align_time"] = pd.to_datetime(prepared["decision_time"])
    else:
        raise ValueError("Centered frame must contain sample_time or decision_time for temporal alignment.")
    prepared = prepared.sort_values("align_time").reset_index(drop=True).copy()
    return prepared, priors, {"combo_name": best_combo_name, "top_k": best_top_k}


def align_common_frame(soft_frame: pd.DataFrame, centered_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = (
        soft_frame[["align_time", "decision_time", "sample_time", "t90", "is_in_spec", "is_out_of_spec", "target_soft_out_of_spec_probability"]]
        .merge(
            centered_frame[["align_time", "target_centered_desirability"]],
            on="align_time",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("align_time")
        .reset_index(drop=True)
    )
    times = common["align_time"]
    soft_aligned = soft_frame.set_index("align_time").loc[times].reset_index()
    centered_aligned = centered_frame.set_index("align_time").loc[times].reset_index()
    return soft_aligned, centered_aligned


def choose_best_fixed_weight(calibration_frame: pd.DataFrame, candidates: list[float]) -> tuple[float, dict[str, float]]:
    best_weight = float(candidates[0])
    best_metrics: dict[str, float] | None = None
    y_true = calibration_frame["y_true_out"].to_numpy(dtype=int)
    for weight in candidates:
        fused = weight * calibration_frame["p_fail_soft"].to_numpy(dtype=float) + (1.0 - weight) * calibration_frame["p_fail_center_proxy"].to_numpy(dtype=float)
        brier = float(np.mean((y_true - fused) ** 2))
        ap = float(average_precision_score(y_true, fused)) if len(np.unique(y_true)) > 1 else float("nan")
        candidate_metrics = {"calibration_brier": brier, "calibration_ap": ap}
        if best_metrics is None or brier < best_metrics["calibration_brier"] - 1e-12 or (
            abs(brier - best_metrics["calibration_brier"]) <= 1e-12 and ap > best_metrics["calibration_ap"]
        ):
            best_weight = float(weight)
            best_metrics = candidate_metrics
    assert best_metrics is not None
    return best_weight, best_metrics


def fit_logistic_fusion(calibration_frame: pd.DataFrame) -> LogisticRegression | None:
    y = calibration_frame["y_true_out"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return None
    x = calibration_frame[["p_fail_soft", "p_fail_center_proxy", "disagreement"]].to_numpy(dtype=float)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(x, y)
    return model


def build_calibration_features(p_fail_soft: np.ndarray, p_fail_center_proxy: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "p_fail_soft": np.clip(p_fail_soft.astype(float), 0.0, 1.0),
            "p_fail_center_proxy": np.clip(p_fail_center_proxy.astype(float), 0.0, 1.0),
        }
    )
    frame["disagreement"] = (frame["p_fail_soft"] - frame["p_fail_center_proxy"]).abs()
    return frame


def drop_duplicate_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.columns.duplicated().any():
        return frame
    return frame.loc[:, ~frame.columns.duplicated()].copy()


def apply_low_risk_guardrail(
    fused_risk: np.ndarray,
    p_fail_soft: np.ndarray,
    p_fail_center_proxy: np.ndarray,
    tau_low: float,
    soft_ceiling: float,
    center_ceiling: float,
) -> np.ndarray:
    adjusted = np.clip(fused_risk.astype(float), 0.0, 1.0).copy()
    low_mask = adjusted < float(tau_low)
    blocked = low_mask & (
        (np.clip(p_fail_soft.astype(float), 0.0, 1.0) > float(soft_ceiling))
        | (np.clip(p_fail_center_proxy.astype(float), 0.0, 1.0) > float(center_ceiling))
    )
    adjusted[blocked] = float(tau_low)
    return adjusted


def choose_best_guardrail(
    calibration_frame: pd.DataFrame,
    fused_risk: np.ndarray,
    tau_low: float,
    tau_high: float,
    soft_candidates: list[float],
    center_candidates: list[float],
    max_false_clear_rate: float,
) -> tuple[dict[str, float], dict[str, float]]:
    best_choice: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None
    y_true = calibration_frame["y_true_out"].to_numpy(dtype=int)
    p_soft = calibration_frame["p_fail_soft"].to_numpy(dtype=float)
    p_center = calibration_frame["p_fail_center_proxy"].to_numpy(dtype=float)
    for soft_ceiling in soft_candidates:
        for center_ceiling in center_candidates:
            guarded = apply_low_risk_guardrail(
                fused_risk=fused_risk,
                p_fail_soft=p_soft,
                p_fail_center_proxy=p_center,
                tau_low=tau_low,
                soft_ceiling=float(soft_ceiling),
                center_ceiling=float(center_ceiling),
            )
            metrics = risk_metrics(y_true_out=y_true, risk_score=guarded, tau_low=tau_low, tau_high=tau_high)
            if metrics["false_clear_rate"] > float(max_false_clear_rate):
                continue
            choice = {
                "soft_ceiling": float(soft_ceiling),
                "center_ceiling": float(center_ceiling),
            }
            if best_metrics is None:
                best_choice = choice
                best_metrics = metrics
                continue
            if metrics["low_risk_rate"] > best_metrics["low_risk_rate"] + 1e-12:
                best_choice = choice
                best_metrics = metrics
                continue
            if abs(metrics["low_risk_rate"] - best_metrics["low_risk_rate"]) <= 1e-12 and metrics["brier"] < best_metrics["brier"] - 1e-12:
                best_choice = choice
                best_metrics = metrics
                continue
            if (
                abs(metrics["low_risk_rate"] - best_metrics["low_risk_rate"]) <= 1e-12
                and abs(metrics["brier"] - best_metrics["brier"]) <= 1e-12
                and metrics["hard_out_ap_diagnostic"] > best_metrics["hard_out_ap_diagnostic"]
            ):
                best_choice = choice
                best_metrics = metrics
    if best_choice is None or best_metrics is None:
        best_choice = {
            "soft_ceiling": float(min(soft_candidates)),
            "center_ceiling": float(min(center_candidates)),
        }
        guarded = apply_low_risk_guardrail(
            fused_risk=fused_risk,
            p_fail_soft=p_soft,
            p_fail_center_proxy=p_center,
            tau_low=tau_low,
            soft_ceiling=best_choice["soft_ceiling"],
            center_ceiling=best_choice["center_ceiling"],
        )
        best_metrics = risk_metrics(y_true_out=y_true, risk_score=guarded, tau_low=tau_low, tau_high=tau_high)
        best_metrics["fallback_due_to_no_feasible_candidate"] = 1.0
    else:
        best_metrics["fallback_due_to_no_feasible_candidate"] = 0.0
    return best_choice, best_metrics


def fit_soft_midrank_model(
    p_center_cal: np.ndarray,
    y_cal_out: np.ndarray,
    mid_mask_cal: np.ndarray,
    min_samples: int,
) -> tuple[str, Any]:
    mid_mask_cal = mid_mask_cal.astype(bool)
    if int(np.sum(mid_mask_cal)) < int(min_samples):
        return "identity", None
    x_mid = np.clip(p_center_cal[mid_mask_cal].astype(float), 0.0, 1.0)
    y_mid = y_cal_out[mid_mask_cal].astype(int)
    if len(np.unique(y_mid)) < 2 or len(np.unique(x_mid)) < 2:
        return "identity", None
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    model.fit(x_mid, y_mid.astype(float))
    return "isotonic", model


def apply_soft_midrank(
    p_soft: np.ndarray,
    p_center_proxy: np.ndarray,
    tau_low: float,
    tau_high: float,
    rank_model_kind: str,
    rank_model: Any,
) -> np.ndarray:
    adjusted = np.clip(p_soft.astype(float), 0.0, 1.0).copy()
    mid_mask = (adjusted >= float(tau_low)) & (adjusted < float(tau_high))
    if not mid_mask.any():
        return adjusted
    if rank_model_kind == "isotonic" and rank_model is not None:
        ranked = np.clip(rank_model.predict(np.clip(p_center_proxy[mid_mask].astype(float), 0.0, 1.0)), 0.0, 1.0)
    else:
        ranked = np.clip(p_center_proxy[mid_mask].astype(float), 0.0, 1.0)
    band_width = max(float(tau_high) - float(tau_low) - 1e-6, 1e-6)
    adjusted[mid_mask] = float(tau_low) + band_width * ranked
    return adjusted


def write_report(path: Path, summary_rows: list[dict[str, Any]], fold_selection_rows: list[dict[str, Any]], experiment_meta: dict[str, Any]) -> None:
    lines = [
        "# Soft + Centered Fusion MVP Summary",
        "",
        "## Scope",
        "",
        "- First-round MVP validation for a dual-head fusion layer.",
        "- This run uses a tail-calibration split inside each outer train fold.",
        "- Base heads remain frozen in structure; only the fusion layer is new.",
        "",
        "## Experiment Meta",
        "",
        json.dumps(experiment_meta, ensure_ascii=False, indent=2),
        "",
        "## Fold Fusion Choices",
        "",
        json.dumps(fold_selection_rows, ensure_ascii=False, indent=2),
        "",
        "## Aggregated Summary",
        "",
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MVP fusion between soft target risk head and centered desirability head.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    report_dir = resolve_path(config_path.parent, config["paths"]["report_dir"])
    soft_config_path = resolve_path(config_path.parent, config["paths"]["soft_config_path"])
    centered_config_path = resolve_path(config_path.parent, config["paths"]["centered_config_path"])
    if artifact_dir is None or report_dir is None or soft_config_path is None or centered_config_path is None:
        raise ValueError("artifact_dir, report_dir, soft_config_path, and centered_config_path must be configured.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    soft_config = load_config(soft_config_path)
    centered_config = load_config(centered_config_path)
    soft_frame = prepare_soft_frame(soft_config_path, soft_config, str(config["soft_head"]["variant_name"]))
    centered_frame, centered_priors, centered_choice = prepare_centered_frame(centered_config_path, centered_config)
    soft_aligned, centered_aligned = align_common_frame(soft_frame, centered_frame)

    outer_splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    fold_selection_rows: list[dict[str, Any]] = []

    soft_feature_columns = [column for column in soft_aligned.columns if "__" in column]
    centered_base_feature_columns = [
        column
        for column in centered_aligned.columns
        if "__" in column and "_dyn_" not in column and not column.startswith(RESERVED_AUX_PREFIXES)
    ]

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splitter.split(soft_aligned), start=1):
        soft_train_full = soft_aligned.iloc[train_idx].copy().reset_index(drop=True)
        soft_test = soft_aligned.iloc[test_idx].copy().reset_index(drop=True)
        centered_train_full = centered_aligned.iloc[train_idx].copy().reset_index(drop=True)
        centered_test = centered_aligned.iloc[test_idx].copy().reset_index(drop=True)

        base_idx, cal_idx = choose_calibration_tail_indices(
            train_len=len(soft_train_full),
            calibration_fraction=float(config["fusion"]["calibration_fraction"]),
            min_calibration_samples=int(config["fusion"]["min_calibration_samples"]),
            min_base_train_samples=int(config["fusion"]["min_base_train_samples"]),
        )

        soft_base_train = soft_train_full.iloc[base_idx].copy().reset_index(drop=True)
        soft_cal = soft_train_full.iloc[cal_idx].copy().reset_index(drop=True)
        centered_base_train = centered_train_full.iloc[base_idx].copy().reset_index(drop=True)
        centered_cal = centered_train_full.iloc[cal_idx].copy().reset_index(drop=True)

        soft_selected, _ = select_features_fold(
            train_x=soft_base_train[soft_feature_columns],
            train_y=soft_base_train[str(soft_config["label_fuzziness"]["target_name"])],
            task_type="regression",
            top_k=int(config["soft_head"]["top_k"]),
        )
        if not soft_selected:
            soft_selected = list(soft_feature_columns)

        soft_label = str(soft_config["label_fuzziness"]["target_name"])
        soft_train_df = soft_base_train[soft_selected + [soft_label]].copy()
        soft_cal_df = soft_cal[soft_selected + [soft_label]].copy()
        soft_test_df = soft_test[soft_selected + [soft_label]].copy()
        soft_model_path = artifact_dir / f"ag_soft_head_{run_id}_fold{fold_idx}"
        soft_predictor, soft_model_best = fit_predictor(
            train_df=soft_train_df,
            label=soft_label,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=soft_model_path,
            ag_config=soft_config["autogluon"],
        )
        p_soft_cal = soft_predictor.predict(soft_cal_df.drop(columns=[soft_label])).to_numpy(dtype=float)
        p_soft_test = soft_predictor.predict(soft_test_df.drop(columns=[soft_label])).to_numpy(dtype=float)
        p_soft_cal = np.clip(p_soft_cal, 0.0, 1.0)
        p_soft_test = np.clip(p_soft_test, 0.0, 1.0)

        centered_combo = next(spec for spec in combo_specs("centered_desirability", centered_priors) if spec["name"] == centered_choice["combo_name"])
        centered_train_x, centered_cal_x = compose_fold_features(centered_base_train, centered_cal, centered_base_feature_columns, centered_priors, centered_combo)
        _, centered_test_x = compose_fold_features(centered_base_train, centered_test, centered_base_feature_columns, centered_priors, centered_combo)
        centered_train_x = drop_duplicate_feature_columns(centered_train_x)
        centered_cal_x = drop_duplicate_feature_columns(centered_cal_x)
        centered_test_x = drop_duplicate_feature_columns(centered_test_x)
        centered_train_clean, centered_cal_clean, centered_cleaned_cols = preclean_features(
            centered_train_x,
            centered_cal_x,
            max_missing_ratio=float(centered_config["preclean"]["max_missing_ratio"]),
            unique_threshold=int(centered_config["preclean"]["near_constant_unique_threshold"]),
        )
        _, centered_test_clean, _ = preclean_features(
            centered_train_x,
            centered_test_x,
            max_missing_ratio=float(centered_config["preclean"]["max_missing_ratio"]),
            unique_threshold=int(centered_config["preclean"]["near_constant_unique_threshold"]),
        )
        centered_selected = select_features_fold(
            train_x=centered_train_clean,
            train_y=centered_base_train["target_centered_desirability"],
            task_type="regression",
            top_k=int(centered_choice["top_k"]),
        )[0]
        if not centered_selected:
            centered_selected = centered_cleaned_cols

        centered_train_sel = centered_train_clean[centered_selected].copy()
        centered_cal_sel = centered_cal_clean[centered_selected].copy()
        centered_test_sel = centered_test_clean[centered_selected].copy()
        centered_model_path = artifact_dir / f"ag_centered_head_{run_id}_fold{fold_idx}"
        centered_predictor, centered_model_best = fit_predictor(
            train_df=pd.concat([centered_train_sel, centered_base_train[["target_centered_desirability"]]], axis=1).copy(),
            label="target_centered_desirability",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=centered_model_path,
            ag_config=centered_config["autogluon"],
        )
        q_center_cal = centered_predictor.predict(centered_cal_sel).to_numpy(dtype=float)
        q_center_test = centered_predictor.predict(centered_test_sel).to_numpy(dtype=float)
        p_center_cal = 1.0 - np.clip(q_center_cal, 0.0, 1.0)
        p_center_test = 1.0 - np.clip(q_center_test, 0.0, 1.0)

        y_cal_out = soft_cal["is_out_of_spec"].to_numpy(dtype=int)
        y_test_out = soft_test["is_out_of_spec"].to_numpy(dtype=int)
        calibration_frame = build_calibration_features(p_soft_cal, p_center_cal)
        calibration_frame["y_true_out"] = y_cal_out

        best_weight, weight_metrics = choose_best_fixed_weight(calibration_frame, [float(x) for x in config["fusion"]["fixed_weight_candidates"]])
        fixed_cal = np.clip(
            best_weight * p_soft_cal + (1.0 - best_weight) * p_center_cal,
            0.0,
            1.0,
        )
        fixed_test = np.clip(best_weight * p_soft_test + (1.0 - best_weight) * p_center_test, 0.0, 1.0)

        logistic_model = fit_logistic_fusion(calibration_frame)
        test_feature_frame = build_calibration_features(p_soft_test, p_center_test)
        if logistic_model is None:
            logistic_test = fixed_test.copy()
            logistic_info: dict[str, Any] = {"fallback_to_fixed_weight": True}
            logistic_cal = fixed_cal.copy()
        else:
            logistic_cal = logistic_model.predict_proba(calibration_frame[["p_fail_soft", "p_fail_center_proxy", "disagreement"]].to_numpy(dtype=float))[:, 1]
            logistic_test = logistic_model.predict_proba(test_feature_frame[["p_fail_soft", "p_fail_center_proxy", "disagreement"]].to_numpy(dtype=float))[:, 1]
            logistic_info = {
                "fallback_to_fixed_weight": False,
                "intercept": float(logistic_model.intercept_[0]),
                "coef_p_fail_soft": float(logistic_model.coef_[0][0]),
                "coef_p_fail_center_proxy": float(logistic_model.coef_[0][1]),
                "coef_disagreement": float(logistic_model.coef_[0][2]),
            }

        logistic_guardrail_choice = None
        logistic_guardrail_metrics = None
        logistic_guarded_test = None
        if "guardrail" in config.get("fusion", {}):
            guardrail_cfg = config["fusion"]["guardrail"]
            logistic_guardrail_choice, logistic_guardrail_metrics = choose_best_guardrail(
                calibration_frame=calibration_frame,
                fused_risk=np.clip(logistic_cal.astype(float), 0.0, 1.0),
                tau_low=float(config["fusion"]["tau_low"]),
                tau_high=float(config["fusion"]["tau_high"]),
                soft_candidates=[float(x) for x in guardrail_cfg["soft_ceiling_candidates"]],
                center_candidates=[float(x) for x in guardrail_cfg["center_ceiling_candidates"]],
                max_false_clear_rate=float(guardrail_cfg["max_false_clear_rate"]),
            )
            logistic_guarded_test = apply_low_risk_guardrail(
                fused_risk=np.clip(logistic_test.astype(float), 0.0, 1.0),
                p_fail_soft=p_soft_test,
                p_fail_center_proxy=p_center_test,
                tau_low=float(config["fusion"]["tau_low"]),
                soft_ceiling=float(logistic_guardrail_choice["soft_ceiling"]),
                center_ceiling=float(logistic_guardrail_choice["center_ceiling"]),
            )

        tau_low = float(config["fusion"]["tau_low"])
        tau_high = float(config["fusion"]["tau_high"])
        mid_mask_cal = (np.clip(p_soft_cal.astype(float), 0.0, 1.0) >= tau_low) & (np.clip(p_soft_cal.astype(float), 0.0, 1.0) < tau_high)
        mid_mask_test = (np.clip(p_soft_test.astype(float), 0.0, 1.0) >= tau_low) & (np.clip(p_soft_test.astype(float), 0.0, 1.0) < tau_high)
        midrank_kind = "disabled"
        midrank_model = None
        soft_midrank_test = None
        midrank_info = None
        if "midrank" in config.get("fusion", {}):
            midrank_cfg = config["fusion"]["midrank"]
            midrank_kind, midrank_model = fit_soft_midrank_model(
                p_center_cal=p_center_cal,
                y_cal_out=y_cal_out,
                mid_mask_cal=mid_mask_cal,
                min_samples=int(midrank_cfg["min_calibration_samples"]),
            )
            soft_midrank_test = apply_soft_midrank(
                p_soft=p_soft_test,
                p_center_proxy=p_center_test,
                tau_low=tau_low,
                tau_high=tau_high,
                rank_model_kind=midrank_kind,
                rank_model=midrank_model,
            )
            midrank_info = {
                "kind": midrank_kind,
                "calibration_mid_sample_count": int(np.sum(mid_mask_cal)),
                "test_mid_sample_count": int(np.sum(mid_mask_test)),
            }

        methods = {
            "soft_only": p_soft_test,
            "center_proxy_only": p_center_test,
            "fixed_weight_fused": fixed_test,
            "logistic_fused": np.clip(logistic_test.astype(float), 0.0, 1.0),
        }
        if logistic_guarded_test is not None:
            methods["logistic_fused_guardrailed"] = np.clip(logistic_guarded_test.astype(float), 0.0, 1.0)
        if soft_midrank_test is not None:
            methods["soft_midrank_centered"] = np.clip(soft_midrank_test.astype(float), 0.0, 1.0)
        for method_name, scores in methods.items():
            metrics = risk_metrics(
                y_true_out=y_test_out,
                risk_score=scores,
                tau_low=tau_low,
                tau_high=tau_high,
            )
            metrics.update(subset_risk_metrics(y_true_out=y_test_out, risk_score=scores, mask=mid_mask_test, prefix="soft_mid_band"))
            metric_rows.append(
                {
                    "fold": int(fold_idx),
                    "method": method_name,
                    "samples_test": int(len(soft_test)),
                    **metrics,
                }
            )

        for row_idx in range(len(soft_test)):
            prediction_rows.append(
                {
                    "fold": int(fold_idx),
                    "decision_time": soft_test.loc[row_idx, "decision_time"],
                    "sample_time": soft_test.loc[row_idx, "sample_time"],
                    "t90": float(soft_test.loc[row_idx, "t90"]),
                    "is_in_spec": int(soft_test.loc[row_idx, "is_in_spec"]),
                    "is_out_of_spec": int(soft_test.loc[row_idx, "is_out_of_spec"]),
                    "target_soft_out_of_spec_probability": float(soft_test.loc[row_idx, soft_label]),
                    "target_centered_desirability": float(centered_test.loc[row_idx, "target_centered_desirability"]),
                    "p_fail_soft": float(p_soft_test[row_idx]),
                    "q_center": float(q_center_test[row_idx]),
                    "p_fail_center_proxy": float(p_center_test[row_idx]),
                    "p_fail_fixed_weight": float(fixed_test[row_idx]),
                    "p_fail_logistic": float(np.clip(logistic_test[row_idx], 0.0, 1.0)),
                    "p_fail_logistic_guardrailed": float(np.clip(logistic_guarded_test[row_idx], 0.0, 1.0)) if logistic_guarded_test is not None else float("nan"),
                    "p_fail_soft_midrank_centered": float(np.clip(soft_midrank_test[row_idx], 0.0, 1.0)) if soft_midrank_test is not None else float("nan"),
                }
            )

        fold_selection_rows.append(
            {
                "fold": int(fold_idx),
                "soft_model_best": soft_model_best,
                "centered_model_best": centered_model_best,
                "soft_selected_feature_count": int(len(soft_selected)),
                "centered_selected_feature_count": int(len(centered_selected)),
                "calibration_samples": int(len(soft_cal)),
                "base_train_samples": int(len(soft_base_train)),
                "fixed_weight": float(best_weight),
                "fixed_weight_selection": weight_metrics,
                "logistic_info": logistic_info,
                "logistic_guardrail_choice": logistic_guardrail_choice,
                "logistic_guardrail_calibration_metrics": logistic_guardrail_metrics,
                "midrank_info": midrank_info,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    summary_rows: list[dict[str, Any]] = []
    for method_name, method_df in metrics_df.groupby("method", sort=False):
        row = {
            "method": method_name,
            "mean_brier": float(method_df["brier"].mean()),
            "mean_hard_out_ap_diagnostic": float(method_df["hard_out_ap_diagnostic"].mean()),
            "mean_hard_out_auc_diagnostic": float(method_df["hard_out_auc_diagnostic"].mean()),
            "mean_false_clear_rate": float(method_df["false_clear_rate"].mean()),
            "mean_low_risk_rate": float(method_df["low_risk_rate"].mean()),
            "mean_high_risk_rate": float(method_df["high_risk_rate"].mean()),
            "mean_retest_rate": float(method_df["retest_rate"].mean()),
            "mean_soft_mid_band_sample_rate": float(method_df["soft_mid_band_sample_rate"].mean()),
            "mean_soft_mid_band_positive_rate": float(method_df["soft_mid_band_positive_rate"].mean()),
            "mean_soft_mid_band_ap_diagnostic": float(method_df["soft_mid_band_ap_diagnostic"].mean()),
            "mean_soft_mid_band_auc_diagnostic": float(method_df["soft_mid_band_auc_diagnostic"].mean()),
        }
        summary_rows.append(row)
    summary_rows = sorted(summary_rows, key=lambda row: (row["mean_brier"], -row["mean_hard_out_ap_diagnostic"], row["method"]))

    experiment_meta = {
        "mvp_type": "tail_calibration_fusion",
        "common_sample_count": int(len(soft_aligned)),
        "soft_variant_name": str(config["soft_head"]["variant_name"]),
        "soft_top_k": int(config["soft_head"]["top_k"]),
        "centered_combo_name": str(centered_choice["combo_name"]),
        "centered_top_k": int(centered_choice["top_k"]),
        "centered_stage1_variant": str(centered_priors["stage1_variant"]),
        "tau_low": float(config["fusion"]["tau_low"]),
        "tau_high": float(config["fusion"]["tau_high"]),
        "calibration_fraction": float(config["fusion"]["calibration_fraction"]),
    }

    predictions_path = artifact_dir / "soft_centered_fusion_mvp_predictions.csv"
    metrics_path = artifact_dir / "soft_centered_fusion_mvp_fold_metrics.csv"
    summary_path = artifact_dir / "soft_centered_fusion_mvp_summary.json"
    report_path = report_dir / "soft_centered_fusion_mvp_summary.md"
    fold_selection_path = artifact_dir / "soft_centered_fusion_mvp_fold_selection.json"

    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fold_selection_path.write_text(json.dumps(fold_selection_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, summary_rows, fold_selection_rows, experiment_meta)
    print(
        json.dumps(
            {
                "predictions_path": str(predictions_path),
                "metrics_path": str(metrics_path),
                "summary_path": str(summary_path),
                "fold_selection_path": str(fold_selection_path),
                "report_path": str(report_path),
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
