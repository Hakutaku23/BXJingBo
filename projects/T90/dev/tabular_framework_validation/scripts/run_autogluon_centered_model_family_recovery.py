from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
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
from run_autogluon_stage4_interactions import build_interaction_frame
from run_autogluon_stage5_quality import build_quality_table
from run_autogluon_stage7_final_selection import preclean_features, supervised_select


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_centered_model_family_recovery.yaml"


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def build_current_centered_frame(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    labeled = build_label_frame(config_path, config)
    dcs = load_dcs_frame(
        resolve_path(config_path.parent, config["paths"]["dcs_main_path"]),
        resolve_path(config_path.parent, config["paths"].get("dcs_supplemental_path")),
    )
    tau_minutes = int(config["snapshot"]["lag_minutes"])
    total_window_minutes = int(config["snapshot"]["total_window_minutes"])
    min_points_per_window = int(config["snapshot"]["min_points_per_window"])
    stats = list(config["snapshot"]["stats"])

    snapshot = build_stage2_table(
        labeled_samples=labeled,
        dcs=dcs,
        tau_minutes=tau_minutes,
        window_minutes=total_window_minutes,
        stats=stats,
        enabled_dynamic_features=[],
        min_points_per_window=min_points_per_window,
    )
    if snapshot.empty:
        raise ValueError("Centered snapshot is empty.")

    interaction_frame = build_interaction_frame(
        snapshot=snapshot,
        tau_minutes=tau_minutes,
        window_minutes=total_window_minutes,
        package_name=str(config["packages"]["interaction_package"]),
    )
    interaction_frame = pd.concat([snapshot[["decision_time"]].copy(), interaction_frame], axis=1)
    quality_table = build_quality_table(
        labeled_samples=labeled,
        dcs=dcs,
        tau_minutes=tau_minutes,
        window_minutes=total_window_minutes,
        enabled_features=list(config["packages"]["quality_features"]),
        min_points_per_window=min_points_per_window,
    )
    frame = snapshot.merge(interaction_frame, on="decision_time", how="inner").merge(
        quality_table,
        on="decision_time",
        how="inner",
    )
    return frame.sort_values("decision_time").reset_index(drop=True)


def trained_model_names(leaderboard: pd.DataFrame) -> list[str]:
    if leaderboard is None or leaderboard.empty or "model" not in leaderboard.columns:
        return []
    return leaderboard["model"].astype(str).tolist()


def trained_family_presence(model_names: list[str]) -> dict[str, bool]:
    joined = " | ".join(model_names).upper()
    return {
        "GBM": "LIGHTGBM" in joined or "GBM" in joined,
        "XGB": "XGB" in joined,
        "CAT": "CAT" in joined,
        "RF": "RANDOMFOREST" in joined or "RF" in joined,
        "XT": "EXTRATREES" in joined or "XT" in joined,
        "NN_TORCH": "NN_TORCH" in joined or "TORCH" in joined,
        "FASTAI": "FASTAI" in joined,
        "WEIGHTED_ENSEMBLE": "WEIGHTEDENSEMBLE" in joined or "WEIGHTED_ENSEMBLE" in joined,
    }


def evaluate_pool(
    pool_name: str,
    pool_hyperparameters: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    artifact_dir: Path,
    splitter: TimeSeriesSplit,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame | None]:
    rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, float]] = []
    raw_feature_count = 0
    cleaned_feature_count = 0
    selected_feature_count = 0
    selected_features_fold1: list[str] = []
    fold1_leaderboard: pd.DataFrame | None = None

    label = str(config["task"]["label"])
    top_k = int(config["task"]["top_k"])

    ag_config = dict(config["autogluon"])
    ag_config["hyperparameters"] = pool_hyperparameters

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(frame), start=1):
        train = frame.iloc[train_idx].copy().reset_index(drop=True)
        test = frame.iloc[test_idx].copy().reset_index(drop=True)

        train_x = train[feature_columns]
        test_x = test[feature_columns]
        raw_feature_count = int(train_x.shape[1])

        train_clean, test_clean, cleaned_cols = preclean_features(
            train_x,
            test_x,
            max_missing_ratio=float(config["preclean"]["max_missing_ratio"]),
            unique_threshold=int(config["preclean"]["near_constant_unique_threshold"]),
        )
        cleaned_feature_count = int(len(cleaned_cols))

        y_train = train[label].to_numpy(dtype=float)
        y_test = test[label].to_numpy(dtype=float)
        selected_cols = supervised_select(
            train_clean,
            y_train,
            task_name="centered_desirability",
            top_k=top_k,
        )
        if not selected_cols:
            selected_cols = list(train_clean.columns)
        selected_feature_count = int(len(selected_cols))
        if fold_idx == 1:
            selected_features_fold1 = list(selected_cols)

        baseline = make_regression_baseline()
        baseline.fit(train_clean[selected_cols], y_train)
        baseline_pred = baseline.predict(test_clean[selected_cols]).astype(float)
        baseline_metric = regression_metrics(
            y_true=y_test,
            pred=baseline_pred,
            in_spec_flag=test["is_in_spec"].to_numpy(dtype=int),
        )

        model_path = artifact_dir / f"ag_centered_model_recovery_{pool_name}_{run_id}_fold{fold_idx}"
        ag_pred, model_best = fit_autogluon_fold(
            train_df=pd.concat([train_clean[selected_cols].copy(), train[[label]].copy()], axis=1),
            test_df=pd.concat([test_clean[selected_cols].copy(), test[[label]].copy()], axis=1),
            label=label,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=ag_config,
        )
        ag_metric = regression_metrics(
            y_true=y_test,
            pred=ag_pred.astype(float),
            in_spec_flag=test["is_in_spec"].to_numpy(dtype=int),
        )

        if fold_idx == 1:
            predictor = TabularPredictor.load(str(model_path))
            fold1_leaderboard = predictor.leaderboard(
                pd.concat([test_clean[selected_cols].copy(), test[[label]].copy()], axis=1),
                silent=True,
            )

        for framework_name, metrics in (("simple_baseline", baseline_metric), ("autogluon", ag_metric)):
            row = {
                "pool_name": pool_name,
                "fold": fold_idx,
                "framework": framework_name,
                "raw_feature_count": raw_feature_count,
                "cleaned_feature_count": cleaned_feature_count,
                "selected_feature_count": selected_feature_count,
            }
            row.update(metrics)
            rows.append(row)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "baseline_mae": float(baseline_metric["mae"]),
                "autogluon_mae": float(ag_metric["mae"]),
                "autogluon_rmse": float(ag_metric["rmse"]),
                "autogluon_rank_correlation": float(ag_metric["rank_correlation"]),
                "autogluon_in_spec_auc": float(ag_metric["in_spec_auc_from_desirability"]),
                "model_best": str(model_best),
            }
        )

    leaderboard_models = trained_model_names(fold1_leaderboard) if fold1_leaderboard is not None else []
    summary = {
        "pool_name": pool_name,
        "sample_count": int(len(frame)),
        "raw_feature_count": raw_feature_count,
        "cleaned_feature_count": cleaned_feature_count,
        "selected_feature_count": selected_feature_count,
        "selected_features_fold1": selected_features_fold1,
        "baseline_mean_mae": _safe_mean([item["baseline_mae"] for item in fold_metrics]),
        "autogluon_mean_mae": _safe_mean([item["autogluon_mae"] for item in fold_metrics]),
        "autogluon_mean_rmse": _safe_mean([item["autogluon_rmse"] for item in fold_metrics]),
        "autogluon_mean_rank_correlation": _safe_mean([item["autogluon_rank_correlation"] for item in fold_metrics]),
        "autogluon_mean_in_spec_auc": _safe_mean([item["autogluon_in_spec_auc"] for item in fold_metrics]),
        "trained_models_fold1": leaderboard_models,
        "trained_family_presence_fold1": trained_family_presence(leaderboard_models),
    }
    return rows, summary, fold1_leaderboard


def write_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        path.write_text("# Centered Model Family Recovery\n\nNo valid model pool was evaluated.\n", encoding="utf-8")
        return
    lines = [
        "# Centered Model Family Recovery",
        "",
        "## Scope",
        "",
        "- Task: centered_desirability only",
        "- Input recipe fixed to the current best centered line",
        "- Only the AutoGluon model family pool is changed",
        "",
        "## Ranked Summary Rows",
        "",
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover and recheck broader AutoGluon model families on the best centered line.")
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

    frame = build_current_centered_frame(config_path, config)
    feature_columns = [column for column in frame.columns if "__" in column]
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    leaderboard_tables: list[pd.DataFrame] = []

    for pool in config["model_pools"]:
        pool_name = str(pool["name"])
        pool_hyperparameters = dict(pool["hyperparameters"])
        rows, summary, leaderboard = evaluate_pool(
            pool_name=pool_name,
            pool_hyperparameters=pool_hyperparameters,
            frame=frame,
            feature_columns=feature_columns,
            config=config,
            artifact_dir=artifact_dir,
            splitter=splitter,
            run_id=run_id,
        )
        all_rows.extend(rows)
        summary_rows.append(summary)
        if leaderboard is not None and not leaderboard.empty:
            board = leaderboard.copy()
            board.insert(0, "pool_name", pool_name)
            leaderboard_tables.append(board)

    current_ref_mae = min(float(row["autogluon_mean_mae"]) for row in summary_rows)
    for row in summary_rows:
        row["current_best_pool_mae"] = current_ref_mae
        row["delta_vs_best_pool"] = float(row["autogluon_mean_mae"] - current_ref_mae)
        row["is_best_pool"] = bool(row["autogluon_mean_mae"] == current_ref_mae)

    summary_rows = sorted(summary_rows, key=lambda item: (float(item["autogluon_mean_mae"]), str(item["pool_name"])))
    best_pool = summary_rows[0] if summary_rows else None

    results_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)
    leaderboard_df = pd.concat(leaderboard_tables, ignore_index=True) if leaderboard_tables else pd.DataFrame()

    results_df.to_csv(artifact_dir / "centered_model_family_recovery_results.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(artifact_dir / "centered_model_family_recovery_summary_rows.csv", index=False, encoding="utf-8-sig")
    leaderboard_df.to_csv(artifact_dir / "centered_model_family_recovery_fold1_leaderboards.csv", index=False, encoding="utf-8-sig")

    summary_payload = {
        "phase": str(config["phase"]),
        "task_name": str(config["task"]["name"]),
        "best_pool": best_pool,
        "summary_rows": summary_rows,
    }
    (artifact_dir / "centered_model_family_recovery_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(report_dir / "centered_model_family_recovery_summary.md", summary_rows)


if __name__ == "__main__":
    main()
