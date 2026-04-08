from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_stage2_feature_engineering import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    regression_metrics,
    resolve_path,
    select_features_fold,
)
from run_autogluon_soft_probability_weak_compression_search import build_variant_snapshot


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_centered_same_fe_as_soft.yaml"


def build_centered_on_soft_reference(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshot, audit = build_variant_snapshot(config_path, config, "current_whole_window_ref")
    center = float(config["target_spec"]["center"])
    tolerance = float(config["target_spec"]["tolerance"])
    snapshot = snapshot.copy()
    snapshot["target_centered_desirability"] = np.maximum(0.0, 1.0 - (snapshot["t90"] - center).abs() / tolerance)
    audit = {
        **audit,
        "task_name": "centered_desirability",
        "shared_x_recipe": "whole_window_range_position",
    }
    return snapshot.sort_values("sample_time").reset_index(drop=True), audit


def run_experiment(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    snapshot, snapshot_audit = build_centered_on_soft_reference(config_path, config)
    feature_columns = [column for column in snapshot.columns if "__" in column]
    label = "target_centered_desirability"
    top_k = int(config["selection"]["top_k"])
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    if artifact_dir is None:
        raise ValueError("artifact_dir must be configured.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    selected_features_fold1: list[str] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
        train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
        test = snapshot.iloc[test_idx].copy().reset_index(drop=True)
        selected_features, _ = select_features_fold(
            train_x=train[feature_columns],
            train_y=train[label],
            task_type="regression",
            top_k=top_k,
        )
        if fold_idx == 1:
            selected_features_fold1 = list(selected_features)

        train_df = train[selected_features + [label]].copy()
        test_df = test[selected_features + [label]].copy()

        baseline = make_regression_baseline()
        baseline.fit(train_df[selected_features], train_df[label].to_numpy(dtype=float))
        baseline_pred = baseline.predict(test_df[selected_features]).astype(float)
        baseline_metrics = regression_metrics(
            test_df[label].to_numpy(dtype=float),
            baseline_pred,
            test["is_in_spec"].to_numpy(dtype=int),
        )

        model_path = artifact_dir / f"ag_centered_same_fe_soft_{run_id}_fold{fold_idx}"
        framework_pred, model_best = fit_autogluon_fold(
            train_df=train_df,
            test_df=test_df,
            label=label,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )
        framework_metrics = regression_metrics(
            test_df[label].to_numpy(dtype=float),
            framework_pred.astype(float),
            test["is_in_spec"].to_numpy(dtype=int),
        )

        rows.append(
            {
                "framework": "simple_baseline_centered_same_fe_as_soft",
                "fold": int(fold_idx),
                "feature_count_selected": int(len(selected_features)),
                **baseline_metrics,
            }
        )
        rows.append(
            {
                "framework": "autogluon_centered_same_fe_as_soft",
                "fold": int(fold_idx),
                "feature_count_selected": int(len(selected_features)),
                "autogluon_model_best": model_best,
                **framework_metrics,
            }
        )

        fold_summaries.append(
            {
                "fold": int(fold_idx),
                "baseline": baseline_metrics,
                "autogluon": framework_metrics,
            }
        )

    results = pd.DataFrame(rows)
    baseline_rows = results[results["framework"] == "simple_baseline_centered_same_fe_as_soft"]
    ag_rows = results[results["framework"] == "autogluon_centered_same_fe_as_soft"]
    summary = {
        "task_name": "centered_desirability",
        "shared_x_recipe": "whole_window_range_position",
        "sample_count": int(len(snapshot)),
        "raw_feature_count": int(len(feature_columns)),
        "selected_feature_count": int(np.mean(results["feature_count_selected"])),
        "selected_features_fold1": selected_features_fold1,
        "baseline_mean_mae": float(baseline_rows["mae"].mean()),
        "autogluon_mean_mae": float(ag_rows["mae"].mean()),
        "baseline_mean_rmse": float(baseline_rows["rmse"].mean()),
        "autogluon_mean_rmse": float(ag_rows["rmse"].mean()),
        "baseline_mean_in_spec_auc_from_desirability": float(baseline_rows["in_spec_auc_from_desirability"].mean()),
        "autogluon_mean_in_spec_auc_from_desirability": float(ag_rows["in_spec_auc_from_desirability"].mean()),
        "positive_signal": bool(ag_rows["mae"].mean() < baseline_rows["mae"].mean()),
        "fold_summaries": fold_summaries,
    }
    return results, summary, snapshot_audit


def write_report(path: Path, summary: dict[str, Any], snapshot_audit: dict[str, Any]) -> None:
    lines = [
        "# Centered Desirability With Soft-Target X Recipe",
        "",
        "## Purpose",
        "",
        "- Re-run centered_desirability using the same X-side feature engineering recipe that currently performs best on the soft target branch.",
        "- This creates a direct same-X comparison between the two branches.",
        "",
        "## Summary",
        "",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "",
        "## Snapshot Audit",
        "",
        json.dumps(snapshot_audit, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run centered_desirability on the same X recipe as the soft target branch.")
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

    results, summary, snapshot_audit = run_experiment(config_path, config)

    (artifact_dir / "centered_same_fe_as_soft_results.csv").write_text(results.to_csv(index=False), encoding="utf-8")
    (artifact_dir / "centered_same_fe_as_soft_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(report_dir / "centered_same_fe_as_soft_summary.md", summary, snapshot_audit)


if __name__ == "__main__":
    main()
