from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_autogluon_stage1_quickcheck import (
    binary_metrics,
    build_stage1_snapshot_table,
    fit_autogluon_fold,
    load_config,
    make_binary_baseline,
    make_regression_baseline,
    regression_metrics,
    resolve_path,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage0_baseline.yaml"


def make_multiclass_baseline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1500,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def align_probability_frame(
    proba: pd.DataFrame | np.ndarray,
    class_labels: list[int],
    observed_labels: list[int] | np.ndarray,
) -> np.ndarray:
    if isinstance(proba, pd.DataFrame):
        aligned = proba.reindex(columns=class_labels, fill_value=0.0).to_numpy(dtype=float)
    else:
        observed = list(observed_labels)
        frame = pd.DataFrame(proba, columns=observed)
        aligned = frame.reindex(columns=class_labels, fill_value=0.0).to_numpy(dtype=float)
    row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return aligned / row_sum


def multiclass_metrics(y_true: np.ndarray, proba: np.ndarray, class_labels: list[int]) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "multiclass_log_loss": float(log_loss(y_true, proba, labels=class_labels)),
    }


def fit_autogluon_multiclass(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label: str,
    model_path: Path,
    ag_config: dict[str, Any],
) -> tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame]:
    predictor = TabularPredictor(
        label=label,
        problem_type="multiclass",
        eval_metric="log_loss",
        path=str(model_path),
        verbosity=int(ag_config["verbosity"]),
    )
    predictor.fit(
        train_data=train_df,
        presets=ag_config["presets"],
        time_limit=int(ag_config["time_limit_seconds"]),
        hyperparameters=ag_config.get("hyperparameters"),
    )
    test_x = test_df.drop(columns=[label])
    proba = predictor.predict_proba(test_x)
    leaderboard = predictor.leaderboard(test_df, silent=True)
    feature_importance = predictor.feature_importance(test_df, silent=True)
    return proba, str(predictor.model_best), leaderboard, feature_importance


def build_stage0_snapshot_table(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshot, audit = build_stage1_snapshot_table(config_path, config)
    if "target_five_bin" in snapshot.columns:
        snapshot["target_five_bin"] = snapshot["target_five_bin"].astype(int)
    return snapshot, audit


def build_feature_catalog(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in snapshot.columns:
        if "__" not in column:
            continue
        sensor, stat = column.split("__", 1)
        rows.append(
            {
                "feature_name": column,
                "sensor": sensor,
                "stat": stat,
                "family": "stage0_baseline_snapshot",
            }
        )
    return pd.DataFrame(rows).sort_values(["sensor", "stat"]).reset_index(drop=True)


def write_stage0_audit(
    path: Path,
    config: dict[str, Any],
    stage_summary: dict[str, Any],
    snapshot_audit: dict[str, Any],
    feature_catalog: pd.DataFrame,
) -> None:
    lines = [
        "# Stage 0 Baseline Audit",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- “Uncleaned” refers to the source state, not to a ban on all preprocessing.",
        "- Stage 0 only uses minimal necessary preprocessing and the simplest causal snapshot baseline.",
        "",
        "## Stage 0 Processing",
        "",
        "- unified decision_time based on sample_time alignment",
        "- 120-minute causal window snapshot",
        "- stats: mean / std / min / max / last / range / delta",
        "- drop all-NaN features",
        "- drop constant features",
        "",
        "## Labels",
        "",
        "- low_risk: y < 8.2",
        "- high_risk: y > 8.7",
        "- centered_desirability: q(y) = max(0, 1 - |y - 8.45| / 0.25)",
        "- five_bin: [-inf, 8.0, 8.2, 8.7, 8.9, inf]",
        "",
        "## Validation",
        "",
        f"- TimeSeriesSplit(n_splits={config['validation']['n_splits']})",
        "- Baseline and AutoGluon use the same split.",
        "",
        "## Snapshot Audit",
        "",
        json.dumps(snapshot_audit, ensure_ascii=False, indent=2),
        "",
        "## Stage Summary",
        "",
        json.dumps(stage_summary, ensure_ascii=False, indent=2),
        "",
        "## Feature Catalog Summary",
        "",
        f"- feature_count: {len(feature_catalog)}",
        f"- sensor_count: {feature_catalog['sensor'].nunique() if not feature_catalog.empty else 0}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage0(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    snapshot, snapshot_audit = build_stage0_snapshot_table(config_path, config)
    feature_columns = [column for column in snapshot.columns if "__" in column]
    feature_catalog = build_feature_catalog(snapshot)
    all_five_bin_labels = sorted(snapshot["target_five_bin"].dropna().astype(int).unique().tolist()) if "target_five_bin" in snapshot.columns else []
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    if artifact_dir is None:
        raise ValueError("artifact_dir must be configured.")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    stage_summary: dict[str, Any] = {"tasks": {}, "stage0_positive_signal": False}
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    leaderboards: dict[str, pd.DataFrame] = {}
    feature_importances: dict[str, pd.DataFrame] = {}

    task_defs: list[dict[str, Any]] = []
    if config["tasks"]["low_risk"]["enabled"]:
        task_defs.append({"task_name": "low_risk", "label": "target_low_risk", "type": "binary", "eval_metric": "average_precision"})
    if config["tasks"]["high_risk"]["enabled"]:
        task_defs.append({"task_name": "high_risk", "label": "target_high_risk", "type": "binary", "eval_metric": "average_precision"})
    if config["tasks"]["centered_desirability"]["enabled"]:
        task_defs.append({"task_name": "centered_desirability", "label": "target_centered_desirability", "type": "regression", "eval_metric": "root_mean_squared_error"})
    if config["tasks"]["five_bin"]["enabled"]:
        task_defs.append({"task_name": "five_bin", "label": "target_five_bin", "type": "multiclass", "eval_metric": "log_loss"})

    for task in task_defs:
        task_name = task["task_name"]
        label = task["label"]
        fold_summaries: list[dict[str, Any]] = []
        baseline_agg: list[dict[str, Any]] = []
        ag_agg: list[dict[str, Any]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
            train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
            test = snapshot.iloc[test_idx].copy().reset_index(drop=True)
            train_df = train[feature_columns + [label]].copy()
            test_df = test[feature_columns + [label]].copy()

            if task["type"] == "binary":
                baseline = make_binary_baseline()
                baseline.fit(train_df[feature_columns], train_df[label].to_numpy(dtype=int))
                baseline_scores = baseline.predict_proba(test_df[feature_columns])[:, 1]
                baseline_metric = binary_metrics(test_df[label].to_numpy(dtype=int), baseline_scores)

                model_path = artifact_dir / f"ag_stage0_{task_name}_{run_id}_fold{fold_idx}"
                ag_scores, model_best = fit_autogluon_fold(
                    train_df=train_df,
                    test_df=test_df,
                    label=label,
                    problem_type="binary",
                    eval_metric=task["eval_metric"],
                    model_path=model_path,
                    ag_config=config["autogluon"],
                )
                ag_metric = binary_metrics(test_df[label].to_numpy(dtype=int), ag_scores)
            elif task["type"] == "regression":
                baseline = make_regression_baseline()
                baseline.fit(train_df[feature_columns], train_df[label].to_numpy(dtype=float))
                baseline_scores = baseline.predict(test_df[feature_columns]).astype(float)
                baseline_metric = regression_metrics(
                    test_df[label].to_numpy(dtype=float),
                    baseline_scores,
                    test["is_in_spec"].to_numpy(dtype=int),
                )

                model_path = artifact_dir / f"ag_stage0_{task_name}_{run_id}_fold{fold_idx}"
                ag_scores, model_best = fit_autogluon_fold(
                    train_df=train_df,
                    test_df=test_df,
                    label=label,
                    problem_type="regression",
                    eval_metric=task["eval_metric"],
                    model_path=model_path,
                    ag_config=config["autogluon"],
                )
                ag_metric = regression_metrics(
                    test_df[label].to_numpy(dtype=float),
                    ag_scores.astype(float),
                    test["is_in_spec"].to_numpy(dtype=int),
                )
            else:
                baseline = make_multiclass_baseline()
                train_y = train_df[label].to_numpy(dtype=int)
                test_y = test_df[label].to_numpy(dtype=int)
                baseline.fit(train_df[feature_columns], train_y)
                baseline_proba_raw = baseline.predict_proba(test_df[feature_columns]).astype(float)
                baseline_proba = align_probability_frame(
                    baseline_proba_raw,
                    class_labels=all_five_bin_labels,
                    observed_labels=baseline.named_steps["clf"].classes_,
                )
                baseline_metric = multiclass_metrics(test_y, baseline_proba, class_labels=all_five_bin_labels)

                model_path = artifact_dir / f"ag_stage0_{task_name}_{run_id}_fold{fold_idx}"
                ag_proba_frame, model_best, leaderboard, feature_importance = fit_autogluon_multiclass(
                    train_df=train_df,
                    test_df=test_df,
                    label=label,
                    model_path=model_path,
                    ag_config=config["autogluon"],
                )
                ag_proba = align_probability_frame(
                    ag_proba_frame,
                    class_labels=all_five_bin_labels,
                    observed_labels=ag_proba_frame.columns.tolist(),
                )
                ag_metric = multiclass_metrics(test_y, ag_proba, class_labels=all_five_bin_labels)
                if fold_idx == 1:
                    leaderboards[task_name] = leaderboard
                    feature_importances[task_name] = feature_importance

            if task["type"] != "multiclass" and fold_idx == 1:
                # Keep at least one leaderboard/feature importance artifact per stage task.
                predictor = TabularPredictor.load(str(model_path))
                leaderboards[task_name] = predictor.leaderboard(test_df, silent=True)
                feature_importances[task_name] = predictor.feature_importance(test_df, silent=True)

            for framework_name, metrics in (("simple_baseline", baseline_metric), ("autogluon", ag_metric)):
                row = {
                    "stage": "stage0",
                    "task_name": task_name,
                    "framework": framework_name,
                    "fold": int(fold_idx),
                    "samples_train": int(len(train_df)),
                    "samples_test": int(len(test_df)),
                    "lookback_minutes": int(config["snapshot"]["lookback_minutes"]),
                    "feature_count": int(len(feature_columns)),
                    "data_source_condition": config["data_source_statement"]["source_condition"],
                    **metrics,
                }
                if framework_name == "autogluon":
                    row["autogluon_model_best"] = model_best
                    ag_agg.append(metrics)
                else:
                    baseline_agg.append(metrics)
                rows.append(row)

            fold_summaries.append({"fold": int(fold_idx), "baseline": baseline_metric, "autogluon": ag_metric})

        summary_row: dict[str, Any] = {"fold_summaries": fold_summaries}
        if task["type"] == "binary":
            baseline_ap = float(np.nanmean([m["ap"] for m in baseline_agg]))
            ag_ap = float(np.nanmean([m["ap"] for m in ag_agg]))
            summary_row["baseline_mean_ap"] = baseline_ap
            summary_row["autogluon_mean_ap"] = ag_ap
            summary_row["positive_signal"] = ag_ap > baseline_ap
        elif task["type"] == "regression":
            baseline_mae = float(np.nanmean([m["mae"] for m in baseline_agg]))
            ag_mae = float(np.nanmean([m["mae"] for m in ag_agg]))
            summary_row["baseline_mean_mae"] = baseline_mae
            summary_row["autogluon_mean_mae"] = ag_mae
            summary_row["positive_signal"] = ag_mae < baseline_mae
        else:
            baseline_loss = float(np.nanmean([m["multiclass_log_loss"] for m in baseline_agg]))
            ag_loss = float(np.nanmean([m["multiclass_log_loss"] for m in ag_agg]))
            summary_row["baseline_mean_multiclass_log_loss"] = baseline_loss
            summary_row["autogluon_mean_multiclass_log_loss"] = ag_loss
            summary_row["positive_signal"] = ag_loss < baseline_loss
        stage_summary["tasks"][task_name] = summary_row

    stage_summary["stage0_positive_signal"] = any(task_info["positive_signal"] for task_info in stage_summary["tasks"].values())
    return pd.DataFrame(rows), stage_summary, snapshot_audit, feature_catalog, leaderboards, feature_importances


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon stage 0 baseline construction and evaluation.")
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

    results, stage_summary, snapshot_audit, feature_catalog, leaderboards, feature_importances = run_stage0(config_path, config)
    snapshot_table, _ = build_stage0_snapshot_table(config_path, config)

    features_path = artifact_dir / "stage0_baseline_features.csv"
    results_path = artifact_dir / "stage0_baseline_results.csv"
    feature_catalog_path = artifact_dir / "stage0_baseline_feature_catalog.csv"
    summary_path = artifact_dir / "stage0_baseline_summary.json"
    audit_path = report_dir / "stage0_baseline_audit.md"

    snapshot_table.to_csv(features_path, index=False, encoding="utf-8-sig")
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    feature_catalog.to_csv(feature_catalog_path, index=False, encoding="utf-8-sig")
    for task_name, leaderboard in leaderboards.items():
        leaderboard.to_csv(artifact_dir / f"stage0_baseline_leaderboard_{task_name}.csv", index=False, encoding="utf-8-sig")
    for task_name, feature_importance in feature_importances.items():
        feature_importance.to_csv(artifact_dir / f"stage0_baseline_feature_importance_{task_name}.csv", encoding="utf-8-sig")

    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "framework": "AutoGluon",
                "phase": "stage0_baseline",
                "stage_summary": stage_summary,
                "snapshot_audit": snapshot_audit,
                "feature_catalog_summary": {
                    "feature_count": int(len(feature_catalog)),
                    "sensor_count": int(feature_catalog["sensor"].nunique()) if not feature_catalog.empty else 0,
                },
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )

    write_stage0_audit(audit_path, config, stage_summary, snapshot_audit, feature_catalog)
    print(
        json.dumps(
            {
                "features_path": str(features_path),
                "results_path": str(results_path),
                "feature_catalog_path": str(feature_catalog_path),
                "summary_path": str(summary_path),
                "audit_path": str(audit_path),
                "stage_summary": stage_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
