from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from autogluon.tabular import TabularPredictor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import TargetSpec, add_out_of_spec_labels, build_dcs_feature_table, load_dcs_frame, load_lims_samples


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage2_feature_engineering.yaml"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_path(base: Path, relative: str | None) -> Path | None:
    if not relative:
        return None
    return (base / relative).resolve()


def make_binary_baseline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1200, solver="lbfgs")),
        ]
    )


def make_regression_baseline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]
    )


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    metrics: dict[str, float] = {
        "ap": float(average_precision_score(y_true, scores)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "unacceptable_miss_rate": float(1.0 - recall_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def regression_metrics(y_true: np.ndarray, pred: np.ndarray, in_spec_flag: np.ndarray) -> dict[str, float]:
    rank_corr = pd.Series(y_true).corr(pd.Series(pred), method="spearman")
    auc_like = float("nan")
    if len(np.unique(in_spec_flag)) > 1:
        auc_like = float(roc_auc_score(in_spec_flag, pred))
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "rank_correlation": float(rank_corr) if pd.notna(rank_corr) else float("nan"),
        "in_spec_auc_from_desirability": auc_like,
    }


def fit_autogluon_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label: str,
    problem_type: str,
    eval_metric: str,
    model_path: Path,
    ag_config: dict[str, Any],
) -> tuple[np.ndarray, str]:
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
    if problem_type == "binary":
        proba = predictor.predict_proba(test_df.drop(columns=[label]))
        positive_col = 1 if 1 in proba.columns else proba.columns[-1]
        return proba[positive_col].to_numpy(dtype=float), str(predictor.model_best)
    pred = predictor.predict(test_df.drop(columns=[label]))
    return pred.to_numpy(dtype=float), str(predictor.model_best)


def select_features_fold(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    task_type: str,
    top_k: int,
) -> tuple[list[str], dict[str, float]]:
    imputer = SimpleImputer(strategy="median")
    x_filled = imputer.fit_transform(train_x)
    if task_type == "binary":
        scores = mutual_info_classif(x_filled, train_y.to_numpy(dtype=int), discrete_features=False, random_state=42)
    else:
        scores = mutual_info_regression(x_filled, train_y.to_numpy(dtype=float), random_state=42)
    ranking = pd.Series(scores, index=train_x.columns).sort_values(ascending=False)
    selected = ranking.head(min(top_k, len(ranking))).index.tolist()
    score_map = {str(k): float(v) for k, v in ranking.head(min(top_k, len(ranking))).items()}
    return selected, score_map


def build_stage2_snapshot_table(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = config["paths"]
    spec = TargetSpec(
        center=float(config["target_spec"]["center"]),
        tolerance=float(config["target_spec"]["tolerance"]),
    )
    lims, _ = load_lims_samples(resolve_path(config_path.parent, paths["lims_path"]))
    labeled = add_out_of_spec_labels(lims, spec).dropna(subset=["t90"]).copy()
    dcs = load_dcs_frame(
        resolve_path(config_path.parent, paths["dcs_main_path"]),
        resolve_path(config_path.parent, paths.get("dcs_supplemental_path")),
    )
    raw = build_dcs_feature_table(
        labeled_samples=labeled,
        dcs=dcs,
        window_minutes=int(config["snapshot"]["lookback_minutes"]),
        min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
    )
    if raw.empty:
        raise ValueError("Stage 2 snapshot table is empty.")

    allowed_stats = set(config["snapshot"]["stats"])
    feature_cols: list[str] = []
    for column in raw.columns:
        if "__" not in column:
            continue
        _, stat = column.split("__", 1)
        if stat in allowed_stats:
            feature_cols.append(column)
    keep_cols = ["sample_time", "t90", "is_in_spec", "is_out_of_spec", "is_above_spec", "is_below_spec", *feature_cols]
    table = raw[keep_cols].copy()

    max_cols = [column for column in feature_cols if column.endswith("__max")]
    for max_col in max_cols:
        sensor = max_col[:-5]
        min_col = f"{sensor}__min"
        range_col = f"{sensor}__range"
        if min_col in table.columns and range_col not in table.columns:
            table[range_col] = table[max_col] - table[min_col]

    feature_columns = [column for column in table.columns if "__" in column]
    dropped_all_nan = [column for column in feature_columns if table[column].isna().all()]
    table = table.drop(columns=dropped_all_nan)
    feature_columns = [column for column in table.columns if "__" in column]

    dropped_constant = []
    for column in feature_columns:
        valid = table[column].dropna()
        if not valid.empty and valid.nunique() <= 1:
            dropped_constant.append(column)
    table = table.drop(columns=dropped_constant)
    feature_columns = [column for column in table.columns if "__" in column]

    missing_threshold = float(config["cleaning"]["drop_feature_missing_ratio_above"])
    dropped_high_missing = [column for column in feature_columns if float(table[column].isna().mean()) > missing_threshold]
    table = table.drop(columns=dropped_high_missing)
    feature_columns = [column for column in table.columns if "__" in column]

    corr_threshold = float(config["cleaning"]["drop_high_corr_threshold"])
    dropped_high_corr: list[str] = []
    corr_frame = table[feature_columns].copy()
    corr_frame = corr_frame.fillna(corr_frame.median(numeric_only=True))
    if not corr_frame.empty:
        corr = corr_frame.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for column in upper.columns:
            if any(upper[column] > corr_threshold):
                dropped_high_corr.append(column)
    if dropped_high_corr:
        table = table.drop(columns=dropped_high_corr)
    feature_columns = [column for column in table.columns if "__" in column]

    tasks = config["tasks"]
    if tasks["high_risk"]["enabled"]:
        table["target_high_risk"] = (table["t90"] > float(tasks["high_risk"]["threshold"])).astype(int)
    if tasks["centered_desirability"]["enabled"]:
        center = float(config["target_spec"]["center"])
        tol = float(config["target_spec"]["tolerance"])
        table["target_centered_desirability"] = np.maximum(0.0, 1.0 - (table["t90"] - center).abs() / tol)

    audit = {
        "starting_source_condition": config["data_source_statement"]["source_condition"],
        "stage2_new_processing": [
            "richer_snapshot_stats_including_slope_and_valid_ratio",
            "drop_high_missing_features",
            "drop_near_duplicate_high_correlation_features",
            "fold_internal_supervised_feature_selection",
        ],
        "lookback_minutes": int(config["snapshot"]["lookback_minutes"]),
        "raw_samples_after_alignment": int(len(raw)),
        "snapshot_samples": int(len(table)),
        "feature_count_after_global_cleaning": int(len(feature_columns)),
        "dropped_all_nan_features": int(len(dropped_all_nan)),
        "dropped_constant_features": int(len(dropped_constant)),
        "dropped_high_missing_features": int(len(dropped_high_missing)),
        "dropped_high_corr_features": int(len(dropped_high_corr)),
    }
    return table.sort_values("sample_time").reset_index(drop=True), audit


def run_stage2(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    snapshot, snapshot_audit = build_stage2_snapshot_table(config_path, config)
    all_feature_columns = [column for column in snapshot.columns if "__" in column]
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    stage_summary: dict[str, Any] = {"tasks": {}, "stage2_positive_signal": False}
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    selection_details: dict[str, list[dict[str, Any]]] = {}

    task_defs: list[dict[str, Any]] = []
    if config["tasks"]["high_risk"]["enabled"]:
        task_defs.append(
            {
                "task_name": "high_risk",
                "label": "target_high_risk",
                "type": "binary",
                "eval_metric": "average_precision",
                "top_k": int(config["selection"]["high_risk_top_k"]),
            }
        )
    if config["tasks"]["centered_desirability"]["enabled"]:
        task_defs.append(
            {
                "task_name": "centered_desirability",
                "label": "target_centered_desirability",
                "type": "regression",
                "eval_metric": "root_mean_squared_error",
                "top_k": int(config["selection"]["centered_desirability_top_k"]),
            }
        )

    for task in task_defs:
        task_name = task["task_name"]
        label = task["label"]
        fold_summaries: list[dict[str, Any]] = []
        agg_for_framework: list[dict[str, Any]] = []
        agg_for_baseline: list[dict[str, Any]] = []
        selection_details[task_name] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
            train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
            test = snapshot.iloc[test_idx].copy().reset_index(drop=True)
            selected_features, top_scores = select_features_fold(
                train_x=train[all_feature_columns],
                train_y=train[label],
                task_type=task["type"],
                top_k=int(task["top_k"]),
            )
            selection_details[task_name].append(
                {
                    "fold": int(fold_idx),
                    "selected_feature_count": int(len(selected_features)),
                    "top_feature_scores": top_scores,
                }
            )

            train_df = train[selected_features + [label]].copy()
            test_df = test[selected_features + [label]].copy()

            if task["type"] == "binary":
                baseline = make_binary_baseline()
                baseline.fit(train_df[selected_features], train_df[label].to_numpy(dtype=int))
                baseline_scores = baseline.predict_proba(test_df[selected_features])[:, 1]
                baseline_metric = binary_metrics(test_df[label].to_numpy(dtype=int), baseline_scores)

                model_path = artifact_dir / f"ag_stage2_{task_name}_{run_id}_fold{fold_idx}"
                framework_scores, model_best = fit_autogluon_fold(
                    train_df=train_df,
                    test_df=test_df,
                    label=label,
                    problem_type="binary",
                    eval_metric=task["eval_metric"],
                    model_path=model_path,
                    ag_config=config["autogluon"],
                )
                framework_metric = binary_metrics(test_df[label].to_numpy(dtype=int), framework_scores)
            else:
                baseline = make_regression_baseline()
                baseline.fit(train_df[selected_features], train_df[label].to_numpy(dtype=float))
                baseline_scores = baseline.predict(test_df[selected_features])
                baseline_metric = regression_metrics(
                    test_df[label].to_numpy(dtype=float),
                    baseline_scores.astype(float),
                    test["is_in_spec"].to_numpy(dtype=int),
                )

                model_path = artifact_dir / f"ag_stage2_{task_name}_{run_id}_fold{fold_idx}"
                framework_scores, model_best = fit_autogluon_fold(
                    train_df=train_df,
                    test_df=test_df,
                    label=label,
                    problem_type="regression",
                    eval_metric=task["eval_metric"],
                    model_path=model_path,
                    ag_config=config["autogluon"],
                )
                framework_metric = regression_metrics(
                    test_df[label].to_numpy(dtype=float),
                    framework_scores.astype(float),
                    test["is_in_spec"].to_numpy(dtype=int),
                )

            for framework_name, metrics in [("simple_baseline_stage2", baseline_metric), ("autogluon_stage2", framework_metric)]:
                row = {
                    "stage": "stage2",
                    "task_name": task_name,
                    "framework": framework_name,
                    "fold": int(fold_idx),
                    "samples_train": int(len(train_df)),
                    "samples_test": int(len(test_df)),
                    "lookback_minutes": int(config["snapshot"]["lookback_minutes"]),
                    "feature_count_selected": int(len(selected_features)),
                    "data_source_condition": config["data_source_statement"]["source_condition"],
                    **metrics,
                }
                if framework_name == "autogluon_stage2":
                    row["autogluon_model_best"] = model_best
                    agg_for_framework.append(metrics)
                else:
                    agg_for_baseline.append(metrics)
                rows.append(row)

            fold_summaries.append(
                {
                    "fold": int(fold_idx),
                    "selected_feature_count": int(len(selected_features)),
                    "baseline": baseline_metric,
                    "autogluon": framework_metric,
                }
            )

        summary_row: dict[str, Any] = {
            "fold_summaries": fold_summaries,
            "selection_summary": selection_details[task_name],
        }
        if task["type"] == "binary":
            baseline_ap = float(np.nanmean([m["ap"] for m in agg_for_baseline]))
            framework_ap = float(np.nanmean([m["ap"] for m in agg_for_framework]))
            summary_row["baseline_mean_ap"] = baseline_ap
            summary_row["autogluon_mean_ap"] = framework_ap
            summary_row["positive_signal"] = framework_ap > baseline_ap
        else:
            baseline_mae = float(np.nanmean([m["mae"] for m in agg_for_baseline]))
            framework_mae = float(np.nanmean([m["mae"] for m in agg_for_framework]))
            summary_row["baseline_mean_mae"] = baseline_mae
            summary_row["autogluon_mean_mae"] = framework_mae
            summary_row["positive_signal"] = framework_mae < baseline_mae
        stage_summary["tasks"][task_name] = summary_row

    stage_summary["stage2_positive_signal"] = any(task_info["positive_signal"] for task_info in stage_summary["tasks"].values())
    return pd.DataFrame(rows), stage_summary, snapshot_audit


def write_audit_markdown(path: Path, config: dict[str, Any], stage_summary: dict[str, Any], snapshot_audit: dict[str, Any]) -> None:
    lines = [
        "# Tabular Framework Validation Audit - Stage 2",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- Uncleaned refers to source state, not a ban on preprocessing.",
        "",
        "## Stage 1 Processing Baseline",
        "",
        "- basic time alignment",
        "- causal 120-minute snapshot window construction",
        "- numeric format conversion",
        "- framework-required missing-value handling inside model / framework",
        "- drop all-NaN features",
        "- drop constant features",
        "",
        "## Stage 2 Additional Processing",
        "",
        "- richer snapshot stats: slope, valid_ratio, range",
        "- global high-missing feature dropping",
        "- global near-duplicate high-correlation feature dropping",
        "- fold-internal supervised feature selection",
        "",
        "## Time Split",
        "",
        f"- TimeSeriesSplit(n_splits={config['validation']['n_splits']})",
        "",
        "## Task Types",
        "",
        "- AutoGluon binary evaluation for high-risk head",
        "- AutoGluon regression evaluation for centered desirability",
        "",
        "## Stage Conclusions",
        "",
        json.dumps(stage_summary, ensure_ascii=False, indent=2),
        "",
        "## Snapshot Audit",
        "",
        json.dumps(snapshot_audit, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon stage 2 feature engineering validation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    report_dir = resolve_path(config_path.parent, config["paths"]["report_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results, stage_summary, snapshot_audit = run_stage2(config_path, config)
    results_path = artifact_dir / "tabular_framework_validation_stage2_results.csv"
    summary_path = artifact_dir / "tabular_framework_validation_stage2_summary.json"
    audit_path = report_dir / "tabular_framework_validation_stage2_audit.md"
    snapshot_path = artifact_dir / "stage2_snapshot_feature_table.csv"

    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "framework": "AutoGluon",
                "phase": "stage2_feature_engineering",
                "stage_summary": stage_summary,
                "snapshot_audit": snapshot_audit,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    snapshot_table, _ = build_stage2_snapshot_table(config_path, config)
    snapshot_table.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
    write_audit_markdown(audit_path, config, stage_summary, snapshot_audit)
    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "audit_path": str(audit_path),
                "stage_summary": stage_summary,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
