from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_stage0_baseline import (
    fit_autogluon_multiclass,
    make_multiclass_baseline,
    multiclass_metrics,
)
from run_autogluon_stage1_lag_scale import build_label_frame
from run_autogluon_stage1_quickcheck import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    regression_metrics,
    resolve_path,
)
from run_autogluon_stage2_dynamic_morphology import build_stage2_table
from run_autogluon_stage3_regime_state import build_state_package
from run_autogluon_stage4_interactions import build_interaction_frame


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage5_quality.yaml"


def stage4_reference(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config_path.parent, config["paths"]["stage4_summary_path"])
    if path is None or not path.exists():
        raise ValueError("stage4_summary_path must exist before running S5 quality.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(data)


def select_effective_prior_packages(stage4_summary: pd.DataFrame) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for task_name in ("centered_desirability", "five_bin"):
        task_rows = stage4_summary[stage4_summary["task_name"] == task_name]
        if not task_rows.empty:
            selected[task_name] = task_rows.iloc[0].to_dict()
    return selected


def _longest_constant_run(values: np.ndarray, eps: float = 1e-12) -> int:
    if len(values) == 0:
        return 0
    best = 1
    current = 1
    for idx in range(1, len(values)):
        if abs(values[idx] - values[idx - 1]) <= eps:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def summarize_quality_window(
    window: pd.DataFrame,
    enabled_features: list[str],
    time_column: str = "time",
) -> dict[str, float]:
    if window.empty:
        return {}

    numeric_columns = [column for column in window.columns if column != time_column]
    result: dict[str, float] = {}
    window_end = pd.to_datetime(window[time_column], errors="coerce").max() if time_column in window.columns else None

    for column in numeric_columns:
        series = pd.to_numeric(window[column], errors="coerce")
        valid = series.dropna()
        valid_count = int(valid.shape[0])
        total_count = int(series.shape[0])
        valid_values = valid.to_numpy(dtype=float)
        diffs = np.diff(valid_values) if len(valid_values) > 1 else np.array([], dtype=float)
        nonzero_mask = np.abs(diffs) > 1e-12

        if "missing_ratio" in enabled_features:
            result[f"quality__{column}__missing_ratio"] = float(1.0 - (valid_count / total_count if total_count else 0.0))
        if "valid_count" in enabled_features:
            result[f"quality__{column}__valid_count"] = float(valid_count)
        if "freeze_length" in enabled_features:
            result[f"quality__{column}__freeze_length"] = float(_longest_constant_run(valid_values))
        if "freeze_ratio" in enabled_features:
            result[f"quality__{column}__freeze_ratio"] = float(np.mean(~nonzero_mask)) if len(diffs) > 0 else 0.0
        if "max_jump" in enabled_features:
            result[f"quality__{column}__max_jump"] = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0
        if "time_since_last_jump" in enabled_features:
            if len(diffs) == 0 or not np.any(nonzero_mask) or window_end is None:
                minutes = float(total_count)
            else:
                valid_times = pd.to_datetime(window.loc[series.notna(), time_column], errors="coerce").reset_index(drop=True)
                jump_idx = int(np.where(nonzero_mask)[0][-1] + 1)
                minutes = float((window_end - valid_times.iloc[jump_idx]).total_seconds() / 60.0)
            result[f"quality__{column}__time_since_last_jump"] = minutes
        if "update_irregularity" in enabled_features:
            if time_column in window.columns and valid_count > 1:
                valid_times = pd.to_datetime(window.loc[series.notna(), time_column], errors="coerce")
                delta_minutes = valid_times.diff().dt.total_seconds().dropna().to_numpy(dtype=float) / 60.0
                irregularity = float(np.std(delta_minutes, ddof=0)) if len(delta_minutes) > 0 else 0.0
            else:
                irregularity = 0.0
            result[f"quality__{column}__update_irregularity"] = irregularity
    return result


def build_quality_table(
    labeled_samples: pd.DataFrame,
    dcs: pd.DataFrame,
    tau_minutes: int,
    window_minutes: int,
    enabled_features: list[str],
    min_points_per_window: int,
) -> pd.DataFrame:
    sample_frame = labeled_samples.copy()
    sample_frame["sample_time"] = pd.to_datetime(sample_frame["sample_time"], errors="coerce")
    sample_frame = sample_frame.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)

    dcs_frame = dcs.copy()
    dcs_frame["time"] = pd.to_datetime(dcs_frame["time"], errors="coerce")
    dcs_frame = dcs_frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")

    rows: list[dict[str, Any]] = []
    for record in sample_frame.itertuples(index=False):
        decision_time = pd.Timestamp(record.sample_time)
        window_end = decision_time - pd.Timedelta(minutes=tau_minutes)
        window_start = window_end - pd.Timedelta(minutes=window_minutes)
        left = np.searchsorted(dcs_times, window_start.to_datetime64(), side="left")
        right = np.searchsorted(dcs_times, window_end.to_datetime64(), side="right")
        window = dcs_frame.iloc[left:right]
        if len(window) < min_points_per_window:
            continue
        row = {"decision_time": decision_time, "quality_rows_in_window": int(len(window))}
        row.update(summarize_quality_window(window, enabled_features=enabled_features, time_column="time"))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["decision_time", "quality_rows_in_window"])
    return pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)


def build_feature_catalog(
    current_features: list[str],
    quality_features: list[str],
    task_name: str,
    package_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in current_features:
        if "__" in column:
            sensor, suffix = column.split("__", 1)
        else:
            sensor, suffix = "meta", column
        family = "regime_state" if column.startswith("state__") else ("process_interaction" if column.startswith("interaction__") else "validated_current")
        rows.append(
            {
                "task_name": task_name,
                "package_name": package_name,
                "feature_name": column,
                "sensor": sensor,
                "feature_suffix": suffix,
                "family": family,
            }
        )
    for column in quality_features:
        _, sensor, metric = column.split("__", 2)
        rows.append(
            {
                "task_name": task_name,
                "package_name": package_name,
                "feature_name": column,
                "sensor": sensor,
                "feature_suffix": metric,
                "family": "quality",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon S5 data-quality validation.")
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

    labeled = build_label_frame(config_path, config)
    dcs = load_dcs_frame(
        resolve_path(config_path.parent, config["paths"]["dcs_main_path"]),
        resolve_path(config_path.parent, config["paths"].get("dcs_supplemental_path")),
    )
    stage4_summary = stage4_reference(config_path, config)
    selected_inputs = select_effective_prior_packages(stage4_summary)
    all_five_bin_labels = sorted(labeled["target_five_bin"].dropna().astype(int).unique().tolist())
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    feature_catalog_parts: list[pd.DataFrame] = []
    leaderboards: dict[str, pd.DataFrame] = {}
    feature_importances: dict[str, pd.DataFrame] = {}

    for task_name in ("centered_desirability", "five_bin"):
        if task_name not in selected_inputs:
            continue
        selected = selected_inputs[task_name]
        tau = int(selected["tau_minutes"])
        window = int(selected["window_minutes"])

        snapshot = build_stage2_table(
            labeled_samples=labeled,
            dcs=dcs,
            tau_minutes=tau,
            window_minutes=window,
            stats=list(config["snapshot"]["stats"]),
            enabled_dynamic_features=[],
            min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
        )
        if snapshot.empty:
            continue
        base_feature_columns = [column for column in snapshot.columns if "__" in column and "_dyn_" not in column]

        package_summaries: list[dict[str, Any]] = []
        for package in config["quality_packages"]:
            package_name = str(package["name"])
            quality_table = build_quality_table(
                labeled_samples=labeled,
                dcs=dcs,
                tau_minutes=tau,
                window_minutes=window,
                enabled_features=list(package["features"]),
                min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
            )
            merged = snapshot.merge(quality_table, on="decision_time", how="inner")
            if merged.empty:
                continue

            current_feature_count = 0
            quality_feature_count = 0
            full_feature_count = 0
            fold_summaries: list[dict[str, Any]] = []
            agg: dict[str, list[dict[str, float]]] = {
                "simple_baseline::current": [],
                f"simple_baseline::{package_name}": [],
                "autogluon::current": [],
                f"autogluon::{package_name}": [],
            }

            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(merged), start=1):
                train = merged.iloc[train_idx].copy().reset_index(drop=True)
                test = merged.iloc[test_idx].copy().reset_index(drop=True)

                state_package_name = selected.get("selected_state_package")
                if state_package_name:
                    if state_package_name == "time_context":
                        state_package = {"name": "time_context", "include_time_context": True, "cluster_count": 0}
                    elif state_package_name == "cluster_k5":
                        state_package = {"name": "cluster_k5", "include_time_context": False, "cluster_count": 5}
                    else:
                        state_package = {"name": "time_plus_cluster_k5", "include_time_context": True, "cluster_count": 5}
                    train_state, test_state, _ = build_state_package(train, test, base_feature_columns, state_package)
                else:
                    train_state = pd.DataFrame(index=train.index)
                    test_state = pd.DataFrame(index=test.index)

                interaction_name = selected.get("interaction_package")
                if interaction_name and interaction_name != "current":
                    train_interactions = build_interaction_frame(train, tau, window, interaction_name)
                    test_interactions = build_interaction_frame(test, tau, window, interaction_name)
                else:
                    train_interactions = pd.DataFrame(index=train.index)
                    test_interactions = pd.DataFrame(index=test.index)

                train_current = pd.concat([train[base_feature_columns], train_state, train_interactions], axis=1)
                test_current = pd.concat([test[base_feature_columns], test_state, test_interactions], axis=1)
                current_feature_columns = list(train_current.columns)
                current_feature_count = len(current_feature_columns)

                quality_feature_columns = [column for column in train.columns if column.startswith("quality__")]
                train_full = pd.concat([train_current, train[quality_feature_columns]], axis=1)
                test_full = pd.concat([test_current, test[quality_feature_columns]], axis=1)
                full_feature_columns = list(train_full.columns)
                quality_feature_count = len(quality_feature_columns)
                full_feature_count = len(full_feature_columns)

                if fold_idx == 1:
                    feature_catalog_parts.append(build_feature_catalog(current_feature_columns, quality_feature_columns, task_name, package_name))

                if task_name == "centered_desirability":
                    label = "target_centered_desirability"
                    baseline = make_regression_baseline()
                    baseline.fit(train_current[current_feature_columns], train[label].to_numpy(dtype=float))
                    current_pred = baseline.predict(test_current[current_feature_columns]).astype(float)
                    current_metric = regression_metrics(test[label].to_numpy(dtype=float), current_pred, test["is_in_spec"].to_numpy(dtype=int))

                    baseline.fit(train_full[full_feature_columns], train[label].to_numpy(dtype=float))
                    full_pred = baseline.predict(test_full[full_feature_columns]).astype(float)
                    full_metric = regression_metrics(test[label].to_numpy(dtype=float), full_pred, test["is_in_spec"].to_numpy(dtype=int))

                    model_path_current = artifact_dir / f"ag_stage5_quality_{task_name}_current_{package_name}_{run_id}_fold{fold_idx}"
                    ag_current_pred, model_best_current = fit_autogluon_fold(
                        train_df=pd.concat([train_current, train[[label]]], axis=1).copy(),
                        test_df=pd.concat([test_current, test[[label]]], axis=1).copy(),
                        label=label,
                        problem_type="regression",
                        eval_metric="root_mean_squared_error",
                        model_path=model_path_current,
                        ag_config=config["autogluon"],
                    )
                    ag_current_metric = regression_metrics(test[label].to_numpy(dtype=float), ag_current_pred.astype(float), test["is_in_spec"].to_numpy(dtype=int))

                    model_path_full = artifact_dir / f"ag_stage5_quality_{task_name}_{package_name}_{run_id}_fold{fold_idx}"
                    ag_full_pred, model_best_full = fit_autogluon_fold(
                        train_df=pd.concat([train_full, train[[label]]], axis=1).copy(),
                        test_df=pd.concat([test_full, test[[label]]], axis=1).copy(),
                        label=label,
                        problem_type="regression",
                        eval_metric="root_mean_squared_error",
                        model_path=model_path_full,
                        ag_config=config["autogluon"],
                    )
                    ag_full_metric = regression_metrics(test[label].to_numpy(dtype=float), ag_full_pred.astype(float), test["is_in_spec"].to_numpy(dtype=int))
                else:
                    label = "target_five_bin"
                    train_y = train[label].to_numpy(dtype=int)
                    test_y = test[label].to_numpy(dtype=int)
                    baseline = make_multiclass_baseline()
                    baseline.fit(train_current[current_feature_columns], train_y)
                    current_proba = baseline.predict_proba(test_current[current_feature_columns]).astype(float)
                    current_frame = pd.DataFrame(current_proba, columns=baseline.named_steps["clf"].classes_.tolist()).reindex(columns=all_five_bin_labels, fill_value=0.0)
                    current_frame = current_frame.div(current_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    current_metric = multiclass_metrics(test_y, current_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels)

                    baseline.fit(train_full[full_feature_columns], train_y)
                    full_proba = baseline.predict_proba(test_full[full_feature_columns]).astype(float)
                    full_frame = pd.DataFrame(full_proba, columns=baseline.named_steps["clf"].classes_.tolist()).reindex(columns=all_five_bin_labels, fill_value=0.0)
                    full_frame = full_frame.div(full_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    full_metric = multiclass_metrics(test_y, full_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels)

                    model_path_current = artifact_dir / f"ag_stage5_quality_{task_name}_current_{package_name}_{run_id}_fold{fold_idx}"
                    ag_current_frame, model_best_current, leaderboard_current, feature_importance_current = fit_autogluon_multiclass(
                        train_df=pd.concat([train_current, train[[label]]], axis=1).copy(),
                        test_df=pd.concat([test_current, test[[label]]], axis=1).copy(),
                        label=label,
                        model_path=model_path_current,
                        ag_config=config["autogluon"],
                    )
                    ag_current_frame = ag_current_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    ag_current_frame = ag_current_frame.div(ag_current_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    ag_current_metric = multiclass_metrics(test_y, ag_current_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels)

                    model_path_full = artifact_dir / f"ag_stage5_quality_{task_name}_{package_name}_{run_id}_fold{fold_idx}"
                    ag_full_frame, model_best_full, leaderboard_full, feature_importance_full = fit_autogluon_multiclass(
                        train_df=pd.concat([train_full, train[[label]]], axis=1).copy(),
                        test_df=pd.concat([test_full, test[[label]]], axis=1).copy(),
                        label=label,
                        model_path=model_path_full,
                        ag_config=config["autogluon"],
                    )
                    ag_full_frame = ag_full_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    ag_full_frame = ag_full_frame.div(ag_full_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    ag_full_metric = multiclass_metrics(test_y, ag_full_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels)
                    if fold_idx == 1:
                        leaderboards[f"{task_name}::current::{package_name}"] = leaderboard_current
                        feature_importances[f"{task_name}::current::{package_name}"] = feature_importance_current
                        leaderboards[f"{task_name}::{package_name}"] = leaderboard_full
                        feature_importances[f"{task_name}::{package_name}"] = feature_importance_full

                if task_name == "centered_desirability" and fold_idx == 1:
                    from autogluon.tabular import TabularPredictor
                    predictor_current = TabularPredictor.load(str(model_path_current))
                    leaderboards[f"{task_name}::current::{package_name}"] = predictor_current.leaderboard(pd.concat([test_current, test[[label]]], axis=1).copy(), silent=True)
                    feature_importances[f"{task_name}::current::{package_name}"] = predictor_current.feature_importance(pd.concat([test_current, test[[label]]], axis=1).copy(), silent=True)
                    predictor_full = TabularPredictor.load(str(model_path_full))
                    leaderboards[f"{task_name}::{package_name}"] = predictor_full.leaderboard(pd.concat([test_full, test[[label]]], axis=1).copy(), silent=True)
                    feature_importances[f"{task_name}::{package_name}"] = predictor_full.feature_importance(pd.concat([test_full, test[[label]]], axis=1).copy(), silent=True)

                for framework_name, feature_package, metrics, feature_count, model_best in [
                    ("simple_baseline", "current", current_metric, len(current_feature_columns), None),
                    ("simple_baseline", package_name, full_metric, len(full_feature_columns), None),
                    ("autogluon", "current", ag_current_metric, len(current_feature_columns), model_best_current),
                    ("autogluon", package_name, ag_full_metric, len(full_feature_columns), model_best_full),
                ]:
                    row = {
                        "stage": "stage5_quality",
                        "task_name": task_name,
                        "selected_stage1_variant": selected["selected_stage1_variant"],
                        "selected_state_package": selected.get("selected_state_package"),
                        "selected_interaction_package": selected.get("interaction_package"),
                        "tau_minutes": tau,
                        "window_minutes": window,
                        "quality_package": package_name,
                        "feature_package": feature_package,
                        "framework": framework_name,
                        "fold": int(fold_idx),
                        "samples_train": int(len(train)),
                        "samples_test": int(len(test)),
                        "current_feature_count": int(current_feature_count),
                        "quality_feature_count": int(quality_feature_count),
                        "feature_count": int(feature_count),
                        "data_source_condition": config["data_source_statement"]["source_condition"],
                        **metrics,
                    }
                    if model_best is not None:
                        row["autogluon_model_best"] = model_best
                    rows.append(row)
                    agg[f"{framework_name}::{feature_package}"].append(metrics)

                fold_summaries.append({"fold": int(fold_idx), "simple_baseline_current": current_metric, "simple_baseline_quality": full_metric, "autogluon_current": ag_current_metric, "autogluon_quality": ag_full_metric})

            if task_name == "centered_desirability":
                current_baseline = float(np.nanmean([m["mae"] for m in agg["simple_baseline::current"]]))
                quality_baseline = float(np.nanmean([m["mae"] for m in agg[f"simple_baseline::{package_name}"]]))
                current_ag = float(np.nanmean([m["mae"] for m in agg["autogluon::current"]]))
                quality_ag = float(np.nanmean([m["mae"] for m in agg[f"autogluon::{package_name}"]]))
                package_summaries.append({"task_name": task_name, "quality_package": package_name, "selected_stage1_variant": selected["selected_stage1_variant"], "selected_state_package": selected.get("selected_state_package"), "selected_interaction_package": selected.get("interaction_package"), "tau_minutes": tau, "window_minutes": window, "current_feature_count": int(current_feature_count), "quality_feature_count": int(quality_feature_count), "full_feature_count": int(full_feature_count), "baseline_current_mean_mae": current_baseline, "baseline_quality_mean_mae": quality_baseline, "autogluon_current_mean_mae": current_ag, "autogluon_quality_mean_mae": quality_ag, "quality_beats_current": quality_ag < current_ag, "fold_summaries": fold_summaries})
            else:
                current_baseline = float(np.nanmean([m["multiclass_log_loss"] for m in agg["simple_baseline::current"]]))
                quality_baseline = float(np.nanmean([m["multiclass_log_loss"] for m in agg[f"simple_baseline::{package_name}"]]))
                current_ag = float(np.nanmean([m["multiclass_log_loss"] for m in agg["autogluon::current"]]))
                quality_ag = float(np.nanmean([m["multiclass_log_loss"] for m in agg[f"autogluon::{package_name}"]]))
                package_summaries.append({"task_name": task_name, "quality_package": package_name, "selected_stage1_variant": selected["selected_stage1_variant"], "selected_state_package": selected.get("selected_state_package"), "selected_interaction_package": selected.get("interaction_package"), "tau_minutes": tau, "window_minutes": window, "current_feature_count": int(current_feature_count), "quality_feature_count": int(quality_feature_count), "full_feature_count": int(full_feature_count), "baseline_current_mean_multiclass_log_loss": current_baseline, "baseline_quality_mean_multiclass_log_loss": quality_baseline, "autogluon_current_mean_multiclass_log_loss": current_ag, "autogluon_quality_mean_multiclass_log_loss": quality_ag, "quality_beats_current": quality_ag < current_ag, "fold_summaries": fold_summaries})

        if package_summaries:
            best = sorted(package_summaries, key=(lambda row: row["autogluon_quality_mean_mae"]) if task_name == "centered_desirability" else (lambda row: row["autogluon_quality_mean_multiclass_log_loss"]))[0]
            summary_rows.append(best)

    results_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    feature_catalog = pd.concat(feature_catalog_parts, ignore_index=True) if feature_catalog_parts else pd.DataFrame()

    results_path = artifact_dir / "stage5_quality_results.csv"
    summary_path = artifact_dir / "stage5_quality_summary.json"
    features_path = artifact_dir / "stage5_quality_features.csv"
    audit_path = report_dir / "stage5_quality_summary.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    feature_catalog.to_csv(features_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_df.to_dict(orient="records"), stream, ensure_ascii=False, indent=2)

    for _, row in summary_df.iterrows():
        task_name = str(row["task_name"])
        package_name = str(row["quality_package"])
        for key in (f"{task_name}::current::{package_name}", f"{task_name}::{package_name}"):
            if key in leaderboards:
                leaderboards[key].to_csv(artifact_dir / f"stage5_quality_leaderboard_{key.replace('::', '_')}.csv", index=False, encoding="utf-8-sig")
            if key in feature_importances:
                feature_importances[key].to_csv(artifact_dir / f"stage5_quality_feature_importance_{key.replace('::', '_')}.csv", encoding="utf-8-sig")

    audit_lines = [
        "# Stage 5 Quality Summary",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- Stage 5 adds controlled data-quality / sensor-health features on top of the current best validated task-specific input package.",
        "- This stage remains restricted to `centered_desirability` and `five_bin`.",
        "",
        "## Selected Prior Inputs",
        "",
        json.dumps(selected_inputs, ensure_ascii=False, indent=2, default=str),
        "",
        "## Quality Packages",
        "",
        json.dumps(config["quality_packages"], ensure_ascii=False, indent=2),
        "",
        "## Best Summary Rows",
        "",
        json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
    ]
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")

    print(json.dumps({"results_path": str(results_path), "summary_path": str(summary_path), "features_path": str(features_path), "audit_path": str(audit_path), "summary": summary_df.to_dict(orient="records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
