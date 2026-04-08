from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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
from run_autogluon_stage2_dynamic_morphology import (
    build_stage2_table,
    select_best_stage1_variants,
    stage1_reference,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage3_regime_state.yaml"


def build_time_context_features(decision_times: pd.Series) -> pd.DataFrame:
    times = pd.to_datetime(decision_times, errors="coerce")
    hour = times.dt.hour.fillna(0).astype(int)
    minute = times.dt.minute.fillna(0).astype(int)
    dayofweek = times.dt.dayofweek.fillna(0).astype(int)

    minute_of_day = (hour * 60 + minute).to_numpy(dtype=float)
    hour_angle = 2.0 * np.pi * minute_of_day / 1440.0
    dow_angle = 2.0 * np.pi * dayofweek.to_numpy(dtype=float) / 7.0

    shift_morning = ((hour >= 8) & (hour < 16)).astype(int)
    shift_evening = ((hour >= 16) & (hour < 24)).astype(int)
    shift_night = ((hour >= 0) & (hour < 8)).astype(int)

    return pd.DataFrame(
        {
            "state__time_hour_sin": np.sin(hour_angle),
            "state__time_hour_cos": np.cos(hour_angle),
            "state__time_dow_sin": np.sin(dow_angle),
            "state__time_dow_cos": np.cos(dow_angle),
            "state__shift_morning": shift_morning,
            "state__shift_evening": shift_evening,
            "state__shift_night": shift_night,
        },
        index=decision_times.index,
    )


def build_cluster_features(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    cluster_count: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_imputed = imputer.fit_transform(train_base)
    test_imputed = imputer.transform(test_base)
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    clusterer = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=20)
    train_assign = clusterer.fit_predict(train_scaled)
    test_assign = clusterer.predict(test_scaled)
    train_dist = clusterer.transform(train_scaled)
    test_dist = clusterer.transform(test_scaled)

    def make_frame(assignments: np.ndarray, distances: np.ndarray, index: pd.Index) -> pd.DataFrame:
        data: dict[str, Any] = {}
        for idx in range(cluster_count):
            data[f"state__cluster_{cluster_count}_dist_{idx}"] = distances[:, idx]
            data[f"state__cluster_{cluster_count}_is_{idx}"] = (assignments == idx).astype(int)

        min_dist = np.min(distances, axis=1)
        sorted_dist = np.sort(distances, axis=1)
        second_dist = sorted_dist[:, 1] if cluster_count > 1 else min_dist
        inv_dist = 1.0 / (distances + 1e-6)
        inv_prob = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        data[f"state__cluster_{cluster_count}_min_dist"] = min_dist
        data[f"state__cluster_{cluster_count}_margin"] = second_dist - min_dist
        for idx in range(cluster_count):
            data[f"state__cluster_{cluster_count}_prob_{idx}"] = inv_prob[:, idx]
        return pd.DataFrame(data, index=index)

    return make_frame(train_assign, train_dist, train_base.index), make_frame(test_assign, test_dist, test_base.index)


def build_state_package(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_feature_columns: list[str],
    package: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    if bool(package.get("include_time_context", False)):
        train_parts.append(build_time_context_features(train["decision_time"]))
        test_parts.append(build_time_context_features(test["decision_time"]))

    cluster_count = int(package.get("cluster_count", 0) or 0)
    if cluster_count > 0:
        train_cluster, test_cluster = build_cluster_features(
            train_base=train[base_feature_columns],
            test_base=test[base_feature_columns],
            cluster_count=cluster_count,
        )
        train_parts.append(train_cluster)
        test_parts.append(test_cluster)

    if train_parts:
        train_state = pd.concat(train_parts, axis=1)
        test_state = pd.concat(test_parts, axis=1)
        feature_columns = list(train_state.columns)
    else:
        train_state = pd.DataFrame(index=train.index)
        test_state = pd.DataFrame(index=test.index)
        feature_columns = []

    return train_state, test_state, feature_columns


def build_feature_catalog(feature_columns: list[str], task_name: str, package_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        if "__" in column:
            sensor, suffix = column.split("__", 1)
        else:
            sensor, suffix = "state", column
        family = "regime_state" if column.startswith("state__") else "lag_scale_base"
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
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon S3 regime/state validation.")
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
    stage1_summary = stage1_reference(config_path, config)
    selected_variants = select_best_stage1_variants(stage1_summary)
    all_five_bin_labels = sorted(labeled["target_five_bin"].dropna().astype(int).unique().tolist())
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    feature_catalog_parts: list[pd.DataFrame] = []
    leaderboards: dict[str, pd.DataFrame] = {}
    feature_importances: dict[str, pd.DataFrame] = {}

    for task_name in ("centered_desirability", "five_bin"):
        if task_name not in selected_variants:
            continue
        ref_row = selected_variants[task_name]
        tau = int(ref_row["tau_minutes"])
        window = int(ref_row["window_minutes"])

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
        for package in config["regime_packages"]:
            package_name = str(package["name"])
            fold_summaries: list[dict[str, Any]] = []
            agg: dict[str, list[dict[str, float]]] = {
                "simple_baseline::base_stats": [],
                f"simple_baseline::{package_name}": [],
                "autogluon::base_stats": [],
                f"autogluon::{package_name}": [],
            }
            package_feature_count = 0
            combined_feature_count = 0

            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
                train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
                test = snapshot.iloc[test_idx].copy().reset_index(drop=True)

                train_state, test_state, state_feature_columns = build_state_package(
                    train=train,
                    test=test,
                    base_feature_columns=base_feature_columns,
                    package=package,
                )
                package_feature_count = len(state_feature_columns)
                train_pkg = pd.concat([train[base_feature_columns], train_state], axis=1)
                test_pkg = pd.concat([test[base_feature_columns], test_state], axis=1)
                package_feature_columns = list(train_pkg.columns)
                combined_feature_count = len(package_feature_columns)
                if fold_idx == 1:
                    feature_catalog_parts.append(
                        build_feature_catalog(base_feature_columns + state_feature_columns, task_name, package_name)
                    )

                if task_name == "centered_desirability":
                    label = "target_centered_desirability"
                    baseline = make_regression_baseline()

                    baseline.fit(train[base_feature_columns], train[label].to_numpy(dtype=float))
                    base_pred = baseline.predict(test[base_feature_columns]).astype(float)
                    base_metric = regression_metrics(
                        test[label].to_numpy(dtype=float),
                        base_pred,
                        test["is_in_spec"].to_numpy(dtype=int),
                    )

                    baseline.fit(train_pkg[package_feature_columns], train[label].to_numpy(dtype=float))
                    pkg_pred = baseline.predict(test_pkg[package_feature_columns]).astype(float)
                    pkg_metric = regression_metrics(
                        test[label].to_numpy(dtype=float),
                        pkg_pred,
                        test["is_in_spec"].to_numpy(dtype=int),
                    )

                    model_path_base = artifact_dir / f"ag_stage3_regime_{task_name}_base_{run_id}_fold{fold_idx}"
                    ag_base_pred, model_best_base = fit_autogluon_fold(
                        train_df=train[base_feature_columns + [label]].copy(),
                        test_df=test[base_feature_columns + [label]].copy(),
                        label=label,
                        problem_type="regression",
                        eval_metric="root_mean_squared_error",
                        model_path=model_path_base,
                        ag_config=config["autogluon"],
                    )
                    ag_base_metric = regression_metrics(
                        test[label].to_numpy(dtype=float),
                        ag_base_pred.astype(float),
                        test["is_in_spec"].to_numpy(dtype=int),
                    )

                    model_path_pkg = artifact_dir / f"ag_stage3_regime_{task_name}_{package_name}_{run_id}_fold{fold_idx}"
                    ag_pkg_pred, model_best_pkg = fit_autogluon_fold(
                        train_df=pd.concat([train_pkg, train[[label]]], axis=1).copy(),
                        test_df=pd.concat([test_pkg, test[[label]]], axis=1).copy(),
                        label=label,
                        problem_type="regression",
                        eval_metric="root_mean_squared_error",
                        model_path=model_path_pkg,
                        ag_config=config["autogluon"],
                    )
                    ag_pkg_metric = regression_metrics(
                        test[label].to_numpy(dtype=float),
                        ag_pkg_pred.astype(float),
                        test["is_in_spec"].to_numpy(dtype=int),
                    )
                else:
                    label = "target_five_bin"
                    train_y = train[label].to_numpy(dtype=int)
                    test_y = test[label].to_numpy(dtype=int)
                    baseline = make_multiclass_baseline()

                    baseline.fit(train[base_feature_columns], train_y)
                    base_proba = baseline.predict_proba(test[base_feature_columns]).astype(float)
                    base_frame = pd.DataFrame(base_proba, columns=baseline.named_steps["clf"].classes_.tolist())
                    base_frame = base_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    base_frame = base_frame.div(base_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    base_metric = multiclass_metrics(test_y, base_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels)

                    baseline.fit(train_pkg[package_feature_columns], train_y)
                    pkg_proba = baseline.predict_proba(test_pkg[package_feature_columns]).astype(float)
                    pkg_frame = pd.DataFrame(pkg_proba, columns=baseline.named_steps["clf"].classes_.tolist())
                    pkg_frame = pkg_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    pkg_frame = pkg_frame.div(pkg_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    pkg_metric = multiclass_metrics(test_y, pkg_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels)

                    model_path_base = artifact_dir / f"ag_stage3_regime_{task_name}_base_{run_id}_fold{fold_idx}"
                    ag_base_frame, model_best_base, leaderboard_base, feature_importance_base = fit_autogluon_multiclass(
                        train_df=train[base_feature_columns + [label]].copy(),
                        test_df=test[base_feature_columns + [label]].copy(),
                        label=label,
                        model_path=model_path_base,
                        ag_config=config["autogluon"],
                    )
                    ag_base_frame = ag_base_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    ag_base_frame = ag_base_frame.div(ag_base_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    ag_base_metric = multiclass_metrics(
                        test_y,
                        ag_base_frame.to_numpy(dtype=float),
                        class_labels=all_five_bin_labels,
                    )

                    model_path_pkg = artifact_dir / f"ag_stage3_regime_{task_name}_{package_name}_{run_id}_fold{fold_idx}"
                    ag_pkg_frame, model_best_pkg, leaderboard_pkg, feature_importance_pkg = fit_autogluon_multiclass(
                        train_df=pd.concat([train_pkg, train[[label]]], axis=1).copy(),
                        test_df=pd.concat([test_pkg, test[[label]]], axis=1).copy(),
                        label=label,
                        model_path=model_path_pkg,
                        ag_config=config["autogluon"],
                    )
                    ag_pkg_frame = ag_pkg_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    ag_pkg_frame = ag_pkg_frame.div(ag_pkg_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    ag_pkg_metric = multiclass_metrics(
                        test_y,
                        ag_pkg_frame.to_numpy(dtype=float),
                        class_labels=all_five_bin_labels,
                    )
                    if fold_idx == 1:
                        leaderboards[f"{task_name}::base_stats"] = leaderboard_base
                        feature_importances[f"{task_name}::base_stats"] = feature_importance_base
                        leaderboards[f"{task_name}::{package_name}"] = leaderboard_pkg
                        feature_importances[f"{task_name}::{package_name}"] = feature_importance_pkg

                if task_name == "centered_desirability" and fold_idx == 1:
                    from autogluon.tabular import TabularPredictor

                    predictor_base = TabularPredictor.load(str(model_path_base))
                    leaderboards[f"{task_name}::base_stats"] = predictor_base.leaderboard(
                        test[base_feature_columns + [label]].copy(), silent=True
                    )
                    feature_importances[f"{task_name}::base_stats"] = predictor_base.feature_importance(
                        test[base_feature_columns + [label]].copy(), silent=True
                    )

                    predictor_pkg = TabularPredictor.load(str(model_path_pkg))
                    leaderboards[f"{task_name}::{package_name}"] = predictor_pkg.leaderboard(
                        pd.concat([test_pkg, test[[label]]], axis=1).copy(), silent=True
                    )
                    feature_importances[f"{task_name}::{package_name}"] = predictor_pkg.feature_importance(
                        pd.concat([test_pkg, test[[label]]], axis=1).copy(), silent=True
                    )

                fold_pairs = [
                    ("simple_baseline", "base_stats", base_metric, len(base_feature_columns), None),
                    ("simple_baseline", package_name, pkg_metric, len(package_feature_columns), None),
                    ("autogluon", "base_stats", ag_base_metric, len(base_feature_columns), model_best_base),
                    ("autogluon", package_name, ag_pkg_metric, len(package_feature_columns), model_best_pkg),
                ]
                for framework_name, feature_package, metrics, feature_count, model_best in fold_pairs:
                    row = {
                        "stage": "stage3_regime_state",
                        "task_name": task_name,
                        "selected_stage1_variant": ref_row["variant_name"],
                        "tau_minutes": int(ref_row["tau_minutes"]),
                        "window_minutes": int(ref_row["window_minutes"]),
                        "state_package": package_name,
                        "feature_package": feature_package,
                        "framework": framework_name,
                        "fold": int(fold_idx),
                        "samples_train": int(len(train)),
                        "samples_test": int(len(test)),
                        "base_feature_count": int(len(base_feature_columns)),
                        "state_feature_count": int(package_feature_count),
                        "feature_count": int(feature_count),
                        "data_source_condition": config["data_source_statement"]["source_condition"],
                        **metrics,
                    }
                    if model_best is not None:
                        row["autogluon_model_best"] = model_best
                    rows.append(row)
                    agg[f"{framework_name}::{feature_package}"].append(metrics)

                fold_summaries.append(
                    {
                        "fold": int(fold_idx),
                        "simple_baseline_base": base_metric,
                        "simple_baseline_state": pkg_metric,
                        "autogluon_base": ag_base_metric,
                        "autogluon_state": ag_pkg_metric,
                    }
                )

            if task_name == "centered_desirability":
                base_baseline = float(np.nanmean([m["mae"] for m in agg["simple_baseline::base_stats"]]))
                state_baseline = float(np.nanmean([m["mae"] for m in agg[f"simple_baseline::{package_name}"]]))
                base_ag = float(np.nanmean([m["mae"] for m in agg["autogluon::base_stats"]]))
                state_ag = float(np.nanmean([m["mae"] for m in agg[f"autogluon::{package_name}"]]))
                stage1_ref = float(ref_row["autogluon_mean_mae"])
                package_summaries.append(
                    {
                        "task_name": task_name,
                        "state_package": package_name,
                        "selected_stage1_variant": ref_row["variant_name"],
                        "tau_minutes": int(ref_row["tau_minutes"]),
                        "window_minutes": int(ref_row["window_minutes"]),
                        "base_feature_count": int(len(base_feature_columns)),
                        "state_feature_count": int(package_feature_count),
                        "full_feature_count": int(combined_feature_count),
                        "baseline_base_mean_mae": base_baseline,
                        "baseline_state_mean_mae": state_baseline,
                        "autogluon_base_mean_mae": base_ag,
                        "autogluon_state_mean_mae": state_ag,
                        "stage1_autogluon_ref": stage1_ref,
                        "state_beats_same_window_base": state_ag < base_ag,
                        "state_beats_stage1_best": state_ag < stage1_ref,
                        "fold_summaries": fold_summaries,
                    }
                )
            else:
                base_baseline = float(
                    np.nanmean([m["multiclass_log_loss"] for m in agg["simple_baseline::base_stats"]])
                )
                state_baseline = float(
                    np.nanmean([m["multiclass_log_loss"] for m in agg[f"simple_baseline::{package_name}"]])
                )
                base_ag = float(np.nanmean([m["multiclass_log_loss"] for m in agg["autogluon::base_stats"]]))
                state_ag = float(np.nanmean([m["multiclass_log_loss"] for m in agg[f"autogluon::{package_name}"]]))
                stage1_ref = float(ref_row["autogluon_mean_multiclass_log_loss"])
                package_summaries.append(
                    {
                        "task_name": task_name,
                        "state_package": package_name,
                        "selected_stage1_variant": ref_row["variant_name"],
                        "tau_minutes": int(ref_row["tau_minutes"]),
                        "window_minutes": int(ref_row["window_minutes"]),
                        "base_feature_count": int(len(base_feature_columns)),
                        "state_feature_count": int(package_feature_count),
                        "full_feature_count": int(combined_feature_count),
                        "baseline_base_mean_multiclass_log_loss": base_baseline,
                        "baseline_state_mean_multiclass_log_loss": state_baseline,
                        "autogluon_base_mean_multiclass_log_loss": base_ag,
                        "autogluon_state_mean_multiclass_log_loss": state_ag,
                        "stage1_autogluon_ref": stage1_ref,
                        "state_beats_same_window_base": state_ag < base_ag,
                        "state_beats_stage1_best": state_ag < stage1_ref,
                        "fold_summaries": fold_summaries,
                    }
                )

        if package_summaries:
            if task_name == "centered_desirability":
                best = sorted(package_summaries, key=lambda row: row["autogluon_state_mean_mae"])[0]
            else:
                best = sorted(package_summaries, key=lambda row: row["autogluon_state_mean_multiclass_log_loss"])[0]
            summary_rows.append(best)

    results_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    feature_catalog = pd.concat(feature_catalog_parts, ignore_index=True) if feature_catalog_parts else pd.DataFrame()

    results_path = artifact_dir / "stage3_regime_results.csv"
    summary_path = artifact_dir / "stage3_regime_summary.json"
    features_path = artifact_dir / "stage3_regime_features.csv"
    audit_path = report_dir / "stage3_regime_summary.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    feature_catalog.to_csv(features_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_df.to_dict(orient="records"), stream, ensure_ascii=False, indent=2)

    for task_name in ("centered_desirability", "five_bin"):
        if task_name not in summary_df["task_name"].tolist():
            continue
        best_package = summary_df[summary_df["task_name"] == task_name].iloc[0]["state_package"]
        for feature_package in ("base_stats", best_package):
            key = f"{task_name}::{feature_package}"
            if key in leaderboards:
                leaderboards[key].to_csv(
                    artifact_dir / f"stage3_regime_leaderboard_{task_name}_{feature_package}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            if key in feature_importances:
                feature_importances[key].to_csv(
                    artifact_dir / f"stage3_regime_feature_importance_{task_name}_{feature_package}.csv",
                    encoding="utf-8-sig",
                )

    audit_lines = [
        "# Stage 3 Regime / State Summary",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- Stage 3 adds controlled regime/state features on top of the best S1 lag-scale packages.",
        "- This stage remains restricted to `centered_desirability` and `five_bin`.",
        "",
        "## Selected S1 Base Packages",
        "",
        json.dumps(selected_variants, ensure_ascii=False, indent=2, default=str),
        "",
        "## Regime Packages",
        "",
        json.dumps(config["regime_packages"], ensure_ascii=False, indent=2),
        "",
        "## Best Summary Rows",
        "",
        json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
    ]
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "features_path": str(features_path),
                "audit_path": str(audit_path),
                "summary": summary_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
