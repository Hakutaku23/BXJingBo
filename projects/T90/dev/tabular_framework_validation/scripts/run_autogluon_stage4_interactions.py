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
from run_autogluon_stage2_dynamic_morphology import build_stage2_table, select_best_stage1_variants, stage1_reference
from run_autogluon_stage3_regime_state import build_state_package


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage4_interactions.yaml"


REACTOR_BALANCE_PAIRS = [
    ("TI_C51007A_S_PV_CV", "TI_C51007B_S_PV_CV"),
    ("PI_C51006A_S_PV_CV", "PI_C51006B_S_PV_CV"),
    ("II_CM510A_PV_CV", "II_CM510B_PV_CV"),
    ("TI_C51101A_S_PV_CV", "TI_C51101B_S_PV_CV"),
    ("PI_C51101A_S_PV_CV", "PI_C51101B_S_PV_CV"),
    ("II_CM511A_PV_CV", "II_CM511B_PV_CV"),
    ("TI_C54002_PV_F_CV", "TI_C54003_PV_F_CV"),
]

FLOW_BALANCE_PAIRS = [
    ("FIC_C51001_PV_F_CV", "FIC_C51003_PV_F_CV"),
    ("FIC_C51401_PV_F_CV", "FIC_C30501_PV_F_CV"),
    ("FIC_C51401_PV_F_CV", "FI_C51005_S_PV_CV"),
    ("FIC_C51401_PV_F_CV", "FIC_C51003_PV_F_CV"),
]

THERMAL_COUPLING_PAIRS = [
    ("TI_C50604_PV_F_CV", "TICA_C53005A_PV_F_CV"),
    ("TI_C50604_PV_F_CV", "TI_C54002_PV_F_CV"),
    ("TI_C50604_PV_F_CV", "TI_C54003_PV_F_CV"),
]

INTERACTION_STATS = ["mean", "last"]


def stage3_reference(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config_path.parent, config["paths"]["stage3_summary_path"])
    if path is None or not path.exists():
        raise ValueError("stage3_summary_path must exist before running S4 interactions.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(data)


def select_effective_prior_packages(
    stage1_summary: pd.DataFrame,
    stage3_summary: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    selected_stage1 = select_best_stage1_variants(stage1_summary)
    effective: dict[str, dict[str, Any]] = {}
    for task_name, stage1_row in selected_stage1.items():
        row = {
            "selected_stage1_variant": stage1_row["variant_name"],
            "tau_minutes": int(stage1_row["tau_minutes"]),
            "window_minutes": int(stage1_row["window_minutes"]),
            "state_package": None,
        }
        stage3_task = stage3_summary[stage3_summary["task_name"] == task_name]
        if not stage3_task.empty:
            best_stage3 = stage3_task.iloc[0].to_dict()
            if bool(best_stage3.get("state_beats_stage1_best", False)):
                row["state_package"] = best_stage3["state_package"]
        effective[task_name] = row
    return effective


def interaction_feature_specs(package_name: str) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []

    if package_name in {"reactor_balance", "combined_core"}:
        for left, right in REACTOR_BALANCE_PAIRS:
            for stat in INTERACTION_STATS:
                specs.append((left, right, f"{stat}__diff"))
                specs.append((left, right, f"{stat}__absdiff"))

    if package_name in {"flow_balance", "combined_core"}:
        for left, right in FLOW_BALANCE_PAIRS:
            for stat in INTERACTION_STATS:
                specs.append((left, right, f"{stat}__diff"))
                specs.append((left, right, f"{stat}__ratio"))

    if package_name in {"thermal_coupling", "combined_core"}:
        for left, right in THERMAL_COUPLING_PAIRS:
            for stat in INTERACTION_STATS:
                specs.append((left, right, f"{stat}__diff"))
                specs.append((left, right, f"{stat}__absdiff"))

    return specs


def build_interaction_frame(
    snapshot: pd.DataFrame,
    tau_minutes: int,
    window_minutes: int,
    package_name: str,
) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    specs = interaction_feature_specs(package_name)
    for left_sensor, right_sensor, op_spec in specs:
        stat, op = op_spec.split("__", 1)
        left_col = f"{left_sensor}__lag{tau_minutes}_win{window_minutes}_{stat}"
        right_col = f"{right_sensor}__lag{tau_minutes}_win{window_minutes}_{stat}"
        if left_col not in snapshot.columns or right_col not in snapshot.columns:
            continue

        left = pd.to_numeric(snapshot[left_col], errors="coerce")
        right = pd.to_numeric(snapshot[right_col], errors="coerce")
        if op == "diff":
            values = left - right
            suffix = "diff"
        elif op == "absdiff":
            values = (left - right).abs()
            suffix = "absdiff"
        elif op == "ratio":
            values = left / right.replace(0.0, np.nan)
            suffix = "ratio"
        else:
            continue
        rows[f"interaction__{left_sensor}__{right_sensor}__{stat}_{suffix}"] = values

    if not rows:
        return pd.DataFrame(index=snapshot.index)
    return pd.DataFrame(rows, index=snapshot.index)


def build_feature_catalog(current_features: list[str], interaction_features: list[str], task_name: str, package_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in current_features:
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
    for column in interaction_features:
        rows.append(
            {
                "task_name": task_name,
                "package_name": package_name,
                "feature_name": column,
                "sensor": "interaction",
                "feature_suffix": column.replace("interaction__", "", 1),
                "family": "process_interaction",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon S4 process interaction validation.")
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
    stage3_summary = stage3_reference(config_path, config)
    selected_inputs = select_effective_prior_packages(stage1_summary, stage3_summary)
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
        for package in config["interaction_packages"]:
            package_name = str(package["name"])
            fold_summaries: list[dict[str, Any]] = []
            agg: dict[str, list[dict[str, float]]] = {
                "simple_baseline::current": [],
                f"simple_baseline::{package_name}": [],
                "autogluon::current": [],
                f"autogluon::{package_name}": [],
            }
            current_feature_count = 0
            interaction_feature_count = 0
            full_feature_count = 0

            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
                train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
                test = snapshot.iloc[test_idx].copy().reset_index(drop=True)

                state_package_name = selected.get("state_package")
                if state_package_name:
                    if state_package_name == "time_context":
                        state_package = {"name": "time_context", "include_time_context": True, "cluster_count": 0}
                    elif state_package_name == "cluster_k5":
                        state_package = {"name": "cluster_k5", "include_time_context": False, "cluster_count": 5}
                    else:
                        state_package = {"name": "time_plus_cluster_k5", "include_time_context": True, "cluster_count": 5}
                    train_state, test_state, state_feature_columns = build_state_package(
                        train=train,
                        test=test,
                        base_feature_columns=base_feature_columns,
                        package=state_package,
                    )
                else:
                    train_state = pd.DataFrame(index=train.index)
                    test_state = pd.DataFrame(index=test.index)
                    state_feature_columns = []

                train_current = pd.concat([train[base_feature_columns], train_state], axis=1)
                test_current = pd.concat([test[base_feature_columns], test_state], axis=1)
                current_feature_columns = list(train_current.columns)
                current_feature_count = len(current_feature_columns)

                train_interactions = build_interaction_frame(train, tau, window, package_name)
                test_interactions = build_interaction_frame(test, tau, window, package_name)
                interaction_feature_columns = list(train_interactions.columns)
                interaction_feature_count = len(interaction_feature_columns)

                train_full = pd.concat([train_current, train_interactions], axis=1)
                test_full = pd.concat([test_current, test_interactions], axis=1)
                full_feature_columns = list(train_full.columns)
                full_feature_count = len(full_feature_columns)

                if fold_idx == 1:
                    feature_catalog_parts.append(
                        build_feature_catalog(current_feature_columns, interaction_feature_columns, task_name, package_name)
                    )

                if task_name == "centered_desirability":
                    label = "target_centered_desirability"
                    baseline = make_regression_baseline()

                    baseline.fit(train_current[current_feature_columns], train[label].to_numpy(dtype=float))
                    current_pred = baseline.predict(test_current[current_feature_columns]).astype(float)
                    current_metric = regression_metrics(
                        test[label].to_numpy(dtype=float),
                        current_pred,
                        test["is_in_spec"].to_numpy(dtype=int),
                    )

                    baseline.fit(train_full[full_feature_columns], train[label].to_numpy(dtype=float))
                    full_pred = baseline.predict(test_full[full_feature_columns]).astype(float)
                    full_metric = regression_metrics(
                        test[label].to_numpy(dtype=float),
                        full_pred,
                        test["is_in_spec"].to_numpy(dtype=int),
                    )

                    model_path_current = artifact_dir / f"ag_stage4_interaction_{task_name}_current_{package_name}_{run_id}_fold{fold_idx}"
                    ag_current_pred, model_best_current = fit_autogluon_fold(
                        train_df=pd.concat([train_current, train[[label]]], axis=1).copy(),
                        test_df=pd.concat([test_current, test[[label]]], axis=1).copy(),
                        label=label,
                        problem_type="regression",
                        eval_metric="root_mean_squared_error",
                        model_path=model_path_current,
                        ag_config=config["autogluon"],
                    )
                    ag_current_metric = regression_metrics(
                        test[label].to_numpy(dtype=float),
                        ag_current_pred.astype(float),
                        test["is_in_spec"].to_numpy(dtype=int),
                    )

                    model_path_full = artifact_dir / f"ag_stage4_interaction_{task_name}_{package_name}_{run_id}_fold{fold_idx}"
                    ag_full_pred, model_best_full = fit_autogluon_fold(
                        train_df=pd.concat([train_full, train[[label]]], axis=1).copy(),
                        test_df=pd.concat([test_full, test[[label]]], axis=1).copy(),
                        label=label,
                        problem_type="regression",
                        eval_metric="root_mean_squared_error",
                        model_path=model_path_full,
                        ag_config=config["autogluon"],
                    )
                    ag_full_metric = regression_metrics(
                        test[label].to_numpy(dtype=float),
                        ag_full_pred.astype(float),
                        test["is_in_spec"].to_numpy(dtype=int),
                    )
                else:
                    label = "target_five_bin"
                    train_y = train[label].to_numpy(dtype=int)
                    test_y = test[label].to_numpy(dtype=int)
                    baseline = make_multiclass_baseline()

                    baseline.fit(train_current[current_feature_columns], train_y)
                    current_proba = baseline.predict_proba(test_current[current_feature_columns]).astype(float)
                    current_frame = pd.DataFrame(current_proba, columns=baseline.named_steps["clf"].classes_.tolist())
                    current_frame = current_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    current_frame = current_frame.div(current_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    current_metric = multiclass_metrics(
                        test_y, current_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels
                    )

                    baseline.fit(train_full[full_feature_columns], train_y)
                    full_proba = baseline.predict_proba(test_full[full_feature_columns]).astype(float)
                    full_frame = pd.DataFrame(full_proba, columns=baseline.named_steps["clf"].classes_.tolist())
                    full_frame = full_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    full_frame = full_frame.div(full_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    full_metric = multiclass_metrics(
                        test_y, full_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels
                    )

                    model_path_current = artifact_dir / f"ag_stage4_interaction_{task_name}_current_{package_name}_{run_id}_fold{fold_idx}"
                    ag_current_frame, model_best_current, leaderboard_current, feature_importance_current = fit_autogluon_multiclass(
                        train_df=pd.concat([train_current, train[[label]]], axis=1).copy(),
                        test_df=pd.concat([test_current, test[[label]]], axis=1).copy(),
                        label=label,
                        model_path=model_path_current,
                        ag_config=config["autogluon"],
                    )
                    ag_current_frame = ag_current_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    ag_current_frame = ag_current_frame.div(ag_current_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    ag_current_metric = multiclass_metrics(
                        test_y, ag_current_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels
                    )

                    model_path_full = artifact_dir / f"ag_stage4_interaction_{task_name}_{package_name}_{run_id}_fold{fold_idx}"
                    ag_full_frame, model_best_full, leaderboard_full, feature_importance_full = fit_autogluon_multiclass(
                        train_df=pd.concat([train_full, train[[label]]], axis=1).copy(),
                        test_df=pd.concat([test_full, test[[label]]], axis=1).copy(),
                        label=label,
                        model_path=model_path_full,
                        ag_config=config["autogluon"],
                    )
                    ag_full_frame = ag_full_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    ag_full_frame = ag_full_frame.div(ag_full_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    ag_full_metric = multiclass_metrics(
                        test_y, ag_full_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels
                    )
                    if fold_idx == 1:
                        leaderboards[f"{task_name}::current::{package_name}"] = leaderboard_current
                        feature_importances[f"{task_name}::current::{package_name}"] = feature_importance_current
                        leaderboards[f"{task_name}::{package_name}"] = leaderboard_full
                        feature_importances[f"{task_name}::{package_name}"] = feature_importance_full

                if task_name == "centered_desirability" and fold_idx == 1:
                    from autogluon.tabular import TabularPredictor

                    predictor_current = TabularPredictor.load(str(model_path_current))
                    leaderboards[f"{task_name}::current::{package_name}"] = predictor_current.leaderboard(
                        pd.concat([test_current, test[[label]]], axis=1).copy(), silent=True
                    )
                    feature_importances[f"{task_name}::current::{package_name}"] = predictor_current.feature_importance(
                        pd.concat([test_current, test[[label]]], axis=1).copy(), silent=True
                    )

                    predictor_full = TabularPredictor.load(str(model_path_full))
                    leaderboards[f"{task_name}::{package_name}"] = predictor_full.leaderboard(
                        pd.concat([test_full, test[[label]]], axis=1).copy(), silent=True
                    )
                    feature_importances[f"{task_name}::{package_name}"] = predictor_full.feature_importance(
                        pd.concat([test_full, test[[label]]], axis=1).copy(), silent=True
                    )

                fold_pairs = [
                    ("simple_baseline", "current", current_metric, len(current_feature_columns), None),
                    ("simple_baseline", package_name, full_metric, len(full_feature_columns), None),
                    ("autogluon", "current", ag_current_metric, len(current_feature_columns), model_best_current),
                    ("autogluon", package_name, ag_full_metric, len(full_feature_columns), model_best_full),
                ]
                for framework_name, feature_package, metrics, feature_count, model_best in fold_pairs:
                    row = {
                        "stage": "stage4_interactions",
                        "task_name": task_name,
                        "selected_stage1_variant": selected["selected_stage1_variant"],
                        "selected_state_package": selected.get("state_package"),
                        "tau_minutes": int(selected["tau_minutes"]),
                        "window_minutes": int(selected["window_minutes"]),
                        "interaction_package": package_name,
                        "feature_package": feature_package,
                        "framework": framework_name,
                        "fold": int(fold_idx),
                        "samples_train": int(len(train)),
                        "samples_test": int(len(test)),
                        "current_feature_count": int(current_feature_count),
                        "interaction_feature_count": int(interaction_feature_count),
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
                        "simple_baseline_current": current_metric,
                        "simple_baseline_interaction": full_metric,
                        "autogluon_current": ag_current_metric,
                        "autogluon_interaction": ag_full_metric,
                    }
                )

            if task_name == "centered_desirability":
                current_baseline = float(np.nanmean([m["mae"] for m in agg["simple_baseline::current"]]))
                interaction_baseline = float(np.nanmean([m["mae"] for m in agg[f"simple_baseline::{package_name}"]]))
                current_ag = float(np.nanmean([m["mae"] for m in agg["autogluon::current"]]))
                interaction_ag = float(np.nanmean([m["mae"] for m in agg[f"autogluon::{package_name}"]]))
                package_summaries.append(
                    {
                        "task_name": task_name,
                        "interaction_package": package_name,
                        "selected_stage1_variant": selected["selected_stage1_variant"],
                        "selected_state_package": selected.get("state_package"),
                        "tau_minutes": int(selected["tau_minutes"]),
                        "window_minutes": int(selected["window_minutes"]),
                        "current_feature_count": int(current_feature_count),
                        "interaction_feature_count": int(interaction_feature_count),
                        "full_feature_count": int(full_feature_count),
                        "baseline_current_mean_mae": current_baseline,
                        "baseline_interaction_mean_mae": interaction_baseline,
                        "autogluon_current_mean_mae": current_ag,
                        "autogluon_interaction_mean_mae": interaction_ag,
                        "interaction_beats_current": interaction_ag < current_ag,
                        "fold_summaries": fold_summaries,
                    }
                )
            else:
                current_baseline = float(
                    np.nanmean([m["multiclass_log_loss"] for m in agg["simple_baseline::current"]])
                )
                interaction_baseline = float(
                    np.nanmean([m["multiclass_log_loss"] for m in agg[f"simple_baseline::{package_name}"]])
                )
                current_ag = float(np.nanmean([m["multiclass_log_loss"] for m in agg["autogluon::current"]]))
                interaction_ag = float(
                    np.nanmean([m["multiclass_log_loss"] for m in agg[f"autogluon::{package_name}"]])
                )
                package_summaries.append(
                    {
                        "task_name": task_name,
                        "interaction_package": package_name,
                        "selected_stage1_variant": selected["selected_stage1_variant"],
                        "selected_state_package": selected.get("state_package"),
                        "tau_minutes": int(selected["tau_minutes"]),
                        "window_minutes": int(selected["window_minutes"]),
                        "current_feature_count": int(current_feature_count),
                        "interaction_feature_count": int(interaction_feature_count),
                        "full_feature_count": int(full_feature_count),
                        "baseline_current_mean_multiclass_log_loss": current_baseline,
                        "baseline_interaction_mean_multiclass_log_loss": interaction_baseline,
                        "autogluon_current_mean_multiclass_log_loss": current_ag,
                        "autogluon_interaction_mean_multiclass_log_loss": interaction_ag,
                        "interaction_beats_current": interaction_ag < current_ag,
                        "fold_summaries": fold_summaries,
                    }
                )

        if package_summaries:
            if task_name == "centered_desirability":
                best = sorted(package_summaries, key=lambda row: row["autogluon_interaction_mean_mae"])[0]
            else:
                best = sorted(package_summaries, key=lambda row: row["autogluon_interaction_mean_multiclass_log_loss"])[0]
            summary_rows.append(best)

    results_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    feature_catalog = pd.concat(feature_catalog_parts, ignore_index=True) if feature_catalog_parts else pd.DataFrame()

    results_path = artifact_dir / "stage4_interaction_results.csv"
    summary_path = artifact_dir / "stage4_interaction_summary.json"
    features_path = artifact_dir / "stage4_interaction_features.csv"
    audit_path = report_dir / "stage4_interaction_summary.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    feature_catalog.to_csv(features_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_df.to_dict(orient="records"), stream, ensure_ascii=False, indent=2)

    for _, row in summary_df.iterrows():
        task_name = str(row["task_name"])
        package_name = str(row["interaction_package"])
        for key in (f"{task_name}::current::{package_name}", f"{task_name}::{package_name}"):
            if key in leaderboards:
                leaderboards[key].to_csv(
                    artifact_dir / f"stage4_interaction_leaderboard_{key.replace('::', '_')}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            if key in feature_importances:
                feature_importances[key].to_csv(
                    artifact_dir / f"stage4_interaction_feature_importance_{key.replace('::', '_')}.csv",
                    encoding="utf-8-sig",
                )

    audit_lines = [
        "# Stage 4 Interaction Summary",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- Stage 4 adds a limited set of process-interaction features on top of the currently validated input package for each task.",
        "- This stage remains restricted to `centered_desirability` and `five_bin`.",
        "",
        "## Selected Prior Inputs",
        "",
        json.dumps(selected_inputs, ensure_ascii=False, indent=2, default=str),
        "",
        "## Interaction Packages",
        "",
        json.dumps(config["interaction_packages"], ensure_ascii=False, indent=2),
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
