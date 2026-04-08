from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
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
from run_autogluon_stage3_regime_state import build_state_package
from run_autogluon_stage4_interactions import build_interaction_frame
from run_autogluon_stage5_quality import build_quality_table


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage6_centered_quality.yaml"


def stage5_reference(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(config_path.parent, config["paths"]["stage5_summary_path"])
    if path is None or not path.exists():
        raise ValueError("stage5_summary_path must exist before running S6 centered-quality.")
    rows = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(rows)
    task_rows = frame[frame["task_name"] == "centered_desirability"]
    if task_rows.empty:
        raise ValueError("No centered_desirability row found in stage5 summary.")
    return task_rows.iloc[0].to_dict()


def apply_input_override(selected: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    override = config.get("override_input_package")
    if not override:
        return selected
    updated = dict(selected)
    for key in (
        "selected_stage1_variant",
        "selected_state_package",
        "selected_interaction_package",
        "quality_package",
        "tau_minutes",
        "window_minutes",
    ):
        if key in override:
            updated[key] = override[key]
    return updated


def _longest_within_band(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_centered_window(
    window: pd.DataFrame,
    enabled_features: list[str],
    time_column: str = "time",
) -> dict[str, float]:
    if window.empty:
        return {}

    numeric_columns = [column for column in window.columns if column != time_column]
    result: dict[str, float] = {}

    for column in numeric_columns:
        series = pd.to_numeric(window[column], errors="coerce").dropna()
        values = series.to_numpy(dtype=float)
        if len(values) == 0:
            continue
        mu = float(np.mean(values))
        sigma = float(np.std(values, ddof=0))
        denom = sigma if sigma > 1e-8 else 1.0
        z = (values - mu) / denom

        if "mean_abs_z" in enabled_features:
            result[f"centered__{column}__mean_abs_z"] = float(np.mean(np.abs(z)))
        if "center_band_ratio" in enabled_features:
            result[f"centered__{column}__center_band_ratio"] = float(np.mean(np.abs(z) <= 0.5))
        if "center_hold_ratio" in enabled_features:
            hold = _longest_within_band(np.abs(z) <= 0.5)
            result[f"centered__{column}__center_hold_ratio"] = float(hold / max(len(z), 1))
        if "last_z" in enabled_features:
            result[f"centered__{column}__last_z"] = float(z[-1])
        if "late_shift_z" in enabled_features:
            late = z[max(0, len(z) - max(2, len(z) // 4)) :]
            result[f"centered__{column}__late_shift_z"] = float(np.mean(late))
        if "tail_bias" in enabled_features:
            upper = float(np.mean(z > 0.5))
            lower = float(np.mean(z < -0.5))
            result[f"centered__{column}__tail_bias"] = upper - lower
        if "reversion_score" in enabled_features:
            if len(z) > 1:
                time_idx = np.arange(len(z), dtype=float)
                slope = float(np.polyfit(time_idx, z, deg=1)[0])
            else:
                slope = 0.0
            result[f"centered__{column}__reversion_score"] = float(-z[-1] * slope)

    return result


def build_centered_quality_table(
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
        row = {"decision_time": decision_time}
        row.update(summarize_centered_window(window, enabled_features=enabled_features, time_column="time"))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["decision_time"])
    return pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)


def build_feature_catalog(current_features: list[str], centered_features: list[str], package_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in current_features:
        if "__" in column:
            sensor, suffix = column.split("__", 1)
        else:
            sensor, suffix = "meta", column
        family = "quality" if column.startswith("quality__") else ("process_interaction" if column.startswith("interaction__") else ("regime_state" if column.startswith("state__") else "validated_current"))
        rows.append(
            {
                "task_name": "centered_desirability",
                "package_name": package_name,
                "feature_name": column,
                "sensor": sensor,
                "feature_suffix": suffix,
                "family": family,
            }
        )
    for column in centered_features:
        _, sensor, metric = column.split("__", 2)
        rows.append(
            {
                "task_name": "centered_desirability",
                "package_name": package_name,
                "feature_name": column,
                "sensor": sensor,
                "feature_suffix": metric,
                "family": "centered_quality",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon S6 centered-quality validation.")
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
    selected = apply_input_override(stage5_reference(config_path, config), config)
    tau = int(selected["tau_minutes"])
    window = int(selected["window_minutes"])
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

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
        raise ValueError("S6 snapshot is empty.")
    base_feature_columns = [column for column in snapshot.columns if "__" in column and "_dyn_" not in column]

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    feature_catalog_parts: list[pd.DataFrame] = []
    leaderboards: dict[str, pd.DataFrame] = {}
    feature_importances: dict[str, pd.DataFrame] = {}

    for package in config["centered_quality_packages"]:
        package_name = str(package["name"])
        centered_table = build_centered_quality_table(
            labeled_samples=labeled,
            dcs=dcs,
            tau_minutes=tau,
            window_minutes=window,
            enabled_features=list(package["features"]),
            min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
        )
        merged = snapshot.merge(centered_table, on="decision_time", how="inner")
        if merged.empty:
            continue

        current_feature_count = 0
        centered_feature_count = 0
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

            interaction_name = selected.get("selected_interaction_package")
            if interaction_name and interaction_name != "current":
                train_interactions = build_interaction_frame(train, tau, window, interaction_name)
                test_interactions = build_interaction_frame(test, tau, window, interaction_name)
            else:
                train_interactions = pd.DataFrame(index=train.index)
                test_interactions = pd.DataFrame(index=test.index)

            quality_name = selected.get("quality_package")
            if quality_name:
                quality_features = {
                    "missing_freeze": ["missing_ratio", "valid_count", "freeze_length", "freeze_ratio"],
                    "jump_irregularity": ["max_jump", "time_since_last_jump", "update_irregularity"],
                    "combined_quality": ["missing_ratio", "valid_count", "freeze_length", "freeze_ratio", "max_jump", "time_since_last_jump", "update_irregularity"],
                }
                quality_table = build_quality_table(
                    labeled_samples=labeled,
                    dcs=dcs,
                    tau_minutes=tau,
                    window_minutes=window,
                    enabled_features=quality_features[quality_name],
                    min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
                )
                train_quality = train.merge(quality_table, on="decision_time", how="left")[[column for column in quality_table.columns if column.startswith("quality__")]]
                test_quality = test.merge(quality_table, on="decision_time", how="left")[[column for column in quality_table.columns if column.startswith("quality__")]]
            else:
                train_quality = pd.DataFrame(index=train.index)
                test_quality = pd.DataFrame(index=test.index)

            train_current = pd.concat([train[base_feature_columns], train_state, train_interactions, train_quality], axis=1)
            test_current = pd.concat([test[base_feature_columns], test_state, test_interactions, test_quality], axis=1)
            current_feature_columns = list(train_current.columns)
            current_feature_count = len(current_feature_columns)

            centered_feature_columns = [column for column in train.columns if column.startswith("centered__")]
            train_full = pd.concat([train_current, train[centered_feature_columns]], axis=1)
            test_full = pd.concat([test_current, test[centered_feature_columns]], axis=1)
            full_feature_columns = list(train_full.columns)
            centered_feature_count = len(centered_feature_columns)
            full_feature_count = len(full_feature_columns)

            if fold_idx == 1:
                feature_catalog_parts.append(build_feature_catalog(current_feature_columns, centered_feature_columns, package_name))

            label = "target_centered_desirability"
            baseline = make_regression_baseline()
            baseline.fit(train_current[current_feature_columns], train[label].to_numpy(dtype=float))
            current_pred = baseline.predict(test_current[current_feature_columns]).astype(float)
            current_metric = regression_metrics(test[label].to_numpy(dtype=float), current_pred, test["is_in_spec"].to_numpy(dtype=int))

            baseline.fit(train_full[full_feature_columns], train[label].to_numpy(dtype=float))
            full_pred = baseline.predict(test_full[full_feature_columns]).astype(float)
            full_metric = regression_metrics(test[label].to_numpy(dtype=float), full_pred, test["is_in_spec"].to_numpy(dtype=int))

            model_path_current = artifact_dir / f"ag_stage6_centered_current_{package_name}_{run_id}_fold{fold_idx}"
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

            model_path_full = artifact_dir / f"ag_stage6_centered_{package_name}_{run_id}_fold{fold_idx}"
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

            if fold_idx == 1:
                from autogluon.tabular import TabularPredictor

                predictor_current = TabularPredictor.load(str(model_path_current))
                leaderboards[f"current::{package_name}"] = predictor_current.leaderboard(pd.concat([test_current, test[[label]]], axis=1).copy(), silent=True)
                feature_importances[f"current::{package_name}"] = predictor_current.feature_importance(pd.concat([test_current, test[[label]]], axis=1).copy(), silent=True)

                predictor_full = TabularPredictor.load(str(model_path_full))
                leaderboards[f"full::{package_name}"] = predictor_full.leaderboard(pd.concat([test_full, test[[label]]], axis=1).copy(), silent=True)
                feature_importances[f"full::{package_name}"] = predictor_full.feature_importance(pd.concat([test_full, test[[label]]], axis=1).copy(), silent=True)

            for framework_name, feature_package, metrics, feature_count, model_best in [
                ("simple_baseline", "current", current_metric, len(current_feature_columns), None),
                ("simple_baseline", package_name, full_metric, len(full_feature_columns), None),
                ("autogluon", "current", ag_current_metric, len(current_feature_columns), model_best_current),
                ("autogluon", package_name, ag_full_metric, len(full_feature_columns), model_best_full),
            ]:
                row = {
                    "stage": "stage6_centered_quality",
                    "task_name": "centered_desirability",
                    "selected_stage1_variant": selected["selected_stage1_variant"],
                    "selected_state_package": selected.get("selected_state_package"),
                    "selected_interaction_package": selected.get("selected_interaction_package"),
                    "selected_quality_package": selected.get("quality_package"),
                    "tau_minutes": tau,
                    "window_minutes": window,
                    "centered_quality_package": package_name,
                    "feature_package": feature_package,
                    "framework": framework_name,
                    "fold": int(fold_idx),
                    "samples_train": int(len(train)),
                    "samples_test": int(len(test)),
                    "current_feature_count": int(current_feature_count),
                    "centered_feature_count": int(centered_feature_count),
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
                    "simple_baseline_centered": full_metric,
                    "autogluon_current": ag_current_metric,
                    "autogluon_centered": ag_full_metric,
                }
            )

        current_baseline = float(np.nanmean([m["mae"] for m in agg["simple_baseline::current"]]))
        centered_baseline = float(np.nanmean([m["mae"] for m in agg[f"simple_baseline::{package_name}"]]))
        current_ag = float(np.nanmean([m["mae"] for m in agg["autogluon::current"]]))
        centered_ag = float(np.nanmean([m["mae"] for m in agg[f"autogluon::{package_name}"]]))
        summary_rows.append(
            {
                "task_name": "centered_desirability",
                "centered_quality_package": package_name,
                "selected_stage1_variant": selected["selected_stage1_variant"],
                "selected_state_package": selected.get("selected_state_package"),
                "selected_interaction_package": selected.get("selected_interaction_package"),
                "selected_quality_package": selected.get("quality_package"),
                "tau_minutes": tau,
                "window_minutes": window,
                "current_feature_count": int(current_feature_count),
                "centered_feature_count": int(centered_feature_count),
                "full_feature_count": int(full_feature_count),
                "baseline_current_mean_mae": current_baseline,
                "baseline_centered_mean_mae": centered_baseline,
                "autogluon_current_mean_mae": current_ag,
                "autogluon_centered_mean_mae": centered_ag,
                "centered_beats_current": centered_ag < current_ag,
                "fold_summaries": fold_summaries,
            }
        )

    results_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("autogluon_centered_mean_mae", ascending=True).reset_index(drop=True)
    feature_catalog = pd.concat(feature_catalog_parts, ignore_index=True) if feature_catalog_parts else pd.DataFrame()

    results_path = artifact_dir / "stage6_centered_quality_results.csv"
    summary_path = artifact_dir / "stage6_centered_quality_summary.json"
    features_path = artifact_dir / "stage6_centered_quality_features.csv"
    audit_path = report_dir / "stage6_centered_quality_summary.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    feature_catalog.to_csv(features_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_df.to_dict(orient="records"), stream, ensure_ascii=False, indent=2)

    for _, row in summary_df.iterrows():
        package_name = str(row["centered_quality_package"])
        for key in (f"current::{package_name}", f"full::{package_name}"):
            if key in leaderboards:
                leaderboards[key].to_csv(artifact_dir / f"stage6_centered_quality_leaderboard_{key.replace('::', '_')}.csv", index=False, encoding="utf-8-sig")
            if key in feature_importances:
                feature_importances[key].to_csv(artifact_dir / f"stage6_centered_quality_feature_importance_{key.replace('::', '_')}.csv", encoding="utf-8-sig")

    audit_lines = [
        "# Stage 6 Centered-Quality Summary",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- Stage 6 adds centered-quality process features only for `centered_desirability`.",
        "- This stage starts from the best validated centered input package from S5.",
        "",
        "## Selected Prior Input",
        "",
        json.dumps(selected, ensure_ascii=False, indent=2, default=str),
        "",
        "## Centered-Quality Packages",
        "",
        json.dumps(config["centered_quality_packages"], ensure_ascii=False, indent=2),
        "",
        "## Summary Rows",
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
