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


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame, summarize_numeric_window


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage2_dynamic_morphology.yaml"


def stage1_reference(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config_path.parent, config["paths"]["stage1_summary_path"])
    if path is None or not path.exists():
        raise ValueError("stage1_summary_path must exist before running S2 dynamic morphology.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(data)


def select_best_stage1_variants(stage1_summary: pd.DataFrame) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    centered = stage1_summary[stage1_summary["task_name"] == "centered_desirability"].sort_values(
        "autogluon_mean_mae", ascending=True
    )
    if not centered.empty:
        selected["centered_desirability"] = centered.iloc[0].to_dict()
    five_bin = stage1_summary[stage1_summary["task_name"] == "five_bin"].sort_values(
        "autogluon_mean_multiclass_log_loss", ascending=True
    )
    if not five_bin.empty:
        selected["five_bin"] = five_bin.iloc[0].to_dict()
    return selected


def summarize_dynamic_window(
    window: pd.DataFrame,
    enabled_features: list[str],
    time_column: str = "time",
) -> dict[str, float]:
    if window.empty:
        return {}

    numeric_columns = [column for column in window.columns if column != time_column]
    result: dict[str, float] = {}

    if time_column in window.columns:
        time_values = pd.to_datetime(window[time_column], errors="coerce")
        time_index = ((time_values - time_values.iloc[0]).dt.total_seconds() / 60.0).to_numpy()
    else:
        time_index = np.arange(len(window), dtype=float)

    for column in numeric_columns:
        series = pd.to_numeric(window[column], errors="coerce")
        valid_mask = series.notna().to_numpy()
        valid_y = series[valid_mask].to_numpy(dtype=float)
        if len(valid_y) == 0:
            continue
        valid_x = time_index[valid_mask]
        diffs = np.diff(valid_y) if len(valid_y) > 1 else np.array([], dtype=float)

        nonzero_diffs = diffs[diffs != 0.0]
        nonzero_signs = np.sign(nonzero_diffs)
        level_std = float(np.std(valid_y, ddof=0)) if len(valid_y) > 1 else 0.0
        diff_std = float(np.std(diffs, ddof=0)) if len(diffs) > 1 else 0.0

        if "slope" in enabled_features:
            if len(valid_y) > 1 and np.nanstd(valid_x) > 0:
                result[f"{column}__dyn_slope"] = float(np.polyfit(valid_x, valid_y, deg=1)[0])
            else:
                result[f"{column}__dyn_slope"] = 0.0
        if "diff_mean" in enabled_features:
            result[f"{column}__dyn_diff_mean"] = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        if "diff_std" in enabled_features:
            result[f"{column}__dyn_diff_std"] = diff_std
        if "diff_max_abs" in enabled_features:
            result[f"{column}__dyn_diff_max_abs"] = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0
        if "sign_change_count" in enabled_features:
            sign_changes = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1])) if len(nonzero_signs) > 1 else 0
            result[f"{column}__dyn_sign_change_count"] = float(sign_changes)
        if "upward_ratio" in enabled_features:
            result[f"{column}__dyn_upward_ratio"] = float(np.mean(diffs > 0)) if len(diffs) > 0 else 0.0
        if "downward_ratio" in enabled_features:
            result[f"{column}__dyn_downward_ratio"] = float(np.mean(diffs < 0)) if len(diffs) > 0 else 0.0
        if "local_autocorr" in enabled_features:
            if len(valid_y) > 2 and np.std(valid_y[:-1], ddof=0) > 0 and np.std(valid_y[1:], ddof=0) > 0:
                result[f"{column}__dyn_local_autocorr"] = float(np.corrcoef(valid_y[:-1], valid_y[1:])[0, 1])
            else:
                result[f"{column}__dyn_local_autocorr"] = 0.0
        if "volatility_ratio" in enabled_features:
            denom = level_std if level_std > 1e-8 else 1.0
            result[f"{column}__dyn_volatility_ratio"] = float(diff_std / denom)

    return result


def build_stage2_table(
    labeled_samples: pd.DataFrame,
    dcs: pd.DataFrame,
    tau_minutes: int,
    window_minutes: int,
    stats: list[str],
    enabled_dynamic_features: list[str],
    min_points_per_window: int,
) -> pd.DataFrame:
    sample_frame = labeled_samples.copy()
    sample_frame["sample_time"] = pd.to_datetime(sample_frame["sample_time"], errors="coerce")
    sample_frame = sample_frame.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)

    dcs_frame = dcs.copy()
    dcs_frame["time"] = pd.to_datetime(dcs_frame["time"], errors="coerce")
    dcs_frame = dcs_frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")

    rows: list[dict[str, object]] = []
    for record in sample_frame.itertuples(index=False):
        decision_time = pd.Timestamp(record.sample_time)
        window_end = decision_time - pd.Timedelta(minutes=tau_minutes)
        window_start = window_end - pd.Timedelta(minutes=window_minutes)
        left = np.searchsorted(dcs_times, window_start.to_datetime64(), side="left")
        right = np.searchsorted(dcs_times, window_end.to_datetime64(), side="right")
        window = dcs_frame.iloc[left:right]
        if len(window) < min_points_per_window:
            continue

        raw_stats = summarize_numeric_window(window, time_column="time")
        filtered_stats: dict[str, float] = {}
        for key, value in raw_stats.items():
            sensor, stat = key.split("__", 1)
            if stat in stats:
                filtered_stats[f"{sensor}__lag{tau_minutes}_win{window_minutes}_{stat}"] = float(value)

        dynamic_stats = summarize_dynamic_window(
            window=window,
            enabled_features=enabled_dynamic_features,
            time_column="time",
        )
        renamed_dynamic = {
            f"{sensor}__lag{tau_minutes}_win{window_minutes}_{suffix}": float(value)
            for sensor_suffix, value in dynamic_stats.items()
            for sensor, suffix in [sensor_suffix.split("__", 1)]
        }

        row = {
            "decision_time": decision_time,
            "sample_time": decision_time,
            "t90": getattr(record, "t90", np.nan),
            "is_in_spec": getattr(record, "is_in_spec", False),
            "is_out_of_spec": getattr(record, "is_out_of_spec", False),
            "is_above_spec": getattr(record, "is_above_spec", False),
            "is_below_spec": getattr(record, "is_below_spec", False),
            "target_centered_desirability": getattr(record, "target_centered_desirability", np.nan),
            "target_five_bin": getattr(record, "target_five_bin", pd.NA),
            "lag_minutes": int(tau_minutes),
            "window_minutes": int(window_minutes),
            "rows_in_window": int(len(window)),
        }
        row.update(filtered_stats)
        row.update(renamed_dynamic)
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)
    if "target_five_bin" in frame.columns:
        frame = frame.dropna(subset=["target_five_bin"]).copy()
        frame["target_five_bin"] = frame["target_five_bin"].astype(int)
    return frame


def build_feature_catalog(snapshot: pd.DataFrame, task_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in snapshot.columns:
        if "__" not in column:
            continue
        sensor, suffix = column.split("__", 1)
        family = "dynamic_morphology" if "_dyn_" in column else "lag_scale_base"
        rows.append(
            {
                "task_name": task_name,
                "feature_name": column,
                "sensor": sensor,
                "feature_suffix": suffix,
                "family": family,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon S2 dynamic morphology validation.")
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
        variant_name = f"{task_name}__lag{tau}_win{window}"

        snapshot = build_stage2_table(
            labeled_samples=labeled,
            dcs=dcs,
            tau_minutes=tau,
            window_minutes=window,
            stats=list(config["snapshot"]["stats"]),
            enabled_dynamic_features=list(config["dynamic_pack"]["enabled_features"]),
            min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
        )
        if snapshot.empty:
            continue

        base_feature_columns = [column for column in snapshot.columns if "__" in column and "_dyn_" not in column]
        dynamic_feature_columns = [column for column in snapshot.columns if "_dyn_" in column]
        full_feature_columns = [*base_feature_columns, *dynamic_feature_columns]
        feature_catalog_parts.append(build_feature_catalog(snapshot[full_feature_columns], task_name))

        fold_summaries: list[dict[str, Any]] = []
        agg: dict[str, list[dict[str, float]]] = {
            "simple_baseline::base_stats": [],
            "simple_baseline::base_plus_dynamic": [],
            "autogluon::base_stats": [],
            "autogluon::base_plus_dynamic": [],
        }

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
            train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
            test = snapshot.iloc[test_idx].copy().reset_index(drop=True)

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

                baseline.fit(train[full_feature_columns], train[label].to_numpy(dtype=float))
                full_pred = baseline.predict(test[full_feature_columns]).astype(float)
                full_metric = regression_metrics(
                    test[label].to_numpy(dtype=float),
                    full_pred,
                    test["is_in_spec"].to_numpy(dtype=int),
                )

                model_path_base = artifact_dir / f"ag_stage2_dynamic_{task_name}_base_{run_id}_fold{fold_idx}"
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

                model_path_full = artifact_dir / f"ag_stage2_dynamic_{task_name}_full_{run_id}_fold{fold_idx}"
                ag_full_pred, model_best_full = fit_autogluon_fold(
                    train_df=train[full_feature_columns + [label]].copy(),
                    test_df=test[full_feature_columns + [label]].copy(),
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

                baseline.fit(train[base_feature_columns], train_y)
                base_proba = baseline.predict_proba(test[base_feature_columns]).astype(float)
                base_frame = pd.DataFrame(base_proba, columns=baseline.named_steps["clf"].classes_.tolist())
                base_frame = base_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                base_frame = base_frame.div(base_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                base_metric = multiclass_metrics(test_y, base_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels)

                baseline.fit(train[full_feature_columns], train_y)
                full_proba = baseline.predict_proba(test[full_feature_columns]).astype(float)
                full_frame = pd.DataFrame(full_proba, columns=baseline.named_steps["clf"].classes_.tolist())
                full_frame = full_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                full_frame = full_frame.div(full_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                full_metric = multiclass_metrics(test_y, full_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels)

                model_path_base = artifact_dir / f"ag_stage2_dynamic_{task_name}_base_{run_id}_fold{fold_idx}"
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

                model_path_full = artifact_dir / f"ag_stage2_dynamic_{task_name}_full_{run_id}_fold{fold_idx}"
                ag_full_frame, model_best_full, leaderboard_full, feature_importance_full = fit_autogluon_multiclass(
                    train_df=train[full_feature_columns + [label]].copy(),
                    test_df=test[full_feature_columns + [label]].copy(),
                    label=label,
                    model_path=model_path_full,
                    ag_config=config["autogluon"],
                )
                ag_full_frame = ag_full_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                ag_full_frame = ag_full_frame.div(ag_full_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                ag_full_metric = multiclass_metrics(
                    test_y,
                    ag_full_frame.to_numpy(dtype=float),
                    class_labels=all_five_bin_labels,
                )
                if fold_idx == 1:
                    leaderboards[f"{task_name}::base_stats"] = leaderboard_base
                    feature_importances[f"{task_name}::base_stats"] = feature_importance_base
                    leaderboards[f"{task_name}::base_plus_dynamic"] = leaderboard_full
                    feature_importances[f"{task_name}::base_plus_dynamic"] = feature_importance_full

            if task_name == "centered_desirability" and fold_idx == 1:
                from autogluon.tabular import TabularPredictor

                predictor_base = TabularPredictor.load(str(model_path_base))
                leaderboards[f"{task_name}::base_stats"] = predictor_base.leaderboard(
                    test[base_feature_columns + [label]].copy(), silent=True
                )
                feature_importances[f"{task_name}::base_stats"] = predictor_base.feature_importance(
                    test[base_feature_columns + [label]].copy(), silent=True
                )

                predictor_full = TabularPredictor.load(str(model_path_full))
                leaderboards[f"{task_name}::base_plus_dynamic"] = predictor_full.leaderboard(
                    test[full_feature_columns + [label]].copy(), silent=True
                )
                feature_importances[f"{task_name}::base_plus_dynamic"] = predictor_full.feature_importance(
                    test[full_feature_columns + [label]].copy(), silent=True
                )

            fold_pairs = [
                ("simple_baseline", "base_stats", base_metric, len(base_feature_columns), None),
                ("simple_baseline", "base_plus_dynamic", full_metric, len(full_feature_columns), None),
                ("autogluon", "base_stats", ag_base_metric, len(base_feature_columns), model_best_base),
                ("autogluon", "base_plus_dynamic", ag_full_metric, len(full_feature_columns), model_best_full),
            ]
            for framework_name, feature_package, metrics, feature_count, model_best in fold_pairs:
                row = {
                    "stage": "stage2_dynamic_morphology",
                    "task_name": task_name,
                    "selected_stage1_variant": ref_row["variant_name"],
                    "tau_minutes": tau,
                    "window_minutes": window,
                    "feature_package": feature_package,
                    "framework": framework_name,
                    "fold": int(fold_idx),
                    "samples_train": int(len(train)),
                    "samples_test": int(len(test)),
                    "base_feature_count": int(len(base_feature_columns)),
                    "dynamic_feature_count": int(len(dynamic_feature_columns)),
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
                    "simple_baseline_dynamic": full_metric,
                    "autogluon_base": ag_base_metric,
                    "autogluon_dynamic": ag_full_metric,
                }
            )

        if task_name == "centered_desirability":
            base_baseline = float(np.nanmean([m["mae"] for m in agg["simple_baseline::base_stats"]]))
            full_baseline = float(np.nanmean([m["mae"] for m in agg["simple_baseline::base_plus_dynamic"]]))
            base_ag = float(np.nanmean([m["mae"] for m in agg["autogluon::base_stats"]]))
            full_ag = float(np.nanmean([m["mae"] for m in agg["autogluon::base_plus_dynamic"]]))
            stage1_ref = float(ref_row["autogluon_mean_mae"])
            summary_rows.append(
                {
                    "task_name": task_name,
                    "selected_stage1_variant": ref_row["variant_name"],
                    "tau_minutes": tau,
                    "window_minutes": window,
                    "base_feature_count": int(len(base_feature_columns)),
                    "dynamic_feature_count": int(len(dynamic_feature_columns)),
                    "full_feature_count": int(len(full_feature_columns)),
                    "baseline_base_mean_mae": base_baseline,
                    "baseline_dynamic_mean_mae": full_baseline,
                    "autogluon_base_mean_mae": base_ag,
                    "autogluon_dynamic_mean_mae": full_ag,
                    "stage1_autogluon_ref": stage1_ref,
                    "dynamic_beats_same_window_base": full_ag < base_ag,
                    "dynamic_beats_stage1_best": full_ag < stage1_ref,
                    "fold_summaries": fold_summaries,
                }
            )
        else:
            base_baseline = float(
                np.nanmean([m["multiclass_log_loss"] for m in agg["simple_baseline::base_stats"]])
            )
            full_baseline = float(
                np.nanmean([m["multiclass_log_loss"] for m in agg["simple_baseline::base_plus_dynamic"]])
            )
            base_ag = float(np.nanmean([m["multiclass_log_loss"] for m in agg["autogluon::base_stats"]]))
            full_ag = float(np.nanmean([m["multiclass_log_loss"] for m in agg["autogluon::base_plus_dynamic"]]))
            stage1_ref = float(ref_row["autogluon_mean_multiclass_log_loss"])
            summary_rows.append(
                {
                    "task_name": task_name,
                    "selected_stage1_variant": ref_row["variant_name"],
                    "tau_minutes": tau,
                    "window_minutes": window,
                    "base_feature_count": int(len(base_feature_columns)),
                    "dynamic_feature_count": int(len(dynamic_feature_columns)),
                    "full_feature_count": int(len(full_feature_columns)),
                    "baseline_base_mean_multiclass_log_loss": base_baseline,
                    "baseline_dynamic_mean_multiclass_log_loss": full_baseline,
                    "autogluon_base_mean_multiclass_log_loss": base_ag,
                    "autogluon_dynamic_mean_multiclass_log_loss": full_ag,
                    "stage1_autogluon_ref": stage1_ref,
                    "dynamic_beats_same_window_base": full_ag < base_ag,
                    "dynamic_beats_stage1_best": full_ag < stage1_ref,
                    "fold_summaries": fold_summaries,
                }
            )

    results_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    feature_catalog = pd.concat(feature_catalog_parts, ignore_index=True) if feature_catalog_parts else pd.DataFrame()

    results_path = artifact_dir / "stage2_dynamic_results.csv"
    summary_path = artifact_dir / "stage2_dynamic_summary.json"
    features_path = artifact_dir / "stage2_dynamic_features.csv"
    audit_path = report_dir / "stage2_dynamic_summary.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    feature_catalog.to_csv(features_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_df.to_dict(orient="records"), stream, ensure_ascii=False, indent=2)

    for task_name in ("centered_desirability", "five_bin"):
        for feature_package in ("base_stats", "base_plus_dynamic"):
            key = f"{task_name}::{feature_package}"
            if key in leaderboards:
                leaderboards[key].to_csv(
                    artifact_dir / f"stage2_dynamic_leaderboard_{task_name}_{feature_package}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            if key in feature_importances:
                feature_importances[key].to_csv(
                    artifact_dir / f"stage2_dynamic_feature_importance_{task_name}_{feature_package}.csv",
                    encoding="utf-8-sig",
                )

    audit_lines = [
        "# Stage 2 Dynamic Morphology Summary",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- Stage 2 adds a controlled dynamic morphology pack on top of the best S1 lag-scale package.",
        "- This stage remains restricted to `centered_desirability` and `five_bin`.",
        "",
        "## Dynamic Feature Pack",
        "",
        json.dumps(config["dynamic_pack"]["enabled_features"], ensure_ascii=False, indent=2),
        "",
        "## Selected S1 Base Packages",
        "",
        json.dumps(selected_variants, ensure_ascii=False, indent=2, default=str),
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
