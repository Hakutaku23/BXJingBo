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

from core import TargetSpec, add_out_of_spec_labels, load_dcs_frame, load_lims_samples, summarize_numeric_window

DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage1_lag_scale.yaml"


def build_label_frame(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    paths = config["paths"]
    spec = TargetSpec(
        center=float(config["target_spec"]["center"]),
        tolerance=float(config["target_spec"]["tolerance"]),
    )
    lims, _ = load_lims_samples(resolve_path(config_path.parent, paths["lims_path"]))
    labeled = add_out_of_spec_labels(lims, spec).dropna(subset=["t90"]).copy()
    center = float(config["target_spec"]["center"])
    tol = float(config["target_spec"]["tolerance"])
    labeled["target_centered_desirability"] = np.maximum(0.0, 1.0 - (labeled["t90"] - center).abs() / tol)
    bins = config["tasks"]["five_bin"]["bins"]
    labeled["target_five_bin"] = pd.cut(labeled["t90"], bins=bins, labels=False, include_lowest=True).astype("Int64")
    return labeled


def build_lag_scale_table(
    labeled_samples: pd.DataFrame,
    dcs: pd.DataFrame,
    tau_minutes: int,
    window_minutes: int,
    stats: list[str],
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

        raw_features = summarize_numeric_window(window, time_column="time")
        filtered: dict[str, float] = {}
        for key, value in raw_features.items():
            sensor, stat = key.split("__", 1)
            if stat in stats:
                filtered[f"{sensor}__lag{tau_minutes}_win{window_minutes}_{stat}"] = float(value)

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
        row.update(filtered)
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)
    if "target_five_bin" in frame.columns:
        frame = frame.dropna(subset=["target_five_bin"]).copy()
        frame["target_five_bin"] = frame["target_five_bin"].astype(int)
    return frame


def build_feature_catalog(snapshot: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in snapshot.columns:
        if "__" not in column:
            continue
        sensor, suffix = column.split("__", 1)
        rows.append(
            {
                "variant_name": variant_name,
                "feature_name": column,
                "sensor": sensor,
                "feature_suffix": suffix,
                "family": "stage1_lag_scale",
            }
        )
    return pd.DataFrame(rows)


def stage0_reference(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(config_path.parent, config["paths"]["stage0_summary_path"])
    if path is None or not path.exists():
        raise ValueError("stage0_summary_path must exist before running S1 lag-scale.")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon S1 lag-scale feature package validation.")
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
    stage0_summary = stage0_reference(config_path, config)
    all_five_bin_labels = sorted(labeled["target_five_bin"].dropna().astype(int).unique().tolist())

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    feature_catalog_parts: list[pd.DataFrame] = []
    leaderboards: dict[str, pd.DataFrame] = {}
    feature_importances: dict[str, pd.DataFrame] = {}
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    for tau in config["lag_scale"]["taus"]:
        for window in config["lag_scale"]["windows"]:
            variant_name = f"lag{int(tau)}_win{int(window)}"
            snapshot = build_lag_scale_table(
                labeled_samples=labeled,
                dcs=dcs,
                tau_minutes=int(tau),
                window_minutes=int(window),
                stats=list(config["snapshot"]["stats"]),
                min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
            )
            feature_columns = [column for column in snapshot.columns if "__" in column]
            feature_catalog_parts.append(build_feature_catalog(snapshot, variant_name))

            if snapshot.empty or not feature_columns:
                continue

            for task_name in ("centered_desirability", "five_bin"):
                fold_summaries: list[dict[str, Any]] = []
                baseline_agg: list[dict[str, Any]] = []
                ag_agg: list[dict[str, Any]] = []

                for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
                    train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
                    test = snapshot.iloc[test_idx].copy().reset_index(drop=True)

                    if task_name == "centered_desirability":
                        label = "target_centered_desirability"
                        baseline = make_regression_baseline()
                        baseline.fit(train[feature_columns], train[label].to_numpy(dtype=float))
                        baseline_pred = baseline.predict(test[feature_columns]).astype(float)
                        baseline_metric = regression_metrics(
                            test[label].to_numpy(dtype=float),
                            baseline_pred,
                            test["is_in_spec"].to_numpy(dtype=int),
                        )

                        model_path = artifact_dir / f"ag_stage1_lag_scale_{variant_name}_{task_name}_{run_id}_fold{fold_idx}"
                        ag_pred, model_best = fit_autogluon_fold(
                            train_df=train[feature_columns + [label]].copy(),
                            test_df=test[feature_columns + [label]].copy(),
                            label=label,
                            problem_type="regression",
                            eval_metric="root_mean_squared_error",
                            model_path=model_path,
                            ag_config=config["autogluon"],
                        )
                        ag_metric = regression_metrics(
                            test[label].to_numpy(dtype=float),
                            ag_pred.astype(float),
                            test["is_in_spec"].to_numpy(dtype=int),
                        )
                    else:
                        label = "target_five_bin"
                        baseline = make_multiclass_baseline()
                        train_y = train[label].to_numpy(dtype=int)
                        test_y = test[label].to_numpy(dtype=int)
                        baseline.fit(train[feature_columns], train_y)
                        baseline_proba = baseline.predict_proba(test[feature_columns]).astype(float)
                        observed_labels = baseline.named_steps["clf"].classes_.tolist()
                        baseline_frame = pd.DataFrame(baseline_proba, columns=observed_labels)
                        baseline_frame = baseline_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                        baseline_frame = baseline_frame.div(baseline_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                        baseline_metric = multiclass_metrics(test_y, baseline_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels)

                        model_path = artifact_dir / f"ag_stage1_lag_scale_{variant_name}_{task_name}_{run_id}_fold{fold_idx}"
                        ag_proba_frame, model_best, leaderboard, feature_importance = fit_autogluon_multiclass(
                            train_df=train[feature_columns + [label]].copy(),
                            test_df=test[feature_columns + [label]].copy(),
                            label=label,
                            model_path=model_path,
                            ag_config=config["autogluon"],
                        )
                        ag_proba_frame = ag_proba_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                        ag_proba_frame = ag_proba_frame.div(ag_proba_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                        ag_metric = multiclass_metrics(test_y, ag_proba_frame.to_numpy(dtype=float), class_labels=all_five_bin_labels)
                        if fold_idx == 1:
                            leaderboards[f"{task_name}::{variant_name}"] = leaderboard
                            feature_importances[f"{task_name}::{variant_name}"] = feature_importance

                    if task_name == "centered_desirability" and fold_idx == 1:
                        from autogluon.tabular import TabularPredictor

                        predictor = TabularPredictor.load(str(model_path))
                        leaderboards[f"{task_name}::{variant_name}"] = predictor.leaderboard(
                            test[feature_columns + [label]].copy(), silent=True
                        )
                        feature_importances[f"{task_name}::{variant_name}"] = predictor.feature_importance(
                            test[feature_columns + [label]].copy(), silent=True
                        )

                    for framework_name, metrics in (("simple_baseline", baseline_metric), ("autogluon", ag_metric)):
                        rows.append(
                            {
                                "stage": "stage1_lag_scale",
                                "variant_name": variant_name,
                                "tau_minutes": int(tau),
                                "window_minutes": int(window),
                                "task_name": task_name,
                                "framework": framework_name,
                                "fold": int(fold_idx),
                                "samples_train": int(len(train)),
                                "samples_test": int(len(test)),
                                "feature_count": int(len(feature_columns)),
                                "data_source_condition": config["data_source_statement"]["source_condition"],
                                **metrics,
                                **({"autogluon_model_best": model_best} if framework_name == "autogluon" else {}),
                            }
                        )
                    baseline_agg.append(baseline_metric)
                    ag_agg.append(ag_metric)
                    fold_summaries.append({"fold": int(fold_idx), "baseline": baseline_metric, "autogluon": ag_metric})

                if task_name == "centered_desirability":
                    baseline_mean = float(np.nanmean([m["mae"] for m in baseline_agg]))
                    ag_mean = float(np.nanmean([m["mae"] for m in ag_agg]))
                    stage0_ref = float(stage0_summary["stage_summary"]["tasks"]["centered_desirability"]["autogluon_mean_mae"])
                    summary_rows.append(
                        {
                            "variant_name": variant_name,
                            "tau_minutes": int(tau),
                            "window_minutes": int(window),
                            "task_name": task_name,
                            "feature_count": int(len(feature_columns)),
                            "baseline_mean_mae": baseline_mean,
                            "autogluon_mean_mae": ag_mean,
                            "stage0_autogluon_ref": stage0_ref,
                            "beats_stage0": ag_mean < stage0_ref,
                            "positive_signal": ag_mean < baseline_mean,
                            "fold_summaries": fold_summaries,
                        }
                    )
                else:
                    baseline_mean = float(np.nanmean([m["multiclass_log_loss"] for m in baseline_agg]))
                    ag_mean = float(np.nanmean([m["multiclass_log_loss"] for m in ag_agg]))
                    stage0_ref = float(stage0_summary["stage_summary"]["tasks"]["five_bin"]["autogluon_mean_multiclass_log_loss"])
                    summary_rows.append(
                        {
                            "variant_name": variant_name,
                            "tau_minutes": int(tau),
                            "window_minutes": int(window),
                            "task_name": task_name,
                            "feature_count": int(len(feature_columns)),
                            "baseline_mean_multiclass_log_loss": baseline_mean,
                            "autogluon_mean_multiclass_log_loss": ag_mean,
                            "stage0_autogluon_ref": stage0_ref,
                            "beats_stage0": ag_mean < stage0_ref,
                            "positive_signal": ag_mean < baseline_mean,
                            "fold_summaries": fold_summaries,
                        }
                    )

    results_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    feature_catalog = pd.concat(feature_catalog_parts, ignore_index=True) if feature_catalog_parts else pd.DataFrame()

    summary_json = summary_df.to_dict(orient="records")
    results_path = artifact_dir / "stage1_lag_scale_results.csv"
    summary_path = artifact_dir / "stage1_lag_scale_summary.json"
    features_path = artifact_dir / "stage1_lag_scale_features.csv"
    audit_path = report_dir / "stage1_lag_scale_summary.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    feature_catalog.to_csv(features_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_json, stream, ensure_ascii=False, indent=2)

    lines = [
        "# Stage 1 Lag-Scale Summary",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- Stage 1 adds lag-scale packages without introducing future leakage.",
        "- This stage now focuses only on `centered_desirability` and `five_bin` by user decision.",
        "",
        "## Grid",
        "",
        f"- taus: {config['lag_scale']['taus']}",
        f"- windows: {config['lag_scale']['windows']}",
        "",
        "## Reference",
        "",
        "- Each lag-scale package is compared against the S0 AutoGluon baseline.",
        "",
        "## Summary Rows",
        "",
        json.dumps(summary_json, ensure_ascii=False, indent=2),
    ]
    audit_path.write_text("\n".join(lines), encoding="utf-8")

    for task_name in ("centered_desirability", "five_bin"):
        task_rows = summary_df[summary_df["task_name"] == task_name].copy()
        if task_rows.empty:
            continue
        if task_name == "centered_desirability":
            best_row = task_rows.sort_values("autogluon_mean_mae", ascending=True).iloc[0]
        else:
            best_row = task_rows.sort_values("autogluon_mean_multiclass_log_loss", ascending=True).iloc[0]
        key = f"{task_name}::{best_row['variant_name']}"
        if key in leaderboards:
            leaderboards[key].to_csv(
                artifact_dir / f"stage1_lag_scale_leaderboard_best_{task_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if key in feature_importances:
            feature_importances[key].to_csv(
                artifact_dir / f"stage1_lag_scale_feature_importance_best_{task_name}.csv",
                encoding="utf-8-sig",
            )

    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "features_path": str(features_path),
                "audit_path": str(audit_path),
                "best_centered_desirability": (
                    summary_df[summary_df["task_name"] == "centered_desirability"]
                    .sort_values("autogluon_mean_mae", ascending=True)
                    .head(1)
                    .to_dict(orient="records")
                ),
                "best_five_bin": (
                    summary_df[summary_df["task_name"] == "five_bin"]
                    .sort_values("autogluon_mean_multiclass_log_loss", ascending=True)
                    .head(1)
                    .to_dict(orient="records")
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
