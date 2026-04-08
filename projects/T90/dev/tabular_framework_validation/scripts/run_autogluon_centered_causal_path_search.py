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
from run_autogluon_stage4_interactions import build_interaction_frame
from run_autogluon_stage5_quality import build_quality_table
from run_autogluon_stage7_final_selection import preclean_features, supervised_select


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_centered_causal_path_search.yaml"


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def build_causal_path_snapshot_table(
    labeled_samples: pd.DataFrame,
    dcs: pd.DataFrame,
    tau_minutes: int,
    total_window_minutes: int,
    step_minutes: int,
    min_points_per_window: int,
) -> pd.DataFrame:
    if total_window_minutes % step_minutes != 0:
        raise ValueError(f"step_minutes={step_minutes} must evenly divide total_window_minutes={total_window_minutes}.")

    sample_frame = labeled_samples.copy()
    sample_frame["sample_time"] = pd.to_datetime(sample_frame["sample_time"], errors="coerce")
    sample_frame = sample_frame.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)

    dcs_frame = dcs.copy()
    dcs_frame["time"] = pd.to_datetime(dcs_frame["time"], errors="coerce")
    dcs_frame = dcs_frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    numeric_columns = [column for column in dcs_frame.columns if column != "time"]
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")

    point_count = total_window_minutes // step_minutes
    rows: list[dict[str, Any]] = []

    for record in sample_frame.itertuples(index=False):
        decision_time = pd.Timestamp(record.sample_time)
        window_end = decision_time - pd.Timedelta(minutes=tau_minutes)
        window_start = window_end - pd.Timedelta(minutes=total_window_minutes)
        left = np.searchsorted(dcs_times, window_start.to_datetime64(), side="left")
        right = np.searchsorted(dcs_times, window_end.to_datetime64(), side="right")
        window = dcs_frame.iloc[left:right].copy()
        if len(window) < min_points_per_window:
            continue

        resampled = (
            window.set_index("time")[numeric_columns]
            .sort_index()
            .resample(f"{step_minutes}min")
            .last()
        )
        expected_index = pd.date_range(start=window_start, periods=point_count, freq=f"{step_minutes}min")
        resampled = resampled.reindex(expected_index).ffill()

        feature_row: dict[str, float] = {}
        for point_idx, (_, row_vals) in enumerate(resampled.iterrows()):
            for sensor in numeric_columns:
                value = pd.to_numeric(row_vals[sensor], errors="coerce")
                feature_row[
                    f"{sensor}__lag{tau_minutes}_win{total_window_minutes}_pathstep{step_minutes}_pt{point_idx:02d}"
                ] = float(value) if pd.notna(value) else np.nan

        row = {
            "decision_time": decision_time,
            "sample_time": decision_time,
            "t90": getattr(record, "t90", np.nan),
            "is_in_spec": getattr(record, "is_in_spec", False),
            "is_out_of_spec": getattr(record, "is_out_of_spec", False),
            "is_above_spec": getattr(record, "is_above_spec", False),
            "is_below_spec": getattr(record, "is_below_spec", False),
            "target_centered_desirability": getattr(record, "target_centered_desirability", np.nan),
            "lag_minutes": int(tau_minutes),
            "window_minutes": int(total_window_minutes),
            "step_minutes": int(step_minutes),
            "point_count": int(point_count),
            "rows_in_window": int(len(window)),
        }
        row.update(feature_row)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)


def build_variant_feature_catalog(features: list[str], variant_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in features:
        if column.startswith("interaction__"):
            sensor = "interaction"
            suffix = column.replace("interaction__", "", 1)
            family = "process_interaction"
        elif column.startswith("quality__"):
            parts = column.split("__", 2)
            sensor = parts[1] if len(parts) > 2 else "quality"
            suffix = parts[2] if len(parts) > 2 else column
            family = "quality"
        else:
            sensor, suffix = column.split("__", 1)
            family = "causal_path" if "_pathstep" in suffix else "whole_window_lag_scale"
        rows.append(
            {
                "variant_name": variant_name,
                "feature_name": column,
                "sensor": sensor,
                "feature_suffix": suffix,
                "family": family,
            }
        )
    return pd.DataFrame(rows)


def evaluate_variant(
    variant_name: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    top_k: int,
    config: dict[str, Any],
    artifact_dir: Path,
    splitter: TimeSeriesSplit,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, float]] = []
    raw_feature_count = 0
    cleaned_feature_count = 0
    selected_feature_count = 0
    selected_features_fold1: list[str] = []

    label = str(config["task"]["label"])

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

        model_path = artifact_dir / f"ag_centered_path_{variant_name}_topk{top_k}_{run_id}_fold{fold_idx}"
        ag_pred, model_best = fit_autogluon_fold(
            train_df=pd.concat([train_clean[selected_cols].copy(), train[[label]].copy()], axis=1),
            test_df=pd.concat([test_clean[selected_cols].copy(), test[[label]].copy()], axis=1),
            label=label,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )
        ag_metric = regression_metrics(
            y_true=y_test,
            pred=ag_pred.astype(float),
            in_spec_flag=test["is_in_spec"].to_numpy(dtype=int),
        )

        for framework_name, metrics in (("simple_baseline", baseline_metric), ("autogluon", ag_metric)):
            row = {
                "variant_name": variant_name,
                "top_k": int(top_k),
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

    summary = {
        "variant_name": variant_name,
        "top_k": int(top_k),
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
    }
    return rows, summary


def write_report(path: Path, summary_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    if not summary_rows:
        path.write_text("# Centered Causal Path Search\n\nNo valid variants were evaluated.\n", encoding="utf-8")
        return

    best = summary_rows[0]
    current_ref = next((row for row in summary_rows if row["variant_name"] == "current_whole_window_ref"), None)
    lines = [
        "# Centered Causal Path Search",
        "",
        "## Scope",
        "",
        "- Task: centered_desirability only",
        "- Total causal window fixed at lag120_win60",
        "- The current centered best line is used as the direct reference",
        "- The experiment replaces the whole-window summary snapshot with lower-compression causal path features",
        "",
        "## Fixed Settings",
        "",
        f"- lag_minutes: {config['snapshot']['lag_minutes']}",
        f"- total_window_minutes: {config['snapshot']['total_window_minutes']}",
        f"- top_k_candidates: {config['path_encoding']['top_k_candidates']}",
        f"- interaction_package: {config['packages']['interaction_package']}",
        "- quality_package: combined_quality",
        "",
        "## Best Variant",
        "",
        json.dumps(best, ensure_ascii=False, indent=2),
        "",
    ]
    if current_ref is not None:
        lines.extend(
            [
                "## Current Reference",
                "",
                json.dumps(current_ref, ensure_ascii=False, indent=2),
                "",
            ]
        )
    lines.extend(
        [
            "## Ranked Summary Rows",
            "",
            json.dumps(summary_rows, ensure_ascii=False, indent=2),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Centered-only causal-path search on top of the validated centered line.")
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

    tau_minutes = int(config["snapshot"]["lag_minutes"])
    total_window_minutes = int(config["snapshot"]["total_window_minutes"])
    stats = list(config["snapshot"]["stats"])
    min_points_per_window = int(config["snapshot"]["min_points_per_window"])

    whole_snapshot = build_stage2_table(
        labeled_samples=labeled,
        dcs=dcs,
        tau_minutes=tau_minutes,
        window_minutes=total_window_minutes,
        stats=stats,
        enabled_dynamic_features=[],
        min_points_per_window=min_points_per_window,
    )
    if whole_snapshot.empty:
        raise ValueError("Whole-window centered snapshot is empty.")

    interaction_frame = build_interaction_frame(
        snapshot=whole_snapshot,
        tau_minutes=tau_minutes,
        window_minutes=total_window_minutes,
        package_name=str(config["packages"]["interaction_package"]),
    )
    interaction_frame = pd.concat([whole_snapshot[["decision_time"]].copy(), interaction_frame], axis=1)
    quality_table = build_quality_table(
        labeled_samples=labeled,
        dcs=dcs,
        tau_minutes=tau_minutes,
        window_minutes=total_window_minutes,
        enabled_features=list(config["packages"]["quality_features"]),
        min_points_per_window=min_points_per_window,
    )

    current_frame = whole_snapshot.merge(interaction_frame, on="decision_time", how="inner").merge(
        quality_table,
        on="decision_time",
        how="inner",
    )

    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    feature_catalog_parts: list[pd.DataFrame] = []

    current_features = [column for column in current_frame.columns if "__" in column]
    for top_k in config["path_encoding"]["top_k_candidates"]:
        top_k = int(top_k)
        rows, summary = evaluate_variant(
            variant_name="current_whole_window_ref",
            frame=current_frame,
            feature_columns=current_features,
            top_k=top_k,
            config=config,
            artifact_dir=artifact_dir,
            splitter=splitter,
            run_id=run_id,
        )
        all_rows.extend(rows)
        summary["step_minutes"] = None
        summary["point_count"] = 1
        feature_catalog_parts.append(build_variant_feature_catalog(current_features, f"current_whole_window_ref_topk{top_k}"))
        summary_rows.append(summary)

    for step_minutes in config["path_encoding"]["step_minutes_candidates"]:
        step_minutes = int(step_minutes)
        path_snapshot = build_causal_path_snapshot_table(
            labeled_samples=labeled,
            dcs=dcs,
            tau_minutes=tau_minutes,
            total_window_minutes=total_window_minutes,
            step_minutes=step_minutes,
            min_points_per_window=min_points_per_window,
        )
        if path_snapshot.empty:
            continue
        path_frame = path_snapshot.merge(interaction_frame, on="decision_time", how="inner").merge(
            quality_table,
            on="decision_time",
            how="inner",
        )
        path_features = [column for column in path_frame.columns if "__" in column]
        variant_name = f"path_step_{step_minutes:02d}"
        for top_k in config["path_encoding"]["top_k_candidates"]:
            top_k = int(top_k)
            rows, summary = evaluate_variant(
                variant_name=variant_name,
                frame=path_frame,
                feature_columns=path_features,
                top_k=top_k,
                config=config,
                artifact_dir=artifact_dir,
                splitter=splitter,
                run_id=run_id,
            )
            all_rows.extend(rows)
            summary["step_minutes"] = step_minutes
            summary["point_count"] = int(total_window_minutes // step_minutes)
            feature_catalog_parts.append(build_variant_feature_catalog(path_features, f"{variant_name}_topk{top_k}"))
            summary_rows.append(summary)

    current_ref_mae = min(
        float(row["autogluon_mean_mae"]) for row in summary_rows if row["variant_name"] == "current_whole_window_ref"
    )
    for row in summary_rows:
        row["current_ref_best_autogluon_mae"] = current_ref_mae
        row["delta_vs_current_ref"] = float(row["autogluon_mean_mae"] - current_ref_mae)
        row["beats_current_ref"] = bool(row["autogluon_mean_mae"] < current_ref_mae)

    summary_rows = sorted(summary_rows, key=lambda item: (float(item["autogluon_mean_mae"]), str(item["variant_name"]), int(item["top_k"])))
    best_variant = summary_rows[0] if summary_rows else None

    results_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)
    feature_catalog_df = (
        pd.concat(feature_catalog_parts, ignore_index=True).sort_values(["variant_name", "family", "feature_name"]).reset_index(drop=True)
        if feature_catalog_parts
        else pd.DataFrame()
    )

    results_path = artifact_dir / "centered_causal_path_search_results.csv"
    summary_path = artifact_dir / "centered_causal_path_search_summary.json"
    catalog_path = artifact_dir / "centered_causal_path_feature_catalog.csv"
    report_path = report_dir / "centered_causal_path_search_summary.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(artifact_dir / "centered_causal_path_search_summary_rows.csv", index=False, encoding="utf-8-sig")
    feature_catalog_df.to_csv(catalog_path, index=False, encoding="utf-8-sig")
    summary_payload = {
        "phase": str(config["phase"]),
        "task_name": str(config["task"]["name"]),
        "best_variant": best_variant,
        "summary_rows": summary_rows,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, summary_rows, config)


if __name__ == "__main__":
    main()
