from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_stage2_feature_engineering import (
    build_stage2_snapshot_table,
    fit_autogluon_fold,
    load_config,
    resolve_path,
    select_features_fold,
)
from run_autogluon_stage2_soft_probability import (
    build_soft_probability_snapshot,
    make_soft_probability_target,
    soft_probability_metrics,
)
from run_autogluon_stage2_soft_probability_x_enrichment import add_range_position_features


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import TargetSpec, add_out_of_spec_labels, load_dcs_frame, load_lims_samples, summarize_numeric_window


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_soft_probability_weak_compression_search.yaml"


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _prepare_soft_labeled_samples(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = config["paths"]
    spec = TargetSpec(
        center=float(config["target_spec"]["center"]),
        tolerance=float(config["target_spec"]["tolerance"]),
    )
    lims, _ = load_lims_samples(resolve_path(config_path.parent, paths["lims_path"]))
    labeled = add_out_of_spec_labels(lims, spec).dropna(subset=["t90"]).copy()
    labeled["sample_time"] = pd.to_datetime(labeled["sample_time"], errors="coerce")
    labeled = labeled.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)
    labeled[str(config["label_fuzziness"]["target_name"])] = make_soft_probability_target(
        labeled["t90"],
        center=float(config["target_spec"]["center"]),
        tolerance=float(config["target_spec"]["tolerance"]),
        boundary_softness=float(config["label_fuzziness"]["boundary_softness"]),
        rule=str(config["label_fuzziness"]["rule"]),
    )

    dcs = load_dcs_frame(
        resolve_path(config_path.parent, paths["dcs_main_path"]),
        resolve_path(config_path.parent, paths.get("dcs_supplemental_path")),
    )
    dcs["time"] = pd.to_datetime(dcs["time"], errors="coerce")
    dcs = dcs.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return labeled, dcs


def build_segmented_snapshot_table(
    labeled_samples: pd.DataFrame,
    dcs: pd.DataFrame,
    total_window_minutes: int,
    segment_minutes: int,
    stats: list[str],
    min_points_per_window: int,
    min_points_per_segment: int,
    target_label: str,
) -> pd.DataFrame:
    if total_window_minutes % segment_minutes != 0:
        raise ValueError(f"segment_minutes={segment_minutes} must evenly divide total_window_minutes={total_window_minutes}.")

    sample_frame = labeled_samples.copy()
    dcs_frame = dcs.copy()
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")
    segment_count = total_window_minutes // segment_minutes
    rows: list[dict[str, Any]] = []

    for record in sample_frame.itertuples(index=False):
        decision_time = pd.Timestamp(record.sample_time)
        window_end = decision_time
        window_start = window_end - pd.Timedelta(minutes=total_window_minutes)
        left = np.searchsorted(dcs_times, window_start.to_datetime64(), side="left")
        right = np.searchsorted(dcs_times, window_end.to_datetime64(), side="right")
        window = dcs_frame.iloc[left:right]
        if len(window) < min_points_per_window:
            continue

        feature_row: dict[str, float] = {}
        valid_sample = True
        for segment_idx in range(segment_count):
            segment_start = window_start + pd.Timedelta(minutes=segment_idx * segment_minutes)
            segment_end = segment_start + pd.Timedelta(minutes=segment_minutes)
            seg_left = np.searchsorted(dcs_times, segment_start.to_datetime64(), side="left")
            seg_right = np.searchsorted(dcs_times, segment_end.to_datetime64(), side="right")
            segment_window = dcs_frame.iloc[seg_left:seg_right]
            if len(segment_window) < min_points_per_segment:
                valid_sample = False
                break

            raw_stats = summarize_numeric_window(segment_window, time_column="time")
            for key, value in raw_stats.items():
                sensor, stat = key.split("__", 1)
                if stat in stats:
                    feature_row[
                        f"{sensor}__win{total_window_minutes}_seg{segment_idx + 1:02d}_len{segment_minutes}_{stat}"
                    ] = float(value)

        if not valid_sample:
            continue

        row = {
            "sample_time": decision_time,
            "t90": getattr(record, "t90", np.nan),
            "is_in_spec": getattr(record, "is_in_spec", False),
            "is_out_of_spec": getattr(record, "is_out_of_spec", False),
            "is_above_spec": getattr(record, "is_above_spec", False),
            "is_below_spec": getattr(record, "is_below_spec", False),
            target_label: getattr(record, target_label, np.nan),
            "window_minutes": int(total_window_minutes),
            "segment_minutes": int(segment_minutes),
            "segment_count": int(segment_count),
            "rows_in_window": int(len(window)),
        }
        row.update(feature_row)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)
    _add_range_columns(table)
    return table


def build_causal_path_snapshot_table(
    labeled_samples: pd.DataFrame,
    dcs: pd.DataFrame,
    total_window_minutes: int,
    step_minutes: int,
    min_points_per_window: int,
    target_label: str,
) -> pd.DataFrame:
    if total_window_minutes % step_minutes != 0:
        raise ValueError(f"step_minutes={step_minutes} must evenly divide total_window_minutes={total_window_minutes}.")

    sample_frame = labeled_samples.copy()
    dcs_frame = dcs.copy()
    numeric_columns = [column for column in dcs_frame.columns if column != "time"]
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")
    point_count = total_window_minutes // step_minutes
    rows: list[dict[str, Any]] = []

    for record in sample_frame.itertuples(index=False):
        decision_time = pd.Timestamp(record.sample_time)
        window_end = decision_time
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
                feature_row[f"{sensor}__win{total_window_minutes}_pathstep{step_minutes}_pt{point_idx:02d}"] = (
                    float(value) if pd.notna(value) else np.nan
                )

        row = {
            "sample_time": decision_time,
            "t90": getattr(record, "t90", np.nan),
            "is_in_spec": getattr(record, "is_in_spec", False),
            "is_out_of_spec": getattr(record, "is_out_of_spec", False),
            "is_above_spec": getattr(record, "is_above_spec", False),
            "is_below_spec": getattr(record, "is_below_spec", False),
            target_label: getattr(record, target_label, np.nan),
            "window_minutes": int(total_window_minutes),
            "step_minutes": int(step_minutes),
            "point_count": int(point_count),
            "rows_in_window": int(len(window)),
        }
        row.update(feature_row)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)


def _add_range_columns(table: pd.DataFrame) -> None:
    feature_columns = [column for column in table.columns if "__" in column]
    max_cols = [column for column in feature_columns if column.endswith("_max")]
    for max_col in max_cols:
        min_col = max_col[:-4] + "_min"
        range_col = max_col[:-4] + "_range"
        if min_col in table.columns and range_col not in table.columns:
            table[range_col] = table[max_col] - table[min_col]


def build_variant_snapshot(config_path: Path, config: dict[str, Any], variant_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_label = str(config["label_fuzziness"]["target_name"])
    if variant_name == "current_whole_window_ref":
        snapshot, audit = build_soft_probability_snapshot(config_path, config)
        snapshot, added_count = add_range_position_features(snapshot)
        feature_columns = [column for column in snapshot.columns if "__" in column]
        dropped_all_nan = [column for column in feature_columns if snapshot[column].isna().all()]
        if dropped_all_nan:
            snapshot = snapshot.drop(columns=dropped_all_nan)
        feature_columns = [column for column in snapshot.columns if "__" in column]
        dropped_constant = []
        for column in feature_columns:
            valid = snapshot[column].dropna()
            if not valid.empty and valid.nunique() <= 1:
                dropped_constant.append(column)
        if dropped_constant:
            snapshot = snapshot.drop(columns=dropped_constant)
        audit = {
            **audit,
            "variant_name": variant_name,
            "representation_family": "whole_window_range_position",
            "added_feature_count_before_post_clean": int(added_count),
            "feature_count_after_variant_build": int(len([column for column in snapshot.columns if "__" in column])),
        }
        return snapshot.sort_values("sample_time").reset_index(drop=True), audit

    labeled_samples, dcs = _prepare_soft_labeled_samples(config_path, config)
    lookback = int(config["snapshot"]["lookback_minutes"])
    min_points_per_window = int(config["snapshot"]["min_points_per_window"])
    audit: dict[str, Any]
    if variant_name.startswith("segmented_len_"):
        segment_minutes = int(variant_name.rsplit("_", 1)[-1])
        snapshot = build_segmented_snapshot_table(
            labeled_samples=labeled_samples,
            dcs=dcs,
            total_window_minutes=lookback,
            segment_minutes=segment_minutes,
            stats=[str(item) for item in config["snapshot"]["segmented_stats"]],
            min_points_per_window=min_points_per_window,
            min_points_per_segment=int(config["snapshot"]["min_points_per_segment"]),
            target_label=target_label,
        )
        audit = {
            "variant_name": variant_name,
            "representation_family": "segmented_stats",
            "segment_minutes": int(segment_minutes),
            "feature_count_after_variant_build": int(len([column for column in snapshot.columns if "__" in column])) if not snapshot.empty else 0,
        }
    elif variant_name.startswith("path_step_"):
        step_minutes = int(variant_name.rsplit("_", 1)[-1])
        snapshot = build_causal_path_snapshot_table(
            labeled_samples=labeled_samples,
            dcs=dcs,
            total_window_minutes=lookback,
            step_minutes=step_minutes,
            min_points_per_window=min_points_per_window,
            target_label=target_label,
        )
        audit = {
            "variant_name": variant_name,
            "representation_family": "causal_path",
            "step_minutes": int(step_minutes),
            "feature_count_after_variant_build": int(len([column for column in snapshot.columns if "__" in column])) if not snapshot.empty else 0,
        }
    else:
        raise ValueError(f"Unsupported variant: {variant_name}")

    return snapshot.sort_values("sample_time").reset_index(drop=True), audit


def evaluate_variant(
    variant_name: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
    artifact_dir: Path,
    splitter: TimeSeriesSplit,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_columns = [column for column in frame.columns if "__" in column]
    label = str(config["label_fuzziness"]["target_name"])
    top_k = int(config["selection"]["soft_probability_top_k"])
    rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, float]] = []
    raw_feature_count = 0
    selected_feature_count = 0
    selected_features_fold1: list[str] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(frame), start=1):
        train = frame.iloc[train_idx].copy().reset_index(drop=True)
        test = frame.iloc[test_idx].copy().reset_index(drop=True)
        raw_feature_count = int(len(feature_columns))
        selected_features, _ = select_features_fold(
            train_x=train[feature_columns],
            train_y=train[label],
            task_type="regression",
            top_k=top_k,
        )
        if not selected_features:
            selected_features = list(feature_columns)
        selected_feature_count = int(len(selected_features))
        if fold_idx == 1:
            selected_features_fold1 = list(selected_features)

        train_df = train[selected_features + [label]].copy()
        test_df = test[selected_features + [label]].copy()

        from run_autogluon_stage2_feature_engineering import make_regression_baseline

        baseline = make_regression_baseline()
        baseline.fit(train_df[selected_features], train_df[label].to_numpy(dtype=float))
        baseline_pred = baseline.predict(test_df[selected_features]).astype(float)
        baseline_metrics = soft_probability_metrics(
            test_df[label].to_numpy(dtype=float),
            baseline_pred,
            test["is_out_of_spec"].to_numpy(dtype=int),
        )

        model_path = artifact_dir / f"ag_soft_weak_{variant_name}_{run_id}_fold{fold_idx}"
        ag_pred, model_best = fit_autogluon_fold(
            train_df=train_df,
            test_df=test_df,
            label=label,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )
        ag_metrics = soft_probability_metrics(
            test_df[label].to_numpy(dtype=float),
            ag_pred.astype(float),
            test["is_out_of_spec"].to_numpy(dtype=int),
        )

        for framework_name, metrics in (
            ("simple_baseline_stage2_soft_probability", baseline_metrics),
            ("autogluon_stage2_soft_probability", ag_metrics),
        ):
            row = {
                "variant_name": variant_name,
                "fold": fold_idx,
                "framework": framework_name,
                "raw_feature_count": raw_feature_count,
                "selected_feature_count": selected_feature_count,
            }
            row.update(metrics)
            if framework_name == "autogluon_stage2_soft_probability":
                row["autogluon_model_best"] = model_best
            rows.append(row)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "baseline_soft_mae": float(baseline_metrics["soft_mae"]),
                "autogluon_soft_mae": float(ag_metrics["soft_mae"]),
                "baseline_soft_brier": float(baseline_metrics["soft_brier"]),
                "autogluon_soft_brier": float(ag_metrics["soft_brier"]),
                "autogluon_hard_out_ap": float(ag_metrics["hard_out_ap_diagnostic"]),
                "model_best": str(model_best),
            }
        )

    summary = {
        "variant_name": variant_name,
        "sample_count": int(len(frame)),
        "raw_feature_count": raw_feature_count,
        "selected_feature_count": selected_feature_count,
        "selected_features_fold1": selected_features_fold1,
        "baseline_mean_soft_mae": _safe_mean([item["baseline_soft_mae"] for item in fold_metrics]),
        "autogluon_mean_soft_mae": _safe_mean([item["autogluon_soft_mae"] for item in fold_metrics]),
        "baseline_mean_soft_brier": _safe_mean([item["baseline_soft_brier"] for item in fold_metrics]),
        "autogluon_mean_soft_brier": _safe_mean([item["autogluon_soft_brier"] for item in fold_metrics]),
        "autogluon_mean_hard_out_ap_diagnostic": _safe_mean([item["autogluon_hard_out_ap"] for item in fold_metrics]),
    }
    return rows, summary


def build_variant_feature_catalog(features: list[str], variant_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in features:
        if "__" not in column:
            continue
        sensor, suffix = column.split("__", 1)
        if "_seg" in suffix:
            family = "segmented_stats"
        elif "_pathstep" in suffix:
            family = "causal_path"
        elif "position_in_range" in suffix or "gap_ratio" in suffix:
            family = "range_position"
        else:
            family = "whole_window_stats"
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


def write_audit(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    best_row = min(summary_rows, key=lambda row: row["autogluon_mean_soft_brier"])
    lines = [
        "# Soft Probability Weak Compression Search",
        "",
        "## Scope",
        "",
        "- Task: soft probability branch only",
        "- Keep the validated fuzzy Y design fixed",
        "- Compare the current whole-window range-position reference against weaker-compression X representations",
        "",
        "## Ranked Summary Rows",
        "",
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        "",
        "## Best Variant By AutoGluon Soft Brier",
        "",
        json.dumps(best_row, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weak-compression search for the AutoGluon soft probability branch.")
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

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))

    variant_names = ["current_whole_window_ref"]
    variant_names += [f"segmented_len_{int(v)}" for v in config["snapshot"]["segmented_lengths"]]
    variant_names += [f"path_step_{int(v)}" for v in config["snapshot"]["path_steps"]]

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []
    catalogs: list[pd.DataFrame] = []

    for variant_name in variant_names:
        frame, audit = build_variant_snapshot(config_path, config, variant_name)
        if frame.empty:
            continue
        rows, summary = evaluate_variant(
            variant_name=variant_name,
            frame=frame,
            config=config,
            artifact_dir=artifact_dir,
            splitter=splitter,
            run_id=run_id,
        )
        summary.update(audit)
        if variant_name == "current_whole_window_ref":
            summary["current_reference_soft_brier"] = float(summary["autogluon_mean_soft_brier"])
            summary["delta_vs_current_reference"] = 0.0
        summary_rows.append(summary)
        detail_rows.append(pd.DataFrame(rows))
        catalogs.append(build_variant_feature_catalog([column for column in frame.columns if "__" in column], variant_name))

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise ValueError("No valid variant finished in soft probability weak-compression search.")
    ref_brier = float(summary_df.loc[summary_df["variant_name"] == "current_whole_window_ref", "autogluon_mean_soft_brier"].iloc[0])
    summary_df["delta_vs_current_reference"] = summary_df["autogluon_mean_soft_brier"] - ref_brier
    summary_df["beats_current_reference"] = summary_df["autogluon_mean_soft_brier"] < ref_brier
    summary_df = summary_df.sort_values(
        by=["autogluon_mean_soft_brier", "autogluon_mean_soft_mae", "variant_name"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    detail_df = pd.concat(detail_rows, ignore_index=True)
    catalog_df = pd.concat(catalogs, ignore_index=True)

    summary_rows_path = artifact_dir / "soft_probability_weak_compression_search_summary_rows.csv"
    results_path = artifact_dir / "soft_probability_weak_compression_search_results.csv"
    catalog_path = artifact_dir / "soft_probability_weak_compression_feature_catalog.csv"
    summary_json_path = artifact_dir / "soft_probability_weak_compression_search_summary.json"
    report_path = report_dir / "soft_probability_weak_compression_search_summary.md"

    summary_df.to_csv(summary_rows_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    catalog_df.to_csv(catalog_path, index=False, encoding="utf-8-sig")
    with summary_json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_df.to_dict(orient="records"), stream, ensure_ascii=False, indent=2)
    write_audit(report_path, summary_df.to_dict(orient="records"))

    print(
        json.dumps(
            {
                "summary_rows_path": str(summary_rows_path),
                "results_path": str(results_path),
                "catalog_path": str(catalog_path),
                "summary_json_path": str(summary_json_path),
                "report_path": str(report_path),
                "best_variant": summary_df.iloc[0].to_dict(),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
