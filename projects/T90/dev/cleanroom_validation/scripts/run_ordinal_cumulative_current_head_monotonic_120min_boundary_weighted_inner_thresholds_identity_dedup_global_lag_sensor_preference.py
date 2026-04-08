from __future__ import annotations

import argparse
import ast
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from run_ordinal_cumulative_current_head import (
    discover_lims_path,
    feature_columns,
    format_threshold,
    load_lims_data,
    preclean_features,
    selected_feature_columns,
)
from run_ordinal_cumulative_current_head_monotonic_120min_ablation_v2 import (
    read_point_metadata,
    sensor_display_name,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup import (
    BUSINESS_LABELS,
    CLEANROOM_DIR,
    PROJECT_DIR,
    evaluate_prediction_set,
    fit_cumulative_business_probabilities_inner_thresholds_only,
    load_config as load_baseline_config,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag import (
    build_shifted_feature_rows,
    lag_candidate_name,
)
from run_ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head import (
    load_identity_deduped_combined_dcs,
)
from run_ordinal_cumulative_current_head import probability_argmax, assign_labels


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = (
    CLEANROOM_DIR
    / "configs"
    / "ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag_sensor_preference.yaml"
)
BASE_LAG_CONFIG_PATH = (
    CLEANROOM_DIR
    / "configs"
    / "ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag.yaml"
)


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_list_cell(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = ast.literal_eval(text)
    return [str(item) for item in parsed]


def align_to_sample_times(frame: pd.DataFrame, sample_times: list[pd.Timestamp]) -> pd.DataFrame:
    time_index = {timestamp: order for order, timestamp in enumerate(sample_times)}
    filtered = frame.loc[frame["sample_time"].isin(time_index)].copy()
    filtered["_order"] = filtered["sample_time"].map(time_index)
    return filtered.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def evaluate_sensor_subset(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    sensors: list[str],
    feature_cols: list[str],
    thresholds: list[float],
    baseline_config: dict[str, object],
    weight_candidate: dict[str, object],
) -> dict[str, object]:
    selected_features = selected_feature_columns(feature_cols, sensors)
    cumulative, business_probabilities = fit_cumulative_business_probabilities_inner_thresholds_only(
        train_frame,
        test_frame,
        features=selected_features,
        thresholds=thresholds,
        config=baseline_config,
        candidate=weight_candidate,
    )
    predicted = probability_argmax(business_probabilities, BUSINESS_LABELS)
    metrics = evaluate_prediction_set(test_frame, business_probabilities, predicted)
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "warning_AP": float(metrics["boundary_warning_average_precision"]),
        "unacceptable_AP": float(metrics["clearly_unacceptable_average_precision"]),
        "boundary_high_conf_non_warning_rate": float(metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"]),
        "selected_feature_count": int(len(selected_features)),
    }


def safe_mean(frame: pd.DataFrame, column: str) -> float:
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return math.nan
    return float(series.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare which selected sensors support L60 overall gains versus L180 boundary caution under the frozen lag cleanroom."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    baseline_config = load_baseline_config(BASE_LAG_CONFIG_PATH)
    base_summary_path = CLEANROOM_DIR / str(config["base_global_lag_summary_json"])
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))

    outputs = config["outputs"]
    artifacts_dir = CLEANROOM_DIR / str(outputs["artifacts_dir"])
    reports_dir = CLEANROOM_DIR / str(outputs["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    point_metadata = read_point_metadata(PROJECT_DIR / str(config["point_metadata_path"]))
    compare_lags = [int(item) for item in config["compare_lags"]]
    lag_names = [lag_candidate_name("L", lag) for lag in compare_lags]
    thresholds = [float(item) for item in baseline_config["labels"]["cumulative_thresholds"]]

    selection_csv = Path(base_summary["artifacts"]["baseline_selection_csv"])
    selection_frame = pd.read_csv(selection_csv)
    selection_frame["selected_sensors"] = selection_frame["selected_sensors"].apply(parse_list_cell)

    aligned_sample_times = [pd.Timestamp(item) for item in base_summary["aligned_sample_times"]]
    lims_path = discover_lims_path(PROJECT_DIR / "data", str(baseline_config["data"]["lims_glob"]))
    lims, _ = load_lims_data(lims_path)
    dcs, _dcs_audit, _alias_frame = load_identity_deduped_combined_dcs(
        PROJECT_DIR / str(baseline_config["data"]["dcs_main_path"]),
        PROJECT_DIR / str(baseline_config["data"]["dcs_supplemental_path"]),
        baseline_config,
    )

    cleaned_frames: dict[str, pd.DataFrame] = {}
    feature_cols_by_lag: dict[str, list[str]] = {}
    for lag in compare_lags:
        lag_name = lag_candidate_name("L", lag)
        feature_rows = build_shifted_feature_rows(
            lims,
            dcs,
            lookback_minutes=int(baseline_config["features"]["lookback_minutes"]),
            lag_minutes=lag,
            stats=list(baseline_config["features"]["window_statistics"]),
        )
        feature_rows = assign_labels(feature_rows, baseline_config)
        feature_rows = align_to_sample_times(feature_rows, aligned_sample_times)
        feature_rows, _preclean_summary = preclean_features(feature_rows, baseline_config)
        feature_rows = align_to_sample_times(feature_rows, aligned_sample_times)
        cleaned_frames[lag_name] = feature_rows
        feature_cols_by_lag[lag_name] = feature_columns(feature_rows)

    base_name = lag_names[0]
    outer_tscv = TimeSeriesSplit(n_splits=int(baseline_config["validation"]["n_splits"]))
    split_indices = list(outer_tscv.split(cleaned_frames[base_name]))

    per_sensor_rows: list[dict[str, object]] = []
    aggregate: dict[str, list[dict[str, object]]] = defaultdict(list)

    for fold_index, (train_index, test_index) in enumerate(split_indices, start=1):
        selection_row = selection_frame.loc[selection_frame["fold"] == fold_index].iloc[0]
        selected_sensors = list(selection_row["selected_sensors"])
        weight_candidate = {
            "candidate_name": str(selection_row["chosen_candidate_name"]),
            "boundary_weight": float(selection_row["chosen_boundary_weight"]),
            "warning_weight": float(selection_row["chosen_warning_weight"]),
        }

        baseline_metrics_by_lag: dict[str, dict[str, object]] = {}
        for lag_name in lag_names:
            frame = cleaned_frames[lag_name]
            train_frame = frame.iloc[train_index].copy()
            test_frame = frame.iloc[test_index].copy()
            baseline_metrics_by_lag[lag_name] = evaluate_sensor_subset(
                train_frame,
                test_frame,
                sensors=selected_sensors,
                feature_cols=feature_cols_by_lag[lag_name],
                thresholds=thresholds,
                baseline_config=baseline_config,
                weight_candidate=weight_candidate,
            )

        for sensor in selected_sensors:
            row: dict[str, object] = {
                "fold": int(fold_index),
                "sensor": sensor,
                "sensor_display_name": sensor_display_name(sensor, point_metadata),
                "selection_weight_candidate": str(selection_row["chosen_candidate_name"]),
            }
            ablated_sensors = [item for item in selected_sensors if item != sensor]

            for lag_name in lag_names:
                frame = cleaned_frames[lag_name]
                train_frame = frame.iloc[train_index].copy()
                test_frame = frame.iloc[test_index].copy()
                baseline_metrics = baseline_metrics_by_lag[lag_name]
                ablated_metrics = evaluate_sensor_subset(
                    train_frame,
                    test_frame,
                    sensors=ablated_sensors,
                    feature_cols=feature_cols_by_lag[lag_name],
                    thresholds=thresholds,
                    baseline_config=baseline_config,
                    weight_candidate=weight_candidate,
                )
                row[f"{lag_name}_baseline_macro_f1"] = baseline_metrics["macro_f1"]
                row[f"{lag_name}_delta_macro_f1"] = ablated_metrics["macro_f1"] - baseline_metrics["macro_f1"]
                row[f"{lag_name}_delta_balanced_accuracy"] = ablated_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
                row[f"{lag_name}_delta_warning_AP"] = ablated_metrics["warning_AP"] - baseline_metrics["warning_AP"]
                row[f"{lag_name}_delta_unacceptable_AP"] = ablated_metrics["unacceptable_AP"] - baseline_metrics["unacceptable_AP"]
                row[f"{lag_name}_delta_boundary_high_conf_non_warning"] = (
                    ablated_metrics["boundary_high_conf_non_warning_rate"] - baseline_metrics["boundary_high_conf_non_warning_rate"]
                )

            row["macro_f1_importance_gap_L60_minus_L180"] = row["L180_delta_macro_f1"] - row["L60_delta_macro_f1"]
            row["balanced_accuracy_importance_gap_L60_minus_L180"] = (
                row["L180_delta_balanced_accuracy"] - row["L60_delta_balanced_accuracy"]
            )
            row["warning_AP_importance_gap_L180_minus_L60"] = row["L60_delta_warning_AP"] - row["L180_delta_warning_AP"]
            row["boundary_guard_importance_gap_L180_minus_L60"] = (
                row["L180_delta_boundary_high_conf_non_warning"] - row["L60_delta_boundary_high_conf_non_warning"]
            )
            per_sensor_rows.append(row)
            aggregate[sensor].append(row)

    per_sensor_df = pd.DataFrame(per_sensor_rows).sort_values(["sensor", "fold"]).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for sensor, rows in aggregate.items():
        frame = pd.DataFrame(rows)
        summary_rows.append(
            {
                "sensor": sensor,
                "sensor_display_name": sensor_display_name(sensor, point_metadata),
                "selection_count": int(len(frame)),
                "mean_L60_delta_macro_f1": safe_mean(frame, "L60_delta_macro_f1"),
                "mean_L180_delta_macro_f1": safe_mean(frame, "L180_delta_macro_f1"),
                "mean_L60_delta_balanced_accuracy": safe_mean(frame, "L60_delta_balanced_accuracy"),
                "mean_L180_delta_balanced_accuracy": safe_mean(frame, "L180_delta_balanced_accuracy"),
                "mean_L60_delta_warning_AP": safe_mean(frame, "L60_delta_warning_AP"),
                "mean_L180_delta_warning_AP": safe_mean(frame, "L180_delta_warning_AP"),
                "mean_L60_delta_unacceptable_AP": safe_mean(frame, "L60_delta_unacceptable_AP"),
                "mean_L180_delta_unacceptable_AP": safe_mean(frame, "L180_delta_unacceptable_AP"),
                "mean_L60_delta_boundary_high_conf_non_warning": safe_mean(frame, "L60_delta_boundary_high_conf_non_warning"),
                "mean_L180_delta_boundary_high_conf_non_warning": safe_mean(frame, "L180_delta_boundary_high_conf_non_warning"),
                "macro_f1_importance_gap_L60_minus_L180": safe_mean(frame, "macro_f1_importance_gap_L60_minus_L180"),
                "balanced_accuracy_importance_gap_L60_minus_L180": safe_mean(frame, "balanced_accuracy_importance_gap_L60_minus_L180"),
                "warning_AP_importance_gap_L180_minus_L60": safe_mean(frame, "warning_AP_importance_gap_L180_minus_L60"),
                "boundary_guard_importance_gap_L180_minus_L60": safe_mean(frame, "boundary_guard_importance_gap_L180_minus_L60"),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("sensor").reset_index(drop=True)

    l60_support_candidates = summary_df[
        (summary_df["mean_L60_delta_macro_f1"] < 0)
        | (summary_df["mean_L60_delta_balanced_accuracy"] < 0)
    ].copy()
    l180_boundary_candidates = summary_df[
        (summary_df["mean_L180_delta_warning_AP"] < 0)
        | (summary_df["mean_L180_delta_boundary_high_conf_non_warning"] > 0)
    ].copy()

    l60_overall_support = l60_support_candidates.sort_values(
        ["macro_f1_importance_gap_L60_minus_L180", "balanced_accuracy_importance_gap_L60_minus_L180"],
        ascending=[False, False],
    ).head(10)
    l180_boundary_support = l180_boundary_candidates.sort_values(
        ["warning_AP_importance_gap_L180_minus_L60", "boundary_guard_importance_gap_L180_minus_L60"],
        ascending=[False, False],
    ).head(10)

    results_csv = artifacts_dir / "global_lag_sensor_preference_per_fold.csv"
    summary_csv = artifacts_dir / "global_lag_sensor_preference_summary.csv"
    summary_json = artifacts_dir / "global_lag_sensor_preference_summary.json"
    report_md = reports_dir / "global_lag_sensor_preference_summary.md"

    per_sensor_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    summary_payload = {
        "experiment_name": config["experiment_name"],
        "base_global_lag_summary_json": str(base_summary_path),
        "compare_lags": compare_lags,
        "top_l60_overall_support_sensors": l60_overall_support.to_dict(orient="records"),
        "top_l180_boundary_support_sensors": l180_boundary_support.to_dict(orient="records"),
        "l60_support_candidate_count": int(len(l60_support_candidates)),
        "l180_boundary_candidate_count": int(len(l180_boundary_candidates)),
        "artifacts": {
            "results_csv": str(results_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Global Lag Sensor Preference Summary",
        "",
        "- Base comparison: L60 vs L180 under the frozen lag cleanroom",
        "- Control: same L0-selected sensors and same per-fold weight candidate reused across both lag arms",
        "- Interpretation rule:",
        "  - more negative ablation delta means the sensor is more helpful to that lag arm",
        "  - larger `macro_f1_importance_gap_L60_minus_L180` means the sensor supports L60 overall performance more",
        "  - larger `warning_AP_importance_gap_L180_minus_L60` or `boundary_guard_importance_gap_L180_minus_L60` means the sensor supports L180 boundary caution more",
        "",
        "## Top Sensors Supporting L60 Overall",
        "",
    ]
    for row in l60_overall_support.itertuples(index=False):
        label = row.sensor_display_name if pd.notna(row.sensor_display_name) and row.sensor_display_name else row.sensor
        lines.append(
            f"- {row.sensor} ({label}): "
            f"macro_gap={row.macro_f1_importance_gap_L60_minus_L180:.4f}, "
            f"balanced_gap={row.balanced_accuracy_importance_gap_L60_minus_L180:.4f}"
        )
    lines.extend(
        [
            "",
            "## Top Sensors Supporting L180 Boundary Caution",
            "",
        ]
    )
    for row in l180_boundary_support.itertuples(index=False):
        label = row.sensor_display_name if pd.notna(row.sensor_display_name) and row.sensor_display_name else row.sensor
        lines.append(
            f"- {row.sensor} ({label}): "
            f"warning_gap={row.warning_AP_importance_gap_L180_minus_L60:.4f}, "
            f"boundary_guard_gap={row.boundary_guard_importance_gap_L180_minus_L60:.4f}"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
