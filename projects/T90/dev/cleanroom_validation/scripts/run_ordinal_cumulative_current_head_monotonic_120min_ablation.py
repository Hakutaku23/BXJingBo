from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from run_ordinal_cumulative_current_head import (
    apply_monotonic_correction,
    assign_labels,
    binary_ap,
    boundary_overconfidence_stats,
    build_feature_rows,
    business_probabilities_from_intervals,
    clip_and_renormalize,
    cumulative_to_interval_probabilities,
    discover_lims_path,
    feature_columns,
    fit_threshold_model,
    format_threshold,
    load_combined_dcs,
    load_config as load_ewma_config,
    load_lims_data,
    predict_threshold_probability,
    probability_argmax,
)


THIS_DIR = Path(__file__).resolve().parent
CLEANROOM_DIR = THIS_DIR.parent
PROJECT_DIR = CLEANROOM_DIR.parents[1]
DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "ordinal_cumulative_current_head_monotonic_120min_ablation.yaml"
BASELINE_CONFIG_PATH = CLEANROOM_DIR / "configs" / "ordinal_cumulative_current_head_monotonic_120min.yaml"
BUSINESS_LABELS = ["acceptable", "warning", "unacceptable"]


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_point_metadata(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_excel(path)
    if "变量" not in frame.columns or "文件名" not in frame.columns:
        return {}
    mapping: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        raw_name = str(getattr(row, "文件名", "")).strip()
        display = str(getattr(row, "变量", "")).strip()
        if not raw_name or not display:
            continue
        normalized = raw_name.replace("B4-", "").replace("/", "_").replace(".", "_")
        normalized = normalized.replace("__", "_")
        mapping[normalized] = display
    return mapping


def sensor_display_name(sensor: str, metadata: dict[str, str]) -> str | None:
    key = sensor.replace("__", "_")
    direct = metadata.get(key)
    if direct:
        return direct
    simplified = key.replace("_PV_F_CV", "").replace("_PV_CV", "").replace("_S_PV_CV", "_S").replace("_PV", "")
    return metadata.get(simplified)


def evaluate_sensor_set(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    sensors: list[str],
    all_feature_columns: list[str],
    thresholds: list[float],
    config: dict[str, object],
) -> dict[str, object]:
    selected_sensor_set = set(sensors)
    selected_features = sorted([col for col in all_feature_columns if col.split("__", 1)[0] in selected_sensor_set])
    cumulative = pd.DataFrame(index=test_frame.index)
    for threshold in thresholds:
        key = format_threshold(threshold)
        target_key = f"target_lt_{key}"
        model = fit_threshold_model(train_frame, selected_features, target_key, config)
        cumulative[f"lt_{key}"] = predict_threshold_probability(model, test_frame[selected_features])

    if bool(config["models"]["apply_monotonic_correction"]):
        cumulative = apply_monotonic_correction(cumulative)
    business = business_probabilities_from_intervals(clip_and_renormalize(cumulative_to_interval_probabilities(cumulative)))
    pred = probability_argmax(business, BUSINESS_LABELS)
    result = {
        "macro_f1": float(f1_score(test_frame["business_label"], pred, labels=BUSINESS_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(test_frame["business_label"], pred)),
        "core_qualified_average_precision": binary_ap(test_frame["business_label"].eq("acceptable").astype(int), business["acceptable"]),
        "boundary_warning_average_precision": binary_ap(test_frame["business_label"].eq("warning").astype(int), business["warning"]),
        "clearly_unacceptable_average_precision": binary_ap(test_frame["business_label"].eq("unacceptable").astype(int), business["unacceptable"]),
    }
    boundary_frame = pd.DataFrame(
        {
            "boundary_any_flag": test_frame["boundary_any_flag"],
            "pred": pred,
            "acceptable": business["acceptable"],
            "unacceptable": business["unacceptable"],
        },
        index=test_frame.index,
    )
    result["boundary_overconfidence"] = boundary_overconfidence_stats(
        boundary_frame,
        pred_col="pred",
        acceptable_prob_col="acceptable",
        unacceptable_prob_col="unacceptable",
    )
    result["selected_sensor_count"] = int(len(sensors))
    return result


def positive_or_nan(value: object) -> float | None:
    number = float(value)
    return None if math.isnan(number) else number


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate one selected DCS sensor at a time from the 120min monotonic baseline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    baseline_config = load_ewma_config(BASELINE_CONFIG_PATH)
    base_summary_path = CLEANROOM_DIR / str(config["base_summary_json"])
    baseline_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))

    outputs = config["outputs"]
    artifacts_dir = CLEANROOM_DIR / str(outputs["artifacts_dir"])
    reports_dir = CLEANROOM_DIR / str(outputs["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    point_metadata = read_point_metadata(PROJECT_DIR / str(config["point_metadata_path"]))
    lims_path = discover_lims_path(PROJECT_DIR / "data", str(baseline_config["data"]["lims_glob"]))
    dcs, _ = load_combined_dcs(
        PROJECT_DIR / str(baseline_config["data"]["dcs_main_path"]),
        PROJECT_DIR / str(baseline_config["data"]["dcs_supplemental_path"]),
    )
    lims, _ = load_lims_data(lims_path)
    feature_rows = build_feature_rows(
        lims,
        dcs,
        lookback_minutes=int(baseline_config["baseline"]["lookback_minutes"]) if "baseline" in baseline_config else int(baseline_config["features"]["lookback_minutes"]),
        stats=list(baseline_config["baseline"]["window_statistics"]) if "baseline" in baseline_config else list(baseline_config["features"]["window_statistics"]),
    )
    feature_rows = assign_labels(feature_rows, baseline_config)
    all_feature_columns = feature_columns(feature_rows)
    thresholds = [float(item) for item in baseline_config["labels"]["cumulative_thresholds"]]

    n_splits = int(baseline_summary["data_summary"]["n_splits_used"])
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_summaries = {int(row["fold"]): row for row in baseline_summary["metrics_summary"]["per_fold"]}

    per_sensor_rows: list[dict[str, object]] = []
    aggregate: dict[str, list[dict[str, object]]] = defaultdict(list)

    for fold_index, (train_index, test_index) in enumerate(tscv.split(feature_rows), start=1):
        train_frame = feature_rows.iloc[train_index].copy()
        test_frame = feature_rows.iloc[test_index].copy()
        baseline_fold = fold_summaries[fold_index]
        selected_sensors = list(baseline_fold["selected_sensors"])
        baseline_metrics = evaluate_sensor_set(
            train_frame,
            test_frame,
            sensors=selected_sensors,
            all_feature_columns=all_feature_columns,
            thresholds=thresholds,
            config=baseline_config,
        )
        for sensor in selected_sensors:
            ablated_sensors = [item for item in selected_sensors if item != sensor]
            ablated_metrics = evaluate_sensor_set(
                train_frame,
                test_frame,
                sensors=ablated_sensors,
                all_feature_columns=all_feature_columns,
                thresholds=thresholds,
                config=baseline_config,
            )
            row = {
                "fold": int(fold_index),
                "sensor": sensor,
                "sensor_display_name": sensor_display_name(sensor, point_metadata),
                "baseline_macro_f1": baseline_metrics["macro_f1"],
                "ablated_macro_f1": ablated_metrics["macro_f1"],
                "delta_macro_f1": ablated_metrics["macro_f1"] - baseline_metrics["macro_f1"],
                "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
                "ablated_balanced_accuracy": ablated_metrics["balanced_accuracy"],
                "delta_balanced_accuracy": ablated_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
                "baseline_core_AP": positive_or_nan(baseline_metrics["core_qualified_average_precision"]),
                "ablated_core_AP": positive_or_nan(ablated_metrics["core_qualified_average_precision"]),
                "delta_core_AP": positive_or_nan(ablated_metrics["core_qualified_average_precision"] - baseline_metrics["core_qualified_average_precision"]),
                "baseline_warning_AP": positive_or_nan(baseline_metrics["boundary_warning_average_precision"]),
                "ablated_warning_AP": positive_or_nan(ablated_metrics["boundary_warning_average_precision"]),
                "delta_warning_AP": positive_or_nan(ablated_metrics["boundary_warning_average_precision"] - baseline_metrics["boundary_warning_average_precision"]),
                "baseline_boundary_high_conf_non_warning": baseline_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                "ablated_boundary_high_conf_non_warning": ablated_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                "delta_boundary_high_conf_non_warning_reduction": baseline_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"] - ablated_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
            per_sensor_rows.append(row)
            aggregate[sensor].append(row)

    summary_rows: list[dict[str, object]] = []
    for sensor, rows in aggregate.items():
        frame = pd.DataFrame(rows)
        positive_macro = int((frame["delta_macro_f1"] > 0).sum())
        positive_bal = int((frame["delta_balanced_accuracy"] > 0).sum())
        positive_boundary = int((frame["delta_boundary_high_conf_non_warning_reduction"] > 0).sum())
        summary_rows.append(
            {
                "sensor": sensor,
                "sensor_display_name": sensor_display_name(sensor, point_metadata),
                "selection_count": int(len(frame)),
                "mean_delta_macro_f1": float(frame["delta_macro_f1"].mean()),
                "mean_delta_balanced_accuracy": float(frame["delta_balanced_accuracy"].mean()),
                "mean_delta_core_AP": float(frame["delta_core_AP"].dropna().mean()) if frame["delta_core_AP"].notna().any() else math.nan,
                "mean_delta_warning_AP": float(frame["delta_warning_AP"].dropna().mean()) if frame["delta_warning_AP"].notna().any() else math.nan,
                "mean_boundary_high_conf_reduction": float(frame["delta_boundary_high_conf_non_warning_reduction"].mean()),
                "folds_macro_improved": positive_macro,
                "folds_balanced_accuracy_improved": positive_bal,
                "folds_boundary_reduced": positive_boundary,
                "suspected_negative_gain": bool(
                    len(frame) >= 2
                    and frame["delta_macro_f1"].mean() > 0
                    and (
                        frame["delta_balanced_accuracy"].mean() > 0
                        or frame["delta_boundary_high_conf_non_warning_reduction"].mean() > 0
                    )
                ),
            }
        )

    per_sensor_df = pd.DataFrame(per_sensor_rows).sort_values(["sensor", "fold"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["suspected_negative_gain", "mean_delta_macro_f1", "mean_delta_balanced_accuracy", "mean_boundary_high_conf_reduction"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    suspected = summary_df[summary_df["suspected_negative_gain"]].copy()
    stable_selected_counter = Counter()
    for row in baseline_summary["metrics_summary"]["per_fold"]:
        stable_selected_counter.update(row["selected_sensors"])

    results_csv = artifacts_dir / "sensor_ablation_per_fold.csv"
    summary_csv = artifacts_dir / "sensor_ablation_summary.csv"
    summary_json = artifacts_dir / "sensor_ablation_summary.json"
    report_md = reports_dir / "sensor_ablation_summary.md"

    json_summary = {
        "experiment_name": config["experiment_name"],
        "baseline_reference": str(base_summary_path),
        "selected_sensor_count_per_fold": 40,
        "selected_sensor_union_count": int(len(stable_selected_counter)),
        "suspected_negative_gain_sensor_count": int(len(suspected)),
        "suspected_negative_gain_sensors": suspected.to_dict(orient="records"),
    }

    per_sensor_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(json_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Sensor Ablation Summary",
        "",
        "- Baseline: 120min monotonic ordinal/cumulative topk=40",
        "- Protocol: remove one selected sensor at a time inside the already-selected fold-local sensor set",
        f"- Selected sensors per fold: 40",
        f"- Selected sensor union across folds: {len(stable_selected_counter)}",
        f"- Suspected negative-gain sensors: {len(suspected)}",
        "",
        "## Suspected Negative-Gain Sensors",
        "",
    ]
    if suspected.empty:
        lines.append("- None under the current ablation rule.")
    else:
        for row in suspected.itertuples(index=False):
            label = row.sensor_display_name if pd.notna(row.sensor_display_name) and row.sensor_display_name else row.sensor
            lines.append(
                f"- {row.sensor} ({label}): selection_count={row.selection_count}, "
                f"mean_delta_macro_f1={row.mean_delta_macro_f1:.4f}, "
                f"mean_delta_balanced_accuracy={row.mean_delta_balanced_accuracy:.4f}, "
                f"mean_boundary_high_conf_reduction={row.mean_boundary_high_conf_reduction:.4f}"
            )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
