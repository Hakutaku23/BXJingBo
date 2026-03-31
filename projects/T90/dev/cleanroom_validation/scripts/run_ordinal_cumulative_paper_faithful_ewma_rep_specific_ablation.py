from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

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
    load_lims_data,
    preclean_features,
    predict_threshold_probability,
    probability_argmax,
)
from run_ordinal_cumulative_paper_faithful_ewma_current_head import build_recursive_ewma_rows, load_combined_dcs


THIS_DIR = Path(__file__).resolve().parent
CLEANROOM_DIR = THIS_DIR.parent
PROJECT_DIR = CLEANROOM_DIR.parents[1]
DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "ordinal_cumulative_paper_faithful_ewma_rep_specific_ablation.yaml"
SEARCH_CONFIG_PATH = CLEANROOM_DIR / "configs" / "ordinal_cumulative_paper_faithful_ewma_rep_specific_current_head.yaml"
BUSINESS_LABELS = ["acceptable", "warning", "unacceptable"]


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def discover_point_metadata_path(path: Path) -> Path | None:
    if path.exists():
        return path
    matches = sorted((PROJECT_DIR / "data").glob("*点位*.xlsx"))
    if matches:
        return matches[0]
    return None


def read_point_metadata(path: Path) -> dict[str, str]:
    candidate_path = discover_point_metadata_path(path)
    if candidate_path is None:
        return {}
    frame = pd.read_excel(candidate_path)
    column_lookup = {str(column).strip(): column for column in frame.columns}
    variable_col = column_lookup.get("变量")
    file_col = column_lookup.get("文件名")
    if variable_col is None or file_col is None:
        return {}
    mapping: dict[str, str] = {}
    for _, row in frame.iterrows():
        raw_name = str(row[file_col]).strip()
        display = str(row[variable_col]).strip()
        if not raw_name or not display or raw_name.lower() == "nan" or display.lower() == "nan":
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


def positive_or_nan(value: object) -> float | None:
    number = float(value)
    return None if math.isnan(number) else number


def evaluate_ewma_sensor_set(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    sensors: list[str],
    ewma_feature_cols: list[str],
    thresholds: list[float],
    config: dict[str, object],
) -> dict[str, object]:
    sensor_set = set(sensors)
    selected_features = sorted([col for col in ewma_feature_cols if col.split("__", 1)[0] in sensor_set])
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
    boundary_frame = pd.DataFrame(
        {
            "boundary_any_flag": test_frame["boundary_any_flag"],
            "pred": pred,
            "acceptable": business["acceptable"],
            "unacceptable": business["unacceptable"],
        },
        index=test_frame.index,
    )
    return {
        "macro_f1": float(f1_score(test_frame["business_label"], pred, labels=BUSINESS_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(test_frame["business_label"], pred)),
        "core_qualified_average_precision": binary_ap(test_frame["business_label"].eq("acceptable").astype(int), business["acceptable"]),
        "boundary_warning_average_precision": binary_ap(test_frame["business_label"].eq("warning").astype(int), business["warning"]),
        "clearly_unacceptable_average_precision": binary_ap(test_frame["business_label"].eq("unacceptable").astype(int), business["unacceptable"]),
        "boundary_overconfidence": boundary_overconfidence_stats(
            boundary_frame,
            pred_col="pred",
            acceptable_prob_col="acceptable",
            unacceptable_prob_col="unacceptable",
        ),
        "selected_sensor_count": int(len(sensors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate one EWMA-selected sensor at a time from the best representation-specific EWMA subset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    search_config = load_config(SEARCH_CONFIG_PATH)
    best_summary_path = CLEANROOM_DIR / str(config["best_summary_json"])
    best_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
    best_combo = best_summary["best_ewma_combination"]

    outputs = config["outputs"]
    artifacts_dir = CLEANROOM_DIR / str(outputs["artifacts_dir"])
    reports_dir = CLEANROOM_DIR / str(outputs["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    point_metadata = read_point_metadata(PROJECT_DIR / str(config["point_metadata_path"]))
    main_dcs_path = PROJECT_DIR / str(search_config["data"]["dcs_main_path"])
    supplemental_path = PROJECT_DIR / str(search_config["data"]["dcs_supplemental_path"])
    lims_path = discover_lims_path(PROJECT_DIR / "data", str(search_config["data"]["lims_glob"]))
    lims, _ = load_lims_data(lims_path)
    dcs, _ = load_combined_dcs(main_dcs_path, supplemental_path)

    baseline_rows = build_feature_rows(
        lims,
        dcs,
        lookback_minutes=int(search_config["baseline"]["lookback_minutes"]),
        stats=list(search_config["baseline"]["window_statistics"]),
    )
    baseline_rows = assign_labels(baseline_rows, search_config)
    baseline_rows, _ = preclean_features(baseline_rows, search_config)
    candidate_sensors = sorted({col.split("__", 1)[0] for col in feature_columns(baseline_rows)})

    ewma_rows, ewma_audit = build_recursive_ewma_rows(
        lims,
        dcs,
        sensors=candidate_sensors,
        tau_minutes=int(best_combo["tau_minutes"]),
        window_minutes=int(best_combo["window_minutes"]),
        lambda_value=float(best_combo["lambda"]),
        min_rows_per_window=int(search_config["ewma"]["min_rows_per_window"]),
        min_valid_points_per_sensor=int(search_config["ewma"]["min_valid_points_per_sensor"]),
    )
    ewma_rows = assign_labels(ewma_rows, search_config)
    ewma_feature_cols = [col for col in ewma_rows.columns if col.endswith("__ewma_recursive")]

    common = baseline_rows.merge(
        ewma_rows[["sample_time", *ewma_feature_cols]],
        on="sample_time",
        how="inner",
    ).sort_values("sample_time").reset_index(drop=True)
    thresholds = [float(item) for item in search_config["labels"]["cumulative_thresholds"]]

    tscv = TimeSeriesSplit(n_splits=int(search_config["validation"]["n_splits"]))
    fold_summaries = {int(row["fold"]): row for row in best_summary["best_ewma_metrics_summary"]["per_fold"]}

    per_sensor_rows: list[dict[str, object]] = []
    aggregate: dict[str, list[dict[str, object]]] = defaultdict(list)
    selected_counts: list[int] = []
    selected_counter = Counter()

    for fold_index, (train_index, test_index) in enumerate(tscv.split(common), start=1):
        train_frame = common.iloc[train_index].copy()
        test_frame = common.iloc[test_index].copy()
        selected_sensors = list(fold_summaries[fold_index]["ewma_selected_sensors"])
        selected_counts.append(len(selected_sensors))
        selected_counter.update(selected_sensors)

        baseline_metrics = evaluate_ewma_sensor_set(
            train_frame,
            test_frame,
            sensors=selected_sensors,
            ewma_feature_cols=ewma_feature_cols,
            thresholds=thresholds,
            config=search_config,
        )
        for sensor in selected_sensors:
            ablated_sensors = [item for item in selected_sensors if item != sensor]
            ablated_metrics = evaluate_ewma_sensor_set(
                train_frame,
                test_frame,
                sensors=ablated_sensors,
                ewma_feature_cols=ewma_feature_cols,
                thresholds=thresholds,
                config=search_config,
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
                "delta_boundary_high_conf_non_warning_reduction": (
                    baseline_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"]
                    - ablated_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"]
                ),
            }
            per_sensor_rows.append(row)
            aggregate[sensor].append(row)

    summary_rows: list[dict[str, object]] = []
    for sensor, rows in aggregate.items():
        frame = pd.DataFrame(rows)
        positive_macro = int((frame["delta_macro_f1"] > 0).sum())
        positive_balanced = int((frame["delta_balanced_accuracy"] > 0).sum())
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
                "folds_balanced_accuracy_improved": positive_balanced,
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

    results_csv = artifacts_dir / "ewma_sensor_ablation_per_fold.csv"
    summary_csv = artifacts_dir / "ewma_sensor_ablation_summary.csv"
    summary_json = artifacts_dir / "ewma_sensor_ablation_summary.json"
    report_md = reports_dir / "ewma_sensor_ablation_summary.md"

    json_summary = {
        "experiment_name": config["experiment_name"],
        "best_summary_reference": str(best_summary_path),
        "best_ewma_combination": best_combo,
        "ewma_audit": ewma_audit,
        "selected_sensor_count_per_fold": selected_counts,
        "selected_sensor_union_count": int(len(selected_counter)),
        "suspected_negative_gain_sensor_count": int(len(suspected)),
        "suspected_negative_gain_sensors": suspected.to_dict(orient="records"),
    }

    per_sensor_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(json_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# EWMA Sensor Ablation Summary",
        "",
        "- Baseline: representation-specific EWMA best combination",
        "- Protocol: remove one EWMA-selected sensor at a time inside each fold-local EWMA subset",
        f"- Best combination: tau={best_combo['tau_minutes']}, W={best_combo['window_minutes']}, lambda={best_combo['lambda']}",
        f"- Selected sensors per fold: {selected_counts}",
        f"- Selected sensor union across folds: {len(selected_counter)}",
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
