from __future__ import annotations

import argparse
import json
import math
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
    calibration_error,
    clip_and_renormalize,
    cumulative_to_interval_probabilities,
    discover_lims_path,
    feature_columns,
    fit_threshold_model,
    format_threshold,
    load_dcs_data,
    load_lims_data,
    monotonicity_summary,
    preclean_features,
    predict_threshold_probability,
    probability_argmax,
    selected_feature_columns,
    select_topk_sensors,
)


THIS_DIR = Path(__file__).resolve().parent
CLEANROOM_DIR = THIS_DIR.parent
PROJECT_DIR = CLEANROOM_DIR.parents[1]
DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "ordinal_cumulative_paper_faithful_ewma_current_head.yaml"
BUSINESS_LABELS = ["acceptable", "warning", "unacceptable"]


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_combined_dcs(main_path: Path, supplemental_path: Path | None) -> tuple[pd.DataFrame, dict[str, object]]:
    main = load_dcs_data(main_path)
    audit = {
        "main_rows": int(len(main)),
        "main_sensor_count": int(len([c for c in main.columns if c != "time"])),
        "supplemental_used": False,
        "supplemental_rows": 0,
        "supplemental_sensor_count": 0,
        "shared_time_count": 0,
    }
    if supplemental_path is None or not supplemental_path.exists():
        return main, audit

    supplemental = load_dcs_data(supplemental_path)
    merged = main.merge(supplemental, on="time", how="outer")
    audit.update(
        {
            "supplemental_used": True,
            "supplemental_rows": int(len(supplemental)),
            "supplemental_sensor_count": int(len([c for c in supplemental.columns if c != "time"])),
            "shared_time_count": int(len(set(main["time"]) & set(supplemental["time"]))),
            "combined_rows": int(len(merged)),
            "combined_sensor_count": int(len([c for c in merged.columns if c != "time"])),
        }
    )
    return merged.sort_values("time").reset_index(drop=True), audit


def recursive_ewma_last(series: pd.Series, lambda_value: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return math.nan
    return float(values.ewm(alpha=1.0 - lambda_value, adjust=False).mean().iloc[-1])


def build_recursive_ewma_rows(
    lims: pd.DataFrame,
    dcs: pd.DataFrame,
    *,
    sensors: list[str],
    tau_minutes: int,
    window_minutes: int,
    lambda_value: float,
    min_rows_per_window: int,
    min_valid_points_per_sensor: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    labeled = lims.dropna(subset=["sample_time", "t90"]).copy().sort_values("sample_time").reset_index(drop=True)
    dcs = dcs.sort_values("time").reset_index(drop=True)
    dcs_times = dcs["time"].to_numpy(dtype="datetime64[ns]")

    rows: list[dict[str, object]] = []
    dropped_window = 0
    sensor_missing_counter = 0
    sensor_total_counter = 0
    for record in labeled.itertuples(index=False):
        sample_time = pd.Timestamp(record.sample_time)
        t_end = sample_time - pd.Timedelta(minutes=tau_minutes)
        t_start = t_end - pd.Timedelta(minutes=window_minutes)
        left = np.searchsorted(dcs_times, t_start.to_datetime64(), side="left")
        right = np.searchsorted(dcs_times, t_end.to_datetime64(), side="right")
        window = dcs.iloc[left:right]
        if len(window) < min_rows_per_window:
            dropped_window += 1
            continue

        row = {"sample_time": sample_time, "t90": float(record.t90)}
        for sensor in sensors:
            sensor_total_counter += 1
            values = pd.to_numeric(window[sensor], errors="coerce").dropna()
            if len(values) < min_valid_points_per_sensor:
                row[f"{sensor}__ewma_recursive"] = math.nan
                sensor_missing_counter += 1
            else:
                row[f"{sensor}__ewma_recursive"] = recursive_ewma_last(values, lambda_value)
        rows.append(row)

    audit = {
        "tau_minutes": int(tau_minutes),
        "window_minutes": int(window_minutes),
        "lambda": float(lambda_value),
        "dropped_due_to_short_window": int(dropped_window),
        "sensor_missing_feature_ratio": float(sensor_missing_counter / sensor_total_counter) if sensor_total_counter else math.nan,
    }
    return pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True), audit


def evaluate_representation(
    frame: pd.DataFrame,
    *,
    baseline_feature_cols: list[str],
    ewma_feature_cols: list[str],
    config: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    n_splits = int(config["validation"]["n_splits"])
    thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = frame[
        ["sample_time", "t90", "business_label", "is_unacceptable", "boundary_any_flag"]
        + [column for column in frame.columns if column.startswith("target_lt_")]
    ].copy()
    for prefix in ["base", "ewma"]:
        for threshold in thresholds:
            results[f"{prefix}_cum_prob_lt_{format_threshold(threshold)}"] = np.nan
        for label in BUSINESS_LABELS:
            results[f"{prefix}_business_prob_{label}"] = np.nan
    per_fold: list[dict[str, object]] = []

    for fold_index, (train_index, test_index) in enumerate(tscv.split(frame), start=1):
        train_frame = frame.iloc[train_index].copy()
        test_frame = frame.iloc[test_index].copy()
        selected_sensors, _ = select_topk_sensors(train_frame, baseline_feature_cols, topk=int(config["screening"]["topk_sensors"]))
        base_features = selected_feature_columns(baseline_feature_cols, selected_sensors)
        ewma_features = selected_feature_columns(ewma_feature_cols, selected_sensors)

        base_cumulative = pd.DataFrame(index=test_frame.index)
        ewma_cumulative = pd.DataFrame(index=test_frame.index)
        for threshold in thresholds:
            key = format_threshold(threshold)
            target_key = f"target_lt_{key}"

            base_model = fit_threshold_model(train_frame, base_features, target_key, config)
            base_cumulative[f"lt_{key}"] = predict_threshold_probability(base_model, test_frame[base_features])

            ewma_model = fit_threshold_model(train_frame, ewma_features, target_key, config)
            ewma_cumulative[f"lt_{key}"] = predict_threshold_probability(ewma_model, test_frame[ewma_features])

        if bool(config["models"]["apply_monotonic_correction"]):
            base_cumulative = apply_monotonic_correction(base_cumulative)
            ewma_cumulative = apply_monotonic_correction(ewma_cumulative)

        base_business = business_probabilities_from_intervals(clip_and_renormalize(cumulative_to_interval_probabilities(base_cumulative)))
        ewma_business = business_probabilities_from_intervals(clip_and_renormalize(cumulative_to_interval_probabilities(ewma_cumulative)))

        for threshold in thresholds:
            key = format_threshold(threshold)
            results.loc[test_frame.index, f"base_cum_prob_lt_{key}"] = base_cumulative[f"lt_{key}"]
            results.loc[test_frame.index, f"ewma_cum_prob_lt_{key}"] = ewma_cumulative[f"lt_{key}"]
        for label in BUSINESS_LABELS:
            results.loc[test_frame.index, f"base_business_prob_{label}"] = base_business[label]
            results.loc[test_frame.index, f"ewma_business_prob_{label}"] = ewma_business[label]

        fold_view = pd.DataFrame(index=test_frame.index)
        fold_view["business_label"] = test_frame["business_label"]
        fold_view["base_pred"] = probability_argmax(base_business, BUSINESS_LABELS)
        fold_view["ewma_pred"] = probability_argmax(ewma_business, BUSINESS_LABELS)
        per_fold.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "selected_sensor_count": int(len(selected_sensors)),
                "base_macro_f1": float(f1_score(fold_view["business_label"], fold_view["base_pred"], labels=BUSINESS_LABELS, average="macro", zero_division=0)),
                "ewma_macro_f1": float(f1_score(fold_view["business_label"], fold_view["ewma_pred"], labels=BUSINESS_LABELS, average="macro", zero_division=0)),
            }
        )

    for prefix in ["base", "ewma"]:
        prob_frame = results[[f"{prefix}_business_prob_{label}" for label in BUSINESS_LABELS]].rename(
            columns={f"{prefix}_business_prob_{label}": label for label in BUSINESS_LABELS}
        )
        results[f"{prefix}_business_pred"] = probability_argmax(prob_frame, BUSINESS_LABELS)

    base_metrics = {
        "macro_f1": float(f1_score(results["business_label"], results["base_business_pred"], labels=BUSINESS_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(results["business_label"], results["base_business_pred"])),
        "core_qualified_average_precision": binary_ap(results["business_label"].eq("acceptable").astype(int), results["base_business_prob_acceptable"]),
        "boundary_warning_average_precision": binary_ap(results["business_label"].eq("warning").astype(int), results["base_business_prob_warning"]),
        "clearly_unacceptable_average_precision": binary_ap(results["business_label"].eq("unacceptable").astype(int), results["base_business_prob_unacceptable"]),
        "boundary_overconfidence": boundary_overconfidence_stats(
            results.rename(columns={"base_business_pred": "pred", "base_business_prob_acceptable": "acceptable", "base_business_prob_unacceptable": "unacceptable"}),
            pred_col="pred",
            acceptable_prob_col="acceptable",
            unacceptable_prob_col="unacceptable",
        ),
    }
    ewma_metrics = {
        "macro_f1": float(f1_score(results["business_label"], results["ewma_business_pred"], labels=BUSINESS_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(results["business_label"], results["ewma_business_pred"])),
        "core_qualified_average_precision": binary_ap(results["business_label"].eq("acceptable").astype(int), results["ewma_business_prob_acceptable"]),
        "boundary_warning_average_precision": binary_ap(results["business_label"].eq("warning").astype(int), results["ewma_business_prob_warning"]),
        "clearly_unacceptable_average_precision": binary_ap(results["business_label"].eq("unacceptable").astype(int), results["ewma_business_prob_unacceptable"]),
        "boundary_overconfidence": boundary_overconfidence_stats(
            results.rename(columns={"ewma_business_pred": "pred", "ewma_business_prob_acceptable": "acceptable", "ewma_business_prob_unacceptable": "unacceptable"}),
            pred_col="pred",
            acceptable_prob_col="acceptable",
            unacceptable_prob_col="unacceptable",
        ),
        "monotonicity": monotonicity_summary(
            results.rename(columns={f"ewma_cum_prob_lt_{format_threshold(t)}": f"cum_prob_lt_{format_threshold(t)}" for t in thresholds}),
            thresholds,
        ),
    }
    return {"baseline_simple": base_metrics, "ewma_treatment": ewma_metrics, "per_fold": per_fold}, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-faithful EWMA against the ordinal/cumulative simple baseline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    outputs = config["outputs"]
    artifacts_dir = CLEANROOM_DIR / str(outputs["artifacts_dir"])
    reports_dir = CLEANROOM_DIR / str(outputs["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    main_dcs_path = PROJECT_DIR / str(config["data"]["dcs_main_path"])
    supplemental_path = PROJECT_DIR / str(config["data"]["dcs_supplemental_path"])
    lims_path = discover_lims_path(PROJECT_DIR / "data", str(config["data"]["lims_glob"]))

    lims, _ = load_lims_data(lims_path)
    dcs, dcs_audit = load_combined_dcs(main_dcs_path, supplemental_path)

    baseline_rows = build_feature_rows(
        lims,
        dcs,
        lookback_minutes=int(config["baseline"]["lookback_minutes"]),
        stats=list(config["baseline"]["window_statistics"]),
    )
    baseline_rows = assign_labels(baseline_rows, config)
    baseline_rows, preclean_summary = preclean_features(baseline_rows, config)
    baseline_feature_cols = feature_columns(baseline_rows)
    candidate_sensors = sorted({col.split("__", 1)[0] for col in baseline_feature_cols})

    result_rows: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_results: pd.DataFrame | None = None

    for tau in config["ewma"]["tau_grid"]:
        for window_minutes in config["ewma"]["window_grid"]:
            for lambda_value in config["ewma"]["lambda_grid"]:
                ewma_rows, ewma_audit = build_recursive_ewma_rows(
                    lims,
                    dcs,
                    sensors=candidate_sensors,
                    tau_minutes=int(tau),
                    window_minutes=int(window_minutes),
                    lambda_value=float(lambda_value),
                    min_rows_per_window=int(config["ewma"]["min_rows_per_window"]),
                    min_valid_points_per_sensor=int(config["ewma"]["min_valid_points_per_sensor"]),
                )
                ewma_rows = assign_labels(ewma_rows, config)
                ewma_feature_cols = [col for col in ewma_rows.columns if col.endswith("__ewma_recursive")]

                common = baseline_rows.merge(
                    ewma_rows[["sample_time", *ewma_feature_cols]],
                    on="sample_time",
                    how="inner",
                ).sort_values("sample_time").reset_index(drop=True)
                if len(common) < 500:
                    continue

                metrics_summary, detailed_results = evaluate_representation(
                    common,
                    baseline_feature_cols=baseline_feature_cols,
                    ewma_feature_cols=ewma_feature_cols,
                    config=config,
                )
                row = {
                    "tau_minutes": int(tau),
                    "window_minutes": int(window_minutes),
                    "lambda": float(lambda_value),
                    "common_samples": int(len(common)),
                    "baseline_macro_f1": metrics_summary["baseline_simple"]["macro_f1"],
                    "ewma_macro_f1": metrics_summary["ewma_treatment"]["macro_f1"],
                    "baseline_balanced_accuracy": metrics_summary["baseline_simple"]["balanced_accuracy"],
                    "ewma_balanced_accuracy": metrics_summary["ewma_treatment"]["balanced_accuracy"],
                    "baseline_core_AP": metrics_summary["baseline_simple"]["core_qualified_average_precision"],
                    "ewma_core_AP": metrics_summary["ewma_treatment"]["core_qualified_average_precision"],
                    "baseline_warning_AP": metrics_summary["baseline_simple"]["boundary_warning_average_precision"],
                    "ewma_warning_AP": metrics_summary["ewma_treatment"]["boundary_warning_average_precision"],
                    "baseline_boundary_high_conf_non_warning": metrics_summary["baseline_simple"]["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                    "ewma_boundary_high_conf_non_warning": metrics_summary["ewma_treatment"]["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                    "ewma_any_violation_rate": metrics_summary["ewma_treatment"]["monotonicity"]["any_violation_rate"],
                }
                result_rows.append(row)

                if best_summary is None or (
                    row["ewma_macro_f1"],
                    row["ewma_balanced_accuracy"],
                    row["ewma_warning_AP"],
                ) > (
                    best_summary["candidate"]["ewma_macro_f1"],
                    best_summary["candidate"]["ewma_balanced_accuracy"],
                    best_summary["candidate"]["ewma_warning_AP"],
                ):
                    best_summary = {
                        "candidate": row,
                        "metrics_summary": metrics_summary,
                        "ewma_audit": ewma_audit,
                    }
                    best_results = detailed_results.copy()

    if best_summary is None or best_results is None:
        raise ValueError("No valid EWMA combination was scored.")

    results_frame = pd.DataFrame(result_rows).sort_values(
        ["ewma_macro_f1", "ewma_balanced_accuracy", "ewma_warning_AP"],
        ascending=[False, False, False],
    )
    results_csv = artifacts_dir / "ordinal_cumulative_ewma_results.csv"
    summary_json = artifacts_dir / "ordinal_cumulative_ewma_summary.json"
    best_rows_csv = artifacts_dir / "ordinal_cumulative_ewma_best_feature_rows.csv"
    audit_md = reports_dir / "ordinal_cumulative_ewma_audit.md"

    summary = {
        "experiment_name": config["experiment_name"],
        "data_sources": {
            "dcs_main_path": str(main_dcs_path),
            "dcs_supplemental_path": str(supplemental_path),
            "lims_path": str(lims_path),
        },
        "dcs_audit": dcs_audit,
        "preclean_summary": preclean_summary,
        "baseline_reference": {
            "lookback_minutes": int(config["baseline"]["lookback_minutes"]),
            "topk_sensors": int(config["screening"]["topk_sensors"]),
        },
        "best_ewma_combination": best_summary["candidate"],
        "best_ewma_metrics_summary": best_summary["metrics_summary"],
        "best_ewma_audit": best_summary["ewma_audit"],
        "artifacts": {
            "results_csv": str(results_csv),
            "summary_json": str(summary_json),
            "best_feature_rows_csv": str(best_rows_csv),
            "audit_md": str(audit_md),
        },
    }

    results_frame.to_csv(results_csv, index=False, encoding="utf-8-sig")
    best_results.to_csv(best_rows_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_lines = [
        "# Ordinal / Cumulative EWMA Audit",
        "",
        "- Task formulation: ordinal / cumulative current-head",
        "- Baseline representation: simple 120min causal window statistics",
        "- Treatment representation: paper-faithful recursive EWMA condensed sample",
        f"- DCS supplemental file used: {dcs_audit['supplemental_used']}",
        "- Supplemental DCS was merged as an extra source and later handled by feature de-duplication / shared fold screening.",
        "",
        "## Best EWMA Combination",
        "",
        json.dumps(best_summary["candidate"], ensure_ascii=False, indent=2),
    ]
    audit_md.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
