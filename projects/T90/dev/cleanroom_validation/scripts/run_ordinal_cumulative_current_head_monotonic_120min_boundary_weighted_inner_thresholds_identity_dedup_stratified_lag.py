from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from run_ordinal_cumulative_current_head import (
    assign_labels,
    discover_lims_path,
    feature_columns,
    load_lims_data,
    preclean_features,
    probability_argmax,
    selected_feature_columns,
    sensor_name,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup import (
    BUSINESS_LABELS,
    CLEANROOM_DIR,
    PROJECT_DIR,
    aggregate_metrics,
    evaluate_prediction_set,
    evaluate_predictions,
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


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = (
    CLEANROOM_DIR
    / "configs"
    / "ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag.yaml"
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
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [str(item) for item in ast.literal_eval(text)]


def align_to_sample_times(frame: pd.DataFrame, sample_times: list[pd.Timestamp]) -> pd.DataFrame:
    time_index = {timestamp: order for order, timestamp in enumerate(sample_times)}
    filtered = frame.loc[frame["sample_time"].isin(time_index)].copy()
    filtered["_order"] = filtered["sample_time"].map(time_index)
    return filtered.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def build_cleaned_lag_frame(
    lag_minutes: int,
    *,
    lims: pd.DataFrame,
    dcs: pd.DataFrame,
    baseline_config: dict[str, object],
    aligned_sample_times: list[pd.Timestamp],
) -> pd.DataFrame:
    frame = build_shifted_feature_rows(
        lims,
        dcs,
        lookback_minutes=int(baseline_config["features"]["lookback_minutes"]),
        lag_minutes=int(lag_minutes),
        stats=list(baseline_config["features"]["window_statistics"]),
    )
    frame = assign_labels(frame, baseline_config)
    frame = align_to_sample_times(frame, aligned_sample_times)
    frame, _preclean_summary = preclean_features(frame, baseline_config)
    return align_to_sample_times(frame, aligned_sample_times)


def build_hybrid_frame(
    short_frame: pd.DataFrame,
    slow_frame: pd.DataFrame,
    *,
    slow_sensors: list[str],
) -> pd.DataFrame:
    slow_sensor_set = set(slow_sensors)
    hybrid = short_frame.copy()
    short_features = set(feature_columns(short_frame))
    slow_features = set(feature_columns(slow_frame))
    shared_features = sorted(short_features & slow_features)
    for column in shared_features:
        if sensor_name(column) in slow_sensor_set:
            hybrid[column] = slow_frame[column].to_numpy()
    return hybrid


def evaluate_candidate(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    selected_sensors: list[str],
    thresholds: list[float],
    baseline_config: dict[str, object],
    weight_candidate: dict[str, object],
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    candidate_feature_cols = feature_columns(train_frame)
    selected_features = selected_feature_columns(candidate_feature_cols, selected_sensors)
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
    metrics["selected_feature_count"] = int(len(selected_features))
    return business_probabilities, predicted, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a small stratified-lag cleanroom by mixing L180 only into a few boundary-preferred sensors and keeping the rest at L60."
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

    selection_csv = Path(base_summary["artifacts"]["baseline_selection_csv"])
    selection_frame = pd.read_csv(selection_csv)
    selection_frame["selected_sensors"] = selection_frame["selected_sensors"].apply(parse_list_cell)

    aligned_sample_times = [pd.Timestamp(item) for item in base_summary["aligned_sample_times"]]
    short_lag = int(config["stratified_lag"]["base_short_lag_minutes"])
    slow_lag = int(config["stratified_lag"]["slow_lag_minutes"])
    short_name = lag_candidate_name("L", short_lag)
    slow_name = lag_candidate_name("L", slow_lag)
    thresholds = [float(item) for item in baseline_config["labels"]["cumulative_thresholds"]]

    lims_path = discover_lims_path(PROJECT_DIR / "data", str(baseline_config["data"]["lims_glob"]))
    lims, _ = load_lims_data(lims_path)
    dcs, _dcs_audit, _alias_frame = load_identity_deduped_combined_dcs(
        PROJECT_DIR / str(baseline_config["data"]["dcs_main_path"]),
        PROJECT_DIR / str(baseline_config["data"]["dcs_supplemental_path"]),
        baseline_config,
    )

    short_frame = build_cleaned_lag_frame(
        short_lag,
        lims=lims,
        dcs=dcs,
        baseline_config=baseline_config,
        aligned_sample_times=aligned_sample_times,
    )
    slow_frame = build_cleaned_lag_frame(
        slow_lag,
        lims=lims,
        dcs=dcs,
        baseline_config=baseline_config,
        aligned_sample_times=aligned_sample_times,
    )

    candidate_frames: dict[str, pd.DataFrame] = {
        short_name: short_frame,
        slow_name: slow_frame,
    }
    candidate_groups: list[dict[str, object]] = []
    for group in config["stratified_lag"]["hybrid_groups"]:
        candidate_name = str(group["candidate_name"])
        slow_sensors = [str(item) for item in group["l180_sensors"]]
        candidate_frames[candidate_name] = build_hybrid_frame(
            short_frame,
            slow_frame,
            slow_sensors=slow_sensors,
        )
        candidate_groups.append(
            {
                "candidate_name": candidate_name,
                "l180_sensors": slow_sensors,
                "l180_sensor_count": int(len(slow_sensors)),
            }
        )

    predictions = short_frame[
        ["sample_time", "t90", "business_label", "boundary_any_flag", "is_unacceptable"]
    ].copy()
    for candidate_name in candidate_frames:
        for label in BUSINESS_LABELS:
            predictions[f"{candidate_name}_business_prob_{label}"] = pd.NA
        predictions[f"{candidate_name}_business_pred"] = pd.NA

    outer_tscv = TimeSeriesSplit(n_splits=int(baseline_config["validation"]["n_splits"]))
    split_indices = list(outer_tscv.split(short_frame))

    per_fold_rows: list[dict[str, object]] = []
    eval_rows_by_candidate: dict[str, list[dict[str, object]]] = {name: [] for name in candidate_frames}

    for fold_index, (train_index, test_index) in enumerate(split_indices, start=1):
        selection_row = selection_frame.loc[selection_frame["fold"] == fold_index].iloc[0]
        selected_sensors = list(selection_row["selected_sensors"])
        weight_candidate = {
            "candidate_name": str(selection_row["chosen_candidate_name"]),
            "boundary_weight": float(selection_row["chosen_boundary_weight"]),
            "warning_weight": float(selection_row["chosen_warning_weight"]),
        }

        for candidate_name, frame in candidate_frames.items():
            train_frame = frame.iloc[train_index].copy()
            test_frame = frame.iloc[test_index].copy()
            business_probabilities, predicted, metrics = evaluate_candidate(
                train_frame,
                test_frame,
                selected_sensors=selected_sensors,
                thresholds=thresholds,
                baseline_config=baseline_config,
                weight_candidate=weight_candidate,
            )
            for label in BUSINESS_LABELS:
                predictions.loc[test_frame.index, f"{candidate_name}_business_prob_{label}"] = business_probabilities[label]
            predictions.loc[test_frame.index, f"{candidate_name}_business_pred"] = predicted

            eval_rows_by_candidate[candidate_name].append(
                {
                    "fold": fold_index,
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "core_qualified_average_precision": metrics["core_qualified_average_precision"],
                    "boundary_warning_average_precision": metrics["boundary_warning_average_precision"],
                    "clearly_unacceptable_average_precision": metrics["clearly_unacceptable_average_precision"],
                    "boundary_high_confidence_non_warning_rate": metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                }
            )
            per_fold_rows.append(
                {
                    "fold": fold_index,
                    "candidate_name": candidate_name,
                    "selected_sensor_count_from_L0": int(len(selected_sensors)),
                    "selected_feature_count": int(metrics["selected_feature_count"]),
                    "chosen_candidate_name": str(selection_row["chosen_candidate_name"]),
                    "chosen_boundary_weight": float(selection_row["chosen_boundary_weight"]),
                    "chosen_warning_weight": float(selection_row["chosen_warning_weight"]),
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "core_qualified_average_precision": metrics["core_qualified_average_precision"],
                    "boundary_warning_average_precision": metrics["boundary_warning_average_precision"],
                    "clearly_unacceptable_average_precision": metrics["clearly_unacceptable_average_precision"],
                    "boundary_high_confidence_non_warning_rate": metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                }
            )

    candidate_summaries: dict[str, dict[str, object]] = {}
    candidate_fold_means: dict[str, dict[str, object]] = {}
    base_summary_candidate: dict[str, object] | None = None

    for candidate_name in candidate_frames:
        predictions[f"{candidate_name}_business_pred"] = predictions[f"{candidate_name}_business_pred"].astype("string")
        for label in BUSINESS_LABELS:
            predictions[f"{candidate_name}_business_prob_{label}"] = pd.to_numeric(
                predictions[f"{candidate_name}_business_prob_{label}"],
                errors="coerce",
            )
        scored = predictions.loc[predictions[f"{candidate_name}_business_pred"].notna()].copy()
        summary = evaluate_predictions(
            scored,
            pred_col=f"{candidate_name}_business_pred",
            acceptable_prob_col=f"{candidate_name}_business_prob_acceptable",
            warning_prob_col=f"{candidate_name}_business_prob_warning",
            unacceptable_prob_col=f"{candidate_name}_business_prob_unacceptable",
        )
        fold_mean = aggregate_metrics([{k: v for k, v in item.items() if k != "fold"} for item in eval_rows_by_candidate[candidate_name]])
        candidate_summaries[candidate_name] = summary
        candidate_fold_means[candidate_name] = fold_mean
        if candidate_name == short_name:
            base_summary_candidate = summary

    if base_summary_candidate is None:
        raise RuntimeError("Missing base short-lag candidate summary.")

    comparison_vs_short: list[dict[str, object]] = []
    for candidate_name, summary in candidate_summaries.items():
        comparison_vs_short.append(
            {
                "candidate_name": candidate_name,
                "macro_f1": summary["macro_f1"],
                "macro_f1_delta_vs_short": summary["macro_f1"] - base_summary_candidate["macro_f1"],
                "balanced_accuracy": summary["balanced_accuracy"],
                "balanced_accuracy_delta_vs_short": summary["balanced_accuracy"] - base_summary_candidate["balanced_accuracy"],
                "core_AP": summary["core_qualified_average_precision"],
                "core_AP_delta_vs_short": summary["core_qualified_average_precision"] - base_summary_candidate["core_qualified_average_precision"],
                "warning_AP": summary["boundary_warning_average_precision"],
                "warning_AP_delta_vs_short": summary["boundary_warning_average_precision"] - base_summary_candidate["boundary_warning_average_precision"],
                "unacceptable_AP": summary["clearly_unacceptable_average_precision"],
                "unacceptable_AP_delta_vs_short": summary["clearly_unacceptable_average_precision"] - base_summary_candidate["clearly_unacceptable_average_precision"],
                "boundary_high_confidence_non_warning_rate": summary["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                "boundary_high_conf_non_warning_delta_vs_short": (
                    summary["boundary_overconfidence"]["high_confidence_non_warning_rate"]
                    - base_summary_candidate["boundary_overconfidence"]["high_confidence_non_warning_rate"]
                ),
            }
        )

    comparison_frame = pd.DataFrame(comparison_vs_short)
    hybrid_frame = comparison_frame.loc[~comparison_frame["candidate_name"].isin({short_name, slow_name})].copy()
    best_macro_hybrid = (
        hybrid_frame.sort_values(["macro_f1", "balanced_accuracy"], ascending=[False, False]).iloc[0].to_dict()
        if not hybrid_frame.empty
        else None
    )
    best_boundary_hybrid = (
        hybrid_frame.sort_values(
            ["warning_AP", "boundary_high_confidence_non_warning_rate", "unacceptable_AP"],
            ascending=[False, True, False],
        ).iloc[0].to_dict()
        if not hybrid_frame.empty
        else None
    )

    per_fold_csv = artifacts_dir / "stratified_lag_per_fold.csv"
    results_csv = artifacts_dir / "stratified_lag_results.csv"
    summary_json = artifacts_dir / "stratified_lag_summary.json"
    report_md = reports_dir / "stratified_lag_summary.md"

    pd.DataFrame(per_fold_rows).to_csv(per_fold_csv, index=False, encoding="utf-8-sig")
    predictions.to_csv(results_csv, index=False, encoding="utf-8-sig")

    summary_payload = {
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "base_global_lag_summary_json": str(base_summary_path),
        "candidate_groups": candidate_groups,
        "candidate_summaries": candidate_summaries,
        "candidate_fold_means": candidate_fold_means,
        "comparison_vs_short_lag": comparison_vs_short,
        "best_macro_hybrid": best_macro_hybrid,
        "best_boundary_hybrid": best_boundary_hybrid,
        "artifacts": {
            "per_fold_csv": str(per_fold_csv),
            "results_csv": str(results_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stratified Lag Summary",
        "",
        "- Base short lag: L60",
        "- Slow lag used only on a small selected sensor subset: L180",
        "- Selection control: reuses the same L0-selected sensors and fold-local weight candidate from the global lag cleanroom",
        "",
        "## Pooled Results",
        "",
    ]
    for row in sorted(comparison_vs_short, key=lambda item: item["candidate_name"]):
        lines.append(
            f"- {row['candidate_name']}: "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
            f"warning_AP={row['warning_AP']:.4f}, "
            f"unacceptable_AP={row['unacceptable_AP']:.4f}, "
            f"boundary_high_conf_non_warning={row['boundary_high_confidence_non_warning_rate']:.4f}"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
