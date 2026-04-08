from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from run_ordinal_cumulative_current_head import (
    discover_lims_path,
    feature_columns,
    format_threshold,
    load_lims_data,
    preclean_features,
    probability_argmax,
    select_topk_sensors,
    selected_feature_columns,
    sensor_name,
    summarize_window,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup import (
    BUSINESS_LABELS,
    CLEANROOM_DIR,
    PROJECT_DIR,
    aggregate_metrics,
    assign_labels,
    build_candidate_grid,
    evaluate_prediction_set,
    evaluate_predictions,
    fit_cumulative_business_probabilities_inner_thresholds_only,
    load_config,
    monotonicity_summary,
    select_candidate_on_train,
)
from run_ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head import (
    load_identity_deduped_combined_dcs,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = (
    CLEANROOM_DIR
    / "configs"
    / "ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag.yaml"
)


def lag_candidate_name(prefix: str, lag_minutes: int) -> str:
    return f"{prefix}{int(lag_minutes)}"


def build_shifted_feature_rows(
    lims: pd.DataFrame,
    dcs: pd.DataFrame,
    *,
    lookback_minutes: int,
    lag_minutes: int,
    stats: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labeled = lims.dropna(subset=["sample_time", "t90"]).copy()
    dcs = dcs.sort_values("time").reset_index(drop=True)
    time_values = dcs["time"].to_numpy()
    lookback = pd.Timedelta(minutes=int(lookback_minutes))
    lag = pd.Timedelta(minutes=int(lag_minutes))

    for record in labeled.itertuples(index=False):
        sample_time = pd.Timestamp(record.sample_time)
        window_end = sample_time - lag
        window_start = window_end - lookback
        left = np.searchsorted(time_values, window_start.to_datetime64(), side="left")
        right = np.searchsorted(time_values, window_end.to_datetime64(), side="right")
        window = dcs.iloc[left:right]
        if window.empty:
            continue

        row: dict[str, object] = {
            "sample_time": sample_time,
            "window_start": window_start,
            "window_end": window_end,
            "window_row_count": int(len(window)),
            "lag_minutes": int(lag_minutes),
            "sheet_name": getattr(record, "sheet_name", "unknown"),
            "sample_name": getattr(record, "sample_name", None),
            "t90": float(record.t90),
        }
        row.update(summarize_window(window, stats))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)


def align_candidate_frames(
    candidate_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, int]]:
    common_times: set[pd.Timestamp] | None = None
    raw_row_counts: dict[str, int] = {}
    for name, frame in candidate_frames.items():
        raw_row_counts[name] = int(len(frame))
        times = set(pd.to_datetime(frame["sample_time"]))
        common_times = times if common_times is None else common_times & times
    aligned_times = sorted(common_times or [])
    aligned_time_set = set(aligned_times)

    aligned: dict[str, pd.DataFrame] = {}
    for name, frame in candidate_frames.items():
        filtered = frame.loc[frame["sample_time"].isin(aligned_time_set)].copy()
        aligned[name] = filtered.sort_values("sample_time").reset_index(drop=True)
    return aligned, [str(ts) for ts in aligned_times], raw_row_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate global lag shift sensitivity on top of the frozen strongest identity-deduped threshold-oriented cleanroom."
    )
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

    lookback_minutes = int(config["features"]["lookback_minutes"])
    stats = list(config["features"]["window_statistics"])
    thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]
    lag_grid_minutes = [int(item) for item in config["lag_validation"]["lag_grid_minutes"]]
    anchor_prefix = str(config["lag_validation"]["anchor_name_prefix"])

    lims, _ = load_lims_data(lims_path)
    dcs, dcs_audit, alias_frame = load_identity_deduped_combined_dcs(
        main_dcs_path,
        supplemental_path,
        config,
    )

    raw_candidate_frames: dict[str, pd.DataFrame] = {}
    lag_rows: list[dict[str, object]] = []
    for lag_minutes in lag_grid_minutes:
        candidate_name = lag_candidate_name(anchor_prefix, lag_minutes)
        frame = build_shifted_feature_rows(
            lims,
            dcs,
            lookback_minutes=lookback_minutes,
            lag_minutes=lag_minutes,
            stats=stats,
        )
        frame = assign_labels(frame, config)
        raw_candidate_frames[candidate_name] = frame
        lag_rows.append(
            {
                "candidate_name": candidate_name,
                "lag_minutes": lag_minutes,
                "raw_rows_before_alignment": int(len(frame)),
            }
        )

    aligned_frames, aligned_times, raw_row_counts = align_candidate_frames(raw_candidate_frames)
    cleaned_frames: dict[str, pd.DataFrame] = {}
    preclean_by_candidate: dict[str, dict[str, object]] = {}
    feature_cols_by_candidate: dict[str, list[str]] = {}

    for row in lag_rows:
        candidate_name = str(row["candidate_name"])
        aligned_frame = aligned_frames[candidate_name]
        cleaned_frame, preclean_summary = preclean_features(aligned_frame, config)
        cleaned_frame = cleaned_frame.sort_values("sample_time").reset_index(drop=True)
        cleaned_frames[candidate_name] = cleaned_frame
        preclean_by_candidate[candidate_name] = preclean_summary
        feature_cols_by_candidate[candidate_name] = feature_columns(cleaned_frame)
        row["aligned_rows_after_intersection"] = int(len(aligned_frame))
        row["rows_after_preclean"] = int(len(cleaned_frame))
        row["feature_count_after_preclean"] = int(len(feature_cols_by_candidate[candidate_name]))

    base_candidate_name = lag_candidate_name(anchor_prefix, 0)
    base_frame = cleaned_frames[base_candidate_name]
    base_feature_cols = feature_cols_by_candidate[base_candidate_name]
    candidates = build_candidate_grid(config)

    outer_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    predictions = base_frame[
        ["sample_time", "t90", "business_label", "boundary_any_flag", "is_unacceptable"]
        + [column for column in base_frame.columns if column.startswith("target_lt_")]
    ].copy()

    for row in lag_rows:
        candidate_name = str(row["candidate_name"])
        for threshold in thresholds:
            key = format_threshold(threshold)
            predictions[f"{candidate_name}_cum_prob_lt_{key}"] = pd.NA
        for label in BUSINESS_LABELS:
            predictions[f"{candidate_name}_business_prob_{label}"] = pd.NA
        predictions[f"{candidate_name}_business_pred"] = pd.NA

    lag_eval_rows: dict[str, list[dict[str, object]]] = {str(row["candidate_name"]): [] for row in lag_rows}
    per_fold_rows: list[dict[str, object]] = []
    baseline_selection_rows: list[dict[str, object]] = []
    inner_search_rows: list[dict[str, object]] = []

    for fold_index, (train_index, test_index) in enumerate(outer_tscv.split(base_frame), start=1):
        train_base = base_frame.iloc[train_index].copy()
        chosen_candidate, candidate_frame = select_candidate_on_train(
            train_base,
            feature_cols=base_feature_cols,
            thresholds=thresholds,
            config=config,
            candidates=candidates,
        )
        candidate_frame.insert(0, "fold", fold_index)
        inner_search_rows.extend(candidate_frame.to_dict(orient="records"))

        selected_sensors, selected_scores = select_topk_sensors(
            train_base,
            base_feature_cols,
            topk=int(config["screening"]["topk_sensors"]),
        )
        baseline_selection_rows.append(
            {
                "fold": fold_index,
                "chosen_candidate_name": chosen_candidate["candidate_name"],
                "chosen_boundary_weight": chosen_candidate["boundary_weight"],
                "chosen_warning_weight": chosen_candidate["warning_weight"],
                "selected_sensor_count": int(len(selected_sensors)),
                "selected_sensors": selected_sensors,
                "selected_sensor_scores": selected_scores,
            }
        )

        for row in lag_rows:
            candidate_name = str(row["candidate_name"])
            lag_minutes = int(row["lag_minutes"])
            candidate_frame_full = cleaned_frames[candidate_name]
            train_frame = candidate_frame_full.iloc[train_index].copy()
            test_frame = candidate_frame_full.iloc[test_index].copy()
            candidate_feature_cols = feature_cols_by_candidate[candidate_name]
            selected_features = selected_feature_columns(candidate_feature_cols, selected_sensors)
            available_selected_sensors = sorted({sensor_name(column) for column in selected_features})

            cumulative, business_probabilities = fit_cumulative_business_probabilities_inner_thresholds_only(
                train_frame,
                test_frame,
                features=selected_features,
                thresholds=thresholds,
                config=config,
                candidate=chosen_candidate,
            )
            predicted = probability_argmax(business_probabilities, BUSINESS_LABELS)
            metrics = evaluate_prediction_set(test_frame, business_probabilities, predicted)

            for threshold in thresholds:
                key = format_threshold(threshold)
                predictions.loc[test_frame.index, f"{candidate_name}_cum_prob_lt_{key}"] = cumulative[f"lt_{key}"]
            for label in BUSINESS_LABELS:
                predictions.loc[test_frame.index, f"{candidate_name}_business_prob_{label}"] = business_probabilities[label]
            predictions.loc[test_frame.index, f"{candidate_name}_business_pred"] = predicted

            lag_eval_rows[candidate_name].append(
                {
                    "fold": fold_index,
                    "lag_minutes": lag_minutes,
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
                    "lag_minutes": lag_minutes,
                    "selected_sensor_count_from_L0": int(len(selected_sensors)),
                    "available_selected_sensor_count": int(len(available_selected_sensors)),
                    "available_selected_sensors": available_selected_sensors,
                    "selected_feature_count": int(len(selected_features)),
                    "chosen_candidate_name": chosen_candidate["candidate_name"],
                    "chosen_boundary_weight": chosen_candidate["boundary_weight"],
                    "chosen_warning_weight": chosen_candidate["warning_weight"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "core_qualified_average_precision": metrics["core_qualified_average_precision"],
                    "boundary_warning_average_precision": metrics["boundary_warning_average_precision"],
                    "clearly_unacceptable_average_precision": metrics["clearly_unacceptable_average_precision"],
                    "boundary_high_confidence_non_warning_rate": metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                }
            )

    for row in lag_rows:
        candidate_name = str(row["candidate_name"])
        predictions[f"{candidate_name}_business_pred"] = predictions[f"{candidate_name}_business_pred"].astype("string")
        for label in BUSINESS_LABELS:
            predictions[f"{candidate_name}_business_prob_{label}"] = pd.to_numeric(
                predictions[f"{candidate_name}_business_prob_{label}"],
                errors="coerce",
            )
        for threshold in thresholds:
            key = format_threshold(threshold)
            predictions[f"{candidate_name}_cum_prob_lt_{key}"] = pd.to_numeric(
                predictions[f"{candidate_name}_cum_prob_lt_{key}"],
                errors="coerce",
            )

    lag_summaries: dict[str, dict[str, object]] = {}
    lag_fold_means: dict[str, dict[str, object]] = {}
    lag_monotonicity: dict[str, dict[str, object]] = {}
    comparison_rows: list[dict[str, object]] = []
    base_summary: dict[str, object] | None = None

    for row in lag_rows:
        candidate_name = str(row["candidate_name"])
        scored = predictions.loc[predictions[f"{candidate_name}_business_pred"].notna()].copy()
        summary = evaluate_predictions(
            scored,
            pred_col=f"{candidate_name}_business_pred",
            acceptable_prob_col=f"{candidate_name}_business_prob_acceptable",
            warning_prob_col=f"{candidate_name}_business_prob_warning",
            unacceptable_prob_col=f"{candidate_name}_business_prob_unacceptable",
        )
        monotonicity = monotonicity_summary(
            scored.rename(
                columns={
                    f"{candidate_name}_cum_prob_lt_{format_threshold(threshold)}": f"cum_prob_lt_{format_threshold(threshold)}"
                    for threshold in thresholds
                }
            ),
            thresholds,
        )
        fold_mean = aggregate_metrics([{k: v for k, v in item.items() if k not in {"fold", "lag_minutes"}} for item in lag_eval_rows[candidate_name]])
        lag_summaries[candidate_name] = summary
        lag_fold_means[candidate_name] = fold_mean
        lag_monotonicity[candidate_name] = monotonicity
        if candidate_name == base_candidate_name:
            base_summary = summary

    if base_summary is None:
        raise RuntimeError("Failed to compute base L0 summary.")

    for row in lag_rows:
        candidate_name = str(row["candidate_name"])
        summary = lag_summaries[candidate_name]
        comparison_rows.append(
            {
                "candidate_name": candidate_name,
                "lag_minutes": int(row["lag_minutes"]),
                "macro_f1": summary["macro_f1"],
                "macro_f1_delta_vs_L0": summary["macro_f1"] - base_summary["macro_f1"],
                "balanced_accuracy": summary["balanced_accuracy"],
                "balanced_accuracy_delta_vs_L0": summary["balanced_accuracy"] - base_summary["balanced_accuracy"],
                "core_qualified_average_precision": summary["core_qualified_average_precision"],
                "core_AP_delta_vs_L0": summary["core_qualified_average_precision"] - base_summary["core_qualified_average_precision"],
                "boundary_warning_average_precision": summary["boundary_warning_average_precision"],
                "warning_AP_delta_vs_L0": summary["boundary_warning_average_precision"] - base_summary["boundary_warning_average_precision"],
                "clearly_unacceptable_average_precision": summary["clearly_unacceptable_average_precision"],
                "unacceptable_AP_delta_vs_L0": summary["clearly_unacceptable_average_precision"] - base_summary["clearly_unacceptable_average_precision"],
                "boundary_high_confidence_non_warning_rate": summary["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                "boundary_high_conf_non_warning_delta_vs_L0": (
                    summary["boundary_overconfidence"]["high_confidence_non_warning_rate"]
                    - base_summary["boundary_overconfidence"]["high_confidence_non_warning_rate"]
                ),
            }
        )

    l180_name = lag_candidate_name(anchor_prefix, 180)
    l120_name = lag_candidate_name(anchor_prefix, 120)
    l240_name = lag_candidate_name(anchor_prefix, 240)
    per_fold_frame = pd.DataFrame(per_fold_rows)
    l180_frame = per_fold_frame.loc[per_fold_frame["candidate_name"] == l180_name].copy()
    l0_frame = per_fold_frame.loc[per_fold_frame["candidate_name"] == base_candidate_name].copy()
    l120_frame = per_fold_frame.loc[per_fold_frame["candidate_name"] == l120_name].copy()
    l240_frame = per_fold_frame.loc[per_fold_frame["candidate_name"] == l240_name].copy()

    l180_fold_signal = {
        "macro_f1_better_than_L0_fold_count": int((l180_frame["macro_f1"].to_numpy() > l0_frame["macro_f1"].to_numpy()).sum()),
        "balanced_accuracy_better_than_L0_fold_count": int((l180_frame["balanced_accuracy"].to_numpy() > l0_frame["balanced_accuracy"].to_numpy()).sum()),
        "warning_AP_better_than_L0_fold_count": int(
            (l180_frame["boundary_warning_average_precision"].to_numpy() > l0_frame["boundary_warning_average_precision"].to_numpy()).sum()
        ),
        "unacceptable_AP_not_worse_than_L0_fold_count": int(
            (l180_frame["clearly_unacceptable_average_precision"].to_numpy() >= l0_frame["clearly_unacceptable_average_precision"].to_numpy()).sum()
        ),
        "boundary_high_conf_non_warning_better_than_L0_fold_count": int(
            (
                l180_frame["boundary_high_confidence_non_warning_rate"].to_numpy()
                < l0_frame["boundary_high_confidence_non_warning_rate"].to_numpy()
            ).sum()
        ),
        "macro_f1_better_than_L120_fold_count": int((l180_frame["macro_f1"].to_numpy() > l120_frame["macro_f1"].to_numpy()).sum()),
        "macro_f1_better_than_L240_fold_count": int((l180_frame["macro_f1"].to_numpy() > l240_frame["macro_f1"].to_numpy()).sum()),
    }

    per_fold_csv = artifacts_dir / "global_lag_per_fold.csv"
    baseline_selection_csv = artifacts_dir / "global_lag_L0_selection_per_fold.csv"
    inner_search_csv = artifacts_dir / "global_lag_L0_inner_search.csv"
    results_csv = artifacts_dir / "global_lag_results.csv"
    lag_audit_csv = artifacts_dir / "global_lag_candidate_audit.csv"
    alias_csv = artifacts_dir / "sensor_identity_alias_pairs.csv"
    summary_json = artifacts_dir / "global_lag_summary.json"
    report_md = reports_dir / "global_lag_summary.md"

    pd.DataFrame(per_fold_rows).to_csv(per_fold_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(baseline_selection_rows).to_csv(baseline_selection_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(inner_search_rows).to_csv(inner_search_csv, index=False, encoding="utf-8-sig")
    predictions.to_csv(results_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(lag_rows).to_csv(lag_audit_csv, index=False, encoding="utf-8-sig")
    alias_frame.to_csv(alias_csv, index=False, encoding="utf-8-sig")

    summary = {
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "data_paths": {
            "dcs_main_path": str(main_dcs_path),
            "dcs_supplemental_path": str(supplemental_path),
            "lims_path": str(lims_path),
        },
        "dcs_audit": dcs_audit,
        "identity_alias_pairs": alias_frame.to_dict(orient="records"),
        "lag_candidates": lag_rows,
        "aligned_sample_times": aligned_times,
        "raw_row_counts": raw_row_counts,
        "data_summary": {
            "raw_lims_rows": int(len(lims)),
            "common_rows_after_lag_intersection": int(len(base_frame)),
            "n_splits_used": int(config["validation"]["n_splits"]),
            "inner_n_splits_used": int(config["validation"]["inner_n_splits"]),
            "topk_sensors_fixed_from_L0": int(config["screening"]["topk_sensors"]),
        },
        "preclean_summary_by_candidate": preclean_by_candidate,
        "l0_selection_per_fold": baseline_selection_rows,
        "per_fold": per_fold_rows,
        "lag_summaries": lag_summaries,
        "lag_fold_means": lag_fold_means,
        "lag_monotonicity": lag_monotonicity,
        "comparison_vs_L0": comparison_rows,
        "l180_fold_signal": l180_fold_signal,
        "artifacts": {
            "per_fold_csv": str(per_fold_csv),
            "baseline_selection_csv": str(baseline_selection_csv),
            "inner_search_csv": str(inner_search_csv),
            "results_csv": str(results_csv),
            "lag_audit_csv": str(lag_audit_csv),
            "alias_csv": str(alias_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Global Lag Validation Summary",
        "",
        "- Frozen anchor: strongest threshold-oriented cleanroom (`simple 120min + topk40 + identity de-dup + monotonic cumulative + inner-threshold weighting`)",
        "- Variable changed: only the DCS window anchor shift",
        "- Selection control: fold-local sensors and weight candidate selected on `L0` train only, then reused by all lag arms",
        f"- Common scored rows across lag arms: {len(base_frame)}",
        "",
        "## Pooled Results",
        "",
    ]
    for comparison in sorted(comparison_rows, key=lambda item: int(item["lag_minutes"])):
        lines.append(
            f"- {comparison['candidate_name']} ({comparison['lag_minutes']} min): "
            f"macro_f1={comparison['macro_f1']:.4f}, "
            f"balanced_accuracy={comparison['balanced_accuracy']:.4f}, "
            f"warning_AP={comparison['boundary_warning_average_precision']:.4f}, "
            f"unacceptable_AP={comparison['clearly_unacceptable_average_precision']:.4f}, "
            f"boundary_high_conf_non_warning={comparison['boundary_high_confidence_non_warning_rate']:.4f}"
        )
    lines.extend(
        [
            "",
            "## L180 Fold Signal",
            "",
            f"- macro_f1 better than L0 folds: {l180_fold_signal['macro_f1_better_than_L0_fold_count']}/5",
            f"- balanced_accuracy better than L0 folds: {l180_fold_signal['balanced_accuracy_better_than_L0_fold_count']}/5",
            f"- warning_AP better than L0 folds: {l180_fold_signal['warning_AP_better_than_L0_fold_count']}/5",
            f"- unacceptable_AP not worse than L0 folds: {l180_fold_signal['unacceptable_AP_not_worse_than_L0_fold_count']}/5",
            f"- boundary overconfidence better than L0 folds: {l180_fold_signal['boundary_high_conf_non_warning_better_than_L0_fold_count']}/5",
            f"- macro_f1 better than L120 folds: {l180_fold_signal['macro_f1_better_than_L120_fold_count']}/5",
            f"- macro_f1 better than L240 folds: {l180_fold_signal['macro_f1_better_than_L240_fold_count']}/5",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
