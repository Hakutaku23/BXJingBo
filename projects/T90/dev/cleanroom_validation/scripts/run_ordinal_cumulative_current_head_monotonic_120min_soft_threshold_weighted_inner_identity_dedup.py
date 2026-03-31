from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from run_ordinal_cumulative_current_head import (
    ConstantProbabilityModel,
    apply_monotonic_correction,
    assign_labels,
    build_feature_rows,
    build_logistic_pipeline,
    business_probabilities_from_intervals,
    clip_and_renormalize,
    cumulative_to_interval_probabilities,
    discover_lims_path,
    format_threshold,
    load_lims_data,
    monotonicity_summary,
    predict_threshold_probability,
    preclean_features,
    probability_argmax,
    select_topk_sensors,
    selected_feature_columns,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_identity_dedup import (
    BUSINESS_LABELS,
    CLEANROOM_DIR,
    PROJECT_DIR,
    aggregate_metrics,
    evaluate_prediction_set,
    evaluate_predictions,
    feature_columns,
    load_config,
)
from run_ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head import load_identity_deduped_combined_dcs


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "ordinal_cumulative_current_head_monotonic_120min_soft_threshold_weighted_inner_identity_dedup.yaml"


def build_candidate_grid(config: dict[str, object]) -> list[dict[str, object]]:
    objective_cfg = config["training_objective"]
    candidates = [{"candidate_name": "default", "peak_weight": 1.0, "radius": None}]
    for peak_weight in objective_cfg["peak_weight_grid"]:
        for radius in objective_cfg["radius_grid"]:
            peak_weight = float(peak_weight)
            radius = float(radius)
            candidates.append(
                {
                    "candidate_name": f"peak_{peak_weight:.2f}_radius_{radius:.2f}",
                    "peak_weight": peak_weight,
                    "radius": radius,
                }
            )
    return candidates


def build_soft_sample_weight(
    train_frame: pd.DataFrame,
    threshold: float,
    candidate: dict[str, object],
) -> pd.Series | None:
    peak_weight = float(candidate["peak_weight"])
    radius = candidate["radius"]
    if peak_weight == 1.0 or radius is None:
        return None
    radius = float(radius)
    distance = (train_frame["t90"] - float(threshold)).abs()
    closeness = (1.0 - (distance / radius)).clip(lower=0.0, upper=1.0)
    weights = 1.0 + (peak_weight - 1.0) * closeness
    if pd.Series(weights).eq(1.0).all():
        return None
    return pd.Series(weights, index=train_frame.index, dtype=float)


def fit_threshold_model_with_optional_sample_weight(
    train_frame: pd.DataFrame,
    features: list[str],
    threshold_key: str,
    config: dict[str, object],
    sample_weight: pd.Series | None,
) -> object:
    target = train_frame[threshold_key].astype(int)
    if target.nunique() < 2:
        return ConstantProbabilityModel(probability=float(target.iloc[0]))
    model_cfg = config["models"]
    model = build_logistic_pipeline(
        max_iter=int(model_cfg["logistic_max_iter"]),
        class_weight=str(model_cfg["class_weight"]),
        multiclass=False,
    )
    if sample_weight is None:
        model.fit(train_frame[features], target)
    else:
        model.fit(train_frame[features], target, model__sample_weight=sample_weight.to_numpy(dtype=float))
    return model


def fit_cumulative_business_probabilities(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    features: list[str],
    thresholds: list[float],
    config: dict[str, object],
    candidate: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weighted_thresholds = {float(item) for item in config["training_objective"]["weighted_thresholds"]}
    cumulative = pd.DataFrame(index=test_frame.index)
    for threshold in thresholds:
        key = format_threshold(threshold)
        target_key = f"target_lt_{key}"
        sample_weight = build_soft_sample_weight(train_frame, float(threshold), candidate) if float(threshold) in weighted_thresholds else None
        model = fit_threshold_model_with_optional_sample_weight(train_frame, features, target_key, config, sample_weight)
        cumulative[f"lt_{key}"] = predict_threshold_probability(model, test_frame[features])

    if bool(config["models"]["apply_monotonic_correction"]):
        cumulative = apply_monotonic_correction(cumulative)
    intervals = clip_and_renormalize(cumulative_to_interval_probabilities(cumulative))
    return cumulative, business_probabilities_from_intervals(intervals)


def select_candidate_on_train(
    train_frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    thresholds: list[float],
    config: dict[str, object],
    candidates: list[dict[str, object]],
) -> tuple[dict[str, object], pd.DataFrame]:
    inner_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["inner_n_splits"]))
    candidate_rows: list[dict[str, object]] = []

    for candidate in candidates:
        inner_rows: list[dict[str, object]] = []
        for inner_train_index, inner_val_index in inner_tscv.split(train_frame):
            inner_train = train_frame.iloc[inner_train_index].copy()
            inner_val = train_frame.iloc[inner_val_index].copy()
            selected_sensors, _ = select_topk_sensors(
                inner_train,
                feature_cols,
                topk=int(config["screening"]["topk_sensors"]),
            )
            selected_features = selected_feature_columns(feature_cols, selected_sensors)
            _cumulative, probabilities = fit_cumulative_business_probabilities(
                inner_train,
                inner_val,
                features=selected_features,
                thresholds=thresholds,
                config=config,
                candidate=candidate,
            )
            predicted = probability_argmax(probabilities, BUSINESS_LABELS)
            metrics = evaluate_prediction_set(inner_val, probabilities, predicted)
            inner_rows.append(
                {
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "core_qualified_average_precision": metrics["core_qualified_average_precision"],
                    "boundary_warning_average_precision": metrics["boundary_warning_average_precision"],
                    "clearly_unacceptable_average_precision": metrics["clearly_unacceptable_average_precision"],
                    "boundary_high_confidence_non_warning_rate": metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                }
            )

        aggregate = aggregate_metrics(inner_rows)
        candidate_rows.append(
            {
                "candidate_name": candidate["candidate_name"],
                "peak_weight": candidate["peak_weight"],
                "radius": candidate["radius"],
                **aggregate,
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows).sort_values("candidate_name").reset_index(drop=True)
    default_row = candidate_frame.loc[candidate_frame["candidate_name"] == "default"].iloc[0]
    macro_tol = float(config["training_objective"]["inner_macro_f1_tolerance"])
    bal_tol = float(config["training_objective"]["inner_balanced_accuracy_tolerance"])
    eligible = candidate_frame[
        (candidate_frame["macro_f1"] >= float(default_row["macro_f1"]) - macro_tol)
        & (candidate_frame["balanced_accuracy"] >= float(default_row["balanced_accuracy"]) - bal_tol)
    ].copy()
    if eligible.empty:
        eligible = candidate_frame.loc[candidate_frame["candidate_name"] == "default"].copy()

    chosen = eligible.sort_values(
        [
            "boundary_high_confidence_non_warning_rate",
            "boundary_warning_average_precision",
            "macro_f1",
            "balanced_accuracy",
        ],
        ascending=[True, False, False, False],
    ).iloc[0]
    return chosen.to_dict(), candidate_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Test soft distance-to-threshold weighting on the strongest identity-deduped simple baseline.")
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
    dcs, dcs_audit, alias_frame = load_identity_deduped_combined_dcs(main_dcs_path, supplemental_path, config)
    feature_rows = build_feature_rows(
        lims,
        dcs,
        lookback_minutes=int(config["features"]["lookback_minutes"]),
        stats=list(config["features"]["window_statistics"]),
    )
    feature_rows = assign_labels(feature_rows, config)
    feature_rows, preclean_summary = preclean_features(feature_rows, config)
    feature_cols = feature_columns(feature_rows)
    thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]
    candidates = build_candidate_grid(config)
    weighted_thresholds = [float(item) for item in config["training_objective"]["weighted_thresholds"]]

    outer_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    predictions = feature_rows[
        ["sample_time", "t90", "business_label", "boundary_any_flag", "is_unacceptable"]
        + [column for column in feature_rows.columns if column.startswith("target_lt_")]
    ].copy()
    for threshold in thresholds:
        key = format_threshold(threshold)
        predictions[f"default_cum_prob_lt_{key}"] = pd.NA
        predictions[f"soft_cum_prob_lt_{key}"] = pd.NA
    for label in BUSINESS_LABELS:
        predictions[f"default_business_prob_{label}"] = pd.NA
        predictions[f"soft_business_prob_{label}"] = pd.NA
    predictions["default_business_pred"] = pd.NA
    predictions["soft_business_pred"] = pd.NA

    per_fold_rows: list[dict[str, object]] = []
    selected_candidate_rows: list[dict[str, object]] = []
    inner_search_rows: list[dict[str, object]] = []
    default_eval_rows: list[dict[str, object]] = []
    soft_eval_rows: list[dict[str, object]] = []
    default_candidate = {"candidate_name": "default", "peak_weight": 1.0, "radius": None}

    for fold_index, (train_index, test_index) in enumerate(outer_tscv.split(feature_rows), start=1):
        train_frame = feature_rows.iloc[train_index].copy()
        test_frame = feature_rows.iloc[test_index].copy()

        chosen_candidate, candidate_frame = select_candidate_on_train(
            train_frame,
            feature_cols=feature_cols,
            thresholds=thresholds,
            config=config,
            candidates=candidates,
        )
        candidate_frame.insert(0, "fold", fold_index)
        inner_search_rows.extend(candidate_frame.to_dict(orient="records"))
        selected_candidate_rows.append({"fold": fold_index, **chosen_candidate})

        selected_sensors, selected_scores = select_topk_sensors(
            train_frame,
            feature_cols,
            topk=int(config["screening"]["topk_sensors"]),
        )
        selected_features = selected_feature_columns(feature_cols, selected_sensors)
        default_cumulative, default_probabilities = fit_cumulative_business_probabilities(
            train_frame,
            test_frame,
            features=selected_features,
            thresholds=thresholds,
            config=config,
            candidate=default_candidate,
        )
        soft_cumulative, soft_probabilities = fit_cumulative_business_probabilities(
            train_frame,
            test_frame,
            features=selected_features,
            thresholds=thresholds,
            config=config,
            candidate=chosen_candidate,
        )

        default_pred = probability_argmax(default_probabilities, BUSINESS_LABELS)
        soft_pred = probability_argmax(soft_probabilities, BUSINESS_LABELS)

        for threshold in thresholds:
            key = format_threshold(threshold)
            predictions.loc[test_frame.index, f"default_cum_prob_lt_{key}"] = default_cumulative[f"lt_{key}"]
            predictions.loc[test_frame.index, f"soft_cum_prob_lt_{key}"] = soft_cumulative[f"lt_{key}"]
        for label in BUSINESS_LABELS:
            predictions.loc[test_frame.index, f"default_business_prob_{label}"] = default_probabilities[label]
            predictions.loc[test_frame.index, f"soft_business_prob_{label}"] = soft_probabilities[label]
        predictions.loc[test_frame.index, "default_business_pred"] = default_pred
        predictions.loc[test_frame.index, "soft_business_pred"] = soft_pred

        default_metrics = evaluate_prediction_set(test_frame, default_probabilities, default_pred)
        soft_metrics = evaluate_prediction_set(test_frame, soft_probabilities, soft_pred)
        default_eval_rows.append(
            {
                "fold": fold_index,
                "macro_f1": default_metrics["macro_f1"],
                "balanced_accuracy": default_metrics["balanced_accuracy"],
                "core_qualified_average_precision": default_metrics["core_qualified_average_precision"],
                "boundary_warning_average_precision": default_metrics["boundary_warning_average_precision"],
                "clearly_unacceptable_average_precision": default_metrics["clearly_unacceptable_average_precision"],
                "boundary_high_confidence_non_warning_rate": default_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )
        soft_eval_rows.append(
            {
                "fold": fold_index,
                "macro_f1": soft_metrics["macro_f1"],
                "balanced_accuracy": soft_metrics["balanced_accuracy"],
                "core_qualified_average_precision": soft_metrics["core_qualified_average_precision"],
                "boundary_warning_average_precision": soft_metrics["boundary_warning_average_precision"],
                "clearly_unacceptable_average_precision": soft_metrics["clearly_unacceptable_average_precision"],
                "boundary_high_confidence_non_warning_rate": soft_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )

        per_fold_rows.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "selected_sensor_count": int(len(selected_sensors)),
                "selected_sensors": selected_sensors,
                "selected_sensor_scores": selected_scores,
                "chosen_candidate_name": chosen_candidate["candidate_name"],
                "chosen_peak_weight": chosen_candidate["peak_weight"],
                "chosen_radius": chosen_candidate["radius"],
                "weighted_thresholds": weighted_thresholds,
                "default_macro_f1": default_metrics["macro_f1"],
                "soft_macro_f1": soft_metrics["macro_f1"],
                "default_balanced_accuracy": default_metrics["balanced_accuracy"],
                "soft_balanced_accuracy": soft_metrics["balanced_accuracy"],
                "default_boundary_high_conf_non_warning_rate": default_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                "soft_boundary_high_conf_non_warning_rate": soft_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )

    for label in BUSINESS_LABELS:
        predictions[f"default_business_prob_{label}"] = pd.to_numeric(predictions[f"default_business_prob_{label}"], errors="coerce")
        predictions[f"soft_business_prob_{label}"] = pd.to_numeric(predictions[f"soft_business_prob_{label}"], errors="coerce")
    for threshold in thresholds:
        key = format_threshold(threshold)
        predictions[f"default_cum_prob_lt_{key}"] = pd.to_numeric(predictions[f"default_cum_prob_lt_{key}"], errors="coerce")
        predictions[f"soft_cum_prob_lt_{key}"] = pd.to_numeric(predictions[f"soft_cum_prob_lt_{key}"], errors="coerce")

    scored_predictions = predictions.loc[
        predictions["default_business_pred"].notna() & predictions["soft_business_pred"].notna()
    ].copy()

    default_summary = evaluate_predictions(
        scored_predictions,
        pred_col="default_business_pred",
        acceptable_prob_col="default_business_prob_acceptable",
        warning_prob_col="default_business_prob_warning",
        unacceptable_prob_col="default_business_prob_unacceptable",
    )
    soft_summary = evaluate_predictions(
        scored_predictions,
        pred_col="soft_business_pred",
        acceptable_prob_col="soft_business_prob_acceptable",
        warning_prob_col="soft_business_prob_warning",
        unacceptable_prob_col="soft_business_prob_unacceptable",
    )
    default_monotonicity = monotonicity_summary(
        scored_predictions.rename(columns={f"default_cum_prob_lt_{format_threshold(t)}": f"cum_prob_lt_{format_threshold(t)}" for t in thresholds}),
        thresholds,
    )
    soft_monotonicity = monotonicity_summary(
        scored_predictions.rename(columns={f"soft_cum_prob_lt_{format_threshold(t)}": f"cum_prob_lt_{format_threshold(t)}" for t in thresholds}),
        thresholds,
    )
    default_fold_mean = aggregate_metrics([{k: v for k, v in row.items() if k != "fold"} for row in default_eval_rows])
    soft_fold_mean = aggregate_metrics([{k: v for k, v in row.items() if k != "fold"} for row in soft_eval_rows])
    improvement = {
        "macro_f1_delta": soft_summary["macro_f1"] - default_summary["macro_f1"],
        "balanced_accuracy_delta": soft_summary["balanced_accuracy"] - default_summary["balanced_accuracy"],
        "core_qualified_average_precision_delta": soft_summary["core_qualified_average_precision"] - default_summary["core_qualified_average_precision"],
        "boundary_warning_average_precision_delta": soft_summary["boundary_warning_average_precision"] - default_summary["boundary_warning_average_precision"],
        "clearly_unacceptable_average_precision_delta": soft_summary["clearly_unacceptable_average_precision"] - default_summary["clearly_unacceptable_average_precision"],
        "boundary_high_confidence_non_warning_rate_delta": soft_summary["boundary_overconfidence"]["high_confidence_non_warning_rate"] - default_summary["boundary_overconfidence"]["high_confidence_non_warning_rate"],
    }

    per_fold_csv = artifacts_dir / "soft_threshold_weighted_per_fold.csv"
    selected_candidate_csv = artifacts_dir / "soft_threshold_weighted_selected_candidate_per_fold.csv"
    inner_search_csv = artifacts_dir / "soft_threshold_weighted_inner_search.csv"
    results_csv = artifacts_dir / "soft_threshold_weighted_results.csv"
    alias_csv = artifacts_dir / "sensor_identity_alias_pairs.csv"
    summary_json = artifacts_dir / "soft_threshold_weighted_summary.json"
    report_md = reports_dir / "soft_threshold_weighted_summary.md"

    pd.DataFrame(per_fold_rows).to_csv(per_fold_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_candidate_rows).to_csv(selected_candidate_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(inner_search_rows).to_csv(inner_search_csv, index=False, encoding="utf-8-sig")
    scored_predictions.to_csv(results_csv, index=False, encoding="utf-8-sig")
    alias_frame.to_csv(alias_csv, index=False, encoding="utf-8-sig")

    summary = {
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "weighted_thresholds": weighted_thresholds,
        "data_paths": {
            "dcs_main_path": str(main_dcs_path),
            "dcs_supplemental_path": str(supplemental_path),
            "lims_path": str(lims_path),
        },
        "dcs_audit": dcs_audit,
        "identity_alias_pairs": alias_frame.to_dict(orient="records"),
        "data_summary": {
            "aligned_rows": int(len(feature_rows)),
            "raw_common_rows": int(len(predictions)),
            "scored_rows": int(len(scored_predictions)),
            "n_splits_used": int(config["validation"]["n_splits"]),
            "inner_n_splits_used": int(config["validation"]["inner_n_splits"]),
        },
        "preclean_summary": preclean_summary,
        "candidate_grid_size": int(len(candidates)),
        "default_summary": default_summary,
        "soft_summary": soft_summary,
        "default_monotonicity": default_monotonicity,
        "soft_monotonicity": soft_monotonicity,
        "default_fold_mean": default_fold_mean,
        "soft_fold_mean": soft_fold_mean,
        "improvement_summary": improvement,
        "per_fold": per_fold_rows,
        "selected_candidate_rows": selected_candidate_rows,
        "artifacts": {
            "per_fold_csv": str(per_fold_csv),
            "selected_candidate_csv": str(selected_candidate_csv),
            "inner_search_csv": str(inner_search_csv),
            "results_csv": str(results_csv),
            "alias_csv": str(alias_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Soft Threshold-Weighted Summary",
        "",
        "- Baseline representation: simple 120min monotonic ordinal/cumulative",
        "- DCS source: merge_data + merge_data_otr with strict sensor-identity de-dup",
        f"- Weighted thresholds only: {weighted_thresholds}",
        "- Intervention: linear soft weighting by distance to threshold on inner thresholds only",
        f"- Candidate grid size: {len(candidates)}",
        f"- Scored rows: {len(scored_predictions)}",
        f"- Default macro_f1: {default_summary['macro_f1']:.4f}",
        f"- Soft macro_f1: {soft_summary['macro_f1']:.4f}",
        f"- Default balanced_accuracy: {default_summary['balanced_accuracy']:.4f}",
        f"- Soft balanced_accuracy: {soft_summary['balanced_accuracy']:.4f}",
        f"- Default core_AP: {default_summary['core_qualified_average_precision']:.4f}",
        f"- Soft core_AP: {soft_summary['core_qualified_average_precision']:.4f}",
        f"- Default warning_AP: {default_summary['boundary_warning_average_precision']:.4f}",
        f"- Soft warning_AP: {soft_summary['boundary_warning_average_precision']:.4f}",
        f"- Default unacceptable_AP: {default_summary['clearly_unacceptable_average_precision']:.4f}",
        f"- Soft unacceptable_AP: {soft_summary['clearly_unacceptable_average_precision']:.4f}",
        f"- Default boundary high-confidence non-warning: {default_summary['boundary_overconfidence']['high_confidence_non_warning_rate']:.4f}",
        f"- Soft boundary high-confidence non-warning: {soft_summary['boundary_overconfidence']['high_confidence_non_warning_rate']:.4f}",
        "",
        "## Selected Candidate Per Fold",
        "",
    ]
    for row in selected_candidate_rows:
        lines.append(
            f"- fold {row['fold']}: {row['candidate_name']} "
            f"(peak_weight={row['peak_weight']}, radius={row['radius']})"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
