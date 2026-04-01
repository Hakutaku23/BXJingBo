from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from run_distributional_reject_option_current_head import (
    BIN_LABELS,
    boundary_subset_stats,
    build_soft_distribution_labels,
    business_probabilities_from_bins,
    entropy_summary,
    fit_distribution_model,
    json_ready,
    load_config,
    multiclass_brier_score,
    multiclass_log_loss,
    normalized_entropy,
    predict_distribution_probabilities,
)
from run_ordinal_cumulative_current_head import (
    build_feature_rows,
    clip_and_renormalize,
    cumulative_to_interval_probabilities,
    discover_lims_path,
    feature_columns,
    load_lims_data,
    preclean_features,
    probability_argmax,
    select_topk_sensors,
    selected_feature_columns,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_identity_dedup import (
    BUSINESS_LABELS,
    aggregate_metrics,
    assign_labels,
    binary_ap,
    boundary_overconfidence_stats,
    evaluate_prediction_set,
    evaluate_predictions,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup import (
    CLEANROOM_DIR,
    PROJECT_DIR,
    build_candidate_grid,
    fit_cumulative_business_probabilities_inner_thresholds_only,
    select_candidate_on_train as select_reference_candidate_on_train,
)
from run_ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head import load_identity_deduped_combined_dcs


DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "distributional_reject_option_current_head_treatment_c.yaml"
REJECT_LABEL = "retest"


def build_reject_candidates(config: dict[str, object]) -> list[dict[str, object]]:
    reject_cfg = config["reject"]
    candidates = [
        {
            "candidate_name": "default_no_reject",
            "tau_conf": None,
            "tau_entropy": None,
            "use_reject": False,
        }
    ]
    for tau_conf in reject_cfg["confidence_threshold_grid"]:
        for tau_entropy in reject_cfg["entropy_threshold_grid"]:
            candidates.append(
                {
                    "candidate_name": f"conf_{float(tau_conf):.2f}_entropy_{float(tau_entropy):.2f}",
                    "tau_conf": float(tau_conf),
                    "tau_entropy": float(tau_entropy),
                    "use_reject": True,
                }
            )
    return candidates


def apply_reject_rule(
    bin_probabilities: pd.DataFrame,
    business_probabilities: pd.DataFrame,
    *,
    candidate: dict[str, object],
    eps: float,
) -> tuple[pd.Series, pd.Series]:
    base_pred = probability_argmax(business_probabilities, BUSINESS_LABELS)
    rejected = pd.Series(False, index=bin_probabilities.index)
    if bool(candidate["use_reject"]):
        max_business_prob = business_probabilities[BUSINESS_LABELS].max(axis=1)
        entropy = normalized_entropy(bin_probabilities, labels=BIN_LABELS, eps=eps)
        rejected = (max_business_prob < float(candidate["tau_conf"])) | (entropy > float(candidate["tau_entropy"]))
    pred = base_pred.copy()
    pred.loc[rejected] = REJECT_LABEL
    return pred, rejected


def evaluate_treatment_predictions(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    rejected_col: str,
    acceptable_prob_col: str,
    warning_prob_col: str,
    unacceptable_prob_col: str,
) -> dict[str, object]:
    rejected = frame[rejected_col].astype(bool)
    covered = frame.loc[~rejected].copy()
    boundary_all = frame.loc[frame["boundary_any_flag"]].copy()
    boundary_covered = covered.loc[covered["boundary_any_flag"]].copy()

    reject_rate = float(rejected.mean())
    coverage = float((~rejected).mean())

    if covered.empty:
        covered_macro_f1 = math.nan
        covered_balanced_accuracy = math.nan
        covered_warning_ap = math.nan
        covered_unacceptable_ap = math.nan
        covered_boundary_high_conf = math.nan
    else:
        covered_macro_f1 = float(
            f1_score(
                covered["business_label"],
                covered[pred_col],
                labels=BUSINESS_LABELS,
                average="macro",
                zero_division=0,
            )
        )
        covered_balanced_accuracy = float(
            balanced_accuracy_score(covered["business_label"], covered[pred_col])
        )
        covered_warning_ap = binary_ap(
            covered["business_label"].eq("warning").astype(int),
            covered[warning_prob_col],
        )
        covered_unacceptable_ap = binary_ap(
            covered["business_label"].eq("unacceptable").astype(int),
            covered[unacceptable_prob_col],
        )
        if boundary_covered.empty:
            covered_boundary_high_conf = math.nan
        else:
            covered_boundary_high_conf = boundary_overconfidence_stats(
                boundary_covered,
                pred_col=pred_col,
                acceptable_prob_col=acceptable_prob_col,
                unacceptable_prob_col=unacceptable_prob_col,
            )["high_confidence_non_warning_rate"]

    unacceptable_covered = covered.loc[covered["business_label"].eq("unacceptable")]
    if unacceptable_covered.empty:
        non_rejected_unacceptable_miss_rate = None
    else:
        non_rejected_unacceptable_miss_rate = float(
            (unacceptable_covered[pred_col] != "unacceptable").mean()
        )

    boundary_reject_rate = float(boundary_all[rejected_col].astype(bool).mean()) if not boundary_all.empty else math.nan

    return {
        "reject_rate": reject_rate,
        "decision_coverage": coverage,
        "covered_macro_f1": covered_macro_f1,
        "covered_balanced_accuracy": covered_balanced_accuracy,
        "covered_warning_average_precision": covered_warning_ap,
        "covered_unacceptable_average_precision": covered_unacceptable_ap,
        "non_rejected_unacceptable_miss_rate": non_rejected_unacceptable_miss_rate,
        "boundary_reject_rate": boundary_reject_rate,
        "boundary_high_confidence_non_warning_rate": covered_boundary_high_conf,
    }


def aggregate_treatment_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    result: dict[str, object] = {}
    for column in frame.columns:
        if column == "fold":
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        result[column] = float(series.mean()) if series.notna().any() else None
    return result


def select_reject_candidate_on_train(
    train_frame: pd.DataFrame,
    *,
    selected_features: list[str],
    config: dict[str, object],
    candidates: list[dict[str, object]],
    eps: float,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, float]]:
    inner_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["inner_n_splits"]))
    baseline_rows: list[dict[str, float]] = []
    candidate_rows: list[dict[str, object]] = []

    for candidate in candidates:
        inner_rows: list[dict[str, object]] = []
        for inner_train_index, inner_val_index in inner_tscv.split(train_frame):
            inner_train = train_frame.iloc[inner_train_index].copy()
            inner_val = train_frame.iloc[inner_val_index].copy()
            model = fit_distribution_model(
                inner_train,
                features=selected_features,
                config=config,
                bin_labels=BIN_LABELS,
            )
            bin_prob = predict_distribution_probabilities(
                model,
                inner_val,
                features=selected_features,
                bin_labels=BIN_LABELS,
            )
            business_prob = business_probabilities_from_bins(bin_prob)
            base_pred = probability_argmax(business_prob, BUSINESS_LABELS)

            if candidate["candidate_name"] == "default_no_reject":
                baseline_rows.append(
                    {
                        "macro_f1": float(
                            f1_score(
                                inner_val["business_label"],
                                base_pred,
                                labels=BUSINESS_LABELS,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(inner_val["business_label"], base_pred)
                        ),
                    }
                )

            pred, rejected = apply_reject_rule(
                bin_prob,
                business_prob,
                candidate=candidate,
                eps=eps,
            )
            eval_frame = inner_val.copy()
            eval_frame["treatment_pred"] = pred
            eval_frame["treatment_rejected"] = rejected
            eval_frame["treatment_business_prob_acceptable"] = business_prob["acceptable"]
            eval_frame["treatment_business_prob_warning"] = business_prob["warning"]
            eval_frame["treatment_business_prob_unacceptable"] = business_prob["unacceptable"]
            metrics = evaluate_treatment_predictions(
                eval_frame,
                pred_col="treatment_pred",
                rejected_col="treatment_rejected",
                acceptable_prob_col="treatment_business_prob_acceptable",
                warning_prob_col="treatment_business_prob_warning",
                unacceptable_prob_col="treatment_business_prob_unacceptable",
            )
            inner_rows.append(metrics)

        aggregate = aggregate_treatment_metrics(inner_rows)
        candidate_rows.append({**candidate, **aggregate})

    baseline_metrics = {
        "macro_f1": float(pd.DataFrame(baseline_rows)["macro_f1"].mean()),
        "balanced_accuracy": float(pd.DataFrame(baseline_rows)["balanced_accuracy"].mean()),
    }
    candidate_frame = pd.DataFrame(candidate_rows).sort_values("candidate_name").reset_index(drop=True)
    reject_cfg = config["reject"]
    eligible = candidate_frame[
        (pd.to_numeric(candidate_frame["reject_rate"], errors="coerce") <= float(reject_cfg["max_reject_rate"]))
        & (pd.to_numeric(candidate_frame["covered_macro_f1"], errors="coerce") >= baseline_metrics["macro_f1"] - float(reject_cfg["inner_macro_f1_tolerance"]))
        & (pd.to_numeric(candidate_frame["covered_balanced_accuracy"], errors="coerce") >= baseline_metrics["balanced_accuracy"] - float(reject_cfg["inner_balanced_accuracy_tolerance"]))
    ].copy()
    if eligible.empty:
        eligible = candidate_frame.loc[candidate_frame["candidate_name"] == "default_no_reject"].copy()

    for column in ["non_rejected_unacceptable_miss_rate", "boundary_high_confidence_non_warning_rate"]:
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    chosen = eligible.sort_values(
        [
            "boundary_high_confidence_non_warning_rate",
            "non_rejected_unacceptable_miss_rate",
            "covered_macro_f1",
            "reject_rate",
        ],
        ascending=[True, True, False, True],
        na_position="last",
    ).iloc[0]
    return chosen.to_dict(), candidate_frame, baseline_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first simple reject-option treatment on top of the distributional branch.")
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
    eps = float(config["distributional"]["entropy_log_epsilon"])

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
    feature_rows = build_soft_distribution_labels(feature_rows, config)
    feature_cols = [
        column
        for column in feature_columns(feature_rows)
        if not column.startswith("soft_target_") and column not in {"hard_bin_label", "soft_label_zone"}
    ]

    reference_candidates = build_candidate_grid(config)
    reference_thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]
    reject_candidates = build_reject_candidates(config)

    feature_rows_csv = artifacts_dir / "distributional_reject_feature_rows.csv"
    results_csv = artifacts_dir / "distributional_reject_results.csv"
    summary_json = artifacts_dir / "distributional_reject_summary.json"
    per_fold_csv = artifacts_dir / "distributional_reject_per_fold.csv"
    reference_candidate_csv = artifacts_dir / "distributional_reject_reference_candidate_per_fold.csv"
    reject_candidate_csv = artifacts_dir / "distributional_reject_selected_candidate_per_fold.csv"
    reject_inner_search_csv = artifacts_dir / "distributional_reject_inner_search.csv"
    alias_csv = artifacts_dir / "sensor_identity_alias_pairs.csv"
    summary_md = reports_dir / "distributional_reject_summary.md"

    outer_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    predictions = feature_rows[
        [
            "sample_time",
            "t90",
            "business_label",
            "boundary_low_flag",
            "boundary_high_flag",
            "boundary_any_flag",
            "is_unacceptable",
            "hard_bin_label",
            "soft_label_zone",
        ]
        + [f"soft_target_{label}" for label in BIN_LABELS]
    ].copy()
    for label in BIN_LABELS:
        predictions[f"distributional_bin_prob_{label}"] = pd.NA
    for label in BUSINESS_LABELS:
        predictions[f"distributional_business_prob_{label}"] = pd.NA
    predictions["distributional_business_pred"] = pd.NA
    predictions["treatment_business_pred"] = pd.NA
    predictions["treatment_rejected"] = pd.NA
    predictions["selected_reject_candidate"] = pd.NA

    per_fold_rows: list[dict[str, object]] = []
    reference_candidate_rows: list[dict[str, object]] = []
    reject_selected_rows: list[dict[str, object]] = []
    reject_inner_search_rows: list[dict[str, object]] = []
    distributional_eval_rows: list[dict[str, object]] = []
    treatment_eval_rows: list[dict[str, object]] = []
    distributional_distribution_rows: list[dict[str, float]] = []

    for fold_index, (train_index, test_index) in enumerate(outer_tscv.split(feature_rows), start=1):
        train_frame = feature_rows.iloc[train_index].copy()
        test_frame = feature_rows.iloc[test_index].copy()

        chosen_reference_candidate, _ = select_reference_candidate_on_train(
            train_frame,
            feature_cols=feature_cols,
            thresholds=reference_thresholds,
            config=config,
            candidates=reference_candidates,
        )
        reference_candidate_rows.append({"fold": fold_index, **chosen_reference_candidate})

        selected_sensors, selected_scores = select_topk_sensors(
            train_frame,
            feature_cols,
            topk=int(config["screening"]["topk_sensors"]),
        )
        selected_features = selected_feature_columns(feature_cols, selected_sensors)

        chosen_reject_candidate, reject_search_frame, baseline_b_inner = select_reject_candidate_on_train(
            train_frame,
            selected_features=selected_features,
            config=config,
            candidates=reject_candidates,
            eps=eps,
        )
        reject_search_frame.insert(0, "fold", fold_index)
        reject_inner_search_rows.extend(reject_search_frame.to_dict(orient="records"))
        reject_selected_rows.append({"fold": fold_index, **chosen_reject_candidate, **baseline_b_inner})

        _reference_cumulative, _reference_business_prob = fit_cumulative_business_probabilities_inner_thresholds_only(
            train_frame,
            test_frame,
            features=selected_features,
            thresholds=reference_thresholds,
            config=config,
            candidate=chosen_reference_candidate,
        )
        distribution_model = fit_distribution_model(
            train_frame,
            features=selected_features,
            config=config,
            bin_labels=BIN_LABELS,
        )
        distributional_bin_prob = predict_distribution_probabilities(
            distribution_model,
            test_frame,
            features=selected_features,
            bin_labels=BIN_LABELS,
        )
        distributional_business_prob = business_probabilities_from_bins(distributional_bin_prob)
        distributional_pred = probability_argmax(distributional_business_prob, BUSINESS_LABELS)
        treatment_pred, treatment_rejected = apply_reject_rule(
            distributional_bin_prob,
            distributional_business_prob,
            candidate=chosen_reject_candidate,
            eps=eps,
        )

        for label in BIN_LABELS:
            predictions.loc[test_frame.index, f"distributional_bin_prob_{label}"] = distributional_bin_prob[label]
        for label in BUSINESS_LABELS:
            predictions.loc[test_frame.index, f"distributional_business_prob_{label}"] = distributional_business_prob[label]
        predictions.loc[test_frame.index, "distributional_business_pred"] = distributional_pred
        predictions.loc[test_frame.index, "treatment_business_pred"] = treatment_pred
        predictions.loc[test_frame.index, "treatment_rejected"] = treatment_rejected
        predictions.loc[test_frame.index, "selected_reject_candidate"] = str(chosen_reject_candidate["candidate_name"])

        distributional_metrics = evaluate_prediction_set(test_frame, distributional_business_prob, distributional_pred)
        distributional_eval_rows.append(
            {
                "fold": fold_index,
                "macro_f1": distributional_metrics["macro_f1"],
                "balanced_accuracy": distributional_metrics["balanced_accuracy"],
                "core_qualified_average_precision": distributional_metrics["core_qualified_average_precision"],
                "boundary_warning_average_precision": distributional_metrics["boundary_warning_average_precision"],
                "clearly_unacceptable_average_precision": distributional_metrics["clearly_unacceptable_average_precision"],
                "boundary_high_confidence_non_warning_rate": distributional_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )

        treatment_frame = test_frame.copy()
        treatment_frame["treatment_business_pred"] = treatment_pred
        treatment_frame["treatment_rejected"] = treatment_rejected
        treatment_frame["distributional_business_prob_acceptable"] = distributional_business_prob["acceptable"]
        treatment_frame["distributional_business_prob_warning"] = distributional_business_prob["warning"]
        treatment_frame["distributional_business_prob_unacceptable"] = distributional_business_prob["unacceptable"]
        treatment_metrics = evaluate_treatment_predictions(
            treatment_frame,
            pred_col="treatment_business_pred",
            rejected_col="treatment_rejected",
            acceptable_prob_col="distributional_business_prob_acceptable",
            warning_prob_col="distributional_business_prob_warning",
            unacceptable_prob_col="distributional_business_prob_unacceptable",
        )
        treatment_eval_rows.append({"fold": fold_index, **treatment_metrics})

        distributional_distribution_rows.append(
            {
                "fold": fold_index,
                "multiclass_brier_score": multiclass_brier_score(distributional_bin_prob, test_frame, BIN_LABELS),
                "negative_log_loss": multiclass_log_loss(distributional_bin_prob, test_frame, labels=BIN_LABELS, eps=eps),
                "normalized_entropy_mean": entropy_summary(distributional_bin_prob, labels=BIN_LABELS, eps=eps)["mean"],
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
                "reference_candidate_name": chosen_reference_candidate["candidate_name"],
                "selected_reject_candidate_name": chosen_reject_candidate["candidate_name"],
                "distributional_macro_f1": distributional_metrics["macro_f1"],
                "distributional_balanced_accuracy": distributional_metrics["balanced_accuracy"],
                "treatment_reject_rate": treatment_metrics["reject_rate"],
                "treatment_coverage": treatment_metrics["decision_coverage"],
                "treatment_covered_macro_f1": treatment_metrics["covered_macro_f1"],
                "treatment_covered_balanced_accuracy": treatment_metrics["covered_balanced_accuracy"],
                "treatment_boundary_reject_rate": treatment_metrics["boundary_reject_rate"],
                "treatment_boundary_high_confidence_non_warning_rate": treatment_metrics["boundary_high_confidence_non_warning_rate"],
            }
        )

    for label in BIN_LABELS:
        predictions[f"distributional_bin_prob_{label}"] = pd.to_numeric(predictions[f"distributional_bin_prob_{label}"], errors="coerce")
    for label in BUSINESS_LABELS:
        predictions[f"distributional_business_prob_{label}"] = pd.to_numeric(predictions[f"distributional_business_prob_{label}"], errors="coerce")

    scored_predictions = predictions.loc[predictions["distributional_business_pred"].notna()].copy()
    treatment_scored = scored_predictions.copy()
    treatment_scored["treatment_rejected"] = treatment_scored["treatment_rejected"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    distributional_summary = evaluate_predictions(
        scored_predictions,
        pred_col="distributional_business_pred",
        acceptable_prob_col="distributional_business_prob_acceptable",
        warning_prob_col="distributional_business_prob_warning",
        unacceptable_prob_col="distributional_business_prob_unacceptable",
    )
    treatment_summary = evaluate_treatment_predictions(
        treatment_scored,
        pred_col="treatment_business_pred",
        rejected_col="treatment_rejected",
        acceptable_prob_col="distributional_business_prob_acceptable",
        warning_prob_col="distributional_business_prob_warning",
        unacceptable_prob_col="distributional_business_prob_unacceptable",
    )

    distributional_distribution_summary = {
        "multiclass_brier_score": multiclass_brier_score(
            scored_predictions.rename(columns={f"distributional_bin_prob_{label}": label for label in BIN_LABELS}),
            scored_predictions,
            BIN_LABELS,
        ),
        "negative_log_loss": multiclass_log_loss(
            scored_predictions.rename(columns={f"distributional_bin_prob_{label}": label for label in BIN_LABELS}),
            scored_predictions,
            labels=BIN_LABELS,
            eps=eps,
        ),
        "entropy": entropy_summary(
            scored_predictions.rename(columns={f"distributional_bin_prob_{label}": label for label in BIN_LABELS}),
            labels=BIN_LABELS,
            eps=eps,
        ),
        "calibration_error": None,
    }

    low_boundary_mask = treatment_scored["t90"].between(7.9, 8.3, inclusive="both")
    high_boundary_mask = treatment_scored["t90"].between(8.6, 8.8, inclusive="both")
    treatment_boundary_diagnostics = {
        "7_9_8_3": boundary_subset_stats(
            treatment_scored.loc[low_boundary_mask & ~treatment_scored["treatment_rejected"]],
            pred_col="treatment_business_pred",
            acceptable_prob_col="distributional_business_prob_acceptable",
            unacceptable_prob_col="distributional_business_prob_unacceptable",
        ),
        "8_6_8_8": boundary_subset_stats(
            treatment_scored.loc[high_boundary_mask & ~treatment_scored["treatment_rejected"]],
            pred_col="treatment_business_pred",
            acceptable_prob_col="distributional_business_prob_acceptable",
            unacceptable_prob_col="distributional_business_prob_unacceptable",
        ),
    }

    distributional_fold_mean = aggregate_metrics([{k: v for k, v in row.items() if k != "fold"} for row in distributional_eval_rows])
    treatment_fold_mean = aggregate_treatment_metrics([{k: v for k, v in row.items() if k != "fold"} for row in treatment_eval_rows])
    distributional_distribution_fold_mean = {
        "multiclass_brier_score": float(pd.DataFrame(distributional_distribution_rows)["multiclass_brier_score"].mean()),
        "negative_log_loss": float(pd.DataFrame(distributional_distribution_rows)["negative_log_loss"].mean()),
        "normalized_entropy_mean": float(pd.DataFrame(distributional_distribution_rows)["normalized_entropy_mean"].mean()),
    }

    improvement = {
        "covered_macro_f1_minus_distributional_macro_f1": (
            treatment_summary["covered_macro_f1"] - distributional_summary["macro_f1"]
            if treatment_summary["covered_macro_f1"] is not None
            else None
        ),
        "covered_balanced_accuracy_minus_distributional_balanced_accuracy": (
            treatment_summary["covered_balanced_accuracy"] - distributional_summary["balanced_accuracy"]
            if treatment_summary["covered_balanced_accuracy"] is not None
            else None
        ),
        "boundary_high_confidence_non_warning_rate_delta": (
            treatment_summary["boundary_high_confidence_non_warning_rate"] - distributional_summary["boundary_overconfidence"]["high_confidence_non_warning_rate"]
            if treatment_summary["boundary_high_confidence_non_warning_rate"] is not None
            else None
        ),
    }

    feature_rows.to_csv(feature_rows_csv, index=False, encoding="utf-8-sig")
    scored_predictions.to_csv(results_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_fold_rows).to_csv(per_fold_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_candidate_rows).to_csv(reference_candidate_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(reject_selected_rows).to_csv(reject_candidate_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(reject_inner_search_rows).to_csv(reject_inner_search_csv, index=False, encoding="utf-8-sig")
    alias_frame.to_csv(alias_csv, index=False, encoding="utf-8-sig")

    summary = {
        "experiment_name": config["experiment_name"],
        "branch_stage": config["branch_stage"],
        "config_path": str(config_path),
        "data_paths": {
            "dcs_main_path": str(main_dcs_path),
            "dcs_supplemental_path": str(supplemental_path),
            "lims_path": str(lims_path),
        },
        "dcs_audit": dcs_audit,
        "identity_alias_pairs": alias_frame.to_dict(orient="records"),
        "data_summary": {
            "aligned_rows": int(len(feature_rows)),
            "scored_rows": int(len(scored_predictions)),
            "n_splits_used": int(config["validation"]["n_splits"]),
            "inner_n_splits_used": int(config["validation"]["inner_n_splits"]),
        },
        "preclean_summary": preclean_summary,
        "distributional_summary": distributional_summary,
        "distributional_distribution_summary": distributional_distribution_summary,
        "treatment_summary": treatment_summary,
        "distributional_fold_mean": distributional_fold_mean,
        "distributional_distribution_fold_mean": distributional_distribution_fold_mean,
        "treatment_fold_mean": treatment_fold_mean,
        "boundary_diagnostics": treatment_boundary_diagnostics,
        "improvement_summary": improvement,
        "per_fold": per_fold_rows,
        "reference_candidate_rows": reference_candidate_rows,
        "reject_selected_rows": reject_selected_rows,
        "artifacts": {
            "feature_rows_csv": str(feature_rows_csv),
            "results_csv": str(results_csv),
            "per_fold_csv": str(per_fold_csv),
            "reference_candidate_csv": str(reference_candidate_csv),
            "reject_candidate_csv": str(reject_candidate_csv),
            "reject_inner_search_csv": str(reject_inner_search_csv),
            "alias_csv": str(alias_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    summary_json.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Distributional Reject Option Treatment-C Summary",
        "",
        "- Branch stage: treatment_c_simple_reject",
        "- Base model: same 5-bin distributional prediction as Baseline B",
        "- Decision layer: simple reject / retest using confidence + entropy",
        f"- Scored rows: {len(scored_predictions)}",
        f"- Distributional no-reject macro_f1: {distributional_summary['macro_f1']:.4f}",
        f"- Treatment coverage: {treatment_summary['decision_coverage']:.4f}",
        f"- Treatment reject_rate: {treatment_summary['reject_rate']:.4f}",
        f"- Treatment covered_macro_f1: {treatment_summary['covered_macro_f1']:.4f}",
        f"- Treatment covered_balanced_accuracy: {treatment_summary['covered_balanced_accuracy']:.4f}",
        f"- Treatment non-rejected unacceptable miss rate: {treatment_summary['non_rejected_unacceptable_miss_rate']}",
        f"- Treatment boundary_reject_rate: {treatment_summary['boundary_reject_rate']:.4f}",
        f"- Treatment boundary high-confidence non-warning: {treatment_summary['boundary_high_confidence_non_warning_rate']:.4f}",
        "",
        "## Selected Reject Candidate Per Fold",
        "",
    ]
    for row in reject_selected_rows:
        lines.append(
            f"- fold {row['fold']}: {row['candidate_name']} "
            f"(tau_conf={row['tau_conf']}, tau_entropy={row['tau_entropy']}, baseline_b_macro_f1={row['macro_f1']:.4f})"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
