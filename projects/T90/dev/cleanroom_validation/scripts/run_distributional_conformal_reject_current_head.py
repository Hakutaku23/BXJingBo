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

from run_distributional_reject_option_current_head import (
    BIN_LABELS,
    build_soft_distribution_labels,
    business_probabilities_from_bins,
    fit_distribution_model,
    json_ready,
    multiclass_brier_score,
    multiclass_log_loss,
    predict_distribution_probabilities,
)
from run_ordinal_cumulative_current_head import (
    build_feature_rows,
    discover_lims_path,
    feature_columns,
    load_lims_data,
    preclean_features,
    select_topk_sensors,
    selected_feature_columns,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_identity_dedup import (
    BUSINESS_LABELS,
    assign_labels,
    binary_ap,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup import (
    CLEANROOM_DIR,
    PROJECT_DIR,
)
from run_ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head import load_identity_deduped_combined_dcs


DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "distributional_conformal_reject_current_head.yaml"
RETEST_LABEL = "retest"
BIN_TO_BUSINESS = {
    "bin_1": "unacceptable",
    "bin_2": "warning",
    "bin_3": "acceptable",
    "bin_4": "warning",
    "bin_5": "unacceptable",
}


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_proper_train_and_calibration(train_frame: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    conformal_cfg = config["conformal"]
    fraction = float(conformal_cfg["calibration_fraction"])
    min_calibration_rows = int(conformal_cfg["min_calibration_rows"])
    min_proper_train_rows = int(conformal_cfg["min_proper_train_rows"])
    n_rows = len(train_frame)

    calibration_rows = max(min_calibration_rows, int(math.ceil(n_rows * fraction)))
    calibration_rows = min(calibration_rows, n_rows - min_proper_train_rows)
    if calibration_rows <= 0:
        raise ValueError("Training fold is too small for conformal calibration split.")

    split_index = n_rows - calibration_rows
    proper_train = train_frame.iloc[:split_index].copy()
    calibration = train_frame.iloc[split_index:].copy()
    if proper_train.empty or calibration.empty:
        raise ValueError("Invalid proper-train/calibration split.")
    return proper_train, calibration


def aps_scores(probabilities: pd.DataFrame, true_labels: pd.Series) -> np.ndarray:
    scores: list[float] = []
    matrix = probabilities[BIN_LABELS].to_numpy(dtype=float)
    labels = true_labels.astype(str).tolist()
    for row_prob, true_label in zip(matrix, labels):
        order = np.argsort(-row_prob)
        sorted_prob = row_prob[order]
        sorted_labels = [BIN_LABELS[idx] for idx in order]
        cumulative = np.cumsum(sorted_prob)
        true_position = sorted_labels.index(true_label)
        scores.append(float(cumulative[true_position]))
    return np.asarray(scores, dtype=float)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    rank = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(scores[rank])


def aps_prediction_sets(probabilities: pd.DataFrame, qhat: float) -> list[list[str]]:
    prediction_sets: list[list[str]] = []
    matrix = probabilities[BIN_LABELS].to_numpy(dtype=float)
    for row_prob in matrix:
        order = np.argsort(-row_prob)
        sorted_prob = row_prob[order]
        cumulative = np.cumsum(sorted_prob)
        cutoff = int(np.searchsorted(cumulative, qhat, side="left"))
        chosen = [BIN_LABELS[idx] for idx in order[: cutoff + 1]]
        prediction_sets.append(chosen)
    return prediction_sets


def business_set_from_bin_set(bin_set: list[str]) -> list[str]:
    return sorted(set(BIN_TO_BUSINESS[label] for label in bin_set))


def operational_decision_from_set(bin_set: list[str]) -> str:
    business_set = business_set_from_bin_set(bin_set)
    if len(business_set) == 1:
        return business_set[0]
    return RETEST_LABEL


def set_metrics(
    frame: pd.DataFrame,
    *,
    prediction_sets: list[list[str]],
    operational_decisions: list[str],
    business_probabilities: pd.DataFrame,
) -> dict[str, object]:
    eval_frame = frame.copy()
    eval_frame["prediction_set"] = ["|".join(item) for item in prediction_sets]
    eval_frame["set_size"] = [len(item) for item in prediction_sets]
    eval_frame["business_set"] = ["|".join(business_set_from_bin_set(item)) for item in prediction_sets]
    eval_frame["business_set_size"] = [len(business_set_from_bin_set(item)) for item in prediction_sets]
    eval_frame["operational_decision"] = operational_decisions
    eval_frame["rejected"] = eval_frame["operational_decision"].eq(RETEST_LABEL)
    eval_frame["business_prob_acceptable"] = business_probabilities["acceptable"].to_numpy(dtype=float)
    eval_frame["business_prob_warning"] = business_probabilities["warning"].to_numpy(dtype=float)
    eval_frame["business_prob_unacceptable"] = business_probabilities["unacceptable"].to_numpy(dtype=float)

    set_contains_true = [
        eval_frame.iloc[idx]["hard_bin_label"] in prediction_sets[idx]
        for idx in range(len(eval_frame))
    ]
    eval_frame["set_contains_true"] = set_contains_true

    covered = eval_frame.loc[~eval_frame["rejected"]].copy()
    non_boundary = eval_frame.loc[~eval_frame["boundary_any_flag"]].copy()
    boundary = eval_frame.loc[eval_frame["boundary_any_flag"]].copy()

    if covered.empty:
        covered_macro_f1 = math.nan
        covered_balanced_accuracy = math.nan
        covered_warning_ap = math.nan
        covered_unacceptable_ap = math.nan
    else:
        covered_macro_f1 = float(
            f1_score(
                covered["business_label"],
                covered["operational_decision"],
                labels=BUSINESS_LABELS,
                average="macro",
                zero_division=0,
            )
        )
        covered_balanced_accuracy = float(
            balanced_accuracy_score(covered["business_label"], covered["operational_decision"])
        )
        covered_warning_ap = binary_ap(
            covered["business_label"].eq("warning").astype(int),
            covered["business_prob_warning"],
        )
        covered_unacceptable_ap = binary_ap(
            covered["business_label"].eq("unacceptable").astype(int),
            covered["business_prob_unacceptable"],
        )

    return {
        "hard_bin_coverage": float(eval_frame["set_contains_true"].mean()),
        "mean_set_size": float(eval_frame["set_size"].mean()),
        "p90_set_size": float(eval_frame["set_size"].quantile(0.90)),
        "singleton_rate": float(eval_frame["set_size"].eq(1).mean()),
        "ambiguous_business_set_rate": float(eval_frame["business_set_size"].gt(1).mean()),
        "retest_rate": float(eval_frame["rejected"].mean()),
        "decision_coverage": float((~eval_frame["rejected"]).mean()),
        "boundary_retest_rate": float(boundary["rejected"].mean()) if not boundary.empty else math.nan,
        "non_boundary_decision_coverage": float((~non_boundary["rejected"]).mean()) if not non_boundary.empty else math.nan,
        "boundary_mean_set_size": float(boundary["set_size"].mean()) if not boundary.empty else math.nan,
        "covered_macro_f1": covered_macro_f1,
        "covered_balanced_accuracy": covered_balanced_accuracy,
        "covered_warning_average_precision": covered_warning_ap,
        "covered_unacceptable_average_precision": covered_unacceptable_ap,
    }


def select_alpha_on_calibration(
    calibration_frame: pd.DataFrame,
    calibration_probabilities: pd.DataFrame,
    business_probabilities: pd.DataFrame,
    config: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    conformal_cfg = config["conformal"]
    candidate_rows: list[dict[str, object]] = []
    for alpha in [float(item) for item in conformal_cfg["alpha_grid"]]:
        scores = aps_scores(calibration_probabilities, calibration_frame["hard_bin_label"])
        qhat = conformal_quantile(scores, alpha)
        prediction_sets = aps_prediction_sets(calibration_probabilities, qhat)
        operational_decisions = [operational_decision_from_set(item) for item in prediction_sets]
        metrics = set_metrics(
            calibration_frame,
            prediction_sets=prediction_sets,
            operational_decisions=operational_decisions,
            business_probabilities=business_probabilities,
        )
        boundary_focus = metrics["boundary_retest_rate"] - metrics["retest_rate"]
        candidate_rows.append(
            {
                "alpha": alpha,
                "qhat": qhat,
                "nominal_coverage": 1.0 - alpha,
                "boundary_focus": boundary_focus,
                **metrics,
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows).sort_values("alpha").reset_index(drop=True)
    coverage_slack = float(conformal_cfg["coverage_slack"])
    min_decision_coverage = float(conformal_cfg["min_decision_coverage"])
    eligible = candidate_frame.loc[
        (candidate_frame["hard_bin_coverage"] >= candidate_frame["nominal_coverage"] - coverage_slack)
        & (candidate_frame["decision_coverage"] >= min_decision_coverage)
    ].copy()
    if eligible.empty:
        eligible = candidate_frame.copy()
    chosen = eligible.sort_values(
        ["boundary_focus", "hard_bin_coverage", "mean_set_size", "decision_coverage"],
        ascending=[False, False, True, False],
    ).iloc[0]
    return chosen.to_dict(), candidate_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run soft 5-bin distributional prediction with split conformal / quantile reject option."
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

    feature_rows_csv = artifacts_dir / "distributional_conformal_feature_rows.csv"
    results_csv = artifacts_dir / "distributional_conformal_results.csv"
    summary_json = artifacts_dir / "distributional_conformal_summary.json"
    per_fold_csv = artifacts_dir / "distributional_conformal_per_fold.csv"
    alpha_search_csv = artifacts_dir / "distributional_conformal_alpha_search.csv"
    alias_csv = artifacts_dir / "sensor_identity_alias_pairs.csv"
    summary_md = reports_dir / "distributional_conformal_summary.md"

    predictions = feature_rows[
        [
            "sample_time",
            "t90",
            "business_label",
            "boundary_low_flag",
            "boundary_high_flag",
            "boundary_any_flag",
            "hard_bin_label",
            "soft_label_zone",
        ]
        + [f"soft_target_{label}" for label in BIN_LABELS]
    ].copy()
    for label in BIN_LABELS:
        predictions[f"bin_prob_{label}"] = pd.NA
    predictions["prediction_set"] = pd.NA
    predictions["business_set"] = pd.NA
    predictions["operational_decision"] = pd.NA
    predictions["rejected"] = pd.NA
    predictions["selected_alpha"] = pd.NA
    predictions["selected_qhat"] = pd.NA

    outer_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    per_fold_rows: list[dict[str, object]] = []
    alpha_search_rows: list[dict[str, object]] = []

    for fold_index, (train_index, test_index) in enumerate(outer_tscv.split(feature_rows), start=1):
        train_frame = feature_rows.iloc[train_index].copy()
        test_frame = feature_rows.iloc[test_index].copy()

        selected_sensors, selected_scores = select_topk_sensors(
            train_frame,
            feature_cols,
            topk=int(config["screening"]["topk_sensors"]),
        )
        selected_features = selected_feature_columns(feature_cols, selected_sensors)

        proper_train, calibration = split_proper_train_and_calibration(train_frame, config)
        model = fit_distribution_model(
            proper_train,
            features=selected_features,
            config=config,
            bin_labels=BIN_LABELS,
        )

        calibration_probabilities = predict_distribution_probabilities(
            model,
            calibration,
            features=selected_features,
            bin_labels=BIN_LABELS,
        )
        calibration_business_probabilities = business_probabilities_from_bins(calibration_probabilities)
        chosen_alpha, alpha_frame = select_alpha_on_calibration(
            calibration,
            calibration_probabilities,
            calibration_business_probabilities,
            config,
        )
        alpha_frame.insert(0, "fold", fold_index)
        alpha_search_rows.extend(alpha_frame.to_dict(orient="records"))

        test_probabilities = predict_distribution_probabilities(
            model,
            test_frame,
            features=selected_features,
            bin_labels=BIN_LABELS,
        )
        test_business_probabilities = business_probabilities_from_bins(test_probabilities)
        test_prediction_sets = aps_prediction_sets(test_probabilities, float(chosen_alpha["qhat"]))
        test_operational_decisions = [operational_decision_from_set(item) for item in test_prediction_sets]
        fold_metrics = set_metrics(
            test_frame,
            prediction_sets=test_prediction_sets,
            operational_decisions=test_operational_decisions,
            business_probabilities=test_business_probabilities,
        )

        for label in BIN_LABELS:
            predictions.loc[test_frame.index, f"bin_prob_{label}"] = test_probabilities[label]
        predictions.loc[test_frame.index, "prediction_set"] = ["|".join(item) for item in test_prediction_sets]
        predictions.loc[test_frame.index, "business_set"] = [
            "|".join(business_set_from_bin_set(item)) for item in test_prediction_sets
        ]
        predictions.loc[test_frame.index, "operational_decision"] = test_operational_decisions
        predictions.loc[test_frame.index, "rejected"] = [item == RETEST_LABEL for item in test_operational_decisions]
        predictions.loc[test_frame.index, "selected_alpha"] = float(chosen_alpha["alpha"])
        predictions.loc[test_frame.index, "selected_qhat"] = float(chosen_alpha["qhat"])

        per_fold_rows.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_frame)),
                "proper_train_rows": int(len(proper_train)),
                "calibration_rows": int(len(calibration)),
                "test_rows": int(len(test_frame)),
                "selected_sensor_count": int(len(selected_sensors)),
                "selected_sensors": selected_sensors,
                "selected_sensor_scores": selected_scores,
                "selected_alpha": float(chosen_alpha["alpha"]),
                "selected_qhat": float(chosen_alpha["qhat"]),
                **fold_metrics,
            }
        )

    for label in BIN_LABELS:
        predictions[f"bin_prob_{label}"] = pd.to_numeric(predictions[f"bin_prob_{label}"], errors="coerce")
    predictions["rejected"] = predictions["rejected"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    scored_predictions = predictions.loc[predictions["operational_decision"].notna()].copy()
    business_probabilities = scored_predictions.rename(
        columns={f"bin_prob_{label}": label for label in BIN_LABELS}
    )
    business_probabilities = business_probabilities[BIN_LABELS]
    business_probabilities = business_probabilities_from_bins(business_probabilities)

    final_metrics = set_metrics(
        scored_predictions,
        prediction_sets=[str(item).split("|") for item in scored_predictions["prediction_set"]],
        operational_decisions=scored_predictions["operational_decision"].astype(str).tolist(),
        business_probabilities=business_probabilities,
    )

    distribution_quality = {
        "multiclass_brier_score": multiclass_brier_score(
            scored_predictions.rename(columns={f"bin_prob_{label}": label for label in BIN_LABELS}),
            scored_predictions,
            BIN_LABELS,
        ),
        "negative_log_loss": multiclass_log_loss(
            scored_predictions.rename(columns={f"bin_prob_{label}": label for label in BIN_LABELS}),
            scored_predictions,
            labels=BIN_LABELS,
            eps=eps,
        ),
    }

    per_fold_frame = pd.DataFrame(per_fold_rows)
    fold_mean = {
        column: float(pd.to_numeric(per_fold_frame[column], errors="coerce").mean())
        for column in per_fold_frame.columns
        if column not in {"fold", "selected_sensors", "selected_sensor_scores"}
        and pd.to_numeric(per_fold_frame[column], errors="coerce").notna().any()
    }

    feature_rows.to_csv(feature_rows_csv, index=False, encoding="utf-8-sig")
    scored_predictions.to_csv(results_csv, index=False, encoding="utf-8-sig")
    per_fold_frame.to_csv(per_fold_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(alpha_search_rows).to_csv(alpha_search_csv, index=False, encoding="utf-8-sig")
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
        "soft_label_design": {
            "bin_labels": BIN_LABELS,
            "bin_edges": config["distributional"]["bin_edges"],
            "soft_inner_thresholds": config["distributional"]["soft_inner_thresholds"],
            "soft_radius": config["distributional"]["soft_radius"],
            "soft_zone_counts": feature_rows["soft_label_zone"].value_counts().to_dict(),
        },
        "conformal_design": {
            "alpha_grid": config["conformal"]["alpha_grid"],
            "calibration_fraction": config["conformal"]["calibration_fraction"],
            "min_calibration_rows": config["conformal"]["min_calibration_rows"],
            "decision_rule": "unique_business_set_else_retest",
            "method": "split_conformal_aps_quantile",
        },
        "data_summary": {
            "aligned_rows": int(len(feature_rows)),
            "scored_rows": int(len(scored_predictions)),
            "n_splits_used": int(config["validation"]["n_splits"]),
        },
        "preclean_summary": preclean_summary,
        "primary_summary": final_metrics,
        "distribution_quality": distribution_quality,
        "fold_mean": fold_mean,
        "per_fold": per_fold_rows,
        "artifacts": {
            "feature_rows_csv": str(feature_rows_csv),
            "results_csv": str(results_csv),
            "per_fold_csv": str(per_fold_csv),
            "alpha_search_csv": str(alpha_search_csv),
            "alias_csv": str(alias_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    summary_json.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Distributional Conformal Reject Summary",
        "",
        "- Branch stage: conformal_quantile_reject_first_pass",
        "- Target: soft 5-bin distribution + split conformal APS quantile calibration + reject",
        f"- Scored rows: {len(scored_predictions)}",
        f"- Hard-bin set coverage: {final_metrics['hard_bin_coverage']:.4f}",
        f"- Mean set size: {final_metrics['mean_set_size']:.4f}",
        f"- Singleton rate: {final_metrics['singleton_rate']:.4f}",
        f"- Ambiguous-business-set rate: {final_metrics['ambiguous_business_set_rate']:.4f}",
        f"- Retest rate: {final_metrics['retest_rate']:.4f}",
        f"- Boundary retest rate: {final_metrics['boundary_retest_rate']:.4f}",
        f"- Non-boundary decision coverage: {final_metrics['non_boundary_decision_coverage']:.4f}",
        f"- Covered macro_f1: {final_metrics['covered_macro_f1']:.4f}",
        f"- Covered balanced_accuracy: {final_metrics['covered_balanced_accuracy']:.4f}",
        "",
        "## Selected Alpha Per Fold",
        "",
    ]
    for row in per_fold_rows:
        lines.append(
            f"- fold {row['fold']}: alpha={row['selected_alpha']:.2f}, qhat={row['selected_qhat']:.4f}, "
            f"coverage={row['hard_bin_coverage']:.4f}, retest={row['retest_rate']:.4f}"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

