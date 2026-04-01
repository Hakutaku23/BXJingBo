from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from run_ordinal_cumulative_current_head import (
    build_feature_rows,
    build_logistic_pipeline,
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


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "distributional_reject_option_current_head.yaml"
BIN_LABELS = ["bin_1", "bin_2", "bin_3", "bin_4", "bin_5"]


class ConstantMulticlassModel:
    def __init__(self, classes: list[str], probability: dict[str, float]) -> None:
        self.classes_ = np.array(classes, dtype=object)
        self._probability = probability

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rows = len(X)
        matrix = np.zeros((rows, len(self.classes_)), dtype=float)
        for idx, label in enumerate(self.classes_):
            matrix[:, idx] = float(self._probability.get(str(label), 0.0))
        return matrix


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def hard_bin_label(value: float, edges: list[float]) -> str:
    if value < edges[0]:
        return "bin_1"
    if value < edges[1]:
        return "bin_2"
    if value < edges[2]:
        return "bin_3"
    if value < edges[3]:
        return "bin_4"
    return "bin_5"


def local_soft_distribution(value: float, threshold: float, radius: float) -> tuple[float, float] | None:
    left = threshold - radius
    right = threshold + radius
    if value < left or value > right:
        return None
    right_mass = (value - left) / (2.0 * radius)
    right_mass = float(np.clip(right_mass, 0.0, 1.0))
    return 1.0 - right_mass, right_mass


def build_soft_distribution_labels(frame: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    dist_cfg = config["distributional"]
    edges = [float(item) for item in dist_cfg["bin_edges"]]
    thresholds = [float(item) for item in dist_cfg["soft_inner_thresholds"]]
    radius = float(dist_cfg["soft_radius"])

    frame = frame.copy()
    for label in BIN_LABELS:
        frame[f"soft_target_{label}"] = 0.0

    hard_labels: list[str] = []
    soft_zone_tags: list[str] = []

    for index, value in frame["t90"].astype(float).items():
        masses = {label: 0.0 for label in BIN_LABELS}
        zone = "hard"
        around_82 = local_soft_distribution(value, thresholds[0], radius)
        around_87 = local_soft_distribution(value, thresholds[1], radius)
        if around_82 is not None:
            masses["bin_2"], masses["bin_3"] = around_82
            zone = "soft_8_2"
        elif around_87 is not None:
            masses["bin_3"], masses["bin_4"] = around_87
            zone = "soft_8_7"
        else:
            masses[hard_bin_label(value, edges)] = 1.0
        for label in BIN_LABELS:
            frame.at[index, f"soft_target_{label}"] = masses[label]
        hard_labels.append(hard_bin_label(value, edges))
        soft_zone_tags.append(zone)

    frame["hard_bin_label"] = hard_labels
    frame["soft_label_zone"] = soft_zone_tags
    return frame


def expand_soft_targets_for_training(
    train_frame: pd.DataFrame,
    *,
    features: list[str],
    bin_labels: list[str],
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    expanded_frames: list[pd.DataFrame] = []
    labels: list[str] = []
    weights: list[float] = []

    for label in bin_labels:
        weight_col = f"soft_target_{label}"
        subset = train_frame.loc[train_frame[weight_col] > 0, features + [weight_col]].copy()
        if subset.empty:
            continue
        expanded_frames.append(subset[features].copy())
        labels.extend([label] * len(subset))
        weights.extend(subset[weight_col].astype(float).tolist())

    if not expanded_frames:
        raise ValueError("No non-zero soft labels available for training expansion.")

    expanded = pd.concat(expanded_frames, ignore_index=True)
    return expanded, pd.Series(labels, dtype=str), np.asarray(weights, dtype=float)


def fit_distribution_model(
    train_frame: pd.DataFrame,
    *,
    features: list[str],
    config: dict[str, object],
    bin_labels: list[str],
) -> object:
    expanded_X, expanded_y, sample_weight = expand_soft_targets_for_training(
        train_frame,
        features=features,
        bin_labels=bin_labels,
    )
    class_counts = expanded_y.value_counts()
    if len(class_counts) == 1:
        only_label = str(class_counts.index[0])
        probability = {label: 0.0 for label in bin_labels}
        probability[only_label] = 1.0
        return ConstantMulticlassModel(bin_labels, probability)

    model_cfg = config["models"]
    model = build_logistic_pipeline(
        max_iter=int(model_cfg["logistic_max_iter"]),
        class_weight=str(model_cfg["class_weight"]),
        multiclass=True,
    )
    model.fit(expanded_X, expanded_y, model__sample_weight=sample_weight)
    return model


def predict_distribution_probabilities(
    model: object,
    frame: pd.DataFrame,
    *,
    features: list[str],
    bin_labels: list[str],
) -> pd.DataFrame:
    raw = model.predict_proba(frame[features])
    classes = [str(item) for item in model.classes_]
    probabilities = pd.DataFrame(0.0, index=frame.index, columns=bin_labels, dtype=float)
    for idx, label in enumerate(classes):
        probabilities[label] = raw[:, idx]
    return clip_and_renormalize(probabilities)


def business_probabilities_from_bins(probabilities: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=probabilities.index)
    result["acceptable"] = probabilities["bin_3"]
    result["warning"] = probabilities["bin_2"] + probabilities["bin_4"]
    result["unacceptable"] = probabilities["bin_1"] + probabilities["bin_5"]
    return result


def multiclass_brier_score(probabilities: pd.DataFrame, target_frame: pd.DataFrame, labels: list[str]) -> float:
    actual = target_frame[[f"soft_target_{label}" for label in labels]].to_numpy(dtype=float)
    predicted = probabilities[labels].to_numpy(dtype=float)
    return float(np.mean(np.sum((predicted - actual) ** 2, axis=1)))


def multiclass_log_loss(
    probabilities: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    labels: list[str],
    eps: float,
) -> float:
    actual = target_frame[[f"soft_target_{label}" for label in labels]].to_numpy(dtype=float)
    predicted = np.clip(probabilities[labels].to_numpy(dtype=float), eps, 1.0)
    return float(np.mean(-np.sum(actual * np.log(predicted), axis=1)))


def normalized_entropy(probabilities: pd.DataFrame, *, labels: list[str], eps: float) -> pd.Series:
    clipped = np.clip(probabilities[labels].to_numpy(dtype=float), eps, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    normalizer = math.log(len(labels))
    return pd.Series(entropy / normalizer, index=probabilities.index, dtype=float)


def entropy_summary(probabilities: pd.DataFrame, *, labels: list[str], eps: float) -> dict[str, float]:
    entropy = normalized_entropy(probabilities, labels=labels, eps=eps)
    return {
        "mean": float(entropy.mean()),
        "median": float(entropy.median()),
        "p90": float(entropy.quantile(0.90)),
        "max": float(entropy.max()),
    }


def false_clear_rate(frame: pd.DataFrame, *, pred_col: str) -> float | None:
    subset = frame.loc[frame["business_label"].eq("unacceptable")].copy()
    if subset.empty:
        return None
    return float((subset[pred_col] != "unacceptable").mean())


def acceptable_confidence(frame: pd.DataFrame, *, acceptable_prob_col: str) -> float | None:
    subset = frame.loc[frame["business_label"].eq("acceptable")].copy()
    if subset.empty:
        return None
    return float(pd.to_numeric(subset[acceptable_prob_col], errors="coerce").mean())


def boundary_subset_stats(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    acceptable_prob_col: str,
    unacceptable_prob_col: str,
) -> dict[str, object]:
    if frame.empty:
        return {"samples": 0}
    overconfidence = boundary_overconfidence_stats(
        frame,
        pred_col=pred_col,
        acceptable_prob_col=acceptable_prob_col,
        unacceptable_prob_col=unacceptable_prob_col,
    )
    return {
        "samples": int(len(frame)),
        "high_confidence_non_warning_rate": overconfidence["high_confidence_non_warning_rate"],
        "unacceptable_false_clear_rate": false_clear_rate(frame, pred_col=pred_col),
        "acceptable_mean_probability": acceptable_confidence(frame, acceptable_prob_col=acceptable_prob_col),
    }


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the first distributional interval-prediction cleanroom baseline without reject option."
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
    feature_cols = feature_columns(feature_rows)
    feature_cols = [
        column
        for column in feature_cols
        if not column.startswith("soft_target_") and column not in {"hard_bin_label", "soft_label_zone"}
    ]

    reference_candidates = build_candidate_grid(config)
    reference_thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]
    soft_eps = float(config["distributional"]["entropy_log_epsilon"])

    feature_rows_csv = artifacts_dir / "distributional_reject_feature_rows.csv"
    results_csv = artifacts_dir / "distributional_reject_results.csv"
    summary_json = artifacts_dir / "distributional_reject_summary.json"
    per_fold_csv = artifacts_dir / "distributional_reject_per_fold.csv"
    reference_candidate_csv = artifacts_dir / "distributional_reject_reference_candidate_per_fold.csv"
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
        predictions[f"reference_bin_prob_{label}"] = pd.NA
        predictions[f"distributional_bin_prob_{label}"] = pd.NA
    for label in BUSINESS_LABELS:
        predictions[f"reference_business_prob_{label}"] = pd.NA
        predictions[f"distributional_business_prob_{label}"] = pd.NA
    predictions["reference_business_pred"] = pd.NA
    predictions["distributional_business_pred"] = pd.NA

    per_fold_rows: list[dict[str, object]] = []
    reference_candidate_rows: list[dict[str, object]] = []
    reference_eval_rows: list[dict[str, object]] = []
    distributional_eval_rows: list[dict[str, object]] = []
    reference_distribution_rows: list[dict[str, float]] = []
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

        reference_cumulative, reference_business_prob = fit_cumulative_business_probabilities_inner_thresholds_only(
            train_frame,
            test_frame,
            features=selected_features,
            thresholds=reference_thresholds,
            config=config,
            candidate=chosen_reference_candidate,
        )
        reference_intervals = clip_and_renormalize(cumulative_to_interval_probabilities(reference_cumulative))
        reference_bin_prob = reference_intervals.rename(
            columns={
                "lt_8_0": "bin_1",
                "between_8_0_8_2": "bin_2",
                "between_8_2_8_7": "bin_3",
                "between_8_7_8_9": "bin_4",
                "ge_8_9": "bin_5",
            }
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

        reference_pred = probability_argmax(reference_business_prob, BUSINESS_LABELS)
        distributional_pred = probability_argmax(distributional_business_prob, BUSINESS_LABELS)

        for label in BIN_LABELS:
            predictions.loc[test_frame.index, f"reference_bin_prob_{label}"] = reference_bin_prob[label]
            predictions.loc[test_frame.index, f"distributional_bin_prob_{label}"] = distributional_bin_prob[label]
        for label in BUSINESS_LABELS:
            predictions.loc[test_frame.index, f"reference_business_prob_{label}"] = reference_business_prob[label]
            predictions.loc[test_frame.index, f"distributional_business_prob_{label}"] = distributional_business_prob[label]
        predictions.loc[test_frame.index, "reference_business_pred"] = reference_pred
        predictions.loc[test_frame.index, "distributional_business_pred"] = distributional_pred

        reference_metrics = evaluate_prediction_set(test_frame, reference_business_prob, reference_pred)
        distributional_metrics = evaluate_prediction_set(test_frame, distributional_business_prob, distributional_pred)
        reference_eval_rows.append(
            {
                "fold": fold_index,
                "macro_f1": reference_metrics["macro_f1"],
                "balanced_accuracy": reference_metrics["balanced_accuracy"],
                "core_qualified_average_precision": reference_metrics["core_qualified_average_precision"],
                "boundary_warning_average_precision": reference_metrics["boundary_warning_average_precision"],
                "clearly_unacceptable_average_precision": reference_metrics["clearly_unacceptable_average_precision"],
                "boundary_high_confidence_non_warning_rate": reference_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )
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

        reference_distribution_rows.append(
            {
                "fold": fold_index,
                "multiclass_brier_score": multiclass_brier_score(reference_bin_prob, test_frame, BIN_LABELS),
                "negative_log_loss": multiclass_log_loss(reference_bin_prob, test_frame, labels=BIN_LABELS, eps=soft_eps),
                "normalized_entropy_mean": entropy_summary(reference_bin_prob, labels=BIN_LABELS, eps=soft_eps)["mean"],
            }
        )
        distributional_distribution_rows.append(
            {
                "fold": fold_index,
                "multiclass_brier_score": multiclass_brier_score(distributional_bin_prob, test_frame, BIN_LABELS),
                "negative_log_loss": multiclass_log_loss(distributional_bin_prob, test_frame, labels=BIN_LABELS, eps=soft_eps),
                "normalized_entropy_mean": entropy_summary(distributional_bin_prob, labels=BIN_LABELS, eps=soft_eps)["mean"],
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
                "reference_boundary_weight": chosen_reference_candidate["boundary_weight"],
                "reference_warning_weight": chosen_reference_candidate["warning_weight"],
                "reference_macro_f1": reference_metrics["macro_f1"],
                "distributional_macro_f1": distributional_metrics["macro_f1"],
                "reference_balanced_accuracy": reference_metrics["balanced_accuracy"],
                "distributional_balanced_accuracy": distributional_metrics["balanced_accuracy"],
                "reference_brier_score": reference_distribution_rows[-1]["multiclass_brier_score"],
                "distributional_brier_score": distributional_distribution_rows[-1]["multiclass_brier_score"],
                "reference_boundary_high_conf_non_warning_rate": reference_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                "distributional_boundary_high_conf_non_warning_rate": distributional_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )

    for label in BIN_LABELS:
        predictions[f"reference_bin_prob_{label}"] = pd.to_numeric(predictions[f"reference_bin_prob_{label}"], errors="coerce")
        predictions[f"distributional_bin_prob_{label}"] = pd.to_numeric(predictions[f"distributional_bin_prob_{label}"], errors="coerce")
    for label in BUSINESS_LABELS:
        predictions[f"reference_business_prob_{label}"] = pd.to_numeric(predictions[f"reference_business_prob_{label}"], errors="coerce")
        predictions[f"distributional_business_prob_{label}"] = pd.to_numeric(predictions[f"distributional_business_prob_{label}"], errors="coerce")

    scored_predictions = predictions.loc[
        predictions["reference_business_pred"].notna() & predictions["distributional_business_pred"].notna()
    ].copy()

    reference_summary = evaluate_predictions(
        scored_predictions,
        pred_col="reference_business_pred",
        acceptable_prob_col="reference_business_prob_acceptable",
        warning_prob_col="reference_business_prob_warning",
        unacceptable_prob_col="reference_business_prob_unacceptable",
    )
    distributional_summary = evaluate_predictions(
        scored_predictions,
        pred_col="distributional_business_pred",
        acceptable_prob_col="distributional_business_prob_acceptable",
        warning_prob_col="distributional_business_prob_warning",
        unacceptable_prob_col="distributional_business_prob_unacceptable",
    )

    reference_distribution_summary = {
        "multiclass_brier_score": multiclass_brier_score(
            scored_predictions.rename(columns={f"reference_bin_prob_{label}": label for label in BIN_LABELS}),
            scored_predictions,
            BIN_LABELS,
        ),
        "negative_log_loss": multiclass_log_loss(
            scored_predictions.rename(columns={f"reference_bin_prob_{label}": label for label in BIN_LABELS}),
            scored_predictions,
            labels=BIN_LABELS,
            eps=soft_eps,
        ),
        "entropy": entropy_summary(
            scored_predictions.rename(columns={f"reference_bin_prob_{label}": label for label in BIN_LABELS}),
            labels=BIN_LABELS,
            eps=soft_eps,
        ),
        "calibration_error": None,
    }
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
            eps=soft_eps,
        ),
        "entropy": entropy_summary(
            scored_predictions.rename(columns={f"distributional_bin_prob_{label}": label for label in BIN_LABELS}),
            labels=BIN_LABELS,
            eps=soft_eps,
        ),
        "calibration_error": None,
    }

    low_boundary_mask = scored_predictions["t90"].between(7.9, 8.3, inclusive="both")
    high_boundary_mask = scored_predictions["t90"].between(8.6, 8.8, inclusive="both")
    boundary_diagnostics = {
        "7_9_8_3": {
            "reference": boundary_subset_stats(
                scored_predictions.loc[low_boundary_mask],
                pred_col="reference_business_pred",
                acceptable_prob_col="reference_business_prob_acceptable",
                unacceptable_prob_col="reference_business_prob_unacceptable",
            ),
            "distributional": boundary_subset_stats(
                scored_predictions.loc[low_boundary_mask],
                pred_col="distributional_business_pred",
                acceptable_prob_col="distributional_business_prob_acceptable",
                unacceptable_prob_col="distributional_business_prob_unacceptable",
            ),
        },
        "8_6_8_8": {
            "reference": boundary_subset_stats(
                scored_predictions.loc[high_boundary_mask],
                pred_col="reference_business_pred",
                acceptable_prob_col="reference_business_prob_acceptable",
                unacceptable_prob_col="reference_business_prob_unacceptable",
            ),
            "distributional": boundary_subset_stats(
                scored_predictions.loc[high_boundary_mask],
                pred_col="distributional_business_pred",
                acceptable_prob_col="distributional_business_prob_acceptable",
                unacceptable_prob_col="distributional_business_prob_unacceptable",
            ),
        },
    }

    reference_fold_mean = aggregate_metrics([{k: v for k, v in row.items() if k != "fold"} for row in reference_eval_rows])
    distributional_fold_mean = aggregate_metrics([{k: v for k, v in row.items() if k != "fold"} for row in distributional_eval_rows])
    reference_distribution_fold_mean = {
        "multiclass_brier_score": float(pd.DataFrame(reference_distribution_rows)["multiclass_brier_score"].mean()),
        "negative_log_loss": float(pd.DataFrame(reference_distribution_rows)["negative_log_loss"].mean()),
        "normalized_entropy_mean": float(pd.DataFrame(reference_distribution_rows)["normalized_entropy_mean"].mean()),
    }
    distributional_distribution_fold_mean = {
        "multiclass_brier_score": float(pd.DataFrame(distributional_distribution_rows)["multiclass_brier_score"].mean()),
        "negative_log_loss": float(pd.DataFrame(distributional_distribution_rows)["negative_log_loss"].mean()),
        "normalized_entropy_mean": float(pd.DataFrame(distributional_distribution_rows)["normalized_entropy_mean"].mean()),
    }

    improvement = {
        "macro_f1_delta": distributional_summary["macro_f1"] - reference_summary["macro_f1"],
        "balanced_accuracy_delta": distributional_summary["balanced_accuracy"] - reference_summary["balanced_accuracy"],
        "core_qualified_average_precision_delta": distributional_summary["core_qualified_average_precision"] - reference_summary["core_qualified_average_precision"],
        "boundary_warning_average_precision_delta": distributional_summary["boundary_warning_average_precision"] - reference_summary["boundary_warning_average_precision"],
        "clearly_unacceptable_average_precision_delta": distributional_summary["clearly_unacceptable_average_precision"] - reference_summary["clearly_unacceptable_average_precision"],
        "boundary_high_confidence_non_warning_rate_delta": distributional_summary["boundary_overconfidence"]["high_confidence_non_warning_rate"] - reference_summary["boundary_overconfidence"]["high_confidence_non_warning_rate"],
        "multiclass_brier_score_delta": distributional_distribution_summary["multiclass_brier_score"] - reference_distribution_summary["multiclass_brier_score"],
        "negative_log_loss_delta": distributional_distribution_summary["negative_log_loss"] - reference_distribution_summary["negative_log_loss"],
    }

    feature_rows.to_csv(feature_rows_csv, index=False, encoding="utf-8-sig")
    scored_predictions.to_csv(results_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_fold_rows).to_csv(per_fold_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_candidate_rows).to_csv(reference_candidate_csv, index=False, encoding="utf-8-sig")
    alias_frame.to_csv(alias_csv, index=False, encoding="utf-8-sig")

    soft_zone_counts = feature_rows["soft_label_zone"].value_counts().to_dict()
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
            "soft_zone_counts": soft_zone_counts,
        },
        "data_summary": {
            "aligned_rows": int(len(feature_rows)),
            "raw_common_rows": int(len(predictions)),
            "scored_rows": int(len(scored_predictions)),
            "n_splits_used": int(config["validation"]["n_splits"]),
            "inner_n_splits_used": int(config["validation"]["inner_n_splits"]),
        },
        "preclean_summary": preclean_summary,
        "reference_summary": reference_summary,
        "distributional_summary": distributional_summary,
        "reference_distribution_summary": reference_distribution_summary,
        "distributional_distribution_summary": distributional_distribution_summary,
        "reference_fold_mean": reference_fold_mean,
        "distributional_fold_mean": distributional_fold_mean,
        "reference_distribution_fold_mean": reference_distribution_fold_mean,
        "distributional_distribution_fold_mean": distributional_distribution_fold_mean,
        "boundary_diagnostics": boundary_diagnostics,
        "improvement_summary": improvement,
        "per_fold": per_fold_rows,
        "reference_candidate_rows": reference_candidate_rows,
        "artifacts": {
            "feature_rows_csv": str(feature_rows_csv),
            "results_csv": str(results_csv),
            "per_fold_csv": str(per_fold_csv),
            "reference_candidate_csv": str(reference_candidate_csv),
            "alias_csv": str(alias_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    summary_json.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Distributional Reject Option Baseline-B Summary",
        "",
        "- Branch stage: baseline_b_only",
        "- Reference: frozen ordinal / cumulative main line",
        "- Treatment: 5-bin distributional prediction without reject",
        f"- Scored rows: {len(scored_predictions)}",
        f"- Soft radius: {config['distributional']['soft_radius']}",
        f"- Reference macro_f1: {reference_summary['macro_f1']:.4f}",
        f"- Distributional macro_f1: {distributional_summary['macro_f1']:.4f}",
        f"- Reference balanced_accuracy: {reference_summary['balanced_accuracy']:.4f}",
        f"- Distributional balanced_accuracy: {distributional_summary['balanced_accuracy']:.4f}",
        f"- Reference warning_AP: {reference_summary['boundary_warning_average_precision']:.4f}",
        f"- Distributional warning_AP: {distributional_summary['boundary_warning_average_precision']:.4f}",
        f"- Reference unacceptable_AP: {reference_summary['clearly_unacceptable_average_precision']:.4f}",
        f"- Distributional unacceptable_AP: {distributional_summary['clearly_unacceptable_average_precision']:.4f}",
        f"- Reference boundary high-confidence non-warning: {reference_summary['boundary_overconfidence']['high_confidence_non_warning_rate']:.4f}",
        f"- Distributional boundary high-confidence non-warning: {distributional_summary['boundary_overconfidence']['high_confidence_non_warning_rate']:.4f}",
        f"- Reference multiclass Brier: {reference_distribution_summary['multiclass_brier_score']:.4f}",
        f"- Distributional multiclass Brier: {distributional_distribution_summary['multiclass_brier_score']:.4f}",
        f"- Reference log loss: {reference_distribution_summary['negative_log_loss']:.4f}",
        f"- Distributional log loss: {distributional_distribution_summary['negative_log_loss']:.4f}",
        "",
        "## Selected Frozen Reference Candidate Per Fold",
        "",
    ]
    for row in reference_candidate_rows:
        lines.append(
            f"- fold {row['fold']}: {row['candidate_name']} "
            f"(boundary_weight={row['boundary_weight']}, warning_weight={row['warning_weight']})"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
