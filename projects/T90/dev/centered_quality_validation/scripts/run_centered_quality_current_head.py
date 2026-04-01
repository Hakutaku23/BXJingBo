from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
CENTERED_DIR = THIS_DIR.parent
PROJECT_DIR = CENTERED_DIR.parents[1]
LEGACY_SCRIPT_DIR = PROJECT_DIR / "dev" / "cleanroom_validation" / "scripts"
if str(LEGACY_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_SCRIPT_DIR))

from run_ordinal_cumulative_current_head import (  # noqa: E402
    ConstantProbabilityModel,
    apply_monotonic_correction,
    assign_labels,
    binary_ap,
    build_feature_rows,
    build_logistic_pipeline,
    business_probabilities_from_intervals,
    clip_and_renormalize,
    cumulative_to_interval_probabilities,
    discover_lims_path,
    feature_columns,
    format_threshold,
    load_lims_data,
    predict_threshold_probability,
    probability_argmax,
    preclean_features,
    select_topk_sensors,
    selected_feature_columns,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_identity_dedup import (  # noqa: E402
    BUSINESS_LABELS,
    build_candidate_grid,
    evaluate_prediction_set,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup import (  # noqa: E402
    fit_cumulative_business_probabilities_inner_thresholds_only,
    select_candidate_on_train,
)
from run_ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head import (  # noqa: E402
    load_identity_deduped_combined_dcs,
)


DEFAULT_CONFIG_PATH = CENTERED_DIR / "configs" / "centered_quality_current_head.yaml"
CENTERED_LABELS = ["unacceptable", "acceptable", "premium"]
CENTERED_ACTION_LABELS = ["unacceptable", "acceptable", "premium", "retest"]


@dataclass
class ConstantRegressor:
    value: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), float(self.value), dtype=float)


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def desirability_score(values: pd.Series, center: float, half_width: float) -> pd.Series:
    return (1.0 - (values.astype(float) - center).abs() / half_width).clip(lower=0.0, upper=1.0)


def candidate_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "true_desirability",
        "centered_truth",
        "target_lower_risk",
        "target_upper_risk",
        "center_core_flag",
        "margin_in_spec_flag",
    }
    return [column for column in feature_columns(frame) if column not in blocked]


def fit_ridge_model(train_frame: pd.DataFrame, features: list[str], target_col: str, alpha: float) -> Pipeline | ConstantRegressor:
    target = pd.to_numeric(train_frame[target_col], errors="coerce")
    if target.nunique(dropna=True) < 2:
        return ConstantRegressor(float(target.dropna().iloc[0]) if target.notna().any() else 0.0)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )
    model.fit(train_frame[features], target)
    return model


def predict_clipped_regression(model: Pipeline | ConstantRegressor, frame: pd.DataFrame) -> pd.Series:
    values = model.predict(frame)
    return pd.Series(np.clip(values, 0.0, 1.0), index=frame.index, dtype=float)


def centered_truth_labels(frame: pd.DataFrame, premium_true_threshold: float) -> pd.Series:
    return pd.Series(
        np.select(
            [
                frame["is_unacceptable"].astype(bool),
                frame["true_desirability"] >= premium_true_threshold,
            ],
            ["unacceptable", "premium"],
            default="acceptable",
        ),
        index=frame.index,
        dtype="object",
    )


def centered_decision(lower_risk: pd.Series, upper_risk: pd.Series, desirability: pd.Series, config: dict[str, object]) -> pd.Series:
    decision_cfg = config["decision"]
    max_risk = pd.concat([lower_risk.astype(float), upper_risk.astype(float)], axis=1).max(axis=1)
    decision = np.select(
        [
            max_risk >= float(decision_cfg["unacceptable_risk_threshold"]),
            (max_risk < float(decision_cfg["premium_risk_ceiling"])) & (desirability >= float(decision_cfg["premium_quality_threshold"])),
            (max_risk < float(decision_cfg["acceptable_risk_ceiling"])) & (desirability >= float(decision_cfg["acceptable_quality_floor"])),
        ],
        ["unacceptable", "premium", "acceptable"],
        default="retest",
    )
    return pd.Series(decision, index=lower_risk.index, dtype="object")


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    valid = pd.concat([actual.astype(float), predicted.astype(float)], axis=1).dropna()
    if valid.empty:
        return math.nan
    return float((valid.iloc[:, 0] - valid.iloc[:, 1]).abs().mean())


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    valid = pd.concat([actual.astype(float), predicted.astype(float)], axis=1).dropna()
    if valid.empty:
        return math.nan
    return float(np.sqrt(np.mean((valid.iloc[:, 0] - valid.iloc[:, 1]) ** 2)))


def spearman_corr(actual: pd.Series, predicted: pd.Series) -> float:
    valid = pd.concat([actual.astype(float), predicted.astype(float)], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return math.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman"))


def precision_recall_for_positive(actual: pd.Series, predicted_positive: pd.Series) -> dict[str, float]:
    actual_bool = actual.astype(bool)
    pred_bool = predicted_positive.astype(bool)
    tp = int((actual_bool & pred_bool).sum())
    pred_pos = int(pred_bool.sum())
    actual_pos = int(actual_bool.sum())
    return {
        "precision": float(tp / pred_pos) if pred_pos else math.nan,
        "recall": float(tp / actual_pos) if actual_pos else math.nan,
    }


def risk_summary(frame: pd.DataFrame, lower_risk_col: str, upper_risk_col: str, decision_col: str) -> dict[str, object]:
    unacceptable_mask = frame["is_unacceptable"].astype(bool)
    false_clear = unacceptable_mask & frame[decision_col].isin(["acceptable", "premium"])
    retest = unacceptable_mask & frame[decision_col].eq("retest")
    predicted_unacceptable = frame[decision_col].eq("unacceptable")
    return {
        "lower_risk_average_precision": binary_ap(frame["target_lower_risk"], frame[lower_risk_col]),
        "upper_risk_average_precision": binary_ap(frame["target_upper_risk"], frame[upper_risk_col]),
        "unacceptable_recall": float(predicted_unacceptable[unacceptable_mask].mean()) if unacceptable_mask.any() else math.nan,
        "unacceptable_false_clear_rate": float(false_clear.mean()) if unacceptable_mask.any() else math.nan,
        "unacceptable_retest_rate": float(retest.mean()) if unacceptable_mask.any() else math.nan,
    }


def desirability_summary(frame: pd.DataFrame, predicted_col: str, premium_true_threshold: float) -> dict[str, object]:
    premium_truth = frame["centered_truth"].eq("premium")
    premium_pred = frame[predicted_col].astype(float) >= premium_true_threshold
    center_core = frame["center_core_flag"].astype(bool)
    margin = frame["margin_in_spec_flag"].astype(bool)
    center_mean = float(frame.loc[center_core, predicted_col].mean()) if center_core.any() else math.nan
    margin_mean = float(frame.loc[margin, predicted_col].mean()) if margin.any() else math.nan
    return {
        "mae": mae(frame["true_desirability"], frame[predicted_col]),
        "rmse": rmse(frame["true_desirability"], frame[predicted_col]),
        "spearman": spearman_corr(frame["true_desirability"], frame[predicted_col]),
        "premium_precision_recall": precision_recall_for_positive(premium_truth, premium_pred),
        "center_core_prediction_mean": center_mean,
        "margin_prediction_mean": margin_mean,
        "center_margin_gap": float(center_mean - margin_mean) if not math.isnan(center_mean) and not math.isnan(margin_mean) else math.nan,
    }


def decision_summary(frame: pd.DataFrame, decision_col: str) -> dict[str, object]:
    actual = frame["centered_truth"].astype(str)
    predicted = frame[decision_col].astype(str)
    covered_mask = predicted.ne("retest")
    boundary_mask = frame["boundary_any_flag"].astype(bool)
    non_boundary_mask = ~boundary_mask
    covered = frame.loc[covered_mask].copy()
    return {
        "macro_f1_vs_truth": float(f1_score(actual, predicted, labels=CENTERED_LABELS, average="macro", zero_division=0)),
        "decision_coverage": float(covered_mask.mean()),
        "retest_rate": float(predicted.eq("retest").mean()),
        "boundary_retest_rate": float(predicted[boundary_mask].eq("retest").mean()) if boundary_mask.any() else math.nan,
        "non_boundary_decision_coverage": float(covered_mask[non_boundary_mask].mean()) if non_boundary_mask.any() else math.nan,
        "covered_macro_f1": float(f1_score(covered["centered_truth"], covered[decision_col], labels=CENTERED_LABELS, average="macro", zero_division=0)) if not covered.empty else math.nan,
        "covered_balanced_accuracy": float(balanced_accuracy_score(covered["centered_truth"], covered[decision_col])) if not covered.empty else math.nan,
    }


def write_summary_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Centered Quality Current-Head Summary",
        "",
        "## First-pass outcome",
        "",
        f"- aligned samples: {summary['data_summary']['aligned_rows']}",
        f"- preclean features: {summary['preclean_summary']['feature_count_after']}",
        f"- selected sensor union: {summary['selected_sensor_union_count']}",
        f"- stable selected sensors (>=4 folds): {summary['stable_sensor_count']}",
        "",
        "## Baseline A",
        "",
        f"- threshold macro_f1: {summary['baseline_threshold_summary']['macro_f1']:.4f}",
        f"- threshold balanced_accuracy: {summary['baseline_threshold_summary']['balanced_accuracy']:.4f}",
        f"- boundary high-conf non-warning: {summary['baseline_threshold_summary']['boundary_overconfidence']['high_confidence_non_warning_rate']:.4f}",
        "",
        "## Treatment B",
        "",
        f"- desirability MAE: {summary['treatment_b_centered_only_summary']['mae']:.4f}",
        f"- desirability RMSE: {summary['treatment_b_centered_only_summary']['rmse']:.4f}",
        f"- desirability Spearman: {summary['treatment_b_centered_only_summary']['spearman']:.4f}",
        "",
        "## Treatment C",
        "",
        f"- lower-risk AP: {summary['treatment_c_risk_summary']['lower_risk_average_precision']:.4f}",
        f"- upper-risk AP: {summary['treatment_c_risk_summary']['upper_risk_average_precision']:.4f}",
        f"- centered decision macro_f1: {summary['treatment_c_decision_summary']['macro_f1_vs_truth']:.4f}",
        f"- decision coverage: {summary['treatment_c_decision_summary']['decision_coverage']:.4f}",
        f"- retest rate: {summary['treatment_c_decision_summary']['retest_rate']:.4f}",
        f"- covered balanced_accuracy: {summary['treatment_c_decision_summary']['covered_balanced_accuracy']:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first centered-quality cleanroom experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    outputs = config["outputs"]
    artifacts_dir = CENTERED_DIR / str(outputs["artifacts_dir"])
    reports_dir = CENTERED_DIR / str(outputs["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_DIR / "data"
    main_dcs_path = PROJECT_DIR / str(config["data"]["dcs_main_path"])
    supplemental_path = PROJECT_DIR / str(config["data"]["dcs_supplemental_path"])
    lims_path = discover_lims_path(data_dir, str(config["data"]["lims_glob"]))

    lims, _ = load_lims_data(lims_path)
    dcs, dcs_audit, alias_frame = load_identity_deduped_combined_dcs(main_dcs_path, supplemental_path, config)
    feature_rows = build_feature_rows(
        lims,
        dcs,
        lookback_minutes=int(config["features"]["lookback_minutes"]),
        stats=list(config["features"]["window_statistics"]),
    )
    feature_rows = assign_labels(feature_rows, config)
    centered_cfg = config["centered_quality"]
    center = float(centered_cfg["target_center"])
    half_width = float(centered_cfg["tolerance_half_width"])
    premium_true_threshold = float(centered_cfg["premium_true_threshold"])
    feature_rows["true_desirability"] = desirability_score(feature_rows["t90"], center, half_width)
    feature_rows["centered_truth"] = centered_truth_labels(feature_rows, premium_true_threshold)
    feature_rows["target_lower_risk"] = feature_rows["t90"].lt(float(config["labels"]["hard_three_class"]["low"])).astype(int)
    feature_rows["target_upper_risk"] = feature_rows["t90"].gt(float(config["labels"]["hard_three_class"]["high"])).astype(int)
    feature_rows["center_core_flag"] = feature_rows["t90"].sub(center).abs().le(float(centered_cfg["center_core_radius"]))
    min_distance = pd.concat(
        [
            feature_rows["t90"].sub(float(config["labels"]["hard_three_class"]["low"])).abs(),
            feature_rows["t90"].sub(float(config["labels"]["hard_three_class"]["high"])).abs(),
        ],
        axis=1,
    ).min(axis=1)
    feature_rows["margin_in_spec_flag"] = feature_rows["business_label"].eq("acceptable") & min_distance.ge(float(centered_cfg["margin_min_distance"]))

    feature_rows, preclean_summary = preclean_features(feature_rows, config)
    feature_cols = candidate_feature_columns(feature_rows)
    thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]
    baseline_config = dict(config)
    baseline_config["training_objective"] = dict(config["baseline_reference"]["training_objective"])
    candidates = build_candidate_grid(baseline_config)

    predictions = feature_rows[
        [
            "sample_time",
            "t90",
            "business_label",
            "centered_truth",
            "boundary_any_flag",
            "is_unacceptable",
            "target_lower_risk",
            "target_upper_risk",
            "true_desirability",
            "center_core_flag",
            "margin_in_spec_flag",
        ]
    ].copy()
    for column in [
        "baseline_business_pred",
        "treatment_c_decision",
    ]:
        predictions[column] = pd.NA
    for label in BUSINESS_LABELS:
        predictions[f"baseline_business_prob_{label}"] = pd.NA
    for key in ["baseline_lower_risk", "baseline_upper_risk", "baseline_desirability_proxy", "treatment_b_desirability", "treatment_c_lower_risk", "treatment_c_upper_risk", "treatment_c_desirability"]:
        predictions[key] = pd.NA

    outer_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    per_fold_rows: list[dict[str, object]] = []
    candidate_search_rows: list[dict[str, object]] = []

    for fold, (train_index, test_index) in enumerate(outer_tscv.split(feature_rows), start=1):
        train_frame = feature_rows.iloc[train_index].copy()
        test_frame = feature_rows.iloc[test_index].copy()
        chosen_candidate, candidate_frame = select_candidate_on_train(
            train_frame,
            feature_cols=feature_cols,
            thresholds=thresholds,
            config=baseline_config,
            candidates=candidates,
        )
        candidate_frame.insert(0, "fold", fold)
        candidate_search_rows.extend(candidate_frame.to_dict(orient="records"))

        selected_sensors, sensor_scores = select_topk_sensors(train_frame, feature_cols, topk=int(config["screening"]["topk_sensors"]))
        selected_features = selected_feature_columns(feature_cols, selected_sensors)
        baseline_cumulative, baseline_probs = fit_cumulative_business_probabilities_inner_thresholds_only(
            train_frame,
            test_frame,
            features=selected_features,
            thresholds=thresholds,
            config=baseline_config,
            candidate=chosen_candidate,
        )
        baseline_pred = probability_argmax(baseline_probs, BUSINESS_LABELS)
        baseline_lower_risk = baseline_cumulative[f"lt_{format_threshold(float(config['labels']['hard_three_class']['low']))}"]
        baseline_upper_risk = 1.0 - baseline_cumulative[f"lt_{format_threshold(float(config['labels']['hard_three_class']['high']))}"]
        baseline_desirability_proxy = baseline_probs["acceptable"]

        lower_key = f"target_lt_{format_threshold(float(config['labels']['hard_three_class']['low']))}"
        upper_train_target = train_frame["target_upper_risk"].astype(int)
        lower_model = build_logistic_pipeline(max_iter=int(config["models"]["logistic_max_iter"]), class_weight=str(config["models"]["class_weight"]), multiclass=False)
        upper_model = build_logistic_pipeline(max_iter=int(config["models"]["logistic_max_iter"]), class_weight=str(config["models"]["class_weight"]), multiclass=False)
        if train_frame[lower_key].nunique() < 2:
            lower_model = ConstantProbabilityModel(probability=float(train_frame[lower_key].iloc[0]))
        else:
            lower_model.fit(train_frame[selected_features], train_frame[lower_key].astype(int))
        if upper_train_target.nunique() < 2:
            upper_model = ConstantProbabilityModel(probability=float(upper_train_target.iloc[0]))
        else:
            upper_model.fit(train_frame[selected_features], upper_train_target)
        desirability_model = fit_ridge_model(train_frame, selected_features, "true_desirability", alpha=float(config["models"]["ridge_alpha"]))

        treatment_b_desirability = predict_clipped_regression(desirability_model, test_frame[selected_features])
        treatment_c_lower_risk = pd.Series(predict_threshold_probability(lower_model, test_frame[selected_features]), index=test_frame.index, dtype=float)
        treatment_c_upper_risk = pd.Series(predict_threshold_probability(upper_model, test_frame[selected_features]), index=test_frame.index, dtype=float)
        treatment_c_desirability = treatment_b_desirability.copy()
        treatment_c_action = centered_decision(treatment_c_lower_risk, treatment_c_upper_risk, treatment_c_desirability, config)

        predictions.loc[test_frame.index, "baseline_business_pred"] = baseline_pred
        predictions.loc[test_frame.index, "baseline_business_prob_acceptable"] = baseline_probs["acceptable"]
        predictions.loc[test_frame.index, "baseline_business_prob_warning"] = baseline_probs["warning"]
        predictions.loc[test_frame.index, "baseline_business_prob_unacceptable"] = baseline_probs["unacceptable"]
        predictions.loc[test_frame.index, "baseline_lower_risk"] = baseline_lower_risk
        predictions.loc[test_frame.index, "baseline_upper_risk"] = baseline_upper_risk
        predictions.loc[test_frame.index, "baseline_desirability_proxy"] = baseline_desirability_proxy
        predictions.loc[test_frame.index, "treatment_b_desirability"] = treatment_b_desirability
        predictions.loc[test_frame.index, "treatment_c_lower_risk"] = treatment_c_lower_risk
        predictions.loc[test_frame.index, "treatment_c_upper_risk"] = treatment_c_upper_risk
        predictions.loc[test_frame.index, "treatment_c_desirability"] = treatment_c_desirability
        predictions.loc[test_frame.index, "treatment_c_decision"] = treatment_c_action

        fold_eval = evaluate_prediction_set(test_frame, baseline_probs, baseline_pred)
        per_fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "selected_sensor_count": int(len(selected_sensors)),
                "selected_sensors": "|".join(selected_sensors),
                "selected_candidate_name": str(chosen_candidate["candidate_name"]),
                "boundary_weight": float(chosen_candidate["boundary_weight"]),
                "warning_weight": float(chosen_candidate["warning_weight"]),
                "baseline_macro_f1": float(fold_eval["macro_f1"]),
                "baseline_balanced_accuracy": float(fold_eval["balanced_accuracy"]),
                "treatment_c_macro_f1_vs_truth": float(f1_score(test_frame["centered_truth"], treatment_c_action, labels=CENTERED_LABELS, average="macro", zero_division=0)),
                "treatment_c_decision_coverage": float(treatment_c_action.ne("retest").mean()),
                "selected_sensor_scores": json.dumps(sensor_scores, ensure_ascii=False),
            }
        )

    numeric_cols = [col for col in predictions.columns if col not in {"sample_time", "business_label", "centered_truth", "baseline_business_pred", "treatment_c_decision"}]
    for column in numeric_cols:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    scored_predictions = predictions.loc[predictions["baseline_business_pred"].notna()].copy()
    baseline_probabilities = scored_predictions[[f"baseline_business_prob_{label}" for label in BUSINESS_LABELS]].rename(columns={f"baseline_business_prob_{label}": label for label in BUSINESS_LABELS})
    baseline_threshold_summary = evaluate_prediction_set(scored_predictions, baseline_probabilities, scored_predictions["baseline_business_pred"])
    baseline_risk_proxy_summary = risk_summary(scored_predictions, "baseline_lower_risk", "baseline_upper_risk", "baseline_business_pred")
    baseline_desirability_proxy_summary = desirability_summary(scored_predictions, "baseline_desirability_proxy", premium_true_threshold)
    treatment_b_centered_only_summary = desirability_summary(scored_predictions, "treatment_b_desirability", premium_true_threshold)
    treatment_c_risk_summary = risk_summary(scored_predictions, "treatment_c_lower_risk", "treatment_c_upper_risk", "treatment_c_decision")
    treatment_c_desirability_summary = desirability_summary(scored_predictions, "treatment_c_desirability", premium_true_threshold)
    treatment_c_decision_summary = decision_summary(scored_predictions, "treatment_c_decision")

    per_fold_frame = pd.DataFrame(per_fold_rows)
    sensor_counter: dict[str, int] = {}
    for sensors in per_fold_frame["selected_sensors"]:
        for sensor in str(sensors).split("|"):
            if sensor:
                sensor_counter[sensor] = sensor_counter.get(sensor, 0) + 1

    summary = {
        "experiment_name": config["experiment_name"],
        "branch_stage": config["branch_stage"],
        "data_summary": {
            "lims_t90_nonnull": int(lims["t90"].notna().sum()),
            "aligned_rows": int(len(feature_rows)),
            "outer_test_rows": int(len(scored_predictions)),
            "window_row_count_mean": float(feature_rows["window_row_count"].mean()),
            **dcs_audit,
        },
        "preclean_summary": preclean_summary,
        "baseline_threshold_summary": baseline_threshold_summary,
        "baseline_risk_proxy_summary": baseline_risk_proxy_summary,
        "baseline_desirability_proxy_summary": baseline_desirability_proxy_summary,
        "treatment_b_centered_only_summary": treatment_b_centered_only_summary,
        "treatment_c_risk_summary": treatment_c_risk_summary,
        "treatment_c_desirability_summary": treatment_c_desirability_summary,
        "treatment_c_decision_summary": treatment_c_decision_summary,
        "selected_sensor_union_count": int(len(sensor_counter)),
        "stable_sensor_count": int(sum(1 for count in sensor_counter.values() if count >= 4)),
        "selected_sensor_frequency": dict(sorted(sensor_counter.items(), key=lambda item: (-item[1], item[0]))),
        "per_fold": per_fold_rows,
    }

    feature_rows.to_csv(artifacts_dir / "centered_quality_feature_rows.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(artifacts_dir / "centered_quality_results.csv", index=False, encoding="utf-8-sig")
    per_fold_frame.to_csv(artifacts_dir / "centered_quality_per_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_search_rows).to_csv(artifacts_dir / "centered_quality_baseline_candidate_search.csv", index=False, encoding="utf-8-sig")
    alias_frame.to_csv(artifacts_dir / "sensor_identity_alias_pairs.csv", index=False, encoding="utf-8-sig")
    (artifacts_dir / "centered_quality_summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_report(reports_dir / "centered_quality_summary.md", json_ready(summary))
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
