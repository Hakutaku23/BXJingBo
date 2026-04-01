from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit


THIS_DIR = Path(__file__).resolve().parent
CENTERED_DIR = THIS_DIR.parent
PROJECT_DIR = CENTERED_DIR.parents[1]
LEGACY_SCRIPT_DIR = PROJECT_DIR / "dev" / "cleanroom_validation" / "scripts"
if str(LEGACY_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_SCRIPT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_ordinal_cumulative_current_head import (  # noqa: E402
    ConstantProbabilityModel,
    assign_labels,
    binary_ap,
    build_feature_rows,
    build_logistic_pipeline,
    discover_lims_path,
    feature_columns,
    format_threshold,
    load_lims_data,
    predict_threshold_probability,
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
from run_centered_quality_current_head import (  # noqa: E402
    CENTERED_LABELS,
    decision_summary,
    json_ready,
    precision_recall_for_positive,
    risk_summary,
)


DEFAULT_CONFIG_PATH = CENTERED_DIR / "configs" / "centered_quality_center_band_current_head.yaml"


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate_feature_columns_center_band(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "center_band_true",
        "centered_truth",
        "target_lower_risk",
        "target_upper_risk",
    }
    return [column for column in feature_columns(frame) if column not in blocked]


def center_band_truth(frame: pd.DataFrame, config: dict[str, object]) -> pd.Series:
    band_cfg = config["centered_quality"]
    low = float(band_cfg["center_band_low"])
    high = float(band_cfg["center_band_high"])
    return frame["t90"].between(low, high, inclusive="left")


def centered_truth_labels(frame: pd.DataFrame, center_band_col: str) -> pd.Series:
    return pd.Series(
        np.select(
            [
                frame["is_unacceptable"].astype(bool),
                frame[center_band_col].astype(bool),
            ],
            ["unacceptable", "premium"],
            default="acceptable",
        ),
        index=frame.index,
        dtype="object",
    )


def fit_binary_head(train_frame: pd.DataFrame, features: list[str], target_col: str, config: dict[str, object]) -> object:
    target = train_frame[target_col].astype(int)
    if target.nunique() < 2:
        return ConstantProbabilityModel(probability=float(target.iloc[0]))
    model = build_logistic_pipeline(
        max_iter=int(config["models"]["logistic_max_iter"]),
        class_weight=str(config["models"]["class_weight"]),
        multiclass=False,
    )
    model.fit(train_frame[features], target)
    return model


def semantic_candidate_grid(config: dict[str, object]) -> list[dict[str, float | str]]:
    grid_cfg = config["semantic_search"]
    rows: list[dict[str, float | str]] = []
    for clear_thr in grid_cfg["clear_risk_threshold_grid"]:
        for retest_thr in grid_cfg["retest_risk_threshold_grid"]:
            if float(retest_thr) >= float(clear_thr):
                continue
            for premium_thr in grid_cfg["premium_center_threshold_grid"]:
                rows.append(
                    {
                        "candidate_name": f"clear_{float(clear_thr):.2f}_retest_{float(retest_thr):.2f}_premium_{float(premium_thr):.2f}",
                        "clear_risk_threshold": float(clear_thr),
                        "retest_risk_threshold": float(retest_thr),
                        "premium_center_threshold": float(premium_thr),
                    }
                )
    return rows


def semantic_decision(lower_risk: pd.Series, upper_risk: pd.Series, center_prob: pd.Series, candidate: dict[str, float | str]) -> pd.Series:
    max_risk = pd.concat([lower_risk.astype(float), upper_risk.astype(float)], axis=1).max(axis=1)
    decision = np.select(
        [
            max_risk >= float(candidate["clear_risk_threshold"]),
            max_risk >= float(candidate["retest_risk_threshold"]),
            center_prob.astype(float) >= float(candidate["premium_center_threshold"]),
        ],
        ["unacceptable", "retest", "premium"],
        default="acceptable",
    )
    return pd.Series(decision, index=lower_risk.index, dtype="object")


def center_band_summary(frame: pd.DataFrame, center_prob_col: str, decision_col: str) -> dict[str, object]:
    center_truth = frame["center_band_true"].astype(int)
    premium_decision = frame[decision_col].eq("premium")
    center_prob = pd.to_numeric(frame[center_prob_col], errors="coerce")
    return {
        "center_band_average_precision": binary_ap(center_truth, center_prob),
        "premium_precision_recall": precision_recall_for_positive(frame["centered_truth"].eq("premium"), premium_decision),
        "premium_rate": float(premium_decision.mean()),
    }


def semantic_candidate_metrics(frame: pd.DataFrame, decision_col: str) -> dict[str, float]:
    actual = frame["centered_truth"].astype(str)
    predicted = frame[decision_col].astype(str)
    unacceptable_truth = actual.eq("unacceptable")
    false_clear = unacceptable_truth & predicted.isin(["acceptable", "premium"])
    premium_precision = precision_recall_for_positive(actual.eq("premium"), predicted.eq("premium"))["precision"]
    return {
        "macro_f1_vs_truth": float(f1_score(actual, predicted, labels=CENTERED_LABELS, average="macro", zero_division=0)),
        "unacceptable_false_clear_rate": float(false_clear.mean()) if unacceptable_truth.any() else math.nan,
        "unacceptable_rate": float(predicted.eq("unacceptable").mean()),
        "retest_rate": float(predicted.eq("retest").mean()),
        "premium_precision": float(premium_precision) if not math.isnan(premium_precision) else -1.0,
    }


def choose_semantic_candidate(
    train_frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    config: dict[str, object],
    candidates: list[dict[str, float | str]],
) -> tuple[dict[str, float | str], pd.DataFrame]:
    inner_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["inner_n_splits"]))
    candidate_rows: list[dict[str, object]] = []

    for candidate in candidates:
        inner_rows: list[dict[str, float]] = []
        for inner_train_index, inner_val_index in inner_tscv.split(train_frame):
            inner_train = train_frame.iloc[inner_train_index].copy()
            inner_val = train_frame.iloc[inner_val_index].copy()
            selected_sensors, _ = select_topk_sensors(inner_train, feature_cols, topk=int(config["screening"]["topk_sensors"]))
            selected_features = selected_feature_columns(feature_cols, selected_sensors)

            lower_model = fit_binary_head(inner_train, selected_features, f"target_lt_{format_threshold(float(config['labels']['hard_three_class']['low']))}", config)
            upper_model = fit_binary_head(inner_train, selected_features, "target_upper_risk", config)
            center_model = fit_binary_head(inner_train, selected_features, "center_band_true", config)

            lower_risk = pd.Series(predict_threshold_probability(lower_model, inner_val[selected_features]), index=inner_val.index, dtype=float)
            upper_risk = pd.Series(predict_threshold_probability(upper_model, inner_val[selected_features]), index=inner_val.index, dtype=float)
            center_prob = pd.Series(predict_threshold_probability(center_model, inner_val[selected_features]), index=inner_val.index, dtype=float)
            decision = semantic_decision(lower_risk, upper_risk, center_prob, candidate)
            scored = inner_val.copy()
            scored["decision"] = decision
            inner_rows.append(semantic_candidate_metrics(scored, "decision"))

        candidate_rows.append(
            {
                **candidate,
                "macro_f1_vs_truth": float(pd.DataFrame(inner_rows)["macro_f1_vs_truth"].mean()),
                "unacceptable_false_clear_rate": float(pd.DataFrame(inner_rows)["unacceptable_false_clear_rate"].mean()),
                "unacceptable_rate": float(pd.DataFrame(inner_rows)["unacceptable_rate"].mean()),
                "retest_rate": float(pd.DataFrame(inner_rows)["retest_rate"].mean()),
                "premium_precision": float(pd.DataFrame(inner_rows)["premium_precision"].mean()),
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows).sort_values("candidate_name").reset_index(drop=True)
    chosen = candidate_frame.sort_values(
        ["unacceptable_false_clear_rate", "macro_f1_vs_truth", "unacceptable_rate", "premium_precision", "retest_rate"],
        ascending=[True, False, True, False, True],
    ).iloc[0]
    return chosen.to_dict(), candidate_frame


def write_summary_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Center-Band Current-Head Summary",
        "",
        "## Main outcome",
        "",
        f"- aligned samples: {summary['data_summary']['aligned_rows']}",
        f"- scored outer-test rows: {summary['data_summary']['outer_test_rows']}",
        f"- center-band share: {summary['center_band_truth_share']:.4f}",
        "",
        "## Baseline A",
        "",
        f"- threshold macro_f1: {summary['baseline_threshold_summary']['macro_f1']:.4f}",
        f"- threshold balanced_accuracy: {summary['baseline_threshold_summary']['balanced_accuracy']:.4f}",
        "",
        "## Treatment D",
        "",
        f"- center-band AP: {summary['treatment_d_center_band_summary']['center_band_average_precision']:.4f}",
        f"- centered decision macro_f1: {summary['treatment_d_decision_summary']['macro_f1_vs_truth']:.4f}",
        f"- unacceptable false-clear rate: {summary['treatment_d_risk_summary']['unacceptable_false_clear_rate']:.4f}",
        f"- unacceptable recall: {summary['treatment_d_risk_summary']['unacceptable_recall']:.4f}",
        f"- decision coverage: {summary['treatment_d_decision_summary']['decision_coverage']:.4f}",
        f"- retest rate: {summary['treatment_d_decision_summary']['retest_rate']:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the center-band semantics cleanroom experiment.")
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
    feature_rows["center_band_true"] = center_band_truth(feature_rows, config).astype(int)
    feature_rows["centered_truth"] = centered_truth_labels(feature_rows, "center_band_true")
    feature_rows["target_lower_risk"] = feature_rows["t90"].lt(float(config["labels"]["hard_three_class"]["low"])).astype(int)
    feature_rows["target_upper_risk"] = feature_rows["t90"].gt(float(config["labels"]["hard_three_class"]["high"])).astype(int)
    feature_rows, preclean_summary = preclean_features(feature_rows, config)
    feature_cols = candidate_feature_columns_center_band(feature_rows)
    thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]

    baseline_config = dict(config)
    baseline_config["training_objective"] = dict(config["baseline_reference"]["training_objective"])
    baseline_candidates = build_candidate_grid(baseline_config)
    semantic_candidates = semantic_candidate_grid(config)

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
            "center_band_true",
        ]
    ].copy()
    for col in ["baseline_business_pred", "treatment_d_decision"]:
        predictions[col] = pd.NA
    for label in BUSINESS_LABELS:
        predictions[f"baseline_business_prob_{label}"] = pd.NA
    for col in ["baseline_lower_risk", "baseline_upper_risk", "treatment_d_lower_risk", "treatment_d_upper_risk", "treatment_d_center_band_prob"]:
        predictions[col] = pd.NA

    outer_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    per_fold_rows: list[dict[str, object]] = []
    baseline_candidate_rows: list[dict[str, object]] = []
    semantic_candidate_rows: list[dict[str, object]] = []

    for fold, (train_index, test_index) in enumerate(outer_tscv.split(feature_rows), start=1):
        train_frame = feature_rows.iloc[train_index].copy()
        test_frame = feature_rows.iloc[test_index].copy()

        chosen_baseline_candidate, baseline_candidate_frame = select_candidate_on_train(
            train_frame,
            feature_cols=feature_cols,
            thresholds=thresholds,
            config=baseline_config,
            candidates=baseline_candidates,
        )
        baseline_candidate_frame.insert(0, "fold", fold)
        baseline_candidate_rows.extend(baseline_candidate_frame.to_dict(orient="records"))

        chosen_semantic_candidate, semantic_candidate_frame = choose_semantic_candidate(
            train_frame,
            feature_cols=feature_cols,
            config=config,
            candidates=semantic_candidates,
        )
        semantic_candidate_frame.insert(0, "fold", fold)
        semantic_candidate_rows.extend(semantic_candidate_frame.to_dict(orient="records"))

        selected_sensors, sensor_scores = select_topk_sensors(train_frame, feature_cols, topk=int(config["screening"]["topk_sensors"]))
        selected_features = selected_feature_columns(feature_cols, selected_sensors)

        baseline_cumulative, baseline_probs = fit_cumulative_business_probabilities_inner_thresholds_only(
            train_frame,
            test_frame,
            features=selected_features,
            thresholds=thresholds,
            config=baseline_config,
            candidate=chosen_baseline_candidate,
        )

        lower_model = fit_binary_head(train_frame, selected_features, f"target_lt_{format_threshold(float(config['labels']['hard_three_class']['low']))}", config)
        upper_model = fit_binary_head(train_frame, selected_features, "target_upper_risk", config)
        center_model = fit_binary_head(train_frame, selected_features, "center_band_true", config)

        lower_risk = pd.Series(predict_threshold_probability(lower_model, test_frame[selected_features]), index=test_frame.index, dtype=float)
        upper_risk = pd.Series(predict_threshold_probability(upper_model, test_frame[selected_features]), index=test_frame.index, dtype=float)
        center_prob = pd.Series(predict_threshold_probability(center_model, test_frame[selected_features]), index=test_frame.index, dtype=float)
        decision = semantic_decision(lower_risk, upper_risk, center_prob, chosen_semantic_candidate)

        predictions.loc[test_frame.index, "baseline_business_pred"] = baseline_probs[BUSINESS_LABELS].to_numpy().argmax(axis=1)
        predictions.loc[test_frame.index, "baseline_business_pred"] = pd.Series(
            [BUSINESS_LABELS[idx] for idx in baseline_probs[BUSINESS_LABELS].to_numpy().argmax(axis=1)],
            index=test_frame.index,
        )
        for label in BUSINESS_LABELS:
            predictions.loc[test_frame.index, f"baseline_business_prob_{label}"] = baseline_probs[label]
        predictions.loc[test_frame.index, "baseline_lower_risk"] = baseline_cumulative[f"lt_{format_threshold(float(config['labels']['hard_three_class']['low']))}"]
        predictions.loc[test_frame.index, "baseline_upper_risk"] = 1.0 - baseline_cumulative[f"lt_{format_threshold(float(config['labels']['hard_three_class']['high']))}"]
        predictions.loc[test_frame.index, "treatment_d_lower_risk"] = lower_risk
        predictions.loc[test_frame.index, "treatment_d_upper_risk"] = upper_risk
        predictions.loc[test_frame.index, "treatment_d_center_band_prob"] = center_prob
        predictions.loc[test_frame.index, "treatment_d_decision"] = decision

        baseline_eval = evaluate_prediction_set(test_frame, baseline_probs, pd.Series([BUSINESS_LABELS[idx] for idx in baseline_probs[BUSINESS_LABELS].to_numpy().argmax(axis=1)], index=test_frame.index))
        fold_scored = test_frame.copy()
        fold_scored["treatment_d_decision"] = decision
        per_fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "selected_sensor_count": int(len(selected_sensors)),
                "selected_sensors": "|".join(selected_sensors),
                "baseline_candidate_name": str(chosen_baseline_candidate["candidate_name"]),
                "semantic_candidate_name": str(chosen_semantic_candidate["candidate_name"]),
                "clear_risk_threshold": float(chosen_semantic_candidate["clear_risk_threshold"]),
                "retest_risk_threshold": float(chosen_semantic_candidate["retest_risk_threshold"]),
                "premium_center_threshold": float(chosen_semantic_candidate["premium_center_threshold"]),
                "baseline_macro_f1": float(baseline_eval["macro_f1"]),
                "baseline_balanced_accuracy": float(baseline_eval["balanced_accuracy"]),
                "treatment_d_macro_f1_vs_truth": float(f1_score(fold_scored["centered_truth"], fold_scored["treatment_d_decision"], labels=CENTERED_LABELS, average="macro", zero_division=0)),
                "treatment_d_unacceptable_rate": float(fold_scored["treatment_d_decision"].eq("unacceptable").mean()),
                "treatment_d_retest_rate": float(fold_scored["treatment_d_decision"].eq("retest").mean()),
                "selected_sensor_scores": json.dumps(sensor_scores, ensure_ascii=False),
            }
        )

    numeric_cols = [col for col in predictions.columns if col not in {"sample_time", "business_label", "centered_truth", "baseline_business_pred", "treatment_d_decision"}]
    for col in numeric_cols:
        predictions[col] = pd.to_numeric(predictions[col], errors="coerce")
    scored_predictions = predictions.loc[predictions["baseline_business_pred"].notna()].copy()
    baseline_probabilities = scored_predictions[[f"baseline_business_prob_{label}" for label in BUSINESS_LABELS]].rename(columns={f"baseline_business_prob_{label}": label for label in BUSINESS_LABELS})
    baseline_threshold_summary = evaluate_prediction_set(scored_predictions, baseline_probabilities, scored_predictions["baseline_business_pred"])
    treatment_d_risk_summary = risk_summary(scored_predictions, "treatment_d_lower_risk", "treatment_d_upper_risk", "treatment_d_decision")
    treatment_d_center_band_summary = center_band_summary(scored_predictions, "treatment_d_center_band_prob", "treatment_d_decision")
    treatment_d_decision_summary = decision_summary(scored_predictions, "treatment_d_decision")

    per_fold_frame = pd.DataFrame(per_fold_rows)
    sensor_counter: dict[str, int] = {}
    for sensors in per_fold_frame["selected_sensors"]:
        for sensor in str(sensors).split("|"):
            if sensor:
                sensor_counter[sensor] = sensor_counter.get(sensor, 0) + 1

    summary = {
        "experiment_name": config["experiment_name"],
        "branch_stage": config["branch_stage"],
        "assay_resolution_note": "T90 is manually measured and effectively quantized at 0.1 resolution; values like 8.45 are target semantics, not directly observed assay values.",
        "data_summary": {
            "lims_t90_nonnull": int(lims["t90"].notna().sum()),
            "aligned_rows": int(len(feature_rows)),
            "outer_test_rows": int(len(scored_predictions)),
            "window_row_count_mean": float(feature_rows["window_row_count"].mean()),
            **dcs_audit,
        },
        "preclean_summary": preclean_summary,
        "center_band_truth_share": float(scored_predictions["center_band_true"].mean()),
        "baseline_threshold_summary": baseline_threshold_summary,
        "treatment_d_risk_summary": treatment_d_risk_summary,
        "treatment_d_center_band_summary": treatment_d_center_band_summary,
        "treatment_d_decision_summary": treatment_d_decision_summary,
        "selected_sensor_union_count": int(len(sensor_counter)),
        "stable_sensor_count": int(sum(1 for count in sensor_counter.values() if count >= 4)),
        "selected_sensor_frequency": dict(sorted(sensor_counter.items(), key=lambda item: (-item[1], item[0]))),
        "per_fold": per_fold_rows,
    }

    feature_rows.to_csv(artifacts_dir / "centered_quality_center_band_feature_rows.csv", index=False, encoding="utf-8-sig")
    scored_predictions.to_csv(artifacts_dir / "centered_quality_center_band_results.csv", index=False, encoding="utf-8-sig")
    per_fold_frame.to_csv(artifacts_dir / "centered_quality_center_band_per_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(baseline_candidate_rows).to_csv(artifacts_dir / "centered_quality_center_band_baseline_candidate_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(semantic_candidate_rows).to_csv(artifacts_dir / "centered_quality_center_band_semantic_candidate_search.csv", index=False, encoding="utf-8-sig")
    alias_frame.to_csv(artifacts_dir / "sensor_identity_alias_pairs.csv", index=False, encoding="utf-8-sig")
    (artifacts_dir / "centered_quality_center_band_summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_report(reports_dir / "centered_quality_center_band_summary.md", json_ready(summary))
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
