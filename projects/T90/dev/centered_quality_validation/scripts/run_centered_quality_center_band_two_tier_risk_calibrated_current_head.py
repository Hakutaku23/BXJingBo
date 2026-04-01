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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
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
    assign_labels,
    build_feature_rows,
    discover_lims_path,
    format_threshold,
    load_lims_data,
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
from run_centered_quality_center_band_current_head import (  # noqa: E402
    center_band_summary,
    center_band_truth,
    centered_truth_labels,
    fit_binary_head,
    json_ready,
)
from run_centered_quality_center_band_two_tier_risk_current_head import (  # noqa: E402
    CENTERED_LABELS,
    candidate_feature_columns_two_tier,
    semantic_candidate_grid,
    semantic_candidate_metrics,
    two_tier_decision,
)
from run_centered_quality_current_head import decision_summary, risk_summary  # noqa: E402


DEFAULT_CONFIG_PATH = CENTERED_DIR / "configs" / "centered_quality_center_band_two_tier_risk_calibrated_current_head.yaml"


@dataclass
class IdentityCalibrator:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values.astype(float), 0.0, 1.0)


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_train_calibration(frame: pd.DataFrame, calibration_fraction: float, min_calibration_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(frame)
    calibration_rows = max(min_calibration_rows, int(math.ceil(n_rows * calibration_fraction)))
    calibration_rows = min(calibration_rows, max(1, n_rows // 3))
    if n_rows - calibration_rows < 50:
        calibration_rows = max(1, n_rows // 4)
    fit_rows = max(1, n_rows - calibration_rows)
    return frame.iloc[:fit_rows].copy(), frame.iloc[fit_rows:].copy()


def fit_isotonic_calibrator(raw_scores: pd.Series, target: pd.Series) -> IdentityCalibrator | IsotonicRegression:
    valid = pd.DataFrame({"score": pd.to_numeric(raw_scores, errors="coerce"), "target": pd.to_numeric(target, errors="coerce")}).dropna()
    if len(valid) < 20 or valid["target"].nunique() < 2 or valid["score"].nunique() < 2:
        return IdentityCalibrator()
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(valid["score"].to_numpy(dtype=float), valid["target"].to_numpy(dtype=float))
    return calibrator


def apply_calibrator(calibrator: IdentityCalibrator | IsotonicRegression, values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if isinstance(calibrator, IdentityCalibrator):
        calibrated = calibrator.transform(raw)
    else:
        calibrated = calibrator.predict(raw)
    return pd.Series(np.clip(calibrated, 0.0, 1.0), index=values.index, dtype=float)


def fit_and_calibrate_head(
    train_frame: pd.DataFrame,
    *,
    features: list[str],
    target_col: str,
    config: dict[str, object],
) -> tuple[object, IdentityCalibrator | IsotonicRegression]:
    calibration_cfg = config["calibration"]
    fit_frame, calibration_frame = split_train_calibration(
        train_frame,
        calibration_fraction=float(calibration_cfg["calibration_fraction"]),
        min_calibration_rows=int(calibration_cfg["min_calibration_rows"]),
    )
    model = fit_binary_head(fit_frame, features, target_col, config)
    calibration_scores = pd.Series(model.predict_proba(calibration_frame[features])[:, 1], index=calibration_frame.index, dtype=float)
    calibrator = fit_isotonic_calibrator(calibration_scores, calibration_frame[target_col].astype(int))
    return model, calibrator


def choose_semantic_candidate(
    train_frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    config: dict[str, object],
    candidates: list[dict[str, float | str]],
) -> tuple[dict[str, float | str], pd.DataFrame]:
    inner_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["inner_n_splits"]))
    candidate_rows: list[dict[str, object]] = []
    risk_cfg = config["risk_targets"]

    for candidate in candidates:
        inner_rows: list[dict[str, float]] = []
        for inner_train_index, inner_val_index in inner_tscv.split(train_frame):
            inner_train = train_frame.iloc[inner_train_index].copy()
            inner_val = train_frame.iloc[inner_val_index].copy()
            selected_sensors, _ = select_topk_sensors(inner_train, feature_cols, topk=int(config["screening"]["topk_sensors"]))
            selected_features = selected_feature_columns(feature_cols, selected_sensors)

            clear_lower_model, clear_lower_cal = fit_and_calibrate_head(inner_train, features=selected_features, target_col=f"target_lt_{format_threshold(float(risk_cfg['clear_low_threshold']))}", config=config)
            clear_upper_model, clear_upper_cal = fit_and_calibrate_head(inner_train, features=selected_features, target_col=f"target_ge_{format_threshold(float(risk_cfg['clear_high_threshold']))}", config=config)
            retest_lower_model, retest_lower_cal = fit_and_calibrate_head(inner_train, features=selected_features, target_col=f"target_lt_{format_threshold(float(risk_cfg['retest_low_threshold']))}", config=config)
            retest_upper_model, retest_upper_cal = fit_and_calibrate_head(inner_train, features=selected_features, target_col="target_upper_risk", config=config)
            center_model = fit_binary_head(inner_train, selected_features, "center_band_true", config)

            clear_lower = apply_calibrator(clear_lower_cal, pd.Series(clear_lower_model.predict_proba(inner_val[selected_features])[:, 1], index=inner_val.index))
            clear_upper = apply_calibrator(clear_upper_cal, pd.Series(clear_upper_model.predict_proba(inner_val[selected_features])[:, 1], index=inner_val.index))
            retest_lower = apply_calibrator(retest_lower_cal, pd.Series(retest_lower_model.predict_proba(inner_val[selected_features])[:, 1], index=inner_val.index))
            retest_upper = apply_calibrator(retest_upper_cal, pd.Series(retest_upper_model.predict_proba(inner_val[selected_features])[:, 1], index=inner_val.index))
            center_prob = pd.Series(center_model.predict_proba(inner_val[selected_features])[:, 1], index=inner_val.index, dtype=float)
            decision = two_tier_decision(clear_lower, clear_upper, retest_lower, retest_upper, center_prob, candidate)
            scored = inner_val.copy()
            scored["decision"] = decision
            inner_rows.append(semantic_candidate_metrics(scored, "decision"))

        metrics_frame = pd.DataFrame(inner_rows)
        candidate_rows.append(
            {
                **candidate,
                "macro_f1_vs_truth": float(metrics_frame["macro_f1_vs_truth"].mean()),
                "unacceptable_false_clear_rate": float(metrics_frame["unacceptable_false_clear_rate"].mean()),
                "unacceptable_recall": float(metrics_frame["unacceptable_recall"].mean()),
                "unacceptable_rate": float(metrics_frame["unacceptable_rate"].mean()),
                "retest_rate": float(metrics_frame["retest_rate"].mean()),
                "premium_precision": float(metrics_frame["premium_precision"].mean()),
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows).sort_values("candidate_name").reset_index(drop=True)
    chosen = candidate_frame.sort_values(
        ["unacceptable_false_clear_rate", "unacceptable_recall", "macro_f1_vs_truth", "premium_precision", "retest_rate"],
        ascending=[True, False, False, False, True],
    ).iloc[0]
    return chosen.to_dict(), candidate_frame


def write_summary_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Center-Band Two-Tier Risk Calibrated Summary",
        "",
        "## Main outcome",
        "",
        f"- centered decision macro_f1: {summary['treatment_g_decision_summary']['macro_f1_vs_truth']:.4f}",
        f"- unacceptable recall: {summary['treatment_g_risk_summary']['unacceptable_recall']:.4f}",
        f"- unacceptable false-clear rate: {summary['treatment_g_risk_summary']['unacceptable_false_clear_rate']:.4f}",
        f"- premium precision: {summary['treatment_g_center_band_summary']['premium_precision_recall']['precision']:.4f}",
        f"- retest rate: {summary['treatment_g_decision_summary']['retest_rate']:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the calibrated two-tier risk experiment.")
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
    risk_cfg = config["risk_targets"]
    feature_rows[f"target_ge_{format_threshold(float(risk_cfg['clear_high_threshold']))}"] = feature_rows["t90"].ge(float(risk_cfg["clear_high_threshold"])).astype(int)
    feature_rows, preclean_summary = preclean_features(feature_rows, config)
    feature_cols = candidate_feature_columns_two_tier(feature_rows)
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
            f"target_ge_{format_threshold(float(risk_cfg['clear_high_threshold']))}",
        ]
    ].copy()
    for col in ["baseline_business_pred", "treatment_g_decision"]:
        predictions[col] = pd.NA
    for label in BUSINESS_LABELS:
        predictions[f"baseline_business_prob_{label}"] = pd.NA
    for col in [
        "treatment_g_clear_lower_risk",
        "treatment_g_clear_upper_risk",
        "treatment_g_retest_lower_risk",
        "treatment_g_retest_upper_risk",
        "treatment_g_center_band_prob",
    ]:
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

        _baseline_cumulative, baseline_probs = fit_cumulative_business_probabilities_inner_thresholds_only(
            train_frame,
            test_frame,
            features=selected_features,
            thresholds=thresholds,
            config=baseline_config,
            candidate=chosen_baseline_candidate,
        )
        baseline_pred = pd.Series(
            [BUSINESS_LABELS[idx] for idx in baseline_probs[BUSINESS_LABELS].to_numpy().argmax(axis=1)],
            index=test_frame.index,
        )

        clear_lower_model, clear_lower_cal = fit_and_calibrate_head(train_frame, features=selected_features, target_col=f"target_lt_{format_threshold(float(risk_cfg['clear_low_threshold']))}", config=config)
        clear_upper_model, clear_upper_cal = fit_and_calibrate_head(train_frame, features=selected_features, target_col=f"target_ge_{format_threshold(float(risk_cfg['clear_high_threshold']))}", config=config)
        retest_lower_model, retest_lower_cal = fit_and_calibrate_head(train_frame, features=selected_features, target_col=f"target_lt_{format_threshold(float(risk_cfg['retest_low_threshold']))}", config=config)
        retest_upper_model, retest_upper_cal = fit_and_calibrate_head(train_frame, features=selected_features, target_col="target_upper_risk", config=config)
        center_model = fit_binary_head(train_frame, selected_features, "center_band_true", config)

        clear_lower = apply_calibrator(clear_lower_cal, pd.Series(clear_lower_model.predict_proba(test_frame[selected_features])[:, 1], index=test_frame.index))
        clear_upper = apply_calibrator(clear_upper_cal, pd.Series(clear_upper_model.predict_proba(test_frame[selected_features])[:, 1], index=test_frame.index))
        retest_lower = apply_calibrator(retest_lower_cal, pd.Series(retest_lower_model.predict_proba(test_frame[selected_features])[:, 1], index=test_frame.index))
        retest_upper = apply_calibrator(retest_upper_cal, pd.Series(retest_upper_model.predict_proba(test_frame[selected_features])[:, 1], index=test_frame.index))
        center_prob = pd.Series(center_model.predict_proba(test_frame[selected_features])[:, 1], index=test_frame.index, dtype=float)
        decision = two_tier_decision(clear_lower, clear_upper, retest_lower, retest_upper, center_prob, chosen_semantic_candidate)

        predictions.loc[test_frame.index, "baseline_business_pred"] = baseline_pred
        for label in BUSINESS_LABELS:
            predictions.loc[test_frame.index, f"baseline_business_prob_{label}"] = baseline_probs[label]
        predictions.loc[test_frame.index, "treatment_g_clear_lower_risk"] = clear_lower
        predictions.loc[test_frame.index, "treatment_g_clear_upper_risk"] = clear_upper
        predictions.loc[test_frame.index, "treatment_g_retest_lower_risk"] = retest_lower
        predictions.loc[test_frame.index, "treatment_g_retest_upper_risk"] = retest_upper
        predictions.loc[test_frame.index, "treatment_g_center_band_prob"] = center_prob
        predictions.loc[test_frame.index, "treatment_g_decision"] = decision

        fold_scored = test_frame.copy()
        fold_scored["treatment_g_decision"] = decision
        baseline_eval = evaluate_prediction_set(test_frame, baseline_probs, baseline_pred)
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
                "treatment_g_macro_f1_vs_truth": float(f1_score(fold_scored["centered_truth"], fold_scored["treatment_g_decision"], labels=CENTERED_LABELS, average="macro", zero_division=0)),
                "treatment_g_unacceptable_rate": float(fold_scored["treatment_g_decision"].eq("unacceptable").mean()),
                "treatment_g_retest_rate": float(fold_scored["treatment_g_decision"].eq("retest").mean()),
                "selected_sensor_scores": json.dumps(sensor_scores, ensure_ascii=False),
            }
        )

    for col in [c for c in predictions.columns if c not in {"sample_time", "business_label", "centered_truth", "baseline_business_pred", "treatment_g_decision"}]:
        predictions[col] = pd.to_numeric(predictions[col], errors="coerce")
    scored_predictions = predictions.loc[predictions["baseline_business_pred"].notna()].copy()
    baseline_probabilities = scored_predictions[[f"baseline_business_prob_{label}" for label in BUSINESS_LABELS]].rename(columns={f"baseline_business_prob_{label}": label for label in BUSINESS_LABELS})
    baseline_threshold_summary = evaluate_prediction_set(scored_predictions, baseline_probabilities, scored_predictions["baseline_business_pred"])
    treatment_g_risk_summary = risk_summary(scored_predictions, "treatment_g_retest_lower_risk", "treatment_g_retest_upper_risk", "treatment_g_decision")
    treatment_g_center_band_summary = center_band_summary(scored_predictions, "treatment_g_center_band_prob", "treatment_g_decision")
    treatment_g_decision_summary = decision_summary(scored_predictions, "treatment_g_decision")

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
            **dcs_audit,
        },
        "preclean_summary": preclean_summary,
        "baseline_threshold_summary": baseline_threshold_summary,
        "treatment_g_risk_summary": treatment_g_risk_summary,
        "treatment_g_center_band_summary": treatment_g_center_band_summary,
        "treatment_g_decision_summary": treatment_g_decision_summary,
        "selected_sensor_union_count": int(len(sensor_counter)),
        "stable_sensor_count": int(sum(1 for count in sensor_counter.values() if count >= 4)),
        "selected_sensor_frequency": dict(sorted(sensor_counter.items(), key=lambda item: (-item[1], item[0]))),
        "per_fold": per_fold_rows,
    }

    feature_rows.to_csv(artifacts_dir / "centered_quality_center_band_two_tier_risk_calibrated_feature_rows.csv", index=False, encoding="utf-8-sig")
    scored_predictions.to_csv(artifacts_dir / "centered_quality_center_band_two_tier_risk_calibrated_results.csv", index=False, encoding="utf-8-sig")
    per_fold_frame.to_csv(artifacts_dir / "centered_quality_center_band_two_tier_risk_calibrated_per_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(baseline_candidate_rows).to_csv(artifacts_dir / "centered_quality_center_band_two_tier_risk_calibrated_baseline_candidate_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(semantic_candidate_rows).to_csv(artifacts_dir / "centered_quality_center_band_two_tier_risk_calibrated_semantic_candidate_search.csv", index=False, encoding="utf-8-sig")
    alias_frame.to_csv(artifacts_dir / "sensor_identity_alias_pairs.csv", index=False, encoding="utf-8-sig")
    (artifacts_dir / "centered_quality_center_band_two_tier_risk_calibrated_summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_report(reports_dir / "centered_quality_center_band_two_tier_risk_calibrated_summary.md", json_ready(summary))
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
