from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from test_probability_anomaly_guard import (
    DEFAULT_SOURCE_RUN_DIR,
    calibrate_iso_soft,
    choose_guarded_scenarios,
    feature_meta,
    feature_shift_scores,
    fit_predict,
    json_ready,
    probability_metrics,
    screen_features,
    select_reduced_features,
)


def risk_label(score: float, *, p90: float, p95: float, known_quality_anomaly: bool, month_shift_anomaly: bool) -> tuple[str, str]:
    if known_quality_anomaly:
        return "quality_anomaly", "low_quality_anomaly"
    if month_shift_anomaly:
        if score > p95:
            return "month_and_sample_feature_shift", "low_feature_shift"
        return "month_feature_shift", "medium_month_shift"
    if score > p95:
        return "high_feature_shift", "low_feature_shift"
    if score > p90:
        return "medium_feature_shift", "medium_feature_shift"
    return "normal_range", "high"


def safe_metrics(group: pd.DataFrame, prob_col: str) -> dict[str, float]:
    if group.empty:
        return {}
    y = group["is_out_spec_obs"].to_numpy(dtype=int)
    p = np.clip(group[prob_col].to_numpy(dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    out = {
        "samples": int(len(group)),
        "observed_out_spec_rate": float(y.mean()),
        "soft_out_prob_mean": float(group["p_out_soft"].mean()),
        "predicted_prob_mean": float(p.mean()),
        "observed_brier": float(brier_score_loss(y, p)),
        "observed_log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    if len(np.unique(y)) > 1:
        out["observed_average_precision"] = float(average_precision_score(y, p))
        out["observed_roc_auc"] = float(roc_auc_score(y, p))
    else:
        out["observed_average_precision"] = math.nan
        out["observed_roc_auc"] = math.nan
    return out


def build_group_metrics(scored: pd.DataFrame, *, prob_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["scenario", "drift_risk_flag", "calibration_confidence"]
    for keys, sub in scored.groupby(group_cols, dropna=False):
        scenario, risk, confidence = keys
        rows.append(
            {
                "scenario": scenario,
                "probability_column": prob_col,
                "drift_risk_flag": risk,
                "calibration_confidence": confidence,
                **safe_metrics(sub, prob_col),
            }
        )
    return pd.DataFrame(rows)


def build_month_metrics(scored: pd.DataFrame, *, prob_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, month), sub in scored.groupby(["scenario", "month"], dropna=False):
        rows.append(
            {
                "scenario": scenario,
                "month": month,
                "probability_column": prob_col,
                "feature_shift_score_median": float(sub["feature_shift_score"].median()),
                "feature_shift_score_p90": float(sub["feature_shift_score"].quantile(0.90)),
                "month_feature_shift_anomaly": bool(sub["month_feature_shift_anomaly"].iloc[0]),
                "known_quality_anomaly_month": bool(sub["known_quality_anomaly_month"].iloc[0]),
                **safe_metrics(sub, prob_col),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add drift-risk and calibration-confidence flags to probability outputs.")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--run-tag", type=str, default="exp030_probability_drift_confidence")
    parser.add_argument("--anomaly-months", nargs="+", default=["2025-06"])
    parser.add_argument("--max-missing-rate", type=float, default=0.95)
    parser.add_argument("--min-variance", type=float, default=1.0e-12)
    parser.add_argument("--reduced-max-features", type=int, default=300)
    parser.add_argument("--reduced-candidate-pool", type=int, default=900)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    parser.add_argument("--month-shift-quantile", type=float, default=0.95)
    args = parser.parse_args()

    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = args.source_run_dir.parent / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    anomaly_months = set(str(m) for m in args.anomaly_months)
    feature_table_path = args.source_run_dir / "two_stage_feature_table.parquet"
    lag_policy = pd.read_csv(args.source_run_dir / "halogen_lag_policy.csv")
    frame = pd.read_parquet(feature_table_path)
    source_shape = [int(frame.shape[0]), int(frame.shape[1])]
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], errors="coerce")
    frame["month"] = frame["sample_time"].dt.to_period("M").astype(str)
    frame["p_out_soft"] = 1.0 - pd.to_numeric(frame["p_pass_soft"], errors="coerce").clip(0.0, 1.0)
    features = [
        col
        for col in frame.columns
        if col.startswith(("hal__", "hal_lag", "ref__", "lims_ctx__")) and not col.endswith("__count")
    ]

    scenario_rows: list[dict[str, Any]] = []
    scored_parts: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for item in choose_guarded_scenarios(frame, anomaly_months):
        scenario = item["scenario"]
        train = item["train"]
        calib = item["calib"]
        test = item["test"]
        screened, screen_report = screen_features(
            train,
            features,
            max_missing_rate=args.max_missing_rate,
            min_variance=args.min_variance,
        )
        selected, selection_report = select_reduced_features(
            train,
            screened,
            max_features=args.reduced_max_features,
            candidate_pool_size=args.reduced_candidate_pool,
            corr_threshold=args.corr_threshold,
        )
        calib_raw, test_raw = fit_predict(train, calib, test, selected)
        test_iso = calibrate_iso_soft(calib_raw, test_raw, calib)

        calib_shift = feature_shift_scores(train, calib, selected)
        test_shift = feature_shift_scores(train, test, selected)
        p90 = float(calib_shift.quantile(0.90))
        p95 = float(calib_shift.quantile(0.95))
        month_threshold = float(calib_shift.quantile(args.month_shift_quantile))
        month_shift = test.assign(feature_shift_score=test_shift.to_numpy()).groupby("month")["feature_shift_score"].median()
        month_shift_anomaly = {month: float(value) > month_threshold for month, value in month_shift.items()}

        scored = test[
            [
                "sample_id",
                "sample_time",
                "month",
                "t90",
                "p_pass_soft",
                "p_out_soft",
                "is_out_spec_obs",
                "sample_weight",
            ]
        ].copy()
        scored.insert(0, "scenario", scenario)
        scored["prob_out_spec_raw"] = np.clip(test_raw, 0.0, 1.0)
        scored["prob_out_spec_iso_soft"] = np.clip(test_iso, 0.0, 1.0)
        scored["feature_shift_score"] = test_shift.to_numpy()
        scored["known_quality_anomaly_month"] = scored["month"].isin(anomaly_months)
        scored["month_feature_shift_median"] = scored["month"].map(month_shift.to_dict()).astype(float)
        scored["month_feature_shift_anomaly"] = scored["month"].map(month_shift_anomaly).fillna(False).astype(bool)
        labels = [
            risk_label(
                float(row.feature_shift_score),
                p90=p90,
                p95=p95,
                known_quality_anomaly=bool(row.known_quality_anomaly_month),
                month_shift_anomaly=bool(row.month_feature_shift_anomaly),
            )
            for row in scored.itertuples(index=False)
        ]
        scored["drift_risk_flag"] = [item[0] for item in labels]
        scored["calibration_confidence"] = [item[1] for item in labels]
        scored["calib_shift_p90"] = p90
        scored["calib_shift_p95"] = p95
        scored["requested_calib_month"] = item["requested_calib_month"]
        scored["used_calib_month"] = item["used_calib_month"]
        scored["calib_guard_triggered"] = bool(item["calib_guard_triggered"])
        scored_parts.append(scored)

        threshold_rows.append(
            {
                "scenario": scenario,
                "train_samples": int(len(train)),
                "calib_samples": int(len(calib)),
                "test_samples": int(len(test)),
                "requested_calib_month": item["requested_calib_month"],
                "used_calib_month": item["used_calib_month"],
                "calib_guard_triggered": bool(item["calib_guard_triggered"]),
                "selected_feature_count": int(len(selected)),
                "calib_shift_p50": float(calib_shift.quantile(0.50)),
                "calib_shift_p75": float(calib_shift.quantile(0.75)),
                "calib_shift_p90": p90,
                "calib_shift_p95": p95,
                "month_shift_threshold": month_threshold,
                "test_shift_median": float(test_shift.median()),
                "test_shift_p90": float(test_shift.quantile(0.90)),
                "test_shift_p95": float(test_shift.quantile(0.95)),
            }
        )

        for variant, pred in {"raw": test_raw, "iso_soft": test_iso}.items():
            metrics = probability_metrics(test, pred)
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "feature_count": int(len(selected)),
                    "train_samples": int(len(train)),
                    "calib_samples": int(len(calib)),
                    "test_samples": int(len(test)),
                    "requested_calib_month": item["requested_calib_month"],
                    "used_calib_month": item["used_calib_month"],
                    "calib_guard_triggered": bool(item["calib_guard_triggered"]),
                    "test_months": ",".join(sorted(test["month"].unique())),
                    **screen_report,
                    **selection_report,
                    **metrics,
                }
            )
        for rank, feature in enumerate(selected, start=1):
            selected_rows.append({"scenario": scenario, "rank": rank, **feature_meta(feature, lag_policy)})

    scored_df = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    scenario_df = pd.DataFrame(scenario_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    selected_df = pd.DataFrame(selected_rows)
    group_iso = build_group_metrics(scored_df, prob_col="prob_out_spec_iso_soft")
    group_raw = build_group_metrics(scored_df, prob_col="prob_out_spec_raw")
    group_df = pd.concat([group_raw, group_iso], ignore_index=True)
    month_iso = build_month_metrics(scored_df, prob_col="prob_out_spec_iso_soft")
    month_raw = build_month_metrics(scored_df, prob_col="prob_out_spec_raw")
    month_df = pd.concat([month_raw, month_iso], ignore_index=True)
    variable_summary = (
        selected_df.groupby(["scenario", "source", "stage", "original_variable_wildcard"], as_index=False)
        .agg(selected_feature_count=("feature", "count"))
        .sort_values(["scenario", "selected_feature_count"], ascending=[True, False])
        if not selected_df.empty
        else pd.DataFrame()
    )

    scored_df.to_csv(run_dir / "probability_scored_with_drift_flags.csv", index=False, encoding="utf-8-sig")
    scenario_df.to_csv(run_dir / "probability_drift_scenario_metrics.csv", index=False, encoding="utf-8-sig")
    threshold_df.to_csv(run_dir / "drift_thresholds_by_scenario.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(run_dir / "drift_risk_group_metrics.csv", index=False, encoding="utf-8-sig")
    month_df.to_csv(run_dir / "drift_month_metrics.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(run_dir / "selected_drift_confidence_features.csv", index=False, encoding="utf-8-sig")
    variable_summary.to_csv(run_dir / "selected_drift_confidence_variable_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_feature_table": str(feature_table_path),
        "source_shape": source_shape,
        "modeling_shape_after_columns": [int(frame.shape[0]), int(frame.shape[1])],
        "anomaly_months": sorted(anomaly_months),
        "candidate_feature_count": int(len(features)),
        "scenario_count": int(scenario_df["scenario"].nunique()) if not scenario_df.empty else 0,
        "risk_flag_counts": scored_df["drift_risk_flag"].value_counts().to_dict() if not scored_df.empty else {},
        "confidence_counts": scored_df["calibration_confidence"].value_counts().to_dict() if not scored_df.empty else {},
        "scenario_metrics": json_ready(scenario_df.to_dict(orient="records")),
    }
    (run_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
