from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_RUN_DIR = PROJECT_DIR / "outputs" / "20260416_124452_exp025_reassigned_stage_lags_ref10_limsctx"

HAL_RE = re.compile(r"^hal_lag(?P<lag>\d+)__(?P<raw>.+)_(?P<kind>lvl3|mean5|mad5|act5|slope5|range5)_L(?P<L>\d+)$")
REF_RE = re.compile(r"^ref__(?P<root>.+)_(?P<kind>act5|lvl3|mad5|mean5|range5|slope5|ageMin|lastStep|stepCnt30|val)_L(?P<L>\d+)$")
LIMS_RE = re.compile(r"^lims_ctx__(?P<name>.+)__mean$")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def screen_features(train: pd.DataFrame, features: list[str], *, max_missing_rate: float, min_variance: float) -> tuple[list[str], dict[str, Any]]:
    numeric = train[features].apply(pd.to_numeric, errors="coerce")
    missing = numeric.isna().mean()
    kept = [col for col in features if float(missing[col]) <= max_missing_rate]
    variance = numeric[kept].var(skipna=True).fillna(0.0) if kept else pd.Series(dtype=float)
    kept2 = [col for col in kept if float(variance[col]) > min_variance]
    return kept2, {
        "candidate_feature_count": int(len(features)),
        "after_missing_count": int(len(kept)),
        "screened_feature_count": int(len(kept2)),
        "max_missing_rate": float(max_missing_rate),
        "min_variance": float(min_variance),
    }


def select_reduced_features(
    train: pd.DataFrame,
    features: list[str],
    *,
    max_features: int,
    candidate_pool_size: int,
    corr_threshold: float,
) -> tuple[list[str], dict[str, Any]]:
    y = pd.to_numeric(train["p_out_soft"], errors="coerce")
    usable = y.notna()
    x = train.loc[usable, features].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    yv = y.loc[usable].astype(float)
    y_centered = yv - float(yv.mean())
    x_centered = x - x.mean(axis=0)
    denom = np.sqrt((x_centered.pow(2).sum(axis=0) * float((y_centered**2).sum())).clip(lower=1.0e-30))
    scores = (x_centered.mul(y_centered, axis=0).sum(axis=0).abs() / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    candidates = scores.sort_values(ascending=False).head(candidate_pool_size).index.tolist()
    corr = x[candidates].corr().abs() if candidates else pd.DataFrame()
    selected: list[str] = []
    for col in candidates:
        if len(selected) >= max_features:
            break
        if not selected:
            selected.append(col)
            continue
        max_corr = corr.loc[col, selected].max()
        if pd.isna(max_corr) or float(max_corr) < corr_threshold:
            selected.append(col)
    return selected, {
        "candidate_pool_count": int(len(candidates)),
        "selected_feature_count": int(len(selected)),
        "top_ranked_features": candidates[:50],
    }


def feature_meta(feature: str, lag_policy: pd.DataFrame) -> dict[str, Any]:
    m = HAL_RE.match(feature)
    if m:
        raw = m.group("raw")
        policy = lag_policy[lag_policy["raw_column"] == raw]
        return {
            "feature": feature,
            "source": "halogen",
            "stage": policy["stage"].iloc[0] if not policy.empty else "",
            "original_variable_wildcard": f"{raw}_*",
            "lag": f"lag{m.group('lag')}",
            "kind": m.group("kind"),
            "lag_index": f"L{m.group('L')}",
        }
    m = REF_RE.match(feature)
    if m:
        root = m.group("root")
        return {
            "feature": feature,
            "source": "refining",
            "stage": f"ref_{root.split('_', 1)[0]}",
            "original_variable_wildcard": f"{root}_*",
            "lag": "snap10",
            "kind": m.group("kind"),
            "lag_index": f"L{m.group('L')}",
        }
    m = LIMS_RE.match(feature)
    if m:
        name = m.group("name")
        return {
            "feature": feature,
            "source": "lims_context",
            "stage": "lims_context",
            "original_variable_wildcard": f"{name}_*",
            "lag": "context",
            "kind": "mean",
            "lag_index": "",
        }
    return {
        "feature": feature,
        "source": "unknown",
        "stage": "",
        "original_variable_wildcard": "",
        "lag": "",
        "kind": "",
        "lag_index": "",
    }


def fit_predict(train: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_calib = imputer.transform(calib[features])
    x_test = imputer.transform(test[features])
    model = GradientBoostingRegressor(
        loss="huber",
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=10,
        alpha=0.9,
        random_state=20260416,
    )
    sample_weight = train["sample_weight"].to_numpy(dtype=float) if "sample_weight" in train else None
    model.fit(x_train, train["p_out_soft"].to_numpy(dtype=float), sample_weight=sample_weight)
    return np.clip(model.predict(x_calib), 0.0, 1.0), np.clip(model.predict(x_test), 0.0, 1.0)


def calibrate_iso_soft(calib_pred: np.ndarray, test_pred: np.ndarray, calib: pd.DataFrame) -> np.ndarray:
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.asarray(calib_pred, dtype=float), calib["p_out_soft"].to_numpy(dtype=float))
    return np.clip(calibrator.transform(np.asarray(test_pred, dtype=float)), 0.0, 1.0)


def probability_metrics(test: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(pred, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    soft = test["p_out_soft"].to_numpy(dtype=float)
    y = test["is_out_spec_obs"].to_numpy(dtype=int)
    corr = spearmanr(soft, p, nan_policy="omit").correlation if len(test) >= 3 else math.nan
    return {
        "samples_test": int(len(test)),
        "observed_out_spec_rate": float(y.mean()),
        "soft_out_prob_mean": float(soft.mean()),
        "predicted_prob_mean": float(p.mean()),
        "soft_mae": float(mean_absolute_error(soft, p)),
        "soft_rmse": float(np.sqrt(mean_squared_error(soft, p))),
        "soft_spearman": float(corr) if corr is not None and not math.isnan(float(corr)) else math.nan,
        "observed_brier": float(brier_score_loss(y, p)),
        "observed_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "observed_average_precision": float(average_precision_score(y, p)),
        "observed_roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan,
    }


def calibration_bins(test: pd.DataFrame, pred: np.ndarray, *, scenario: str, variant: str, n_bins: int) -> pd.DataFrame:
    work = test[["sample_id", "sample_time", "month", "t90", "p_out_soft", "is_out_spec_obs"]].copy()
    work["pred_prob"] = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    q = min(int(n_bins), max(int(len(work) // 10), 2))
    work["bin"] = pd.qcut(work["pred_prob"].rank(method="first"), q=q, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for bin_id, sub in work.groupby("bin"):
        rows.append(
            {
                "scenario": scenario,
                "variant": variant,
                "bin": int(bin_id),
                "samples": int(len(sub)),
                "pred_prob_mean": float(sub["pred_prob"].mean()),
                "observed_out_spec_rate": float(sub["is_out_spec_obs"].mean()),
                "soft_out_prob_mean": float(sub["p_out_soft"].mean()),
                "t90_mean": float(sub["t90"].mean()),
            }
        )
    return pd.DataFrame(rows)


def ece_from_bins(bins: pd.DataFrame) -> dict[str, float]:
    total = max(float(bins["samples"].sum()), 1.0)
    return {
        "observed_ece": float(((bins["samples"] / total) * (bins["pred_prob_mean"] - bins["observed_out_spec_rate"]).abs()).sum()),
        "soft_label_ece": float(((bins["samples"] / total) * (bins["pred_prob_mean"] - bins["soft_out_prob_mean"]).abs()).sum()),
    }


def month_profile(frame: pd.DataFrame, anomaly_months: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, sub in frame.groupby("month"):
        rows.append(
            {
                "month": month,
                "samples": int(len(sub)),
                "observed_out_spec_rate": float(sub["is_out_spec_obs"].mean()),
                "soft_out_prob_mean": float(sub["p_out_soft"].mean()),
                "t90_mean": float(sub["t90"].mean()),
                "known_anomaly_month": month in anomaly_months,
            }
        )
    return pd.DataFrame(rows).sort_values("month")


def choose_guarded_scenarios(frame: pd.DataFrame, anomaly_months: set[str]) -> list[dict[str, Any]]:
    months = sorted(frame["month"].dropna().astype(str).unique())
    scenarios: list[dict[str, Any]] = []

    base_train = frame[(frame["split"] == "train") & (~frame["month"].isin(anomaly_months))].copy()
    base_calib = frame[(frame["split"] == "valid") & (~frame["month"].isin(anomaly_months))].copy()
    base_test = frame[frame["split"] == "test"].copy()
    scenarios.append(
        {
            "scenario": "fixed_guard_drop_anomaly_train_calib",
            "train": base_train,
            "calib": base_calib,
            "test": base_test,
            "requested_calib_month": ",".join(sorted(frame.loc[frame["split"] == "valid", "month"].unique())),
            "used_calib_month": ",".join(sorted(base_calib["month"].unique())),
            "calib_guard_triggered": False,
        }
    )

    for idx in range(2, len(months)):
        test_month = months[idx]
        requested_calib_month = months[idx - 1]
        historical_months = months[: idx - 1]
        train_months = [m for m in historical_months if m not in anomaly_months]
        calib_month = requested_calib_month
        calib_guard = False
        if requested_calib_month in anomaly_months:
            calib_guard = True
            normal_before = [m for m in historical_months if m not in anomaly_months]
            if not normal_before:
                continue
            calib_month = normal_before[-1]
            train_months = [m for m in normal_before if m != calib_month]
        if not train_months:
            continue
        train = frame[frame["month"].isin(train_months)].copy()
        calib = frame[frame["month"] == calib_month].copy()
        test = frame[frame["month"] == test_month].copy()
        if len(train) >= 250 and len(calib) >= 30 and len(test) >= 30:
            scenarios.append(
                {
                    "scenario": f"rolling_guard_test_{test_month}",
                    "train": train,
                    "calib": calib,
                    "test": test,
                    "requested_calib_month": requested_calib_month,
                    "used_calib_month": calib_month,
                    "calib_guard_triggered": calib_guard,
                }
            )
    return scenarios


def feature_shift_scores(train: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> pd.Series:
    baseline = train[features].apply(pd.to_numeric, errors="coerce")
    target_x = target[features].apply(pd.to_numeric, errors="coerce")
    med = baseline.median(axis=0)
    mad = (baseline - med).abs().median(axis=0).replace(0.0, np.nan)
    z = (target_x - med).abs().div(1.4826 * mad, axis=1)
    z = z.replace([np.inf, -np.inf], np.nan).clip(upper=20.0)
    return z.median(axis=1, skipna=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probability model with anomaly-month training/calibration guard.")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--run-tag", type=str, default="exp029_probability_anomaly_guard")
    parser.add_argument("--anomaly-months", nargs="+", default=["2025-06"])
    parser.add_argument("--max-missing-rate", type=float, default=0.95)
    parser.add_argument("--min-variance", type=float, default=1.0e-12)
    parser.add_argument("--reduced-max-features", type=int, default=300)
    parser.add_argument("--reduced-candidate-pool", type=int, default=900)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    parser.add_argument("--calibration-bins", type=int, default=10)
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

    profile_df = month_profile(frame, anomaly_months)
    profile_df.to_csv(run_dir / "anomaly_month_profile.csv", index=False, encoding="utf-8-sig")

    results: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    shift_rows: list[dict[str, Any]] = []

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
        predictions = {
            "raw_guarded": test_raw,
            "iso_soft_guarded": calibrate_iso_soft(calib_raw, test_raw, calib),
        }
        shift = feature_shift_scores(train, test, selected)
        shift_rows.append(
            {
                "scenario": scenario,
                "test_months": ",".join(sorted(test["month"].unique())),
                "feature_shift_median": float(shift.median(skipna=True)),
                "feature_shift_p90": float(shift.quantile(0.90)),
                "feature_shift_p95": float(shift.quantile(0.95)),
            }
        )
        for variant, pred in predictions.items():
            bins = calibration_bins(test, pred, scenario=scenario, variant=variant, n_bins=args.calibration_bins)
            calibration_frames.append(bins)
            metrics = probability_metrics(test, pred)
            metrics.update(ece_from_bins(bins))
            results.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "feature_count": int(len(selected)),
                    "train_samples": int(len(train)),
                    "calib_samples": int(len(calib)),
                    "test_samples": int(len(test)),
                    "train_months": ",".join(sorted(train["month"].unique())),
                    "requested_calib_month": item["requested_calib_month"],
                    "used_calib_month": item["used_calib_month"],
                    "calib_guard_triggered": bool(item["calib_guard_triggered"]),
                    "test_months": ",".join(sorted(test["month"].unique())),
                    "train_out_spec_rate": float(train["is_out_spec_obs"].mean()),
                    "calib_out_spec_rate": float(calib["is_out_spec_obs"].mean()),
                    "test_out_spec_rate": float(test["is_out_spec_obs"].mean()),
                    "train_soft_out_prob_mean": float(train["p_out_soft"].mean()),
                    "calib_soft_out_prob_mean": float(calib["p_out_soft"].mean()),
                    "test_soft_out_prob_mean": float(test["p_out_soft"].mean()),
                    **screen_report,
                    **selection_report,
                    **metrics,
                }
            )
        for rank, feature in enumerate(selected, start=1):
            selected_rows.append({"scenario": scenario, "rank": rank, **feature_meta(feature, lag_policy)})

    results_df = pd.DataFrame(results)
    selected_df = pd.DataFrame(selected_rows)
    calibration_df = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    shift_df = pd.DataFrame(shift_rows)
    variable_summary = (
        selected_df.groupby(["scenario", "source", "stage", "original_variable_wildcard"], as_index=False)
        .agg(selected_feature_count=("feature", "count"))
        .sort_values(["scenario", "selected_feature_count"], ascending=[True, False])
        if not selected_df.empty
        else pd.DataFrame()
    )

    results_df.to_csv(run_dir / "probability_anomaly_guard_results.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(run_dir / "selected_anomaly_guard_features.csv", index=False, encoding="utf-8-sig")
    variable_summary.to_csv(run_dir / "selected_anomaly_guard_variable_summary.csv", index=False, encoding="utf-8-sig")
    calibration_df.to_csv(run_dir / "probability_anomaly_guard_calibration_bins.csv", index=False, encoding="utf-8-sig")
    shift_df.to_csv(run_dir / "feature_shift_guard_scores.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_run_dir": str(args.source_run_dir),
        "source_feature_table": str(feature_table_path),
        "source_shape": source_shape,
        "modeling_shape_after_guard_columns": [int(frame.shape[0]), int(frame.shape[1])],
        "anomaly_months": sorted(anomaly_months),
        "candidate_feature_count": int(len(features)),
        "scenario_count": int(results_df["scenario"].nunique()) if not results_df.empty else 0,
        "results": json_ready(results_df.to_dict(orient="records")),
    }
    (run_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
