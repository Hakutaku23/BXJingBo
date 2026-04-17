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


def fit_raw_probability(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_valid = imputer.transform(valid[features])
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
    return np.clip(model.predict(x_valid), 0.0, 1.0), np.clip(model.predict(x_test), 0.0, 1.0)


def calibrate_iso_soft(valid_pred: np.ndarray, test_pred: np.ndarray, valid: pd.DataFrame) -> np.ndarray:
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.asarray(valid_pred, dtype=float), valid["p_out_soft"].to_numpy(dtype=float))
    return np.clip(calibrator.transform(np.asarray(test_pred, dtype=float)), 0.0, 1.0)


def probability_metrics(test: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(pred, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    soft = test["p_out_soft"].to_numpy(dtype=float)
    y = test["is_out_spec_obs"].to_numpy(dtype=int)
    corr = spearmanr(soft, p, nan_policy="omit").correlation if len(test) >= 3 else math.nan
    return {
        "samples_test": int(len(test)),
        "test_months": ",".join(sorted(test["month"].astype(str).unique())),
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


def calibration_bins(test: pd.DataFrame, pred: np.ndarray, *, scenario: str, model_variant: str, n_bins: int) -> pd.DataFrame:
    work = test[["sample_id", "sample_time", "month", "t90", "p_out_soft", "is_out_spec_obs"]].copy()
    work["pred_prob"] = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    q = min(int(n_bins), max(int(len(work) // 10), 2))
    work["bin"] = pd.qcut(work["pred_prob"].rank(method="first"), q=q, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for bin_id, sub in work.groupby("bin"):
        rows.append(
            {
                "scenario": scenario,
                "model_variant": model_variant,
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


def scenario_profile(name: str, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    return {
        "scenario": name,
        "train_samples": int(len(train)),
        "valid_samples": int(len(valid)),
        "test_samples": int(len(test)),
        "train_months": ",".join(sorted(train["month"].astype(str).unique())),
        "valid_months": ",".join(sorted(valid["month"].astype(str).unique())),
        "test_months": ",".join(sorted(test["month"].astype(str).unique())),
        "train_out_spec_rate": float(train["is_out_spec_obs"].mean()),
        "valid_out_spec_rate": float(valid["is_out_spec_obs"].mean()),
        "test_out_spec_rate": float(test["is_out_spec_obs"].mean()),
        "train_soft_out_prob_mean": float(train["p_out_soft"].mean()),
        "valid_soft_out_prob_mean": float(valid["p_out_soft"].mean()),
        "test_soft_out_prob_mean": float(test["p_out_soft"].mean()),
    }


def build_scenarios(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    scenarios: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    base_train = frame[frame["split"] == "train"].copy()
    base_valid = frame[frame["split"] == "valid"].copy()
    base_test = frame[frame["split"] == "test"].copy()
    scenarios.append(("existing_split", base_train, base_valid, base_test))
    scenarios.append(("existing_split_train_without_2025_06", base_train[base_train["month"] != "2025-06"].copy(), base_valid, base_test))

    months = sorted(frame["month"].dropna().astype(str).unique())
    for idx in range(2, len(months)):
        train_months = months[: idx - 1]
        valid_month = months[idx - 1]
        test_month = months[idx]
        train = frame[frame["month"].isin(train_months)].copy()
        valid = frame[frame["month"] == valid_month].copy()
        test = frame[frame["month"] == test_month].copy()
        if len(train) >= 300 and len(valid) >= 30 and len(test) >= 30:
            scenarios.append((f"rolling_test_{test_month}", train, valid, test))
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-stratified tests for the T90 out-of-spec probability model.")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--run-tag", type=str, default="exp028_probability_time_tests")
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

    results: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []

    for scenario, train, valid, test in build_scenarios(frame):
        if train.empty or valid.empty or test.empty:
            continue
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
        valid_raw, test_raw = fit_raw_probability(train, valid, test, selected)
        predictions = {
            "raw": test_raw,
            "iso_soft": calibrate_iso_soft(valid_raw, test_raw, valid),
        }
        for variant, pred in predictions.items():
            bins = calibration_bins(test, pred, scenario=scenario, model_variant=variant, n_bins=args.calibration_bins)
            calibration_rows.append(bins)
            metrics = probability_metrics(test, pred)
            metrics.update(ece_from_bins(bins))
            results.append(
                {
                    "scenario": scenario,
                    "model_variant": variant,
                    "feature_count": int(len(selected)),
                    **scenario_profile(scenario, train, valid, test),
                    **screen_report,
                    **selection_report,
                    **metrics,
                }
            )
        profiles.append(scenario_profile(scenario, train, valid, test))
        for rank, feature in enumerate(selected, start=1):
            selected_rows.append({"scenario": scenario, "rank": rank, **feature_meta(feature, lag_policy)})

    result_df = pd.DataFrame(results)
    profile_df = pd.DataFrame(profiles)
    selected_df = pd.DataFrame(selected_rows)
    calibration_df = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    variable_summary = (
        selected_df.groupby(["scenario", "source", "stage", "original_variable_wildcard"], as_index=False)
        .agg(selected_feature_count=("feature", "count"))
        .sort_values(["scenario", "selected_feature_count"], ascending=[True, False])
        if not selected_df.empty
        else pd.DataFrame()
    )

    result_df.to_csv(run_dir / "probability_time_test_results.csv", index=False, encoding="utf-8-sig")
    profile_df.to_csv(run_dir / "probability_time_test_profiles.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(run_dir / "selected_probability_time_features.csv", index=False, encoding="utf-8-sig")
    variable_summary.to_csv(run_dir / "selected_probability_time_variable_summary.csv", index=False, encoding="utf-8-sig")
    calibration_df.to_csv(run_dir / "probability_time_calibration_bins.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_run_dir": str(args.source_run_dir),
        "source_feature_table": str(feature_table_path),
        "source_shape": source_shape,
        "modeling_shape_after_time_target_columns": [int(frame.shape[0]), int(frame.shape[1])],
        "candidate_feature_count": int(len(features)),
        "scenario_count": int(profile_df["scenario"].nunique()) if not profile_df.empty else 0,
        "results": json_ready(result_df.to_dict(orient="records")),
    }
    (run_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
