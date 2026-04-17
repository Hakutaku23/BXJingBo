from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from test_probability_anomaly_guard import (
    DEFAULT_SOURCE_RUN_DIR,
    calibrate_iso_soft,
    feature_meta,
    fit_predict,
    json_ready,
    probability_metrics,
    screen_features,
    select_reduced_features,
)
from test_probability_drift_confidence import risk_label


def safe_metrics(frame: pd.DataFrame, prob_col: str) -> dict[str, float]:
    y = frame["is_out_spec_obs"].to_numpy(dtype=int)
    p = np.clip(frame[prob_col].to_numpy(dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    soft = frame["p_out_soft"].to_numpy(dtype=float)
    out = {
        "samples_test": int(len(frame)),
        "observed_out_spec_rate": float(y.mean()),
        "soft_out_prob_mean": float(soft.mean()),
        "predicted_prob_mean": float(p.mean()),
        "soft_mae": float(mean_absolute_error(soft, p)),
        "soft_rmse": float(np.sqrt(mean_squared_error(soft, p))),
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


def fit_global_bundle(train: pd.DataFrame, calib: pd.DataFrame, features: list[str], *, n_estimators: int) -> tuple[SimpleImputer, GradientBoostingRegressor, IsotonicRegression]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_calib = imputer.transform(calib[features])
    model = GradientBoostingRegressor(
        loss="huber",
        n_estimators=n_estimators,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=10,
        alpha=0.9,
        random_state=20260416,
    )
    sample_weight = train["sample_weight"].to_numpy(dtype=float) if "sample_weight" in train else None
    model.fit(x_train, train["p_out_soft"].to_numpy(dtype=float), sample_weight=sample_weight)
    calib_pred = np.clip(model.predict(x_calib), 0.0, 1.0)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(calib_pred, calib["p_out_soft"].to_numpy(dtype=float))
    return imputer, model, calibrator


def predict_global_bundle(bundle: tuple[SimpleImputer, GradientBoostingRegressor, IsotonicRegression], query: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    imputer, model, calibrator = bundle
    x_query = imputer.transform(query[features])
    raw = np.clip(model.predict(x_query), 0.0, 1.0)
    iso = np.clip(calibrator.transform(raw), 0.0, 1.0)
    return raw, iso


def fit_global_prob(train: pd.DataFrame, calib: pd.DataFrame, query: pd.DataFrame, features: list[str], *, n_estimators: int) -> tuple[np.ndarray, np.ndarray]:
    return predict_global_bundle(fit_global_bundle(train, calib, features, n_estimators=n_estimators), query, features)


def robust_reference(train: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
    numeric = train[features].apply(pd.to_numeric, errors="coerce")
    med = numeric.median(axis=0)
    mad = (numeric - med).abs().median(axis=0)
    scale = (1.4826 * mad).replace(0.0, np.nan)
    fallback = numeric.std(axis=0).replace(0.0, np.nan)
    scale = scale.fillna(fallback).fillna(1.0)
    return med, scale


def transform_robust(frame: pd.DataFrame, features: list[str], med: pd.Series, scale: pd.Series) -> np.ndarray:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    filled = numeric.fillna(med)
    arr = filled.sub(med, axis=1).div(scale, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return arr.to_numpy(dtype=float)


def effective_count(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(np.square(weights).sum())
    if denom <= 0.0:
        return 0.0
    return float(np.square(weights.sum()) / denom)


def make_neighbor_months(months: pd.Series, weights: np.ndarray, topn: int = 4) -> str:
    work = pd.DataFrame({"month": months.astype(str).to_numpy(), "weight": weights})
    agg = work.groupby("month")["weight"].sum().sort_values(ascending=False).head(topn)
    total = max(float(agg.sum()), 1.0e-12)
    return ";".join(f"{idx}:{float(val / total):.3f}" for idx, val in agg.items())


def confidence_from_neighbors(eff_n: float, dist_median: float, calib_p90: float, calib_p95: float, fallback_used: bool) -> str:
    if fallback_used:
        return "fallback_global"
    if eff_n < 15 or dist_median > calib_p95:
        return "low_local"
    if eff_n < 30 or dist_median > calib_p90:
        return "medium_local"
    return "high_local"


def run_online_global(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    anomaly_months: set[str],
    n_estimators: int,
    label_delay_days: float,
) -> pd.DataFrame:
    current_train = train.copy()
    pending: list[pd.Series] = []
    bundle: tuple[SimpleImputer, GradientBoostingRegressor, IsotonicRegression] | None = None
    rows: list[dict[str, Any]] = []
    for pos, (_, query) in enumerate(test.iterrows(), start=1):
        query_df = query.to_frame().T
        query_time = pd.to_datetime(query.get("sample_time"), errors="coerce")
        available_cutoff = query_time - pd.Timedelta(days=float(label_delay_days))
        released: list[pd.Series] = []
        still_pending: list[pd.Series] = []
        for item in pending:
            item_time = pd.to_datetime(item.get("sample_time"), errors="coerce")
            if pd.notna(item_time) and item_time <= available_cutoff:
                released.append(item)
            else:
                still_pending.append(item)
        pending = still_pending
        if released:
            current_train = pd.concat([current_train, pd.DataFrame(released)], ignore_index=True)
            bundle = None
        if bundle is None:
            bundle = fit_global_bundle(current_train, calib, features, n_estimators=n_estimators)
        raw, iso = predict_global_bundle(bundle, query_df, features)
        rows.append(
            {
                "sample_id": query.get("sample_id"),
                "sample_time": query.get("sample_time"),
                "month": query.get("month"),
                "prob_out_spec_online_global_raw": float(raw[0]),
                "prob_out_spec_online_global_iso": float(iso[0]),
                "online_global_train_size": int(len(current_train)),
                "online_global_pending_size": int(len(pending)),
            }
        )
        if str(query.get("month")) not in anomaly_months:
            pending.append(query.copy())
        if pos % 50 == 0:
            print(f"online_global delay={label_delay_days:g}d predicted {pos}/{len(test)}")
    return pd.DataFrame(rows)


def run_jitl(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    static_fallback: np.ndarray,
    drift_flags: pd.Series,
    *,
    anomaly_months: set[str],
    neighbor_k: int,
    min_neighbors: int,
    local_estimators: int,
    model_weight: float,
    recency_half_life_days: float,
    include_calib_in_library: bool,
    calib_shift_p90: float,
    calib_shift_p95: float,
    label_delay_days: float,
    fallback_drift_flags: set[str],
) -> pd.DataFrame:
    med, scale = robust_reference(train, features)
    initial_library = pd.concat([train, calib], ignore_index=True) if include_calib_in_library else train.copy()
    initial_library = initial_library[~initial_library["month"].astype(str).isin(anomaly_months)].copy()
    lib = initial_library.reset_index(drop=True)
    x_lib = transform_robust(lib, features, med, scale)
    y_lib = lib["p_out_soft"].to_numpy(dtype=float)
    sw_lib = lib["sample_weight"].to_numpy(dtype=float) if "sample_weight" in lib else np.ones(len(lib), dtype=float)
    time_lib = pd.to_datetime(lib["sample_time"], errors="coerce")
    pending: list[pd.Series] = []
    rows: list[dict[str, Any]] = []

    for pos, (_, query) in enumerate(test.iterrows(), start=1):
        query_df = query.to_frame().T
        query_time = pd.to_datetime(query.get("sample_time"), errors="coerce")
        available_cutoff = query_time - pd.Timedelta(days=float(label_delay_days))
        released: list[pd.Series] = []
        still_pending: list[pd.Series] = []
        for item in pending:
            item_time = pd.to_datetime(item.get("sample_time"), errors="coerce")
            if pd.notna(item_time) and item_time <= available_cutoff:
                released.append(item)
            else:
                still_pending.append(item)
        pending = still_pending
        if released:
            release_df = pd.DataFrame(released)
            x_release = transform_robust(release_df, features, med, scale)
            x_lib = np.vstack([x_lib, x_release])
            y_lib = np.append(y_lib, release_df["p_out_soft"].to_numpy(dtype=float))
            release_sw = release_df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in release_df else np.ones(len(release_df), dtype=float)
            sw_lib = np.append(sw_lib, release_sw)
            lib = pd.concat([lib, release_df], ignore_index=True)
            time_lib = pd.to_datetime(lib["sample_time"], errors="coerce")
        x_query = transform_robust(query_df, features, med, scale)[0]
        drift_flag = str(drift_flags.iloc[pos - 1])
        drift_fallback = drift_flag in fallback_drift_flags
        distances = np.nanmedian(np.abs(x_lib - x_query), axis=1)
        finite = np.isfinite(distances)
        if not finite.any():
            order = np.arange(len(distances))
        else:
            order = np.argsort(np.where(finite, distances, np.inf))
        chosen = order[: min(neighbor_k, len(order))]
        chosen = chosen[np.isfinite(distances[chosen])]
        fallback_used = len(chosen) < min_neighbors

        if fallback_used:
            local_model_prob = float(static_fallback[pos - 1])
            local_knn_prob = float(static_fallback[pos - 1])
            local_blend_prob = float(static_fallback[pos - 1])
            neighbor_weights = np.array([], dtype=float)
            neighbor_dist = np.array([], dtype=float)
            eff_n = 0.0
            dist_median = math.nan
            dist_p90 = math.nan
            local_obs_rate = math.nan
            local_soft_mean = math.nan
            neighbor_months = ""
        else:
            neighbor_dist = distances[chosen].astype(float)
            bandwidth = max(float(np.nanmedian(neighbor_dist)), 1.0e-6)
            sim = np.exp(-neighbor_dist / bandwidth)
            query_time = pd.to_datetime(query.get("sample_time"), errors="coerce")
            age_days = (query_time - time_lib.iloc[chosen]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
            age_days = np.where(np.isfinite(age_days), np.maximum(age_days, 0.0), 0.0)
            recency = np.exp(-age_days / max(recency_half_life_days, 1.0))
            neighbor_weights = sim * recency * sw_lib[chosen]
            if float(neighbor_weights.sum()) <= 0.0:
                neighbor_weights = np.ones(len(chosen), dtype=float)
            eff_n = effective_count(neighbor_weights)
            dist_median = float(np.nanmedian(neighbor_dist))
            dist_p90 = float(np.nanquantile(neighbor_dist, 0.90))
            local_soft_mean = float(np.average(y_lib[chosen], weights=neighbor_weights))
            local_obs_rate = float(np.average(lib.iloc[chosen]["is_out_spec_obs"].to_numpy(dtype=float), weights=neighbor_weights))
            neighbor_months = make_neighbor_months(lib.iloc[chosen]["month"], neighbor_weights)

            model = GradientBoostingRegressor(
                loss="huber",
                n_estimators=local_estimators,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.9,
                min_samples_leaf=5,
                alpha=0.9,
                random_state=20260416 + pos,
            )
            model.fit(x_lib[chosen], y_lib[chosen], sample_weight=neighbor_weights)
            local_model_prob = float(np.clip(model.predict(x_query.reshape(1, -1))[0], 0.0, 1.0))
            local_knn_prob = float(np.clip(local_soft_mean, 0.0, 1.0))
            local_blend_prob = float(np.clip(model_weight * local_model_prob + (1.0 - model_weight) * local_knn_prob, 0.0, 1.0))

        protected_fallback_used = bool(fallback_used or drift_fallback)
        protected_prob = float(static_fallback[pos - 1]) if protected_fallback_used else local_model_prob
        confidence = confidence_from_neighbors(eff_n, dist_median, calib_shift_p90, calib_shift_p95, fallback_used)
        if drift_fallback:
            confidence = "fallback_high_feature_shift"
        rows.append(
            {
                "sample_id": query.get("sample_id"),
                "sample_time": query.get("sample_time"),
                "month": query.get("month"),
                "prob_out_spec_jitl_model": local_model_prob,
                "prob_out_spec_jitl_knn": local_knn_prob,
                "prob_out_spec_jitl_blend": local_blend_prob,
                "prob_out_spec_jitl_protected": protected_prob,
                "jitl_neighbor_count": int(len(chosen)),
                "jitl_effective_neighbor_count": float(eff_n),
                "jitl_neighbor_distance_median": dist_median,
                "jitl_neighbor_distance_p90": dist_p90,
                "jitl_local_obs_rate": local_obs_rate,
                "jitl_local_soft_mean": local_soft_mean,
                "jitl_neighbor_months": neighbor_months,
                "jitl_fallback_used": bool(fallback_used),
                "jitl_drift_fallback_used": bool(drift_fallback),
                "jitl_protected_fallback_used": bool(protected_fallback_used),
                "jitl_confidence": confidence,
                "jitl_library_size": int(len(lib)),
                "jitl_pending_size": int(len(pending)),
            }
        )

        if str(query.get("month")) not in anomaly_months:
            pending.append(query.copy())
        if pos % 50 == 0:
            print(f"jitl delay={label_delay_days:g}d predicted {pos}/{len(test)}")
    return pd.DataFrame(rows)


def build_drift_flags(train: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame, features: list[str], anomaly_months: set[str], month_shift_quantile: float) -> tuple[pd.DataFrame, dict[str, float]]:
    med, scale = robust_reference(train, features)
    calib_z = transform_robust(calib, features, med, scale)
    test_z = transform_robust(test, features, med, scale)
    calib_shift = pd.Series(np.nanmedian(np.abs(calib_z), axis=1), index=calib.index)
    test_shift = pd.Series(np.nanmedian(np.abs(test_z), axis=1), index=test.index)
    p90 = float(calib_shift.quantile(0.90))
    p95 = float(calib_shift.quantile(0.95))
    month_threshold = float(calib_shift.quantile(month_shift_quantile))
    month_shift = test.assign(feature_shift_score=test_shift.to_numpy()).groupby("month")["feature_shift_score"].median()
    month_shift_anomaly = {month: float(value) > month_threshold for month, value in month_shift.items()}
    out = pd.DataFrame(
        {
            "sample_id": test["sample_id"].to_numpy(),
            "feature_shift_score": test_shift.to_numpy(),
            "known_quality_anomaly_month": test["month"].astype(str).isin(anomaly_months).to_numpy(),
            "month_feature_shift_median": test["month"].map(month_shift.to_dict()).astype(float).to_numpy(),
            "month_feature_shift_anomaly": test["month"].map(month_shift_anomaly).fillna(False).astype(bool).to_numpy(),
        }
    )
    labels = [
        risk_label(
            float(row.feature_shift_score),
            p90=p90,
            p95=p95,
            known_quality_anomaly=bool(row.known_quality_anomaly_month),
            month_shift_anomaly=bool(row.month_feature_shift_anomaly),
        )
        for row in out.itertuples(index=False)
    ]
    out["drift_risk_flag"] = [item[0] for item in labels]
    out["calibration_confidence"] = [item[1] for item in labels]
    return out, {
        "calib_shift_p90": p90,
        "calib_shift_p95": p95,
        "month_shift_threshold": month_threshold,
        "test_shift_median": float(test_shift.median()),
        "test_shift_p90": float(test_shift.quantile(0.90)),
        "test_shift_p95": float(test_shift.quantile(0.95)),
    }


def group_metrics(scored: pd.DataFrame, prob_cols: list[str], group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["label_delay_days", group_col] if "label_delay_days" in scored.columns else [group_col]
    for prob_col in prob_cols:
        for key, sub in scored.groupby(group_cols, dropna=False):
            if len(sub) < 2:
                continue
            if isinstance(key, tuple):
                payload = dict(zip(group_cols, key))
            else:
                payload = {group_col: key}
            rows.append({"probability_column": prob_col, **payload, **safe_metrics(sub, prob_col)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential online and JITL probability experiments.")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--run-tag", type=str, default="exp031_jitl_probability")
    parser.add_argument("--anomaly-months", nargs="+", default=["2025-06"])
    parser.add_argument("--max-missing-rate", type=float, default=0.95)
    parser.add_argument("--min-variance", type=float, default=1.0e-12)
    parser.add_argument("--reduced-max-features", type=int, default=300)
    parser.add_argument("--reduced-candidate-pool", type=int, default=900)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    parser.add_argument("--neighbor-k", type=int, default=80)
    parser.add_argument("--min-neighbors", type=int, default=30)
    parser.add_argument("--local-estimators", type=int, default=80)
    parser.add_argument("--local-model-weight", type=float, default=0.65)
    parser.add_argument("--recency-half-life-days", type=float, default=180.0)
    parser.add_argument("--global-estimators", type=int, default=80)
    parser.add_argument("--month-shift-quantile", type=float, default=0.95)
    parser.add_argument("--label-delays-days", nargs="+", type=float, default=[0.0])
    parser.add_argument("--fallback-drift-flags", nargs="+", default=["high_feature_shift", "month_and_sample_feature_shift", "quality_anomaly"])
    parser.add_argument("--skip-online-global", action="store_true")
    parser.add_argument("--exclude-calib-from-jitl-library", action="store_true")
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

    train = frame[(frame["split"] == "train") & (~frame["month"].isin(anomaly_months))].sort_values("sample_time").copy()
    calib = frame[(frame["split"] == "valid") & (~frame["month"].isin(anomaly_months))].sort_values("sample_time").copy()
    test = frame[frame["split"] == "test"].sort_values("sample_time").copy()

    screened, screen_report = screen_features(train, features, max_missing_rate=args.max_missing_rate, min_variance=args.min_variance)
    selected, selection_report = select_reduced_features(
        train,
        screened,
        max_features=args.reduced_max_features,
        candidate_pool_size=args.reduced_candidate_pool,
        corr_threshold=args.corr_threshold,
    )

    print(f"selected {len(selected)} features from {len(features)} candidates")
    calib_raw, static_raw = fit_predict(train, calib, test, selected)
    static_iso = calibrate_iso_soft(calib_raw, static_raw, calib)
    drift_df, drift_thresholds = build_drift_flags(train, calib, test, selected, anomaly_months, args.month_shift_quantile)

    base_scored = test[["sample_id", "sample_time", "month", "t90", "p_pass_soft", "p_out_soft", "is_out_spec_obs", "sample_weight"]].copy()
    base_scored["prob_out_spec_static_raw"] = np.clip(static_raw, 0.0, 1.0)
    base_scored["prob_out_spec_static_iso"] = np.clip(static_iso, 0.0, 1.0)
    base_scored = base_scored.merge(drift_df, on="sample_id", how="left")

    scored_parts: list[pd.DataFrame] = []
    fallback_drift_flags = {str(item) for item in args.fallback_drift_flags}
    label_delays = sorted({float(item) for item in args.label_delays_days})
    for delay_days in label_delays:
        print(f"running label_delay={delay_days:g} days")
        scored_delay = base_scored.copy()
        if args.skip_online_global:
            online_df = pd.DataFrame({"sample_id": scored_delay["sample_id"]})
        else:
            online_df = run_online_global(
                train,
                calib,
                test,
                selected,
                anomaly_months=anomaly_months,
                n_estimators=args.global_estimators,
                label_delay_days=delay_days,
            )
        scored_delay = scored_delay.merge(online_df, on="sample_id", how="left")

        fallback = scored_delay["prob_out_spec_static_iso"].to_numpy(dtype=float)
        jitl_df = run_jitl(
            train,
            calib,
            test,
            selected,
            fallback,
            scored_delay["drift_risk_flag"],
            anomaly_months=anomaly_months,
            neighbor_k=args.neighbor_k,
            min_neighbors=args.min_neighbors,
            local_estimators=args.local_estimators,
            model_weight=args.local_model_weight,
            recency_half_life_days=args.recency_half_life_days,
            include_calib_in_library=not bool(args.exclude_calib_from_jitl_library),
            calib_shift_p90=drift_thresholds["calib_shift_p90"],
            calib_shift_p95=drift_thresholds["calib_shift_p95"],
            label_delay_days=delay_days,
            fallback_drift_flags=fallback_drift_flags,
        )
        scored_delay = scored_delay.merge(jitl_df, on="sample_id", how="left", suffixes=("", "_jitl"))
        scored_delay["label_delay_days"] = delay_days
        scored_parts.append(scored_delay)

    scored = pd.concat(scored_parts, ignore_index=True)

    prob_cols = [
        "prob_out_spec_static_raw",
        "prob_out_spec_static_iso",
        "prob_out_spec_jitl_knn",
        "prob_out_spec_jitl_model",
        "prob_out_spec_jitl_blend",
        "prob_out_spec_jitl_protected",
    ]
    if not args.skip_online_global:
        prob_cols.extend(["prob_out_spec_online_global_raw", "prob_out_spec_online_global_iso"])

    metrics_rows = []
    for delay_days, sub in scored.groupby("label_delay_days"):
        for col in prob_cols:
            if col in sub.columns:
                metrics_rows.append({"label_delay_days": float(delay_days), "variant": col, **safe_metrics(sub, col)})
    metrics_df = pd.DataFrame(metrics_rows)
    month_metrics = []
    for delay_days, delay_sub in scored.groupby("label_delay_days"):
        for prob_col in prob_cols:
            if prob_col not in delay_sub.columns:
                continue
            for month, sub in delay_sub.groupby("month"):
                month_metrics.append({"label_delay_days": float(delay_days), "probability_column": prob_col, "month": month, **safe_metrics(sub, prob_col)})
    month_metrics_df = pd.DataFrame(month_metrics)
    drift_group_df = group_metrics(scored, prob_cols, "drift_risk_flag")
    jitl_conf_df = group_metrics(scored, prob_cols, "jitl_confidence")

    selected_df = pd.DataFrame([{"rank": rank, **feature_meta(feature, lag_policy)} for rank, feature in enumerate(selected, start=1)])
    variable_summary = (
        selected_df.groupby(["source", "stage", "original_variable_wildcard"], as_index=False)
        .agg(selected_feature_count=("feature", "count"))
        .sort_values("selected_feature_count", ascending=False)
    )

    scored.to_csv(run_dir / "jitl_probability_scored.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(run_dir / "jitl_probability_metrics.csv", index=False, encoding="utf-8-sig")
    month_metrics_df.to_csv(run_dir / "jitl_probability_month_metrics.csv", index=False, encoding="utf-8-sig")
    drift_group_df.to_csv(run_dir / "jitl_probability_drift_group_metrics.csv", index=False, encoding="utf-8-sig")
    jitl_conf_df.to_csv(run_dir / "jitl_probability_confidence_group_metrics.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(run_dir / "jitl_selected_features.csv", index=False, encoding="utf-8-sig")
    variable_summary.to_csv(run_dir / "jitl_selected_variable_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_feature_table": str(feature_table_path),
        "source_shape": source_shape,
        "modeling_shape_after_columns": [int(frame.shape[0]), int(frame.shape[1])],
        "anomaly_months": sorted(anomaly_months),
        "candidate_feature_count": int(len(features)),
        "train_samples": int(len(train)),
        "calib_samples": int(len(calib)),
        "test_samples": int(len(test)),
        "test_months": sorted(test["month"].astype(str).unique().tolist()),
        "label_delays_days": label_delays,
        "screen_report": screen_report,
        "selection_report": {k: v for k, v in selection_report.items() if k != "top_ranked_features"},
        "top_ranked_features": selection_report.get("top_ranked_features", [])[:50],
        "drift_thresholds": drift_thresholds,
        "jitl_config": {
            "neighbor_k": int(args.neighbor_k),
            "min_neighbors": int(args.min_neighbors),
            "local_estimators": int(args.local_estimators),
            "local_model_weight": float(args.local_model_weight),
            "recency_half_life_days": float(args.recency_half_life_days),
            "include_calib_in_library": not bool(args.exclude_calib_from_jitl_library),
            "fallback_drift_flags": sorted(fallback_drift_flags),
        },
        "metrics": metrics_df.to_dict(orient="records"),
        "jitl_confidence_counts": scored["jitl_confidence"].value_counts().to_dict(),
        "drift_risk_counts": scored["drift_risk_flag"].value_counts().to_dict(),
        "jitl_protected_fallback_counts": scored["jitl_protected_fallback_used"].value_counts().to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
