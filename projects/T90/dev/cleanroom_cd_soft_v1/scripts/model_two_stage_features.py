from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cleanroom_cd_soft_v1.config import load_config, resolve_path
from cleanroom_cd_soft_v1.data import load_lims_samples
from cleanroom_cd_soft_v1.labels import add_noise_aware_labels
from cleanroom_cd_soft_v1.split import leakage_audit, make_time_purged_split


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "base.yaml"
DATA_DIR = PROJECT_DIR.parents[1] / "data"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def safe_col(name: str) -> str:
    return str(name).replace("\n", "_").replace("\r", "_").replace(" ", "_")


def infer_halogen_stage(raw_col: str) -> tuple[str, int, str]:
    """Map a DCS point to an approximate process-stage lag from the flow chart."""
    col = str(raw_col).upper()
    if any(token in col for token in ["C500", "C510", "CM510", "51004", "51006", "51007"]):
        return "feed_r510", 180, "feed/V510/R510 first reaction stage"
    if any(token in col for token in ["C511", "CM511"]):
        return "r511", 130, "R511 second reaction stage; about one R510 residence later"
    if any(token in col for token in ["C512", "CM512", "51204", "C517"]):
        return "r512_buffer", 80, "R512 buffer/outlet-temperature substitute region"
    if any(token in col for token in ["C513", "CM513"]):
        return "r513_neutralization", 75, "R513 neutralization stage"
    if any(token in col for token in ["C514", "CM514", "C518", "C516", "C564"]):
        return "r514_additive", 70, "R514 additive/dilution stage"
    if any(token in col for token in ["C530", "CM530", "C30501", "53001", "53002", "53003", "53005"]):
        return "v530_flash", 45, "V530 flash and nearby downstream points"
    if any(token in col for token in ["C532", "CM532", "53202", "53203", "53205", "53252"]):
        return "v532_buffer", 30, "V532 downstream buffer points"
    if any(token in col for token in ["C540", "CM540", "54002", "54003", "54051"]):
        return "v540_t300", 20, "V540/T300 downstream buffer/devolatilization-side points"
    return "unknown_halogen", 120, "unmapped halogenation point; conservative middle-process lag"


def stage_candidate_lags(stage: str, primary_lag: int) -> list[int]:
    candidates = {
        "feed_r510": [150, 180, 210],
        "r511": [100, 130, 160],
        "r512_buffer": [60, 80, 120],
        "r513_neutralization": [60, 75, 120],
        "r514_additive": [45, 70, 100],
        "v530_flash": [15, 45, 75],
        "v532_buffer": [0, 30, 60],
        "v540_t300": [0, 15, 60],
        "unknown_halogen": [60, 120, 180],
    }
    values = candidates.get(stage, [primary_lag])
    return sorted({int(v) for v in values})


def halogen_lag_policy(raw_cols: list[str], *, mode: str, global_lag_minutes: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_col in raw_cols:
        if mode in {"stage_map", "stage_candidates"}:
            stage, lag_minutes, rationale = infer_halogen_stage(raw_col)
        else:
            stage, lag_minutes, rationale = "global", int(global_lag_minutes), "single global halogenation lag"
        candidate_lags = stage_candidate_lags(stage, lag_minutes) if mode == "stage_candidates" else [int(lag_minutes)]
        rows.append(
            {
                "raw_column": raw_col,
                "stage": stage,
                "lag_minutes": int(lag_minutes),
                "candidate_lag_minutes": ",".join(str(v) for v in candidate_lags),
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def asof_snapshot(
    samples: pd.DataFrame,
    values: pd.DataFrame,
    *,
    value_time_col: str,
    lag_minutes: int,
    tolerance_minutes: int,
) -> pd.DataFrame:
    left = samples[["sample_id", "sample_time"]].copy()
    left["_order"] = np.arange(len(left))
    left["asof_time"] = left["sample_time"] - pd.Timedelta(minutes=int(lag_minutes))
    right = values.sort_values(value_time_col).drop_duplicates(subset=[value_time_col], keep="last")
    merged = pd.merge_asof(
        left.sort_values("asof_time"),
        right,
        left_on="asof_time",
        right_on=value_time_col,
        direction="backward",
        tolerance=pd.Timedelta(minutes=int(tolerance_minutes)),
    )
    merged = merged.sort_values("_order").drop(
        columns=["_order", "asof_time", "sample_time", value_time_col],
        errors="ignore",
    )
    return merged.reset_index(drop=True)


def load_refining_features(
    path: Path,
    *,
    time_col: str,
    start_col_1based: int,
    end_col_1based: int,
    include_y_derived: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(path, nrows=0)
    all_cols = list(header.columns)
    selected = all_cols[int(start_col_1based) - 1 : int(end_col_1based)]
    blocked = {"Y", "Lab", "Lab_hold", "Lab_ageMin", "Y_cal"}
    if not include_y_derived:
        selected = [col for col in selected if col not in blocked and not str(col).startswith("Y_L")]
    usecols = [time_col, *selected]
    dtype = {col: "float32" for col in selected}
    frame = pd.read_csv(path, usecols=usecols, dtype=dtype)
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    for col in selected:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    renamed = {col: f"ref__{safe_col(col)}" for col in selected}
    frame = frame.rename(columns=renamed)
    report = {
        "path": str(path),
        "rows": int(len(frame)),
        "time_min": frame[time_col].min().isoformat() if not frame.empty else None,
        "time_max": frame[time_col].max().isoformat() if not frame.empty else None,
        "requested_1based_range": [int(start_col_1based), int(end_col_1based)],
        "include_y_derived": bool(include_y_derived),
        "selected_feature_count": int(len(selected)),
        "selected_first_columns": selected[:20],
        "selected_last_columns": selected[-20:],
    }
    return frame, report


def load_halogen_feature_parts(
    feature_dir: Path,
    samples: pd.DataFrame,
    *,
    lag_mode: str,
    lag_minutes: int,
    tolerance_minutes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = feature_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = manifest["parts"]
    merged_parts: list[pd.DataFrame] = [samples[["sample_id"]].copy()]
    part_reports: list[dict[str, Any]] = []

    raw_cols = [raw_col for part in parts for raw_col in part.get("raw_columns", [])]
    lag_policy = halogen_lag_policy(raw_cols, mode=lag_mode, global_lag_minutes=lag_minutes)
    lag_lookup = lag_policy.set_index("raw_column")["lag_minutes"].to_dict()
    candidate_lag_lookup = {
        row.raw_column: [int(v) for v in str(row.candidate_lag_minutes).split(",") if str(v).strip()]
        for row in lag_policy.itertuples(index=False)
    }
    stage_lookup = lag_policy.set_index("raw_column")["stage"].to_dict()

    for part in parts:
        path = Path(part["path"])
        frame = pd.read_parquet(path)
        snapped_groups: list[pd.DataFrame] = []
        for raw_col in part.get("raw_columns", []):
            raw_feature_cols = [col for col in frame.columns if col.startswith(f"{raw_col}_")]
            if not raw_feature_cols:
                continue
            for candidate_lag in candidate_lag_lookup.get(raw_col, [int(lag_lookup.get(raw_col, lag_minutes))]):
                prefix = f"hal_lag{int(candidate_lag)}__" if lag_mode == "stage_candidates" else "hal__"
                renamed = frame[["time", *raw_feature_cols]].rename(
                    columns={col: f"{prefix}{safe_col(col)}" for col in raw_feature_cols}
                )
                snapped = asof_snapshot(
                    samples,
                    renamed,
                    value_time_col="time",
                    lag_minutes=int(candidate_lag),
                    tolerance_minutes=tolerance_minutes,
                )
                snapped_groups.append(snapped[[col for col in snapped.columns if col != "sample_id"]])
        snapped_part = pd.concat([group.reset_index(drop=True) for group in snapped_groups], axis=1) if snapped_groups else pd.DataFrame(index=samples.index)
        feature_cols = list(snapped_part.columns)
        merged_parts.append(snapped_part)
        part_reports.append(
            {
                "path": str(path),
                "raw_columns": part.get("raw_columns", []),
                "stage_lags": [
                    {
                        "raw_column": raw_col,
                        "stage": stage_lookup.get(raw_col),
                        "lag_minutes": int(lag_lookup.get(raw_col, lag_minutes)),
                        "candidate_lag_minutes": candidate_lag_lookup.get(raw_col, [int(lag_lookup.get(raw_col, lag_minutes))]),
                    }
                    for raw_col in part.get("raw_columns", [])
                ],
                "feature_columns": int(len(feature_cols)),
                "mean_availability": float(snapped_part[feature_cols].notna().mean(axis=1).mean()) if feature_cols else None,
            }
        )

    result = pd.concat([part.reset_index(drop=True) for part in merged_parts], axis=1)
    report = {
        "feature_dir": str(feature_dir),
        "lag_mode": str(lag_mode),
        "lag_minutes": int(lag_minutes),
        "effective_lag_minutes": sorted(
            {
                int(v)
                for item in lag_policy["candidate_lag_minutes"].astype(str)
                for v in item.split(",")
                if v.strip()
            }
        ),
        "tolerance_minutes": int(tolerance_minutes),
        "parts": int(len(parts)),
        "feature_columns": int(result.shape[1] - 1),
        "lag_policy": lag_policy.to_dict(orient="records"),
        "stage_counts": lag_policy["stage"].value_counts().to_dict(),
        "part_reports": part_reports,
        "manifest_total_feature_columns": manifest.get("total_feature_columns"),
    }
    return result, report


def screen_features(
    train: pd.DataFrame,
    features: list[str],
    *,
    max_missing_rate: float,
    min_variance: float,
) -> tuple[list[str], dict[str, Any]]:
    numeric = train[features].apply(pd.to_numeric, errors="coerce")
    missing = numeric.isna().mean()
    kept = [col for col in features if float(missing[col]) <= max_missing_rate]
    variance = numeric[kept].var(skipna=True).fillna(0.0) if kept else pd.Series(dtype=float)
    kept2 = [col for col in kept if float(variance[col]) > min_variance]
    return kept2, {
        "input_feature_count": int(len(features)),
        "after_missing_count": int(len(kept)),
        "after_variance_count": int(len(kept2)),
        "max_missing_rate": float(max_missing_rate),
        "min_variance": float(min_variance),
    }


def select_reduced_features(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    max_features: int,
    candidate_pool_size: int,
    corr_threshold: float,
) -> tuple[list[str], dict[str, Any]]:
    y = pd.to_numeric(train[target], errors="coerce")
    usable = y.notna()
    x = train.loc[usable, features].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    yv = y.loc[usable].astype(float)
    y_centered = yv - float(yv.mean())
    x_centered = x - x.mean(axis=0)
    denom = np.sqrt((x_centered.pow(2).sum(axis=0) * float((y_centered**2).sum())).clip(lower=1.0e-30))
    scores = (x_centered.mul(y_centered, axis=0).sum(axis=0).abs() / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    candidates = scores.sort_values(ascending=False).head(candidate_pool_size).index.tolist()
    if not candidates:
        return [], {"selected_feature_count": 0, "top_ranked_features": []}
    cand_frame = x[candidates]
    corr = cand_frame.corr().abs()
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
        "max_features": int(max_features),
        "correlation_threshold": float(corr_threshold),
        "top_ranked_features": candidates[:30],
    }


def regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    y = y_true.to_numpy(dtype=float)
    p = np.asarray(pred, dtype=float)
    corr = spearmanr(y, p, nan_policy="omit").correlation if len(y) >= 3 else math.nan
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "spearman": float(corr) if corr is not None and not math.isnan(float(corr)) else math.nan,
        "mse": float(np.mean((p - y) ** 2)),
    }


def binary_metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    y = y_true.to_numpy(dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    out = {
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan
    return out


def best_f1_threshold(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    y = y_true.to_numpy(dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    precision, recall, thresholds = precision_recall_curve(y, p)
    if thresholds.size == 0:
        return {"threshold": 0.5, "valid_precision": 0.0, "valid_recall": 0.0, "valid_f1": 0.0}
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1.0e-12)
    idx = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[idx]),
        "valid_precision": float(precision[idx]),
        "valid_recall": float(recall[idx]),
        "valid_f1": float(f1[idx]),
    }


def threshold_metrics(y_true: pd.Series, prob: np.ndarray, threshold: float) -> dict[str, float]:
    y = y_true.to_numpy(dtype=int)
    pred = (np.asarray(prob, dtype=float) >= float(threshold)).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alarm_rate": float(pred.mean()),
        "false_alarm_rate": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "miss_rate": float(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
    }


def fit_regressor(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> tuple[np.ndarray, dict[str, float]]:
    del valid
    pipe_imputer = SimpleImputer(strategy="median")
    x_train = pipe_imputer.fit_transform(train[features])
    x_test = pipe_imputer.transform(test[features])
    y_train = train[target].to_numpy(dtype=float)
    sample_weight = train["sample_weight"].to_numpy(dtype=float) if "sample_weight" in train else None
    model = GradientBoostingRegressor(
        loss="huber",
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=10,
        alpha=0.9,
        random_state=20260415,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    pred = model.predict(x_test)
    if target in {"cd_mean", "p_pass_soft"}:
        pred = np.clip(pred, 0.0, 1.0)
    metrics = regression_metrics(test[target], pred)
    return pred, metrics


def fit_classifier(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_valid = imputer.transform(valid[features])
    x_test = imputer.transform(test[features])
    y_train = train[target].to_numpy(dtype=int)
    sample_weight = train["sample_weight"].to_numpy(dtype=float) if "sample_weight" in train else None
    model = GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=20260415,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    valid_prob = model.predict_proba(x_valid)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    threshold = best_f1_threshold(valid[target], valid_prob)
    metrics = binary_metrics(test[target], test_prob)
    metrics.update({f"valid_{k}": v for k, v in threshold.items() if k != "threshold"})
    metrics.update(threshold_metrics(test[target], test_prob, threshold["threshold"]))
    return valid_prob, test_prob, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Model T90 labels with halogenation and refining precomputed DCS features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-tag", type=str, default="exp014_two_stage_full_reduced")
    parser.add_argument("--hal-feature-dir", type=Path, default=DATA_DIR / "merge_data_volatiles_style_feature_parts")
    parser.add_argument("--refining-path", type=Path, default=DATA_DIR / "output.csv")
    parser.add_argument("--hal-lag-mode", choices=["global", "stage_map", "stage_candidates"], default="global")
    parser.add_argument("--hal-lag-minutes", type=int, default=180)
    parser.add_argument("--ref-lag-minutes", type=int, default=10)
    parser.add_argument("--hal-tolerance-minutes", type=int, default=10)
    parser.add_argument("--ref-tolerance-minutes", type=int, default=5)
    parser.add_argument("--ref-start-col-1based", type=int, default=9)
    parser.add_argument("--ref-end-col-1based", type=int, default=412)
    parser.add_argument("--include-ref-y-derived", action="store_true")
    parser.add_argument("--include-lims-context", action="store_true")
    parser.add_argument("--max-missing-rate", type=float, default=0.95)
    parser.add_argument("--min-variance", type=float, default=1.0e-12)
    parser.add_argument("--reduced-max-features", type=int, default=300)
    parser.add_argument("--reduced-candidate-pool", type=int, default=900)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    output_root = resolve_path(config_path, config["paths"]["output_root"])
    if output_root is None:
        raise ValueError("paths.output_root is required.")
    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    lims_path = resolve_path(config_path, config["paths"]["lims_path"])
    if lims_path is None:
        raise ValueError("paths.lims_path is required.")
    samples = add_noise_aware_labels(load_lims_samples(lims_path, config), config)

    refining, refining_report = load_refining_features(
        args.refining_path,
        time_col=str(config["downstream"].get("time_col", "time")),
        start_col_1based=args.ref_start_col_1based,
        end_col_1based=args.ref_end_col_1based,
        include_y_derived=bool(args.include_ref_y_derived),
    )
    manifest = json.loads((args.hal_feature_dir / "manifest.json").read_text(encoding="utf-8"))
    raw_cols = [raw_col for part in manifest["parts"] for raw_col in part.get("raw_columns", [])]
    hal_policy = halogen_lag_policy(raw_cols, mode=args.hal_lag_mode, global_lag_minutes=args.hal_lag_minutes)
    hal_policy_path = run_dir / "halogen_lag_policy.csv"
    hal_policy.to_csv(hal_policy_path, index=False, encoding="utf-8-sig")
    effective_hal_lags = sorted(
        {
            int(v)
            for item in hal_policy["candidate_lag_minutes"].astype(str)
            for v in item.split(",")
            if v.strip()
        }
    )

    first_part = pd.read_parquet(Path(manifest["parts"][0]["path"]), columns=["time"])
    hal_min = first_part["time"].min()
    hal_max = first_part["time"].max()
    ref_min = refining["time"].min()
    ref_max = refining["time"].max()
    boundary_mask = (
        (samples["sample_time"] - pd.Timedelta(minutes=max(effective_hal_lags)) >= hal_min)
        & (samples["sample_time"] - pd.Timedelta(minutes=min(effective_hal_lags)) <= hal_max)
        & (samples["sample_time"] - pd.Timedelta(minutes=args.ref_lag_minutes) >= ref_min)
        & (samples["sample_time"] - pd.Timedelta(minutes=args.ref_lag_minutes) <= ref_max)
    )
    samples = samples.loc[boundary_mask].sort_values("sample_time").reset_index(drop=True)

    split, split_audit = make_time_purged_split(samples, config)
    samples = samples.merge(split[["sample_id", "split"]], on="sample_id", how="left")
    leak_audit = leakage_audit(split)

    hal_features, hal_report = load_halogen_feature_parts(
        args.hal_feature_dir,
        samples,
        lag_mode=args.hal_lag_mode,
        lag_minutes=args.hal_lag_minutes,
        tolerance_minutes=args.hal_tolerance_minutes,
    )
    ref_snapshot = asof_snapshot(
        samples,
        refining,
        value_time_col="time",
        lag_minutes=args.ref_lag_minutes,
        tolerance_minutes=args.ref_tolerance_minutes,
    )
    ref_feature_cols = [col for col in ref_snapshot.columns if col.startswith("ref__")]
    ref_report = {
        **refining_report,
        "lag_minutes": int(args.ref_lag_minutes),
        "tolerance_minutes": int(args.ref_tolerance_minutes),
        "mean_snapshot_availability": float(ref_snapshot[ref_feature_cols].notna().mean(axis=1).mean()) if ref_feature_cols else None,
    }

    feature_table = samples.merge(hal_features, on="sample_id", how="left").merge(
        ref_snapshot[["sample_id", *ref_feature_cols]],
        on="sample_id",
        how="left",
    )
    feature_table_path = run_dir / "two_stage_feature_table.parquet"
    feature_table.to_parquet(feature_table_path, index=False, compression="zstd")

    features = [col for col in feature_table.columns if col.startswith(("hal__", "hal_lag", "ref__"))]
    if args.include_lims_context:
        features.extend(
            [
                col
                for col in feature_table.columns
                if col.startswith("lims_ctx__") and not col.endswith("__count")
            ]
        )
    train = feature_table[feature_table["split"] == "train"].copy()
    valid = feature_table[feature_table["split"] == "valid"].copy()
    test = feature_table[feature_table["split"] == "test"].copy()
    screened_features, screen_audit = screen_features(
        train,
        features,
        max_missing_rate=args.max_missing_rate,
        min_variance=args.min_variance,
    )

    rows: list[dict[str, Any]] = []
    scored = test[["sample_id", "sample_time", "t90", "cd_mean", "p_pass_soft", "is_out_spec_obs", "sample_weight"]].copy()

    targets = ["cd_mean", "p_pass_soft", "is_out_spec_obs"]
    for mode in ["full", "reduced"]:
        for target in targets:
            problem = "binary" if target == "is_out_spec_obs" else "regression"
            if mode == "full":
                selected = screened_features
                selection_audit = {"mode": "full", **screen_audit}
            else:
                selected, reduced_audit = select_reduced_features(
                    train,
                    screened_features,
                    target,
                    max_features=args.reduced_max_features,
                    candidate_pool_size=args.reduced_candidate_pool,
                    corr_threshold=args.corr_threshold,
                )
                selection_audit = {"mode": "reduced", **screen_audit, **reduced_audit}
            if not selected:
                continue
            if problem == "binary":
                valid_prob, test_prob, metrics = fit_classifier(train, valid, test, selected, target)
                scored[f"{mode}_{target}_prob"] = test_prob
            else:
                pred, metrics = fit_regressor(train, valid, test, selected, target)
                scored[f"{mode}_{target}_pred"] = pred
                if target == "p_pass_soft":
                    prob_fail = 1.0 - pred
                    metrics["out_spec_brier_from_p_pass"] = float(brier_score_loss(test["is_out_spec_obs"].astype(int), prob_fail))
            rows.append(
                {
                    "mode": mode,
                    "target": target,
                    "problem_type": problem,
                    "samples_train": int(len(train)),
                    "samples_valid": int(len(valid)),
                    "samples_test": int(len(test)),
                    "feature_count": int(len(selected)),
                    "selection_audit": json.dumps(json_ready(selection_audit), ensure_ascii=False),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    results_path = run_dir / "two_stage_model_results.csv"
    scored_path = run_dir / "two_stage_scored_test_rows.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "feature_table_path": str(feature_table_path),
        "results_path": str(results_path),
        "scored_test_rows_path": str(scored_path),
        "sample_count_after_two_stage_boundary": int(len(samples)),
        "split_counts": samples["split"].value_counts(dropna=False).to_dict(),
        "raw_feature_count": int(len(features)),
        "screened_feature_count": int(len(screened_features)),
        "include_lims_context_features": bool(args.include_lims_context),
        "halogenation_lag_mode": str(args.hal_lag_mode),
        "halogenation_lag_minutes": int(args.hal_lag_minutes),
        "halogenation_effective_lag_minutes": effective_hal_lags,
        "halogenation_lag_policy_path": str(hal_policy_path),
        "refining_lag_minutes": int(args.ref_lag_minutes),
        "halogenation_report": hal_report,
        "refining_report": ref_report,
        "split_audit": split_audit,
        "leakage_audit": leak_audit,
        "screen_audit": screen_audit,
        "results": json_ready(results.to_dict(orient="records")),
    }
    (run_dir / "two_stage_model_summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
