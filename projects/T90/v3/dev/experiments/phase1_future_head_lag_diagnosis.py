from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(__file__).resolve().parent
V3_ROOT = EXPERIMENT_DIR.parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import TargetSpec, add_out_of_spec_labels, load_dcs_frame, load_lims_samples, summarize_numeric_window


DEFAULT_CONFIG_PATH = V3_ROOT / "config" / "phase1_warning.yaml"
BLOCKED_COLUMNS = {
    "decision_time",
    "feature_end_time",
    "sample_name",
    "t90",
    "is_in_spec",
    "is_out_of_spec",
    "is_above_spec",
    "is_below_spec",
    "window_minutes",
    "lag_minutes",
    "rows_in_window",
}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def resolve_path(config_path: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    return (config_path.parent / relative_path).resolve()


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def score_single_feature(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    valid = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if valid.empty:
        return {
            "valid_samples": 0,
            "valid_ratio": 0.0,
            "auc_sym": None,
            "corr_abs": None,
            "screen_score": 0.0,
        }

    x_values = valid["x"].to_numpy(dtype=float)
    y_values = valid["y"].to_numpy(dtype=int)
    valid_ratio = float(len(valid) / len(x)) if len(x) else 0.0
    if len(np.unique(x_values)) < 2 or len(np.unique(y_values)) < 2:
        return {
            "valid_samples": int(len(valid)),
            "valid_ratio": valid_ratio,
            "auc_sym": None,
            "corr_abs": 0.0,
            "screen_score": 0.0,
        }

    auc = roc_auc_score(y_values, x_values)
    auc_sym = float(max(float(auc), float(1.0 - auc)))
    corr = float(abs(np.corrcoef(x_values, y_values)[0, 1])) if np.nanstd(x_values) > 0 else 0.0
    if np.isnan(corr):
        corr = 0.0
    auc_component = max((auc_sym - 0.5) * 2.0, 0.0)
    screen_score = float(valid_ratio * (0.7 * auc_component + 0.3 * corr))
    return {
        "valid_samples": int(len(valid)),
        "valid_ratio": valid_ratio,
        "auc_sym": auc_sym,
        "corr_abs": corr,
        "screen_score": screen_score,
    }


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in BLOCKED_COLUMNS and "__" in column]


def split_feature_name(name: str) -> tuple[str, str]:
    sensor, stat = name.split("__", 1)
    return sensor, stat


def build_lagged_feature_table(
    labeled_samples: pd.DataFrame,
    dcs: pd.DataFrame,
    lag_minutes: int,
    window_minutes: int,
    min_points_per_window: int,
) -> pd.DataFrame:
    sample_frame = labeled_samples.copy()
    sample_frame["decision_time"] = pd.to_datetime(sample_frame["sample_time"], errors="coerce")
    sample_frame = sample_frame.dropna(subset=["decision_time"]).sort_values("decision_time").reset_index(drop=True)

    dcs_frame = dcs.copy()
    dcs_frame["time"] = pd.to_datetime(dcs_frame["time"], errors="coerce")
    dcs_frame = dcs_frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")

    rows: list[dict[str, Any]] = []
    for record in sample_frame.itertuples(index=False):
        decision_time = pd.Timestamp(record.decision_time)
        feature_end_time = decision_time - pd.Timedelta(minutes=lag_minutes)
        left_boundary = (feature_end_time - pd.Timedelta(minutes=window_minutes)).to_datetime64()
        right_boundary = feature_end_time.to_datetime64()
        left = np.searchsorted(dcs_times, left_boundary, side="left")
        right = np.searchsorted(dcs_times, right_boundary, side="right")
        window = dcs_frame.iloc[left:right]
        if len(window) < min_points_per_window:
            continue

        row = {
            "decision_time": decision_time,
            "feature_end_time": feature_end_time,
            "sample_name": getattr(record, "sample_name", np.nan),
            "t90": getattr(record, "t90", np.nan),
            "is_in_spec": getattr(record, "is_in_spec", False),
            "is_out_of_spec": getattr(record, "is_out_of_spec", False),
            "is_above_spec": getattr(record, "is_above_spec", False),
            "is_below_spec": getattr(record, "is_below_spec", False),
            "lag_minutes": int(lag_minutes),
            "window_minutes": int(window_minutes),
            "rows_in_window": int(len(window)),
        }
        row.update(summarize_numeric_window(window, time_column="time"))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)


def build_future_target(decision_times: pd.Series, full_labeled_samples: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    full = full_labeled_samples.copy()
    full["sample_time"] = pd.to_datetime(full["sample_time"], errors="coerce")
    full = full.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)
    full_times = full["sample_time"].to_numpy(dtype="datetime64[ns]")
    full_labels = full["is_out_of_spec"].astype(int).to_numpy()

    target = np.full(len(decision_times), np.nan, dtype=float)
    for idx, current_time in enumerate(pd.to_datetime(decision_times, errors="coerce")):
        if pd.isna(current_time):
            continue
        future_mask = (full_times > current_time.to_datetime64()) & (
            full_times <= (current_time + pd.Timedelta(minutes=horizon_minutes)).to_datetime64()
        )
        future_indices = np.flatnonzero(future_mask)
        if len(future_indices) == 0:
            continue
        target[idx] = float(full_labels[future_indices].max())
    return pd.Series(target, index=decision_times.index, name=f"future_out_of_spec_{horizon_minutes}m")


def rank_sensors(feature_table: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for feature_name in get_feature_columns(feature_table):
        sensor_name, stat_name = split_feature_name(feature_name)
        metrics = score_single_feature(feature_table[feature_name], target)
        rows.append(
            {
                "feature_name": feature_name,
                "sensor_name": sensor_name,
                "stat_name": stat_name,
                **metrics,
            }
        )
    feature_df = pd.DataFrame(rows).sort_values(
        ["screen_score", "auc_sym", "corr_abs", "feature_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    sensor_df = (
        feature_df.groupby("sensor_name", as_index=False)
        .agg(
            feature_count=("feature_name", "count"),
            best_feature=("feature_name", "first"),
            best_score=("screen_score", "max"),
            mean_score=("screen_score", "mean"),
            best_auc_sym=("auc_sym", "max"),
        )
        .sort_values(["best_score", "mean_score", "best_auc_sym", "sensor_name"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )
    return feature_df, sensor_df


def build_selected_features(feature_table: pd.DataFrame, sensors: list[str], stats: list[str]) -> list[str]:
    sensor_set = set(sensors)
    stat_set = set(stats)
    selected = []
    for column in get_feature_columns(feature_table):
        sensor_name, stat_name = split_feature_name(column)
        if sensor_name in sensor_set and stat_name in stat_set:
            selected.append(column)
    return selected


def evaluate_logistic_subset(feature_table: pd.DataFrame, target: pd.Series, sensors: list[str], stats: list[str]) -> dict[str, Any]:
    selected_features = build_selected_features(feature_table, sensors, stats)
    if not selected_features:
        return {"status": "no_features"}

    work = feature_table.sort_values("decision_time").reset_index(drop=True).copy()
    target_aligned = target.loc[work.index]
    usable_mask = target_aligned.notna()
    work = work.loc[usable_mask].reset_index(drop=True)
    y = target_aligned.loc[usable_mask].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return {"status": "single_class", "samples": int(len(work))}

    X = work[selected_features]
    splitter = TimeSeriesSplit(n_splits=5)
    probabilities = np.full(len(work), np.nan, dtype=float)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")),
        ]
    )
    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        fold_columns = [column for column in selected_features if X_train[column].notna().any()]
        if not fold_columns:
            continue
        model.fit(X_train[fold_columns], y_train)
        probabilities[test_idx] = model.predict_proba(X_test[fold_columns])[:, 1]

    scored_mask = ~np.isnan(probabilities)
    if scored_mask.sum() == 0:
        return {"status": "no_scored_samples", "samples": int(len(work))}

    y_scored = y[scored_mask]
    prob_scored = probabilities[scored_mask]
    thresholds = np.arange(0.05, 1.00, 0.05)
    best = None
    for threshold in thresholds:
        pred = prob_scored >= threshold
        tp = int(np.sum((pred == 1) & (y_scored == 1)))
        fp = int(np.sum((pred == 1) & (y_scored == 0)))
        fn = int(np.sum((pred == 0) & (y_scored == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        row = {"threshold": float(round(threshold, 2)), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
        if best is None or row["f1"] > best["f1"]:
            best = row

    return {
        "status": "ok",
        "samples": int(len(work)),
        "scored_samples": int(scored_mask.sum()),
        "sensor_count": int(len(sensors)),
        "feature_count": int(len(selected_features)),
        "roc_auc": float(roc_auc_score(y_scored, prob_scored)),
        "average_precision": float(average_precision_score(y_scored, prob_scored)),
        "best_threshold": best["threshold"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_f1": best["f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 future-head lag and window diagnosis under fixed causal window-statistics features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--future-horizon-minutes", type=int, default=240)
    parser.add_argument("--lag-hours", type=str, default="0,1,2,4,6,8,12")
    parser.add_argument("--window-minutes", type=str, default="10,20,60,120,240,360,480")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--stats", type=str, default="mean,min,last,max")
    parser.add_argument("--output-prefix", type=str, default="phase1_future_head_lag_diagnosis")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    lag_hours = parse_int_list(args.lag_hours)
    window_minutes_list = parse_int_list(args.window_minutes)
    stats = [item.strip() for item in args.stats.split(",") if item.strip()]

    source_config = config["data_sources"]
    target_spec = TargetSpec(
        center=float(config["target_spec"]["center"]),
        tolerance=float(config["target_spec"]["tolerance"]),
    )
    lims_samples, _ = load_lims_samples(resolve_path(config_path, source_config["lims_path"]))
    labeled = add_out_of_spec_labels(lims_samples, target_spec).dropna(subset=["t90"]).copy()
    dcs = load_dcs_frame(
        resolve_path(config_path, source_config["dcs_main_path"]),
        resolve_path(config_path, source_config.get("dcs_supplemental_path")),
    )

    feature_rows_all: list[pd.DataFrame] = []
    sensor_rows_all: list[pd.DataFrame] = []
    subset_rows: list[dict[str, Any]] = []

    for lag_hour in lag_hours:
        lag_minutes = int(lag_hour * 60)
        for window_minutes in window_minutes_list:
            feature_table = build_lagged_feature_table(
                labeled_samples=labeled,
                dcs=dcs,
                lag_minutes=lag_minutes,
                window_minutes=window_minutes,
                min_points_per_window=int(config["window_search"]["min_points_per_window"]),
            )
            if feature_table.empty:
                subset_rows.append(
                    {
                        "lag_hours": int(lag_hour),
                        "window_minutes": int(window_minutes),
                        "status": "no_feature_rows",
                    }
                )
                continue

            future_target = build_future_target(
                decision_times=feature_table["decision_time"],
                full_labeled_samples=labeled,
                horizon_minutes=int(args.future_horizon_minutes),
            )
            usable_mask = future_target.notna()
            if int(usable_mask.sum()) == 0 or len(np.unique(future_target.loc[usable_mask].astype(int))) < 2:
                subset_rows.append(
                    {
                        "lag_hours": int(lag_hour),
                        "window_minutes": int(window_minutes),
                        "status": "not_enough_future_labels",
                    }
                )
                continue

            feature_df, sensor_df = rank_sensors(feature_table.loc[usable_mask].reset_index(drop=True), future_target.loc[usable_mask].reset_index(drop=True))
            feature_df.insert(0, "lag_hours", int(lag_hour))
            feature_df.insert(1, "window_minutes", int(window_minutes))
            sensor_df.insert(0, "lag_hours", int(lag_hour))
            sensor_df.insert(1, "window_minutes", int(window_minutes))
            feature_rows_all.append(feature_df)
            sensor_rows_all.append(sensor_df)

            top_sensors = sensor_df["sensor_name"].head(int(args.topk)).tolist()
            metrics = evaluate_logistic_subset(feature_table, future_target, top_sensors, stats)
            metrics.update(
                {
                    "lag_hours": int(lag_hour),
                    "window_minutes": int(window_minutes),
                    "topk_sensors": int(args.topk),
                    "stats": ",".join(stats),
                    "sensor_names": top_sensors,
                    "future_horizon_minutes": int(args.future_horizon_minutes),
                }
            )
            subset_rows.append(metrics)

    subset_df = pd.DataFrame(subset_rows)
    ok_df = subset_df[subset_df["status"] == "ok"].copy()
    ok_df = ok_df.sort_values(
        ["roc_auc", "average_precision", "best_f1", "lag_hours", "window_minutes"],
        ascending=[False, False, False, True, True],
    )

    lag_summary = (
        ok_df.groupby("lag_hours", as_index=False)
        .agg(
            best_roc_auc=("roc_auc", "max"),
            best_average_precision=("average_precision", "max"),
            best_f1=("best_f1", "max"),
            best_window_minutes=("window_minutes", "first"),
        )
        .sort_values(["best_roc_auc", "best_average_precision", "best_f1", "lag_hours"], ascending=[False, False, False, True])
    )

    window_summary = (
        ok_df.groupby("window_minutes", as_index=False)
        .agg(
            best_roc_auc=("roc_auc", "max"),
            best_average_precision=("average_precision", "max"),
            best_f1=("best_f1", "max"),
            best_lag_hours=("lag_hours", "first"),
        )
        .sort_values(["best_roc_auc", "best_average_precision", "best_f1", "window_minutes"], ascending=[False, False, False, True])
    )

    best_row = ok_df.iloc[0].to_dict() if not ok_df.empty else None
    summary = {
        "future_horizon_minutes": int(args.future_horizon_minutes),
        "topk_sensors": int(args.topk),
        "stats": stats,
        "best_row": best_row,
        "lag_summary_rows": lag_summary.to_dict(orient="records"),
        "window_summary_rows": window_summary.to_dict(orient="records"),
        "notes": {
            "goal": "Diagnose whether the future-head benefits from explicit lagged causal windows before entering distilled-feature design.",
            "scope": "This stage uses simple causal window-statistics plus sensor screening to isolate lag/window effects before EWMA distillation.",
        },
    }

    if feature_rows_all:
        pd.concat(feature_rows_all, ignore_index=True).to_csv(
            artifact_dir / f"{args.output_prefix}_feature_rankings.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if sensor_rows_all:
        pd.concat(sensor_rows_all, ignore_index=True).to_csv(
            artifact_dir / f"{args.output_prefix}_sensor_rankings.csv",
            index=False,
            encoding="utf-8-sig",
        )
    subset_df.to_csv(artifact_dir / f"{args.output_prefix}_subset_benchmarks.csv", index=False, encoding="utf-8-sig")
    lag_summary.to_csv(artifact_dir / f"{args.output_prefix}_lag_summary.csv", index=False, encoding="utf-8-sig")
    window_summary.to_csv(artifact_dir / f"{args.output_prefix}_window_summary.csv", index=False, encoding="utf-8-sig")
    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
