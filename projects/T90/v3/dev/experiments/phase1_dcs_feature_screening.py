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

from core import TargetSpec, add_out_of_spec_labels, build_dcs_feature_table, load_dcs_frame, load_lims_samples


DEFAULT_CONFIG_PATH = V3_ROOT / "config" / "phase1_warning.yaml"
BLOCKED_COLUMNS = {
    "sample_time",
    "sample_name",
    "t90",
    "is_in_spec",
    "is_out_of_spec",
    "is_above_spec",
    "is_below_spec",
    "window_minutes",
    "rows_in_window",
    "volatile_lab",
    "volatile_online",
    "bromine",
    "calcium_stearate",
    "calcium",
    "stabilizer",
    "mooney",
    "antioxidant",
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


def get_dcs_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in BLOCKED_COLUMNS and "__" in column]


def split_feature_name(feature_name: str) -> tuple[str, str]:
    if "__" not in feature_name:
        return feature_name, "raw"
    sensor, stat = feature_name.split("__", 1)
    return sensor, stat


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
    auc_sym = float(max(auc, 1.0 - auc))
    corr = float(abs(np.corrcoef(x_values, y_values)[0, 1])) if np.nanstd(x_values) > 0 else 0.0
    if np.isnan(corr):
        corr = 0.0

    # Normalize both components into a 0~1 utility score and weight them.
    auc_component = max((auc_sym - 0.5) * 2.0, 0.0)
    corr_component = max(corr, 0.0)
    screen_score = float(valid_ratio * (0.7 * auc_component + 0.3 * corr_component))

    return {
        "valid_samples": int(len(valid)),
        "valid_ratio": valid_ratio,
        "auc_sym": auc_sym,
        "corr_abs": corr,
        "screen_score": screen_score,
    }


def rank_window_features(feature_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = feature_table["is_out_of_spec"].astype(int)
    feature_rows: list[dict[str, Any]] = []
    for feature_name in get_dcs_feature_columns(feature_table):
        sensor, stat = split_feature_name(feature_name)
        metrics = score_single_feature(feature_table[feature_name], y)
        feature_rows.append(
            {
                "feature_name": feature_name,
                "sensor_name": sensor,
                "stat_name": stat,
                **metrics,
            }
        )

    feature_df = pd.DataFrame(feature_rows).sort_values(
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
            mean_auc_sym=("auc_sym", "mean"),
            best_valid_ratio=("valid_ratio", "max"),
        )
        .sort_values(["best_score", "mean_score", "best_auc_sym", "sensor_name"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )

    stat_df = (
        feature_df.groupby("stat_name", as_index=False)
        .agg(
            feature_count=("feature_name", "count"),
            mean_score=("screen_score", "mean"),
            median_score=("screen_score", "median"),
            best_score=("screen_score", "max"),
            mean_auc_sym=("auc_sym", "mean"),
            mean_valid_ratio=("valid_ratio", "mean"),
        )
        .sort_values(["mean_score", "best_score", "stat_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    return feature_df, sensor_df, stat_df


def evaluate_sensor_subset(feature_table: pd.DataFrame, sensor_names: list[str], n_splits: int = 5) -> dict[str, Any]:
    subset_features = [
        column
        for column in get_dcs_feature_columns(feature_table)
        if split_feature_name(column)[0] in set(sensor_names)
    ]
    if not subset_features:
        return {"status": "no_features"}

    usable = feature_table.sort_values("sample_time").reset_index(drop=True)
    y = usable["is_out_of_spec"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return {"status": "single_class", "samples": int(len(usable))}

    X = usable[subset_features]
    splitter = TimeSeriesSplit(n_splits=n_splits)
    probabilities = np.full(len(usable), np.nan, dtype=float)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue

        fold_columns = [column for column in subset_features if X_train[column].notna().any()]
        if not fold_columns:
            continue

        model.fit(X_train[fold_columns], y_train)
        probabilities[test_idx] = model.predict_proba(X_test[fold_columns])[:, 1]

    scored_mask = ~np.isnan(probabilities)
    if scored_mask.sum() == 0:
        return {"status": "no_scored_samples", "samples": int(len(usable))}

    y_scored = y[scored_mask]
    prob_scored = probabilities[scored_mask]
    thresholds = np.arange(0.05, 1.00, 0.05)
    best = None
    for threshold in thresholds:
        predicted = prob_scored >= threshold
        tp = int(np.sum((predicted == 1) & (y_scored == 1)))
        fp = int(np.sum((predicted == 1) & (y_scored == 0)))
        fn = int(np.sum((predicted == 0) & (y_scored == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        row = {
            "threshold": float(round(threshold, 2)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if best is None or row["f1"] > best["f1"]:
            best = row

    return {
        "status": "ok",
        "samples": int(len(usable)),
        "scored_samples": int(scored_mask.sum()),
        "sensor_count": int(len(sensor_names)),
        "feature_count": int(len(subset_features)),
        "roc_auc": float(roc_auc_score(y_scored, prob_scored)),
        "average_precision": float(average_precision_score(y_scored, prob_scored)),
        "best_threshold": best["threshold"] if best else None,
        "best_precision": best["precision"] if best else None,
        "best_recall": best["recall"] if best else None,
        "best_f1": best["f1"] if best else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V3-native DCS feature screening from raw all-point data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--windows", type=str, default="")
    parser.add_argument("--topk", type=str, default="5,10,15,20")
    parser.add_argument("--output-prefix", type=str, default="phase1_dcs_feature_screening")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    candidate_windows = (
        [int(value) for value in args.windows.split(",") if value.strip()]
        if args.windows
        else [int(value) for value in config["window_search"]["candidate_minutes"]]
    )
    topk_values = [int(value) for value in args.topk.split(",") if value.strip()]
    min_points_per_window = int(config["window_search"]["min_points_per_window"])

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
    stat_rows_all: list[pd.DataFrame] = []
    subset_rows: list[dict[str, Any]] = []
    window_summary_rows: list[dict[str, Any]] = []

    for window_minutes in candidate_windows:
        feature_table = build_dcs_feature_table(
            labeled,
            dcs,
            window_minutes=window_minutes,
            min_points_per_window=min_points_per_window,
        )
        feature_df, sensor_df, stat_df = rank_window_features(feature_table)
        feature_df.insert(0, "window_minutes", int(window_minutes))
        sensor_df.insert(0, "window_minutes", int(window_minutes))
        stat_df.insert(0, "window_minutes", int(window_minutes))
        feature_rows_all.append(feature_df)
        sensor_rows_all.append(sensor_df)
        stat_rows_all.append(stat_df)

        top_sensor_names = sensor_df["sensor_name"].tolist()
        for topk in topk_values:
            metrics = evaluate_sensor_subset(feature_table, top_sensor_names[:topk])
            metrics.update(
                {
                    "window_minutes": int(window_minutes),
                    "topk_sensors": int(topk),
                    "sensor_names": top_sensor_names[:topk],
                }
            )
            subset_rows.append(metrics)

        best_subset = None
        if subset_rows:
            current_window_rows = [row for row in subset_rows if row["window_minutes"] == int(window_minutes) and row["status"] == "ok"]
            if current_window_rows:
                best_subset = max(
                    current_window_rows,
                    key=lambda row: (row["roc_auc"], row["average_precision"], row["best_f1"]),
                )

        top_sensor_preview = sensor_df.head(10)["sensor_name"].tolist()
        top_stat_preview = stat_df.head(8)["stat_name"].tolist()
        window_summary_rows.append(
            {
                "window_minutes": int(window_minutes),
                "samples": int(len(feature_table)),
                "positive_ratio": float(feature_table["is_out_of_spec"].mean()) if len(feature_table) else 0.0,
                "top_sensor_1": top_sensor_preview[0] if len(top_sensor_preview) > 0 else None,
                "top_sensor_2": top_sensor_preview[1] if len(top_sensor_preview) > 1 else None,
                "top_sensor_3": top_sensor_preview[2] if len(top_sensor_preview) > 2 else None,
                "top_stat_1": top_stat_preview[0] if len(top_stat_preview) > 0 else None,
                "top_stat_2": top_stat_preview[1] if len(top_stat_preview) > 1 else None,
                "top_stat_3": top_stat_preview[2] if len(top_stat_preview) > 2 else None,
                "best_subset_topk": int(best_subset["topk_sensors"]) if best_subset else None,
                "best_subset_roc_auc": float(best_subset["roc_auc"]) if best_subset else None,
                "best_subset_average_precision": float(best_subset["average_precision"]) if best_subset else None,
                "best_subset_f1": float(best_subset["best_f1"]) if best_subset else None,
            }
        )

    feature_rankings = pd.concat(feature_rows_all, ignore_index=True)
    sensor_rankings = pd.concat(sensor_rows_all, ignore_index=True)
    stat_rankings = pd.concat(stat_rows_all, ignore_index=True)
    subset_benchmarks = pd.DataFrame(subset_rows)
    window_summary = pd.DataFrame(window_summary_rows).sort_values(
        ["best_subset_roc_auc", "best_subset_average_precision", "best_subset_f1", "window_minutes"],
        ascending=[False, False, False, True],
    )

    summary = {
        "phase": config.get("phase", "phase1_warning"),
        "objective": config["objective"]["name"],
        "target_spec": {
            "center": target_spec.center,
            "tolerance": target_spec.tolerance,
            "low": target_spec.low,
            "high": target_spec.high,
        },
        "window_summary_rows": window_summary.to_dict(orient="records"),
        "recommended_next_step": {
            "note": (
                "Use the best-ranked windows and top screened sensors as the next V3 "
                "warning-modeling slice instead of carrying over the legacy point list."
            ),
            "top_windows_by_subset_auc": window_summary["window_minutes"].tolist(),
        },
    }

    feature_rankings.to_csv(
        artifact_dir / f"{args.output_prefix}_feature_rankings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensor_rankings.to_csv(
        artifact_dir / f"{args.output_prefix}_sensor_rankings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stat_rankings.to_csv(
        artifact_dir / f"{args.output_prefix}_stat_rankings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subset_benchmarks.to_csv(
        artifact_dir / f"{args.output_prefix}_subset_benchmarks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    window_summary.to_csv(
        artifact_dir / f"{args.output_prefix}_window_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
