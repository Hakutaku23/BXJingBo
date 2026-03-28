from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
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
DEFAULT_DUAL_HEAD_SUMMARY = V3_ROOT / "dev" / "artifacts" / "phase1_dual_head_current_future_summary.json"


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


def build_future_window_label(frame: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    work = frame.sort_values("sample_time").reset_index(drop=True).copy()
    times = pd.to_datetime(work["sample_time"], errors="coerce")
    labels = work["is_out_of_spec"].astype(int).to_numpy()
    target = np.full(len(work), np.nan, dtype=float)

    for idx, current_time in enumerate(times):
        if pd.isna(current_time):
            continue
        future_mask = (times > current_time) & (times <= current_time + pd.Timedelta(minutes=horizon_minutes))
        future_indices = np.flatnonzero(future_mask.to_numpy(dtype=bool))
        if len(future_indices) == 0:
            continue
        target[idx] = float(labels[future_indices].max())

    return pd.Series(target, index=work.index, name=f"future_out_of_spec_{horizon_minutes}m")


def build_selected_features(frame: pd.DataFrame, sensors: list[str], stats: list[str]) -> list[str]:
    sensor_set = set(sensors)
    stat_set = set(stats)
    selected = []
    for column in frame.columns:
        if "__" not in column:
            continue
        sensor, stat = column.split("__", 1)
        if sensor in sensor_set and stat in stat_set:
            selected.append(column)
    return selected


def make_model(model_name: str) -> Pipeline:
    if model_name == "logistic_balanced":
        return Pipeline(
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
    if model_name == "random_forest_balanced":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if model_name == "extra_trees_balanced":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        max_depth=6,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if model_name == "gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    GradientBoostingClassifier(
                        max_depth=4,
                        learning_rate=0.05,
                        n_estimators=250,
                        min_samples_leaf=20,
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def collect_oof_probabilities(feature_table: pd.DataFrame, selected_features: list[str], target: pd.Series, model_name: str, n_splits: int = 5) -> pd.DataFrame:
    work = feature_table.sort_values("sample_time").reset_index(drop=True).copy()
    target_aligned = target.loc[work.index]
    usable_mask = target_aligned.notna()
    work = work.loc[usable_mask].reset_index(drop=True)
    y = target_aligned.loc[usable_mask].astype(int).to_numpy()

    X = work[selected_features]
    probabilities = np.full(len(work), np.nan, dtype=float)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue

        fold_columns = [column for column in selected_features if X_train[column].notna().any()]
        if not fold_columns:
            continue

        model = make_model(model_name)
        model.fit(X_train[fold_columns], y_train)
        probabilities[test_idx] = model.predict_proba(X_test[fold_columns])[:, 1]

    scored_mask = ~np.isnan(probabilities)
    if scored_mask.sum() == 0:
        return pd.DataFrame()

    result = work.loc[scored_mask, ["sample_time", "t90", "is_out_of_spec"]].copy()
    result["target"] = y[scored_mask]
    result["probability"] = probabilities[scored_mask]
    return result.reset_index(drop=True)


def scan_thresholds(oof_frame: pd.DataFrame) -> list[dict[str, float]]:
    y_true = oof_frame["target"].astype(int).to_numpy()
    prob = oof_frame["probability"].to_numpy(dtype=float)
    rows = []
    for threshold in np.arange(0.05, 1.00, 0.05):
        pred = prob >= threshold
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1": float(f1),
                "predicted_positive_ratio": float(pred.mean()),
            }
        )
    return rows


def choose_threshold(scan_rows: list[dict[str, float]], min_recall: float = 0.80) -> dict[str, float]:
    candidates = [row for row in scan_rows if row["recall"] >= min_recall]
    if not candidates:
        return max(scan_rows, key=lambda row: (row["f1"], row["specificity"]))
    return max(candidates, key=lambda row: (row["specificity"], row["precision"], row["f1"]))


def benchmark_head(
    head_name: str,
    feature_table: pd.DataFrame,
    selected_features: list[str],
    target: pd.Series,
    model_names: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    oof_frames: dict[str, pd.DataFrame] = {}

    for model_name in model_names:
        oof_frame = collect_oof_probabilities(feature_table, selected_features, target, model_name=model_name)
        if oof_frame.empty or len(np.unique(oof_frame["target"])) < 2:
            rows.append({"head_name": head_name, "model_name": model_name, "status": "not_enough_scored_samples"})
            continue

        oof_frames[model_name] = oof_frame
        scan_rows = scan_thresholds(oof_frame)
        threshold_row = choose_threshold(scan_rows)
        rows.append(
            {
                "head_name": head_name,
                "model_name": model_name,
                "status": "ok",
                "samples": int(len(oof_frame)),
                "positive_ratio": float(oof_frame["target"].mean()),
                "roc_auc": float(roc_auc_score(oof_frame["target"], oof_frame["probability"])),
                "average_precision": float(average_precision_score(oof_frame["target"], oof_frame["probability"])),
                "threshold": float(threshold_row["threshold"]),
                "precision": float(threshold_row["precision"]),
                "recall": float(threshold_row["recall"]),
                "specificity": float(threshold_row["specificity"]),
                "f1": float(threshold_row["f1"]),
                "predicted_positive_ratio": float(threshold_row["predicted_positive_ratio"]),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["roc_auc", "average_precision", "f1", "specificity", "model_name"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
    return result, oof_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark different model families independently for current alarm head and future warning head.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dual-head-summary", type=Path, default=DEFAULT_DUAL_HEAD_SUMMARY)
    parser.add_argument("--models", type=str, default="logistic_balanced,random_forest_balanced,extra_trees_balanced,gradient_boosting")
    parser.add_argument("--output-prefix", type=str, default="phase1_dual_head_model_family_benchmark")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dual_summary = json.loads(args.dual_head_summary.read_text(encoding="utf-8"))
    alarm_row = dual_summary["alarm_best_row"]
    warning_row = dual_summary["warning_best_row"]
    future_horizon = int(dual_summary["future_horizon_minutes"])
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]

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

    alarm_table = build_dcs_feature_table(
        labeled,
        dcs,
        window_minutes=int(alarm_row["window_minutes"]),
        min_points_per_window=int(config["window_search"]["min_points_per_window"]),
    )
    warning_table = build_dcs_feature_table(
        labeled,
        dcs,
        window_minutes=int(warning_row["window_minutes"]),
        min_points_per_window=int(config["window_search"]["min_points_per_window"]),
    )

    alarm_features = build_selected_features(alarm_table, list(alarm_row["sensor_names"]), str(alarm_row["stats"]).split(","))
    warning_features = build_selected_features(warning_table, list(warning_row["sensor_names"]), str(warning_row["stats"]).split(","))
    alarm_target = alarm_table["is_out_of_spec"].astype(float)
    warning_target = build_future_window_label(warning_table, horizon_minutes=future_horizon)

    alarm_result, alarm_oof = benchmark_head("current_alarm", alarm_table, alarm_features, alarm_target, model_names)
    warning_result, warning_oof = benchmark_head("future_warning", warning_table, warning_features, warning_target, model_names)

    summary = {
        "future_horizon_minutes": future_horizon,
        "alarm_feature_setup": {
            "window_minutes": int(alarm_row["window_minutes"]),
            "sensor_names": list(alarm_row["sensor_names"]),
            "stats": str(alarm_row["stats"]).split(","),
        },
        "warning_feature_setup": {
            "window_minutes": int(warning_row["window_minutes"]),
            "sensor_names": list(warning_row["sensor_names"]),
            "stats": str(warning_row["stats"]).split(","),
        },
        "alarm_rows": alarm_result.to_dict(orient="records"),
        "warning_rows": warning_result.to_dict(orient="records"),
        "recommended_alarm_model": alarm_result.iloc[0]["model_name"] if not alarm_result.empty else None,
        "recommended_warning_model": warning_result.iloc[0]["model_name"] if not warning_result.empty else None,
    }

    alarm_result.to_csv(artifact_dir / f"{args.output_prefix}_alarm_summary.csv", index=False, encoding="utf-8-sig")
    warning_result.to_csv(artifact_dir / f"{args.output_prefix}_warning_summary.csv", index=False, encoding="utf-8-sig")
    for model_name, oof_frame in alarm_oof.items():
        oof_frame.to_csv(artifact_dir / f"{args.output_prefix}_alarm_{model_name}_oof.csv", index=False, encoding="utf-8-sig")
    for model_name, oof_frame in warning_oof.items():
        oof_frame.to_csv(artifact_dir / f"{args.output_prefix}_warning_{model_name}_oof.csv", index=False, encoding="utf-8-sig")
    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
