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


BASELINE_DIR = Path(__file__).resolve().parent
V3_ROOT = BASELINE_DIR.parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import TargetSpec, add_out_of_spec_labels, build_dcs_feature_table, load_dcs_frame, load_lims_samples


DEFAULT_CONFIG_PATH = V3_ROOT / "config" / "phase1_warning.yaml"
SCREENING_PREFIX = "phase1_dcs_feature_screening"


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


def split_feature_name(feature_name: str) -> tuple[str, str]:
    if "__" not in feature_name:
        return feature_name, "raw"
    sensor, stat = feature_name.split("__", 1)
    return sensor, stat


def build_selected_features(feature_table: pd.DataFrame, sensors: list[str], stats: list[str]) -> list[str]:
    sensor_set = set(sensors)
    stat_set = set(stats)
    return [
        column
        for column in feature_table.columns
        if "__" in column and split_feature_name(column)[0] in sensor_set and split_feature_name(column)[1] in stat_set
    ]


def collect_oof_probabilities(feature_table: pd.DataFrame, selected_features: list[str], n_splits: int = 5) -> pd.DataFrame:
    usable = feature_table.sort_values("sample_time").reset_index(drop=True)
    y = usable["is_out_of_spec"].astype(int).to_numpy()
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

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(usable[selected_features]):
        X_train = usable.iloc[train_idx][selected_features].copy()
        X_test = usable.iloc[test_idx][selected_features].copy()
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue

        fold_columns = [column for column in selected_features if X_train[column].notna().any()]
        if not fold_columns:
            continue

        model.fit(X_train[fold_columns], y_train)
        probabilities[test_idx] = model.predict_proba(X_test[fold_columns])[:, 1]

    result = usable.loc[~np.isnan(probabilities), ["sample_time", "t90", "is_out_of_spec"]].copy()
    result["probability"] = probabilities[~np.isnan(probabilities)]
    return result.reset_index(drop=True)


def threshold_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob >= threshold
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": float(round(threshold, 2)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "predicted_positive_ratio": float(pred.mean()),
    }


def sweep_thresholds(oof_frame: pd.DataFrame) -> list[dict[str, float]]:
    y_true = oof_frame["is_out_of_spec"].astype(int).to_numpy()
    prob = oof_frame["probability"].to_numpy(dtype=float)
    thresholds = np.arange(0.05, 1.00, 0.05)
    return [threshold_metrics(y_true, prob, float(threshold)) for threshold in thresholds]


def pick_warning_threshold(scan_rows: list[dict[str, float]]) -> dict[str, float]:
    candidates = [row for row in scan_rows if row["recall"] >= 0.80]
    if not candidates:
        return max(scan_rows, key=lambda row: (row["f1"], row["specificity"]))
    return max(candidates, key=lambda row: (row["specificity"], row["precision"], row["f1"]))


def pick_alarm_threshold(scan_rows: list[dict[str, float]]) -> dict[str, float]:
    candidates = [row for row in scan_rows if row["precision"] >= 0.23 and row["recall"] >= 0.35]
    if not candidates:
        return max(scan_rows, key=lambda row: (row["precision"], row["specificity"], row["f1"]))
    return max(candidates, key=lambda row: (row["precision"], row["specificity"], row["f1"]))


def apply_raw_two_level(probabilities: np.ndarray, warning_threshold: float, alarm_threshold: float) -> np.ndarray:
    states = np.zeros(len(probabilities), dtype=int)
    states[probabilities >= warning_threshold] = 1
    states[probabilities >= alarm_threshold] = 2
    return states


def apply_hysteresis_two_level(
    probabilities: np.ndarray,
    warning_threshold: float,
    alarm_threshold: float,
    warning_release_ratio: float = 0.80,
    alarm_release_ratio: float = 0.85,
) -> np.ndarray:
    states = np.zeros(len(probabilities), dtype=int)
    warning_off = warning_threshold * warning_release_ratio
    alarm_off = alarm_threshold * alarm_release_ratio
    state = 0
    for idx, value in enumerate(probabilities):
        if state == 0:
            if value >= alarm_threshold:
                state = 2
            elif value >= warning_threshold:
                state = 1
        elif state == 1:
            if value >= alarm_threshold:
                state = 2
            elif value < warning_off:
                state = 0
        else:
            if value >= alarm_off:
                state = 2
            elif value >= warning_threshold:
                state = 1
            elif value < warning_off:
                state = 0
        states[idx] = state
    return states


def summarize_policy(y_true: np.ndarray, states: np.ndarray) -> dict[str, float | int]:
    positive = y_true.astype(int)
    warning_or_alarm = states >= 1
    alarm_only = states == 2

    warning_tp = int(np.sum((warning_or_alarm == 1) & (positive == 1)))
    warning_fp = int(np.sum((warning_or_alarm == 1) & (positive == 0)))
    warning_fn = int(np.sum((warning_or_alarm == 0) & (positive == 1)))
    warning_tn = int(np.sum((warning_or_alarm == 0) & (positive == 0)))

    alarm_tp = int(np.sum((alarm_only == 1) & (positive == 1)))
    alarm_fp = int(np.sum((alarm_only == 1) & (positive == 0)))
    alarm_fn = int(np.sum((alarm_only == 0) & (positive == 1)))
    alarm_tn = int(np.sum((alarm_only == 0) & (positive == 0)))

    def safe_div(num: int, den: int) -> float:
        return float(num / den) if den else 0.0

    switch_count = int(np.sum(states[1:] != states[:-1])) if len(states) > 1 else 0
    return {
        "warning_precision": safe_div(warning_tp, warning_tp + warning_fp),
        "warning_recall": safe_div(warning_tp, warning_tp + warning_fn),
        "warning_specificity": safe_div(warning_tn, warning_tn + warning_fp),
        "alarm_precision": safe_div(alarm_tp, alarm_tp + alarm_fp),
        "alarm_recall": safe_div(alarm_tp, alarm_tp + alarm_fn),
        "alarm_specificity": safe_div(alarm_tn, alarm_tn + alarm_fp),
        "warning_rate": float(warning_or_alarm.mean()),
        "alarm_rate": float(alarm_only.mean()),
        "switch_count": switch_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two-level warning/alarm policy for the compact Phase 1 warning model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--stats", type=str, default="mean,min,last,max,std")
    parser.add_argument("--screening-prefix", type=str, default=SCREENING_PREFIX)
    parser.add_argument("--output-prefix", type=str, default="phase1_warning_alarm_policy")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    source_config = config["data_sources"]
    screening_path = artifact_dir / f"{args.screening_prefix}_sensor_rankings.csv"
    sensor_rankings = pd.read_csv(screening_path)
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
    feature_table = build_dcs_feature_table(
        labeled,
        dcs,
        window_minutes=args.window,
        min_points_per_window=int(config["window_search"]["min_points_per_window"]),
    )

    stats = [item.strip() for item in args.stats.split(",") if item.strip()]
    sensor_names = (
        sensor_rankings[sensor_rankings["window_minutes"] == args.window]["sensor_name"]
        .head(args.topk)
        .tolist()
    )
    selected_features = build_selected_features(feature_table, sensor_names, stats)
    oof_frame = collect_oof_probabilities(feature_table, selected_features)
    scan_rows = sweep_thresholds(oof_frame)
    warning_threshold_row = pick_warning_threshold(scan_rows)
    alarm_threshold_row = pick_alarm_threshold(scan_rows)

    y_true = oof_frame["is_out_of_spec"].astype(int).to_numpy()
    prob = oof_frame["probability"].to_numpy(dtype=float)

    raw_states = apply_raw_two_level(prob, warning_threshold_row["threshold"], alarm_threshold_row["threshold"])
    hysteresis_states = apply_hysteresis_two_level(prob, warning_threshold_row["threshold"], alarm_threshold_row["threshold"])

    raw_policy = summarize_policy(y_true, raw_states)
    hysteresis_policy = summarize_policy(y_true, hysteresis_states)

    summary = {
        "phase": config.get("phase", "phase1_warning"),
        "objective": config["objective"]["name"],
        "selected_window_minutes": int(args.window),
        "selected_topk_sensors": int(args.topk),
        "selected_stats": stats,
        "selected_sensor_names": sensor_names,
        "model_level_metrics": {
            "roc_auc": float(roc_auc_score(y_true, prob)),
            "average_precision": float(average_precision_score(y_true, prob)),
            "samples": int(len(oof_frame)),
        },
        "threshold_scan_rows": scan_rows,
        "recommended_thresholds": {
            "warning": warning_threshold_row,
            "alarm": alarm_threshold_row,
        },
        "policy_rows": [
            {
                "policy_name": "raw_two_level",
                **raw_policy,
            },
            {
                "policy_name": "hysteresis_two_level",
                **hysteresis_policy,
            },
        ],
    }

    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    pd.DataFrame(scan_rows).to_csv(
        artifact_dir / f"{args.output_prefix}_threshold_scan.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summary["policy_rows"]).to_csv(
        artifact_dir / f"{args.output_prefix}_policy_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oof_frame.to_csv(
        artifact_dir / f"{args.output_prefix}_oof_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
