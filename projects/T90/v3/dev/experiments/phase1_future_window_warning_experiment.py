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


def collect_oof_probabilities(feature_table: pd.DataFrame, selected_features: list[str], target: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    work = feature_table.sort_values("sample_time").reset_index(drop=True).copy()
    target_aligned = target.loc[work.index]
    usable_mask = target_aligned.notna()
    work = work.loc[usable_mask].reset_index(drop=True)
    y = target_aligned.loc[usable_mask].astype(int).to_numpy()

    probabilities = np.full(len(work), np.nan, dtype=float)
    X = work[selected_features]

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
    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue

        fold_columns = [column for column in selected_features if X_train[column].notna().any()]
        if not fold_columns:
            continue

        model.fit(X_train[fold_columns], y_train)
        probabilities[test_idx] = model.predict_proba(X_test[fold_columns])[:, 1]

    result = work.loc[~np.isnan(probabilities), ["sample_time", "t90", "is_out_of_spec"]].copy()
    result["future_target"] = y[~np.isnan(probabilities)]
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
    y_true = oof_frame["future_target"].astype(int).to_numpy()
    prob = oof_frame["probability"].to_numpy(dtype=float)
    return [threshold_metrics(y_true, prob, float(threshold)) for threshold in np.arange(0.05, 1.00, 0.05)]


def choose_threshold(scan_rows: list[dict[str, float]]) -> dict[str, float]:
    candidates = [row for row in scan_rows if row["recall"] >= 0.80]
    if not candidates:
        return max(scan_rows, key=lambda row: (row["f1"], row["specificity"]))
    return max(candidates, key=lambda row: (row["specificity"], row["precision"], row["f1"]))


def build_actual_bad_segments(frame: pd.DataFrame, gap_limit_minutes: float) -> list[dict[str, Any]]:
    work = frame.sort_values("sample_time").reset_index(drop=True).copy()
    work["sample_time"] = pd.to_datetime(work["sample_time"], errors="coerce")
    labels = work["is_out_of_spec"].astype(int).to_numpy()
    times = work["sample_time"].to_numpy(dtype="datetime64[ns]")

    segments: list[dict[str, Any]] = []
    start = 0
    for idx in range(1, len(work)):
        label_changed = labels[idx] != labels[idx - 1]
        gap_minutes = (times[idx] - times[idx - 1]) / np.timedelta64(1, "m")
        gap_break = float(gap_minutes) > gap_limit_minutes
        if label_changed or gap_break:
            segments.append({"label": int(labels[start]), "start_idx": int(start), "end_idx": int(idx - 1)})
            start = idx
    segments.append({"label": int(labels[start]), "start_idx": int(start), "end_idx": int(len(work) - 1)})
    return [segment for segment in segments if segment["label"] == 1]


def evaluate_actual_prewarn(oof_frame: pd.DataFrame, threshold: float, horizon_minutes: float) -> dict[str, Any]:
    work = oof_frame.sort_values("sample_time").reset_index(drop=True).copy()
    work["sample_time"] = pd.to_datetime(work["sample_time"], errors="coerce")
    work["warning"] = work["probability"] >= threshold

    intervals = work["sample_time"].diff().dropna().dt.total_seconds() / 60.0
    median_interval = float(intervals.median()) if not intervals.empty else 240.0
    bad_segments = build_actual_bad_segments(work, gap_limit_minutes=median_interval * 2.0)

    prewarn_hits = 0
    start_hits = 0
    lead_times = []
    for segment in bad_segments:
        segment_start_time = work.iloc[segment["start_idx"]]["sample_time"]
        if bool(work.iloc[segment["start_idx"]]["warning"]):
            start_hits += 1

        history = work.iloc[: segment["start_idx"]]
        if history.empty:
            continue
        history = history[history["sample_time"] >= segment_start_time - pd.Timedelta(minutes=horizon_minutes)]
        history = history[history["warning"]]
        if history.empty:
            continue
        prewarn_hits += 1
        last_time = history.iloc[-1]["sample_time"]
        lead_times.append(float((segment_start_time - last_time).total_seconds() / 60.0))

    return {
        "bad_segment_count": int(len(bad_segments)),
        "actual_bad_segment_start_warning_ratio": float(start_hits / len(bad_segments)) if bad_segments else 0.0,
        "actual_bad_segment_prewarn_ratio": float(prewarn_hits / len(bad_segments)) if bad_segments else 0.0,
        "median_actual_prewarn_lead_minutes": float(np.median(lead_times)) if lead_times else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Future-window warning comparison on the current 8-minute mainline feature chain.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--stats", type=str, default="mean,min,last,max")
    parser.add_argument("--screening-prefix", type=str, default="phase1_short_window_feature_screening")
    parser.add_argument("--future-horizons", type=str, default="120,240,360,480")
    parser.add_argument(
        "--same-sample-baseline",
        type=Path,
        default=V3_ROOT / "dev" / "artifacts" / "phase1_event_level_window8_summary.json",
    )
    parser.add_argument("--output-prefix", type=str, default="phase1_future_window_warning")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

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
    feature_table = build_dcs_feature_table(
        labeled,
        dcs,
        window_minutes=args.window,
        min_points_per_window=int(config["window_search"]["min_points_per_window"]),
    )

    screening_path = artifact_dir / f"{args.screening_prefix}_sensor_rankings.csv"
    sensor_rankings = pd.read_csv(screening_path)
    selected_sensors = (
        sensor_rankings[sensor_rankings["window_minutes"] == args.window]["sensor_name"]
        .head(args.topk)
        .tolist()
    )
    selected_stats = [item.strip() for item in args.stats.split(",") if item.strip()]
    selected_features = [
        column
        for column in feature_table.columns
        if "__" in column
        and column.split("__", 1)[0] in set(selected_sensors)
        and column.split("__", 1)[1] in set(selected_stats)
    ]

    same_sample_baseline = json.loads(args.same_sample_baseline.read_text(encoding="utf-8"))
    same_sample_prewarn_ratio = None
    for row in same_sample_baseline["rows"]:
        if row["policy_name"] == "raw_two_level":
            same_sample_prewarn_ratio = row["bad_segment_prewarn_warning_ratio"]

    rows = []
    oof_rows: list[pd.DataFrame] = []
    for horizon in [int(value) for value in args.future_horizons.split(",") if value.strip()]:
        future_target = build_future_window_label(feature_table, horizon_minutes=horizon)
        oof_frame = collect_oof_probabilities(feature_table, selected_features, future_target)
        if oof_frame.empty or len(np.unique(oof_frame["future_target"])) < 2:
            rows.append(
                {
                    "future_horizon_minutes": int(horizon),
                    "status": "not_enough_future_labels",
                }
            )
            continue

        scan_rows = sweep_thresholds(oof_frame)
        threshold_row = choose_threshold(scan_rows)
        model_metrics = {
            "roc_auc": float(roc_auc_score(oof_frame["future_target"], oof_frame["probability"])),
            "average_precision": float(average_precision_score(oof_frame["future_target"], oof_frame["probability"])),
            "samples": int(len(oof_frame)),
            "positive_ratio": float(oof_frame["future_target"].mean()),
        }
        actual_prewarn = evaluate_actual_prewarn(oof_frame, threshold=float(threshold_row["threshold"]), horizon_minutes=float(horizon))

        row = {
            "future_horizon_minutes": int(horizon),
            "status": "ok",
            **model_metrics,
            "threshold": float(threshold_row["threshold"]),
            "precision": float(threshold_row["precision"]),
            "recall": float(threshold_row["recall"]),
            "specificity": float(threshold_row["specificity"]),
            "f1": float(threshold_row["f1"]),
            "actual_bad_segment_prewarn_ratio": float(actual_prewarn["actual_bad_segment_prewarn_ratio"]),
            "actual_bad_segment_start_warning_ratio": float(actual_prewarn["actual_bad_segment_start_warning_ratio"]),
            "median_actual_prewarn_lead_minutes": actual_prewarn["median_actual_prewarn_lead_minutes"],
            "same_sample_baseline_prewarn_ratio": same_sample_prewarn_ratio,
        }
        rows.append(row)

        export = oof_frame.copy()
        export["future_horizon_minutes"] = int(horizon)
        oof_rows.append(export)

    result = pd.DataFrame(rows).sort_values(
        ["actual_bad_segment_prewarn_ratio", "roc_auc", "average_precision", "future_horizon_minutes"],
        ascending=[False, False, False, True],
    )

    summary = {
        "selected_window_minutes": int(args.window),
        "selected_topk_sensors": int(args.topk),
        "selected_stats": selected_stats,
        "selected_sensor_names": selected_sensors,
        "rows": result.to_dict(orient="records"),
    }

    result.to_csv(
        artifact_dir / f"{args.output_prefix}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    if oof_rows:
        pd.concat(oof_rows, ignore_index=True).to_csv(
            artifact_dir / f"{args.output_prefix}_oof_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
