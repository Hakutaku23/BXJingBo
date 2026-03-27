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


def parse_list_argument(value: str, cast=int) -> list[Any]:
    return [cast(item) for item in value.split(",") if item.strip()]


def split_feature_name(feature_name: str) -> tuple[str, str]:
    if "__" not in feature_name:
        return feature_name, "raw"
    sensor, stat = feature_name.split("__", 1)
    return sensor, stat


def score_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    thresholds = np.arange(0.05, 1.00, 0.05)
    best = None
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = int(np.sum((predicted == 1) & (y_true == 1)))
        fp = int(np.sum((predicted == 1) & (y_true == 0)))
        fn = int(np.sum((predicted == 0) & (y_true == 1)))

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
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "best_threshold": best["threshold"] if best else None,
        "best_precision": best["precision"] if best else None,
        "best_recall": best["recall"] if best else None,
        "best_f1": best["f1"] if best else None,
    }


def evaluate_feature_subset(feature_table: pd.DataFrame, selected_features: list[str], n_splits: int = 5) -> dict[str, Any]:
    if not selected_features:
        return {"status": "no_features"}

    usable = feature_table.sort_values("sample_time").reset_index(drop=True)
    y = usable["is_out_of_spec"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return {"status": "single_class", "samples": int(len(usable))}

    X = usable[selected_features]
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

        fold_columns = [column for column in selected_features if X_train[column].notna().any()]
        if not fold_columns:
            continue

        model.fit(X_train[fold_columns], y_train)
        probabilities[test_idx] = model.predict_proba(X_test[fold_columns])[:, 1]

    scored_mask = ~np.isnan(probabilities)
    if scored_mask.sum() == 0:
        return {"status": "no_scored_samples", "samples": int(len(usable))}

    metrics = score_predictions(y[scored_mask], probabilities[scored_mask])
    return {
        "status": "ok",
        "samples": int(len(usable)),
        "scored_samples": int(scored_mask.sum()),
        "feature_count": int(len(selected_features)),
        **metrics,
    }


def build_selected_features(feature_table: pd.DataFrame, sensors: list[str], stats: list[str]) -> list[str]:
    selected = []
    sensor_set = set(sensors)
    stat_set = set(stats)
    for column in feature_table.columns:
        if "__" not in column:
            continue
        sensor, stat = split_feature_name(column)
        if sensor in sensor_set and stat in stat_set:
            selected.append(column)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact Phase 1 warning modeling from screened DCS sensors.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--windows", type=str, default="10,45,50")
    parser.add_argument("--topk", type=str, default="5,10,15")
    parser.add_argument("--stat-sets", type=str, default="mean,min,last,max;mean,min,last,max,std")
    parser.add_argument("--screening-prefix", type=str, default=SCREENING_PREFIX)
    parser.add_argument("--output-prefix", type=str, default="phase1_compact_warning_modeling")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    windows = parse_list_argument(args.windows, int)
    topk_values = parse_list_argument(args.topk, int)
    stat_sets = [[item.strip() for item in chunk.split(",") if item.strip()] for chunk in args.stat_sets.split(";") if chunk.strip()]

    screening_path = artifact_dir / f"{args.screening_prefix}_sensor_rankings.csv"
    sensor_rankings = pd.read_csv(screening_path)

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

    rows: list[dict[str, Any]] = []
    for window_minutes in windows:
        feature_table = build_dcs_feature_table(
            labeled,
            dcs,
            window_minutes=window_minutes,
            min_points_per_window=int(config["window_search"]["min_points_per_window"]),
        )
        window_sensors = sensor_rankings[sensor_rankings["window_minutes"] == window_minutes]["sensor_name"].tolist()
        for topk in topk_values:
            selected_sensors = window_sensors[:topk]
            for stats in stat_sets:
                selected_features = build_selected_features(feature_table, selected_sensors, stats)
                metrics = evaluate_feature_subset(feature_table, selected_features)
                metrics.update(
                    {
                        "window_minutes": int(window_minutes),
                        "topk_sensors": int(topk),
                        "stats": ",".join(stats),
                        "sensor_names": selected_sensors,
                    }
                )
                rows.append(metrics)

    result = pd.DataFrame(rows).sort_values(
        ["roc_auc", "average_precision", "best_f1", "window_minutes", "topk_sensors"],
        ascending=[False, False, False, True, True],
    )
    summary = {
        "phase": config.get("phase", "phase1_warning"),
        "objective": config["objective"]["name"],
        "rows": result.to_dict(orient="records"),
    }

    result.to_csv(
        artifact_dir / f"{args.output_prefix}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
