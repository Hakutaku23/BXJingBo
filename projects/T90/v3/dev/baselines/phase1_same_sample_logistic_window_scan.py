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


def score_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    thresholds = np.arange(0.05, 1.00, 0.05)
    best: dict[str, Any] | None = None

    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = int(np.sum((predicted == 1) & (y_true == 1)))
        fp = int(np.sum((predicted == 1) & (y_true == 0)))
        fn = int(np.sum((predicted == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        row = {
            "threshold": float(round(threshold, 2)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predicted_positive_ratio": float(predicted.mean()),
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


def evaluate_window(feature_table: pd.DataFrame, n_splits: int = 5) -> dict[str, Any]:
    if feature_table.empty:
        return {"status": "no_samples"}

    feature_columns = [column for column in feature_table.columns if column not in BLOCKED_COLUMNS]
    feature_columns = [column for column in feature_columns if feature_table[column].notna().any()]
    if not feature_columns:
        return {"status": "no_features"}

    usable = feature_table.sort_values("sample_time").reset_index(drop=True)
    y = usable["is_out_of_spec"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return {"status": "single_class", "samples": int(len(usable))}

    splitter = TimeSeriesSplit(n_splits=n_splits)
    probabilities = np.full(len(usable), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

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
                    n_jobs=None,
                ),
            ),
        ]
    )

    X = usable[feature_columns]
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2:
            continue

        fold_columns = [column for column in feature_columns if X_train[column].notna().any()]
        if not fold_columns:
            continue

        model.fit(X_train[fold_columns], y_train)
        fold_probability = model.predict_proba(X_test[fold_columns])[:, 1]
        probabilities[test_idx] = fold_probability

        fold_rows.append(
            {
                "fold": int(fold_index),
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
                "test_positive_ratio": float(np.mean(y_test)),
            }
        )

    scored_mask = ~np.isnan(probabilities)
    if scored_mask.sum() == 0:
        return {"status": "no_scored_samples", "samples": int(len(usable))}

    scored = score_probabilities(y[scored_mask], probabilities[scored_mask])
    return {
        "status": "ok",
        "samples": int(len(usable)),
        "scored_samples": int(scored_mask.sum()),
        "feature_count": int(len(feature_columns)),
        "folds": fold_rows,
        **scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a V3 same-sample warning logistic window scan.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--windows", type=str, default="")
    parser.add_argument("--output-prefix", type=str, default="phase1_same_sample_logistic_window_scan")
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
    candidate_windows = (
        [int(value) for value in args.windows.split(",") if value.strip()]
        if args.windows
        else [int(value) for value in config["window_search"]["candidate_minutes"]]
    )
    min_points_per_window = int(config["window_search"]["min_points_per_window"])

    lims_samples, _ = load_lims_samples(resolve_path(config_path, source_config["lims_path"]))
    labeled = add_out_of_spec_labels(lims_samples, target_spec).dropna(subset=["t90"]).copy()
    dcs = load_dcs_frame(
        resolve_path(config_path, source_config["dcs_main_path"]),
        resolve_path(config_path, source_config.get("dcs_supplemental_path")),
    )

    rows: list[dict[str, Any]] = []
    for window_minutes in candidate_windows:
        feature_table = build_dcs_feature_table(
            labeled,
            dcs,
            window_minutes=window_minutes,
            min_points_per_window=min_points_per_window,
        )
        metrics = evaluate_window(feature_table)
        metrics["window_minutes"] = int(window_minutes)
        rows.append(metrics)

    summary = {
        "phase": config.get("phase", "phase1_warning"),
        "objective": config["objective"]["name"],
        "target_spec": {
            "center": target_spec.center,
            "tolerance": target_spec.tolerance,
            "low": target_spec.low,
            "high": target_spec.high,
        },
        "rows": rows,
    }

    csv_path = artifact_dir / f"{args.output_prefix}_summary.csv"
    json_path = artifact_dir / f"{args.output_prefix}_summary.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
