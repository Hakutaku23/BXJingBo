from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from oos_alert_module import (
    DEFAULT_CONFIG_PATH,
    _load_assets,
    _prepare_alert_frame,
    _select_dcs_feature_columns,
)


DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"


def _build_model_registry() -> dict[str, object]:
    return {
        "logistic_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "random_forest_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=4,
                        random_state=42,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        min_samples_leaf=3,
                        random_state=42,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=6,
                        max_iter=250,
                        learning_rate=0.05,
                        min_samples_leaf=20,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def _threshold_scan(actual: pd.Series, score: pd.Series, thresholds: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        predicted = (score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(actual, predicted, zero_division=0)),
                "recall": float(recall_score(actual, predicted, zero_division=0)),
                "f1": float(f1_score(actual, predicted, zero_division=0)),
                "predicted_positive_ratio": float(predicted.mean()),
            }
        )
    return rows


def _summarize(actual: pd.Series, score: pd.Series) -> dict[str, object]:
    thresholds = [round(item, 2) for item in np.arange(0.05, 0.96, 0.05)]
    scan = _threshold_scan(actual, score, thresholds)
    best_threshold = max(
        scan,
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["recall"],
            item["threshold"],
        ),
    )
    return {
        "roc_auc": float(roc_auc_score(actual, score)),
        "average_precision": float(average_precision_score(actual, score)),
        "best_f1_threshold": float(best_threshold["threshold"]),
        "best_f1": float(best_threshold["f1"]),
        "best_precision": float(best_threshold["precision"]),
        "best_recall": float(best_threshold["recall"]),
        "threshold_scan": scan,
    }


def _predict_model(model, train_frame: pd.DataFrame, test_frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    estimator = clone(model)
    estimator.fit(train_frame[feature_columns], train_frame["is_out_of_spec"].astype(int))
    return estimator.predict_proba(test_frame[feature_columns])[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark alternative standalone out-of-spec classifiers for T90.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--splits", type=int, default=5, help="TimeSeriesSplit count.")
    args = parser.parse_args()

    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = results_dir / "oos_alert_model_benchmark_summary.json"
    summary_csv_path = results_dir / "oos_alert_model_benchmark_summary.csv"
    predictions_csv_path = results_dir / "oos_alert_model_benchmark_predictions.csv"

    _config, _stage_policy, base_casebase, ph_casebase = _load_assets(Path(args.config))
    frame = _prepare_alert_frame(base_casebase, ph_casebase).sort_values("sample_time").reset_index(drop=True)
    dcs_features = _select_dcs_feature_columns(frame)
    registry = _build_model_registry()
    ensemble_members = ["logistic_balanced", "random_forest_balanced", "extra_trees_balanced", "hist_gradient_boosting"]

    predictions: dict[str, pd.Series] = {
        name: pd.Series(index=frame.index, dtype=float) for name in registry
    }
    predictions["soft_voting_4"] = pd.Series(index=frame.index, dtype=float)

    tscv = TimeSeriesSplit(n_splits=args.splits)
    for fold_index, (train_index, test_index) in enumerate(tscv.split(frame), start=1):
        print(f"evaluating fold {fold_index}/{args.splits}")
        train_frame = frame.iloc[train_index].copy()
        test_frame = frame.iloc[test_index].copy()

        fold_scores: dict[str, np.ndarray] = {}
        for model_name, model in registry.items():
            fold_scores[model_name] = _predict_model(model, train_frame, test_frame, dcs_features)
            predictions[model_name].iloc[test_index] = fold_scores[model_name]

        soft_score = np.mean([fold_scores[name] for name in ensemble_members], axis=0)
        predictions["soft_voting_4"].iloc[test_index] = soft_score

    scored_mask = pd.DataFrame(predictions).notna().all(axis=1)
    scored_frame = frame.loc[scored_mask, ["sample_time", "t90", "is_in_spec", "is_out_of_spec"]].copy()
    actual = scored_frame["is_out_of_spec"].astype(int)

    summary_rows: list[dict[str, object]] = []
    detailed_summary: dict[str, object] = {
        "target_definition": "1 when T90 is outside 8.45 +/- 0.25, otherwise 0",
        "samples": int(len(frame)),
        "scored_samples": int(scored_mask.sum()),
        "models": {},
    }

    for model_name, score_series in predictions.items():
        score = score_series.loc[scored_mask].astype(float)
        metrics = _summarize(actual, score)
        detailed_summary["models"][model_name] = metrics
        summary_rows.append(
            {
                "model_name": model_name,
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "best_f1_threshold": metrics["best_f1_threshold"],
                "best_f1": metrics["best_f1"],
                "best_precision": metrics["best_precision"],
                "best_recall": metrics["best_recall"],
            }
        )
        scored_frame[f"prob_{model_name}"] = score.values

    best_by_f1 = max(summary_rows, key=lambda row: (row["best_f1"], row["average_precision"], row["roc_auc"]))
    best_by_ap = max(summary_rows, key=lambda row: (row["average_precision"], row["roc_auc"], row["best_f1"]))
    detailed_summary["recommendations"] = {
        "best_by_f1": best_by_f1,
        "best_by_average_precision": best_by_ap,
    }
    detailed_summary["artifacts"] = {
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "predictions_csv": str(predictions_csv_path),
    }

    pd.DataFrame(summary_rows).sort_values(
        ["best_f1", "average_precision", "roc_auc"], ascending=False
    ).to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    scored_frame.to_csv(predictions_csv_path, index=False, encoding="utf-8-sig")
    summary_json_path.write_text(json.dumps(detailed_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(detailed_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
