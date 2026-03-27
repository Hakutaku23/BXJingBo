from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from oos_alert_module import DEFAULT_CONFIG_PATH, _load_assets, _prepare_alert_frame, _select_dcs_feature_columns


DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
TARGET_LOW = 8.45 - 0.25
TARGET_HIGH = 8.45 + 0.25


def _build_small_model_registry() -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "gaussian_nb": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=42,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gb": Pipeline(
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


def _fit_predict_supervised(
    estimator: Pipeline,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
) -> np.ndarray:
    y_train = train_frame[target_column].astype(int)
    if y_train.nunique() < 2:
        constant = float(y_train.iloc[0]) if not y_train.empty else 0.0
        return np.full(len(test_frame), constant, dtype=float)

    model = clone(estimator)
    model.fit(train_frame[feature_columns], y_train)
    return model.predict_proba(test_frame[feature_columns])[:, 1]


def _fit_predict_iforest(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    in_spec_train = train_frame.loc[train_frame["is_in_spec"] == 1, feature_columns]
    if in_spec_train.empty:
        return np.zeros(len(test_frame), dtype=float)

    X_train_ref = scaler.fit_transform(imputer.fit_transform(in_spec_train))
    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train_ref)

    X_train_all = scaler.transform(imputer.transform(train_frame[feature_columns]))
    X_test = scaler.transform(imputer.transform(test_frame[feature_columns]))

    train_scores = -model.decision_function(X_train_all)
    test_scores = -model.decision_function(X_test)
    low = float(np.nanmin(train_scores))
    high = float(np.nanmax(train_scores))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(len(test_frame), dtype=float)
    normalized = (test_scores - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def _threshold_scan(actual: pd.Series, score: pd.Series) -> list[dict[str, object]]:
    thresholds = [round(item, 2) for item in np.arange(0.05, 0.96, 0.05)]
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
    scan = _threshold_scan(actual, score)
    best_threshold = max(
        scan,
        key=lambda row: (row["f1"], row["precision"], row["recall"], row["threshold"]),
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


def _make_method_plot(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    summary_frame = pd.DataFrame(summary_rows)
    ordered = summary_frame.sort_values(["best_f1", "average_precision", "roc_auc"], ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("best_f1", "Best F1"),
        ("average_precision", "Average Precision"),
        ("roc_auc", "ROC AUC"),
    ]
    colors = ["#2563eb", "#f97316", "#16a34a"]
    for ax, (column, title), color in zip(axes, metrics, colors):
        ax.barh(ordered["method_name"], ordered[column], color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("T90 Out-of-Spec Alert Method Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate broader method families for standalone T90 out-of-spec alerting.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--splits", type=int, default=5, help="TimeSeriesSplit count.")
    args = parser.parse_args()

    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = results_dir / "oos_alert_method_experiment_summary.json"
    summary_csv_path = results_dir / "oos_alert_method_experiment_summary.csv"
    predictions_csv_path = results_dir / "oos_alert_method_experiment_predictions.csv"
    plot_png_path = results_dir / "oos_alert_method_experiment_metrics.png"

    _config, _stage_policy, base_casebase, ph_casebase = _load_assets(Path(args.config))
    frame = _prepare_alert_frame(base_casebase, ph_casebase).sort_values("sample_time").reset_index(drop=True)
    frame["is_above_spec"] = (pd.to_numeric(frame["t90"], errors="coerce") > TARGET_HIGH).astype(int)
    frame["is_below_spec"] = (pd.to_numeric(frame["t90"], errors="coerce") < TARGET_LOW).astype(int)
    dcs_features = _select_dcs_feature_columns(frame)

    registry = _build_small_model_registry()
    member_names = list(registry.keys())
    methods = {
        "single_logistic_oos": pd.Series(index=frame.index, dtype=float),
        "single_extra_trees_oos": pd.Series(index=frame.index, dtype=float),
        "committee_soft_vote_oos": pd.Series(index=frame.index, dtype=float),
        "committee_hard_vote_oos": pd.Series(index=frame.index, dtype=float),
        "dual_head_logistic_max": pd.Series(index=frame.index, dtype=float),
        "dual_head_committee_soft_max": pd.Series(index=frame.index, dtype=float),
        "dual_head_committee_hard_max": pd.Series(index=frame.index, dtype=float),
        "iforest_in_spec_anomaly": pd.Series(index=frame.index, dtype=float),
        "hybrid_dual_committee_plus_iforest": pd.Series(index=frame.index, dtype=float),
    }

    tscv = TimeSeriesSplit(n_splits=args.splits)
    for fold_index, (train_index, test_index) in enumerate(tscv.split(frame), start=1):
        print(f"evaluating fold {fold_index}/{args.splits}")
        train_frame = frame.iloc[train_index].copy()
        test_frame = frame.iloc[test_index].copy()

        oos_scores: dict[str, np.ndarray] = {}
        above_scores: dict[str, np.ndarray] = {}
        below_scores: dict[str, np.ndarray] = {}
        for model_name, estimator in registry.items():
            oos_scores[model_name] = _fit_predict_supervised(
                estimator,
                train_frame,
                test_frame,
                feature_columns=dcs_features,
                target_column="is_out_of_spec",
            )
            above_scores[model_name] = _fit_predict_supervised(
                estimator,
                train_frame,
                test_frame,
                feature_columns=dcs_features,
                target_column="is_above_spec",
            )
            below_scores[model_name] = _fit_predict_supervised(
                estimator,
                train_frame,
                test_frame,
                feature_columns=dcs_features,
                target_column="is_below_spec",
            )

        committee_soft_oos = np.mean([oos_scores[name] for name in member_names], axis=0)
        committee_hard_oos = np.mean([(oos_scores[name] >= 0.5).astype(float) for name in member_names], axis=0)
        dual_head_logistic = np.maximum(above_scores["logistic"], below_scores["logistic"])
        dual_head_committee_soft = np.maximum(
            np.mean([above_scores[name] for name in member_names], axis=0),
            np.mean([below_scores[name] for name in member_names], axis=0),
        )
        dual_head_committee_hard = np.maximum(
            np.mean([(above_scores[name] >= 0.5).astype(float) for name in member_names], axis=0),
            np.mean([(below_scores[name] >= 0.5).astype(float) for name in member_names], axis=0),
        )
        iforest_score = _fit_predict_iforest(train_frame, test_frame, feature_columns=dcs_features)
        hybrid_score = np.maximum(dual_head_committee_soft, iforest_score)

        methods["single_logistic_oos"].iloc[test_index] = oos_scores["logistic"]
        methods["single_extra_trees_oos"].iloc[test_index] = oos_scores["extra_trees"]
        methods["committee_soft_vote_oos"].iloc[test_index] = committee_soft_oos
        methods["committee_hard_vote_oos"].iloc[test_index] = committee_hard_oos
        methods["dual_head_logistic_max"].iloc[test_index] = dual_head_logistic
        methods["dual_head_committee_soft_max"].iloc[test_index] = dual_head_committee_soft
        methods["dual_head_committee_hard_max"].iloc[test_index] = dual_head_committee_hard
        methods["iforest_in_spec_anomaly"].iloc[test_index] = iforest_score
        methods["hybrid_dual_committee_plus_iforest"].iloc[test_index] = hybrid_score

    scored_mask = pd.DataFrame(methods).notna().all(axis=1)
    scored_frame = frame.loc[scored_mask, ["sample_time", "t90", "is_in_spec", "is_out_of_spec", "is_above_spec", "is_below_spec"]].copy()
    actual = scored_frame["is_out_of_spec"].astype(int)

    summary_rows: list[dict[str, object]] = []
    detailed_summary: dict[str, object] = {
        "target_definition": "1 when T90 is outside 8.45 +/- 0.25, otherwise 0",
        "samples": int(len(frame)),
        "scored_samples": int(scored_mask.sum()),
        "methods": {},
    }

    for method_name, score_series in methods.items():
        score = score_series.loc[scored_mask].astype(float)
        metrics = _summarize(actual, score)
        detailed_summary["methods"][method_name] = metrics
        summary_rows.append(
            {
                "method_name": method_name,
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "best_f1_threshold": metrics["best_f1_threshold"],
                "best_f1": metrics["best_f1"],
                "best_precision": metrics["best_precision"],
                "best_recall": metrics["best_recall"],
            }
        )
        scored_frame[f"score_{method_name}"] = score.values

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
        "plot_png": str(plot_png_path),
    }

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["best_f1", "average_precision", "roc_auc"], ascending=False
    )
    summary_frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    scored_frame.to_csv(predictions_csv_path, index=False, encoding="utf-8-sig")
    _make_method_plot(summary_rows, plot_png_path)
    summary_json_path.write_text(json.dumps(detailed_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(detailed_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
