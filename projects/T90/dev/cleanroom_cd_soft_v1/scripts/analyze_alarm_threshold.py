from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cleanroom_cd_soft_v1.config import load_config, resolve_path
from train_supervised import feature_columns, json_ready, select_model_features, soft_binary_cross_entropy, tree_model_config


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "base.yaml"


def threshold_metrics(y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pred = s >= float(threshold)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "rows": int(len(y)),
        "actual_out_spec": int(y.sum()),
        "actual_in_spec": int((y == 0).sum()),
        "alarm_count": int(pred.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "false_alarm_share_among_alarms": float(1.0 - precision),
        "false_alarm_rate_among_in_spec": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "miss_rate_among_out_spec": float(fn / (tp + fn)) if (tp + fn) else 0.0,
        "alarm_rate": float(pred.mean()) if len(pred) else 0.0,
    }


def build_threshold_curve(y_true: pd.Series, score: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), score)
    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    denom = curve["precision"] + curve["recall"]
    curve["f1"] = np.where(denom > 0.0, 2.0 * curve["precision"] * curve["recall"] / denom, 0.0)
    curve["false_alarm_share_among_alarms"] = 1.0 - curve["precision"]
    return curve.sort_values("threshold").reset_index(drop=True)


def choose_thresholds(curve: pd.DataFrame, candidate_threshold: float) -> dict[str, float | None]:
    valid_curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["threshold", "precision", "recall", "f1"])
    if valid_curve.empty:
        return {"best_f1": None, "best_precision_at_recall_ge_0p80": None, "closest_to_p80_r80": None, "candidate": float(candidate_threshold)}

    max_f1 = float(valid_curve["f1"].max())
    best_f1_rows = valid_curve[np.isclose(valid_curve["f1"], max_f1)]
    best_f1_threshold = float(best_f1_rows.sort_values(["recall", "threshold"], ascending=[False, True]).iloc[0]["threshold"])

    recall_rows = valid_curve[valid_curve["recall"] >= 0.80].copy()
    best_recall_threshold: float | None = None
    if not recall_rows.empty:
        best_recall_threshold = float(recall_rows.sort_values(["precision", "f1", "threshold"], ascending=[False, False, True]).iloc[0]["threshold"])

    balanced = valid_curve.copy()
    balanced["distance_to_p80_r80"] = (balanced["precision"] - 0.80) ** 2 + (balanced["recall"] - 0.80) ** 2
    closest_threshold = float(balanced.sort_values(["distance_to_p80_r80", "f1"], ascending=[True, False]).iloc[0]["threshold"])

    return {
        "best_f1": best_f1_threshold,
        "best_precision_at_recall_ge_0p80": best_recall_threshold,
        "closest_to_p80_r80": closest_threshold,
        "candidate": float(candidate_threshold),
    }


def plot_pr_curve(y_true: pd.Series, score: np.ndarray, output_path: Path, selected_thresholds: dict[str, float | None]) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), score)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=170)
    ax.plot(recall, precision, color="#2458a6", linewidth=1.6, label="validation PR curve")
    for name, threshold in selected_thresholds.items():
        if threshold is None or len(thresholds) == 0:
            continue
        idx = int(np.abs(thresholds - threshold).argmin())
        ax.scatter(recall[idx], precision[idx], s=38, label=f"{name}: {threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_f1_threshold(curve: pd.DataFrame, output_path: Path, selected_thresholds: dict[str, float | None]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=170)
    ax.plot(curve["threshold"], curve["f1"], color="#c4472d", linewidth=1.5, label="validation F1")
    ax.plot(curve["threshold"], curve["precision"], color="#2458a6", linewidth=1.0, alpha=0.85, label="precision")
    ax.plot(curve["threshold"], curve["recall"], color="#2f7d32", linewidth=1.0, alpha=0.85, label="recall")
    for name, threshold in selected_thresholds.items():
        if threshold is None:
            continue
        ax.axvline(float(threshold), linewidth=0.9, linestyle="--", alpha=0.72, label=f"{name}: {threshold:.3f}")
    ax.set_xlabel("Alarm threshold: P(out-spec)")
    ax.set_ylabel("Score")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an out-spec alarm threshold using validation PR/F1 curves.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-lims-context", action="store_true")
    parser.add_argument("--include-downstream", action="store_true")
    parser.add_argument("--window-minutes", type=int, default=240)
    parser.add_argument("--lag-minutes", type=int, default=None)
    parser.add_argument("--candidate-threshold", type=float, default=0.42)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    prepared_run = args.prepared_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(prepared_run / "feature_table.csv")
    frame = frame[frame["split"].isin(["train", "valid", "test"])].copy()
    train = frame[frame["split"] == "train"].copy()
    valid = frame[frame["split"] == "valid"].copy()
    test = frame[frame["split"] == "test"].copy()

    target = "is_out_spec_obs"
    raw_features = feature_columns(
        frame,
        include_lims_context=bool(args.include_lims_context),
        include_downstream=bool(args.include_downstream),
        window_minutes=args.window_minutes,
        lag_minutes=args.lag_minutes,
    )
    features, feature_audit = select_model_features(train, raw_features, target, config)
    if not features:
        raise ValueError("No features selected for threshold analysis.")

    tree_cfg = tree_model_config(config)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    loss="log_loss",
                    n_estimators=tree_cfg["n_estimators"],
                    learning_rate=tree_cfg["learning_rate"],
                    max_depth=tree_cfg["max_depth"],
                    subsample=tree_cfg["subsample"],
                    min_samples_leaf=tree_cfg["min_samples_leaf"],
                    random_state=tree_cfg["random_state"],
                ),
            ),
        ]
    )
    fit_kwargs = {}
    if "sample_weight" in train.columns:
        fit_kwargs["model__sample_weight"] = train["sample_weight"].to_numpy(dtype=float)
    model.fit(train[features], train[target].astype(int), **fit_kwargs)

    valid_score = model.predict_proba(valid[features])[:, 1]
    test_score = model.predict_proba(test[features])[:, 1]
    threshold_curve = build_threshold_curve(valid[target], valid_score)
    selected = choose_thresholds(threshold_curve, args.candidate_threshold)

    valid_scored = valid[["sample_id", "sample_time", "t90", "is_out_spec_obs", "sample_weight"]].copy()
    valid_scored["out_spec_alarm_probability"] = valid_score
    test_scored = test[["sample_id", "sample_time", "t90", "is_out_spec_obs", "sample_weight"]].copy()
    test_scored["out_spec_alarm_probability"] = test_score

    threshold_rows: list[dict[str, Any]] = []
    for name, threshold in selected.items():
        if threshold is None:
            continue
        threshold_rows.append({"split": "valid", "threshold_name": name, **threshold_metrics(valid[target], valid_score, threshold)})
        threshold_rows.append({"split": "test", "threshold_name": name, **threshold_metrics(test[target], test_score, threshold)})
    threshold_metrics_frame = pd.DataFrame(threshold_rows)

    valid_auc = float(roc_auc_score(valid[target].astype(int), valid_score)) if valid[target].nunique() > 1 else None
    test_auc = float(roc_auc_score(test[target].astype(int), test_score)) if test[target].nunique() > 1 else None
    summary = {
        "prepared_run_dir": str(prepared_run),
        "output_dir": str(output_dir),
        "target": target,
        "include_lims_context": bool(args.include_lims_context),
        "include_downstream": bool(args.include_downstream),
        "window_minutes": int(args.window_minutes) if args.window_minutes is not None else None,
        "lag_minutes": int(args.lag_minutes) if args.lag_minutes is not None else None,
        "candidate_threshold": float(args.candidate_threshold),
        "selected_thresholds": selected,
        "validation_average_precision": float(average_precision_score(valid[target].astype(int), valid_score)),
        "validation_brier": float(brier_score_loss(valid[target].astype(int), valid_score)),
        "validation_roc_auc": valid_auc,
        "test_average_precision": float(average_precision_score(test[target].astype(int), test_score)),
        "test_brier": float(brier_score_loss(test[target].astype(int), test_score)),
        "test_roc_auc": test_auc,
        "feature_selection": feature_audit,
        "tree_config": tree_cfg,
    }

    threshold_curve_path = output_dir / "validation_alarm_threshold_curve.csv"
    threshold_metrics_path = output_dir / "alarm_threshold_metrics.csv"
    valid_scored_path = output_dir / "validation_alarm_scores.csv"
    test_scored_path = output_dir / "test_alarm_scores.csv"
    pr_plot_path = output_dir / "validation_alarm_pr_curve.png"
    f1_plot_path = output_dir / "validation_alarm_f1_threshold_curve.png"
    summary_path = output_dir / "alarm_threshold_summary.json"

    threshold_curve.to_csv(threshold_curve_path, index=False, encoding="utf-8-sig")
    threshold_metrics_frame.to_csv(threshold_metrics_path, index=False, encoding="utf-8-sig")
    valid_scored.to_csv(valid_scored_path, index=False, encoding="utf-8-sig")
    test_scored.to_csv(test_scored_path, index=False, encoding="utf-8-sig")
    plot_pr_curve(valid[target], valid_score, pr_plot_path, selected)
    plot_f1_threshold(threshold_curve, f1_plot_path, selected)
    summary_path.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(json_ready({**summary, "threshold_metrics": threshold_rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
