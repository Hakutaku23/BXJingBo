from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_PREDICTIONS_CSV = DEFAULT_RESULTS_DIR / "oos_alert_experiment_predictions.csv"
DEFAULT_THRESHOLD_SUMMARY = DEFAULT_RESULTS_DIR / "oos_alert_threshold_sweep_summary.json"


def _build_split_summary(total_samples: int, n_splits: int) -> list[dict[str, object]]:
    dummy = np.arange(total_samples)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(dummy), start=1):
        rows.append(
            {
                "fold": fold,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "train_ratio_vs_total": float(len(train_idx) / total_samples),
                "test_ratio_vs_total": float(len(test_idx) / total_samples),
                "train_test_ratio": float(len(train_idx) / len(test_idx)),
                "train_start": int(train_idx[0]),
                "train_end": int(train_idx[-1]),
                "test_start": int(test_idx[0]),
                "test_end": int(test_idx[-1]),
            }
        )
    return rows


def _load_threshold_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize feasibility of the standalone T90 out-of-spec alert model.")
    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS_CSV), help="CSV emitted by oos_alert_experiment.py.")
    parser.add_argument("--threshold-summary", default=str(DEFAULT_THRESHOLD_SUMMARY), help="JSON emitted by oos_alert_threshold_sweep.py.")
    parser.add_argument("--probability-column", default="prob_global_dcs", help="Probability column to visualize.")
    parser.add_argument("--output-prefix", default="", help="Optional prefix for the generated visualization artifacts.")
    parser.add_argument("--splits", type=int, default=5, help="Number of TimeSeriesSplit folds used in the experiment.")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_csv)
    threshold_path = Path(args.threshold_summary)
    results_dir = predictions_path.parent

    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    split_csv_path = results_dir / f"{prefix}oos_alert_split_summary.csv"
    split_json_path = results_dir / f"{prefix}oos_alert_split_summary.json"
    dashboard_png_path = results_dir / f"{prefix}oos_alert_visual_dashboard.png"

    frame = pd.read_csv(predictions_path)
    scored = frame.dropna(subset=["is_out_of_spec", args.probability_column]).copy()
    actual = pd.to_numeric(scored["is_out_of_spec"], errors="coerce").fillna(0).astype(int)
    score = pd.to_numeric(scored[args.probability_column], errors="coerce").fillna(0.0)
    threshold_summary = _load_threshold_summary(threshold_path)

    split_rows = _build_split_summary(total_samples=int(len(frame)), n_splits=args.splits)
    pd.DataFrame(split_rows).to_csv(split_csv_path, index=False, encoding="utf-8-sig")
    split_json_path.write_text(json.dumps(split_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fpr, tpr, _ = roc_curve(actual, score)
    precision_curve, recall_curve, _ = precision_recall_curve(actual, score)
    roc_auc = float(roc_auc_score(actual, score))
    ap = float(average_precision_score(actual, score))

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(fpr, tpr, color="#0f766e", linewidth=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(recall_curve, precision_curve, color="#b45309", linewidth=2, label=f"AP = {ap:.3f}")
    baseline = float(actual.mean())
    ax.hlines(baseline, 0, 1, linestyle="--", color="#9ca3af", linewidth=1, label=f"Base rate = {baseline:.3f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    bins = np.linspace(0.0, 1.0, 31)
    ax.hist(score[actual == 0], bins=bins, alpha=0.65, color="#60a5fa", label="In spec")
    ax.hist(score[actual == 1], bins=bins, alpha=0.65, color="#f97316", label="Out of spec")
    for mode_name, metrics in threshold_summary["recommended_thresholds"].items():
        ax.axvline(float(metrics["threshold"]), linestyle="--", linewidth=1.5, label=f"{mode_name} = {float(metrics['threshold']):.2f}")
    ax.set_title("Alert Score Distribution")
    ax.set_xlabel("Predicted Out-of-Spec Probability")
    ax.set_ylabel("Sample Count")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    threshold_rows = []
    for mode_name, metrics in threshold_summary["recommended_thresholds"].items():
        threshold_rows.append(
            {
                "mode": mode_name,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "fpr": float(metrics["false_positive_rate"]),
            }
        )
    threshold_frame = pd.DataFrame(threshold_rows)
    x = np.arange(len(threshold_frame))
    width = 0.25
    ax.bar(x - width, threshold_frame["precision"], width=width, color="#2563eb", label="Precision")
    ax.bar(x, threshold_frame["recall"], width=width, color="#16a34a", label="Recall")
    ax.bar(x + width, 1 - threshold_frame["fpr"], width=width, color="#dc2626", label="Specificity")
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_frame["mode"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Operating Point Comparison")
    ax.set_ylabel("Metric Value")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Standalone T90 Out-of-Spec Alert Feasibility\n"
        f"scored_samples={len(scored)}, total_samples={len(frame)}, probability_column={args.probability_column}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(dashboard_png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(
        json.dumps(
            {
                "dashboard_png": str(dashboard_png_path),
                "split_csv": str(split_csv_path),
                "split_json": str(split_json_path),
                "roc_auc": roc_auc,
                "average_precision": ap,
                "scored_samples": int(len(scored)),
                "total_samples": int(len(frame)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
