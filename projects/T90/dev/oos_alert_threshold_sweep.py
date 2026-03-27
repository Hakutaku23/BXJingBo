from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_PREDICTIONS_CSV = DEFAULT_RESULTS_DIR / "oos_alert_experiment_predictions.csv"


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _evaluate_threshold(actual: pd.Series, score: pd.Series, threshold: float) -> dict[str, object]:
    predicted = (score >= threshold).astype(int)
    tp = int(((predicted == 1) & (actual == 1)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    false_positive_rate = _safe_divide(fp, fp + tn)
    false_negative_rate = _safe_divide(fn, fn + tp)
    predicted_positive_ratio = _safe_divide(tp + fp, len(actual))
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "threshold": float(round(threshold, 4)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "predicted_positive_ratio": predicted_positive_ratio,
        "f1": f1,
    }


def _choose_balanced(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(
        rows,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["recall"]),
            float(row["threshold"]),
        ),
    )


def _choose_low_miss(rows: list[dict[str, object]], min_recall: float) -> tuple[dict[str, object], dict[str, object]]:
    candidates = [row for row in rows if float(row["recall"]) >= min_recall]
    if candidates:
        chosen = max(
            candidates,
            key=lambda row: (
                float(row["precision"]),
                float(row["f1"]),
                float(row["specificity"]),
                float(row["threshold"]),
            ),
        )
        return chosen, {"minimum_recall": float(min_recall), "fallback_used": False}

    chosen = max(
        rows,
        key=lambda row: (
            float(row["recall"]),
            float(row["precision"]),
            float(row["f1"]),
            -float(row["false_positive_rate"]),
        ),
    )
    return chosen, {"minimum_recall": float(min_recall), "fallback_used": True}


def _choose_low_false_alarm(rows: list[dict[str, object]], min_precision: float) -> tuple[dict[str, object], dict[str, object]]:
    candidates = [row for row in rows if float(row["precision"]) >= min_precision]
    if candidates:
        chosen = max(
            candidates,
            key=lambda row: (
                float(row["recall"]),
                float(row["f1"]),
                float(row["specificity"]),
                -float(row["threshold"]),
            ),
        )
        return chosen, {"minimum_precision": float(min_precision), "fallback_used": False}

    chosen = max(
        rows,
        key=lambda row: (
            float(row["precision"]),
            float(row["specificity"]),
            float(row["f1"]),
            -float(row["predicted_positive_ratio"]),
        ),
    )
    return chosen, {"minimum_precision": float(min_precision), "fallback_used": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose practical operating thresholds for the dedicated T90 out-of-spec alert.")
    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS_CSV), help="CSV emitted by oos_alert_experiment.py.")
    parser.add_argument("--probability-column", default="prob_global_dcs", help="Probability column to sweep.")
    parser.add_argument("--output-prefix", default="", help="Optional prefix for the generated threshold sweep artifacts.")
    parser.add_argument("--grid-step", type=float, default=0.01, help="Threshold scan step.")
    parser.add_argument("--min-recall-low-miss", type=float, default=0.90, help="Recall floor for the low-miss operating point.")
    parser.add_argument("--min-precision-low-false-alarm", type=float, default=0.45, help="Precision floor for the low-false-alarm operating point.")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_csv)
    results_dir = predictions_path.parent
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    summary_json_path = results_dir / f"{prefix}oos_alert_threshold_sweep_summary.json"
    scan_csv_path = results_dir / f"{prefix}oos_alert_threshold_sweep_scan.csv"

    frame = pd.read_csv(predictions_path)
    frame = frame.dropna(subset=["is_out_of_spec", args.probability_column]).copy()
    actual = pd.to_numeric(frame["is_out_of_spec"], errors="coerce").fillna(0).astype(int)
    score = pd.to_numeric(frame[args.probability_column], errors="coerce").fillna(0.0)

    thresholds = np.arange(args.grid_step, 1.0, args.grid_step)
    rows = [_evaluate_threshold(actual, score, float(threshold)) for threshold in thresholds]
    scan_frame = pd.DataFrame(rows)
    scan_frame.to_csv(scan_csv_path, index=False, encoding="utf-8-sig")

    balanced = _choose_balanced(rows)
    low_miss, low_miss_rule = _choose_low_miss(rows, args.min_recall_low_miss)
    low_false_alarm, low_false_rule = _choose_low_false_alarm(rows, args.min_precision_low_false_alarm)

    summary = {
        "target_definition": "1 when T90 is outside 8.45 +/- 0.25, otherwise 0",
        "strategy": args.probability_column,
        "output_prefix": args.output_prefix,
        "samples": int(len(frame)),
        "out_of_spec_ratio": float(actual.mean()),
        "selection_rules": {
            "balanced_f1": {"objective": "maximize_f1"},
            "low_miss": low_miss_rule,
            "low_false_alarm": low_false_rule,
        },
        "recommended_thresholds": {
            "balanced_f1": balanced,
            "low_miss": low_miss,
            "low_false_alarm": low_false_alarm,
        },
        "default_operating_mode": {
            "name": "low_miss",
            "reason": "For product-quality warning, missing a bad batch is usually costlier than raising an extra manual review.",
            "threshold": float(low_miss["threshold"]),
        },
        "artifacts": {
            "scan_csv": str(scan_csv_path),
            "summary_json": str(summary_json_path),
        },
    }

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
