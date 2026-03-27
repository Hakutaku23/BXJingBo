from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_PREDICTIONS_CSV = DEFAULT_RESULTS_DIR / "oos_alert_model_benchmark_predictions.csv"
DEFAULT_THRESHOLD_SUMMARY = DEFAULT_RESULTS_DIR / "logistic_balanced_oos_alert_threshold_sweep_summary.json"


@dataclass(frozen=True)
class PolicySpec:
    name: str
    kind: str
    threshold: float
    off_threshold: float | None = None
    count_k: int | None = None
    count_n: int | None = None
    alpha: float | None = None


def _load_thresholds(path: Path) -> dict[str, float]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: float(payload["threshold"])
        for name, payload in summary["recommended_thresholds"].items()
    }


def _build_policy_specs(thresholds: dict[str, float]) -> list[PolicySpec]:
    low_miss = thresholds["low_miss"]
    balanced = thresholds["balanced_f1"]
    low_false = thresholds["low_false_alarm"]
    return [
        PolicySpec("raw_low_miss", "raw", low_miss),
        PolicySpec("raw_balanced", "raw", balanced),
        PolicySpec("raw_low_false_alarm", "raw", low_false),
        PolicySpec("consecutive_2_balanced", "consecutive", balanced, count_k=2),
        PolicySpec("consecutive_2_low_miss", "consecutive", low_miss, count_k=2),
        PolicySpec("consecutive_3_low_miss", "consecutive", low_miss, count_k=3),
        PolicySpec("k2_of_3_low_miss", "k_of_n", low_miss, count_k=2, count_n=3),
        PolicySpec("k2_of_3_balanced", "k_of_n", balanced, count_k=2, count_n=3),
        PolicySpec("k3_of_5_low_miss", "k_of_n", low_miss, count_k=3, count_n=5),
        PolicySpec("ema_hysteresis_balanced", "ema_hysteresis", balanced, off_threshold=low_miss, alpha=0.5),
        PolicySpec("ema_hysteresis_conservative", "ema_hysteresis", low_false, off_threshold=balanced, alpha=0.4),
    ]


def _apply_policy(score: pd.Series, policy: PolicySpec) -> pd.Series:
    values = score.to_numpy(dtype=float)
    output = np.zeros(len(values), dtype=int)

    if policy.kind == "raw":
        output = (values >= policy.threshold).astype(int)
        return pd.Series(output, index=score.index)

    if policy.kind == "consecutive":
        window = int(policy.count_k or 1)
        above = (values >= policy.threshold).astype(int)
        run = 0
        for i, flag in enumerate(above):
            run = run + 1 if flag else 0
            output[i] = 1 if run >= window else 0
        return pd.Series(output, index=score.index)

    if policy.kind == "k_of_n":
        k = int(policy.count_k or 1)
        n = int(policy.count_n or k)
        above = (values >= policy.threshold).astype(int)
        rolling_sum = pd.Series(above).rolling(window=n, min_periods=1).sum().to_numpy()
        output = (rolling_sum >= k).astype(int)
        return pd.Series(output, index=score.index)

    if policy.kind == "ema_hysteresis":
        alpha = float(policy.alpha or 0.5)
        on_threshold = float(policy.threshold)
        off_threshold = float(policy.off_threshold if policy.off_threshold is not None else policy.threshold)
        ema = values[0] if len(values) else 0.0
        state = 0
        for i, value in enumerate(values):
            ema = alpha * value + (1.0 - alpha) * ema if i > 0 else value
            if state == 0 and ema >= on_threshold:
                state = 1
            elif state == 1 and ema < off_threshold:
                state = 0
            output[i] = state
        return pd.Series(output, index=score.index)

    raise ValueError(f"Unsupported policy kind: {policy.kind}")


def _compute_segments(label: pd.Series) -> list[tuple[int, int, int]]:
    values = label.astype(int).to_numpy()
    if len(values) == 0:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    current = int(values[0])
    for i in range(1, len(values)):
        if int(values[i]) != current:
            segments.append((start, i - 1, current))
            start = i
            current = int(values[i])
    segments.append((start, len(values) - 1, current))
    return segments


def _segment_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    segments = _compute_segments(actual)
    bad_segments = [seg for seg in segments if seg[2] == 1]
    good_segments = [seg for seg in segments if seg[2] == 0]

    bad_hits = 0
    for start, end, _ in bad_segments:
        if predicted.iloc[start : end + 1].max() == 1:
            bad_hits += 1

    good_false_alarm = 0
    for start, end, _ in good_segments:
        if predicted.iloc[start : end + 1].max() == 1:
            good_false_alarm += 1

    return {
        "bad_segment_count": float(len(bad_segments)),
        "good_segment_count": float(len(good_segments)),
        "bad_segment_recall": float(bad_hits / len(bad_segments)) if bad_segments else 0.0,
        "good_segment_false_alarm_ratio": float(good_false_alarm / len(good_segments)) if good_segments else 0.0,
    }


def _signal_stability_metrics(predicted: pd.Series) -> dict[str, float]:
    signal = predicted.astype(int).to_numpy()
    if len(signal) == 0:
        return {
            "alarm_switch_count": 0.0,
            "alarm_on_event_count": 0.0,
            "mean_alarm_run_length": 0.0,
        }

    switches = int(np.abs(np.diff(signal)).sum())
    on_events = int(((signal[1:] == 1) & (signal[:-1] == 0)).sum() + (1 if signal[0] == 1 else 0))

    run_lengths: list[int] = []
    run = 0
    for value in signal:
        if value == 1:
            run += 1
        elif run > 0:
            run_lengths.append(run)
            run = 0
    if run > 0:
        run_lengths.append(run)

    return {
        "alarm_switch_count": float(switches),
        "alarm_on_event_count": float(on_events),
        "mean_alarm_run_length": float(np.mean(run_lengths)) if run_lengths else 0.0,
    }


def _summarize_policy(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    precision = float(precision_score(actual, predicted, zero_division=0))
    recall = float(recall_score(actual, predicted, zero_division=0))
    f1 = float(f1_score(actual, predicted, zero_division=0))

    tp = int(((predicted == 1) & (actual == 1)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())

    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
    false_negative_rate = float(fn / (fn + tp)) if (fn + tp) else 0.0
    predicted_positive_ratio = float(predicted.mean())

    summary = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "predicted_positive_ratio": predicted_positive_ratio,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
    summary.update(_segment_metrics(actual, predicted))
    summary.update(_signal_stability_metrics(predicted))
    return summary


def _choose_recommendation(rows: list[dict[str, object]]) -> dict[str, object]:
    candidates = [
        row
        for row in rows
        if float(row["bad_segment_recall"]) >= 0.85 and float(row["recall"]) >= 0.70
    ]
    if not candidates:
        candidates = rows

    chosen = max(
        candidates,
        key=lambda row: (
            -float(row["good_segment_false_alarm_ratio"]),
            -float(row["false_positive_rate"]),
            float(row["precision"]),
            -float(row["alarm_switch_count"]),
            float(row["bad_segment_recall"]),
            float(row["f1"]),
        ),
    )
    return chosen


def _plot_policy_metrics(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    frame = pd.DataFrame(summary_rows).sort_values(
        ["bad_segment_recall", "precision", "specificity"], ascending=[False, False, False]
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    axes[0].barh(frame["policy_name"], frame["bad_segment_recall"], color="#2563eb")
    axes[0].set_title("Bad Segment Recall")
    axes[0].set_xlim(0, 1)
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].barh(frame["policy_name"], frame["good_segment_false_alarm_ratio"], color="#dc2626")
    axes[1].set_title("Good Segment False Alarm Ratio")
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis="x", alpha=0.2)

    axes[2].barh(frame["policy_name"], frame["alarm_switch_count"], color="#16a34a")
    axes[2].set_title("Alarm Switch Count")
    axes[2].grid(axis="x", alpha=0.2)

    fig.suptitle("Temporal Alarm Policy Comparison on Logistic Alert Score", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate temporal alarm policies on top of logistic T90 out-of-spec scores.")
    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS_CSV), help="Prediction CSV from oos_alert_model_benchmark.py.")
    parser.add_argument("--probability-column", default="prob_logistic_balanced", help="Probability column to use as the base score.")
    parser.add_argument("--threshold-summary", default=str(DEFAULT_THRESHOLD_SUMMARY), help="Threshold summary JSON for the logistic model.")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_csv)
    threshold_summary_path = Path(args.threshold_summary)
    results_dir = predictions_path.parent
    summary_json_path = results_dir / "oos_alert_temporal_policy_summary.json"
    summary_csv_path = results_dir / "oos_alert_temporal_policy_summary.csv"
    plot_png_path = results_dir / "oos_alert_temporal_policy_metrics.png"
    signal_csv_path = results_dir / "oos_alert_temporal_policy_signals.csv"

    frame = pd.read_csv(predictions_path)
    frame = frame.dropna(subset=["is_out_of_spec", args.probability_column]).copy()
    actual = pd.to_numeric(frame["is_out_of_spec"], errors="coerce").fillna(0).astype(int)
    score = pd.to_numeric(frame[args.probability_column], errors="coerce").fillna(0.0)

    thresholds = _load_thresholds(threshold_summary_path)
    policies = _build_policy_specs(thresholds)

    signal_frame = frame[["sample_time", "t90", "is_in_spec", "is_out_of_spec", args.probability_column]].copy()
    signal_frame = signal_frame.rename(columns={args.probability_column: "base_probability"})

    summary_rows: list[dict[str, object]] = []
    detailed_summary: dict[str, object] = {
        "target_definition": "1 when T90 is outside 8.45 +/- 0.25, otherwise 0",
        "base_probability_column": args.probability_column,
        "samples": int(len(frame)),
        "thresholds": thresholds,
        "policies": {},
    }

    for policy in policies:
        predicted = _apply_policy(score, policy)
        signal_frame[f"signal_{policy.name}"] = predicted.values
        metrics = _summarize_policy(actual, predicted)
        policy_payload = {
            "policy_kind": policy.kind,
            "threshold": float(policy.threshold),
            "off_threshold": float(policy.off_threshold) if policy.off_threshold is not None else None,
            "count_k": int(policy.count_k) if policy.count_k is not None else None,
            "count_n": int(policy.count_n) if policy.count_n is not None else None,
            "alpha": float(policy.alpha) if policy.alpha is not None else None,
        }
        policy_payload.update(metrics)
        detailed_summary["policies"][policy.name] = policy_payload
        summary_rows.append({"policy_name": policy.name, **policy_payload})

    recommendation = _choose_recommendation(summary_rows)
    detailed_summary["recommended_policy"] = recommendation
    detailed_summary["artifacts"] = {
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "plot_png": str(plot_png_path),
        "signal_csv": str(signal_csv_path),
    }

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["bad_segment_recall", "precision", "specificity"], ascending=[False, False, False]
    )
    summary_frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    signal_frame.to_csv(signal_csv_path, index=False, encoding="utf-8-sig")
    _plot_policy_metrics(summary_rows, plot_png_path)
    summary_json_path.write_text(json.dumps(detailed_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(detailed_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
