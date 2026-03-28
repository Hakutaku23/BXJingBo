from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(__file__).resolve().parent
V3_ROOT = EXPERIMENT_DIR.parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import TargetSpec, add_out_of_spec_labels, build_dcs_feature_table, load_dcs_frame, load_lims_samples


DEFAULT_CONFIG_PATH = V3_ROOT / "config" / "phase1_warning.yaml"
DEFAULT_BENCHMARK_SUMMARY = V3_ROOT / "dev" / "artifacts" / "phase1_dual_head_model_family_benchmark_summary.json"


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


def build_selected_features(frame: pd.DataFrame, sensors: list[str], stats: list[str]) -> list[str]:
    sensor_set = set(sensors)
    stat_set = set(stats)
    selected = []
    for column in frame.columns:
        if "__" not in column:
            continue
        sensor, stat = column.split("__", 1)
        if sensor in sensor_set and stat in stat_set:
            selected.append(column)
    return selected


def make_model(model_name: str) -> Pipeline:
    if model_name == "logistic_balanced":
        return Pipeline(
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
    if model_name == "random_forest_balanced":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def fit_sigmoid_calibrator(calibration_prob: np.ndarray, calibration_y: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(calibration_y)) < 2:
        return None
    model = LogisticRegression(solver="lbfgs")
    model.fit(calibration_prob.reshape(-1, 1), calibration_y)
    return model


def split_calibration_indices(train_idx: np.ndarray, calibration_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    if len(train_idx) < 20:
        return train_idx, np.array([], dtype=int)
    calibration_size = max(int(round(len(train_idx) * calibration_fraction)), 20)
    calibration_size = min(calibration_size, len(train_idx) - 10)
    if calibration_size <= 0:
        return train_idx, np.array([], dtype=int)
    return train_idx[:-calibration_size], train_idx[-calibration_size:]


def collect_oof_probabilities(X: pd.DataFrame, y: np.ndarray, model_name: str) -> pd.DataFrame:
    splitter = TimeSeriesSplit(n_splits=5)
    raw_prob = np.full(len(X), np.nan, dtype=float)
    sigmoid_prob = np.full(len(X), np.nan, dtype=float)
    isotonic_prob = np.full(len(X), np.nan, dtype=float)

    for train_idx, test_idx in splitter.split(X):
        subtrain_idx, calibration_idx = split_calibration_indices(train_idx)
        if len(subtrain_idx) == 0:
            continue

        y_subtrain = y[subtrain_idx]
        if len(np.unique(y_subtrain)) < 2:
            continue

        model = make_model(model_name)
        model.fit(X.iloc[subtrain_idx], y_subtrain)

        raw_test = model.predict_proba(X.iloc[test_idx])[:, 1]
        raw_prob[test_idx] = raw_test

        if len(calibration_idx) == 0:
            sigmoid_prob[test_idx] = raw_test
            isotonic_prob[test_idx] = raw_test
            continue

        calibration_raw = model.predict_proba(X.iloc[calibration_idx])[:, 1]
        calibration_y = y[calibration_idx]

        sigmoid_model = fit_sigmoid_calibrator(calibration_raw, calibration_y)
        if sigmoid_model is None:
            sigmoid_prob[test_idx] = raw_test
        else:
            sigmoid_prob[test_idx] = sigmoid_model.predict_proba(raw_test.reshape(-1, 1))[:, 1]

        if len(np.unique(calibration_y)) < 2:
            isotonic_prob[test_idx] = raw_test
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(calibration_raw, calibration_y)
            isotonic_prob[test_idx] = iso.transform(raw_test)

    scored_mask = ~np.isnan(raw_prob)
    result = pd.DataFrame(
        {
            "y_true": y[scored_mask],
            "raw_probability": raw_prob[scored_mask],
            "sigmoid_probability": sigmoid_prob[scored_mask],
            "isotonic_probability": isotonic_prob[scored_mask],
        }
    )
    return result


def score_probability_vector(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    prob = np.clip(prob.astype(float), 1e-6, 1.0 - 1e-6)
    thresholds = np.arange(0.05, 1.00, 0.05)
    best = None
    for threshold in thresholds:
        pred = prob >= threshold
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        row = {
            "threshold": float(round(threshold, 2)),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
        }
        if best is None or row["f1"] > best["f1"]:
            best = row

    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob)),
        "best_threshold": float(best["threshold"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_specificity": float(best["specificity"]),
        "best_f1": float(best["f1"]),
    }


def calibration_curve_rows(y_true: np.ndarray, prob: np.ndarray, bin_count: int = 10) -> list[dict[str, float]]:
    bins = np.linspace(0.0, 1.0, bin_count + 1)
    rows = []
    for idx in range(bin_count):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == bin_count - 1:
            mask = (prob >= left) & (prob <= right)
        else:
            mask = (prob >= left) & (prob < right)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "mean_probability": float(prob[mask].mean()),
                "observed_positive_rate": float(y_true[mask].mean()),
                "sample_count": int(mask.sum()),
            }
        )
    return rows


def draw_calibration_plot(rows_by_variant: dict[str, list[dict[str, float]]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="ideal")
    for variant_name, rows in rows_by_variant.items():
        if not rows:
            continue
        x = [row["mean_probability"] for row in rows]
        y = [row["observed_positive_rate"] for row in rows]
        ax.plot(x, y, marker="o", linewidth=1.6, label=variant_name)
    ax.set_title("Phase1 Current Alarm Probability Calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed out-of-spec rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether Phase 1 current-alarm output should move from hard labels to calibrated probabilities.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--benchmark-summary", type=Path, default=DEFAULT_BENCHMARK_SUMMARY)
    parser.add_argument("--models", type=str, default="logistic_balanced,random_forest_balanced")
    parser.add_argument("--output-prefix", type=str, default="phase1_alarm_probability_calibration")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = json.loads(args.benchmark_summary.read_text(encoding="utf-8"))
    alarm_setup = benchmark_summary["alarm_feature_setup"]
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]

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
        window_minutes=int(alarm_setup["window_minutes"]),
        min_points_per_window=int(config["window_search"]["min_points_per_window"]),
    )
    selected_features = build_selected_features(feature_table, list(alarm_setup["sensor_names"]), list(alarm_setup["stats"]))
    work = feature_table.sort_values("sample_time").reset_index(drop=True).copy()
    X = work[selected_features]
    y = work["is_out_of_spec"].astype(int).to_numpy()

    summary_rows: list[dict[str, Any]] = []
    calibration_rows_all: list[dict[str, Any]] = []
    rows_by_variant: dict[str, list[dict[str, float]]] = {}
    probability_exports: list[pd.DataFrame] = []

    for model_name in model_names:
        oof_frame = collect_oof_probabilities(X, y, model_name=model_name)
        if oof_frame.empty:
            continue

        y_true = oof_frame["y_true"].astype(int).to_numpy()
        for variant_name, column_name in [
            ("raw", "raw_probability"),
            ("sigmoid", "sigmoid_probability"),
            ("isotonic", "isotonic_probability"),
        ]:
            prob = oof_frame[column_name].to_numpy(dtype=float)
            metrics = score_probability_vector(y_true, prob)
            row = {
                "model_name": model_name,
                "probability_variant": variant_name,
                **metrics,
            }
            summary_rows.append(row)

            curve_rows = calibration_curve_rows(y_true, prob)
            rows_by_variant[f"{model_name}:{variant_name}"] = curve_rows
            for curve_row in curve_rows:
                calibration_rows_all.append(
                    {
                        "model_name": model_name,
                        "probability_variant": variant_name,
                        **curve_row,
                    }
                )

        export = oof_frame.copy()
        export.insert(0, "model_name", model_name)
        probability_exports.append(export)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["brier_score", "log_loss", "roc_auc", "average_precision", "model_name", "probability_variant"],
        ascending=[True, True, False, False, True, True],
    )
    calibration_df = pd.DataFrame(calibration_rows_all)
    if not summary_df.empty:
        best_row = summary_df.iloc[0].to_dict()
    else:
        best_row = None

    plot_path = artifact_dir / f"{args.output_prefix}_calibration_curve.png"
    draw_calibration_plot(rows_by_variant, plot_path)

    summary = {
        "alarm_feature_setup": alarm_setup,
        "rows": summary_df.to_dict(orient="records"),
        "recommended_probability_output": best_row,
        "notes": {
            "goal": "Evaluate whether the current-alarm head should expose calibrated out-of-spec probability instead of only hard classification.",
            "interpretation": "Probability quality is judged by Brier score, log loss, ROC AUC, and average precision together.",
        },
    }

    summary_df.to_csv(artifact_dir / f"{args.output_prefix}_summary.csv", index=False, encoding="utf-8-sig")
    calibration_df.to_csv(artifact_dir / f"{args.output_prefix}_calibration_rows.csv", index=False, encoding="utf-8-sig")
    if probability_exports:
        pd.concat(probability_exports, ignore_index=True).to_csv(
            artifact_dir / f"{args.output_prefix}_oof_probabilities.csv",
            index=False,
            encoding="utf-8-sig",
        )
    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
