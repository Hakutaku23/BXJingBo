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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
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
CLASS_ORDER = ["below_spec", "in_spec", "above_spec"]


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


def encode_three_class_target(frame: pd.DataFrame) -> np.ndarray:
    target = np.full(len(frame), -1, dtype=int)
    below_mask = frame["is_below_spec"].astype(int).to_numpy() == 1
    in_mask = frame["is_in_spec"].astype(int).to_numpy() == 1
    above_mask = frame["is_above_spec"].astype(int).to_numpy() == 1
    target[below_mask] = 0
    target[in_mask] = 1
    target[above_mask] = 2
    return target


def make_model(model_name: str) -> Pipeline:
    if model_name == "multinomial_logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1200,
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
                        n_estimators=350,
                        max_depth=7,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def collect_oof_probabilities(X: pd.DataFrame, y: np.ndarray, model_name: str) -> pd.DataFrame:
    splitter = TimeSeriesSplit(n_splits=5)
    probabilities = np.full((len(X), len(CLASS_ORDER)), np.nan, dtype=float)

    for train_idx, test_idx in splitter.split(X):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 3:
            continue
        model = make_model(model_name)
        model.fit(X.iloc[train_idx], y_train)
        fold_prob = model.predict_proba(X.iloc[test_idx])
        probabilities[test_idx, :] = fold_prob

    scored_mask = ~np.isnan(probabilities).any(axis=1)
    result = pd.DataFrame(probabilities[scored_mask], columns=[f"prob_{name}" for name in CLASS_ORDER])
    result.insert(0, "y_true", y[scored_mask])
    result["pred_class"] = np.argmax(probabilities[scored_mask], axis=1)
    return result


def score_multiclass_output(oof_frame: pd.DataFrame) -> dict[str, Any]:
    y_true = oof_frame["y_true"].to_numpy(dtype=int)
    prob = oof_frame[[f"prob_{name}" for name in CLASS_ORDER]].to_numpy(dtype=float)
    pred = oof_frame["pred_class"].to_numpy(dtype=int)

    class_rows = []
    for class_idx, class_name in enumerate(CLASS_ORDER):
        true_mask = y_true == class_idx
        pred_mask = pred == class_idx
        tp = int(np.sum(true_mask & pred_mask))
        fn = int(np.sum(true_mask & (~pred_mask)))
        fp = int(np.sum((~true_mask) & pred_mask))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        class_rows.append(
            {
                "class_name": class_name,
                "support": int(true_mask.sum()),
                "precision": float(precision),
                "recall": float(recall),
                "mean_predicted_probability": float(prob[:, class_idx].mean()),
            }
        )

    confusion_rows = []
    for actual_idx, actual_name in enumerate(CLASS_ORDER):
        for pred_idx, pred_name in enumerate(CLASS_ORDER):
            confusion_rows.append(
                {
                    "actual_class": actual_name,
                    "predicted_class": pred_name,
                    "count": int(np.sum((y_true == actual_idx) & (pred == pred_idx))),
                }
            )

    return {
        "samples": int(len(oof_frame)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "multiclass_log_loss": float(log_loss(y_true, prob, labels=[0, 1, 2])),
        "multiclass_brier_score": multiclass_brier_score(y_true, prob, n_classes=3),
        "class_rows": class_rows,
        "confusion_rows": confusion_rows,
    }


def draw_probability_histograms(oof_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    y_true = oof_frame["y_true"].to_numpy(dtype=int)
    for idx, class_name in enumerate(CLASS_ORDER):
        ax = axes[idx]
        prob = oof_frame[f"prob_{class_name}"].to_numpy(dtype=float)
        ax.hist(prob[y_true == idx], bins=20, alpha=0.75, label=f"true {class_name}", color="#2E6F95")
        ax.hist(prob[y_true != idx], bins=20, alpha=0.45, label=f"other classes", color="#C84C31")
        ax.set_title(class_name)
        ax.set_xlabel("predicted probability")
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.set_ylabel("count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate three-class probability output for below-spec / in-spec / above-spec.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--benchmark-summary", type=Path, default=DEFAULT_BENCHMARK_SUMMARY)
    parser.add_argument("--models", type=str, default="multinomial_logistic,random_forest_balanced")
    parser.add_argument("--output-prefix", type=str, default="phase1_three_class_probability")
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
    y = encode_three_class_target(work)
    valid_mask = y >= 0
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    work = work.loc[valid_mask].reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    model_details: dict[str, Any] = {}
    probability_exports: list[pd.DataFrame] = []

    for model_name in model_names:
        oof_frame = collect_oof_probabilities(X, y, model_name=model_name)
        if oof_frame.empty:
            continue
        metrics = score_multiclass_output(oof_frame)
        summary_rows.append(
            {
                "model_name": model_name,
                "samples": metrics["samples"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "multiclass_log_loss": metrics["multiclass_log_loss"],
                "multiclass_brier_score": metrics["multiclass_brier_score"],
            }
        )
        model_details[model_name] = {
            "class_rows": metrics["class_rows"],
            "confusion_rows": metrics["confusion_rows"],
        }

        export = oof_frame.copy()
        export.insert(0, "model_name", model_name)
        export["true_class_name"] = export["y_true"].map({0: "below_spec", 1: "in_spec", 2: "above_spec"})
        export["pred_class_name"] = export["pred_class"].map({0: "below_spec", 1: "in_spec", 2: "above_spec"})
        probability_exports.append(export)
        draw_probability_histograms(export, artifact_dir / f"{args.output_prefix}_{model_name}_hist.png")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1", "balanced_accuracy", "multiclass_log_loss", "multiclass_brier_score", "model_name"],
        ascending=[False, False, True, True, True],
    )
    best_row = summary_df.iloc[0].to_dict() if not summary_df.empty else None

    summary = {
        "class_order": CLASS_ORDER,
        "feature_setup": alarm_setup,
        "rows": summary_df.to_dict(orient="records"),
        "model_details": model_details,
        "recommended_three_class_model": best_row,
        "notes": {
            "goal": "Check whether a direct three-class probability output is more suitable than forcing the boundary problem into a binary decision.",
            "interpretation": "The key outputs are class probabilities for below-spec, in-spec, and above-spec, then downstream warning logic can aggregate them.",
        },
    }

    summary_df.to_csv(artifact_dir / f"{args.output_prefix}_summary.csv", index=False, encoding="utf-8-sig")
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
