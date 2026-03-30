from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

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
DEFAULT_CURRENT_HEAD_SUMMARY = V3_ROOT / "dev" / "artifacts" / "phase1_three_class_feature_research_summary.json"
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


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def encode_three_class_target(frame: pd.DataFrame) -> np.ndarray:
    target = np.full(len(frame), -1, dtype=int)
    target[frame["is_below_spec"].astype(int).to_numpy() == 1] = 0
    target[frame["is_in_spec"].astype(int).to_numpy() == 1] = 1
    target[frame["is_above_spec"].astype(int).to_numpy() == 1] = 2
    return target


def make_model(model_name: str) -> Pipeline:
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
    raise ValueError(f"Unsupported model: {model_name}")


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def build_short_baseline_features(
    labeled: pd.DataFrame,
    dcs: pd.DataFrame,
    sensors: list[str],
    window_minutes: int,
    stats: list[str],
    min_points_per_window: int,
) -> pd.DataFrame:
    base = build_dcs_feature_table(labeled, dcs, window_minutes=window_minutes, min_points_per_window=min_points_per_window)
    keep_columns = [
        "sample_time",
        "t90",
        "is_in_spec",
        "is_out_of_spec",
        "is_above_spec",
        "is_below_spec",
    ]
    sensor_set = set(sensors)
    stat_set = set(stats)
    for column in base.columns:
        if "__" not in column:
            continue
        sensor_name, stat_name = column.split("__", 1)
        if sensor_name in sensor_set and stat_name in stat_set:
            keep_columns.append(column)
    result = base[keep_columns].copy()
    result = result.rename(columns={"sample_time": "decision_time"})
    return result.sort_values("decision_time").reset_index(drop=True)


def compute_ewma_window_features(
    labeled: pd.DataFrame,
    dcs: pd.DataFrame,
    sensors: list[str],
    window_minutes: int,
    ewma_lambda: float,
    min_points_per_window: int,
) -> pd.DataFrame:
    sample_frame = labeled.copy()
    sample_frame["decision_time"] = pd.to_datetime(sample_frame["sample_time"], errors="coerce")
    sample_frame = sample_frame.dropna(subset=["decision_time"]).sort_values("decision_time").reset_index(drop=True)

    dcs_frame = dcs.copy()
    dcs_frame["time"] = pd.to_datetime(dcs_frame["time"], errors="coerce")
    keep_columns = ["time", *[sensor for sensor in sensors if sensor in dcs_frame.columns]]
    dcs_frame = dcs_frame[keep_columns].dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")

    rows: list[dict[str, Any]] = []
    for record in sample_frame.itertuples(index=False):
        decision_time = pd.Timestamp(record.decision_time)
        left_boundary = (decision_time - pd.Timedelta(minutes=window_minutes)).to_datetime64()
        right_boundary = decision_time.to_datetime64()
        left = np.searchsorted(dcs_times, left_boundary, side="left")
        right = np.searchsorted(dcs_times, right_boundary, side="right")
        window = dcs_frame.iloc[left:right]
        if len(window) < min_points_per_window:
            continue

        row = {
            "decision_time": decision_time,
            "t90": getattr(record, "t90", np.nan),
            "is_in_spec": getattr(record, "is_in_spec", False),
            "is_out_of_spec": getattr(record, "is_out_of_spec", False),
            "is_above_spec": getattr(record, "is_above_spec", False),
            "is_below_spec": getattr(record, "is_below_spec", False),
            "ewma_window_minutes": int(window_minutes),
            "ewma_lambda": float(ewma_lambda),
            "rows_in_window": int(len(window)),
        }

        time_values = pd.to_datetime(window["time"], errors="coerce")
        delta_minutes = ((decision_time - time_values).dt.total_seconds() / 60.0).to_numpy(dtype=float)
        weights = np.power(float(ewma_lambda), delta_minutes)

        for sensor in sensors:
            series = pd.to_numeric(window[sensor], errors="coerce")
            valid_mask = series.notna().to_numpy()
            if not valid_mask.any():
                continue
            valid_values = series[valid_mask].to_numpy(dtype=float)
            valid_weights = weights[valid_mask]
            weight_sum = float(valid_weights.sum())
            if weight_sum <= 0:
                continue

            ewma_value = float(np.sum(valid_weights * valid_values) / weight_sum)
            last_value = float(valid_values[-1])
            variance = float(np.sum(valid_weights * (valid_values - ewma_value) ** 2) / weight_sum)
            row[f"{sensor}__ewma"] = ewma_value
            row[f"{sensor}__ewm_std"] = float(np.sqrt(max(variance, 0.0)))
            row[f"{sensor}__last"] = last_value
            row[f"{sensor}__last_minus_ewma"] = float(last_value - ewma_value)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)


def collect_oof_multiclass(X: pd.DataFrame, y: np.ndarray, model_name: str) -> pd.DataFrame:
    splitter = TimeSeriesSplit(n_splits=5)
    probabilities = np.full((len(X), len(CLASS_ORDER)), np.nan, dtype=float)

    for train_idx, test_idx in splitter.split(X):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 3:
            continue
        model = make_model(model_name)
        model.fit(X.iloc[train_idx], y_train)
        probabilities[test_idx, :] = model.predict_proba(X.iloc[test_idx])

    scored_mask = ~np.isnan(probabilities).any(axis=1)
    if scored_mask.sum() == 0:
        return pd.DataFrame()
    result = pd.DataFrame(probabilities[scored_mask], columns=[f"prob_{name}" for name in CLASS_ORDER])
    result.insert(0, "y_true", y[scored_mask])
    result["pred_class"] = np.argmax(probabilities[scored_mask], axis=1)
    return result.reset_index(drop=True)


def score_multiclass_output(oof_frame: pd.DataFrame) -> dict[str, Any]:
    y_true = oof_frame["y_true"].to_numpy(dtype=int)
    prob = oof_frame[[f"prob_{name}" for name in CLASS_ORDER]].to_numpy(dtype=float)
    pred = oof_frame["pred_class"].to_numpy(dtype=int)
    return {
        "samples": int(len(oof_frame)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "multiclass_log_loss": float(log_loss(y_true, prob, labels=[0, 1, 2])),
        "multiclass_brier_score": multiclass_brier_score(y_true, prob, n_classes=3),
    }


def evaluate_representation(frame: pd.DataFrame, feature_columns: list[str], model_name: str) -> dict[str, Any]:
    work = frame.sort_values("decision_time").reset_index(drop=True).copy()
    y = encode_three_class_target(work)
    valid_mask = y >= 0
    work = work.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    if len(np.unique(y)) < 3:
        return {"status": "missing_classes", "samples": int(len(work))}

    X = work[feature_columns]
    oof = collect_oof_multiclass(X, y, model_name=model_name)
    if oof.empty:
        return {"status": "no_scored_samples", "samples": int(len(work))}
    metrics = score_multiclass_output(oof)
    return {"status": "ok", **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Current-head EWMA distillation experiment under the locked v1.2 plan.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--current-head-summary", type=Path, default=DEFAULT_CURRENT_HEAD_SUMMARY)
    parser.add_argument("--ewma-windows", type=str, default="20,60,120")
    parser.add_argument("--ewma-lambdas", type=str, default="0.75,0.85,0.92")
    parser.add_argument("--models", type=str, default="random_forest_balanced,multinomial_logistic")
    parser.add_argument("--output-prefix", type=str, default="phase1_current_head_ewma_distillation")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    current_head_summary = json.loads(args.current_head_summary.read_text(encoding="utf-8"))
    setup = current_head_summary["recommended_current_three_class_setup"]
    sensors = list(setup["sensor_names"])
    baseline_window = int(setup["window_minutes"])
    baseline_stats = str(setup["stats"]).split(",")
    ewma_windows = parse_int_list(args.ewma_windows)
    ewma_lambdas = parse_float_list(args.ewma_lambdas)
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

    baseline_frame = build_short_baseline_features(
        labeled=labeled,
        dcs=dcs,
        sensors=sensors,
        window_minutes=baseline_window,
        stats=baseline_stats,
        min_points_per_window=int(config["window_search"]["min_points_per_window"]),
    )
    baseline_feature_columns = [column for column in baseline_frame.columns if "__" in column]

    rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    ewma_export_rows: list[pd.DataFrame] = []

    for model_name in model_names:
        baseline_metrics = evaluate_representation(baseline_frame, baseline_feature_columns, model_name=model_name)
        baseline_metrics.update(
            {
                "representation_name": "baseline_short_window",
                "model_name": model_name,
                "feature_family": "baseline_short_window",
                "feature_count": int(len(baseline_feature_columns)),
                "ewma_window_minutes": None,
                "ewma_lambda": None,
            }
        )
        rows.append(baseline_metrics)

    for ewma_window in ewma_windows:
        for ewma_lambda in ewma_lambdas:
            ewma_frame = compute_ewma_window_features(
                labeled=labeled,
                dcs=dcs,
                sensors=sensors,
                window_minutes=int(ewma_window),
                ewma_lambda=float(ewma_lambda),
                min_points_per_window=int(config["window_search"]["min_points_per_window"]),
            )
            if ewma_frame.empty:
                continue
            ewma_feature_columns = [column for column in ewma_frame.columns if "__" in column]
            ewma_export = ewma_frame.copy()
            ewma_export_rows.append(ewma_export)

            family_definitions = {
                "ewma_core": [column for column in ewma_feature_columns if column.endswith("__ewma")],
                "ewma_plus_deviation": [
                    column
                    for column in ewma_feature_columns
                    if column.endswith("__ewma") or column.endswith("__ewm_std") or column.endswith("__last_minus_ewma")
                ],
            }

            merged_hybrid = baseline_frame.merge(
                ewma_frame[["decision_time", *ewma_feature_columns]],
                on="decision_time",
                how="inner",
            )
            family_definitions["hybrid_short_plus_ewma"] = [
                *[column for column in merged_hybrid.columns if column in baseline_feature_columns],
                *[
                    column
                    for column in merged_hybrid.columns
                    if column.endswith("__ewma") or column.endswith("__ewm_std") or column.endswith("__last_minus_ewma")
                ],
            ]

            family_frames = {
                "ewma_core": ewma_frame,
                "ewma_plus_deviation": ewma_frame,
                "hybrid_short_plus_ewma": merged_hybrid,
            }

            for family_name, feature_columns in family_definitions.items():
                for model_name in model_names:
                    metrics = evaluate_representation(family_frames[family_name], feature_columns, model_name=model_name)
                    metrics.update(
                        {
                            "representation_name": f"{family_name}_w{ewma_window}_l{ewma_lambda}",
                            "model_name": model_name,
                            "feature_family": family_name,
                            "feature_count": int(len(feature_columns)),
                            "ewma_window_minutes": int(ewma_window),
                            "ewma_lambda": float(ewma_lambda),
                        }
                    )
                    rows.append(metrics)

    result = pd.DataFrame(rows)
    ok_result = result[result["status"] == "ok"].copy()
    ok_result = ok_result.sort_values(
        ["macro_f1", "balanced_accuracy", "multiclass_log_loss", "multiclass_brier_score", "feature_count"],
        ascending=[False, False, True, True, True],
    )
    best_row = ok_result.iloc[0].to_dict() if not ok_result.empty else None

    summary = {
        "baseline_setup": setup,
        "ewma_design": {
            "weight_definition": "w(delta_minutes) = lambda ** delta_minutes",
            "core_outputs": ["ewma", "ewm_std", "last", "last_minus_ewma"],
            "note": "This stage evaluates whether a longer weighted causal window can preserve more operating-state information for the current three-class head.",
        },
        "rows": ok_result.to_dict(orient="records"),
        "recommended_representation": best_row,
    }

    result.to_csv(artifact_dir / f"{args.output_prefix}_summary.csv", index=False, encoding="utf-8-sig")
    if ewma_export_rows:
        pd.concat(ewma_export_rows, ignore_index=True).to_csv(
            artifact_dir / f"{args.output_prefix}_ewma_feature_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )
    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
