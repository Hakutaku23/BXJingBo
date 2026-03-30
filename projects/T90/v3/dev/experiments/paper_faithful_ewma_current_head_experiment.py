from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(__file__).resolve().parent
V3_ROOT = EXPERIMENT_DIR.parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import TargetSpec, add_out_of_spec_labels, load_dcs_frame, load_lims_samples


DEFAULT_CONFIG_PATH = V3_ROOT / "config" / "phase1_warning.yaml"
RESULT_CSV = "paper_faithful_ewma_current_head_results.csv"
SUMMARY_JSON = "paper_faithful_ewma_current_head_summary.json"
FEATURE_ROWS_CSV = "paper_faithful_ewma_current_head_feature_rows.csv"
AUDIT_MD = "paper_faithful_ewma_current_head_audit.md"
CLASS_ORDER = ["below_spec", "in_spec", "above_spec"]


@dataclass
class FeatureConfig:
    baseline_window_minutes: int = 8
    baseline_stats: tuple[str, ...] = ("mean", "min", "last", "max")
    tau_grid: tuple[int, ...] = (0, 120, 240, 360, 480)
    window_grid: tuple[int, ...] = (60, 120, 240, 360, 480)
    lambda_grid: tuple[float, ...] = (0.97, 0.985, 0.995)
    topk_grid: tuple[int, ...] = (20, 40, 80)
    n_splits: int = 5
    min_valid_points_per_sensor: int = 3
    min_rows_per_window: int = 5
    screening_stat_for_sensor: str = "mean"
    supervised_screening_enabled: bool = True


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


def encode_three_class_target(frame: pd.DataFrame) -> np.ndarray:
    target = np.full(len(frame), -1, dtype=int)
    target[frame["is_below_spec"].astype(int).to_numpy() == 1] = 0
    target[frame["is_in_spec"].astype(int).to_numpy() == 1] = 1
    target[frame["is_above_spec"].astype(int).to_numpy() == 1] = 2
    return target


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def aggregate_fold_scores(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    pred = np.argmax(probabilities, axis=1)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "multiclass_log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "multiclass_brier_score": multiclass_brier_score(y_true, probabilities, n_classes=3),
    }


def score_single_feature_multiclass(x: pd.Series, y: np.ndarray) -> float:
    valid = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": y}).dropna()
    if valid.empty or len(np.unique(valid["x"])) < 2 or len(np.unique(valid["y"])) < 3:
        return 0.0
    y_values = valid["y"].to_numpy(dtype=int)
    x_values = valid["x"].to_numpy(dtype=float)
    auc_values: list[float] = []
    for class_id in [0, 1, 2]:
        one_vs_rest = (y_values == class_id).astype(int)
        if one_vs_rest.min() == one_vs_rest.max():
            continue
        try:
            auc_raw = float(roc_auc_score(one_vs_rest, x_values))
        except ValueError:
            continue
        auc_values.append(abs(auc_raw - 0.5) * 2.0)
    return float(np.mean(auc_values)) if auc_values else 0.0


def summarize_numeric_window(window: pd.DataFrame, sensor_columns: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in sensor_columns:
        series = pd.to_numeric(window[column], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            result[f"{column}__last"] = np.nan
            result[f"{column}__mean"] = np.nan
            result[f"{column}__min"] = np.nan
            result[f"{column}__max"] = np.nan
            continue
        result[f"{column}__last"] = float(valid.iloc[-1])
        result[f"{column}__mean"] = float(valid.mean())
        result[f"{column}__min"] = float(valid.min())
        result[f"{column}__max"] = float(valid.max())
    return result


def unsupervised_sensor_preclean(dcs: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    sensor_columns = [column for column in dcs.columns if column != "time"]
    kept: list[str] = []
    reasons: dict[str, str] = {}
    fingerprints: set[tuple[Any, ...]] = set()
    for column in sensor_columns:
        series = pd.to_numeric(dcs[column], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            reasons[column] = "all_missing"
            continue
        missing_ratio = float(series.isna().mean())
        if missing_ratio > 0.6:
            reasons[column] = "high_missingness"
            continue
        if int(non_null.nunique()) <= 5:
            reasons[column] = "too_few_unique_values"
            continue
        if float(non_null.std(ddof=0)) <= 1e-8:
            reasons[column] = "near_constant"
            continue
        sampled = tuple(np.round(non_null.iloc[:120].to_numpy(dtype=float), 8).tolist())
        if sampled in fingerprints:
            reasons[column] = "obvious_duplicate_signature"
            continue
        fingerprints.add(sampled)
        kept.append(column)
    audit = {
        "input_sensor_count": int(len(sensor_columns)),
        "kept_sensor_count": int(len(kept)),
        "dropped_sensor_count": int(len(sensor_columns) - len(kept)),
        "drop_reasons": reasons,
    }
    return kept, audit


def build_baseline_feature_table(
    labeled: pd.DataFrame,
    dcs: pd.DataFrame,
    sensors: list[str],
    window_minutes: int,
    min_rows_per_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_frame = labeled.copy()
    sample_frame["sample_time"] = pd.to_datetime(sample_frame["sample_time"], errors="coerce")
    sample_frame = sample_frame.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)

    dcs_frame = dcs[["time", *sensors]].copy()
    dcs_frame["time"] = pd.to_datetime(dcs_frame["time"], errors="coerce")
    dcs_frame = dcs_frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")

    rows: list[dict[str, Any]] = []
    dropped_for_short_window = 0
    for record in sample_frame.itertuples(index=False):
        decision_time = pd.Timestamp(record.sample_time)
        left_boundary = (decision_time - pd.Timedelta(minutes=window_minutes)).to_datetime64()
        right_boundary = decision_time.to_datetime64()
        left = np.searchsorted(dcs_times, left_boundary, side="left")
        right = np.searchsorted(dcs_times, right_boundary, side="right")
        window = dcs_frame.iloc[left:right]
        if len(window) < min_rows_per_window:
            dropped_for_short_window += 1
            continue
        row = {
            "decision_time": decision_time,
            "t90": getattr(record, "t90", np.nan),
            "is_in_spec": getattr(record, "is_in_spec", False),
            "is_out_of_spec": getattr(record, "is_out_of_spec", False),
            "is_above_spec": getattr(record, "is_above_spec", False),
            "is_below_spec": getattr(record, "is_below_spec", False),
            "crosses_regime_boundary": False,
        }
        row.update(summarize_numeric_window(window, sensors))
        rows.append(row)
    audit = {
        "baseline_window_minutes": int(window_minutes),
        "dropped_due_to_insufficient_window_rows": int(dropped_for_short_window),
    }
    return pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True), audit


def compute_recursive_ewma_features(
    labeled: pd.DataFrame,
    dcs: pd.DataFrame,
    sensors: list[str],
    tau_minutes: int,
    window_minutes: int,
    lambda_value: float,
    min_rows_per_window: int,
    min_valid_points_per_sensor: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_frame = labeled.copy()
    sample_frame["sample_time"] = pd.to_datetime(sample_frame["sample_time"], errors="coerce")
    sample_frame = sample_frame.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)

    dcs_frame = dcs[["time", *sensors]].copy()
    dcs_frame["time"] = pd.to_datetime(dcs_frame["time"], errors="coerce")
    dcs_frame = dcs_frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")

    rows: list[dict[str, Any]] = []
    dropped_for_short_window = 0
    insufficient_sensor_counter = 0
    total_sensor_windows = 0
    for record in sample_frame.itertuples(index=False):
        decision_time = pd.Timestamp(record.sample_time)
        t_end = decision_time - pd.Timedelta(minutes=tau_minutes)
        t_start = t_end - pd.Timedelta(minutes=window_minutes)
        left = np.searchsorted(dcs_times, t_start.to_datetime64(), side="left")
        right = np.searchsorted(dcs_times, t_end.to_datetime64(), side="right")
        window = dcs_frame.iloc[left:right]
        if len(window) < min_rows_per_window:
            dropped_for_short_window += 1
            continue

        row = {
            "decision_time": decision_time,
            "tau_minutes": int(tau_minutes),
            "window_minutes": int(window_minutes),
            "lambda": float(lambda_value),
            "t90": getattr(record, "t90", np.nan),
            "is_in_spec": getattr(record, "is_in_spec", False),
            "is_out_of_spec": getattr(record, "is_out_of_spec", False),
            "is_above_spec": getattr(record, "is_above_spec", False),
            "is_below_spec": getattr(record, "is_below_spec", False),
            "crosses_regime_boundary": False,
        }

        for sensor in sensors:
            total_sensor_windows += 1
            series = pd.to_numeric(window[sensor], errors="coerce").dropna()
            if len(series) < min_valid_points_per_sensor:
                insufficient_sensor_counter += 1
                row[f"{sensor}__ewma_recursive"] = np.nan
                row[f"{sensor}__last"] = np.nan
                row[f"{sensor}__last_minus_ewma_recursive"] = np.nan
                row[f"{sensor}__ewm_std_recursive"] = np.nan
                continue

            ewma_series = series.ewm(alpha=1.0 - lambda_value, adjust=False).mean()
            ewma_value = float(ewma_series.iloc[-1])
            last_value = float(series.iloc[-1])
            ewm_std_series = series.ewm(alpha=1.0 - lambda_value, adjust=False).std(bias=False)
            ewm_std_value = float(ewm_std_series.iloc[-1]) if not pd.isna(ewm_std_series.iloc[-1]) else np.nan

            row[f"{sensor}__ewma_recursive"] = ewma_value
            row[f"{sensor}__last"] = last_value
            row[f"{sensor}__last_minus_ewma_recursive"] = float(last_value - ewma_value)
            row[f"{sensor}__ewm_std_recursive"] = ewm_std_value
        rows.append(row)

    audit = {
        "tau_minutes": int(tau_minutes),
        "window_minutes": int(window_minutes),
        "lambda": float(lambda_value),
        "dropped_due_to_insufficient_window_rows": int(dropped_for_short_window),
        "sensor_windows_with_too_few_valid_points": int(insufficient_sensor_counter),
        "total_sensor_windows": int(total_sensor_windows),
    }
    return pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True), audit


def rank_sensors_on_training_fold(
    baseline_train: pd.DataFrame,
    y_train: np.ndarray,
    candidate_sensors: list[str],
    scoring_stat: str,
) -> list[str]:
    sensor_rows: list[dict[str, Any]] = []
    for sensor in candidate_sensors:
        column = f"{sensor}__{scoring_stat}"
        if column not in baseline_train.columns:
            continue
        score = score_single_feature_multiclass(baseline_train[column], y_train)
        sensor_rows.append({"sensor_name": sensor, "screen_score": score})
    ranking = pd.DataFrame(sensor_rows).sort_values(["screen_score", "sensor_name"], ascending=[False, True])
    return ranking["sensor_name"].tolist()


def evaluate_fold_predictions(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_columns: list[str],
    model_name: str,
) -> tuple[np.ndarray, dict[str, float]]:
    train_matrix = train_frame[feature_columns].copy()
    test_matrix = test_frame[feature_columns].copy()
    valid_feature_columns = [
        column for column in feature_columns
        if not train_matrix[column].isna().all() and not test_matrix[column].isna().all()
    ]
    if not valid_feature_columns:
        raise ValueError("No valid feature columns after fold-local missing-value guard.")
    model = make_model(model_name)
    model.fit(train_matrix[valid_feature_columns], y_train)
    probabilities = model.predict_proba(test_matrix[valid_feature_columns])
    return probabilities, aggregate_fold_scores(y_test, probabilities)


def run_baseline_and_treatment(
    baseline_table: pd.DataFrame,
    ewma_table: pd.DataFrame,
    candidate_sensors: list[str],
    topk_target: int,
    model_name: str,
    screening_stat: str,
    n_splits: int,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    baseline_work = baseline_table.sort_values("decision_time").reset_index(drop=True).copy()
    ewma_work = ewma_table.sort_values("decision_time").reset_index(drop=True).copy()
    merged = baseline_work.merge(
        ewma_work,
        on=["decision_time", "t90", "is_in_spec", "is_out_of_spec", "is_above_spec", "is_below_spec", "crosses_regime_boundary"],
        how="inner",
        suffixes=("_baseline", "_ewma"),
    )
    merged = merged.sort_values("decision_time").reset_index(drop=True)
    y = encode_three_class_target(merged)
    valid_mask = y >= 0
    merged = merged.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]

    splitter = TimeSeriesSplit(n_splits=n_splits)
    baseline_probabilities = np.full((len(merged), len(CLASS_ORDER)), np.nan, dtype=float)
    ewma_probabilities = np.full((len(merged), len(CLASS_ORDER)), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(merged), start=1):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 3 or len(np.unique(y_test)) < 2:
            continue

        train_frame = merged.iloc[train_idx].reset_index(drop=True)
        test_frame = merged.iloc[test_idx].reset_index(drop=True)
        train_baseline_view = train_frame.rename(
            columns={column: column.replace("_baseline", "") for column in train_frame.columns if column.endswith("_baseline")}
        )
        ranked_sensors = rank_sensors_on_training_fold(
            baseline_train=train_baseline_view,
            y_train=y_train,
            candidate_sensors=candidate_sensors,
            scoring_stat=screening_stat,
        )
        actual_topk = min(int(topk_target), len(ranked_sensors))
        selected_sensors = ranked_sensors[:actual_topk]
        baseline_features = [
            f"{sensor}__{stat}_baseline"
            for sensor in selected_sensors
            for stat in ("mean", "min", "last", "max")
            if f"{sensor}__{stat}_baseline" in merged.columns
        ]
        ewma_features = [
            f"{sensor}__{suffix}_ewma"
            for sensor in selected_sensors
            for suffix in ("ewma_recursive", "last", "last_minus_ewma_recursive", "ewm_std_recursive")
            if f"{sensor}__{suffix}_ewma" in merged.columns
        ]
        if not baseline_features or not ewma_features:
            continue

        baseline_fold_prob, baseline_fold_metrics = evaluate_fold_predictions(
            train_frame=train_frame,
            test_frame=test_frame,
            y_train=y_train,
            y_test=y_test,
            feature_columns=baseline_features,
            model_name=model_name,
        )
        ewma_fold_prob, ewma_fold_metrics = evaluate_fold_predictions(
            train_frame=train_frame,
            test_frame=test_frame,
            y_train=y_train,
            y_test=y_test,
            feature_columns=ewma_features,
            model_name=model_name,
        )
        baseline_probabilities[test_idx, :] = baseline_fold_prob
        ewma_probabilities[test_idx, :] = ewma_fold_prob

        fold_rows.append(
            {
                "fold": int(fold_idx),
                "topk_actual": int(actual_topk),
                "selected_sensors": selected_sensors,
                "baseline_macro_f1": baseline_fold_metrics["macro_f1"],
                "baseline_balanced_accuracy": baseline_fold_metrics["balanced_accuracy"],
                "ewma_macro_f1": ewma_fold_metrics["macro_f1"],
                "ewma_balanced_accuracy": ewma_fold_metrics["balanced_accuracy"],
            }
        )

    valid_baseline = ~np.isnan(baseline_probabilities).any(axis=1)
    valid_ewma = ~np.isnan(ewma_probabilities).any(axis=1)
    if valid_baseline.sum() == 0 or valid_ewma.sum() == 0:
        raise ValueError("No valid folds were scored.")

    baseline_metrics = aggregate_fold_scores(y[valid_baseline], baseline_probabilities[valid_baseline])
    ewma_metrics = aggregate_fold_scores(y[valid_ewma], ewma_probabilities[valid_ewma])
    fold_frame = pd.DataFrame(fold_rows)
    return (
        {
            **baseline_metrics,
            "samples": int(valid_baseline.sum()),
            "valid_fold_count": int(len(fold_rows)),
            "topk_actual_median": int(fold_frame["topk_actual"].median()) if not fold_frame.empty else 0,
        },
        {
            **ewma_metrics,
            "samples": int(valid_ewma.sum()),
            "valid_fold_count": int(len(fold_rows)),
            "topk_actual_median": int(fold_frame["topk_actual"].median()) if not fold_frame.empty else 0,
        },
        fold_frame,
    )


def write_audit_markdown(path: Path, summary: dict[str, Any], audit_state: dict[str, Any]) -> None:
    lines = [
        "# Paper-Faithful EWMA Current Head Audit",
        "",
        f"- Recursive EWMA strictly used: {'yes' if audit_state['recursive_ewma_strictly_used'] else 'no'}",
        f"- Explicit tau used: {'yes' if audit_state['explicit_tau_used'] else 'no'}",
        f"- Feature re-screening enabled: {'yes' if audit_state['supervised_rescreening_enabled'] else 'no'}",
        f"- Train-fold-only screening: {'yes' if audit_state['screening_train_fold_only'] else 'no'}",
        f"- Regime transitions identified: {'yes' if audit_state['regime_boundaries_identified'] else 'no'}",
        f"- Samples dropped due to insufficient windows: {audit_state['total_samples_dropped_due_to_short_windows']}",
        "",
        "## Screening Strategy",
        "",
        audit_state["screening_strategy_description"],
        "",
        "## Known Limitations",
        "",
        "- Regime boundaries are not reliably identified from the current data sources, so no no-crossing subset evaluation was available.",
        "- The experiment remains CPU-only and model-family-limited by design.",
        "",
        "## Best EWMA Combination",
        "",
        json.dumps(summary.get("best_ewma_combination", {}), ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-faithful EWMA current-head applicability test.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = FeatureConfig()
    config_path = args.config.resolve()
    project_config = load_config(config_path)
    output_dir = args.output_dir.resolve() if args.output_dir else resolve_path(config_path, project_config["output"]["artifact_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    source_config = project_config["data_sources"]
    target_spec = TargetSpec(
        center=float(project_config["target_spec"]["center"]),
        tolerance=float(project_config["target_spec"]["tolerance"]),
    )
    lims_samples, _ = load_lims_samples(resolve_path(config_path, source_config["lims_path"]))
    labeled = add_out_of_spec_labels(lims_samples, target_spec).dropna(subset=["t90"]).copy()
    dcs = load_dcs_frame(
        resolve_path(config_path, source_config["dcs_main_path"]),
        resolve_path(config_path, source_config.get("dcs_supplemental_path")),
    )

    precleaned_sensors, preclean_audit = unsupervised_sensor_preclean(dcs)
    baseline_table, baseline_audit = build_baseline_feature_table(
        labeled=labeled,
        dcs=dcs,
        sensors=precleaned_sensors,
        window_minutes=config.baseline_window_minutes,
        min_rows_per_window=config.min_rows_per_window,
    )

    result_rows: list[dict[str, Any]] = []
    fold_evidence: dict[str, Any] = {}
    ewma_feature_exports: list[pd.DataFrame] = []
    ewma_combo_audits: list[dict[str, Any]] = []

    model_names = ["multinomial_logistic", "random_forest_balanced"]
    for tau_minutes in config.tau_grid:
        for window_minutes in config.window_grid:
            for lambda_value in config.lambda_grid:
                ewma_table, ewma_audit = compute_recursive_ewma_features(
                    labeled=labeled,
                    dcs=dcs,
                    sensors=precleaned_sensors,
                    tau_minutes=int(tau_minutes),
                    window_minutes=int(window_minutes),
                    lambda_value=float(lambda_value),
                    min_rows_per_window=config.min_rows_per_window,
                    min_valid_points_per_sensor=config.min_valid_points_per_sensor,
                )
                ewma_combo_audits.append(ewma_audit)
                for topk_target in config.topk_grid:
                    for model_name in model_names:
                        baseline_metrics, ewma_metrics, fold_frame = run_baseline_and_treatment(
                            baseline_table=baseline_table,
                            ewma_table=ewma_table,
                            candidate_sensors=precleaned_sensors,
                            topk_target=int(topk_target),
                            model_name=model_name,
                            screening_stat=config.screening_stat_for_sensor,
                            n_splits=config.n_splits,
                        )
                        combo_key = f"{model_name}|{tau_minutes}|{window_minutes}|{lambda_value}|{topk_target}"
                        fold_evidence[combo_key] = fold_frame.to_dict(orient="records")

                        result_rows.append(
                            {
                                "model_name": model_name,
                                "tau_minutes": np.nan,
                                "window_minutes": config.baseline_window_minutes,
                                "lambda": np.nan,
                                "topk_requested": int(topk_target),
                                "topk_sensors": int(baseline_metrics["topk_actual_median"]),
                                "feature_family": "baseline_short_window",
                                "samples": int(baseline_metrics["samples"]),
                                "macro_f1": baseline_metrics["macro_f1"],
                                "balanced_accuracy": baseline_metrics["balanced_accuracy"],
                                "weighted_f1": baseline_metrics["weighted_f1"],
                                "multiclass_log_loss": baseline_metrics["multiclass_log_loss"],
                                "multiclass_brier_score": baseline_metrics["multiclass_brier_score"],
                                "valid_fold_count": int(baseline_metrics["valid_fold_count"]),
                                "crosses_regime_boundary_ratio": np.nan,
                            }
                        )
                        result_rows.append(
                            {
                                "model_name": model_name,
                                "tau_minutes": int(tau_minutes),
                                "window_minutes": int(window_minutes),
                                "lambda": float(lambda_value),
                                "topk_requested": int(topk_target),
                                "topk_sensors": int(ewma_metrics["topk_actual_median"]),
                                "feature_family": "paper_faithful_recursive_ewma",
                                "samples": int(ewma_metrics["samples"]),
                                "macro_f1": ewma_metrics["macro_f1"],
                                "balanced_accuracy": ewma_metrics["balanced_accuracy"],
                                "weighted_f1": ewma_metrics["weighted_f1"],
                                "multiclass_log_loss": ewma_metrics["multiclass_log_loss"],
                                "multiclass_brier_score": ewma_metrics["multiclass_brier_score"],
                                "valid_fold_count": int(ewma_metrics["valid_fold_count"]),
                                "crosses_regime_boundary_ratio": np.nan,
                            }
                        )

                        export_columns = [
                            "decision_time",
                            "tau_minutes",
                            "window_minutes",
                            "lambda",
                            "t90",
                            "is_in_spec",
                            "is_above_spec",
                            "is_below_spec",
                        ] + [
                            column for column in ewma_table.columns if column.endswith("__ewma_recursive")
                        ]
                        export_frame = ewma_table[export_columns].copy()
                        export_frame["topk_sensors"] = int(ewma_metrics["topk_actual_median"])
                        export_frame["model_name"] = model_name
                        export_frame["feature_family"] = "paper_faithful_recursive_ewma"
                        ewma_feature_exports.append(export_frame)

    results = pd.DataFrame(result_rows).drop_duplicates()
    baseline_rows = results[results["feature_family"] == "baseline_short_window"].copy()
    baseline_reference = baseline_rows.sort_values(
        ["macro_f1", "balanced_accuracy", "multiclass_log_loss"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()
    ewma_rows = results[results["feature_family"] == "paper_faithful_recursive_ewma"].copy()
    best_ewma = ewma_rows.sort_values(
        ["macro_f1", "balanced_accuracy", "multiclass_log_loss"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()

    corresponding_fold_key = f"{best_ewma['model_name']}|{int(best_ewma['tau_minutes'])}|{int(best_ewma['window_minutes'])}|{best_ewma['lambda']}|{int(best_ewma['topk_requested'])}"
    fold_rows = fold_evidence.get(corresponding_fold_key, [])
    stable_fold_gain = False
    if fold_rows:
        gain_count = sum(
            1
            for row in fold_rows
            if row["ewma_macro_f1"] > row["baseline_macro_f1"] and row["ewma_balanced_accuracy"] > row["baseline_balanced_accuracy"]
        )
        stable_fold_gain = gain_count >= max(2, len(fold_rows) - 1)

    beats_baseline = bool(
        best_ewma["macro_f1"] > baseline_reference["macro_f1"]
        and best_ewma["balanced_accuracy"] > baseline_reference["balanced_accuracy"]
        and stable_fold_gain
    )

    summary = {
        "question": "Whether paper-faithful recursive EWMA data distillation is applicable to the current current-head three-class task.",
        "baseline_performance": baseline_reference,
        "feature_rescreening_enabled": True,
        "screening_strategy_description": (
            "Label-free global pre-cleaning removes all-missing, high-missingness, near-constant, and obvious-duplicate sensors. "
            "Then, inside each training fold only, sensors are ranked by the univariate three-class screening score of the baseline-window sensor__mean feature. "
            "Baseline and EWMA treatment share the same fold-local selected sensor list under the same top-k."
        ),
        "best_ewma_combination": best_ewma,
        "beats_baseline": beats_baseline,
        "stable_fold_gain": stable_fold_gain,
        "where_advantage_comes_from": (
            "Advantage is attributed only if the same fold-local sensor set is shared between baseline and treatment and the EWMA representation improves both macro_f1 and balanced_accuracy."
        ),
        "whether_result_is_unstable": not stable_fold_gain,
        "known_limitations": [
            "Regime boundaries are not reliably identified in the current data, so no no-crossing subset can be separately evaluated.",
            "Below-spec samples are much rarer than in-spec samples.",
            "Baseline and EWMA are restricted to simple model families to isolate representation effects.",
        ],
        "preclean_audit": preclean_audit,
        "baseline_window_audit": baseline_audit,
        "ewma_combo_audit_examples": ewma_combo_audits[:5],
    }

    results.to_csv(output_dir / RESULT_CSV, index=False, encoding="utf-8-sig")
    with (output_dir / SUMMARY_JSON).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    if ewma_feature_exports:
        pd.concat(ewma_feature_exports, ignore_index=True).to_csv(output_dir / FEATURE_ROWS_CSV, index=False, encoding="utf-8-sig")

    audit_state = {
        "recursive_ewma_strictly_used": True,
        "explicit_tau_used": True,
        "supervised_rescreening_enabled": True,
        "screening_train_fold_only": True,
        "regime_boundaries_identified": False,
        "total_samples_dropped_due_to_short_windows": int(
            baseline_audit["dropped_due_to_insufficient_window_rows"]
            + sum(item["dropped_due_to_insufficient_window_rows"] for item in ewma_combo_audits)
        ),
        "screening_strategy_description": summary["screening_strategy_description"],
        "best_ewma_combination": best_ewma,
    }
    write_audit_markdown(output_dir / AUDIT_MD, summary, audit_state)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
