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


def parse_window_groups(value: str) -> list[list[int]]:
    return [parse_int_list(group) for group in value.split(";") if group.strip()]


def parse_weight_groups(value: str) -> list[list[float]]:
    return [parse_float_list(group) for group in value.split(";") if group.strip()]


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


def feature_column(sensor: str, stat: str) -> str:
    return f"{sensor}__{stat}"


def build_selected_columns(sensor_names: list[str], stats: list[str]) -> list[str]:
    return [feature_column(sensor, stat) for sensor in sensor_names for stat in stats]


def build_window_feature_tables(
    labeled: pd.DataFrame,
    dcs: pd.DataFrame,
    windows: list[int],
    min_points_per_window: int,
) -> dict[int, pd.DataFrame]:
    feature_tables: dict[int, pd.DataFrame] = {}
    for window in windows:
        table = build_dcs_feature_table(
            labeled,
            dcs,
            window_minutes=int(window),
            min_points_per_window=min_points_per_window,
        )
        if table.empty:
            continue
        keep = [
            "sample_time",
            "t90",
            "is_in_spec",
            "is_out_of_spec",
            "is_above_spec",
            "is_below_spec",
        ]
        feature_columns = [column for column in table.columns if "__" in column]
        feature_tables[int(window)] = table[keep + feature_columns].copy()
    return feature_tables


def build_multiscale_representations(
    base_table: pd.DataFrame,
    feature_tables: dict[int, pd.DataFrame],
    window_group: list[int],
    weights: list[float],
    sensor_names: list[str],
    short_stats: list[str],
    long_stats: list[str],
) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    if len(window_group) != len(weights):
        raise ValueError("window_group and weights must have the same length")
    weights_array = np.array(weights, dtype=float)
    weights_array = weights_array / weights_array.sum()
    short_window = int(window_group[0])
    long_windows = [int(window) for window in window_group[1:]]
    long_weights = np.array(weights_array[1:], dtype=float)
    long_weights = long_weights / long_weights.sum() if len(long_weights) else long_weights

    core_stats = list(dict.fromkeys([*short_stats, "std", "delta"]))
    selected_columns = build_selected_columns(sensor_names, core_stats)

    aligned = None
    for window in window_group:
        frame = feature_tables[int(window)].copy()
        rename_map = {
            column: f"w{int(window)}__{column}"
            for column in frame.columns
            if "__" in column and column in selected_columns
        }
        frame = frame.rename(columns=rename_map)
        if aligned is None:
            aligned = frame
        else:
            aligned = aligned.merge(
                frame[["sample_time", *rename_map.values()]],
                on="sample_time",
                how="inner",
            )
    if aligned is None or aligned.empty:
        return {}

    meta_columns = ["sample_time", "t90", "is_in_spec", "is_out_of_spec", "is_above_spec", "is_below_spec"]
    aligned = aligned.sort_values("sample_time").reset_index(drop=True)

    representations: dict[str, tuple[pd.DataFrame, list[str]]] = {}

    concat_columns = [
        f"w{window}__{feature_column(sensor, stat)}"
        for window in window_group
        for sensor in sensor_names
        for stat in short_stats
        if f"w{window}__{feature_column(sensor, stat)}" in aligned.columns
    ]
    representations["multiscale_concat_core"] = (aligned[meta_columns + concat_columns].copy(), concat_columns)

    weighted_frame = aligned[meta_columns].copy()
    weighted_columns: list[str] = []
    for sensor in sensor_names:
        for stat in short_stats:
            candidates = []
            for idx, window in enumerate(window_group):
                column_name = f"w{window}__{feature_column(sensor, stat)}"
                if column_name in aligned.columns:
                    candidates.append((weights_array[idx], column_name))
            if not candidates:
                continue
            new_name = f"{sensor}__weighted_{stat}"
            weighted_frame[new_name] = sum(weight * aligned[column] for weight, column in candidates)
            weighted_columns.append(new_name)
    representations["window_weighted_core"] = (weighted_frame, weighted_columns)

    extended_frame = weighted_frame.copy()
    extended_columns = list(weighted_columns)
    for sensor in sensor_names:
        for stat in ["std", "delta"]:
            candidates = []
            for idx, window in enumerate(window_group):
                column_name = f"w{window}__{feature_column(sensor, stat)}"
                if column_name in aligned.columns:
                    candidates.append((weights_array[idx], column_name))
            if not candidates:
                continue
            new_name = f"{sensor}__weighted_{stat}"
            extended_frame[new_name] = sum(weight * aligned[column] for weight, column in candidates)
            extended_columns.append(new_name)
    representations["window_weighted_plus_variability"] = (extended_frame, extended_columns)

    hybrid_frame = base_table[meta_columns + build_selected_columns(sensor_names, short_stats)].copy()
    rename_short = {
        feature_column(sensor, stat): f"{sensor}__short_{stat}"
        for sensor in sensor_names
        for stat in short_stats
        if feature_column(sensor, stat) in hybrid_frame.columns
    }
    hybrid_frame = hybrid_frame.rename(columns=rename_short)
    hybrid_columns = list(rename_short.values())
    hybrid_frame = hybrid_frame.merge(
        extended_frame[["sample_time", *extended_columns]],
        on="sample_time",
        how="inner",
    )
    hybrid_columns.extend(extended_columns)
    for sensor in sensor_names:
        for stat in ["mean", "last"]:
            short_column = f"{sensor}__short_{stat}"
            long_candidates = []
            for idx, window in enumerate(long_windows):
                column_name = f"w{window}__{feature_column(sensor, stat)}"
                if column_name in aligned.columns:
                    long_candidates.append((long_weights[idx], column_name))
            if short_column not in hybrid_frame.columns or not long_candidates:
                continue
            long_weighted_name = f"{sensor}__weighted_long_{stat}"
            hybrid_frame[long_weighted_name] = sum(weight * aligned[column] for weight, column in long_candidates)
            hybrid_columns.append(long_weighted_name)
            contrast_name = f"{sensor}__short_minus_long_{stat}"
            hybrid_frame[contrast_name] = hybrid_frame[short_column] - hybrid_frame[long_weighted_name]
            hybrid_columns.append(contrast_name)
    representations["hybrid_short_plus_weighted"] = (hybrid_frame, hybrid_columns)

    return representations


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
    work = frame.sort_values("sample_time").reset_index(drop=True).copy()
    y = encode_three_class_target(work)
    valid_mask = y >= 0
    work = work.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    if len(np.unique(y)) < 3:
        return {"status": "missing_classes", "samples": int(len(work))}
    oof = collect_oof_multiclass(work[feature_columns], y, model_name=model_name)
    if oof.empty:
        return {"status": "no_scored_samples", "samples": int(len(work))}
    metrics = score_multiclass_output(oof)
    return {"status": "ok", **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Current-head multiscale weighted fusion experiment under the locked v1.2 plan.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--current-head-summary", type=Path, default=DEFAULT_CURRENT_HEAD_SUMMARY)
    parser.add_argument("--window-groups", type=str, default="8,15,30;8,15,45;8,20,45;8,10,30")
    parser.add_argument("--weight-groups", type=str, default="0.60,0.25,0.15;0.50,0.30,0.20;0.70,0.20,0.10")
    parser.add_argument("--models", type=str, default="random_forest_balanced")
    parser.add_argument("--output-prefix", type=str, default="phase1_current_head_multiscale_weighted_fusion")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    current_head_summary = json.loads(args.current_head_summary.read_text(encoding="utf-8"))
    setup = current_head_summary["recommended_current_three_class_setup"]
    sensor_names = list(setup["sensor_names"])
    short_stats = str(setup["stats"]).split(",")
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    window_groups = parse_window_groups(args.window_groups)
    weight_groups = parse_weight_groups(args.weight_groups)
    unique_windows = sorted({window for group in window_groups for window in group})

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

    feature_tables = build_window_feature_tables(
        labeled=labeled,
        dcs=dcs,
        windows=unique_windows,
        min_points_per_window=int(config["window_search"]["min_points_per_window"]),
    )
    base_window = int(setup["window_minutes"])
    base_table = feature_tables[base_window]
    baseline_columns = build_selected_columns(sensor_names, short_stats)

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        baseline_metrics = evaluate_representation(base_table, baseline_columns, model_name=model_name)
        baseline_metrics.update(
            {
                "representation_name": "baseline_short_window",
                "model_name": model_name,
                "window_group": str([base_window]),
                "weights": None,
                "feature_family": "baseline_short_window",
                "feature_count": int(len(baseline_columns)),
            }
        )
        rows.append(baseline_metrics)

    for window_group in window_groups:
        for weight_group in weight_groups:
            if len(window_group) != len(weight_group):
                continue
            representations = build_multiscale_representations(
                base_table=base_table,
                feature_tables=feature_tables,
                window_group=window_group,
                weights=weight_group,
                sensor_names=sensor_names,
                short_stats=short_stats,
                long_stats=["std", "delta"],
            )
            for family_name, (frame, feature_columns) in representations.items():
                for model_name in model_names:
                    metrics = evaluate_representation(frame, feature_columns, model_name=model_name)
                    metrics.update(
                        {
                            "representation_name": f"{family_name}_w{'-'.join(map(str, window_group))}_a{'-'.join(f'{w:.2f}' for w in weight_group)}",
                            "model_name": model_name,
                            "window_group": ",".join(map(str, window_group)),
                            "weights": ",".join(f"{weight:.2f}" for weight in weight_group),
                            "feature_family": family_name,
                            "feature_count": int(len(feature_columns)),
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
        "multiscale_design": {
            "window_group_candidates": window_groups,
            "window_weight_candidates": weight_groups,
            "short_window_role": "retain the current 8-minute baseline features as the main local-context branch",
            "long_window_role": "inject longer operating-state information through window-level weighted fusion",
            "families": [
                "multiscale_concat_core",
                "window_weighted_core",
                "window_weighted_plus_variability",
                "hybrid_short_plus_weighted",
            ],
        },
        "rows": ok_result.to_dict(orient="records"),
        "recommended_representation": best_row,
    }

    result.to_csv(artifact_dir / f"{args.output_prefix}_summary.csv", index=False, encoding="utf-8-sig")
    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
