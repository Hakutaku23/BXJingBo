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
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(__file__).resolve().parent
V3_ROOT = EXPERIMENT_DIR.parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import TargetSpec, add_out_of_spec_labels, build_dcs_feature_table, load_dcs_frame, load_lims_samples


DEFAULT_CONFIG_PATH = V3_ROOT / "config" / "phase1_warning.yaml"
CLASS_ORDER = ["below_spec", "in_spec", "above_spec"]
BLOCKED_COLUMNS = {
    "sample_time",
    "sample_name",
    "t90",
    "is_in_spec",
    "is_out_of_spec",
    "is_above_spec",
    "is_below_spec",
    "window_minutes",
    "rows_in_window",
    "volatile_lab",
    "volatile_online",
    "bromine",
    "calcium_stearate",
    "calcium",
    "stabilizer",
    "mooney",
    "antioxidant",
}


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


def parse_list_argument(value: str, cast=int) -> list[Any]:
    return [cast(item) for item in value.split(",") if item.strip()]


def parse_stat_sets(value: str) -> list[list[str]]:
    return [[item.strip() for item in chunk.split(",") if item.strip()] for chunk in value.split(";") if chunk.strip()]


def get_dcs_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in BLOCKED_COLUMNS and "__" in column]


def split_feature_name(feature_name: str) -> tuple[str, str]:
    if "__" not in feature_name:
        return feature_name, "raw"
    sensor, stat = feature_name.split("__", 1)
    return sensor, stat


def encode_three_class_target(frame: pd.DataFrame) -> np.ndarray:
    target = np.full(len(frame), -1, dtype=int)
    target[frame["is_below_spec"].astype(int).to_numpy() == 1] = 0
    target[frame["is_in_spec"].astype(int).to_numpy() == 1] = 1
    target[frame["is_above_spec"].astype(int).to_numpy() == 1] = 2
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


def score_single_feature_multiclass(x: pd.Series, y: np.ndarray) -> dict[str, float | int | None]:
    valid = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.Series(y)}).dropna()
    if valid.empty:
        return {
            "valid_samples": 0,
            "valid_ratio": 0.0,
            "macro_auc_sym": None,
            "macro_corr_abs": None,
            "screen_score": 0.0,
        }

    x_values = valid["x"].to_numpy(dtype=float)
    y_values = valid["y"].to_numpy(dtype=int)
    valid_ratio = float(len(valid) / len(x)) if len(x) else 0.0
    if len(np.unique(x_values)) < 2 or len(np.unique(y_values)) < 3:
        return {
            "valid_samples": int(len(valid)),
            "valid_ratio": valid_ratio,
            "macro_auc_sym": None,
            "macro_corr_abs": 0.0,
            "screen_score": 0.0,
        }

    aucs = []
    corrs = []
    for class_idx in range(3):
        y_binary = (y_values == class_idx).astype(int)
        if len(np.unique(y_binary)) < 2:
            continue
        auc = roc_auc_score(y_binary, x_values)
        aucs.append(max(float(auc), float(1.0 - auc)))
        corr = float(abs(np.corrcoef(x_values, y_binary)[0, 1])) if np.nanstd(x_values) > 0 else 0.0
        if np.isnan(corr):
            corr = 0.0
        corrs.append(corr)

    if not aucs:
        return {
            "valid_samples": int(len(valid)),
            "valid_ratio": valid_ratio,
            "macro_auc_sym": None,
            "macro_corr_abs": 0.0,
            "screen_score": 0.0,
        }

    macro_auc_sym = float(np.mean(aucs))
    macro_corr_abs = float(np.mean(corrs)) if corrs else 0.0
    auc_component = max((macro_auc_sym - 0.5) * 2.0, 0.0)
    screen_score = float(valid_ratio * (0.75 * auc_component + 0.25 * macro_corr_abs))
    return {
        "valid_samples": int(len(valid)),
        "valid_ratio": valid_ratio,
        "macro_auc_sym": macro_auc_sym,
        "macro_corr_abs": macro_corr_abs,
        "screen_score": screen_score,
    }


def rank_window_features_multiclass(feature_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = encode_three_class_target(feature_table)
    feature_rows: list[dict[str, Any]] = []
    for feature_name in get_dcs_feature_columns(feature_table):
        sensor, stat = split_feature_name(feature_name)
        metrics = score_single_feature_multiclass(feature_table[feature_name], y)
        feature_rows.append(
            {
                "feature_name": feature_name,
                "sensor_name": sensor,
                "stat_name": stat,
                **metrics,
            }
        )

    feature_df = pd.DataFrame(feature_rows).sort_values(
        ["screen_score", "macro_auc_sym", "macro_corr_abs", "feature_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    sensor_df = (
        feature_df.groupby("sensor_name", as_index=False)
        .agg(
            feature_count=("feature_name", "count"),
            best_feature=("feature_name", "first"),
            best_score=("screen_score", "max"),
            mean_score=("screen_score", "mean"),
            best_macro_auc_sym=("macro_auc_sym", "max"),
        )
        .sort_values(["best_score", "mean_score", "best_macro_auc_sym", "sensor_name"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )
    stat_df = (
        feature_df.groupby("stat_name", as_index=False)
        .agg(
            feature_count=("feature_name", "count"),
            mean_score=("screen_score", "mean"),
            median_score=("screen_score", "median"),
            best_score=("screen_score", "max"),
            mean_macro_auc_sym=("macro_auc_sym", "mean"),
        )
        .sort_values(["mean_score", "best_score", "stat_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return feature_df, sensor_df, stat_df


def build_selected_features(feature_table: pd.DataFrame, sensors: list[str], stats: list[str]) -> list[str]:
    sensor_set = set(sensors)
    stat_set = set(stats)
    selected = []
    for column in feature_table.columns:
        if "__" not in column:
            continue
        sensor, stat = split_feature_name(column)
        if sensor in sensor_set and stat in stat_set:
            selected.append(column)
    return selected


def collect_oof_probabilities_multiclass(X: pd.DataFrame, y: np.ndarray, model_name: str, n_splits: int = 5) -> pd.DataFrame:
    splitter = TimeSeriesSplit(n_splits=n_splits)
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


def evaluate_subset_multiclass(feature_table: pd.DataFrame, sensor_names: list[str], stats: list[str], model_name: str) -> dict[str, Any]:
    selected_features = build_selected_features(feature_table, sensor_names, stats)
    if not selected_features:
        return {"status": "no_features"}

    usable = feature_table.sort_values("sample_time").reset_index(drop=True).copy()
    y = encode_three_class_target(usable)
    valid_mask = y >= 0
    usable = usable.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    if len(np.unique(y)) < 3:
        return {"status": "missing_classes", "samples": int(len(usable))}

    oof = collect_oof_probabilities_multiclass(usable[selected_features], y, model_name=model_name)
    if oof.empty:
        return {"status": "no_scored_samples", "samples": int(len(usable))}

    prob = oof[[f"prob_{name}" for name in CLASS_ORDER]].to_numpy(dtype=float)
    y_true = oof["y_true"].to_numpy(dtype=int)
    pred = oof["pred_class"].to_numpy(dtype=int)
    return {
        "status": "ok",
        "samples": int(len(usable)),
        "scored_samples": int(len(oof)),
        "sensor_count": int(len(sensor_names)),
        "feature_count": int(len(selected_features)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "multiclass_log_loss": float(log_loss(y_true, prob, labels=[0, 1, 2])),
        "multiclass_brier_score": multiclass_brier_score(y_true, prob, n_classes=3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Research windows/sensors/stats again for the current three-class head.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--windows", type=str, default="8,10,12,15,20,30,45")
    parser.add_argument("--topk", type=str, default="5,10,15")
    parser.add_argument("--stat-sets", type=str, default="mean,min,last,max;mean,min,last,max,std")
    parser.add_argument("--models", type=str, default="multinomial_logistic,random_forest_balanced")
    parser.add_argument("--output-prefix", type=str, default="phase1_three_class_feature_research")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    windows = parse_list_argument(args.windows, int)
    topk_values = parse_list_argument(args.topk, int)
    stat_sets = parse_stat_sets(args.stat_sets)
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

    feature_rows_all: list[pd.DataFrame] = []
    sensor_rows_all: list[pd.DataFrame] = []
    stat_rows_all: list[pd.DataFrame] = []
    subset_rows: list[dict[str, Any]] = []
    window_summary_rows: list[dict[str, Any]] = []

    for window_minutes in windows:
        feature_table = build_dcs_feature_table(
            labeled,
            dcs,
            window_minutes=window_minutes,
            min_points_per_window=int(config["window_search"]["min_points_per_window"]),
        )
        feature_df, sensor_df, stat_df = rank_window_features_multiclass(feature_table)
        feature_df.insert(0, "window_minutes", int(window_minutes))
        sensor_df.insert(0, "window_minutes", int(window_minutes))
        stat_df.insert(0, "window_minutes", int(window_minutes))
        feature_rows_all.append(feature_df)
        sensor_rows_all.append(sensor_df)
        stat_rows_all.append(stat_df)

        top_sensor_names = sensor_df["sensor_name"].tolist()
        for topk in topk_values:
            selected_sensors = top_sensor_names[:topk]
            for stat_set in stat_sets:
                for model_name in model_names:
                    metrics = evaluate_subset_multiclass(feature_table, selected_sensors, stat_set, model_name)
                    metrics.update(
                        {
                            "window_minutes": int(window_minutes),
                            "topk_sensors": int(topk),
                            "stats": ",".join(stat_set),
                            "model_name": model_name,
                            "sensor_names": selected_sensors,
                        }
                    )
                    subset_rows.append(metrics)

        current_window_rows = [
            row
            for row in subset_rows
            if row["window_minutes"] == int(window_minutes) and row.get("status") == "ok"
        ]
        best_subset = None
        if current_window_rows:
            best_subset = max(
                current_window_rows,
                key=lambda row: (row["macro_f1"], row["balanced_accuracy"], -row["multiclass_log_loss"], -row["multiclass_brier_score"]),
            )

        top_sensor_preview = sensor_df.head(10)["sensor_name"].tolist()
        top_stat_preview = stat_df.head(8)["stat_name"].tolist()
        window_summary_rows.append(
            {
                "window_minutes": int(window_minutes),
                "samples": int(len(feature_table)),
                "top_sensor_1": top_sensor_preview[0] if len(top_sensor_preview) > 0 else None,
                "top_sensor_2": top_sensor_preview[1] if len(top_sensor_preview) > 1 else None,
                "top_sensor_3": top_sensor_preview[2] if len(top_sensor_preview) > 2 else None,
                "top_stat_1": top_stat_preview[0] if len(top_stat_preview) > 0 else None,
                "top_stat_2": top_stat_preview[1] if len(top_stat_preview) > 1 else None,
                "top_stat_3": top_stat_preview[2] if len(top_stat_preview) > 2 else None,
                "best_subset_topk": int(best_subset["topk_sensors"]) if best_subset else None,
                "best_subset_stats": best_subset["stats"] if best_subset else None,
                "best_subset_model": best_subset["model_name"] if best_subset else None,
                "best_subset_macro_f1": float(best_subset["macro_f1"]) if best_subset else None,
                "best_subset_balanced_accuracy": float(best_subset["balanced_accuracy"]) if best_subset else None,
                "best_subset_log_loss": float(best_subset["multiclass_log_loss"]) if best_subset else None,
            }
        )

    feature_rankings = pd.concat(feature_rows_all, ignore_index=True)
    sensor_rankings = pd.concat(sensor_rows_all, ignore_index=True)
    stat_rankings = pd.concat(stat_rows_all, ignore_index=True)
    subset_benchmarks = pd.DataFrame(subset_rows)
    subset_benchmarks_ok = subset_benchmarks[subset_benchmarks["status"] == "ok"].copy()
    subset_benchmarks_ok = subset_benchmarks_ok.sort_values(
        ["macro_f1", "balanced_accuracy", "multiclass_log_loss", "multiclass_brier_score", "window_minutes", "topk_sensors"],
        ascending=[False, False, True, True, True, True],
    )
    window_summary = pd.DataFrame(window_summary_rows).sort_values(
        ["best_subset_macro_f1", "best_subset_balanced_accuracy", "best_subset_log_loss", "window_minutes"],
        ascending=[False, False, True, True],
    )

    best_row = subset_benchmarks_ok.iloc[0].to_dict() if not subset_benchmarks_ok.empty else None
    summary = {
        "class_order": CLASS_ORDER,
        "window_summary_rows": window_summary.to_dict(orient="records"),
        "recommended_current_three_class_setup": best_row,
        "notes": {
            "goal": "Re-search window, sensors, stats, and model family after changing the current head to a three-class probability target.",
            "interpretation": "The previous binary-selected 8-minute setup should be treated only as a bootstrap baseline, not as the final three-class configuration.",
        },
    }

    feature_rankings.to_csv(artifact_dir / f"{args.output_prefix}_feature_rankings.csv", index=False, encoding="utf-8-sig")
    sensor_rankings.to_csv(artifact_dir / f"{args.output_prefix}_sensor_rankings.csv", index=False, encoding="utf-8-sig")
    stat_rankings.to_csv(artifact_dir / f"{args.output_prefix}_stat_rankings.csv", index=False, encoding="utf-8-sig")
    subset_benchmarks.to_csv(artifact_dir / f"{args.output_prefix}_subset_benchmarks.csv", index=False, encoding="utf-8-sig")
    window_summary.to_csv(artifact_dir / f"{args.output_prefix}_window_summary.csv", index=False, encoding="utf-8-sig")
    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
