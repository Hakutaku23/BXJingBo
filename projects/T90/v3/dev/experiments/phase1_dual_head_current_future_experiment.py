from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(__file__).resolve().parent
V3_ROOT = EXPERIMENT_DIR.parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import TargetSpec, add_out_of_spec_labels, build_dcs_feature_table, load_dcs_frame, load_lims_samples


DEFAULT_CONFIG_PATH = V3_ROOT / "config" / "phase1_warning.yaml"
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


def split_feature_name(feature_name: str) -> tuple[str, str]:
    if "__" not in feature_name:
        return feature_name, "raw"
    sensor, stat = feature_name.split("__", 1)
    return sensor, stat


def get_dcs_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in BLOCKED_COLUMNS and "__" in column]


def score_single_feature(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    valid = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if valid.empty:
        return {
            "valid_samples": 0,
            "valid_ratio": 0.0,
            "auc_sym": None,
            "corr_abs": None,
            "screen_score": 0.0,
        }

    x_values = valid["x"].to_numpy(dtype=float)
    y_values = valid["y"].to_numpy(dtype=int)
    valid_ratio = float(len(valid) / len(x)) if len(x) else 0.0

    if len(np.unique(x_values)) < 2 or len(np.unique(y_values)) < 2:
        return {
            "valid_samples": int(len(valid)),
            "valid_ratio": valid_ratio,
            "auc_sym": None,
            "corr_abs": 0.0,
            "screen_score": 0.0,
        }

    auc = roc_auc_score(y_values, x_values)
    auc_sym = float(max(auc, 1.0 - auc))
    corr = float(abs(np.corrcoef(x_values, y_values)[0, 1])) if np.nanstd(x_values) > 0 else 0.0
    if np.isnan(corr):
        corr = 0.0

    auc_component = max((auc_sym - 0.5) * 2.0, 0.0)
    screen_score = float(valid_ratio * (0.7 * auc_component + 0.3 * corr))
    return {
        "valid_samples": int(len(valid)),
        "valid_ratio": valid_ratio,
        "auc_sym": auc_sym,
        "corr_abs": corr,
        "screen_score": screen_score,
    }


def rank_window_features(feature_table: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, Any]] = []
    for feature_name in get_dcs_feature_columns(feature_table):
        sensor, stat = split_feature_name(feature_name)
        metrics = score_single_feature(feature_table[feature_name], y)
        feature_rows.append(
            {
                "feature_name": feature_name,
                "sensor_name": sensor,
                "stat_name": stat,
                **metrics,
            }
        )

    feature_df = pd.DataFrame(feature_rows).sort_values(
        ["screen_score", "auc_sym", "corr_abs", "feature_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    sensor_df = (
        feature_df.groupby("sensor_name", as_index=False)
        .agg(
            feature_count=("feature_name", "count"),
            best_feature=("feature_name", "first"),
            best_score=("screen_score", "max"),
            mean_score=("screen_score", "mean"),
            best_auc_sym=("auc_sym", "max"),
        )
        .sort_values(["best_score", "mean_score", "best_auc_sym", "sensor_name"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )
    return feature_df, sensor_df


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


def build_future_window_label(frame: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    work = frame.sort_values("sample_time").reset_index(drop=True).copy()
    times = pd.to_datetime(work["sample_time"], errors="coerce")
    labels = work["is_out_of_spec"].astype(int).to_numpy()
    target = np.full(len(work), np.nan, dtype=float)

    for idx, current_time in enumerate(times):
        if pd.isna(current_time):
            continue
        future_mask = (times > current_time) & (times <= current_time + pd.Timedelta(minutes=horizon_minutes))
        future_indices = np.flatnonzero(future_mask.to_numpy(dtype=bool))
        if len(future_indices) == 0:
            continue
        target[idx] = float(labels[future_indices].max())

    return pd.Series(target, index=work.index, name=f"future_out_of_spec_{horizon_minutes}m")


def collect_oof_probabilities(feature_table: pd.DataFrame, selected_features: list[str], target: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    work = feature_table.sort_values("sample_time").reset_index(drop=True).copy()
    target_aligned = target.loc[work.index]
    usable_mask = target_aligned.notna()
    work = work.loc[usable_mask].reset_index(drop=True)
    y = target_aligned.loc[usable_mask].astype(int).to_numpy()
    X = work[selected_features]

    probabilities = np.full(len(work), np.nan, dtype=float)
    model = Pipeline(
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

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue

        fold_columns = [column for column in selected_features if X_train[column].notna().any()]
        if not fold_columns:
            continue

        model.fit(X_train[fold_columns], y_train)
        probabilities[test_idx] = model.predict_proba(X_test[fold_columns])[:, 1]

    scored_mask = ~np.isnan(probabilities)
    if scored_mask.sum() == 0:
        return pd.DataFrame()

    result = work.loc[scored_mask, ["sample_time", "t90", "is_out_of_spec"]].copy()
    result["target"] = y[scored_mask]
    result["probability"] = probabilities[scored_mask]
    return result.reset_index(drop=True)


def scan_thresholds(oof_frame: pd.DataFrame) -> list[dict[str, float]]:
    y_true = oof_frame["target"].astype(int).to_numpy()
    prob = oof_frame["probability"].to_numpy(dtype=float)
    rows = []
    for threshold in np.arange(0.05, 1.00, 0.05):
        pred = prob >= threshold
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1": float(f1),
                "predicted_positive_ratio": float(pred.mean()),
            }
        )
    return rows


def choose_threshold(scan_rows: list[dict[str, float]], min_recall: float = 0.80) -> dict[str, float]:
    candidates = [row for row in scan_rows if row["recall"] >= min_recall]
    if not candidates:
        return max(scan_rows, key=lambda row: (row["f1"], row["specificity"]))
    return max(candidates, key=lambda row: (row["specificity"], row["precision"], row["f1"]))


def evaluate_target_configs(
    feature_tables: dict[int, pd.DataFrame],
    target_name: str,
    target_builder,
    windows: list[int],
    topk_values: list[int],
    stat_sets: list[list[str]],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame], dict[int, pd.Series]]:
    rows: list[dict[str, Any]] = []
    feature_rankings: dict[int, pd.DataFrame] = {}
    sensor_rankings: dict[int, pd.DataFrame] = {}
    targets_by_window: dict[int, pd.Series] = {}

    for window_minutes in windows:
        feature_table = feature_tables[window_minutes]
        target = target_builder(feature_table)
        targets_by_window[window_minutes] = target
        usable_mask = target.notna()
        if int(usable_mask.sum()) == 0:
            continue

        feature_df, sensor_df = rank_window_features(feature_table.loc[usable_mask].reset_index(drop=True), target.loc[usable_mask].reset_index(drop=True))
        feature_df.insert(0, "window_minutes", int(window_minutes))
        feature_df.insert(1, "target_name", target_name)
        sensor_df.insert(0, "window_minutes", int(window_minutes))
        sensor_df.insert(1, "target_name", target_name)
        feature_rankings[window_minutes] = feature_df
        sensor_rankings[window_minutes] = sensor_df

        sensor_names = sensor_df["sensor_name"].tolist()
        for topk in topk_values:
            selected_sensors = sensor_names[:topk]
            for stats in stat_sets:
                selected_features = build_selected_features(feature_table, selected_sensors, stats)
                if not selected_features:
                    continue
                oof_frame = collect_oof_probabilities(feature_table, selected_features, target)
                if oof_frame.empty or len(np.unique(oof_frame["target"])) < 2:
                    rows.append(
                        {
                            "target_name": target_name,
                            "window_minutes": int(window_minutes),
                            "topk_sensors": int(topk),
                            "stats": ",".join(stats),
                            "sensor_names": selected_sensors,
                            "status": "not_enough_scored_samples",
                        }
                    )
                    continue

                scan_rows = scan_thresholds(oof_frame)
                threshold_row = choose_threshold(scan_rows)
                rows.append(
                    {
                        "target_name": target_name,
                        "window_minutes": int(window_minutes),
                        "topk_sensors": int(topk),
                        "stats": ",".join(stats),
                        "sensor_names": selected_sensors,
                        "status": "ok",
                        "samples": int(len(oof_frame)),
                        "positive_ratio": float(oof_frame["target"].mean()),
                        "roc_auc": float(roc_auc_score(oof_frame["target"], oof_frame["probability"])),
                        "average_precision": float(average_precision_score(oof_frame["target"], oof_frame["probability"])),
                        "threshold": float(threshold_row["threshold"]),
                        "precision": float(threshold_row["precision"]),
                        "recall": float(threshold_row["recall"]),
                        "specificity": float(threshold_row["specificity"]),
                        "f1": float(threshold_row["f1"]),
                        "predicted_positive_ratio": float(threshold_row["predicted_positive_ratio"]),
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["roc_auc", "average_precision", "f1", "specificity", "window_minutes", "topk_sensors"],
            ascending=[False, False, False, False, True, True],
        ).reset_index(drop=True)
    return result, feature_rankings, sensor_rankings, targets_by_window


def evaluate_joint_state_machine(
    alarm_feature_table: pd.DataFrame,
    alarm_target: pd.Series,
    alarm_row: pd.Series,
    warning_feature_table: pd.DataFrame,
    warning_target: pd.Series,
    warning_row: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    alarm_features = build_selected_features(
        alarm_feature_table,
        list(alarm_row["sensor_names"]),
        str(alarm_row["stats"]).split(","),
    )
    warning_features = build_selected_features(
        warning_feature_table,
        list(warning_row["sensor_names"]),
        str(warning_row["stats"]).split(","),
    )

    alarm_oof = collect_oof_probabilities(alarm_feature_table, alarm_features, alarm_target).rename(
        columns={"target": "current_target", "probability": "current_probability"}
    )
    warning_oof = collect_oof_probabilities(warning_feature_table, warning_features, warning_target).rename(
        columns={"target": "future_target", "probability": "future_probability"}
    )
    if alarm_oof.empty or warning_oof.empty:
        return pd.DataFrame(), {"status": "missing_oof"}

    merged = alarm_oof.merge(
        warning_oof[["sample_time", "future_target", "future_probability"]],
        on="sample_time",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(), {"status": "no_overlap"}

    alarm_threshold = float(alarm_row["threshold"])
    warning_threshold = float(warning_row["threshold"])
    merged["pred_alarm"] = merged["current_probability"] >= alarm_threshold
    merged["pred_warning"] = (~merged["pred_alarm"]) & (merged["future_probability"] >= warning_threshold)
    merged["pred_state"] = np.where(merged["pred_alarm"], "alarm", np.where(merged["pred_warning"], "warning", "normal"))
    merged["actual_state"] = np.where(
        merged["current_target"] == 1,
        "alarm",
        np.where(merged["future_target"] == 1, "warning", "normal"),
    )

    current_y = merged["current_target"].astype(int).to_numpy()
    current_pred = merged["pred_alarm"].astype(int).to_numpy()
    current_tp = int(np.sum((current_pred == 1) & (current_y == 1)))
    current_fp = int(np.sum((current_pred == 1) & (current_y == 0)))
    current_tn = int(np.sum((current_pred == 0) & (current_y == 0)))
    current_fn = int(np.sum((current_pred == 0) & (current_y == 1)))

    normal_subset = merged[merged["current_target"] == 0].copy()
    future_recall = None
    future_specificity = None
    if not normal_subset.empty and len(normal_subset["future_target"].unique()) > 1:
        future_y = normal_subset["future_target"].astype(int).to_numpy()
        future_pred = normal_subset["pred_warning"].astype(int).to_numpy()
        future_tp = int(np.sum((future_pred == 1) & (future_y == 1)))
        future_fp = int(np.sum((future_pred == 1) & (future_y == 0)))
        future_tn = int(np.sum((future_pred == 0) & (future_y == 0)))
        future_fn = int(np.sum((future_pred == 0) & (future_y == 1)))
        future_recall = future_tp / (future_tp + future_fn) if (future_tp + future_fn) else 0.0
        future_specificity = future_tn / (future_tn + future_fp) if (future_tn + future_fp) else 0.0

    state_accuracy = float((merged["pred_state"] == merged["actual_state"]).mean())
    state_rows = (
        merged.groupby(["actual_state", "pred_state"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["actual_state", "pred_state"])
        .to_dict(orient="records")
    )

    summary = {
        "status": "ok",
        "overlap_samples": int(len(merged)),
        "state_accuracy": state_accuracy,
        "current_alarm_recall": float(current_tp / (current_tp + current_fn)) if (current_tp + current_fn) else 0.0,
        "current_alarm_specificity": float(current_tn / (current_tn + current_fp)) if (current_tn + current_fp) else 0.0,
        "future_warning_recall_given_current_normal": float(future_recall) if future_recall is not None else None,
        "future_warning_specificity_given_current_normal": float(future_specificity) if future_specificity is not None else None,
        "pred_alarm_ratio": float(merged["pred_alarm"].mean()),
        "pred_warning_ratio": float(merged["pred_warning"].mean()),
        "pred_normal_ratio": float((merged["pred_state"] == "normal").mean()),
        "state_confusion_rows": state_rows,
    }
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dual-head Phase 1 design: current alarm + future warning with independent DCS windows.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--alarm-windows", type=str, default="8,10,12,15,20,30")
    parser.add_argument("--warning-windows", type=str, default="8,10,12,15,20,30")
    parser.add_argument("--future-horizon", type=int, default=240)
    parser.add_argument("--topk", type=str, default="5,10,15")
    parser.add_argument("--stat-sets", type=str, default="mean,min,last,max;mean,min,last,max,std")
    parser.add_argument("--output-prefix", type=str, default="phase1_dual_head_current_future")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    alarm_windows = parse_list_argument(args.alarm_windows, int)
    warning_windows = parse_list_argument(args.warning_windows, int)
    all_windows = sorted(set(alarm_windows + warning_windows))
    topk_values = parse_list_argument(args.topk, int)
    stat_sets = parse_stat_sets(args.stat_sets)

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

    feature_tables: dict[int, pd.DataFrame] = {}
    for window_minutes in all_windows:
        feature_tables[window_minutes] = build_dcs_feature_table(
            labeled,
            dcs,
            window_minutes=window_minutes,
            min_points_per_window=int(config["window_search"]["min_points_per_window"]),
        )

    alarm_result, alarm_feature_rankings, alarm_sensor_rankings, alarm_targets = evaluate_target_configs(
        feature_tables=feature_tables,
        target_name="current_alarm",
        target_builder=lambda frame: frame["is_out_of_spec"].astype(float),
        windows=alarm_windows,
        topk_values=topk_values,
        stat_sets=stat_sets,
    )
    warning_result, warning_feature_rankings, warning_sensor_rankings, warning_targets = evaluate_target_configs(
        feature_tables=feature_tables,
        target_name="future_warning",
        target_builder=lambda frame: build_future_window_label(frame, horizon_minutes=int(args.future_horizon)),
        windows=warning_windows,
        topk_values=topk_values,
        stat_sets=stat_sets,
    )

    if alarm_result.empty or warning_result.empty:
        raise RuntimeError("Alarm or warning search returned no valid rows.")

    best_alarm = alarm_result.iloc[0]
    best_warning = warning_result.iloc[0]

    merged_rows, joint_summary = evaluate_joint_state_machine(
        alarm_feature_table=feature_tables[int(best_alarm["window_minutes"])],
        alarm_target=alarm_targets[int(best_alarm["window_minutes"])],
        alarm_row=best_alarm,
        warning_feature_table=feature_tables[int(best_warning["window_minutes"])],
        warning_target=warning_targets[int(best_warning["window_minutes"])],
        warning_row=best_warning,
    )

    summary = {
        "future_horizon_minutes": int(args.future_horizon),
        "alarm_best_row": best_alarm.to_dict(),
        "warning_best_row": best_warning.to_dict(),
        "joint_summary": joint_summary,
        "notes": {
            "design": "Current alarm and future warning are modeled independently and may use different DCS lookback windows.",
            "state_machine": "If current alarm is positive, output alarm; otherwise use the future warning head.",
        },
    }

    alarm_result.to_csv(artifact_dir / f"{args.output_prefix}_alarm_summary.csv", index=False, encoding="utf-8-sig")
    warning_result.to_csv(artifact_dir / f"{args.output_prefix}_warning_summary.csv", index=False, encoding="utf-8-sig")
    if alarm_feature_rankings:
        pd.concat(alarm_feature_rankings.values(), ignore_index=True).to_csv(
            artifact_dir / f"{args.output_prefix}_alarm_feature_rankings.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.concat(alarm_sensor_rankings.values(), ignore_index=True).to_csv(
            artifact_dir / f"{args.output_prefix}_alarm_sensor_rankings.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if warning_feature_rankings:
        pd.concat(warning_feature_rankings.values(), ignore_index=True).to_csv(
            artifact_dir / f"{args.output_prefix}_warning_feature_rankings.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.concat(warning_sensor_rankings.values(), ignore_index=True).to_csv(
            artifact_dir / f"{args.output_prefix}_warning_sensor_rankings.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if not merged_rows.empty:
        merged_rows.to_csv(artifact_dir / f"{args.output_prefix}_joint_rows.csv", index=False, encoding="utf-8-sig")
    with (artifact_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
