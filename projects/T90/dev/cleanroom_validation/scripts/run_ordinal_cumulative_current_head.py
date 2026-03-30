from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
CLEANROOM_DIR = THIS_DIR.parent
PROJECT_DIR = CLEANROOM_DIR.parents[1]
DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "ordinal_cumulative_current_head.yaml"

HARD_LABELS = ["below_spec", "in_spec", "above_spec"]
BUSINESS_LABELS = ["acceptable", "warning", "unacceptable"]


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = float(np.clip(self.probability, 0.0, 1.0))
        p0 = 1.0 - p1
        rows = len(X)
        return np.column_stack([np.full(rows, p0, dtype=float), np.full(rows, p1, dtype=float)])


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_name(name: object) -> str:
    text = str(name).strip()
    return text.replace("\xa0", " ")


def infer_columns(columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in columns:
        key = normalize_name(column)
        if "采样时间" in key:
            mapping["sample_time"] = column
        elif "样品名称" in key:
            mapping["sample_name"] = column
        elif "挥发份" in key and "在线" not in key:
            mapping["volatile_lab"] = column
        elif "在线监测挥发份" in key:
            mapping["volatile_online"] = column
        elif "90" in key.lower():
            mapping["t90"] = column
        elif "溴含量" in key:
            mapping["bromine"] = column
        elif "硬脂酸钙含量" in key:
            mapping["calcium_stearate"] = column
        elif key.startswith("钙含量"):
            mapping["calcium"] = column
        elif "稳定剂" in key:
            mapping["stabilizer"] = column
        elif "门尼粘度" in key:
            mapping["mooney"] = column
        elif "防老剂" in key:
            mapping["antioxidant"] = column
    return mapping


def discover_lims_path(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No LIMS workbook matched pattern: {pattern}")
    return matches[0]


def load_lims_data(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    sheets = pd.read_excel(path, sheet_name=None)
    raw = pd.concat(
        [sheet.assign(sheet_name=name) for name, sheet in sheets.items()],
        ignore_index=True,
    )
    raw.columns = [normalize_name(column) for column in raw.columns]
    column_map = infer_columns(list(raw.columns))
    rename_map = {source: target for target, source in column_map.items()}
    lims = raw.rename(columns=rename_map)

    if "sample_time" not in lims.columns or "t90" not in lims.columns:
        raise ValueError("Failed to infer required LIMS columns: sample_time / t90")

    lims["sample_time"] = pd.to_datetime(lims["sample_time"], errors="coerce")
    numeric_cols = [
        "volatile_lab",
        "volatile_online",
        "t90",
        "bromine",
        "calcium_stearate",
        "calcium",
        "stabilizer",
        "mooney",
        "antioxidant",
    ]
    for column in numeric_cols:
        if column in lims.columns:
            lims[column] = pd.to_numeric(lims[column], errors="coerce")

    aggregation: dict[str, str] = {}
    for column in ["sample_name", "sheet_name", *numeric_cols]:
        if column in lims.columns:
            aggregation[column] = "first"

    grouped = (
        lims.sort_values("sample_time")
        .groupby("sample_time", as_index=False)
        .agg(aggregation)
        .sort_values("sample_time")
        .reset_index(drop=True)
    )
    grouped["sheet_name"] = grouped["sheet_name"].fillna("unknown")
    return grouped, column_map


def load_dcs_data(path: Path) -> pd.DataFrame:
    dcs = pd.read_csv(path)
    dcs["time"] = pd.to_datetime(dcs["time"], errors="coerce")
    dcs = dcs.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    for column in [column for column in dcs.columns if column != "time"]:
        dcs[column] = pd.to_numeric(dcs[column], errors="coerce")
    return dcs


def summarize_window(window: pd.DataFrame, stats: list[str]) -> dict[str, float]:
    row: dict[str, float] = {}
    numeric_window = window.select_dtypes(include=["number"])
    if numeric_window.empty:
        return row

    for column in numeric_window.columns:
        values = numeric_window[column].dropna()
        if values.empty:
            continue
        if "last" in stats:
            row[f"{column}__last"] = float(values.iloc[-1])
        if "mean" in stats:
            row[f"{column}__mean"] = float(values.mean())
        if "std" in stats:
            row[f"{column}__std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        if "min" in stats:
            row[f"{column}__min"] = float(values.min())
        if "max" in stats:
            row[f"{column}__max"] = float(values.max())
        if "range" in stats:
            row[f"{column}__range"] = float(values.max() - values.min())
        if "delta" in stats:
            row[f"{column}__delta"] = float(values.iloc[-1] - values.iloc[0]) if len(values) > 1 else 0.0
    return row


def build_feature_rows(
    lims: pd.DataFrame,
    dcs: pd.DataFrame,
    *,
    lookback_minutes: int,
    stats: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labeled = lims.dropna(subset=["sample_time", "t90"]).copy()
    dcs = dcs.sort_values("time").reset_index(drop=True)
    time_values = dcs["time"].to_numpy()
    lookback = pd.Timedelta(minutes=lookback_minutes)

    for record in labeled.itertuples(index=False):
        sample_time = pd.Timestamp(record.sample_time)
        start_time = sample_time - lookback
        left = np.searchsorted(time_values, start_time.to_datetime64(), side="left")
        right = np.searchsorted(time_values, sample_time.to_datetime64(), side="right")
        window = dcs.iloc[left:right]
        if window.empty:
            continue

        row: dict[str, object] = {
            "sample_time": sample_time,
            "window_start": start_time,
            "window_end": sample_time,
            "window_row_count": int(len(window)),
            "sheet_name": getattr(record, "sheet_name", "unknown"),
            "sample_name": getattr(record, "sample_name", None),
            "t90": float(record.t90),
        }
        row.update(summarize_window(window, stats))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)


def format_threshold(value: float) -> str:
    return str(value).replace(".", "_")


def assign_labels(frame: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    labels = config["labels"]
    hard_low = float(labels["hard_three_class"]["low"])
    hard_high = float(labels["hard_three_class"]["high"])
    thresholds = [float(item) for item in labels["cumulative_thresholds"]]
    low_boundary = [float(item) for item in labels["boundary_low"]]
    high_boundary = [float(item) for item in labels["boundary_high"]]

    frame = frame.copy()
    frame["hard_label"] = np.select(
        [frame["t90"] < hard_low, frame["t90"] >= hard_high],
        ["below_spec", "above_spec"],
        default="in_spec",
    )
    frame["business_label"] = np.select(
        [
            frame["t90"] < thresholds[0],
            frame["t90"] >= thresholds[-1],
            frame["t90"] < thresholds[1],
            frame["t90"] >= thresholds[2],
        ],
        ["unacceptable", "unacceptable", "warning", "warning"],
        default="acceptable",
    )
    frame["is_unacceptable"] = frame["business_label"].eq("unacceptable").astype(int)
    frame["is_out_of_spec"] = frame["hard_label"].ne("in_spec").astype(int)
    frame["boundary_low_flag"] = frame["t90"].between(low_boundary[0], low_boundary[1], inclusive="left")
    frame["boundary_high_flag"] = frame["t90"].between(high_boundary[0], high_boundary[1], inclusive="left")
    frame["boundary_any_flag"] = frame["boundary_low_flag"] | frame["boundary_high_flag"]
    for threshold in thresholds:
        frame[f"target_lt_{format_threshold(threshold)}"] = frame["t90"].lt(threshold).astype(int)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "sample_time",
        "window_start",
        "window_end",
        "window_row_count",
        "sheet_name",
        "sample_name",
        "t90",
        "hard_label",
        "business_label",
        "is_unacceptable",
        "is_out_of_spec",
        "boundary_low_flag",
        "boundary_high_flag",
        "boundary_any_flag",
    }
    blocked.update({column for column in frame.columns if column.startswith("target_lt_")})
    return [column for column in frame.columns if column not in blocked]


def sensor_name(feature_name: str) -> str:
    return feature_name.split("__", 1)[0]


def drop_duplicate_feature_columns(frame: pd.DataFrame, columns: list[str], fill_value: float) -> tuple[list[str], list[str]]:
    seen: dict[tuple[float, ...], str] = {}
    kept: list[str] = []
    dropped: list[str] = []
    for column in columns:
        signature = tuple(pd.to_numeric(frame[column], errors="coerce").fillna(fill_value).round(10).tolist())
        if signature in seen:
            dropped.append(column)
            continue
        seen[signature] = column
        kept.append(column)
    return kept, dropped


def preclean_features(frame: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    features_cfg = config["features"]
    columns = feature_columns(frame)
    start_count = len(columns)

    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    missing_ratio = numeric.isna().mean()
    kept = missing_ratio[missing_ratio <= float(features_cfg["max_missing_ratio"])].index.tolist()
    dropped_missing = sorted(set(columns) - set(kept))

    numeric = numeric[kept]
    nunique = numeric.nunique(dropna=True)
    kept = nunique[nunique > 1].index.tolist()
    dropped_constant = sorted(set(numeric.columns) - set(kept))

    numeric = numeric[kept]
    kept_after_freeze: list[str] = []
    dropped_near_constant: list[str] = []
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            dropped_near_constant.append(column)
            continue
        rounded = series.round(10)
        dominance = float(rounded.value_counts(normalize=True, dropna=False).iloc[0])
        if dominance >= float(features_cfg["near_constant_ratio"]):
            dropped_near_constant.append(column)
        else:
            kept_after_freeze.append(column)

    numeric = numeric[kept_after_freeze]
    deduped, dropped_duplicates = drop_duplicate_feature_columns(
        numeric,
        list(numeric.columns),
        float(features_cfg["duplicate_fill_value"]),
    )
    cleaned = frame.copy()
    cleaned[deduped] = numeric[deduped]

    summary = {
        "feature_count_before": int(start_count),
        "feature_count_after": int(len(deduped)),
        "dropped_missing_count": int(len(dropped_missing)),
        "dropped_constant_count": int(len(dropped_constant)),
        "dropped_near_constant_count": int(len(dropped_near_constant)),
        "dropped_duplicate_count": int(len(dropped_duplicates)),
        "dropped_missing_examples": dropped_missing[:10],
        "dropped_constant_examples": dropped_constant[:10],
        "dropped_near_constant_examples": dropped_near_constant[:10],
        "dropped_duplicate_examples": dropped_duplicates[:10],
    }
    return cleaned, summary


def safe_corr_score(values: pd.Series, target: pd.Series) -> float:
    aligned = pd.concat([pd.to_numeric(values, errors="coerce"), pd.to_numeric(target, errors="coerce")], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0
    if aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return 0.0
    score = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    if pd.isna(score):
        return 0.0
    return float(abs(score))


def select_topk_sensors(train_frame: pd.DataFrame, columns: list[str], topk: int) -> tuple[list[str], dict[str, float]]:
    target = train_frame["is_out_of_spec"].astype(int)
    sensor_scores: dict[str, float] = {}
    for column in columns:
        score = safe_corr_score(train_frame[column], target)
        sensor = sensor_name(column)
        sensor_scores[sensor] = max(sensor_scores.get(sensor, 0.0), score)
    ranked = sorted(sensor_scores.items(), key=lambda item: (-item[1], item[0]))
    selected_sensors = [name for name, _score in ranked[:topk]]
    return selected_sensors, dict(ranked[:topk])


def selected_feature_columns(columns: list[str], sensors: list[str]) -> list[str]:
    sensor_set = set(sensors)
    return sorted([column for column in columns if sensor_name(column) in sensor_set])


def build_logistic_pipeline(max_iter: int, class_weight: str, multiclass: bool) -> Pipeline:
    kwargs = {"max_iter": max_iter, "class_weight": class_weight}
    model = LogisticRegression(**kwargs)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def fit_hard_model(train_frame: pd.DataFrame, features: list[str], config: dict[str, object]) -> Pipeline:
    model_cfg = config["models"]
    model = build_logistic_pipeline(
        max_iter=int(model_cfg["logistic_max_iter"]),
        class_weight=str(model_cfg["class_weight"]),
        multiclass=True,
    )
    model.fit(train_frame[features], train_frame["hard_label"])
    return model


def fit_threshold_model(train_frame: pd.DataFrame, features: list[str], threshold_key: str, config: dict[str, object]) -> Pipeline | ConstantProbabilityModel:
    target = train_frame[threshold_key].astype(int)
    if target.nunique() < 2:
        return ConstantProbabilityModel(probability=float(target.iloc[0]))
    model_cfg = config["models"]
    model = build_logistic_pipeline(
        max_iter=int(model_cfg["logistic_max_iter"]),
        class_weight=str(model_cfg["class_weight"]),
        multiclass=False,
    )
    model.fit(train_frame[features], target)
    return model


def predict_threshold_probability(model: Pipeline | ConstantProbabilityModel, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame)[:, 1]


def calibration_error(actual: pd.Series, probability: pd.Series, bins: int = 10) -> float:
    frame = pd.DataFrame({"actual": actual.astype(float), "probability": probability.astype(float)}).dropna()
    if frame.empty:
        return math.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right >= 1.0:
            bucket = frame[(frame["probability"] >= left) & (frame["probability"] <= right)]
        else:
            bucket = frame[(frame["probability"] >= left) & (frame["probability"] < right)]
        if bucket.empty:
            continue
        total += abs(bucket["actual"].mean() - bucket["probability"].mean()) * (len(bucket) / len(frame))
    return float(total)


def hard_business_probabilities(hard_proba: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=hard_proba.index)
    result["acceptable"] = hard_proba["in_spec"]
    result["warning"] = 0.0
    result["unacceptable"] = hard_proba["below_spec"] + hard_proba["above_spec"]
    return result


def cumulative_to_interval_probabilities(cumulative: pd.DataFrame) -> pd.DataFrame:
    thresholds = list(cumulative.columns)
    intervals = pd.DataFrame(index=cumulative.index)
    intervals["lt_8_0"] = cumulative[thresholds[0]]
    intervals["between_8_0_8_2"] = cumulative[thresholds[1]] - cumulative[thresholds[0]]
    intervals["between_8_2_8_7"] = cumulative[thresholds[2]] - cumulative[thresholds[1]]
    intervals["between_8_7_8_9"] = cumulative[thresholds[3]] - cumulative[thresholds[2]]
    intervals["ge_8_9"] = 1.0 - cumulative[thresholds[3]]
    return intervals


def apply_monotonic_correction(cumulative: pd.DataFrame) -> pd.DataFrame:
    corrected = cumulative.copy()
    ordered_columns = list(corrected.columns)
    running = corrected[ordered_columns[0]].to_numpy(dtype=float)
    corrected[ordered_columns[0]] = np.clip(running, 0.0, 1.0)
    for column in ordered_columns[1:]:
        running = np.maximum(running, corrected[column].to_numpy(dtype=float))
        corrected[column] = np.clip(running, 0.0, 1.0)
    return corrected


def clip_and_renormalize(probabilities: pd.DataFrame) -> pd.DataFrame:
    clipped = probabilities.clip(lower=0.0)
    row_sum = clipped.sum(axis=1)
    safe_sum = row_sum.replace(0.0, 1.0)
    return clipped.div(safe_sum, axis=0)


def business_probabilities_from_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=intervals.index)
    result["acceptable"] = intervals["between_8_2_8_7"]
    result["warning"] = intervals["between_8_0_8_2"] + intervals["between_8_7_8_9"]
    result["unacceptable"] = intervals["lt_8_0"] + intervals["ge_8_9"]
    return result


def probability_argmax(probabilities: pd.DataFrame, labels: list[str]) -> pd.Series:
    indices = probabilities[labels].to_numpy().argmax(axis=1)
    return pd.Series([labels[index] for index in indices], index=probabilities.index)


def multiclass_metrics(actual: pd.Series, predicted: pd.Series, labels: list[str]) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
    }


def binary_ap(actual: pd.Series, score: pd.Series) -> float:
    frame = pd.DataFrame({"actual": actual.astype(int), "score": score.astype(float)}).dropna()
    if frame.empty or frame["actual"].nunique() < 2:
        return math.nan
    return float(average_precision_score(frame["actual"], frame["score"]))


def threshold_metric_table(frame: pd.DataFrame, thresholds: list[float]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for threshold in thresholds:
        key = format_threshold(threshold)
        actual = frame[f"target_lt_{key}"]
        score = frame[f"cum_prob_lt_{key}"]
        valid = score.notna()
        if valid.sum() == 0 or actual[valid].nunique() < 2:
            results[f"lt_{threshold}"] = {"brier_score": math.nan, "calibration_error": math.nan}
            continue
        results[f"lt_{threshold}"] = {
            "brier_score": float(brier_score_loss(actual[valid], score[valid])),
            "calibration_error": calibration_error(actual[valid], score[valid]),
        }
    return results


def summarize_boundary_subset(frame: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    subset = frame.loc[mask].copy()
    if subset.empty:
        return {"samples": 0}
    return {
        "samples": int(len(subset)),
        "t90_min": float(subset["t90"].min()),
        "t90_max": float(subset["t90"].max()),
        "hard_baseline": multiclass_metrics(subset["business_label"], subset["hard_business_pred"], BUSINESS_LABELS),
        "cumulative_treatment": multiclass_metrics(subset["business_label"], subset["cum_business_pred"], BUSINESS_LABELS),
        "hard_unacceptable_ap": binary_ap(subset["is_unacceptable"], subset["hard_unacceptable_prob"]),
        "cumulative_unacceptable_ap": binary_ap(subset["is_unacceptable"], subset["cum_unacceptable_prob"]),
    }


def monotonicity_summary(frame: pd.DataFrame, thresholds: list[float]) -> dict[str, object]:
    columns = [f"cum_prob_lt_{format_threshold(threshold)}" for threshold in thresholds]
    pairwise: dict[str, float] = {}
    any_violation = pd.Series(False, index=frame.index)
    for left, right in zip(columns[:-1], columns[1:]):
        violation = frame[left] > frame[right]
        pairwise[f"{left}->{right}"] = float(violation.mean())
        any_violation = any_violation | violation
    return {
        "pairwise_violation_rate": pairwise,
        "any_violation_rate": float(any_violation.mean()),
        "samples_with_any_violation": int(any_violation.sum()),
    }


def boundary_overconfidence_stats(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    acceptable_prob_col: str,
    unacceptable_prob_col: str,
    confidence_threshold: float = 0.60,
) -> dict[str, object]:
    subset = frame.loc[frame["boundary_any_flag"]].copy()
    if subset.empty:
        return {"samples": 0}
    non_warning = subset[pred_col] != "warning"
    extreme_conf = subset[[acceptable_prob_col, unacceptable_prob_col]].max(axis=1) >= confidence_threshold
    flagged = non_warning & extreme_conf
    return {
        "samples": int(len(subset)),
        "confidence_threshold": float(confidence_threshold),
        "non_warning_rate": float(non_warning.mean()),
        "high_confidence_non_warning_rate": float(flagged.mean()),
        "high_confidence_non_warning_count": int(flagged.sum()),
    }


def reframed_step1_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | dict[str, object]]]:
    hard = {
        "core_qualified_average_precision": binary_ap(
            frame["business_label"].eq("acceptable").astype(int),
            frame["hard_business_prob_acceptable"],
        ),
        "boundary_warning_average_precision": binary_ap(
            frame["business_label"].eq("warning").astype(int),
            frame["hard_business_prob_warning"],
        ),
        "clearly_unacceptable_average_precision": binary_ap(
            frame["business_label"].eq("unacceptable").astype(int),
            frame["hard_business_prob_unacceptable"],
        ),
        "boundary_overconfidence": boundary_overconfidence_stats(
            frame,
            pred_col="hard_business_pred",
            acceptable_prob_col="hard_business_prob_acceptable",
            unacceptable_prob_col="hard_business_prob_unacceptable",
        ),
    }
    cumulative = {
        "core_qualified_average_precision": binary_ap(
            frame["business_label"].eq("acceptable").astype(int),
            frame["cum_business_prob_acceptable"],
        ),
        "boundary_warning_average_precision": binary_ap(
            frame["business_label"].eq("warning").astype(int),
            frame["cum_business_prob_warning"],
        ),
        "clearly_unacceptable_average_precision": binary_ap(
            frame["business_label"].eq("unacceptable").astype(int),
            frame["cum_business_prob_unacceptable"],
        ),
        "boundary_overconfidence": boundary_overconfidence_stats(
            frame,
            pred_col="cum_business_pred",
            acceptable_prob_col="cum_business_prob_acceptable",
            unacceptable_prob_col="cum_business_prob_unacceptable",
        ),
    }
    return {"hard_baseline": hard, "cumulative_treatment": cumulative}


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_audit_report(
    path: Path,
    *,
    config: dict[str, object],
    data_summary: dict[str, object],
    preclean_summary: dict[str, object],
    metrics_summary: dict[str, object],
) -> None:
    thresholds = config["labels"]["cumulative_thresholds"]
    lines = [
        "# Ordinal / Cumulative Current-Head Audit",
        "",
        "## Scope",
        "",
        "- Baseline: hard three-class current-head classification",
        "- Treatment: independent cumulative threshold logistic models",
        "- Main feature family: simple causal window statistics only",
        "- EWMA / PH / stage-aware logic: not used in this run",
        "",
        "## Thresholds",
        "",
        f"- cumulative thresholds: {thresholds}",
        f"- hard baseline low/high: {config['labels']['hard_three_class']['low']} / {config['labels']['hard_three_class']['high']}",
        "",
        "## Data audit",
        "",
        f"- LIMS labeled samples before alignment: {data_summary['lims_t90_nonnull']}",
        f"- aligned feature rows: {data_summary['aligned_rows']}",
        f"- DCS sensors in raw file: {data_summary['dcs_sensor_count']}",
        f"- average window rows: {data_summary['window_row_count_mean']:.2f}",
        "",
        "## Feature pre-cleaning",
        "",
        f"- features before pre-clean: {preclean_summary['feature_count_before']}",
        f"- features after pre-clean: {preclean_summary['feature_count_after']}",
        f"- dropped for missingness: {preclean_summary['dropped_missing_count']}",
        f"- dropped as constant: {preclean_summary['dropped_constant_count']}",
        f"- dropped as near-constant: {preclean_summary['dropped_near_constant_count']}",
        f"- dropped as duplicates: {preclean_summary['dropped_duplicate_count']}",
        "",
        "## Leakage / screening audit",
        "",
        "- supervised sensor screening was fit inside each training fold only",
        "- both compared formulations shared the same screened sensor set within each fold",
        "- no historical selected sensor list or tuned lag/window recommendation was inherited",
        "",
        "## Monotonicity audit",
        "",
        "- cumulative probabilities were modeled independently per threshold",
        f"- monotonic correction applied: {config['models']['apply_monotonic_correction']}",
        f"- interval negative handling for decision collapse: {config['models']['interval_negative_handling']}",
        f"- any-violation rate: {metrics_summary['monotonicity']['any_violation_rate']:.4f}",
        "",
        "## First-run outcome",
        "",
        f"- hard baseline macro_f1: {metrics_summary['overall_metrics']['hard_baseline']['macro_f1']:.4f}",
        f"- hard baseline balanced_accuracy: {metrics_summary['overall_metrics']['hard_baseline']['balanced_accuracy']:.4f}",
        f"- cumulative treatment macro_f1: {metrics_summary['overall_metrics']['cumulative_treatment']['macro_f1']:.4f}",
        f"- cumulative treatment balanced_accuracy: {metrics_summary['overall_metrics']['cumulative_treatment']['balanced_accuracy']:.4f}",
        "",
        "## Reframed step-1 view",
        "",
        f"- hard core-qualified AP: {metrics_summary['reframed_step1_metrics']['hard_baseline']['core_qualified_average_precision']:.4f}",
        f"- cumulative core-qualified AP: {metrics_summary['reframed_step1_metrics']['cumulative_treatment']['core_qualified_average_precision']:.4f}",
        f"- hard boundary-warning AP: {metrics_summary['reframed_step1_metrics']['hard_baseline']['boundary_warning_average_precision']:.4f}",
        f"- cumulative boundary-warning AP: {metrics_summary['reframed_step1_metrics']['cumulative_treatment']['boundary_warning_average_precision']:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first cleanroom ordinal/cumulative current-head experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    data_dir = PROJECT_DIR / "data"
    outputs_cfg = config["outputs"]
    artifacts_dir = CLEANROOM_DIR / str(outputs_cfg["artifacts_dir"])
    reports_dir = CLEANROOM_DIR / str(outputs_cfg["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dcs_path = PROJECT_DIR / str(config["data"]["dcs_path"])
    lims_path = discover_lims_path(data_dir, str(config["data"]["lims_glob"]))
    lims, column_map = load_lims_data(lims_path)
    dcs = load_dcs_data(dcs_path)

    feature_rows = build_feature_rows(
        lims,
        dcs,
        lookback_minutes=int(config["features"]["lookback_minutes"]),
        stats=list(config["features"]["window_statistics"]),
    )
    feature_rows = assign_labels(feature_rows, config)
    cleaned_rows, preclean_summary = preclean_features(feature_rows, config)
    features = feature_columns(cleaned_rows)

    n_rows = len(cleaned_rows)
    requested_splits = int(config["validation"]["n_splits"])
    min_splits = int(config["validation"]["min_splits"])
    n_splits = requested_splits if n_rows >= 1000 else max(min_splits, min(requested_splits, 4))
    if n_splits < min_splits:
        raise ValueError(f"Not enough aligned rows for TimeSeriesSplit with min_splits={min_splits}.")

    results = cleaned_rows[
        [
            "sample_time",
            "window_start",
            "window_end",
            "window_row_count",
            "sheet_name",
            "sample_name",
            "t90",
            "hard_label",
            "business_label",
            "is_unacceptable",
            "is_out_of_spec",
            "boundary_low_flag",
            "boundary_high_flag",
            "boundary_any_flag",
            *[column for column in cleaned_rows.columns if column.startswith("target_lt_")],
        ]
    ].copy()
    results["fold"] = np.nan
    results["selected_sensor_count"] = np.nan
    results["selected_feature_count"] = np.nan

    thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]
    for threshold in thresholds:
        key = format_threshold(threshold)
        results[f"cum_prob_lt_{key}"] = np.nan
    for label in HARD_LABELS:
        results[f"hard_prob_{label}"] = np.nan
    for label in BUSINESS_LABELS:
        results[f"hard_business_prob_{label}"] = np.nan
        results[f"cum_business_prob_{label}"] = np.nan
    for interval_label in ["lt_8_0", "between_8_0_8_2", "between_8_2_8_7", "between_8_7_8_9", "ge_8_9"]:
        results[f"cum_interval_{interval_label}"] = np.nan

    per_fold: list[dict[str, object]] = []
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold_index, (train_index, test_index) in enumerate(tscv.split(cleaned_rows), start=1):
        train_frame = cleaned_rows.iloc[train_index].copy()
        test_frame = cleaned_rows.iloc[test_index].copy()

        selected_sensors, sensor_scores = select_topk_sensors(
            train_frame,
            features,
            topk=int(config["screening"]["topk_sensors"]),
        )
        selected_features = selected_feature_columns(features, selected_sensors)

        hard_model = fit_hard_model(train_frame, selected_features, config)
        hard_proba = pd.DataFrame(
            hard_model.predict_proba(test_frame[selected_features]),
            index=test_frame.index,
            columns=list(hard_model.named_steps["model"].classes_),
        ).reindex(columns=HARD_LABELS, fill_value=0.0)

        cumulative_frame = pd.DataFrame(index=test_frame.index)
        for threshold in thresholds:
            key = format_threshold(threshold)
            threshold_model = fit_threshold_model(train_frame, selected_features, f"target_lt_{key}", config)
            cumulative_frame[f"lt_{key}"] = predict_threshold_probability(threshold_model, test_frame[selected_features])

        if bool(config["models"]["apply_monotonic_correction"]):
            cumulative_frame = apply_monotonic_correction(cumulative_frame)
        intervals = clip_and_renormalize(cumulative_to_interval_probabilities(cumulative_frame))
        hard_business = hard_business_probabilities(hard_proba)
        cumulative_business = business_probabilities_from_intervals(intervals)

        results.loc[test_frame.index, "fold"] = fold_index
        results.loc[test_frame.index, "selected_sensor_count"] = len(selected_sensors)
        results.loc[test_frame.index, "selected_feature_count"] = len(selected_features)
        for label in HARD_LABELS:
            results.loc[test_frame.index, f"hard_prob_{label}"] = hard_proba[label]
        for threshold in thresholds:
            key = format_threshold(threshold)
            results.loc[test_frame.index, f"cum_prob_lt_{key}"] = cumulative_frame[f"lt_{key}"]
        for label in BUSINESS_LABELS:
            results.loc[test_frame.index, f"hard_business_prob_{label}"] = hard_business[label]
            results.loc[test_frame.index, f"cum_business_prob_{label}"] = cumulative_business[label]
        for interval_label in intervals.columns:
            results.loc[test_frame.index, f"cum_interval_{interval_label}"] = intervals[interval_label]

        fold_eval = pd.DataFrame(index=test_frame.index)
        fold_eval["business_label"] = test_frame["business_label"]
        fold_eval["is_unacceptable"] = test_frame["is_unacceptable"]
        fold_eval["hard_business_pred"] = probability_argmax(hard_business, BUSINESS_LABELS)
        fold_eval["cum_business_pred"] = probability_argmax(cumulative_business, BUSINESS_LABELS)
        fold_eval["hard_unacceptable_prob"] = hard_business["unacceptable"]
        fold_eval["cum_unacceptable_prob"] = cumulative_business["unacceptable"]
        per_fold.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "selected_sensors": selected_sensors,
                "selected_sensor_scores": sensor_scores,
                "selected_feature_count": int(len(selected_features)),
                "hard_baseline": multiclass_metrics(fold_eval["business_label"], fold_eval["hard_business_pred"], BUSINESS_LABELS),
                "cumulative_treatment": multiclass_metrics(fold_eval["business_label"], fold_eval["cum_business_pred"], BUSINESS_LABELS),
                "hard_unacceptable_ap": binary_ap(fold_eval["is_unacceptable"], fold_eval["hard_unacceptable_prob"]),
                "cumulative_unacceptable_ap": binary_ap(fold_eval["is_unacceptable"], fold_eval["cum_unacceptable_prob"]),
            }
        )

    results["hard_business_pred"] = probability_argmax(
        results[[f"hard_business_prob_{label}" for label in BUSINESS_LABELS]].rename(
            columns={f"hard_business_prob_{label}": label for label in BUSINESS_LABELS}
        ),
        BUSINESS_LABELS,
    )
    results["cum_business_pred"] = probability_argmax(
        results[[f"cum_business_prob_{label}" for label in BUSINESS_LABELS]].rename(
            columns={f"cum_business_prob_{label}": label for label in BUSINESS_LABELS}
        ),
        BUSINESS_LABELS,
    )
    results["hard_unacceptable_prob"] = results["hard_business_prob_unacceptable"]
    results["cum_unacceptable_prob"] = results["cum_business_prob_unacceptable"]

    overall_metrics = {
        "hard_baseline": multiclass_metrics(results["business_label"], results["hard_business_pred"], BUSINESS_LABELS),
        "cumulative_treatment": multiclass_metrics(results["business_label"], results["cum_business_pred"], BUSINESS_LABELS),
    }
    overall_metrics["hard_baseline"]["unacceptable_average_precision"] = binary_ap(results["is_unacceptable"], results["hard_unacceptable_prob"])
    overall_metrics["cumulative_treatment"]["unacceptable_average_precision"] = binary_ap(results["is_unacceptable"], results["cum_unacceptable_prob"])

    data_summary = {
        "lims_rows_grouped": int(len(lims)),
        "lims_t90_nonnull": int(lims["t90"].notna().sum()),
        "aligned_rows": int(len(results)),
        "dcs_rows": int(len(dcs)),
        "dcs_sensor_count": int(len([column for column in dcs.columns if column != "time"])),
        "window_row_count_mean": float(results["window_row_count"].mean()),
        "window_row_count_min": int(results["window_row_count"].min()),
        "window_row_count_max": int(results["window_row_count"].max()),
        "hard_label_distribution": results["hard_label"].value_counts().to_dict(),
        "business_label_distribution": results["business_label"].value_counts().to_dict(),
        "boundary_low_samples": int(results["boundary_low_flag"].sum()),
        "boundary_high_samples": int(results["boundary_high_flag"].sum()),
        "boundary_any_samples": int(results["boundary_any_flag"].sum()),
        "column_map_keys": sorted(column_map.keys()),
        "n_splits_used": int(n_splits),
    }
    metrics_summary = {
        "overall_metrics": overall_metrics,
        "reframed_step1_metrics": reframed_step1_metrics(results),
        "threshold_metrics": threshold_metric_table(results, thresholds),
        "monotonicity": monotonicity_summary(results, thresholds),
        "boundary_diagnostics": {
            "boundary_low": summarize_boundary_subset(results, results["boundary_low_flag"]),
            "boundary_high": summarize_boundary_subset(results, results["boundary_high_flag"]),
            "boundary_any": summarize_boundary_subset(results, results["boundary_any_flag"]),
        },
        "per_fold": per_fold,
    }

    feature_rows_path = artifacts_dir / "ordinal_cumulative_feature_rows.csv"
    results_path = artifacts_dir / "ordinal_cumulative_results.csv"
    summary_path = artifacts_dir / "ordinal_cumulative_summary.json"
    audit_path = reports_dir / "ordinal_cumulative_current_head_audit.md"

    summary = {
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "project_dir": str(PROJECT_DIR),
        "cleanroom_dir": str(CLEANROOM_DIR),
        "data_paths": {"dcs_path": str(dcs_path), "lims_path": str(lims_path)},
        "data_summary": data_summary,
        "preclean_summary": preclean_summary,
        "metrics_summary": metrics_summary,
        "artifacts": {
            "feature_rows_csv": str(feature_rows_path),
            "results_csv": str(results_path),
            "summary_json": str(summary_path),
            "audit_md": str(audit_path),
        },
    }

    cleaned_rows.to_csv(feature_rows_path, index=False, encoding="utf-8-sig")
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_audit_report(
        audit_path,
        config=config,
        data_summary=data_summary,
        preclean_summary=preclean_summary,
        metrics_summary=metrics_summary,
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
