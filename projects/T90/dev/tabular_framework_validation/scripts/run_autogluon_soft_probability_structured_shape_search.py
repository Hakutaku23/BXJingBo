from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_stage2_feature_engineering import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    resolve_path,
    select_features_fold,
)
from run_autogluon_stage2_soft_probability import (
    soft_probability_metrics,
)
from run_autogluon_stage2_soft_probability_x_enrichment import add_range_position_features
from run_autogluon_soft_probability_weak_compression_search import (
    _prepare_soft_labeled_samples,
    build_variant_snapshot as build_weak_compression_variant_snapshot,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import summarize_numeric_window


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_soft_probability_structured_shape_search.yaml"
EPS = 1e-6


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _sensor_prefixes(frame: pd.DataFrame) -> list[str]:
    return sorted({column.split("__", 1)[0] for column in frame.columns if "__" in column})


def _iter_sensor_windows(
    labeled_samples: pd.DataFrame,
    dcs: pd.DataFrame,
    lookback: int,
    min_points_per_window: int,
):
    dcs_times = dcs["time"].to_numpy(dtype="datetime64[ns]")
    numeric_columns = [column for column in dcs.columns if column != "time"]
    for record in labeled_samples.itertuples(index=False):
        decision_time = pd.Timestamp(record.sample_time)
        window_end = decision_time
        window_start = window_end - pd.Timedelta(minutes=lookback)
        left = np.searchsorted(dcs_times, window_start.to_datetime64(), side="left")
        right = np.searchsorted(dcs_times, window_end.to_datetime64(), side="right")
        window = dcs.iloc[left:right]
        if len(window) < min_points_per_window:
            continue
        yield record, window, numeric_columns


def _nanquantile(values: np.ndarray, q: float) -> float:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.nanquantile(valid, q))


def _structured_shape_features(values: np.ndarray, quantile_levels: list[float]) -> dict[str, float]:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return {}

    count = valid.size
    range_value = float(np.nanmax(valid) - np.nanmin(valid))
    denom = range_value + EPS

    q_values = {q: _nanquantile(valid, q) for q in quantile_levels}
    head = valid[: max(1, count // 3)]
    middle = valid[max(0, count // 3): max(1, (2 * count) // 3)]
    tail = valid[max(0, (2 * count) // 3):]
    head_mean = float(np.nanmean(head)) if head.size else float("nan")
    middle_mean = float(np.nanmean(middle)) if middle.size else float("nan")
    tail_mean = float(np.nanmean(tail)) if tail.size else float("nan")

    argmax_pos = float(np.nanargmax(valid) / max(1, count - 1))
    argmin_pos = float(np.nanargmin(valid) / max(1, count - 1))
    upper_band = q_values.get(0.75, float("nan"))
    lower_band = q_values.get(0.25, float("nan"))
    median = q_values.get(0.50, float("nan"))

    features = {
        "q10": q_values.get(0.10, float("nan")),
        "q25": q_values.get(0.25, float("nan")),
        "q50": median,
        "q75": q_values.get(0.75, float("nan")),
        "q90": q_values.get(0.90, float("nan")),
        "iqr": (q_values.get(0.75, float("nan")) - q_values.get(0.25, float("nan"))),
        "p90_p10_span": (q_values.get(0.90, float("nan")) - q_values.get(0.10, float("nan"))),
        "head_mean": head_mean,
        "middle_mean": middle_mean,
        "tail_mean": tail_mean,
        "tail_minus_head_over_range": (tail_mean - head_mean) / denom,
        "tail_minus_middle_over_range": (tail_mean - middle_mean) / denom,
        "middle_minus_head_over_range": (middle_mean - head_mean) / denom,
        "curvature_over_range": (head_mean - (2.0 * middle_mean) + tail_mean) / denom,
        "argmax_position": argmax_pos,
        "argmin_position": argmin_pos,
        "upper_band_ratio": float(np.mean(valid >= upper_band)) if not np.isnan(upper_band) else float("nan"),
        "lower_band_ratio": float(np.mean(valid <= lower_band)) if not np.isnan(lower_band) else float("nan"),
        "above_median_ratio": float(np.mean(valid >= median)) if not np.isnan(median) else float("nan"),
    }
    return features


def build_structured_shape_snapshot(
    config_path: Path,
    config: dict[str, Any],
    variant_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if variant_name == "current_whole_window_ref":
        return build_weak_compression_variant_snapshot(config_path, config, "current_whole_window_ref")

    base_snapshot, base_audit = build_weak_compression_variant_snapshot(config_path, config, "current_whole_window_ref")
    labeled_samples, dcs = _prepare_soft_labeled_samples(config_path, config)
    lookback = int(config["snapshot"]["lookback_minutes"])
    min_points_per_window = int(config["snapshot"]["min_points_per_window"])
    quantile_levels = [float(q) for q in config["structured_shape"]["quantile_levels"]]
    rows: list[dict[str, Any]] = []

    for record, window, numeric_columns in _iter_sensor_windows(
        labeled_samples=labeled_samples,
        dcs=dcs,
        lookback=lookback,
        min_points_per_window=min_points_per_window,
    ):
        row = {"sample_time": pd.Timestamp(record.sample_time)}

        for sensor in numeric_columns:
            values = pd.to_numeric(window[sensor], errors="coerce").to_numpy(dtype=float)
            features = _structured_shape_features(values, quantile_levels)
            if not features:
                continue

            include_quantile = variant_name in {"shape_quantile_profile", "shape_combined_profile"}
            include_phase = variant_name in {"shape_phase_profile", "shape_combined_profile"}
            for feature_name, feature_value in features.items():
                if feature_name in {"q10", "q25", "q50", "q75", "q90", "iqr", "p90_p10_span"} and not include_quantile:
                    continue
                if feature_name not in {"q10", "q25", "q50", "q75", "q90", "iqr", "p90_p10_span"} and not include_phase:
                    continue
                row[f"{sensor}__shape_{feature_name}"] = float(feature_value) if pd.notna(feature_value) else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame(), {}

    shape_frame = pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)
    snapshot = base_snapshot.merge(shape_frame, on="sample_time", how="inner")
    feature_columns = [column for column in snapshot.columns if "__" in column]
    dropped_all_nan = [column for column in feature_columns if snapshot[column].isna().all()]
    if dropped_all_nan:
        snapshot = snapshot.drop(columns=dropped_all_nan)
    feature_columns = [column for column in snapshot.columns if "__" in column]
    dropped_constant = []
    for column in feature_columns:
        valid = snapshot[column].dropna()
        if not valid.empty and valid.nunique() <= 1:
            dropped_constant.append(column)
    if dropped_constant:
        snapshot = snapshot.drop(columns=dropped_constant)

    audit = {
        **base_audit,
        "lookback_minutes": lookback,
        "snapshot_samples": int(len(snapshot)),
        "soft_probability_label_name": str(config["label_fuzziness"]["target_name"]),
        "soft_probability_rule": str(config["label_fuzziness"]["rule"]),
        "boundary_softness": float(config["label_fuzziness"]["boundary_softness"]),
        "representation_family": f"whole_window_range_position_plus_{variant_name}",
        "feature_count_after_variant_build": int(len([column for column in snapshot.columns if "__" in column])),
        "dropped_all_nan_after_variant": int(len(dropped_all_nan)),
        "dropped_constant_after_variant": int(len(dropped_constant)),
    }
    return snapshot, audit


def run_variant(config_path: Path, config: dict[str, Any], variant_name: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    snapshot, audit = build_structured_shape_snapshot(config_path, config, variant_name)
    feature_columns = [column for column in snapshot.columns if "__" in column]
    label = str(config["label_fuzziness"]["target_name"])
    top_k = int(config["selection"]["soft_probability_top_k"])
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    if artifact_dir is None:
        raise ValueError("artifact_dir must be configured.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, Any]] = []
    selected_fold1: list[str] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
        train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
        test = snapshot.iloc[test_idx].copy().reset_index(drop=True)
        selected_features, _ = select_features_fold(
            train_x=train[feature_columns],
            train_y=train[label],
            task_type="regression",
            top_k=top_k,
        )
        if fold_idx == 1:
            selected_fold1 = list(selected_features)

        train_df = train[selected_features + [label]].copy()
        test_df = test[selected_features + [label]].copy()

        baseline = make_regression_baseline()
        baseline.fit(train_df[selected_features], train_df[label].to_numpy(dtype=float))
        baseline_pred = baseline.predict(test_df[selected_features]).astype(float)
        baseline_metrics = soft_probability_metrics(
            test_df[label].to_numpy(dtype=float),
            baseline_pred,
            test["is_out_of_spec"].to_numpy(dtype=int),
        )

        model_path = artifact_dir / f"ag_soft_shape_{variant_name}_{run_id}_fold{fold_idx}"
        framework_pred, model_best = fit_autogluon_fold(
            train_df=train_df,
            test_df=test_df,
            label=label,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )
        framework_metrics = soft_probability_metrics(
            test_df[label].to_numpy(dtype=float),
            framework_pred.astype(float),
            test["is_out_of_spec"].to_numpy(dtype=int),
        )

        rows.append(
            {
                "variant_name": variant_name,
                "framework": "simple_baseline_stage2_soft_probability",
                "fold": int(fold_idx),
                "feature_count_selected": int(len(selected_features)),
                **baseline_metrics,
            }
        )
        rows.append(
            {
                "variant_name": variant_name,
                "framework": "autogluon_stage2_soft_probability",
                "fold": int(fold_idx),
                "feature_count_selected": int(len(selected_features)),
                "autogluon_model_best": model_best,
                **framework_metrics,
            }
        )

    results = pd.DataFrame(rows)
    baseline_rows = results[results["framework"] == "simple_baseline_stage2_soft_probability"]
    ag_rows = results[results["framework"] == "autogluon_stage2_soft_probability"]
    summary = {
        "variant_name": variant_name,
        **audit,
        "representation_family": audit.get("representation_family", variant_name),
        "sample_count": int(len(snapshot)),
        "raw_feature_count": int(len(feature_columns)),
        "selected_feature_count": int(_safe_mean(list(results["feature_count_selected"]))),
        "selected_features_fold1": selected_fold1,
        "baseline_mean_soft_mae": float(baseline_rows["soft_mae"].mean()),
        "autogluon_mean_soft_mae": float(ag_rows["soft_mae"].mean()),
        "baseline_mean_soft_brier": float(baseline_rows["soft_brier"].mean()),
        "autogluon_mean_soft_brier": float(ag_rows["soft_brier"].mean()),
        "autogluon_mean_hard_out_ap_diagnostic": float(ag_rows["hard_out_ap_diagnostic"].mean()),
    }
    return results, summary, snapshot


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    ranked = sorted(rows, key=lambda row: row["autogluon_mean_soft_brier"])
    best = ranked[0] if ranked else {}
    lines = [
        "# Soft Probability Structured Shape Search",
        "",
        "## Scope",
        "",
        "- Task: soft probability branch only",
        "- Keep the validated fuzzy Y design fixed",
        "- Compare the current whole-window range-position reference against structured shape feature families",
        "",
        "## Ranked Summary Rows",
        "",
        json.dumps(ranked, ensure_ascii=False, indent=2),
        "",
        "## Best Variant By AutoGluon Soft Brier",
        "",
        json.dumps(best, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured shape feature search for the AutoGluon soft probability branch.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    report_dir = resolve_path(config_path.parent, config["paths"]["report_dir"])
    if artifact_dir is None or report_dir is None:
        raise ValueError("Both artifact_dir and report_dir must be configured.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []
    feature_catalog_rows: list[dict[str, Any]] = []

    for variant_name in config["structured_shape"]["variants"]:
        results, summary, snapshot = run_variant(config_path, config, str(variant_name))
        detail_rows.append(results)
        summary_rows.append(summary)
        feature_catalog_rows.append(
            {
                "variant_name": str(variant_name),
                "representation_family": summary["representation_family"],
                "sample_count": summary["sample_count"],
                "raw_feature_count": summary["raw_feature_count"],
                "selected_feature_count": summary["selected_feature_count"],
                "autogluon_mean_soft_brier": summary["autogluon_mean_soft_brier"],
                "autogluon_mean_soft_mae": summary["autogluon_mean_soft_mae"],
            }
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values("autogluon_mean_soft_brier").reset_index(drop=True)
    if not summary_frame.empty:
        current_ref = float(
            summary_frame.loc[summary_frame["variant_name"] == "current_whole_window_ref", "autogluon_mean_soft_brier"].iloc[0]
        )
        summary_frame["current_reference_soft_brier"] = current_ref
        summary_frame["delta_vs_current_reference"] = summary_frame["autogluon_mean_soft_brier"] - current_ref
        summary_frame["beats_current_reference"] = summary_frame["delta_vs_current_reference"] < 0.0

    detail_frame = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    feature_catalog = pd.DataFrame(feature_catalog_rows)

    (artifact_dir / "soft_probability_structured_shape_search_summary_rows.csv").write_text(
        summary_frame.to_csv(index=False), encoding="utf-8"
    )
    (artifact_dir / "soft_probability_structured_shape_search_results.csv").write_text(
        detail_frame.to_csv(index=False), encoding="utf-8"
    )
    (artifact_dir / "soft_probability_structured_shape_feature_catalog.csv").write_text(
        feature_catalog.to_csv(index=False), encoding="utf-8"
    )
    (artifact_dir / "soft_probability_structured_shape_search_summary.json").write_text(
        summary_frame.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(report_dir / "soft_probability_structured_shape_search_summary.md", summary_frame.to_dict(orient="records"))


if __name__ == "__main__":
    main()
