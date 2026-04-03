from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    build_soft_probability_snapshot,
    soft_probability_metrics,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage2_soft_probability_x_enrichment.yaml"
EPS = 1e-6


def _sensor_prefixes(snapshot: pd.DataFrame) -> list[str]:
    return sorted({column.split("__", 1)[0] for column in snapshot.columns if "__" in column})


def _safe_get(snapshot: pd.DataFrame, column: str) -> pd.Series | None:
    return snapshot[column] if column in snapshot.columns else None


def add_relative_dynamics_features(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame = snapshot.copy()
    added = 0
    for sensor in _sensor_prefixes(frame):
        mean_col = _safe_get(frame, f"{sensor}__mean")
        std_col = _safe_get(frame, f"{sensor}__std")
        last_col = _safe_get(frame, f"{sensor}__last")
        delta_col = _safe_get(frame, f"{sensor}__delta")
        slope_col = _safe_get(frame, f"{sensor}__slope")
        range_col = _safe_get(frame, f"{sensor}__range")
        if mean_col is not None and last_col is not None:
            frame[f"{sensor}__last_minus_mean"] = last_col - mean_col
            added += 1
        if delta_col is not None and std_col is not None:
            frame[f"{sensor}__delta_over_std"] = delta_col / (std_col.abs() + EPS)
            added += 1
        if delta_col is not None and range_col is not None:
            frame[f"{sensor}__delta_over_range"] = delta_col / (range_col.abs() + EPS)
            added += 1
        if slope_col is not None and std_col is not None:
            frame[f"{sensor}__slope_over_std"] = slope_col / (std_col.abs() + EPS)
            added += 1
        if slope_col is not None and range_col is not None:
            frame[f"{sensor}__slope_over_range"] = slope_col / (range_col.abs() + EPS)
            added += 1
        if std_col is not None and mean_col is not None:
            frame[f"{sensor}__std_over_abs_mean"] = std_col / (mean_col.abs() + EPS)
            added += 1
        if range_col is not None and mean_col is not None:
            frame[f"{sensor}__range_over_abs_mean"] = range_col / (mean_col.abs() + EPS)
            added += 1
    return frame, added


def add_range_position_features(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame = snapshot.copy()
    added = 0
    for sensor in _sensor_prefixes(frame):
        min_col = _safe_get(frame, f"{sensor}__min")
        max_col = _safe_get(frame, f"{sensor}__max")
        mean_col = _safe_get(frame, f"{sensor}__mean")
        last_col = _safe_get(frame, f"{sensor}__last")
        range_col = _safe_get(frame, f"{sensor}__range")
        if min_col is None or max_col is None or range_col is None:
            continue
        denom = range_col.abs() + EPS
        if last_col is not None:
            frame[f"{sensor}__last_position_in_range"] = (last_col - min_col) / denom
            frame[f"{sensor}__upper_gap_ratio"] = (max_col - last_col) / denom
            frame[f"{sensor}__lower_gap_ratio"] = (last_col - min_col) / denom
            added += 3
        if mean_col is not None:
            frame[f"{sensor}__mean_position_in_range"] = (mean_col - min_col) / denom
            frame[f"{sensor}__last_vs_mean_over_range"] = ((last_col - mean_col) / denom) if last_col is not None else np.nan
            added += 2
    return frame, added


def build_variant_snapshot(config_path: Path, config: dict[str, Any], variant_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshot, audit = build_soft_probability_snapshot(config_path, config)
    added_feature_count = 0
    if variant_name == "base_core":
        pass
    elif variant_name == "relative_dynamics":
        snapshot, added_feature_count = add_relative_dynamics_features(snapshot)
    elif variant_name == "range_position":
        snapshot, added_feature_count = add_range_position_features(snapshot)
    elif variant_name == "combined_enriched":
        snapshot, added_dyn = add_relative_dynamics_features(snapshot)
        snapshot, added_pos = add_range_position_features(snapshot)
        added_feature_count = added_dyn + added_pos
    else:
        raise ValueError(f"Unsupported x enrichment variant: {variant_name}")

    # Drop any newly created all-NaN or constant columns after enrichment.
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
        **audit,
        "x_enrichment_variant": variant_name,
        "x_enrichment_added_feature_count_before_post_clean": int(added_feature_count),
        "x_enrichment_dropped_all_nan_after_enrichment": int(len(dropped_all_nan)),
        "x_enrichment_dropped_constant_after_enrichment": int(len(dropped_constant)),
        "feature_count_after_x_enrichment": int(len([column for column in snapshot.columns if "__" in column])),
    }
    return snapshot.sort_values("sample_time").reset_index(drop=True), audit


def run_variant(config_path: Path, config: dict[str, Any], variant_name: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    snapshot, audit = build_variant_snapshot(config_path, config, variant_name)
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
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
        train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
        test = snapshot.iloc[test_idx].copy().reset_index(drop=True)
        selected_features, _ = select_features_fold(
            train_x=train[feature_columns],
            train_y=train[label],
            task_type="regression",
            top_k=top_k,
        )
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

        model_path = artifact_dir / f"ag_stage2_soft_probability_x_{variant_name}_{run_id}_fold{fold_idx}"
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
        "feature_count_after_x_enrichment": int(audit["feature_count_after_x_enrichment"]),
        "baseline_mean_soft_mae": float(baseline_rows["soft_mae"].mean()),
        "autogluon_mean_soft_mae": float(ag_rows["soft_mae"].mean()),
        "baseline_mean_soft_brier": float(baseline_rows["soft_brier"].mean()),
        "autogluon_mean_soft_brier": float(ag_rows["soft_brier"].mean()),
        "baseline_mean_hard_out_ap_diagnostic": float(baseline_rows["hard_out_ap_diagnostic"].mean()),
        "autogluon_mean_hard_out_ap_diagnostic": float(ag_rows["hard_out_ap_diagnostic"].mean()),
        "positive_signal": bool(ag_rows["soft_brier"].mean() < baseline_rows["soft_brier"].mean()),
    }
    return results, summary, audit


def write_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    best_row = min(rows, key=lambda row: row["autogluon_mean_soft_brier"])
    lines = [
        "# Tabular Framework Validation Audit - Soft Probability X Enrichment",
        "",
        "## Purpose",
        "",
        "- Keep the validated soft-label design fixed.",
        "- Compare additional X-side process-statistic feature bundles derived from the same causal snapshot.",
        "",
        "## Candidate Summary",
        "",
        json.dumps(rows, ensure_ascii=False, indent=2),
        "",
        "## Best Candidate By AutoGluon Soft Brier",
        "",
        json.dumps(best_row, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run X-side enrichment comparison for AutoGluon soft probability branch.")
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
    for variant_name in config["x_enrichment"]["variants"]:
        results, summary, _ = run_variant(config_path, config, str(variant_name))
        summary_rows.append(summary)
        detail_rows.append(results)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["autogluon_mean_soft_brier", "autogluon_mean_soft_mae", "variant_name"],
        ascending=[True, True, True],
    )
    detail_df = pd.concat(detail_rows, ignore_index=True)

    summary_path = artifact_dir / "tabular_framework_validation_soft_probability_x_enrichment_summary.csv"
    details_path = artifact_dir / "tabular_framework_validation_soft_probability_x_enrichment_results.csv"
    json_path = artifact_dir / "tabular_framework_validation_soft_probability_x_enrichment_summary.json"
    audit_path = report_dir / "tabular_framework_validation_soft_probability_x_enrichment_audit.md"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_df.to_dict(orient="records"), stream, ensure_ascii=False, indent=2)
    write_audit(audit_path, summary_df.to_dict(orient="records"))
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "details_path": str(details_path),
                "json_path": str(json_path),
                "audit_path": str(audit_path),
                "best_variant": summary_df.iloc[0].to_dict(),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
