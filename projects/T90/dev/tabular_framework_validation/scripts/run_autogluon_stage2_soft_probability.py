from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_stage2_feature_engineering import (
    build_stage2_snapshot_table,
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    resolve_path,
    select_features_fold,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage2_soft_probability.yaml"


def make_soft_probability_target(
    t90: pd.Series,
    center: float,
    tolerance: float,
    boundary_softness: float,
    rule: str,
) -> pd.Series:
    signed_margin = (t90 - center).abs() - tolerance
    scaled_distance = signed_margin / boundary_softness
    if rule == "logistic_boundary_membership":
        values = 1.0 / (1.0 + np.exp(-scaled_distance))
    elif rule == "tanh_boundary_membership":
        values = 0.5 * (1.0 + np.tanh(scaled_distance))
    elif rule == "linear_boundary_band":
        values = np.clip(0.5 + 0.5 * scaled_distance, 0.0, 1.0)
    elif rule == "smoothstep_boundary_band":
        normalized = np.clip((scaled_distance + 1.0) / 2.0, 0.0, 1.0)
        values = normalized * normalized * (3.0 - 2.0 * normalized)
    else:
        raise ValueError(f"Unsupported soft probability rule: {rule}")
    return pd.Series(values, index=t90.index, dtype=float)


def soft_probability_metrics(
    y_true_soft: np.ndarray,
    pred_soft: np.ndarray,
    hard_out_flag: np.ndarray,
) -> dict[str, float]:
    pred = np.clip(pred_soft.astype(float), 0.0, 1.0)
    rank_corr = pd.Series(y_true_soft).corr(pd.Series(pred), method="spearman")
    metrics: dict[str, float] = {
        "soft_mae": float(np.mean(np.abs(y_true_soft - pred))),
        "soft_rmse": float(np.sqrt(np.mean((y_true_soft - pred) ** 2))),
        "soft_brier": float(np.mean((y_true_soft - pred) ** 2)),
        "rank_correlation": float(rank_corr) if pd.notna(rank_corr) else float("nan"),
    }
    if len(np.unique(hard_out_flag)) > 1:
        metrics["hard_out_ap_diagnostic"] = float(average_precision_score(hard_out_flag, pred))
        metrics["hard_out_auc_diagnostic"] = float(roc_auc_score(hard_out_flag, pred))
    else:
        metrics["hard_out_ap_diagnostic"] = float("nan")
        metrics["hard_out_auc_diagnostic"] = float("nan")
    return metrics


def build_soft_probability_snapshot(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshot, snapshot_audit = build_stage2_snapshot_table(config_path, config)
    center = float(config["target_spec"]["center"])
    tolerance = float(config["target_spec"]["tolerance"])
    boundary_softness = float(config["label_fuzziness"]["boundary_softness"])
    label_name = str(config["label_fuzziness"]["target_name"])
    rule = str(config["label_fuzziness"]["rule"])
    snapshot[label_name] = make_soft_probability_target(
        snapshot["t90"],
        center,
        tolerance,
        boundary_softness,
        rule,
    )
    snapshot_audit = {
        **snapshot_audit,
        "soft_probability_label_name": label_name,
        "soft_probability_rule": str(config["label_fuzziness"]["rule"]),
        "boundary_softness": boundary_softness,
    }
    return snapshot.sort_values("sample_time").reset_index(drop=True), snapshot_audit


def run_soft_probability_stage2(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    snapshot, snapshot_audit = build_soft_probability_snapshot(config_path, config)
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
    fold_summaries: list[dict[str, Any]] = []
    selection_summary: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
        train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
        test = snapshot.iloc[test_idx].copy().reset_index(drop=True)
        selected_features, top_scores = select_features_fold(
            train_x=train[feature_columns],
            train_y=train[label],
            task_type="regression",
            top_k=top_k,
        )
        selection_summary.append(
            {
                "fold": int(fold_idx),
                "selected_feature_count": int(len(selected_features)),
                "top_feature_scores": top_scores,
            }
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

        model_path = artifact_dir / f"ag_stage2_soft_probability_{run_id}_fold{fold_idx}"
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

        for framework_name, metrics in (
            ("simple_baseline_stage2_soft_probability", baseline_metrics),
            ("autogluon_stage2_soft_probability", framework_metrics),
        ):
            row = {
                "stage": "stage2_soft_probability",
                "framework": framework_name,
                "fold": int(fold_idx),
                "samples_train": int(len(train_df)),
                "samples_test": int(len(test_df)),
                "lookback_minutes": int(config["snapshot"]["lookback_minutes"]),
                "feature_count_selected": int(len(selected_features)),
                "data_source_condition": config["data_source_statement"]["source_condition"],
                "boundary_softness": float(config["label_fuzziness"]["boundary_softness"]),
                **metrics,
            }
            if framework_name == "autogluon_stage2_soft_probability":
                row["autogluon_model_best"] = model_best
            rows.append(row)

        fold_summaries.append(
            {
                "fold": int(fold_idx),
                "selected_feature_count": int(len(selected_features)),
                "baseline": baseline_metrics,
                "autogluon": framework_metrics,
            }
        )

    results = pd.DataFrame(rows)
    baseline_rows = results[results["framework"] == "simple_baseline_stage2_soft_probability"]
    ag_rows = results[results["framework"] == "autogluon_stage2_soft_probability"]
    stage_summary = {
        "task_name": "soft_out_of_spec_probability",
        "label_name": label,
        "fold_summaries": fold_summaries,
        "selection_summary": selection_summary,
        "baseline_mean_soft_mae": float(baseline_rows["soft_mae"].mean()),
        "autogluon_mean_soft_mae": float(ag_rows["soft_mae"].mean()),
        "baseline_mean_soft_brier": float(baseline_rows["soft_brier"].mean()),
        "autogluon_mean_soft_brier": float(ag_rows["soft_brier"].mean()),
        "baseline_mean_hard_out_ap_diagnostic": float(baseline_rows["hard_out_ap_diagnostic"].mean()),
        "autogluon_mean_hard_out_ap_diagnostic": float(ag_rows["hard_out_ap_diagnostic"].mean()),
        "positive_signal": bool(ag_rows["soft_brier"].mean() < baseline_rows["soft_brier"].mean()),
    }
    return results, stage_summary, snapshot_audit, snapshot


def write_soft_probability_audit(
    path: Path,
    stage_summary: dict[str, Any],
    snapshot_audit: dict[str, Any],
    config: dict[str, Any],
) -> None:
    lines = [
        "# Tabular Framework Validation Audit - Stage 2 Soft Probability Branch",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- Uncleaned refers to source state, not a ban on preprocessing.",
        "",
        "## Branch Purpose",
        "",
        "- This branch continues the desirability line, but shifts the output into a fuzzy out-of-spec risk probability.",
        "- It treats hard out-of-spec labels only as a diagnostic reference, not as the primary training target.",
        "",
        "## Label Type",
        "",
        "- The target is a continuous fuzzy risk score, not a hard class label.",
        f"- Rule: {config['label_fuzziness']['rule']}.",
        f"- Boundary softness: {config['label_fuzziness']['boundary_softness']}.",
        "- Formula: sigmoid((abs(T90 - center) - tolerance) / boundary_softness).",
        "",
        "## X-side Feature Engineering",
        "",
        "- causal 120-minute snapshot",
        "- stats: mean, std, min, max, last, delta, range, slope, valid_ratio",
        "- drop constant features",
        "- drop high-missing features",
        "- drop near-duplicate high-correlation features",
        "- fold-internal supervised feature selection",
        "",
        "## Stage Conclusions",
        "",
        json.dumps(stage_summary, ensure_ascii=False, indent=2),
        "",
        "## Snapshot Audit",
        "",
        json.dumps(snapshot_audit, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon stage 2 soft out-of-spec probability validation.")
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

    results, stage_summary, snapshot_audit, snapshot = run_soft_probability_stage2(config_path, config)
    results_path = artifact_dir / "tabular_framework_validation_soft_probability_results.csv"
    summary_path = artifact_dir / "tabular_framework_validation_soft_probability_summary.json"
    snapshot_path = artifact_dir / "stage2_soft_probability_snapshot_feature_table.csv"
    audit_path = report_dir / "tabular_framework_validation_soft_probability_audit.md"

    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    snapshot.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "framework": "AutoGluon",
                "phase": "stage2_soft_out_of_spec_probability",
                "stage_summary": stage_summary,
                "snapshot_audit": snapshot_audit,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    write_soft_probability_audit(audit_path, stage_summary, snapshot_audit, config)
    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "snapshot_path": str(snapshot_path),
                "audit_path": str(audit_path),
                "stage_summary": stage_summary,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
