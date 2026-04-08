from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_stage1_lag_scale import build_label_frame
from run_autogluon_stage1_quickcheck import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    regression_metrics,
    resolve_path,
)
from run_autogluon_stage2_dynamic_morphology import build_stage2_table
from run_autogluon_stage5_quality import build_quality_table
from run_autogluon_stage6_centered_quality import build_centered_quality_table
from run_autogluon_stage7_final_selection import (
    build_current_task_frame,
    combo_specs,
    compose_fold_features,
    load_references,
    preclean_features,
    select_task_priors,
    supervised_select,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_centered_desirability_outspec_eval.yaml"


def centered_outspec_metrics(
    y_true_desirability: np.ndarray,
    pred_desirability: np.ndarray,
    in_spec_flag: np.ndarray,
) -> dict[str, float]:
    metrics = regression_metrics(y_true_desirability, pred_desirability, in_spec_flag)
    clipped_desirability = np.clip(pred_desirability.astype(float), 0.0, 1.0)
    hard_out_flag = 1 - in_spec_flag.astype(int)
    hard_out_risk = 1.0 - clipped_desirability
    if len(np.unique(hard_out_flag)) > 1:
        metrics["hard_out_ap_diagnostic"] = float(average_precision_score(hard_out_flag, hard_out_risk))
        metrics["hard_out_auc_diagnostic"] = float(roc_auc_score(hard_out_flag, hard_out_risk))
    else:
        metrics["hard_out_ap_diagnostic"] = float("nan")
        metrics["hard_out_auc_diagnostic"] = float("nan")
    return metrics


def load_best_centered_combo(summary_path: Path) -> dict[str, Any]:
    summary_df = pd.DataFrame(json.loads(summary_path.read_text(encoding="utf-8")))
    centered = summary_df[summary_df["task_name"] == "centered_desirability"].copy()
    if centered.empty:
        raise ValueError("No centered_desirability row found in stage7 summary.")
    centered = centered.sort_values(["autogluon_mean_mae", "combo_name", "top_k"], ascending=[True, True, True])
    return centered.iloc[0].to_dict()


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Centered Desirability Out-of-Spec Diagnostic Summary",
        "",
        "## Purpose",
        "",
        "- Re-evaluate the current best `centered_desirability` solution with the same hard out-of-spec AP/AUC diagnostics used in the soft target branch.",
        "- The diagnostic mapping is `hard_out_risk = 1 - clip(predicted_desirability, 0, 1)`.",
        "",
        "## Selected Centered Solution",
        "",
        json.dumps(
            {
                "combo_name": summary["combo_name"],
                "top_k": summary["top_k"],
                "selected_stage1_variant": summary["selected_stage1_variant"],
                "selected_interaction_package": summary["selected_interaction_package"],
                "selected_quality_package": summary["selected_quality_package"],
                "selected_centered_quality_package": summary["selected_centered_quality_package"],
                "tau_minutes": summary["tau_minutes"],
                "window_minutes": summary["window_minutes"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "## Summary",
        "",
        json.dumps(summary, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best centered_desirability package with hard out-of-spec AP/AUC diagnostics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    report_dir = resolve_path(config_path.parent, config["paths"]["report_dir"])
    stage7_summary_path = resolve_path(config_path.parent, config["paths"]["stage7_summary_path"])
    if artifact_dir is None or report_dir is None or stage7_summary_path is None:
        raise ValueError("artifact_dir, report_dir, and stage7_summary_path must be configured.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    refs = load_references(config_path, config)
    priors = select_task_priors(refs)["centered_desirability"]
    best_combo_summary = load_best_centered_combo(stage7_summary_path)
    best_combo_name = str(best_combo_summary["combo_name"])
    best_top_k = int(best_combo_summary["top_k"])
    combo = next(spec for spec in combo_specs("centered_desirability", priors) if spec["name"] == best_combo_name)

    labeled = build_label_frame(config_path, config)
    dcs = load_dcs_frame(
        resolve_path(config_path.parent, config["paths"]["dcs_main_path"]),
        resolve_path(config_path.parent, config["paths"].get("dcs_supplemental_path")),
    )
    snapshot = build_stage2_table(
        labeled_samples=labeled,
        dcs=dcs,
        tau_minutes=int(priors["tau_minutes"]),
        window_minutes=int(priors["window_minutes"]),
        stats=list(config["snapshot"]["stats"]),
        enabled_dynamic_features=[],
        min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
    )
    base_feature_columns = [column for column in snapshot.columns if "__" in column and "_dyn_" not in column]

    quality_table = None
    if priors.get("quality_package"):
        quality_table = build_quality_table(
            labeled_samples=labeled,
            dcs=dcs,
            tau_minutes=int(priors["tau_minutes"]),
            window_minutes=int(priors["window_minutes"]),
            enabled_features=list(config["quality_feature_map"][priors["quality_package"]]),
            min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
        )

    centered_table = None
    if priors.get("centered_quality_package"):
        centered_table = build_centered_quality_table(
            labeled_samples=labeled,
            dcs=dcs,
            tau_minutes=int(priors["tau_minutes"]),
            window_minutes=int(priors["window_minutes"]),
            enabled_features=list(config["centered_quality_feature_map"][priors["centered_quality_package"]]),
            min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
        )

    prepared = build_current_task_frame("centered_desirability", snapshot, priors, combo, quality_table, centered_table)
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    agg: dict[str, list[dict[str, float]]] = {"simple_baseline": [], "autogluon": []}
    selected_features_fold1: list[str] = []
    raw_feature_count = 0
    cleaned_feature_count = 0
    selected_feature_count = 0

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(prepared), start=1):
        train = prepared.iloc[train_idx].copy().reset_index(drop=True)
        test = prepared.iloc[test_idx].copy().reset_index(drop=True)

        train_x, test_x = compose_fold_features(train, test, base_feature_columns, priors, combo)
        raw_feature_count = int(train_x.shape[1])

        train_clean, test_clean, cleaned_cols = preclean_features(
            train_x,
            test_x,
            max_missing_ratio=float(config["preclean"]["max_missing_ratio"]),
            unique_threshold=int(config["preclean"]["near_constant_unique_threshold"]),
        )
        cleaned_feature_count = int(len(cleaned_cols))

        y_train = train["target_centered_desirability"].to_numpy(dtype=float)
        y_test = test["target_centered_desirability"].to_numpy(dtype=float)
        in_spec_test = test["is_in_spec"].to_numpy(dtype=int)

        selected_cols = supervised_select(train_clean, y_train, "centered_desirability", best_top_k)
        if not selected_cols:
            selected_cols = cleaned_cols
        selected_feature_count = int(len(selected_cols))
        if fold_idx == 1:
            selected_features_fold1 = list(selected_cols)

        train_sel = train_clean[selected_cols].copy()
        test_sel = test_clean[selected_cols].copy()

        baseline = make_regression_baseline()
        baseline.fit(train_sel, y_train)
        base_pred = baseline.predict(test_sel).astype(float)
        base_metric = centered_outspec_metrics(y_test, base_pred, in_spec_test)

        model_path = artifact_dir / f"ag_centered_outspec_eval_{best_combo_name}_top{best_top_k}_{run_id}_fold{fold_idx}"
        ag_pred, model_best = fit_autogluon_fold(
            train_df=pd.concat([train_sel, train[["target_centered_desirability"]]], axis=1).copy(),
            test_df=pd.concat([test_sel, test[["target_centered_desirability"]]], axis=1).copy(),
            label="target_centered_desirability",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )
        ag_metric = centered_outspec_metrics(y_test, ag_pred.astype(float), in_spec_test)

        rows.append(
            {
                "stage": "centered_desirability_outspec_eval",
                "task_name": "centered_desirability",
                "combo_name": best_combo_name,
                "top_k": int(best_top_k),
                "fold": int(fold_idx),
                "framework": "simple_baseline",
                "samples_train": int(len(train)),
                "samples_test": int(len(test)),
                "raw_feature_count": raw_feature_count,
                "cleaned_feature_count": cleaned_feature_count,
                "selected_feature_count": selected_feature_count,
                **base_metric,
            }
        )
        rows.append(
            {
                "stage": "centered_desirability_outspec_eval",
                "task_name": "centered_desirability",
                "combo_name": best_combo_name,
                "top_k": int(best_top_k),
                "fold": int(fold_idx),
                "framework": "autogluon",
                "samples_train": int(len(train)),
                "samples_test": int(len(test)),
                "raw_feature_count": raw_feature_count,
                "cleaned_feature_count": cleaned_feature_count,
                "selected_feature_count": selected_feature_count,
                "autogluon_model_best": model_best,
                **ag_metric,
            }
        )

        agg["simple_baseline"].append(base_metric)
        agg["autogluon"].append(ag_metric)
        fold_summaries.append(
            {
                "fold": int(fold_idx),
                "selected_feature_count": int(len(selected_cols)),
                "baseline": base_metric,
                "autogluon": ag_metric,
            }
        )

    results_df = pd.DataFrame(rows)
    summary = {
        "task_name": "centered_desirability",
        "combo_name": best_combo_name,
        "top_k": int(best_top_k),
        "selected_stage1_variant": str(priors["stage1_variant"]),
        "selected_interaction_package": priors.get("interaction_package"),
        "selected_quality_package": priors.get("quality_package"),
        "selected_centered_quality_package": priors.get("centered_quality_package"),
        "tau_minutes": int(priors["tau_minutes"]),
        "window_minutes": int(priors["window_minutes"]),
        "raw_feature_count": raw_feature_count,
        "cleaned_feature_count": cleaned_feature_count,
        "selected_feature_count": selected_feature_count,
        "selected_features_fold1": selected_features_fold1,
        "baseline_mean_mae": float(np.nanmean([m["mae"] for m in agg["simple_baseline"]])),
        "autogluon_mean_mae": float(np.nanmean([m["mae"] for m in agg["autogluon"]])),
        "baseline_mean_rmse": float(np.nanmean([m["rmse"] for m in agg["simple_baseline"]])),
        "autogluon_mean_rmse": float(np.nanmean([m["rmse"] for m in agg["autogluon"]])),
        "baseline_mean_rank_correlation": float(np.nanmean([m["rank_correlation"] for m in agg["simple_baseline"]])),
        "autogluon_mean_rank_correlation": float(np.nanmean([m["rank_correlation"] for m in agg["autogluon"]])),
        "baseline_mean_in_spec_auc_from_desirability": float(np.nanmean([m["in_spec_auc_from_desirability"] for m in agg["simple_baseline"]])),
        "autogluon_mean_in_spec_auc_from_desirability": float(np.nanmean([m["in_spec_auc_from_desirability"] for m in agg["autogluon"]])),
        "baseline_mean_hard_out_ap_diagnostic": float(np.nanmean([m["hard_out_ap_diagnostic"] for m in agg["simple_baseline"]])),
        "autogluon_mean_hard_out_ap_diagnostic": float(np.nanmean([m["hard_out_ap_diagnostic"] for m in agg["autogluon"]])),
        "baseline_mean_hard_out_auc_diagnostic": float(np.nanmean([m["hard_out_auc_diagnostic"] for m in agg["simple_baseline"]])),
        "autogluon_mean_hard_out_auc_diagnostic": float(np.nanmean([m["hard_out_auc_diagnostic"] for m in agg["autogluon"]])),
        "fold_summaries": fold_summaries,
    }

    results_path = artifact_dir / "centered_desirability_outspec_eval_results.csv"
    summary_path = artifact_dir / "centered_desirability_outspec_eval_summary.json"
    report_path = report_dir / "centered_desirability_outspec_eval_summary.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, summary)
    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
