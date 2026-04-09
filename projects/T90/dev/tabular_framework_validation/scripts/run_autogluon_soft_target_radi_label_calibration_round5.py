from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_label_x_controlled_matrix_round1 import _prepare_labeled, _preclean
from run_autogluon_stage1_quickcheck import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    resolve_path,
)
from run_autogluon_stage2_dynamic_morphology import build_stage2_table
from run_autogluon_stage2_feature_engineering import select_features_fold
from run_autogluon_stage2_soft_probability import (
    make_soft_probability_target,
    soft_probability_metrics,
)
from run_autogluon_stage4_interactions import build_interaction_frame
from run_autogluon_stage5_quality import build_quality_table
from run_autogluon_stage7_final_selection import load_references, select_task_priors
from run_autogluon_soft_target_quality_ablation_round4 import (
    build_inner_oof_predictions,
    choose_soft_thresholds,
    enrich_soft_scored_rows,
    load_round3_ap_floor,
    soft_threshold_candidate_rows,
    summarize_deployability,
    summarize_fold_radi,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_soft_target_radi_label_calibration_round5.yaml"


def build_fixed_full_quality_frame(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    labeled = _prepare_labeled(config_path, config)
    soft_label_name = str(config["label_fuzziness"]["target_name"])
    refs = load_references(config_path, config)
    priors = select_task_priors(refs)["centered_desirability"]
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
    interaction_frame = build_interaction_frame(
        snapshot,
        int(priors["tau_minutes"]),
        int(priors["window_minutes"]),
        str(priors["interaction_package"]),
    )
    quality_table = build_quality_table(
        labeled_samples=labeled,
        dcs=dcs,
        tau_minutes=int(priors["tau_minutes"]),
        window_minutes=int(priors["window_minutes"]),
        enabled_features=list(config["full_quality_features"]),
        min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
    )
    base_feature_columns = [column for column in snapshot.columns if "__" in column and "_dyn_" not in column]
    base_meta = snapshot[
        [
            "sample_time",
            "decision_time",
            "t90",
            "is_in_spec",
            "is_out_of_spec",
            "is_above_spec",
            "is_below_spec",
        ]
        + base_feature_columns
    ].copy()
    base_meta = base_meta.merge(labeled[["sample_time", soft_label_name]], on="sample_time", how="left")
    quality_cols = [column for column in quality_table.columns if column.startswith("quality__")]
    quality_merged = snapshot[["decision_time"]].merge(quality_table, on="decision_time", how="left")
    frame = pd.concat(
        [
            base_meta.reset_index(drop=True),
            interaction_frame.reset_index(drop=True),
            quality_merged[quality_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    return frame.sort_values("sample_time").reset_index(drop=True)


def apply_label_variant(frame: pd.DataFrame, config: dict[str, Any], rule: str, boundary_softness: float) -> pd.DataFrame:
    out = frame.copy()
    target_name = str(config["label_fuzziness"]["target_name"])
    out[target_name] = make_soft_probability_target(
        out["t90"],
        center=float(config["target_spec"]["center"]),
        tolerance=float(config["target_spec"]["tolerance"]),
        boundary_softness=float(boundary_softness),
        rule=str(rule),
    )
    return out


def evaluate_label_variant(
    frame: pd.DataFrame,
    variant_name: str,
    rule: str,
    boundary_softness: float,
    config: dict[str, Any],
    artifact_dir: Path,
    run_id: str,
    ap_floor: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    soft_label_name = str(config["label_fuzziness"]["target_name"])
    top_k = int(config["selection"]["shared_top_k"])
    feature_columns = [column for column in frame.columns if "__" in column]
    spec_low = float(config["radi"]["spec_low"])
    spec_high = float(config["radi"]["spec_high"])
    measurement_error_scenarios = [float(value) for value in config["measurement_error_scenarios"]]
    clear_threshold_grid = [float(value) for value in config["radi"]["clear_threshold_grid"]]
    alert_threshold_grid = [float(value) for value in config["radi"]["alert_threshold_grid"]]

    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    scored_rows_parts: list[pd.DataFrame] = []
    threshold_candidate_parts: list[dict[str, Any]] = []
    fold_summary_parts: list[dict[str, Any]] = []
    selected_features_fold1: list[str] = []
    raw_feature_count = 0
    cleaned_feature_count = 0
    selected_feature_count = 0
    scenario_fold_rows: dict[float, list[dict[str, Any]]] = {scenario: [] for scenario in measurement_error_scenarios}

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(frame), start=1):
        train = frame.iloc[train_idx].copy().reset_index(drop=True)
        test = frame.iloc[test_idx].copy().reset_index(drop=True)
        train_clean, test_clean, cleaned_cols = _preclean(
            train[feature_columns],
            test[feature_columns],
            max_missing_ratio=float(config["preclean"]["max_missing_ratio"]),
            unique_threshold=int(config["preclean"]["near_constant_unique_threshold"]),
        )
        raw_feature_count = int(len(feature_columns))
        cleaned_feature_count = int(len(cleaned_cols))
        selected_features, _ = select_features_fold(
            train_x=train_clean,
            train_y=train[soft_label_name],
            task_type="regression",
            top_k=top_k,
        )
        if not selected_features:
            selected_features = cleaned_cols
        selected_feature_count = int(len(selected_features))
        if fold_idx == 1:
            selected_features_fold1 = list(selected_features)

        train_sel = train_clean[selected_features].copy()
        test_sel = test_clean[selected_features].copy()
        y_train = train[soft_label_name].to_numpy(dtype=float)
        y_test = test[soft_label_name].to_numpy(dtype=float)
        hard_out = test["is_out_of_spec"].to_numpy(dtype=int)

        baseline = make_regression_baseline()
        baseline.fit(train_sel, y_train)
        base_pred = baseline.predict(test_sel).astype(float)
        base_metric = soft_probability_metrics(y_test, base_pred, hard_out)

        model_path = artifact_dir / f"ag_soft_radi_label_round5_{variant_name}_{run_id}_fold{fold_idx}"
        ag_pred, model_best = fit_autogluon_fold(
            train_df=pd.concat([train_sel, train[[soft_label_name]]], axis=1).copy(),
            test_df=pd.concat([test_sel, test[[soft_label_name]]], axis=1).copy(),
            label=soft_label_name,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )
        ag_metric = soft_probability_metrics(y_test, ag_pred.astype(float), hard_out)

        rows.append(
            {
                "variant_name": variant_name,
                "rule": rule,
                "boundary_softness": float(boundary_softness),
                "framework": "simple_baseline",
                "fold": int(fold_idx),
                "raw_feature_count": raw_feature_count,
                "cleaned_feature_count": cleaned_feature_count,
                "selected_feature_count": selected_feature_count,
                **base_metric,
            }
        )
        rows.append(
            {
                "variant_name": variant_name,
                "rule": rule,
                "boundary_softness": float(boundary_softness),
                "framework": "autogluon",
                "fold": int(fold_idx),
                "raw_feature_count": raw_feature_count,
                "cleaned_feature_count": cleaned_feature_count,
                "selected_feature_count": selected_feature_count,
                "autogluon_model_best": model_best,
                **ag_metric,
            }
        )
        fold_summaries.append(
            {
                "fold": int(fold_idx),
                "selected_feature_count": selected_feature_count,
                "baseline_soft_brier": float(base_metric["soft_brier"]),
                "autogluon_soft_brier": float(ag_metric["soft_brier"]),
                "autogluon_hard_out_ap_diagnostic": float(ag_metric["hard_out_ap_diagnostic"]),
                "autogluon_hard_out_auc_diagnostic": float(ag_metric["hard_out_auc_diagnostic"]),
            }
        )

        inner_oof = build_inner_oof_predictions(
            train_outer=train,
            feature_columns=feature_columns,
            soft_label_name=soft_label_name,
            config=config,
            artifact_dir=artifact_dir,
            run_id=run_id,
            variant_name=variant_name,
            outer_fold_idx=fold_idx,
        )
        test_scored_base = test[["sample_time", "t90", "is_out_of_spec"]].copy()
        test_scored_base["pred_score_raw"] = ag_pred.astype(float)

        for scenario in measurement_error_scenarios:
            inner_scored = enrich_soft_scored_rows(inner_oof, scenario, spec_low, spec_high)
            candidates = soft_threshold_candidate_rows(
                inner_scored,
                clear_threshold_grid=clear_threshold_grid,
                alert_threshold_grid=alert_threshold_grid,
                ap_floor=ap_floor,
            )
            best_threshold = choose_soft_thresholds(candidates)
            for row in candidates:
                row.update(
                    {
                        "run_id": run_id,
                        "scheme_name": variant_name,
                        "outer_fold_id": int(fold_idx),
                        "measurement_error_scenario": scenario,
                    }
                )
                threshold_candidate_parts.append(row)

            test_scored = enrich_soft_scored_rows(test_scored_base, scenario, spec_low, spec_high)
            test_scored["run_id"] = run_id
            test_scored["scheme_name"] = variant_name
            test_scored["label_family"] = "soft_target"
            test_scored["feature_recipe_name"] = "lag120_win60_plus_full_interaction_plus_full_quality"
            test_scored["model_framework"] = "autogluon"
            test_scored["outer_fold_id"] = int(fold_idx)
            test_scored["preferred_threshold"] = np.nan
            test_scored["preferred_flag"] = np.nan
            test_scored["clear_threshold"] = float(best_threshold["inner_clear_threshold"])
            test_scored["alert_threshold"] = float(best_threshold["inner_alert_threshold"])
            test_scored["clear_flag"] = (test_scored["pred_outspec_risk_aligned"] <= float(best_threshold["inner_clear_threshold"])).astype(int)
            test_scored["alert_flag"] = (test_scored["pred_outspec_risk_aligned"] >= float(best_threshold["inner_alert_threshold"])).astype(int)
            test_scored["retest_flag"] = 1 - test_scored["clear_flag"] - test_scored["alert_flag"]
            scored_rows_parts.append(test_scored)

            fold_summary = summarize_fold_radi(
                test_scored,
                clear_threshold=float(best_threshold["inner_clear_threshold"]),
                alert_threshold=float(best_threshold["inner_alert_threshold"]),
                ap_floor=ap_floor,
            )
            fold_summary.update(
                {
                    "run_id": run_id,
                    "scheme_name": variant_name,
                    "outer_fold_id": int(fold_idx),
                    "selected_clear_threshold": float(best_threshold["inner_clear_threshold"]),
                    "selected_alert_threshold": float(best_threshold["inner_alert_threshold"]),
                    "measurement_error_scenario": scenario,
                }
            )
            fold_summary_parts.append(fold_summary)
            scenario_fold_rows[scenario].append(fold_summary)

    deploy_parts = [
        {
            "run_id": run_id,
            **summarize_deployability(
                fold_rows=scenario_fold_rows[scenario],
                variant_name=variant_name,
                measurement_error_scenario=scenario,
            ),
        }
        for scenario in measurement_error_scenarios
    ]

    results = pd.DataFrame(rows)
    baseline_rows = results[results["framework"] == "simple_baseline"]
    ag_rows = results[results["framework"] == "autogluon"]
    deploy_df = pd.DataFrame(deploy_parts)
    summary = {
        "variant_name": variant_name,
        "rule": rule,
        "boundary_softness": float(boundary_softness),
        "raw_feature_count": raw_feature_count,
        "cleaned_feature_count": cleaned_feature_count,
        "selected_feature_count": selected_feature_count,
        "selected_features_fold1": selected_features_fold1,
        "baseline_mean_soft_mae": float(baseline_rows["soft_mae"].mean()),
        "autogluon_mean_soft_mae": float(ag_rows["soft_mae"].mean()),
        "baseline_mean_soft_brier": float(baseline_rows["soft_brier"].mean()),
        "autogluon_mean_soft_brier": float(ag_rows["soft_brier"].mean()),
        "baseline_mean_hard_out_ap_diagnostic": float(baseline_rows["hard_out_ap_diagnostic"].mean()),
        "autogluon_mean_hard_out_ap_diagnostic": float(ag_rows["hard_out_ap_diagnostic"].mean()),
        "baseline_mean_hard_out_auc_diagnostic": float(baseline_rows["hard_out_auc_diagnostic"].mean()),
        "autogluon_mean_hard_out_auc_diagnostic": float(ag_rows["hard_out_auc_diagnostic"].mean()),
        "radi_measurement_summaries": deploy_df.to_dict(orient="records"),
        "radi_worst_case_mean": float(deploy_df["radi_mean"].min()) if not deploy_df.empty else float("nan"),
        "radi_worst_case_gate_pass_rate": float(deploy_df["gate_pass_rate"].min()) if not deploy_df.empty else float("nan"),
        "radi_worst_case_status": (
            deploy_df.sort_values(["radi_mean", "measurement_error_scenario"], ascending=[True, True]).iloc[0]["recommended_status"]
            if not deploy_df.empty
            else "FAIL"
        ),
        "fold_summaries": fold_summaries,
    }
    return results, summary, pd.concat(scored_rows_parts, ignore_index=True), pd.DataFrame(threshold_candidate_parts), pd.DataFrame(fold_summary_parts)


def write_report(path: Path, summary_rows: list[dict[str, Any]], ap_floor: float) -> None:
    lines = [
        "# Soft Target RADI-Oriented Label Calibration Round 5",
        "",
        "## Scope",
        "",
        "- Freeze the strongest X-side backbone from round 4.",
        "- Only vary soft-target label mapping rule and boundary softness.",
        "- Re-evaluate with the same RADI offline deployability protocol.",
        "",
        "## AP Floor",
        "",
        f"- `ap_floor = {ap_floor:.6f}`",
        "",
        "## Summary Rows",
        "",
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RADI-oriented label calibration on fixed strongest soft-target backbone.")
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

    ap_floor = load_round3_ap_floor(config_path, config)
    base_frame = build_fixed_full_quality_frame(config_path, config)
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    detail_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    scored_rows_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []
    fold_summary_parts: list[pd.DataFrame] = []

    for variant_name, spec in dict(config["label_variants"]).items():
        variant_frame = apply_label_variant(
            base_frame,
            config=config,
            rule=str(spec["rule"]),
            boundary_softness=float(spec["boundary_softness"]),
        )
        results, summary, scored_rows, threshold_candidates, fold_summary = evaluate_label_variant(
            frame=variant_frame,
            variant_name=str(variant_name),
            rule=str(spec["rule"]),
            boundary_softness=float(spec["boundary_softness"]),
            config=config,
            artifact_dir=artifact_dir,
            run_id=run_id,
            ap_floor=ap_floor,
        )
        detail_parts.append(results)
        summary_rows.append(summary)
        scored_rows_parts.append(scored_rows)
        threshold_parts.append(threshold_candidates)
        fold_summary_parts.append(fold_summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["radi_worst_case_mean", "radi_worst_case_gate_pass_rate", "autogluon_mean_soft_brier", "variant_name"],
        ascending=[False, False, True, True],
    )
    details_df = pd.concat(detail_parts, ignore_index=True)
    scored_rows_df = pd.concat(scored_rows_parts, ignore_index=True)
    threshold_df = pd.concat(threshold_parts, ignore_index=True)
    fold_summary_df = pd.concat(fold_summary_parts, ignore_index=True)
    deploy_rows = [deploy for row in summary_rows for deploy in row["radi_measurement_summaries"]]
    deploy_df = pd.DataFrame(deploy_rows).sort_values(by=["scheme_name", "measurement_error_scenario"], ascending=[True, True])

    summary_path = artifact_dir / "soft_target_radi_label_calibration_round5_summary.json"
    results_path = artifact_dir / "soft_target_radi_label_calibration_round5_results.csv"
    scored_rows_path = artifact_dir / "offline_eval_scored_rows.csv"
    threshold_path = artifact_dir / "offline_soft_threshold_candidates.csv"
    fold_summary_path = artifact_dir / "offline_soft_fold_summary.csv"
    deploy_path = artifact_dir / "offline_soft_deployability_summary.csv"
    report_path = report_dir / "soft_target_radi_label_calibration_round5_summary.md"

    summary_path.write_text(json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    details_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    scored_rows_df.to_csv(scored_rows_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    fold_summary_df.to_csv(fold_summary_path, index=False, encoding="utf-8-sig")
    deploy_df.to_csv(deploy_path, index=False, encoding="utf-8-sig")
    write_report(report_path, summary_df.to_dict(orient="records"), ap_floor)

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "results_path": str(results_path),
                "report_path": str(report_path),
                "offline_scored_rows_path": str(scored_rows_path),
                "offline_soft_threshold_candidates_path": str(threshold_path),
                "offline_soft_fold_summary_path": str(fold_summary_path),
                "offline_soft_deployability_summary_path": str(deploy_path),
                "best_variant": summary_df.iloc[0].to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
