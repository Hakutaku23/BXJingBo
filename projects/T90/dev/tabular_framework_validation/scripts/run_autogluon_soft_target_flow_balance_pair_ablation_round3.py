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
from run_autogluon_stage2_soft_probability import soft_probability_metrics
from run_autogluon_stage5_quality import build_quality_table
from run_autogluon_stage7_final_selection import load_references, select_task_priors


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_soft_target_flow_balance_pair_ablation_round3.yaml"

FLOW_BALANCE_PAIRS = [
    ("FIC_C51001_PV_F_CV", "FIC_C51003_PV_F_CV"),
    ("FIC_C51401_PV_F_CV", "FIC_C30501_PV_F_CV"),
    ("FIC_C51401_PV_F_CV", "FI_C51005_S_PV_CV"),
    ("FIC_C51401_PV_F_CV", "FIC_C51003_PV_F_CV"),
]

FLOW_STATS = ("mean", "last")
FLOW_OPS = ("diff", "ratio")


def _pair_column_names(left: str, right: str) -> list[str]:
    cols: list[str] = []
    for stat in FLOW_STATS:
        for op in FLOW_OPS:
            cols.append(f"interaction__{left}__{right}__{stat}_{op}")
    return cols


def build_pair_ablation_frames(config_path: Path, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
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
    if snapshot.empty:
        raise ValueError("Base lag snapshot is empty.")

    base_feature_columns = [column for column in snapshot.columns if "__" in column and "_dyn_" not in column]
    quality_table = build_quality_table(
        labeled_samples=labeled,
        dcs=dcs,
        tau_minutes=int(priors["tau_minutes"]),
        window_minutes=int(priors["window_minutes"]),
        enabled_features=list(config["quality_feature_map"][priors["quality_package"]]),
        min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
    )
    quality_merged = snapshot[["decision_time"]].merge(quality_table, on="decision_time", how="left")
    quality_cols = [column for column in quality_merged.columns if column.startswith("quality__")]

    meta = snapshot[
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
    meta = meta.merge(labeled[["sample_time", soft_label_name]], on="sample_time", how="left")

    pair_frames: dict[str, pd.DataFrame] = {}
    all_pair_cols: list[str] = []
    for left, right in FLOW_BALANCE_PAIRS:
        cols = _pair_column_names(left, right)
        all_pair_cols.extend(cols)
        pair_name = f"{left}__{right}"
        pair_df = pd.DataFrame(index=snapshot.index)
        for column in cols:
            left_col = f"{left}__lag{int(priors['tau_minutes'])}_win{int(priors['window_minutes'])}_{column.rsplit('__', 1)[-1].split('_')[0]}"
            right_col = f"{right}__lag{int(priors['tau_minutes'])}_win{int(priors['window_minutes'])}_{column.rsplit('__', 1)[-1].split('_')[0]}"
            if left_col not in snapshot.columns or right_col not in snapshot.columns:
                continue
            left_series = pd.to_numeric(snapshot[left_col], errors="coerce")
            right_series = pd.to_numeric(snapshot[right_col], errors="coerce")
            suffix = column.rsplit("__", 1)[-1]
            if suffix.endswith("diff"):
                pair_df[column] = left_series - right_series
            elif suffix.endswith("ratio"):
                pair_df[column] = left_series / right_series.replace(0.0, np.nan)
        pair_frames[pair_name] = pair_df

    all_interactions = pd.concat([pair_frames[f"{left}__{right}"] for left, right in FLOW_BALANCE_PAIRS], axis=1)
    quality_only = quality_merged[quality_cols].reset_index(drop=True)

    frames: dict[str, pd.DataFrame] = {
        "full_all_pairs": pd.concat([meta.reset_index(drop=True), all_interactions.reset_index(drop=True), quality_only], axis=1)
    }

    for left, right in FLOW_BALANCE_PAIRS:
        pair_name = f"{left}__{right}"
        pair_only_name = f"single_{pair_name}"
        drop_name = f"drop_{pair_name}"
        frames[pair_only_name] = pd.concat(
            [meta.reset_index(drop=True), pair_frames[pair_name].reset_index(drop=True), quality_only],
            axis=1,
        )

        remaining_pairs = [
            pair_frames[f"{l}__{r}"]
            for l, r in FLOW_BALANCE_PAIRS
            if not (l == left and r == right)
        ]
        remaining_df = pd.concat(remaining_pairs, axis=1) if remaining_pairs else pd.DataFrame(index=snapshot.index)
        frames[drop_name] = pd.concat(
            [meta.reset_index(drop=True), remaining_df.reset_index(drop=True), quality_only],
            axis=1,
        )

    return {
        name: frame.sort_values("sample_time").reset_index(drop=True)
        for name, frame in frames.items()
    }


def evaluate_variant(
    frame: pd.DataFrame,
    variant_name: str,
    config: dict[str, Any],
    artifact_dir: Path,
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    soft_label_name = str(config["label_fuzziness"]["target_name"])
    top_k = int(config["selection"]["shared_top_k"])
    feature_columns = [column for column in frame.columns if "__" in column]

    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    raw_feature_count = 0
    cleaned_feature_count = 0
    selected_feature_count = 0
    selected_features_fold1: list[str] = []

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

        model_path = artifact_dir / f"ag_soft_target_flow_pair_ablation_{variant_name}_{run_id}_fold{fold_idx}"
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

    results = pd.DataFrame(rows)
    baseline_rows = results[results["framework"] == "simple_baseline"]
    ag_rows = results[results["framework"] == "autogluon"]
    summary = {
        "variant_name": variant_name,
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
        "fold_summaries": fold_summaries,
    }
    return results, summary


def write_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Soft Target Flow-Balance Pair Ablation Round 3",
        "",
        "## Scope",
        "",
        "- Freeze the current strongest pair from round 2: `soft_target + lag_plus_interaction_plus_quality`.",
        "- Keep quality fixed and only dissect the `flow_balance` interaction package.",
        "- Compare full package, single-pair injections, and drop-one-pair variants.",
        "",
        "## Summary Rows",
        "",
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run flow-balance pair ablation for the strongest soft-target line.")
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

    frames = build_pair_ablation_frames(config_path, config)
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    detail_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for variant_name, frame in frames.items():
        results, summary = evaluate_variant(
            frame=frame,
            variant_name=variant_name,
            config=config,
            artifact_dir=artifact_dir,
            run_id=run_id,
        )
        detail_parts.append(results)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["autogluon_mean_soft_brier", "autogluon_mean_hard_out_ap_diagnostic", "variant_name"],
        ascending=[True, False, True],
    )
    details_df = pd.concat(detail_parts, ignore_index=True)

    summary_path = artifact_dir / "soft_target_flow_balance_pair_ablation_round3_summary.json"
    results_path = artifact_dir / "soft_target_flow_balance_pair_ablation_round3_results.csv"
    report_path = report_dir / "soft_target_flow_balance_pair_ablation_round3_summary.md"

    summary_path.write_text(json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    details_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    write_report(report_path, summary_df.to_dict(orient="records"))

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "results_path": str(results_path),
                "report_path": str(report_path),
                "best_variant": summary_df.iloc[0].to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
