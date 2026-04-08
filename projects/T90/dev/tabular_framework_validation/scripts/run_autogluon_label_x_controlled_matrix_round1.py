from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_centered_desirability_outspec_eval import centered_outspec_metrics
from run_autogluon_soft_probability_weak_compression_search import build_variant_snapshot
from run_autogluon_stage1_lag_scale import build_label_frame
from run_autogluon_stage1_quickcheck import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    resolve_path,
)
from run_autogluon_stage2_dynamic_morphology import build_stage2_table
from run_autogluon_stage2_feature_engineering import select_features_fold
from run_autogluon_stage2_soft_probability import make_soft_probability_target, soft_probability_metrics
from run_autogluon_stage5_quality import build_quality_table
from run_autogluon_stage7_final_selection import load_references, select_task_priors
from run_autogluon_stage4_interactions import build_interaction_frame


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_label_x_controlled_matrix_round1.yaml"


def _load_best_centered_combo(summary_path: Path) -> dict[str, Any]:
    summary_df = pd.DataFrame(json.loads(summary_path.read_text(encoding="utf-8")))
    centered = summary_df[summary_df["task_name"] == "centered_desirability"].copy()
    if centered.empty:
        raise ValueError("No centered_desirability row found in stage7 summary.")
    centered = centered.sort_values(["autogluon_mean_mae", "combo_name", "top_k"], ascending=[True, True, True])
    return centered.iloc[0].to_dict()


def _prepare_labeled(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    labeled = build_label_frame(config_path, config).copy()
    soft_name = str(config["label_fuzziness"]["target_name"])
    if soft_name not in labeled.columns:
        labeled[soft_name] = make_soft_probability_target(
            labeled["t90"],
            center=float(config["target_spec"]["center"]),
            tolerance=float(config["target_spec"]["tolerance"]),
            boundary_softness=float(config["label_fuzziness"]["boundary_softness"]),
            rule=str(config["label_fuzziness"]["rule"]),
        )
    labeled["target_centered_desirability"] = np.maximum(
        0.0,
        1.0 - (labeled["t90"] - float(config["target_spec"]["center"])).abs() / float(config["target_spec"]["tolerance"]),
    )
    labeled["sample_time"] = pd.to_datetime(labeled["sample_time"], errors="coerce")
    labeled = labeled.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)
    return labeled


def _build_soft_x_frame(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame, _ = build_variant_snapshot(config_path, config, "current_whole_window_ref")
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], errors="coerce")
    frame = frame.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)
    return frame


def _build_centered_x_frame(config_path: Path, config: dict[str, Any], labeled: pd.DataFrame) -> pd.DataFrame:
    refs = load_references(config_path, config)
    priors = select_task_priors(refs)["centered_desirability"]
    stage7_summary_path = resolve_path(config_path.parent, config["paths"]["stage7_summary_path"])
    if stage7_summary_path is None:
        raise ValueError("stage7_summary_path must be configured.")
    best_combo = _load_best_centered_combo(stage7_summary_path)

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
        raise ValueError("Centered-X snapshot is empty.")

    frame = snapshot.copy()
    if priors.get("quality_package"):
        quality_table = build_quality_table(
            labeled_samples=labeled,
            dcs=dcs,
            tau_minutes=int(priors["tau_minutes"]),
            window_minutes=int(priors["window_minutes"]),
            enabled_features=list(config["quality_feature_map"][priors["quality_package"]]),
            min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
        )
        frame = frame.merge(quality_table, on="decision_time", how="left")

    if priors.get("interaction_package") and str(best_combo["combo_name"]) in {
        "lag_plus_interaction",
        "lag_plus_interaction_plus_quality",
        "lag_plus_interaction_plus_quality_plus_centered",
    }:
        interaction_frame = build_interaction_frame(
            snapshot,
            int(priors["tau_minutes"]),
            int(priors["window_minutes"]),
            str(priors["interaction_package"]),
        )
        frame = pd.concat([frame.reset_index(drop=True), interaction_frame.reset_index(drop=True)], axis=1)

    feature_columns = [column for column in frame.columns if "__" in column]
    meta_columns = [
        "sample_time",
        "decision_time",
        "t90",
        "is_in_spec",
        "is_out_of_spec",
        "is_above_spec",
        "is_below_spec",
        "target_centered_desirability",
    ]
    soft_name = str(config["label_fuzziness"]["target_name"])
    if soft_name in labeled.columns:
        frame = frame.merge(labeled[["sample_time", soft_name]], on="sample_time", how="left")
        meta_columns.append(soft_name)
    keep_columns = [column for column in meta_columns if column in frame.columns] + feature_columns
    frame = frame[keep_columns].copy()
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], errors="coerce")
    frame = frame.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)
    return frame


def _build_union_x_frame(soft_frame: pd.DataFrame, centered_frame: pd.DataFrame, soft_label_name: str) -> pd.DataFrame:
    left_meta = [
        "sample_time",
        "t90",
        "is_in_spec",
        "is_out_of_spec",
        "is_above_spec",
        "is_below_spec",
        "target_centered_desirability",
        soft_label_name,
    ]
    soft_features = [column for column in soft_frame.columns if "__" in column]
    centered_features = [column for column in centered_frame.columns if "__" in column]

    left = soft_frame[left_meta + soft_features].copy()
    right = centered_frame[["sample_time"] + centered_features].copy()
    rename_map = {column: f"centeredX__{column}" for column in centered_features if column in left.columns}
    right = right.rename(columns=rename_map)
    merged = left.merge(right, on="sample_time", how="inner")
    merged = merged.sort_values("sample_time").reset_index(drop=True)
    return merged


def _drop_duplicate_columns(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep: list[str] = []
    seen: set[int] = set()
    for column in train_x.columns:
        hashed = int(pd.util.hash_pandas_object(train_x[column].astype(str), index=False).sum())
        if hashed in seen:
            continue
        seen.add(hashed)
        keep.append(column)
    return train_x[keep].copy(), test_x[keep].copy()


def _preclean(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    max_missing_ratio: float,
    unique_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    keep: list[str] = []
    for column in train_x.columns:
        series = train_x[column]
        if float(series.isna().mean()) > max_missing_ratio:
            continue
        if int(series.nunique(dropna=False)) <= unique_threshold:
            continue
        keep.append(column)
    train_clean = train_x[keep].copy()
    test_clean = test_x[keep].copy()
    train_clean, test_clean = _drop_duplicate_columns(train_clean, test_clean)
    return train_clean, test_clean, list(train_clean.columns)


def _evaluate_task(
    frame: pd.DataFrame,
    task_name: str,
    x_recipe_name: str,
    config: dict[str, Any],
    artifact_dir: Path,
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    soft_label_name = str(config["label_fuzziness"]["target_name"])
    top_k = int(config["selection"]["shared_top_k"])
    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    raw_feature_count = 0
    cleaned_feature_count = 0
    selected_feature_count = 0
    selected_features_fold1: list[str] = []

    feature_columns = [column for column in frame.columns if "__" in column]
    label_name = soft_label_name if task_name == "soft_target" else "target_centered_desirability"
    eval_name = "soft_brier" if task_name == "soft_target" else "mae"

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
            train_y=train[label_name],
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
        y_train = train[label_name].to_numpy(dtype=float)
        y_test = test[label_name].to_numpy(dtype=float)

        baseline = make_regression_baseline()
        baseline.fit(train_sel, y_train)
        base_pred = baseline.predict(test_sel).astype(float)

        model_path = artifact_dir / f"ag_label_x_round1_{task_name}_{x_recipe_name}_{run_id}_fold{fold_idx}"
        ag_pred, model_best = fit_autogluon_fold(
            train_df=pd.concat([train_sel, train[[label_name]]], axis=1).copy(),
            test_df=pd.concat([test_sel, test[[label_name]]], axis=1).copy(),
            label=label_name,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )
        ag_pred = ag_pred.astype(float)

        if task_name == "soft_target":
            base_metric = soft_probability_metrics(
                y_test,
                base_pred,
                test["is_out_of_spec"].to_numpy(dtype=int),
            )
            ag_metric = soft_probability_metrics(
                y_test,
                ag_pred,
                test["is_out_of_spec"].to_numpy(dtype=int),
            )
            primary_metric_name = "soft_brier"
        else:
            base_metric = centered_outspec_metrics(
                y_test,
                base_pred,
                test["is_in_spec"].to_numpy(dtype=int),
            )
            ag_metric = centered_outspec_metrics(
                y_test,
                ag_pred,
                test["is_in_spec"].to_numpy(dtype=int),
            )
            primary_metric_name = "mae"

        for framework_name, metrics, preds in (
            ("simple_baseline", base_metric, base_pred),
            ("autogluon", ag_metric, ag_pred),
        ):
            row = {
                "task_name": task_name,
                "x_recipe_name": x_recipe_name,
                "framework": framework_name,
                "fold": int(fold_idx),
                "samples_train": int(len(train)),
                "samples_test": int(len(test)),
                "raw_feature_count": raw_feature_count,
                "cleaned_feature_count": cleaned_feature_count,
                "selected_feature_count": selected_feature_count,
                **metrics,
            }
            if framework_name == "autogluon":
                row["autogluon_model_best"] = model_best
            rows.append(row)

        fold_summaries.append(
            {
                "fold": int(fold_idx),
                "selected_feature_count": selected_feature_count,
                "baseline_primary": float(base_metric[primary_metric_name]),
                "autogluon_primary": float(ag_metric[primary_metric_name]),
                "autogluon_hard_out_ap_diagnostic": float(ag_metric["hard_out_ap_diagnostic"]),
                "autogluon_hard_out_auc_diagnostic": float(ag_metric["hard_out_auc_diagnostic"]),
            }
        )

    results = pd.DataFrame(rows)
    ag_rows = results[results["framework"] == "autogluon"].copy()
    baseline_rows = results[results["framework"] == "simple_baseline"].copy()

    summary = {
        "task_name": task_name,
        "x_recipe_name": x_recipe_name,
        "raw_feature_count": raw_feature_count,
        "cleaned_feature_count": cleaned_feature_count,
        "selected_feature_count": selected_feature_count,
        "selected_features_fold1": selected_features_fold1,
        "baseline_mean_primary": float(baseline_rows[eval_name].mean()),
        "autogluon_mean_primary": float(ag_rows[eval_name].mean()),
        "baseline_mean_hard_out_ap_diagnostic": float(baseline_rows["hard_out_ap_diagnostic"].mean()),
        "autogluon_mean_hard_out_ap_diagnostic": float(ag_rows["hard_out_ap_diagnostic"].mean()),
        "baseline_mean_hard_out_auc_diagnostic": float(baseline_rows["hard_out_auc_diagnostic"].mean()),
        "autogluon_mean_hard_out_auc_diagnostic": float(ag_rows["hard_out_auc_diagnostic"].mean()),
        "fold_summaries": fold_summaries,
    }
    if task_name == "soft_target":
        summary["baseline_mean_soft_mae"] = float(baseline_rows["soft_mae"].mean())
        summary["autogluon_mean_soft_mae"] = float(ag_rows["soft_mae"].mean())
    else:
        summary["baseline_mean_in_spec_auc_from_desirability"] = float(
            baseline_rows["in_spec_auc_from_desirability"].mean()
        )
        summary["autogluon_mean_in_spec_auc_from_desirability"] = float(
            ag_rows["in_spec_auc_from_desirability"].mean()
        )
    return results, summary


def write_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Label-X Controlled Matrix Round 1",
        "",
        "## Scope",
        "",
        "- Use the same model family: AutoGluon regression",
        "- Use the same time split: TimeSeriesSplit(5)",
        "- Use the same fold-safe feature selection budget within this round",
        "- Only vary label semantics and X recipe family",
        "",
        "## Summary Rows",
        "",
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run round-1 controlled label-vs-X analysis for T90 tabular validation.")
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

    labeled = _prepare_labeled(config_path, config)
    soft_label_name = str(config["label_fuzziness"]["target_name"])
    soft_x_frame = _build_soft_x_frame(config_path, config)
    centered_x_frame = _build_centered_x_frame(config_path, config, labeled)
    union_x_frame = _build_union_x_frame(soft_x_frame, centered_x_frame, soft_label_name)

    x_frames = {
        "soft_x_only": soft_x_frame,
        "centered_x_only": centered_x_frame,
        "union_x": union_x_frame,
    }

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    detail_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for task_name in ("soft_target", "centered_desirability"):
        for x_recipe_name, frame in x_frames.items():
            results, summary = _evaluate_task(
                frame=frame,
                task_name=task_name,
                x_recipe_name=x_recipe_name,
                config=config,
                artifact_dir=artifact_dir,
                run_id=run_id,
            )
            detail_parts.append(results)
            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.concat(detail_parts, ignore_index=True)

    summary_path = artifact_dir / "label_x_controlled_matrix_round1_summary.json"
    results_path = artifact_dir / "label_x_controlled_matrix_round1_results.csv"
    report_path = report_dir / "label_x_controlled_matrix_round1_summary.md"

    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    details_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    write_report(report_path, summary_rows)

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "results_path": str(results_path),
                "report_path": str(report_path),
                "summary_rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
