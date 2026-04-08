from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_stage0_baseline import (
    fit_autogluon_multiclass,
    make_multiclass_baseline,
    multiclass_metrics,
)
from run_autogluon_stage1_lag_scale import build_label_frame
from run_autogluon_stage1_quickcheck import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    regression_metrics,
    resolve_path,
)
from run_autogluon_stage2_dynamic_morphology import build_stage2_table
from run_autogluon_stage3_regime_state import build_state_package
from run_autogluon_stage4_interactions import build_interaction_frame
from run_autogluon_stage5_quality import build_quality_table
from run_autogluon_stage6_centered_quality import build_centered_quality_table


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage7_final_selection.yaml"


def _load_summary(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def load_references(config_path: Path, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    refs: dict[str, pd.DataFrame] = {}
    for key in ("stage1_summary_path", "stage3_summary_path", "stage4_summary_path", "stage5_summary_path", "stage6_summary_path"):
        path = resolve_path(config_path.parent, config["paths"][key])
        if path is None or not path.exists():
            raise ValueError(f"{key} must exist before running S7 final selection.")
        refs[key] = _load_summary(path)
    return refs


def select_task_priors(refs: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    centered_stage6 = refs["stage6_summary_path"]
    centered_row = centered_stage6.iloc[0].to_dict()

    five_stage4 = refs["stage4_summary_path"]
    five_row = five_stage4[five_stage4["task_name"] == "five_bin"].iloc[0].to_dict()

    stage1 = refs["stage1_summary_path"]
    centered_s1 = stage1[(stage1["task_name"] == "centered_desirability") & (stage1["variant_name"] == centered_row["selected_stage1_variant"])].iloc[0].to_dict()
    five_s1 = stage1[(stage1["task_name"] == "five_bin") & (stage1["variant_name"] == five_row["selected_stage1_variant"])].iloc[0].to_dict()

    return {
        "centered_desirability": {
            "tau_minutes": int(centered_row["tau_minutes"]),
            "window_minutes": int(centered_row["window_minutes"]),
            "stage1_variant": centered_row["selected_stage1_variant"],
            "state_package": centered_row.get("selected_state_package"),
            "interaction_package": centered_row.get("selected_interaction_package"),
            "quality_package": centered_row.get("selected_quality_package"),
            "centered_quality_package": centered_row.get("centered_quality_package"),
            "stage0_ref": float(centered_s1["stage0_autogluon_ref"]),
        },
        "five_bin": {
            "tau_minutes": int(five_row["tau_minutes"]),
            "window_minutes": int(five_row["window_minutes"]),
            "stage1_variant": five_row["selected_stage1_variant"],
            "state_package": five_row.get("selected_state_package"),
            "interaction_package": five_row.get("interaction_package"),
            "quality_package": None,
            "centered_quality_package": None,
            "stage0_ref": float(five_s1["stage0_autogluon_ref"]),
        },
    }


def combo_specs(task_name: str, priors: dict[str, Any]) -> list[dict[str, Any]]:
    if task_name == "centered_desirability":
        return [
            {"name": "lag_only", "use_state": False, "use_interaction": False, "use_quality": False, "use_centered_quality": False},
            {"name": "lag_plus_interaction", "use_state": False, "use_interaction": True, "use_quality": False, "use_centered_quality": False},
            {"name": "lag_plus_interaction_plus_quality", "use_state": False, "use_interaction": True, "use_quality": True, "use_centered_quality": False},
            {"name": "lag_plus_interaction_plus_quality_plus_centered", "use_state": False, "use_interaction": True, "use_quality": True, "use_centered_quality": True},
        ]
    return [
        {"name": "lag_only", "use_state": False, "use_interaction": False, "use_quality": False, "use_centered_quality": False},
        {"name": "lag_plus_state", "use_state": True, "use_interaction": False, "use_quality": False, "use_centered_quality": False},
        {"name": "lag_plus_state_plus_interaction", "use_state": True, "use_interaction": True, "use_quality": False, "use_centered_quality": False},
    ]


def approximate_drop_duplicate_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    kept: list[str] = []
    seen: set[int] = set()
    for column in columns:
        series = frame[column]
        hashed = pd.util.hash_pandas_object(series.astype(str), index=False).sum()
        if int(hashed) in seen:
            continue
        seen.add(int(hashed))
        kept.append(column)
    return kept


def preclean_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    max_missing_ratio: float,
    unique_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cols = list(train_x.columns)
    keep: list[str] = []
    for column in cols:
        series = train_x[column]
        if float(series.isna().mean()) > max_missing_ratio:
            continue
        if int(series.nunique(dropna=False)) <= unique_threshold:
            continue
        keep.append(column)
    keep = approximate_drop_duplicate_columns(train_x, keep)
    return train_x[keep].copy(), test_x[keep].copy(), keep


def supervised_select(
    train_x: pd.DataFrame,
    y: np.ndarray,
    task_name: str,
    top_k: int,
) -> list[str]:
    if train_x.empty:
        return []
    x_filled = train_x.copy()
    for column in x_filled.columns:
        series = pd.to_numeric(x_filled[column], errors="coerce")
        if series.notna().any():
            x_filled[column] = series.fillna(series.median())
        else:
            x_filled[column] = 0.0
    try:
        if task_name == "centered_desirability":
            scores = mutual_info_regression(x_filled, y, random_state=42)
        else:
            scores = mutual_info_classif(x_filled, y, random_state=42)
    except Exception:
        return list(train_x.columns[: min(top_k, train_x.shape[1])])
    score_frame = pd.DataFrame({"feature": train_x.columns, "score": scores})
    score_frame = score_frame.sort_values(["score", "feature"], ascending=[False, True]).reset_index(drop=True)
    return score_frame["feature"].head(min(top_k, len(score_frame))).tolist()


def build_current_task_frame(
    task_name: str,
    snapshot: pd.DataFrame,
    priors: dict[str, Any],
    combo: dict[str, Any],
    quality_table: pd.DataFrame | None,
    centered_table: pd.DataFrame | None,
) -> pd.DataFrame:
    frame = snapshot.copy()
    if combo["use_quality"] and quality_table is not None:
        frame = frame.merge(quality_table, on="decision_time", how="left")
    if combo["use_centered_quality"] and centered_table is not None:
        frame = frame.merge(centered_table, on="decision_time", how="left")
    return frame


def choose_state_package(priors: dict[str, Any]) -> dict[str, Any] | None:
    name = priors.get("state_package")
    if not name:
        return None
    if name == "time_context":
        return {"name": "time_context", "include_time_context": True, "cluster_count": 0}
    if name == "cluster_k5":
        return {"name": "cluster_k5", "include_time_context": False, "cluster_count": 5}
    return {"name": "time_plus_cluster_k5", "include_time_context": True, "cluster_count": 5}


def compose_fold_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_feature_columns: list[str],
    priors: dict[str, Any],
    combo: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = [train[base_feature_columns]]
    test_parts = [test[base_feature_columns]]

    if combo["use_state"]:
        package = choose_state_package(priors)
        if package is not None:
            train_state, test_state, _ = build_state_package(train, test, base_feature_columns, package)
            train_parts.append(train_state)
            test_parts.append(test_state)

    if combo["use_interaction"] and priors.get("interaction_package"):
        interaction_name = str(priors["interaction_package"])
        train_parts.append(build_interaction_frame(train, int(priors["tau_minutes"]), int(priors["window_minutes"]), interaction_name))
        test_parts.append(build_interaction_frame(test, int(priors["tau_minutes"]), int(priors["window_minutes"]), interaction_name))

    if combo["use_quality"]:
        quality_cols = [column for column in train.columns if column.startswith("quality__")]
        if quality_cols:
            train_parts.append(train[quality_cols])
            test_parts.append(test[quality_cols])

    if combo["use_centered_quality"]:
        centered_cols = [column for column in train.columns if column.startswith("centered__")]
        if centered_cols:
            train_parts.append(train[centered_cols])
            test_parts.append(test[centered_cols])

    return pd.concat(train_parts, axis=1), pd.concat(test_parts, axis=1)


def build_feature_catalog(task_name: str, combo_name: str, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in features:
        if "__" in column:
            sensor, suffix = column.split("__", 1)
        else:
            sensor, suffix = "meta", column
        if column.startswith("interaction__"):
            family = "process_interaction"
        elif column.startswith("quality__"):
            family = "quality"
        elif column.startswith("centered__"):
            family = "centered_quality"
        elif column.startswith("state__"):
            family = "regime_state"
        else:
            family = "lag_scale"
        rows.append(
            {
                "task_name": task_name,
                "combo_name": combo_name,
                "feature_name": column,
                "sensor": sensor,
                "feature_suffix": suffix,
                "family": family,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon S7 final feature selection.")
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

    refs = load_references(config_path, config)
    priors_by_task = select_task_priors(refs)
    labeled = build_label_frame(config_path, config)
    dcs = load_dcs_frame(
        resolve_path(config_path.parent, config["paths"]["dcs_main_path"]),
        resolve_path(config_path.parent, config["paths"].get("dcs_supplemental_path")),
    )
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    final_feature_tables: list[pd.DataFrame] = []
    feature_catalog_parts: list[pd.DataFrame] = []

    for task_name, priors in priors_by_task.items():
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
            continue
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

        task_summary_candidates: list[dict[str, Any]] = []
        for combo in combo_specs(task_name, priors):
            prepared = build_current_task_frame(task_name, snapshot, priors, combo, quality_table, centered_table)
            top_k_candidates = list(config["tasks"][task_name]["top_k_candidates"])
            for top_k in top_k_candidates:
                fold_summaries: list[dict[str, Any]] = []
                agg: dict[str, list[dict[str, float]]] = {"simple_baseline": [], "autogluon": []}
                raw_feature_count = 0
                cleaned_feature_count = 0
                selected_feature_count = 0
                chosen_cols_fold1: list[str] = []

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

                    if task_name == "centered_desirability":
                        y_train = train["target_centered_desirability"].to_numpy(dtype=float)
                        y_test = test["target_centered_desirability"].to_numpy(dtype=float)
                    else:
                        y_train = train["target_five_bin"].to_numpy(dtype=int)
                        y_test = test["target_five_bin"].to_numpy(dtype=int)

                    selected_cols = supervised_select(train_clean, y_train, task_name, int(top_k))
                    if not selected_cols:
                        selected_cols = cleaned_cols
                    selected_feature_count = int(len(selected_cols))
                    if fold_idx == 1:
                        chosen_cols_fold1 = list(selected_cols)
                        feature_catalog_parts.append(build_feature_catalog(task_name, f"{combo['name']}__top{top_k}", selected_cols))

                    train_sel = train_clean[selected_cols].copy()
                    test_sel = test_clean[selected_cols].copy()

                    if task_name == "centered_desirability":
                        baseline = make_regression_baseline()
                        baseline.fit(train_sel, y_train)
                        base_pred = baseline.predict(test_sel).astype(float)
                        base_metric = regression_metrics(y_test, base_pred, test["is_in_spec"].to_numpy(dtype=int))

                        model_path = artifact_dir / f"ag_stage7_{task_name}_{combo['name']}_top{top_k}_{run_id}_fold{fold_idx}"
                        ag_pred, model_best = fit_autogluon_fold(
                            train_df=pd.concat([train_sel, train[["target_centered_desirability"]]], axis=1).copy(),
                            test_df=pd.concat([test_sel, test[["target_centered_desirability"]]], axis=1).copy(),
                            label="target_centered_desirability",
                            problem_type="regression",
                            eval_metric="root_mean_squared_error",
                            model_path=model_path,
                            ag_config=config["autogluon"],
                        )
                        ag_metric = regression_metrics(y_test, ag_pred.astype(float), test["is_in_spec"].to_numpy(dtype=int))
                    else:
                        all_labels = sorted(prepared["target_five_bin"].dropna().astype(int).unique().tolist())
                        baseline = make_multiclass_baseline()
                        baseline.fit(train_sel, y_train)
                        base_proba = baseline.predict_proba(test_sel).astype(float)
                        base_frame = pd.DataFrame(base_proba, columns=baseline.named_steps["clf"].classes_.tolist()).reindex(columns=all_labels, fill_value=0.0)
                        base_frame = base_frame.div(base_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                        base_metric = multiclass_metrics(y_test, base_frame.to_numpy(dtype=float), class_labels=all_labels)

                        model_path = artifact_dir / f"ag_stage7_{task_name}_{combo['name']}_top{top_k}_{run_id}_fold{fold_idx}"
                        ag_frame, model_best, _, _ = fit_autogluon_multiclass(
                            train_df=pd.concat([train_sel, train[["target_five_bin"]]], axis=1).copy(),
                            test_df=pd.concat([test_sel, test[["target_five_bin"]]], axis=1).copy(),
                            label="target_five_bin",
                            model_path=model_path,
                            ag_config=config["autogluon"],
                        )
                        ag_frame = ag_frame.reindex(columns=all_labels, fill_value=0.0)
                        ag_frame = ag_frame.div(ag_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                        ag_metric = multiclass_metrics(y_test, ag_frame.to_numpy(dtype=float), class_labels=all_labels)

                    rows.append(
                        {
                            "stage": "stage7_final_selection",
                            "task_name": task_name,
                            "combo_name": combo["name"],
                            "top_k": int(top_k),
                            "fold": int(fold_idx),
                            "samples_train": int(len(train)),
                            "samples_test": int(len(test)),
                            "raw_feature_count": raw_feature_count,
                            "cleaned_feature_count": cleaned_feature_count,
                            "selected_feature_count": selected_feature_count,
                            "framework": "simple_baseline",
                            "data_source_condition": config["data_source_statement"]["source_condition"],
                            **base_metric,
                        }
                    )
                    rows.append(
                        {
                            "stage": "stage7_final_selection",
                            "task_name": task_name,
                            "combo_name": combo["name"],
                            "top_k": int(top_k),
                            "fold": int(fold_idx),
                            "samples_train": int(len(train)),
                            "samples_test": int(len(test)),
                            "raw_feature_count": raw_feature_count,
                            "cleaned_feature_count": cleaned_feature_count,
                            "selected_feature_count": selected_feature_count,
                            "framework": "autogluon",
                            "data_source_condition": config["data_source_statement"]["source_condition"],
                            "autogluon_model_best": model_best,
                            **ag_metric,
                        }
                    )
                    agg["simple_baseline"].append(base_metric)
                    agg["autogluon"].append(ag_metric)
                    fold_summaries.append({"fold": int(fold_idx), "selected_columns_fold_size": int(len(selected_cols)), "baseline": base_metric, "autogluon": ag_metric})

                if task_name == "centered_desirability":
                    baseline_mean = float(np.nanmean([m["mae"] for m in agg["simple_baseline"]]))
                    ag_mean = float(np.nanmean([m["mae"] for m in agg["autogluon"]]))
                    task_summary_candidates.append(
                        {
                            "task_name": task_name,
                            "combo_name": combo["name"],
                            "top_k": int(top_k),
                            "stage0_autogluon_ref": float(priors["stage0_ref"]),
                            "baseline_mean_mae": baseline_mean,
                            "autogluon_mean_mae": ag_mean,
                            "beats_stage0": ag_mean < float(priors["stage0_ref"]),
                            "raw_feature_count": raw_feature_count,
                            "cleaned_feature_count": cleaned_feature_count,
                            "selected_feature_count": selected_feature_count,
                            "selected_features_fold1": chosen_cols_fold1,
                            "fold_summaries": fold_summaries,
                        }
                    )
                else:
                    baseline_mean = float(np.nanmean([m["multiclass_log_loss"] for m in agg["simple_baseline"]]))
                    ag_mean = float(np.nanmean([m["multiclass_log_loss"] for m in agg["autogluon"]]))
                    task_summary_candidates.append(
                        {
                            "task_name": task_name,
                            "combo_name": combo["name"],
                            "top_k": int(top_k),
                            "stage0_autogluon_ref": float(priors["stage0_ref"]),
                            "baseline_mean_multiclass_log_loss": baseline_mean,
                            "autogluon_mean_multiclass_log_loss": ag_mean,
                            "beats_stage0": ag_mean < float(priors["stage0_ref"]),
                            "raw_feature_count": raw_feature_count,
                            "cleaned_feature_count": cleaned_feature_count,
                            "selected_feature_count": selected_feature_count,
                            "selected_features_fold1": chosen_cols_fold1,
                            "fold_summaries": fold_summaries,
                        }
                    )

        if not task_summary_candidates:
            continue
        if task_name == "centered_desirability":
            best = sorted(task_summary_candidates, key=lambda row: row["autogluon_mean_mae"])[0]
        else:
            best = sorted(task_summary_candidates, key=lambda row: row["autogluon_mean_multiclass_log_loss"])[0]
        summary_rows.append(best)

        best_combo = next(combo for combo in combo_specs(task_name, priors) if combo["name"] == best["combo_name"])
        full_prepared = build_current_task_frame(task_name, snapshot, priors, best_combo, quality_table, centered_table)
        full_x, _ = compose_fold_features(full_prepared, full_prepared, base_feature_columns, priors, best_combo)
        full_clean, _, cleaned_cols = preclean_features(
            full_x,
            full_x,
            max_missing_ratio=float(config["preclean"]["max_missing_ratio"]),
            unique_threshold=int(config["preclean"]["near_constant_unique_threshold"]),
        )
        if task_name == "centered_desirability":
            y_full = full_prepared["target_centered_desirability"].to_numpy(dtype=float)
        else:
            y_full = full_prepared["target_five_bin"].to_numpy(dtype=int)
        selected_cols = supervised_select(full_clean, y_full, task_name, int(best["top_k"]))
        if not selected_cols:
            selected_cols = cleaned_cols
        final_table = pd.concat(
            [
                pd.DataFrame(
                    {
                        "task_name": task_name,
                        "decision_time": full_prepared["decision_time"],
                        "sample_time": full_prepared["sample_time"],
                        "t90": full_prepared["t90"],
                        "best_combo_name": best["combo_name"],
                        "best_top_k": int(best["top_k"]),
                    }
                ),
                full_clean[selected_cols].reset_index(drop=True),
            ],
            axis=1,
        )
        final_feature_tables.append(final_table)

    results_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    feature_catalog = pd.concat(feature_catalog_parts, ignore_index=True) if feature_catalog_parts else pd.DataFrame()
    final_feature_table = pd.concat(final_feature_tables, ignore_index=True, sort=False) if final_feature_tables else pd.DataFrame()

    results_path = artifact_dir / "stage7_final_results.csv"
    summary_path = artifact_dir / "stage7_final_summary.json"
    feature_table_path = artifact_dir / "stage7_final_feature_table.csv"
    feature_catalog_path = artifact_dir / "stage7_final_feature_catalog.csv"
    audit_path = report_dir / "stage7_final_summary.md"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary_df.to_json(summary_path, orient="records", force_ascii=False, indent=2)
    final_feature_table.to_csv(feature_table_path, index=False, encoding="utf-8-sig")
    feature_catalog.to_csv(feature_catalog_path, index=False, encoding="utf-8-sig")

    audit_lines = [
        "# Stage 7 Final Summary",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- Stage 7 combines only previously validated positive feature packages.",
        "- Unsupervised pre-cleaning and supervised feature selection are both performed in a fold-safe manner.",
        "",
        "## Selected Priors",
        "",
        json.dumps(priors_by_task, ensure_ascii=False, indent=2),
        "",
        "## Summary Rows",
        "",
        json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
    ]
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "feature_table_path": str(feature_table_path),
                "feature_catalog_path": str(feature_catalog_path),
                "audit_path": str(audit_path),
                "summary": summary_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
