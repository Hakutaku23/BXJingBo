from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
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


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[4]
V3_ROOT = WORKSPACE_ROOT / "projects" / "T90" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import load_dcs_frame, summarize_numeric_window


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_lag_reality_check.yaml"


def _load_stage1_reference(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config_path.parent, config["paths"].get("stage1_summary_path"))
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def _window_feature_frame(
    window: pd.DataFrame,
    tau_minutes: int,
    window_minutes: int,
    stats: set[str],
) -> dict[str, float]:
    raw_features = summarize_numeric_window(window, time_column="time")
    filtered: dict[str, float] = {}
    for key, value in raw_features.items():
        sensor, stat = key.split("__", 1)
        if stat in stats:
            filtered[f"{sensor}__{stat}"] = float(value)
    return filtered


def build_variant_snapshot(
    labeled_samples: pd.DataFrame,
    dcs: pd.DataFrame,
    variant: dict[str, Any],
    stats: list[str],
    min_points_per_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_frame = labeled_samples.copy()
    sample_frame["sample_time"] = pd.to_datetime(sample_frame["sample_time"], errors="coerce")
    sample_frame = sample_frame.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)

    dcs_frame = dcs.copy()
    dcs_frame["time"] = pd.to_datetime(dcs_frame["time"], errors="coerce")
    dcs_frame = dcs_frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")

    default_tau = int(variant["default_tau_minutes"])
    window_minutes = int(variant["window_minutes"])
    tau_overrides = {str(key): int(value) for key, value in variant.get("tau_overrides", {}).items()}
    unique_taus = sorted({default_tau, *tau_overrides.values()})
    stats_set = set(stats)

    rows: list[dict[str, Any]] = []
    feature_catalog_rows: list[dict[str, Any]] = []
    seen_feature_catalog: set[tuple[str, str, int]] = set()

    for record in sample_frame.itertuples(index=False):
        decision_time = pd.Timestamp(record.sample_time)
        tau_feature_maps: dict[int, dict[str, float]] = {}
        valid = True
        for tau in unique_taus:
            window_end = decision_time - pd.Timedelta(minutes=tau)
            window_start = window_end - pd.Timedelta(minutes=window_minutes)
            left = np.searchsorted(dcs_times, window_start.to_datetime64(), side="left")
            right = np.searchsorted(dcs_times, window_end.to_datetime64(), side="right")
            window = dcs_frame.iloc[left:right]
            if len(window) < min_points_per_window:
                valid = False
                break
            tau_feature_maps[tau] = _window_feature_frame(
                window=window,
                tau_minutes=tau,
                window_minutes=window_minutes,
                stats=stats_set,
            )
        if not valid:
            continue

        row = {
            "decision_time": decision_time,
            "sample_time": decision_time,
            "t90": getattr(record, "t90", np.nan),
            "is_in_spec": getattr(record, "is_in_spec", False),
            "is_out_of_spec": getattr(record, "is_out_of_spec", False),
            "is_above_spec": getattr(record, "is_above_spec", False),
            "is_below_spec": getattr(record, "is_below_spec", False),
            "target_centered_desirability": getattr(record, "target_centered_desirability", np.nan),
            "target_five_bin": getattr(record, "target_five_bin", pd.NA),
            "variant_name": str(variant["name"]),
            "window_minutes": window_minutes,
            "default_tau_minutes": default_tau,
        }

        feature_names = set()
        for tau_features in tau_feature_maps.values():
            feature_names.update(tau_features.keys())

        for feature_name in sorted(feature_names):
            sensor, stat = feature_name.split("__", 1)
            sensor_tau = tau_overrides.get(sensor, default_tau)
            if feature_name in tau_feature_maps[sensor_tau]:
                row[feature_name] = tau_feature_maps[sensor_tau][feature_name]
                catalog_key = (sensor, stat, sensor_tau)
                if catalog_key not in seen_feature_catalog:
                    feature_catalog_rows.append(
                        {
                            "variant_name": str(variant["name"]),
                            "feature_name": feature_name,
                            "sensor": sensor,
                            "stat": stat,
                            "tau_minutes": sensor_tau,
                            "window_minutes": window_minutes,
                            "lag_strategy": "override" if sensor in tau_overrides else "default",
                        }
                    )
                    seen_feature_catalog.add(catalog_key)

        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)
    if not frame.empty and "target_five_bin" in frame.columns:
        frame = frame.dropna(subset=["target_five_bin"]).copy()
        frame["target_five_bin"] = frame["target_five_bin"].astype(int)
    catalog = pd.DataFrame(feature_catalog_rows).sort_values(["tau_minutes", "sensor", "stat"]).reset_index(drop=True)
    return frame, catalog


def align_common_snapshot_time(
    variant_tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    common_times: set[pd.Timestamp] | None = None
    for frame in variant_tables.values():
        times = set(pd.to_datetime(frame["decision_time"]))
        common_times = times if common_times is None else common_times & times
    common_times = common_times or set()
    common_index = pd.DataFrame({"decision_time": sorted(common_times)})

    aligned: dict[str, pd.DataFrame] = {}
    for variant_name, frame in variant_tables.items():
        merged = common_index.merge(frame, on="decision_time", how="left", validate="one_to_one")
        merged = merged.sort_values("decision_time").reset_index(drop=True)
        aligned[variant_name] = merged
    return aligned, common_index


def write_markdown_report(
    report_path: Path,
    discovered_scripts: list[str],
    summary_rows: list[dict[str, Any]],
    common_sample_count: int,
    stage1_reference: pd.DataFrame,
) -> None:
    tasks_present = sorted({str(row["task_name"]) for row in summary_rows})
    lines = [
        "# AutoGluon Lag Reality Check",
        "",
        "## Discovered AutoGluon Experiment Scripts",
        "",
    ]
    for item in discovered_scripts:
        lines.append(f"- `{item}`")

    lines.extend(
        [
            "",
            "## Shared Evaluation Setting",
            "",
            f"- common_samples_scored: {common_sample_count}",
            f"- labels reused from AutoGluon branch: {', '.join(f'`{task}`' for task in tasks_present)}",
            "- metrics reused from AutoGluon branch:",
        ]
    )
    if "centered_desirability" in tasks_present:
        lines.append("- `centered_desirability`: `MAE / RMSE / rank_correlation / in_spec_auc_from_desirability`")
    if "five_bin" in tasks_present:
        lines.append("- `five_bin`: `macro_f1 / balanced_accuracy / multiclass_log_loss`")
    lines.extend(["", "## Historical Stage 1 References", ""])
    if stage1_reference.empty:
        lines.append("- unavailable")
    else:
        for task_name in tasks_present:
            task_rows = stage1_reference[stage1_reference["task_name"] == task_name].copy()
            if task_rows.empty:
                continue
            if task_name == "centered_desirability":
                best_row = task_rows.sort_values("autogluon_mean_mae", ascending=True).iloc[0]
                lines.append(
                    f"- `{task_name}` best historical lag package: `{best_row['variant_name']}`"
                    f", autogluon_mean_mae = {best_row['autogluon_mean_mae']:.4f}"
                )
            else:
                best_row = task_rows.sort_values("autogluon_mean_multiclass_log_loss", ascending=True).iloc[0]
                lines.append(
                    f"- `{task_name}` best historical lag package: `{best_row['variant_name']}`"
                    f", autogluon_mean_multiclass_log_loss = {best_row['autogluon_mean_multiclass_log_loss']:.4f}"
                )

    lines.extend(["", "## Variant Summary", ""])
    for row in summary_rows:
        variant_name = row["variant_name"]
        task_name = row["task_name"]
        lines.append(f"### `{variant_name}` / `{task_name}`")
        lines.append("")
        if task_name == "centered_desirability":
            lines.append(
                f"- autogluon_mean_mae: {row['autogluon_mean_mae']:.4f}"
                f" | autogluon_mean_rmse: {row['autogluon_mean_rmse']:.4f}"
                f" | autogluon_mean_rank_correlation: {row['autogluon_mean_rank_correlation']:.4f}"
                f" | autogluon_mean_in_spec_auc: {row['autogluon_mean_in_spec_auc_from_desirability']:.4f}"
            )
        else:
            lines.append(
                f"- autogluon_mean_macro_f1: {row['autogluon_mean_macro_f1']:.4f}"
                f" | autogluon_mean_balanced_accuracy: {row['autogluon_mean_balanced_accuracy']:.4f}"
                f" | autogluon_mean_multiclass_log_loss: {row['autogluon_mean_multiclass_log_loss']:.4f}"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon lag reality cross-check with reused label semantics.")
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

    labeled = build_label_frame(config_path, config)
    dcs = load_dcs_frame(
        resolve_path(config_path.parent, config["paths"]["dcs_main_path"]),
        resolve_path(config_path.parent, config["paths"].get("dcs_supplemental_path")),
    )
    stage1_reference = _load_stage1_reference(config_path, config)
    all_five_bin_labels = sorted(labeled["target_five_bin"].dropna().astype(int).unique().tolist())
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    enabled_tasks = [
        task_name
        for task_name in ("centered_desirability", "five_bin")
        if bool(config.get("tasks", {}).get(task_name, {}).get("enabled", False))
    ]
    if not enabled_tasks:
        raise ValueError("At least one task must be enabled.")

    variant_tables: dict[str, pd.DataFrame] = {}
    feature_catalog_parts: list[pd.DataFrame] = []
    for variant in config["variants"]:
        frame, catalog = build_variant_snapshot(
            labeled_samples=labeled,
            dcs=dcs,
            variant=variant,
            stats=list(config["snapshot"]["stats"]),
            min_points_per_window=int(config["snapshot"]["min_points_per_window"]),
        )
        if frame.empty:
            continue
        variant_name = str(variant["name"])
        variant_tables[variant_name] = frame
        if not catalog.empty:
            feature_catalog_parts.append(catalog)

    if not variant_tables:
        raise ValueError("No lag variants produced a non-empty snapshot table.")

    variant_tables, common_index = align_common_snapshot_time(variant_tables)
    common_sample_count = int(len(common_index))
    if common_sample_count == 0:
        raise ValueError("No common samples remained after aligning lag variants.")

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    leaderboards: dict[str, pd.DataFrame] = {}
    feature_importances: dict[str, pd.DataFrame] = {}
    discovered_scripts = sorted(path.name for path in SCRIPT_DIR.glob("run_autogluon_*.py"))

    for variant_name, snapshot in variant_tables.items():
        feature_columns = [column for column in snapshot.columns if "__" in column]
        if not feature_columns:
            continue

        for task_name in enabled_tasks:
            fold_summaries: list[dict[str, Any]] = []
            baseline_agg: list[dict[str, Any]] = []
            ag_agg: list[dict[str, Any]] = []

            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
                train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
                test = snapshot.iloc[test_idx].copy().reset_index(drop=True)

                if task_name == "centered_desirability":
                    label = "target_centered_desirability"
                    baseline = make_regression_baseline()
                    baseline.fit(train[feature_columns], train[label].to_numpy(dtype=float))
                    baseline_pred = baseline.predict(test[feature_columns]).astype(float)
                    baseline_metric = regression_metrics(
                        test[label].to_numpy(dtype=float),
                        baseline_pred,
                        test["is_in_spec"].to_numpy(dtype=int),
                    )

                    model_path = artifact_dir / f"ag_lag_reality_{variant_name}_{task_name}_{run_id}_fold{fold_idx}"
                    ag_pred, model_best = fit_autogluon_fold(
                        train_df=train[feature_columns + [label]].copy(),
                        test_df=test[feature_columns + [label]].copy(),
                        label=label,
                        problem_type="regression",
                        eval_metric="root_mean_squared_error",
                        model_path=model_path,
                        ag_config=config["autogluon"],
                    )
                    ag_metric = regression_metrics(
                        test[label].to_numpy(dtype=float),
                        ag_pred.astype(float),
                        test["is_in_spec"].to_numpy(dtype=int),
                    )
                else:
                    label = "target_five_bin"
                    train_y = train[label].to_numpy(dtype=int)
                    test_y = test[label].to_numpy(dtype=int)

                    baseline = make_multiclass_baseline()
                    baseline.fit(train[feature_columns], train_y)
                    baseline_proba = baseline.predict_proba(test[feature_columns]).astype(float)
                    observed_labels = baseline.named_steps["clf"].classes_.tolist()
                    baseline_frame = pd.DataFrame(baseline_proba, columns=observed_labels)
                    baseline_frame = baseline_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    baseline_frame = baseline_frame.div(baseline_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    baseline_metric = multiclass_metrics(
                        test_y,
                        baseline_frame.to_numpy(dtype=float),
                        class_labels=all_five_bin_labels,
                    )

                    model_path = artifact_dir / f"ag_lag_reality_{variant_name}_{task_name}_{run_id}_fold{fold_idx}"
                    ag_proba_frame, model_best, leaderboard, feature_importance = fit_autogluon_multiclass(
                        train_df=train[feature_columns + [label]].copy(),
                        test_df=test[feature_columns + [label]].copy(),
                        label=label,
                        model_path=model_path,
                        ag_config=config["autogluon"],
                    )
                    ag_proba_frame = ag_proba_frame.reindex(columns=all_five_bin_labels, fill_value=0.0)
                    ag_proba_frame = ag_proba_frame.div(ag_proba_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
                    ag_metric = multiclass_metrics(
                        test_y,
                        ag_proba_frame.to_numpy(dtype=float),
                        class_labels=all_five_bin_labels,
                    )
                    if fold_idx == 1:
                        leaderboards[f"{variant_name}::{task_name}"] = leaderboard
                        feature_importances[f"{variant_name}::{task_name}"] = feature_importance

                for framework_name, metrics in (("simple_baseline", baseline_metric), ("autogluon", ag_metric)):
                    rows.append(
                        {
                            "stage": "lag_reality_check",
                            "variant_name": variant_name,
                            "task_name": task_name,
                            "framework": framework_name,
                            "fold": int(fold_idx),
                            "samples_train": int(len(train)),
                            "samples_test": int(len(test)),
                            "feature_count": int(len(feature_columns)),
                            "common_samples_scored": common_sample_count,
                            "data_source_condition": config["data_source_statement"]["source_condition"],
                            **metrics,
                            **({"autogluon_model_best": model_best} if framework_name == "autogluon" else {}),
                        }
                    )
                baseline_agg.append(baseline_metric)
                ag_agg.append(ag_metric)
                fold_summaries.append({"fold": int(fold_idx), "baseline": baseline_metric, "autogluon": ag_metric})

            if task_name == "centered_desirability":
                summary_rows.append(
                    {
                        "variant_name": variant_name,
                        "task_name": task_name,
                        "feature_count": int(len(feature_columns)),
                        "common_samples_scored": common_sample_count,
                        "baseline_mean_mae": float(np.nanmean([m["mae"] for m in baseline_agg])),
                        "autogluon_mean_mae": float(np.nanmean([m["mae"] for m in ag_agg])),
                        "baseline_mean_rmse": float(np.nanmean([m["rmse"] for m in baseline_agg])),
                        "autogluon_mean_rmse": float(np.nanmean([m["rmse"] for m in ag_agg])),
                        "baseline_mean_rank_correlation": float(np.nanmean([m["rank_correlation"] for m in baseline_agg])),
                        "autogluon_mean_rank_correlation": float(np.nanmean([m["rank_correlation"] for m in ag_agg])),
                        "baseline_mean_in_spec_auc_from_desirability": float(np.nanmean([m["in_spec_auc_from_desirability"] for m in baseline_agg])),
                        "autogluon_mean_in_spec_auc_from_desirability": float(np.nanmean([m["in_spec_auc_from_desirability"] for m in ag_agg])),
                        "fold_summaries": fold_summaries,
                    }
                )
            else:
                summary_rows.append(
                    {
                        "variant_name": variant_name,
                        "task_name": task_name,
                        "feature_count": int(len(feature_columns)),
                        "common_samples_scored": common_sample_count,
                        "baseline_mean_macro_f1": float(np.nanmean([m["macro_f1"] for m in baseline_agg])),
                        "autogluon_mean_macro_f1": float(np.nanmean([m["macro_f1"] for m in ag_agg])),
                        "baseline_mean_balanced_accuracy": float(np.nanmean([m["balanced_accuracy"] for m in baseline_agg])),
                        "autogluon_mean_balanced_accuracy": float(np.nanmean([m["balanced_accuracy"] for m in ag_agg])),
                        "baseline_mean_multiclass_log_loss": float(np.nanmean([m["multiclass_log_loss"] for m in baseline_agg])),
                        "autogluon_mean_multiclass_log_loss": float(np.nanmean([m["multiclass_log_loss"] for m in ag_agg])),
                        "fold_summaries": fold_summaries,
                    }
                )

    results_df = pd.DataFrame(rows)
    summary_path = artifact_dir / "lag_reality_check_summary.json"
    results_path = artifact_dir / "lag_reality_check_results.csv"
    feature_catalog_path = artifact_dir / "lag_reality_check_feature_catalog.csv"
    common_samples_path = artifact_dir / "lag_reality_check_common_samples.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    common_index.to_csv(common_samples_path, index=False, encoding="utf-8-sig")
    if feature_catalog_parts:
        pd.concat(feature_catalog_parts, ignore_index=True).to_csv(feature_catalog_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for key, leaderboard in leaderboards.items():
        safe_key = key.replace("::", "_")
        leaderboard.to_csv(artifact_dir / f"lag_reality_check_leaderboard_{safe_key}.csv", index=False, encoding="utf-8-sig")
    for key, importance in feature_importances.items():
        safe_key = key.replace("::", "_")
        importance.to_csv(artifact_dir / f"lag_reality_check_feature_importance_{safe_key}.csv", index=False, encoding="utf-8-sig")

    report_path = report_dir / "lag_reality_check_summary.md"
    write_markdown_report(
        report_path=report_path,
        discovered_scripts=discovered_scripts,
        summary_rows=summary_rows,
        common_sample_count=common_sample_count,
        stage1_reference=stage1_reference,
    )


if __name__ == "__main__":
    main()
