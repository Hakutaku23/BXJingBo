from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core import build_casebase_from_windows, get_context_sensors, get_target_range, load_runtime_config
from evaluate_calcium_accuracy import summarize_calcium_accuracy
from judge_calcium_acceptance import judge_acceptance
from ph_augmented_window_experiment import attach_ph_features
from ph_lag_experiment import build_ph_features, load_ph_data
from ph_segmented_window_experiment import prune_empty_numeric_columns
from stage_identifier_experiment import (
    assign_stage_labels,
    build_policy_recommendation,
    build_stage_policy_rows,
    choose_stage_count,
    evaluate_stage_counts,
)
from test import (
    build_windows_and_outcomes,
    evaluate_recommendations,
    load_dcs_data,
    load_lims_grouped,
    summarize_results,
)
from window_size_experiment import summarize_bromine_accuracy


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_PH_PATH = PROJECT_DIR / "data" / "B4-AI-C53001A.PV.F_CV.xlsx"
DEFAULT_DCS_WINDOW = 50
DEFAULT_STAGE_COUNTS = "2,3,4,5,6"
DEFAULT_PH_LAGS = "120,240,300"


def parse_int_list(value: str, *, minimum: int) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    values = sorted({int(item) for item in items})
    if not values:
        raise ValueError("At least one integer value must be provided.")
    if any(item < minimum for item in values):
        raise ValueError(f"All values must be >= {minimum}.")
    return values


def load_sources(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], tuple[float, float], str]:
    config = load_runtime_config(config_path)
    sensors = get_context_sensors(config)
    target_range = get_target_range(config)
    time_column = str(config.get("window", {}).get("time_column", "time"))
    data_sources = config.get("data_sources", {})
    dcs = load_dcs_data(data_sources.get("dcs_paths", []), sensors=sensors, time_column=time_column)
    lims = load_lims_grouped(PROJECT_DIR / str(data_sources["lims_path"]))
    return dcs, lims, sensors, target_range, time_column


def summarize_combined_results(results: pd.DataFrame) -> dict[str, object]:
    replay_summary = summarize_results(results)
    calcium_summary = summarize_calcium_accuracy(results)
    bromine_summary = summarize_bromine_accuracy(results)
    judgement = judge_acceptance(
        calcium_summary,
        mae_threshold=0.05,
        p90_threshold=0.10,
        max_threshold=0.25,
        in_spec_range_ratio_threshold=0.55,
        success_ratio_threshold=1.0,
    )
    return {
        "replay_summary": replay_summary,
        "calcium_summary": calcium_summary,
        "bromine_summary": bromine_summary,
        "judgement": judgement,
    }


def build_metric_row(name: str, results: pd.DataFrame) -> dict[str, object]:
    summary = summarize_combined_results(results)
    replay = summary["replay_summary"]
    calcium = summary["calcium_summary"]
    bromine = summary["bromine_summary"]
    aligned_samples = int(replay["aligned_samples"])
    success_ratio = float(replay["successful_recommendations"] / aligned_samples) if aligned_samples else 0.0
    row = {
        "strategy": name,
        "aligned_samples": aligned_samples,
        "successful_recommendations": int(replay["successful_recommendations"]),
        "failed_recommendations": int(replay["failed_recommendations"]),
        "success_ratio": success_ratio,
        "in_spec_samples": int(replay["in_spec_samples"]),
        "calcium_mae": calcium["overall"]["mean_abs_error"],
        "calcium_p90": calcium["overall"]["p90_abs_error"],
        "calcium_max_error": calcium["overall"]["max_abs_error"],
        "bromine_mae": bromine["overall"]["mean_abs_error"],
        "bromine_p90": bromine["overall"]["p90_abs_error"],
        "in_spec_calcium_range_ratio": replay["in_spec_actual_calcium_inside_recommended_range_ratio"],
        "in_spec_bromine_range_ratio": replay["in_spec_actual_bromine_inside_recommended_range_ratio"],
        "overall_calcium_range_ratio": replay["actual_calcium_inside_recommended_range_ratio"],
        "overall_bromine_range_ratio": replay["actual_bromine_inside_recommended_range_ratio"],
        "acceptance_passed": bool(summary["judgement"]["overall_passed"]),
    }
    row["composite_score"] = (
        0.35 * (1.0 / (1.0 + float(row["calcium_mae"]) / 0.05))
        + 0.15 * (1.0 / (1.0 + float(row["bromine_mae"]) / 0.025))
        + 0.20 * float(row["in_spec_calcium_range_ratio"])
        + 0.10 * float(row["in_spec_bromine_range_ratio"])
        + 0.10 * float(row["success_ratio"])
        + 0.10
    )
    return row


def create_strategy_plot(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("calcium_mae", "Calcium MAE"),
        ("bromine_mae", "Bromine MAE"),
        ("in_spec_calcium_range_ratio", "In-spec calcium coverage"),
        ("composite_score", "Composite score"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (column, title) in zip(axes.flatten(), metrics):
        axis.bar(summary["strategy"], summary[column], color=["#4c78a8", "#f58518"])
        axis.set_title(title)
        axis.set_ylabel(column)
        axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev-only stage-aware prototype for second-phase T90 recommendation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--ph-path", default=str(DEFAULT_PH_PATH), help="PH Excel path.")
    parser.add_argument("--window-minutes", type=int, default=DEFAULT_DCS_WINDOW, help="Fixed DCS window size.")
    parser.add_argument("--ph-feature-window", type=int, default=50, help="PH rolling feature window size in minutes.")
    parser.add_argument("--ph-lags", default=DEFAULT_PH_LAGS, help="Comma-separated PH lags to evaluate for stage policy.")
    parser.add_argument("--stage-counts", default=DEFAULT_STAGE_COUNTS, help="Comma-separated stage counts for stage identifier selection.")
    parser.add_argument("--min-stage-samples", type=int, default=120, help="Minimum samples per stage for a valid stage-count candidate.")
    parser.add_argument("--enable-threshold", type=float, default=0.002, help="Minimum composite-score gain needed to enable PH in a stage.")
    parser.add_argument("--max-recommended-lag", type=int, default=240, help="Largest PH lag allowed in the stage-aware prototype.")
    parser.add_argument("--tolerance", type=int, default=2, help="Merge tolerance in minutes.")
    parser.add_argument("--output-prefix", default="", help="Optional output filename prefix.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on evaluated samples. Default 0 means all aligned samples.")
    parser.add_argument("--neighbor-count", type=int, default=150, help="Neighborhood size for recommendation replay.")
    parser.add_argument("--local-neighbor-count", type=int, default=80, help="Local neighborhood size for control recommendation.")
    parser.add_argument("--probability-threshold", type=float, default=0.60, help="In-spec probability threshold.")
    parser.add_argument("--grid-points", type=int, default=31, help="Grid density for calcium/bromine search.")
    args = parser.parse_args()

    ph_lags = parse_int_list(args.ph_lags, minimum=0)
    stage_counts = parse_int_list(args.stage_counts, minimum=2)
    output_prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = results_dir / f"{output_prefix}stage_aware_prototype_summary.csv"
    summary_json_path = results_dir / f"{output_prefix}stage_aware_prototype_summary.json"
    plot_path = results_dir / f"{output_prefix}stage_aware_prototype_metrics.png"

    dcs, lims, sensors, target_range, time_column = load_sources(Path(args.config))
    windows, outcomes = build_windows_and_outcomes(dcs, lims, args.window_minutes, time_column=time_column)
    base_casebase = build_casebase_from_windows(
        windows,
        outcomes,
        target_low=target_range[0],
        target_high=target_range[1],
        include_sensors=sensors,
    )

    stage_count_summary, context_columns, imputer, scaler = evaluate_stage_counts(
        base_casebase,
        stage_counts=stage_counts,
        min_stage_samples=args.min_stage_samples,
    )
    chosen_stage_count = choose_stage_count(stage_count_summary)
    staged_casebase, _ = assign_stage_labels(
        base_casebase,
        stage_count=chosen_stage_count,
        context_columns=context_columns,
        imputer=imputer,
        scaler=scaler,
    )

    ph = load_ph_data(args.ph_path)
    ph_features = build_ph_features(ph, feature_window_minutes=args.ph_feature_window)
    stage_lookup = staged_casebase[["sample_time", "stage_id", "stage_name"]].drop_duplicates()
    staged_augmented_by_lag: dict[int, pd.DataFrame] = {}
    for lag_minutes in ph_lags:
        augmented = attach_ph_features(
            base_casebase,
            lims,
            ph_features,
            lag_minutes=lag_minutes,
            tolerance_minutes=args.tolerance,
        )
        staged_augmented_by_lag[lag_minutes] = augmented.merge(stage_lookup, on="sample_time", how="inner")

    policy_summary = build_stage_policy_rows(
        staged_casebase,
        staged_augmented_by_lag,
        lags=ph_lags,
        limit=args.limit,
        neighbor_count=args.neighbor_count,
        local_neighbor_count=args.local_neighbor_count,
        probability_threshold=args.probability_threshold,
        grid_points=args.grid_points,
        mae_threshold=0.05,
        p90_threshold=0.10,
        max_threshold=0.25,
        in_spec_range_ratio_threshold=0.55,
        success_ratio_threshold=1.0,
    )
    best_by_stage = (
        policy_summary.loc[policy_summary["lag_minutes"] >= 0]
        .sort_values(["stage_name", "delta_composite_score_vs_baseline"], ascending=[True, False])
        .groupby("stage_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    policy = build_policy_recommendation(
        best_by_stage,
        enable_threshold=args.enable_threshold,
        max_recommended_lag=args.max_recommended_lag,
    )

    baseline_parts: list[pd.DataFrame] = []
    prototype_parts: list[pd.DataFrame] = []
    policy_map = {item["stage_name"]: item for item in policy["recommendations"]}
    for stage_name, stage_frame in staged_casebase.groupby("stage_name"):
        baseline_stage = prune_empty_numeric_columns(stage_frame.drop(columns=["stage_id", "stage_name"]).reset_index(drop=True))
        baseline_stage_results = evaluate_recommendations(
            baseline_stage,
            limit=args.limit,
            neighbor_count=args.neighbor_count,
            local_neighbor_count=args.local_neighbor_count,
            probability_threshold=args.probability_threshold,
            grid_points=args.grid_points,
        )
        baseline_stage_results["selected_strategy"] = "baseline"
        baseline_stage_results["stage_name"] = stage_name
        baseline_parts.append(baseline_stage_results)

        recommendation = policy_map[stage_name]
        if recommendation["enable_ph"]:
            selected_lag = int(recommendation["best_lag_minutes"])
            selected_casebase = staged_augmented_by_lag[selected_lag]
            selected_stage = selected_casebase.loc[selected_casebase["stage_name"] == stage_name].drop(columns=["stage_id", "stage_name"]).reset_index(drop=True)
            selected_stage = prune_empty_numeric_columns(selected_stage)
            stage_results = evaluate_recommendations(
                selected_stage,
                limit=args.limit,
                neighbor_count=args.neighbor_count,
                local_neighbor_count=args.local_neighbor_count,
                probability_threshold=args.probability_threshold,
                grid_points=args.grid_points,
            )
            stage_results["selected_strategy"] = f"ph_{selected_lag}"
        else:
            stage_results = baseline_stage_results.copy()
            stage_results["selected_strategy"] = "baseline"
        stage_results["stage_name"] = stage_name
        prototype_parts.append(stage_results)

    baseline_results = pd.concat(baseline_parts, ignore_index=True)
    prototype_results = pd.concat(prototype_parts, ignore_index=True)

    baseline_row = build_metric_row("baseline_50min_dcs", baseline_results)
    prototype_row = build_metric_row("stage_aware_prototype", prototype_results)
    summary_frame = pd.DataFrame([baseline_row, prototype_row])
    summary_frame["delta_vs_baseline_calcium_mae"] = summary_frame["calcium_mae"] - baseline_row["calcium_mae"]
    summary_frame["delta_vs_baseline_composite_score"] = summary_frame["composite_score"] - baseline_row["composite_score"]
    summary_frame["delta_vs_baseline_in_spec_calcium_range_ratio"] = (
        summary_frame["in_spec_calcium_range_ratio"] - baseline_row["in_spec_calcium_range_ratio"]
    )
    create_strategy_plot(summary_frame, plot_path)

    report = {
        "tested_stage_counts": stage_count_summary.to_dict(orient="records"),
        "chosen_stage_count": chosen_stage_count,
        "policy_recommendation": policy,
        "strategy_rows": summary_frame.to_dict(orient="records"),
        "stage_policy_rows": best_by_stage.to_dict(orient="records"),
        "plot_path": str(plot_path),
    }

    summary_frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    summary_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"summary_csv_saved_to={summary_csv_path}")
    print(f"summary_json_saved_to={summary_json_path}")
    print(f"plot_saved_to={plot_path}")


if __name__ == "__main__":
    main()
