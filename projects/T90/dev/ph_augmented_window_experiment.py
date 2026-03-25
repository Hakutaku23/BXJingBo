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
from test import (
    build_windows_and_outcomes,
    evaluate_recommendations,
    load_dcs_data,
    load_lims_grouped,
    summarize_results,
)
from evaluate_calcium_accuracy import summarize_calcium_accuracy
from judge_calcium_acceptance import judge_acceptance
from ph_lag_experiment import build_lagged_dataset, build_ph_features, load_ph_data
from window_size_experiment import summarize_bromine_accuracy


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_PH_PATH = PROJECT_DIR / "data" / "B4-AI-C53001A.PV.F_CV.xlsx"
DEFAULT_DCS_WINDOW = 50


def parse_lags(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    lags = sorted({int(item) for item in items})
    if not lags:
        raise ValueError("At least one lag must be provided.")
    if any(lag < 0 for lag in lags):
        raise ValueError("Lag values must be non-negative integers.")
    return lags


def load_sources(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], list[str], tuple[float, float], str]:
    config = load_runtime_config(config_path)
    sensors = get_context_sensors(config)
    target_range = get_target_range(config)
    time_column = str(config.get("window", {}).get("time_column", "time"))
    data_sources = config.get("data_sources", {})
    dcs = load_dcs_data(data_sources.get("dcs_paths", []), sensors=sensors, time_column=time_column)
    lims = load_lims_grouped(PROJECT_DIR / str(data_sources["lims_path"]))
    return dcs, lims, config, sensors, target_range, time_column


def attach_ph_features(
    casebase: pd.DataFrame,
    lims: pd.DataFrame,
    ph_features: pd.DataFrame,
    *,
    lag_minutes: int,
    tolerance_minutes: int,
) -> pd.DataFrame:
    aligned = build_lagged_dataset(
        lims=lims,
        ph_features=ph_features,
        lag_minutes=lag_minutes,
        tolerance_minutes=tolerance_minutes,
    )
    ph_columns = ["sample_time", "ph_point", "ph_mean", "ph_std", "ph_min", "ph_max", "ph_delta", "alignment_error_minutes"]
    ph_augmented = aligned[ph_columns].copy()
    merged = casebase.merge(ph_augmented, on="sample_time", how="inner")
    merged = merged.dropna(subset=["ph_point", "ph_mean", "ph_std", "ph_min", "ph_max", "ph_delta"]).reset_index(drop=True)
    return merged


def evaluate_casebase(
    casebase: pd.DataFrame,
    *,
    limit: int,
    neighbor_count: int,
    local_neighbor_count: int,
    probability_threshold: float,
    grid_points: int,
    mae_threshold: float,
    p90_threshold: float,
    max_threshold: float,
    in_spec_range_ratio_threshold: float,
    success_ratio_threshold: float,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    results = evaluate_recommendations(
        casebase,
        limit=limit,
        neighbor_count=neighbor_count,
        local_neighbor_count=local_neighbor_count,
        probability_threshold=probability_threshold,
        grid_points=grid_points,
    )
    replay_summary = summarize_results(results)
    calcium_summary = summarize_calcium_accuracy(results)
    bromine_summary = summarize_bromine_accuracy(results)
    judgement = judge_acceptance(
        calcium_summary,
        mae_threshold=mae_threshold,
        p90_threshold=p90_threshold,
        max_threshold=max_threshold,
        in_spec_range_ratio_threshold=in_spec_range_ratio_threshold,
        success_ratio_threshold=success_ratio_threshold,
    )
    return replay_summary, calcium_summary, bromine_summary, judgement


def build_summary_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values("lag_minutes").reset_index(drop=True)
    if frame.empty:
        return frame

    max_aligned = float(frame["aligned_samples"].max()) if not frame["aligned_samples"].isna().all() else 1.0
    frame["sample_ratio"] = frame["aligned_samples"] / max_aligned
    frame["calcium_mae_score"] = 1.0 / (1.0 + frame["calcium_mae"] / 0.05)
    frame["bromine_mae_score"] = 1.0 / (1.0 + frame["bromine_mae"] / 0.025)
    frame["composite_score"] = (
        0.35 * frame["calcium_mae_score"]
        + 0.15 * frame["bromine_mae_score"]
        + 0.20 * frame["in_spec_calcium_range_ratio"]
        + 0.10 * frame["in_spec_bromine_range_ratio"]
        + 0.10 * frame["success_ratio"]
        + 0.10 * frame["sample_ratio"]
    )
    return frame


def create_plot(summary: pd.DataFrame, baseline: dict[str, object], output_path: Path, focus_low: int, focus_high: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].plot(summary["lag_minutes"], summary["aligned_samples"], marker="o", color="#1f77b4")
    axes[0, 0].axhline(float(baseline["aligned_samples"]), linestyle="--", color="#444444", linewidth=1.0, label="50min baseline")
    axes[0, 0].axvspan(focus_low, focus_high, color="#ffdd99", alpha=0.25)
    axes[0, 0].set_title("Aligned samples with PH lag")
    axes[0, 0].set_xlabel("PH lag minutes")
    axes[0, 0].set_ylabel("Aligned samples")
    axes[0, 0].legend()

    axes[0, 1].plot(summary["lag_minutes"], summary["calcium_mae"], marker="o", label="calcium MAE", color="#d62728")
    axes[0, 1].plot(summary["lag_minutes"], summary["bromine_mae"], marker="o", label="bromine MAE", color="#2ca02c")
    axes[0, 1].axhline(float(baseline["calcium_mae"]), linestyle="--", color="#d62728", linewidth=1.0, alpha=0.7)
    axes[0, 1].axhline(float(baseline["bromine_mae"]), linestyle="--", color="#2ca02c", linewidth=1.0, alpha=0.7)
    axes[0, 1].axvspan(focus_low, focus_high, color="#ffdd99", alpha=0.25)
    axes[0, 1].set_title("Recommendation error with PH lag")
    axes[0, 1].set_xlabel("PH lag minutes")
    axes[0, 1].set_ylabel("Absolute error")
    axes[0, 1].legend()

    axes[1, 0].plot(summary["lag_minutes"], summary["in_spec_calcium_range_ratio"], marker="o", label="in-spec calcium range ratio", color="#9467bd")
    axes[1, 0].plot(summary["lag_minutes"], summary["in_spec_bromine_range_ratio"], marker="o", label="in-spec bromine range ratio", color="#8c564b")
    axes[1, 0].axhline(float(baseline["in_spec_calcium_range_ratio"]), linestyle="--", color="#9467bd", linewidth=1.0, alpha=0.7)
    axes[1, 0].axhline(float(baseline["in_spec_bromine_range_ratio"]), linestyle="--", color="#8c564b", linewidth=1.0, alpha=0.7)
    axes[1, 0].axvspan(focus_low, focus_high, color="#ffdd99", alpha=0.25)
    axes[1, 0].set_title("In-spec coverage with PH lag")
    axes[1, 0].set_xlabel("PH lag minutes")
    axes[1, 0].set_ylabel("Coverage ratio")
    axes[1, 0].legend()

    axes[1, 1].plot(summary["lag_minutes"], summary["composite_score"], marker="o", color="#ff7f0e")
    axes[1, 1].axhline(float(baseline["composite_score"]), linestyle="--", color="#444444", linewidth=1.0, label="50min baseline")
    axes[1, 1].scatter(
        summary.loc[summary["acceptance_passed"], "lag_minutes"],
        summary.loc[summary["acceptance_passed"], "composite_score"],
        color="#2ca02c",
        s=50,
        label="acceptance passed",
        zorder=3,
    )
    axes[1, 1].axvspan(focus_low, focus_high, color="#ffdd99", alpha=0.25)
    axes[1, 1].set_title("Composite score with PH lag")
    axes[1, 1].set_xlabel("PH lag minutes")
    axes[1, 1].set_ylabel("Heuristic score")
    axes[1, 1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_report(summary: pd.DataFrame, baseline: dict[str, object], lags: list[int], plot_path: Path, focus_low: int, focus_high: int) -> dict[str, object]:
    if summary.empty:
        return {
            "tested_lags": lags,
            "message": "No valid PH-augmented experiment rows were produced.",
            "plot_path": str(plot_path),
            "baseline": baseline,
        }

    best_by_score = summary.sort_values("composite_score", ascending=False).iloc[0]
    best_by_calcium_mae = summary.sort_values("calcium_mae", ascending=True).iloc[0]
    best_by_calcium_range = summary.sort_values("in_spec_calcium_range_ratio", ascending=False).iloc[0]
    focus = summary.loc[summary["lag_minutes"].between(focus_low, focus_high)].copy()
    best_focus = None if focus.empty else focus.sort_values("composite_score", ascending=False).iloc[0]

    return {
        "tested_lags": lags,
        "plot_path": str(plot_path),
        "focus_window_minutes": [focus_low, focus_high],
        "baseline": baseline,
        "best_lag_by_composite_score": int(best_by_score["lag_minutes"]),
        "best_lag_by_calcium_mae": int(best_by_calcium_mae["lag_minutes"]),
        "best_lag_by_in_spec_calcium_range_ratio": int(best_by_calcium_range["lag_minutes"]),
        "best_focus_lag_by_composite_score": None if best_focus is None else int(best_focus["lag_minutes"]),
        "supports_process_claim": bool(best_focus is not None and float(best_focus["composite_score"]) >= float(baseline["composite_score"])),
        "rows": summary.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether lagged PH improves the 50min DCS recommendation baseline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--ph-path", default=str(DEFAULT_PH_PATH), help="PH Excel path.")
    parser.add_argument("--window-minutes", type=int, default=DEFAULT_DCS_WINDOW, help="Fixed DCS window size for the joint experiment.")
    parser.add_argument("--ph-feature-window", type=int, default=50, help="PH rolling feature window size in minutes.")
    parser.add_argument("--lags", default="0,30,60,90,120,150,180,210,240,270,300", help="Comma-separated PH lags in minutes.")
    parser.add_argument("--tolerance", type=int, default=2, help="Merge tolerance in minutes.")
    parser.add_argument("--output-prefix", default="", help="Optional output filename prefix.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on evaluated samples. Default 0 means all aligned samples.")
    parser.add_argument("--neighbor-count", type=int, default=150, help="Neighborhood size for recommendation replay.")
    parser.add_argument("--local-neighbor-count", type=int, default=80, help="Local neighborhood size for control recommendation.")
    parser.add_argument("--probability-threshold", type=float, default=0.60, help="In-spec probability threshold for feasible recommendation points.")
    parser.add_argument("--grid-points", type=int, default=31, help="Grid density for calcium/bromine recommendation search.")
    parser.add_argument("--mae-threshold", type=float, default=0.05, help="Acceptance threshold for calcium MAE.")
    parser.add_argument("--p90-threshold", type=float, default=0.10, help="Acceptance threshold for calcium P90 error.")
    parser.add_argument("--max-threshold", type=float, default=0.25, help="Acceptance threshold for calcium worst-case error.")
    parser.add_argument("--in-spec-range-ratio-threshold", type=float, default=0.55, help="Acceptance threshold for in-spec calcium range coverage.")
    parser.add_argument("--success-ratio-threshold", type=float, default=1.0, help="Acceptance threshold for recommendation success ratio.")
    parser.add_argument("--focus-low", type=int, default=180, help="Lower lag bound of the process-claimed focus window.")
    parser.add_argument("--focus-high", type=int, default=270, help="Upper lag bound of the process-claimed focus window.")
    args = parser.parse_args()

    lags = parse_lags(args.lags)
    output_prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = results_dir / f"{output_prefix}ph_augmented_window_experiment_summary.csv"
    summary_json_path = results_dir / f"{output_prefix}ph_augmented_window_experiment_summary.json"
    plot_path = results_dir / f"{output_prefix}ph_augmented_window_experiment_metrics.png"

    dcs, lims, _config, sensors, target_range, time_column = load_sources(Path(args.config))
    windows, outcomes = build_windows_and_outcomes(dcs, lims, args.window_minutes, time_column=time_column)
    base_casebase = build_casebase_from_windows(
        windows,
        outcomes,
        target_low=target_range[0],
        target_high=target_range[1],
        include_sensors=sensors,
    )

    replay_summary, calcium_summary, bromine_summary, judgement = evaluate_casebase(
        base_casebase,
        limit=args.limit,
        neighbor_count=args.neighbor_count,
        local_neighbor_count=args.local_neighbor_count,
        probability_threshold=args.probability_threshold,
        grid_points=args.grid_points,
        mae_threshold=args.mae_threshold,
        p90_threshold=args.p90_threshold,
        max_threshold=args.max_threshold,
        in_spec_range_ratio_threshold=args.in_spec_range_ratio_threshold,
        success_ratio_threshold=args.success_ratio_threshold,
    )
    aligned_samples = int(replay_summary["aligned_samples"])
    baseline = {
        "window_minutes": int(args.window_minutes),
        "aligned_samples": aligned_samples,
        "success_ratio": float(replay_summary["successful_recommendations"] / aligned_samples) if aligned_samples else 0.0,
        "calcium_mae": calcium_summary["overall"]["mean_abs_error"],
        "calcium_p90": calcium_summary["overall"]["p90_abs_error"],
        "calcium_max_error": calcium_summary["overall"]["max_abs_error"],
        "bromine_mae": bromine_summary["overall"]["mean_abs_error"],
        "bromine_p90": bromine_summary["overall"]["p90_abs_error"],
        "in_spec_calcium_range_ratio": replay_summary["in_spec_actual_calcium_inside_recommended_range_ratio"],
        "in_spec_bromine_range_ratio": replay_summary["in_spec_actual_bromine_inside_recommended_range_ratio"],
        "overall_calcium_range_ratio": replay_summary["actual_calcium_inside_recommended_range_ratio"],
        "overall_bromine_range_ratio": replay_summary["actual_bromine_inside_recommended_range_ratio"],
        "acceptance_passed": bool(judgement["overall_passed"]),
    }
    baseline["sample_ratio"] = 1.0
    baseline["calcium_mae_score"] = 1.0 / (1.0 + float(baseline["calcium_mae"]) / 0.05)
    baseline["bromine_mae_score"] = 1.0 / (1.0 + float(baseline["bromine_mae"]) / 0.025)
    baseline["composite_score"] = (
        0.35 * baseline["calcium_mae_score"]
        + 0.15 * baseline["bromine_mae_score"]
        + 0.20 * float(baseline["in_spec_calcium_range_ratio"])
        + 0.10 * float(baseline["in_spec_bromine_range_ratio"])
        + 0.10 * float(baseline["success_ratio"])
        + 0.10 * baseline["sample_ratio"]
    )

    ph = load_ph_data(args.ph_path)
    ph_features = build_ph_features(ph, feature_window_minutes=args.ph_feature_window)

    experiment_rows: list[dict[str, object]] = []
    for lag_minutes in lags:
        print(f"running PH-augmented experiment: lag={lag_minutes} minutes")
        casebase = attach_ph_features(
            base_casebase,
            lims,
            ph_features,
            lag_minutes=lag_minutes,
            tolerance_minutes=args.tolerance,
        )
        replay_summary, calcium_summary, bromine_summary, judgement = evaluate_casebase(
            casebase,
            limit=args.limit,
            neighbor_count=args.neighbor_count,
            local_neighbor_count=args.local_neighbor_count,
            probability_threshold=args.probability_threshold,
            grid_points=args.grid_points,
            mae_threshold=args.mae_threshold,
            p90_threshold=args.p90_threshold,
            max_threshold=args.max_threshold,
            in_spec_range_ratio_threshold=args.in_spec_range_ratio_threshold,
            success_ratio_threshold=args.success_ratio_threshold,
        )
        aligned_samples = int(replay_summary["aligned_samples"])
        success_ratio = float(replay_summary["successful_recommendations"] / aligned_samples) if aligned_samples else 0.0
        experiment_rows.append(
            {
                "lag_minutes": lag_minutes,
                "aligned_samples": aligned_samples,
                "successful_recommendations": int(replay_summary["successful_recommendations"]),
                "failed_recommendations": int(replay_summary["failed_recommendations"]),
                "success_ratio": success_ratio,
                "in_spec_samples": int(replay_summary["in_spec_samples"]),
                "calcium_mae": calcium_summary["overall"]["mean_abs_error"],
                "calcium_p90": calcium_summary["overall"]["p90_abs_error"],
                "calcium_max_error": calcium_summary["overall"]["max_abs_error"],
                "bromine_mae": bromine_summary["overall"]["mean_abs_error"],
                "bromine_p90": bromine_summary["overall"]["p90_abs_error"],
                "in_spec_calcium_range_ratio": replay_summary["in_spec_actual_calcium_inside_recommended_range_ratio"],
                "in_spec_bromine_range_ratio": replay_summary["in_spec_actual_bromine_inside_recommended_range_ratio"],
                "overall_calcium_range_ratio": replay_summary["actual_calcium_inside_recommended_range_ratio"],
                "overall_bromine_range_ratio": replay_summary["actual_bromine_inside_recommended_range_ratio"],
                "acceptance_passed": bool(judgement["overall_passed"]),
            }
        )

    summary_frame = build_summary_frame(experiment_rows)
    summary_frame["delta_calcium_mae_vs_baseline"] = summary_frame["calcium_mae"] - float(baseline["calcium_mae"])
    summary_frame["delta_bromine_mae_vs_baseline"] = summary_frame["bromine_mae"] - float(baseline["bromine_mae"])
    summary_frame["delta_in_spec_calcium_range_ratio_vs_baseline"] = (
        summary_frame["in_spec_calcium_range_ratio"] - float(baseline["in_spec_calcium_range_ratio"])
    )
    summary_frame["delta_composite_score_vs_baseline"] = summary_frame["composite_score"] - float(baseline["composite_score"])

    create_plot(summary_frame, baseline, plot_path, focus_low=args.focus_low, focus_high=args.focus_high)
    report = build_report(summary_frame, baseline, lags, plot_path, focus_low=args.focus_low, focus_high=args.focus_high)

    summary_frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    summary_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"summary_csv_saved_to={summary_csv_path}")
    print(f"summary_json_saved_to={summary_json_path}")
    print(f"plot_saved_to={plot_path}")


if __name__ == "__main__":
    main()
