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


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_WINDOWS = (5, 10, 15, 20, 30, 45, 60)


def summarize_bromine_accuracy(results: pd.DataFrame) -> dict[str, object]:
    successful = results.loc[results["error_message"].isna()].copy()
    in_spec = successful.loc[successful["is_in_spec"]]
    out_of_spec = successful.loc[~successful["is_in_spec"]]

    def safe_stat(series: pd.Series, fn: str) -> float | None:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return None
        if fn == "mean":
            return float(clean.mean())
        if fn == "median":
            return float(clean.median())
        if fn == "max":
            return float(clean.max())
        if fn == "p90":
            return float(clean.quantile(0.90))
        raise ValueError(f"Unsupported stat function: {fn}")

    def safe_ratio(frame: pd.DataFrame, column: str) -> float | None:
        if frame.empty:
            return None
        return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).mean())

    return {
        "overall": {
            "mean_abs_error": safe_stat(successful["bromine_abs_error_to_best"], "mean"),
            "median_abs_error": safe_stat(successful["bromine_abs_error_to_best"], "median"),
            "p90_abs_error": safe_stat(successful["bromine_abs_error_to_best"], "p90"),
            "max_abs_error": safe_stat(successful["bromine_abs_error_to_best"], "max"),
            "inside_recommended_range_ratio": safe_ratio(successful, "actual_bromine_inside_range"),
        },
        "in_spec_only": {
            "mean_abs_error": safe_stat(in_spec["bromine_abs_error_to_best"], "mean"),
            "median_abs_error": safe_stat(in_spec["bromine_abs_error_to_best"], "median"),
            "p90_abs_error": safe_stat(in_spec["bromine_abs_error_to_best"], "p90"),
            "max_abs_error": safe_stat(in_spec["bromine_abs_error_to_best"], "max"),
            "inside_recommended_range_ratio": safe_ratio(in_spec, "actual_bromine_inside_range"),
        },
        "out_of_spec_only": {
            "mean_abs_error": safe_stat(out_of_spec["bromine_abs_error_to_best"], "mean"),
            "median_abs_error": safe_stat(out_of_spec["bromine_abs_error_to_best"], "median"),
            "p90_abs_error": safe_stat(out_of_spec["bromine_abs_error_to_best"], "p90"),
            "max_abs_error": safe_stat(out_of_spec["bromine_abs_error_to_best"], "max"),
            "inside_recommended_range_ratio": safe_ratio(out_of_spec, "actual_bromine_inside_range"),
        },
    }


def parse_windows(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    windows = sorted({int(item) for item in items})
    if not windows:
        raise ValueError("At least one window size must be provided.")
    if any(window <= 1 for window in windows):
        raise ValueError("Window sizes must be integers greater than 1.")
    return windows


def load_sources(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], list[str], tuple[float, float], str]:
    config = load_runtime_config(config_path)
    sensors = get_context_sensors(config)
    target_range = get_target_range(config)
    time_column = str(config.get("window", {}).get("time_column", "time"))
    data_sources = config.get("data_sources", {})
    dcs = load_dcs_data(data_sources.get("dcs_paths", []), sensors=sensors, time_column=time_column)
    lims = load_lims_grouped(PROJECT_DIR / str(data_sources["lims_path"]))
    return dcs, lims, config, sensors, target_range, time_column


def evaluate_window_size(
    *,
    dcs: pd.DataFrame,
    lims: pd.DataFrame,
    window_minutes: int,
    sensors: list[str],
    target_range: tuple[float, float],
    time_column: str,
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
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    windows, outcomes = build_windows_and_outcomes(dcs, lims, window_minutes, time_column=time_column)
    casebase = build_casebase_from_windows(
        windows,
        outcomes,
        target_low=target_range[0],
        target_high=target_range[1],
        include_sensors=sensors,
    )
    results = evaluate_recommendations(
        casebase,
        limit=limit,
        neighbor_count=neighbor_count,
        local_neighbor_count=local_neighbor_count,
        probability_threshold=probability_threshold,
        grid_points=grid_points,
    )
    summary = summarize_results(results)
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
    return results, summary, calcium_summary, bromine_summary, judgement


def build_experiment_table(experiment_rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(experiment_rows).sort_values("window_minutes").reset_index(drop=True)
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
    return frame.sort_values("window_minutes").reset_index(drop=True)


def create_window_experiment_plot(summary_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].plot(summary_frame["window_minutes"], summary_frame["aligned_samples"], marker="o", color="#1f77b4")
    axes[0, 0].set_title("Aligned samples by window size")
    axes[0, 0].set_xlabel("Window minutes")
    axes[0, 0].set_ylabel("Aligned samples")

    axes[0, 1].plot(summary_frame["window_minutes"], summary_frame["calcium_mae"], marker="o", label="calcium MAE", color="#d62728")
    axes[0, 1].plot(summary_frame["window_minutes"], summary_frame["bromine_mae"], marker="o", label="bromine MAE", color="#2ca02c")
    axes[0, 1].set_title("Recommendation error by window size")
    axes[0, 1].set_xlabel("Window minutes")
    axes[0, 1].set_ylabel("Absolute error")
    axes[0, 1].legend()

    axes[1, 0].plot(summary_frame["window_minutes"], summary_frame["in_spec_calcium_range_ratio"], marker="o", label="in-spec calcium range ratio", color="#9467bd")
    axes[1, 0].plot(summary_frame["window_minutes"], summary_frame["in_spec_bromine_range_ratio"], marker="o", label="in-spec bromine range ratio", color="#8c564b")
    axes[1, 0].set_title("In-spec range coverage by window size")
    axes[1, 0].set_xlabel("Window minutes")
    axes[1, 0].set_ylabel("Coverage ratio")
    axes[1, 0].legend()

    axes[1, 1].plot(summary_frame["window_minutes"], summary_frame["composite_score"], marker="o", color="#ff7f0e")
    axes[1, 1].scatter(
        summary_frame.loc[summary_frame["acceptance_passed"], "window_minutes"],
        summary_frame.loc[summary_frame["acceptance_passed"], "composite_score"],
        color="#2ca02c",
        s=50,
        label="acceptance passed",
        zorder=3,
    )
    axes[1, 1].set_title("Composite score by window size")
    axes[1, 1].set_xlabel("Window minutes")
    axes[1, 1].set_ylabel("Heuristic score")
    axes[1, 1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_experiment_report(summary_frame: pd.DataFrame, windows: list[int], plot_path: Path) -> dict[str, object]:
    if summary_frame.empty:
        return {
            "tested_windows": windows,
            "message": "No valid experiment rows were produced.",
            "plot_path": str(plot_path),
        }

    best_by_score = summary_frame.sort_values("composite_score", ascending=False).iloc[0]
    best_by_calcium_mae = summary_frame.sort_values("calcium_mae", ascending=True).iloc[0]
    best_by_bromine_mae = summary_frame.sort_values("bromine_mae", ascending=True).iloc[0]
    best_by_calcium_range = summary_frame.sort_values("in_spec_calcium_range_ratio", ascending=False).iloc[0]

    return {
        "tested_windows": windows,
        "plot_path": str(plot_path),
        "scoring_note": "composite_score is a heuristic that balances calcium MAE, bromine MAE, in-spec range coverage, recommendation success, and aligned sample count.",
        "best_window_by_composite_score": int(best_by_score["window_minutes"]),
        "best_window_by_calcium_mae": int(best_by_calcium_mae["window_minutes"]),
        "best_window_by_bromine_mae": int(best_by_bromine_mae["window_minutes"]),
        "best_window_by_in_spec_calcium_range_ratio": int(best_by_calcium_range["window_minutes"]),
        "rows": summary_frame.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep multiple DCS window sizes and compare recommendation quality in dev only.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--windows", default=",".join(str(item) for item in DEFAULT_WINDOWS), help="Comma-separated window sizes in minutes.")
    parser.add_argument("--output-prefix", default="", help="Optional prefix for output artifact filenames.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on evaluated samples per window. Default 0 means all aligned samples.")
    parser.add_argument("--neighbor-count", type=int, default=150, help="Neighborhood size for recommendation replay.")
    parser.add_argument("--local-neighbor-count", type=int, default=80, help="Local neighborhood size for control recommendation.")
    parser.add_argument("--probability-threshold", type=float, default=0.60, help="In-spec probability threshold for feasible recommendation points.")
    parser.add_argument("--grid-points", type=int, default=31, help="Grid density for calcium/bromine recommendation search.")
    parser.add_argument("--mae-threshold", type=float, default=0.05, help="Acceptance threshold for calcium MAE.")
    parser.add_argument("--p90-threshold", type=float, default=0.10, help="Acceptance threshold for calcium P90 error.")
    parser.add_argument("--max-threshold", type=float, default=0.25, help="Acceptance threshold for calcium worst-case error.")
    parser.add_argument("--in-spec-range-ratio-threshold", type=float, default=0.55, help="Acceptance threshold for in-spec calcium range coverage.")
    parser.add_argument("--success-ratio-threshold", type=float, default=1.0, help="Acceptance threshold for recommendation success ratio.")
    args = parser.parse_args()

    config_path = Path(args.config)
    windows = parse_windows(args.windows)
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{args.output_prefix}_" if args.output_prefix else ""

    dcs, lims, _config, sensors, target_range, time_column = load_sources(config_path)

    experiment_rows: list[dict[str, object]] = []
    for window_minutes in windows:
        print(f"running window experiment: {window_minutes} minutes")
        _results, replay_summary, calcium_summary, bromine_summary, judgement = evaluate_window_size(
            dcs=dcs,
            lims=lims,
            window_minutes=window_minutes,
            sensors=sensors,
            target_range=target_range,
            time_column=time_column,
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
        success_ratio = (
            float(replay_summary["successful_recommendations"] / aligned_samples)
            if aligned_samples > 0
            else 0.0
        )
        experiment_rows.append(
            {
                "window_minutes": window_minutes,
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

    summary_frame = build_experiment_table(experiment_rows)
    plot_path = results_dir / f"{output_prefix}window_size_experiment_metrics.png"
    create_window_experiment_plot(summary_frame, plot_path)

    report = build_experiment_report(summary_frame, windows, plot_path)
    summary_csv_path = results_dir / f"{output_prefix}window_size_experiment_summary.csv"
    summary_json_path = results_dir / f"{output_prefix}window_size_experiment_summary.json"
    summary_frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    summary_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"summary_csv_saved_to={summary_csv_path}")
    print(f"summary_json_saved_to={summary_json_path}")
    print(f"plot_saved_to={plot_path}")


if __name__ == "__main__":
    main()
