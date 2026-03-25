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
from ph_augmented_window_experiment import attach_ph_features, evaluate_casebase, parse_lags
from ph_lag_experiment import build_ph_features, load_ph_data
from test import build_windows_and_outcomes, load_dcs_data, load_lims_grouped


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_PH_PATH = PROJECT_DIR / "data" / "B4-AI-C53001A.PV.F_CV.xlsx"
DEFAULT_DCS_WINDOW = 50


def load_sources(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], tuple[float, float], str]:
    config = load_runtime_config(config_path)
    sensors = get_context_sensors(config)
    target_range = get_target_range(config)
    time_column = str(config.get("window", {}).get("time_column", "time"))
    data_sources = config.get("data_sources", {})
    dcs = load_dcs_data(data_sources.get("dcs_paths", []), sensors=sensors, time_column=time_column)
    lims = load_lims_grouped(PROJECT_DIR / str(data_sources["lims_path"]))
    return dcs, lims, sensors, target_range, time_column


def assign_segments(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    segmented = frame.copy()
    segmented["segment"] = pd.to_datetime(segmented["sample_time"]).dt.to_period(freq).astype(str)
    return segmented


def prune_empty_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preserved = {"sample_time", "t90", "is_in_spec", "calcium", "bromine"}
    numeric = frame.select_dtypes(include=["number"]).columns.tolist()
    removable = [column for column in numeric if column not in preserved and frame[column].notna().sum() == 0]
    if not removable:
        return frame
    return frame.drop(columns=removable)


def create_heatmap(summary: pd.DataFrame, value_column: str, output_path: Path, title: str) -> None:
    pivot = summary.pivot(index="segment", columns="lag_minutes", values=value_column).sort_index()
    fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(pivot.index))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(column) for column in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("PH lag minutes")
    ax.set_ylabel("Segment")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.set_ylabel(value_column, rotation=90)

    for row_index, segment in enumerate(pivot.index):
        for column_index, lag in enumerate(pivot.columns):
            value = pivot.loc[segment, lag]
            if pd.isna(value):
                continue
            ax.text(column_index, row_index, f"{value:.4f}", ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_report(summary: pd.DataFrame, best_by_segment: pd.DataFrame, *, freq: str, lags: list[int], focus_low: int, focus_high: int, heatmap_paths: dict[str, str]) -> dict[str, object]:
    focus_best = best_by_segment.loc[best_by_segment["lag_minutes"].between(focus_low, focus_high)]
    consensus = (
        best_by_segment["lag_minutes"].value_counts().sort_index().rename_axis("lag_minutes").reset_index(name="winning_segments")
        if not best_by_segment.empty
        else pd.DataFrame(columns=["lag_minutes", "winning_segments"])
    )
    return {
        "segment_frequency": freq,
        "tested_lags": lags,
        "focus_window_minutes": [focus_low, focus_high],
        "segment_count": int(best_by_segment["segment"].nunique()) if not best_by_segment.empty else 0,
        "segments_supporting_process_claim": int(len(focus_best)),
        "segment_share_supporting_process_claim": float(len(focus_best) / len(best_by_segment)) if len(best_by_segment) else 0.0,
        "winning_lag_consensus": consensus.to_dict(orient="records"),
        "best_lag_by_segment": best_by_segment.to_dict(orient="records"),
        "heatmaps": heatmap_paths,
        "rows": summary.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment the 50min DCS + PH lag experiment by time period.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--ph-path", default=str(DEFAULT_PH_PATH), help="PH Excel path.")
    parser.add_argument("--window-minutes", type=int, default=DEFAULT_DCS_WINDOW, help="Fixed DCS window size for the segmented experiment.")
    parser.add_argument("--ph-feature-window", type=int, default=50, help="PH rolling feature window size in minutes.")
    parser.add_argument("--lags", default="120,240,300", help="Comma-separated PH lags in minutes.")
    parser.add_argument("--segment-freq", default="Q", help="Pandas period frequency for segmentation. Default Q.")
    parser.add_argument("--min-segment-samples", type=int, default=80, help="Minimum aligned samples per segment to keep it in the report.")
    parser.add_argument("--tolerance", type=int, default=2, help="Merge tolerance in minutes.")
    parser.add_argument("--output-prefix", default="", help="Optional output filename prefix.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on evaluated samples per segment. Default 0 means all aligned samples.")
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

    summary_csv_path = results_dir / f"{output_prefix}ph_segmented_window_experiment_summary.csv"
    summary_json_path = results_dir / f"{output_prefix}ph_segmented_window_experiment_summary.json"
    composite_heatmap_path = results_dir / f"{output_prefix}ph_segmented_window_experiment_composite_heatmap.png"
    calcium_heatmap_path = results_dir / f"{output_prefix}ph_segmented_window_experiment_calcium_heatmap.png"

    dcs, lims, sensors, target_range, time_column = load_sources(Path(args.config))
    windows, outcomes = build_windows_and_outcomes(dcs, lims, args.window_minutes, time_column=time_column)
    base_casebase = build_casebase_from_windows(
        windows,
        outcomes,
        target_low=target_range[0],
        target_high=target_range[1],
        include_sensors=sensors,
    )
    base_casebase = assign_segments(base_casebase, args.segment_freq)

    ph = load_ph_data(args.ph_path)
    ph_features = build_ph_features(ph, feature_window_minutes=args.ph_feature_window)

    rows: list[dict[str, object]] = []
    segments = sorted(segment for segment, count in base_casebase["segment"].value_counts().items() if count >= args.min_segment_samples)
    augmented_by_lag = {
        lag_minutes: assign_segments(
            attach_ph_features(
                base_casebase.drop(columns=["segment"]),
                lims,
                ph_features,
                lag_minutes=lag_minutes,
                tolerance_minutes=args.tolerance,
            ),
            args.segment_freq,
        )
        for lag_minutes in lags
    }

    for segment in segments:
        print(f"running segmented baseline: {segment}")
        baseline_casebase = base_casebase.loc[base_casebase["segment"] == segment].drop(columns=["segment"]).reset_index(drop=True)
        baseline_casebase = prune_empty_numeric_columns(baseline_casebase)
        replay_summary, calcium_summary, bromine_summary, judgement = evaluate_casebase(
            baseline_casebase,
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
        baseline_row = {
            "segment": segment,
            "lag_minutes": -1,
            "aligned_samples": aligned_samples,
            "successful_recommendations": int(replay_summary["successful_recommendations"]),
            "failed_recommendations": int(replay_summary["failed_recommendations"]),
            "success_ratio": float(replay_summary["successful_recommendations"] / aligned_samples) if aligned_samples else 0.0,
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
        baseline_row["composite_score"] = (
            0.35 * (1.0 / (1.0 + float(baseline_row["calcium_mae"]) / 0.05))
            + 0.15 * (1.0 / (1.0 + float(baseline_row["bromine_mae"]) / 0.025))
            + 0.20 * float(baseline_row["in_spec_calcium_range_ratio"])
            + 0.10 * float(baseline_row["in_spec_bromine_range_ratio"])
            + 0.10 * float(baseline_row["success_ratio"])
            + 0.10
        )
        rows.append(baseline_row)

        for lag_minutes in lags:
            print(f"running segmented PH-augmented experiment: {segment} lag={lag_minutes}")
            casebase = augmented_by_lag[lag_minutes].loc[augmented_by_lag[lag_minutes]["segment"] == segment].drop(columns=["segment"]).reset_index(drop=True)
            casebase = prune_empty_numeric_columns(casebase)
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
            row = {
                "segment": segment,
                "lag_minutes": lag_minutes,
                "aligned_samples": aligned_samples,
                "successful_recommendations": int(replay_summary["successful_recommendations"]),
                "failed_recommendations": int(replay_summary["failed_recommendations"]),
                "success_ratio": float(replay_summary["successful_recommendations"] / aligned_samples) if aligned_samples else 0.0,
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
            row["composite_score"] = (
                0.35 * (1.0 / (1.0 + float(row["calcium_mae"]) / 0.05))
                + 0.15 * (1.0 / (1.0 + float(row["bromine_mae"]) / 0.025))
                + 0.20 * float(row["in_spec_calcium_range_ratio"])
                + 0.10 * float(row["in_spec_bromine_range_ratio"])
                + 0.10 * float(row["success_ratio"])
                + 0.10
            )
            rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["segment", "lag_minutes"]).reset_index(drop=True)
    baselines = summary.loc[summary["lag_minutes"] == -1, ["segment", "calcium_mae", "bromine_mae", "in_spec_calcium_range_ratio", "composite_score"]].rename(
        columns={
            "calcium_mae": "baseline_calcium_mae",
            "bromine_mae": "baseline_bromine_mae",
            "in_spec_calcium_range_ratio": "baseline_in_spec_calcium_range_ratio",
            "composite_score": "baseline_composite_score",
        }
    )
    summary = summary.merge(baselines, on="segment", how="left")
    summary["delta_calcium_mae_vs_baseline"] = summary["calcium_mae"] - summary["baseline_calcium_mae"]
    summary["delta_bromine_mae_vs_baseline"] = summary["bromine_mae"] - summary["baseline_bromine_mae"]
    summary["delta_in_spec_calcium_range_ratio_vs_baseline"] = summary["in_spec_calcium_range_ratio"] - summary["baseline_in_spec_calcium_range_ratio"]
    summary["delta_composite_score_vs_baseline"] = summary["composite_score"] - summary["baseline_composite_score"]

    experiment_only = summary.loc[summary["lag_minutes"] >= 0].copy()
    best_by_segment = experiment_only.sort_values(["segment", "composite_score"], ascending=[True, False]).groupby("segment", as_index=False).head(1).reset_index(drop=True)

    create_heatmap(experiment_only, "delta_composite_score_vs_baseline", composite_heatmap_path, "PH lag composite improvement vs 50min baseline")
    create_heatmap(experiment_only, "delta_in_spec_calcium_range_ratio_vs_baseline", calcium_heatmap_path, "PH lag calcium coverage improvement vs 50min baseline")

    heatmap_paths = {
        "composite_delta_heatmap": str(composite_heatmap_path),
        "calcium_range_delta_heatmap": str(calcium_heatmap_path),
    }
    report = build_report(
        summary,
        best_by_segment,
        freq=args.segment_freq,
        lags=lags,
        focus_low=args.focus_low,
        focus_high=args.focus_high,
        heatmap_paths=heatmap_paths,
    )

    summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    summary_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"summary_csv_saved_to={summary_csv_path}")
    print(f"summary_json_saved_to={summary_json_path}")
    print(f"composite_heatmap_saved_to={composite_heatmap_path}")
    print(f"calcium_heatmap_saved_to={calcium_heatmap_path}")


if __name__ == "__main__":
    main()
