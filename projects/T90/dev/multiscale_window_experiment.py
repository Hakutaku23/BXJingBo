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

from core import encode_dcs_window, get_context_sensors, get_target_range, load_runtime_config
from test import (
    _first_valid,
    evaluate_recommendations,
    load_dcs_data,
    load_lims_grouped,
    summarize_results,
)
from evaluate_calcium_accuracy import summarize_calcium_accuracy
from judge_calcium_acceptance import judge_acceptance
from window_size_experiment import summarize_bromine_accuracy


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_COMBINATIONS = ((15,), (20,), (30,), (60,), (15, 30), (15, 45), (15, 60), (20, 60))


def parse_combinations(value: str) -> list[tuple[int, ...]]:
    combinations: list[tuple[int, ...]] = []
    for raw_group in value.split(","):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        group = tuple(sorted({int(item.strip()) for item in raw_group.split("+") if item.strip()}))
        if not group:
            continue
        if any(window <= 1 for window in group):
            raise ValueError("Window sizes must be integers greater than 1.")
        combinations.append(group)
    if not combinations:
        raise ValueError("At least one window combination must be provided.")
    return combinations


def combination_label(combination: tuple[int, ...]) -> str:
    return "+".join(str(item) for item in combination)


def load_sources(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], tuple[float, float], str]:
    config = load_runtime_config(config_path)
    sensors = get_context_sensors(config)
    target_range = get_target_range(config)
    time_column = str(config.get("window", {}).get("time_column", "time"))
    data_sources = config.get("data_sources", {})
    dcs = load_dcs_data(data_sources.get("dcs_paths", []), sensors=sensors, time_column=time_column)
    lims = load_lims_grouped(PROJECT_DIR / str(data_sources["lims_path"]))
    return dcs, lims, sensors, target_range, time_column


def build_multiscale_casebase(
    *,
    dcs: pd.DataFrame,
    lims: pd.DataFrame,
    window_minutes_list: tuple[int, ...],
    sensors: list[str],
    target_range: tuple[float, float],
    time_column: str,
) -> pd.DataFrame:
    dcs = dcs.sort_values(time_column).reset_index(drop=True)
    time_index = pd.to_datetime(dcs[time_column]).to_numpy(dtype="datetime64[ns]")
    max_window = max(window_minutes_list)

    rows: list[dict[str, object]] = []
    required_columns = ["sample_time", "t90", "calcium", "bromine"]
    available_columns = [column for column in lims.columns if column not in required_columns]

    for row in lims.itertuples(index=False):
        sample_time = pd.Timestamp(row.sample_time)
        if pd.isna(sample_time) or pd.isna(row.t90) or pd.isna(row.calcium) or pd.isna(row.bromine):
            continue

        end_index = time_index.searchsorted(sample_time.to_datetime64(), side="right") - 1
        if end_index < max_window - 1:
            continue

        base_window = dcs.iloc[end_index - max_window + 1 : end_index + 1].copy()
        if len(base_window) != max_window:
            continue

        base_times = pd.to_datetime(base_window[time_column])
        if sample_time - base_times.iloc[-1] > pd.Timedelta(minutes=2):
            continue
        if base_times.iloc[-1] - base_times.iloc[0] > pd.Timedelta(minutes=max_window + 2):
            continue

        encoded_features: dict[str, float] = {}
        valid = True
        for window_minutes in window_minutes_list:
            window = base_window.tail(window_minutes).reset_index(drop=True)
            window_times = pd.to_datetime(window[time_column])
            if len(window) != window_minutes:
                valid = False
                break
            if window_times.iloc[-1] - window_times.iloc[0] > pd.Timedelta(minutes=window_minutes + 2):
                valid = False
                break
            encoded = encode_dcs_window(window, include_sensors=sensors)
            for feature_name, feature_value in encoded.items():
                encoded_features[f"w{window_minutes}__{feature_name}"] = feature_value

        if not valid:
            continue

        row_dict = {
            "sample_time": sample_time,
            "t90": float(row.t90),
            "calcium": float(row.calcium),
            "bromine": float(row.bromine),
            "is_in_spec": int(target_range[0] <= float(row.t90) <= target_range[1]),
        }
        source_row = row._asdict()
        for column in available_columns:
            row_dict[column] = _first_valid(pd.Series([source_row.get(column)]))
        row_dict.update(encoded_features)
        rows.append(row_dict)

    return pd.DataFrame(rows)


def run_combination_experiment(
    *,
    casebase: pd.DataFrame,
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
    frame = pd.DataFrame(rows)
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
    return frame.sort_values("composite_score", ascending=False).reset_index(drop=True)


def create_plot(summary_frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = summary_frame.sort_values("combination_label").reset_index(drop=True)
    labels = plot_frame["combination_label"].tolist()
    x = list(range(len(labels)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].bar(x, plot_frame["aligned_samples"], color="#4c78a8")
    axes[0, 0].set_title("Aligned samples by window combination")
    axes[0, 0].set_xticks(x, labels, rotation=30)
    axes[0, 0].set_ylabel("Aligned samples")

    axes[0, 1].plot(x, plot_frame["calcium_mae"], marker="o", label="calcium MAE", color="#d62728")
    axes[0, 1].plot(x, plot_frame["bromine_mae"], marker="o", label="bromine MAE", color="#2ca02c")
    axes[0, 1].set_title("Recommendation error by window combination")
    axes[0, 1].set_xticks(x, labels, rotation=30)
    axes[0, 1].set_ylabel("Absolute error")
    axes[0, 1].legend()

    axes[1, 0].plot(x, plot_frame["in_spec_calcium_range_ratio"], marker="o", label="in-spec calcium range ratio", color="#9467bd")
    axes[1, 0].plot(x, plot_frame["in_spec_bromine_range_ratio"], marker="o", label="in-spec bromine range ratio", color="#8c564b")
    axes[1, 0].set_title("In-spec range coverage")
    axes[1, 0].set_xticks(x, labels, rotation=30)
    axes[1, 0].set_ylabel("Coverage ratio")
    axes[1, 0].legend()

    axes[1, 1].bar(
        x,
        plot_frame["composite_score"],
        color=["#2ca02c" if passed else "#ff7f0e" for passed in plot_frame["acceptance_passed"]],
    )
    axes[1, 1].set_title("Composite score and acceptance")
    axes[1, 1].set_xticks(x, labels, rotation=30)
    axes[1, 1].set_ylabel("Composite score")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_report(summary_frame: pd.DataFrame, plot_path: Path, combinations: list[tuple[int, ...]]) -> dict[str, object]:
    if summary_frame.empty:
        return {
            "tested_combinations": [combination_label(item) for item in combinations],
            "message": "No valid multiscale experiment rows were produced.",
            "plot_path": str(plot_path),
        }

    best_by_score = summary_frame.sort_values("composite_score", ascending=False).iloc[0]
    best_by_calcium_mae = summary_frame.sort_values("calcium_mae", ascending=True).iloc[0]
    best_by_bromine_mae = summary_frame.sort_values("bromine_mae", ascending=True).iloc[0]

    return {
        "tested_combinations": [combination_label(item) for item in combinations],
        "plot_path": str(plot_path),
        "scoring_note": "composite_score is a heuristic that balances calcium MAE, bromine MAE, in-spec range coverage, recommendation success, and aligned sample count.",
        "best_combination_by_composite_score": best_by_score["combination_label"],
        "best_combination_by_calcium_mae": best_by_calcium_mae["combination_label"],
        "best_combination_by_bromine_mae": best_by_bromine_mae["combination_label"],
        "rows": summary_frame.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiscale DCS window combinations in dev only.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument(
        "--combinations",
        default=",".join(combination_label(item) for item in DEFAULT_COMBINATIONS),
        help="Comma-separated window combinations such as 15,20,15+30,15+60.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on evaluated samples per combination. Default 0 means all aligned samples.")
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
    combinations = parse_combinations(args.combinations)
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    dcs, lims, sensors, target_range, time_column = load_sources(config_path)

    rows: list[dict[str, object]] = []
    for combination in combinations:
        label = combination_label(combination)
        print(f"running multiscale experiment: {label}")
        casebase = build_multiscale_casebase(
            dcs=dcs,
            lims=lims,
            window_minutes_list=combination,
            sensors=sensors,
            target_range=target_range,
            time_column=time_column,
        )
        replay_summary, calcium_summary, bromine_summary, judgement = run_combination_experiment(
            casebase=casebase,
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
        success_ratio = float(replay_summary["successful_recommendations"] / aligned_samples) if aligned_samples > 0 else 0.0
        rows.append(
            {
                "combination_label": label,
                "window_count": len(combination),
                "max_window_minutes": max(combination),
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

    summary_frame = build_summary_frame(rows)
    summary_csv_path = results_dir / "multiscale_window_experiment_summary.csv"
    summary_json_path = results_dir / "multiscale_window_experiment_summary.json"
    plot_path = results_dir / "multiscale_window_experiment_metrics.png"

    summary_frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    create_plot(summary_frame, plot_path)
    report = build_report(summary_frame, plot_path, combinations)
    summary_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"summary_csv_saved_to={summary_csv_path}")
    print(f"summary_json_saved_to={summary_json_path}")
    print(f"plot_saved_to={plot_path}")


if __name__ == "__main__":
    main()
