from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core import build_casebase_from_windows, get_context_sensors, get_target_range, load_runtime_config
from ph_augmented_window_experiment import attach_ph_features, evaluate_casebase, parse_lags
from ph_lag_experiment import build_ph_features, load_ph_data
from ph_segmented_window_experiment import prune_empty_numeric_columns
from test import build_windows_and_outcomes, load_dcs_data, load_lims_grouped


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_PH_PATH = PROJECT_DIR / "data" / "B4-AI-C53001A.PV.F_CV.xlsx"
DEFAULT_DCS_WINDOW = 50
BLOCKED_COLUMNS = {"sample_time", "t90", "is_in_spec", "calcium", "bromine"}


def parse_stage_counts(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    counts = sorted({int(item) for item in items})
    if not counts:
        raise ValueError("At least one stage count must be provided.")
    if any(count < 2 for count in counts):
        raise ValueError("Stage counts must be integers greater than or equal to 2.")
    return counts


def load_sources(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], tuple[float, float], str]:
    config = load_runtime_config(config_path)
    sensors = get_context_sensors(config)
    target_range = get_target_range(config)
    time_column = str(config.get("window", {}).get("time_column", "time"))
    data_sources = config.get("data_sources", {})
    dcs = load_dcs_data(data_sources.get("dcs_paths", []), sensors=sensors, time_column=time_column)
    lims = load_lims_grouped(PROJECT_DIR / str(data_sources["lims_path"]))
    return dcs, lims, sensors, target_range, time_column


def get_context_columns(casebase: pd.DataFrame) -> list[str]:
    numeric_columns = casebase.select_dtypes(include=["number"]).columns.tolist()
    return [column for column in numeric_columns if column not in BLOCKED_COLUMNS and casebase[column].notna().sum() > 0]


def evaluate_stage_counts(
    casebase: pd.DataFrame,
    *,
    stage_counts: list[int],
    min_stage_samples: int,
) -> tuple[pd.DataFrame, list[str], SimpleImputer, StandardScaler]:
    context_columns = get_context_columns(casebase)
    if not context_columns:
        raise ValueError("No usable DCS context columns were found for stage clustering.")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = scaler.fit_transform(imputer.fit_transform(casebase[context_columns]))

    rows: list[dict[str, object]] = []
    for stage_count in stage_counts:
        model = KMeans(n_clusters=stage_count, random_state=42, n_init=20)
        labels = model.fit_predict(matrix)
        counts = pd.Series(labels).value_counts().sort_index()
        silhouette = float(silhouette_score(matrix, labels))
        rows.append(
            {
                "stage_count": stage_count,
                "silhouette_score": silhouette,
                "min_stage_samples": int(counts.min()),
                "max_stage_samples": int(counts.max()),
                "stage_sample_counts": {f"stage_{int(index)}": int(value) for index, value in counts.items()},
                "meets_min_stage_samples": bool(int(counts.min()) >= min_stage_samples),
            }
        )

    summary = pd.DataFrame(rows).sort_values("stage_count").reset_index(drop=True)
    return summary, context_columns, imputer, scaler


def choose_stage_count(summary: pd.DataFrame) -> int:
    valid = summary.loc[summary["meets_min_stage_samples"]].copy()
    if valid.empty:
        raise ValueError("No tested stage count satisfied the minimum stage sample requirement.")
    best = valid.sort_values(["silhouette_score", "stage_count"], ascending=[False, True]).iloc[0]
    return int(best["stage_count"])


def assign_stage_labels(
    casebase: pd.DataFrame,
    *,
    stage_count: int,
    context_columns: list[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> tuple[pd.DataFrame, KMeans]:
    matrix = scaler.transform(imputer.transform(casebase[context_columns]))
    model = KMeans(n_clusters=stage_count, random_state=42, n_init=20)
    labels = model.fit_predict(matrix)
    staged = casebase.copy()
    staged["stage_id"] = labels.astype(int)
    staged["stage_name"] = staged["stage_id"].map(lambda value: f"stage_{int(value)}")
    return staged, model


def describe_stage_profiles(
    staged_casebase: pd.DataFrame,
    *,
    context_columns: list[str],
) -> pd.DataFrame:
    overall = staged_casebase[context_columns].mean(numeric_only=True)
    rows: list[dict[str, object]] = []
    for stage_name, frame in staged_casebase.groupby("stage_name"):
        stage_mean = frame[context_columns].mean(numeric_only=True)
        deltas = (stage_mean - overall).abs().sort_values(ascending=False).head(8)
        rows.append(
            {
                "stage_name": stage_name,
                "samples": int(len(frame)),
                "in_spec_ratio": float(pd.to_numeric(frame["is_in_spec"], errors="coerce").mean()),
                "t90_mean": float(pd.to_numeric(frame["t90"], errors="coerce").mean()),
                "calcium_mean": float(pd.to_numeric(frame["calcium"], errors="coerce").mean()),
                "bromine_mean": float(pd.to_numeric(frame["bromine"], errors="coerce").mean()),
                "top_context_shifts": {column: float(deltas[column]) for column in deltas.index},
            }
        )
    return pd.DataFrame(rows).sort_values("stage_name").reset_index(drop=True)


def build_stage_policy_rows(
    staged_casebase: pd.DataFrame,
    staged_augmented_by_lag: dict[int, pd.DataFrame],
    *,
    lags: list[int],
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
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage_name, baseline_stage in staged_casebase.groupby("stage_name"):
        print(f"evaluating stage baseline: {stage_name}")
        baseline_casebase = prune_empty_numeric_columns(baseline_stage.drop(columns=["stage_id", "stage_name"]).reset_index(drop=True))
        replay_summary, calcium_summary, bromine_summary, judgement = evaluate_casebase(
            baseline_casebase,
            limit=limit,
            neighbor_count=neighbor_count,
            local_neighbor_count=local_neighbor_count,
            probability_threshold=probability_threshold,
            grid_points=grid_points,
            mae_threshold=mae_threshold,
            p90_threshold=p90_threshold,
            max_threshold=max_threshold,
            in_spec_range_ratio_threshold=in_spec_range_ratio_threshold,
            success_ratio_threshold=success_ratio_threshold,
        )
        aligned_samples = int(replay_summary["aligned_samples"])
        baseline_row = {
            "stage_name": stage_name,
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
            print(f"evaluating PH stage policy: {stage_name} lag={lag_minutes}")
            stage_augmented = staged_augmented_by_lag[lag_minutes].loc[staged_augmented_by_lag[lag_minutes]["stage_name"] == stage_name]
            casebase = prune_empty_numeric_columns(stage_augmented.drop(columns=["stage_id", "stage_name"]).reset_index(drop=True))
            replay_summary, calcium_summary, bromine_summary, judgement = evaluate_casebase(
                casebase,
                limit=limit,
                neighbor_count=neighbor_count,
                local_neighbor_count=local_neighbor_count,
                probability_threshold=probability_threshold,
                grid_points=grid_points,
                mae_threshold=mae_threshold,
                p90_threshold=p90_threshold,
                max_threshold=max_threshold,
                in_spec_range_ratio_threshold=in_spec_range_ratio_threshold,
                success_ratio_threshold=success_ratio_threshold,
            )
            aligned_samples = int(replay_summary["aligned_samples"])
            row = {
                "stage_name": stage_name,
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

    summary = pd.DataFrame(rows).sort_values(["stage_name", "lag_minutes"]).reset_index(drop=True)
    baselines = summary.loc[summary["lag_minutes"] == -1, ["stage_name", "calcium_mae", "in_spec_calcium_range_ratio", "composite_score"]].rename(
        columns={
            "calcium_mae": "baseline_calcium_mae",
            "in_spec_calcium_range_ratio": "baseline_in_spec_calcium_range_ratio",
            "composite_score": "baseline_composite_score",
        }
    )
    summary = summary.merge(baselines, on="stage_name", how="left")
    summary["delta_calcium_mae_vs_baseline"] = summary["calcium_mae"] - summary["baseline_calcium_mae"]
    summary["delta_in_spec_calcium_range_ratio_vs_baseline"] = summary["in_spec_calcium_range_ratio"] - summary["baseline_in_spec_calcium_range_ratio"]
    summary["delta_composite_score_vs_baseline"] = summary["composite_score"] - summary["baseline_composite_score"]
    return summary


def create_policy_heatmap(summary: pd.DataFrame, value_column: str, output_path: Path, title: str) -> None:
    experiment_only = summary.loc[summary["lag_minutes"] >= 0].copy()
    pivot = experiment_only.pivot(index="stage_name", columns="lag_minutes", values=value_column).sort_index()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.8 * len(pivot.index))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(column) for column in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("PH lag minutes")
    ax.set_ylabel("DCS stage")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.set_ylabel(value_column, rotation=90)

    for row_index, stage_name in enumerate(pivot.index):
        for column_index, lag in enumerate(pivot.columns):
            value = pivot.loc[stage_name, lag]
            if pd.isna(value):
                continue
            ax.text(column_index, row_index, f"{value:.4f}", ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_policy_recommendation(best_by_stage: pd.DataFrame, *, enable_threshold: float, max_recommended_lag: int) -> dict[str, object]:
    recommendations: list[dict[str, object]] = []
    ph_enabled_count = 0
    long_lag_count = 0
    for row in best_by_stage.itertuples(index=False):
        enable_ph = bool(row.delta_composite_score_vs_baseline >= enable_threshold and row.lag_minutes <= max_recommended_lag)
        if enable_ph:
            ph_enabled_count += 1
        if int(row.lag_minutes) > 240:
            long_lag_count += 1
        recommendations.append(
            {
                "stage_name": row.stage_name,
                "best_lag_minutes": int(row.lag_minutes),
                "enable_ph": enable_ph,
                "delta_composite_score_vs_baseline": float(row.delta_composite_score_vs_baseline),
                "delta_calcium_mae_vs_baseline": float(row.delta_calcium_mae_vs_baseline),
                "delta_in_spec_calcium_range_ratio_vs_baseline": float(row.delta_in_spec_calcium_range_ratio_vs_baseline),
                "reason": (
                    "enable stage-specific PH enhancement"
                    if enable_ph
                    else "keep 50min DCS-only baseline"
                ),
            }
        )

    total = len(recommendations)
    return {
        "enable_threshold": enable_threshold,
        "max_recommended_lag_minutes": max_recommended_lag,
        "ph_enabled_stage_count": ph_enabled_count,
        "stage_count": total,
        "ph_enabled_stage_ratio": float(ph_enabled_count / total) if total else 0.0,
        "long_lag_candidate_count": long_lag_count,
        "recommendations": recommendations,
        "use_ph_in_v2": bool(ph_enabled_count > 0),
        "use_long_lag_over_4h_in_v2": False,
    }


def build_report(
    stage_count_summary: pd.DataFrame,
    chosen_stage_count: int,
    stage_profiles: pd.DataFrame,
    policy_summary: pd.DataFrame,
    policy_recommendation: dict[str, object],
    heatmap_paths: dict[str, str],
) -> dict[str, object]:
    experiment_only = policy_summary.loc[policy_summary["lag_minutes"] >= 0].copy()
    best_by_stage = experiment_only.sort_values(["stage_name", "delta_composite_score_vs_baseline"], ascending=[True, False]).groupby("stage_name", as_index=False).head(1).reset_index(drop=True)
    return {
        "tested_stage_counts": stage_count_summary.to_dict(orient="records"),
        "chosen_stage_count": chosen_stage_count,
        "stage_profiles": stage_profiles.to_dict(orient="records"),
        "best_policy_by_stage": best_by_stage.to_dict(orient="records"),
        "policy_recommendation": policy_recommendation,
        "heatmaps": heatmap_paths,
        "rows": policy_summary.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DCS-only stage identifier and decide whether PH should be enabled by stage.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--ph-path", default=str(DEFAULT_PH_PATH), help="PH Excel path.")
    parser.add_argument("--window-minutes", type=int, default=DEFAULT_DCS_WINDOW, help="Fixed DCS window size for the stage experiment.")
    parser.add_argument("--ph-feature-window", type=int, default=50, help="PH rolling feature window size in minutes.")
    parser.add_argument("--lags", default="120,240,300", help="Comma-separated PH lags in minutes.")
    parser.add_argument("--stage-counts", default="2,3,4,5,6", help="Comma-separated stage counts to evaluate with k-means.")
    parser.add_argument("--min-stage-samples", type=int, default=120, help="Minimum samples per stage for a stage-count candidate to be valid.")
    parser.add_argument("--enable-threshold", type=float, default=0.002, help="Minimum composite-score gain needed to enable PH in a stage.")
    parser.add_argument("--max-recommended-lag", type=int, default=240, help="Largest PH lag that will be recommended for second-phase use.")
    parser.add_argument("--tolerance", type=int, default=2, help="Merge tolerance in minutes.")
    parser.add_argument("--output-prefix", default="", help="Optional output filename prefix.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on evaluated samples per stage. Default 0 means all aligned samples.")
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

    lags = parse_lags(args.lags)
    stage_counts = parse_stage_counts(args.stage_counts)
    output_prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = results_dir / f"{output_prefix}stage_identifier_experiment_summary.csv"
    summary_json_path = results_dir / f"{output_prefix}stage_identifier_experiment_summary.json"
    composite_heatmap_path = results_dir / f"{output_prefix}stage_identifier_experiment_composite_heatmap.png"
    calcium_heatmap_path = results_dir / f"{output_prefix}stage_identifier_experiment_calcium_heatmap.png"

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
    staged_casebase, _model = assign_stage_labels(
        base_casebase,
        stage_count=chosen_stage_count,
        context_columns=context_columns,
        imputer=imputer,
        scaler=scaler,
    )
    stage_profiles = describe_stage_profiles(staged_casebase, context_columns=context_columns)

    ph = load_ph_data(args.ph_path)
    ph_features = build_ph_features(ph, feature_window_minutes=args.ph_feature_window)
    stage_lookup = staged_casebase[["sample_time", "stage_id", "stage_name"]].drop_duplicates()
    staged_augmented_by_lag: dict[int, pd.DataFrame] = {}
    for lag_minutes in lags:
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
        lags=lags,
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
    create_policy_heatmap(policy_summary, "delta_composite_score_vs_baseline", composite_heatmap_path, "Stage policy: composite improvement vs 50min baseline")
    create_policy_heatmap(policy_summary, "delta_in_spec_calcium_range_ratio_vs_baseline", calcium_heatmap_path, "Stage policy: calcium coverage improvement vs 50min baseline")

    best_by_stage = (
        policy_summary.loc[policy_summary["lag_minutes"] >= 0]
        .sort_values(["stage_name", "delta_composite_score_vs_baseline"], ascending=[True, False])
        .groupby("stage_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    policy_recommendation = build_policy_recommendation(
        best_by_stage,
        enable_threshold=args.enable_threshold,
        max_recommended_lag=args.max_recommended_lag,
    )
    heatmap_paths = {
        "composite_delta_heatmap": str(composite_heatmap_path),
        "calcium_range_delta_heatmap": str(calcium_heatmap_path),
    }
    report = build_report(
        stage_count_summary,
        chosen_stage_count,
        stage_profiles,
        policy_summary,
        policy_recommendation,
        heatmap_paths,
    )

    policy_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    summary_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"summary_csv_saved_to={summary_csv_path}")
    print(f"summary_json_saved_to={summary_json_path}")
    print(f"composite_heatmap_saved_to={composite_heatmap_path}")
    print(f"calcium_heatmap_saved_to={calcium_heatmap_path}")


if __name__ == "__main__":
    main()
