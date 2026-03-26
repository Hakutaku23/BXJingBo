from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core import (
    build_casebase_from_windows,
    get_context_sensors,
    get_target_range,
    load_runtime_config,
    save_casebase_dataset,
)
from ph_augmented_window_experiment import attach_ph_features
from ph_lag_experiment import build_ph_features, load_ph_data
from ph_segmented_window_experiment import prune_empty_numeric_columns
from stage_identifier_experiment import (
    assign_stage_labels,
    build_policy_recommendation,
    build_stage_policy_rows,
    choose_stage_count,
    describe_stage_profiles,
    evaluate_stage_counts,
)
from test import build_windows_and_outcomes, load_dcs_data, load_lims_grouped


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_ASSETS_DIR = PROJECT_DIR / "assets"
DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_STAGE_COUNTS = [2, 3, 4, 5, 6]
DEFAULT_POLICY_LAGS = [120, 240, 300]


def _load_sources(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], tuple[float, float], dict[str, object]]:
    config = load_runtime_config(config_path)
    sensors = get_context_sensors(config)
    target_range = get_target_range(config)
    window_config = config.get("window", {})
    data_sources = config.get("data_sources", {})
    time_column = str(window_config.get("time_column", "time"))
    dcs = load_dcs_data(data_sources.get("dcs_paths", []), sensors=sensors, time_column=time_column)
    lims = load_lims_grouped(PROJECT_DIR / str(data_sources["lims_path"]))
    ph = load_ph_data(PROJECT_DIR / str(data_sources["ph_path"]))
    return dcs, lims, ph, sensors, target_range, config


def _serialize_stage_policy(
    *,
    config: dict[str, object],
    target_range: tuple[float, float],
    context_columns: list[str],
    stage_model,
    imputer,
    scaler,
    chosen_stage_count: int,
    stage_count_summary: pd.DataFrame,
    stage_profiles: pd.DataFrame,
    policy: dict[str, object],
) -> dict[str, object]:
    chosen_row = stage_count_summary.loc[stage_count_summary["stage_count"] == chosen_stage_count].iloc[0]
    stage_centroids = {}
    for index, center in enumerate(stage_model.cluster_centers_):
        stage_centroids[f"stage_{index}"] = {
            column: float(center[position])
            for position, column in enumerate(context_columns)
        }

    recommendations = policy.get("recommendations", [])
    recommendation_map = {
        item["stage_name"]: {
            "best_lag_minutes": int(item["best_lag_minutes"]),
            "enable_ph": bool(item["enable_ph"]),
            "delta_composite_score_vs_baseline": float(item["delta_composite_score_vs_baseline"]),
            "delta_calcium_mae_vs_baseline": float(item["delta_calcium_mae_vs_baseline"]),
            "delta_in_spec_calcium_range_ratio_vs_baseline": float(item["delta_in_spec_calcium_range_ratio_vs_baseline"]),
            "reason": str(item["reason"]),
        }
        for item in recommendations
    }

    return {
        "version": "2.0.0",
        "target_range": {"low": float(target_range[0]), "high": float(target_range[1])},
        "window": config.get("window", {}),
        "ph": config.get("ph", {}),
        "stage_identifier": {
            "stage_count": int(chosen_stage_count),
            "silhouette_score": float(chosen_row["silhouette_score"]),
            "min_stage_samples": int(chosen_row["min_stage_samples"]),
            "max_stage_samples": int(chosen_row["max_stage_samples"]),
            "stage_sample_counts": chosen_row["stage_sample_counts"],
            "context_columns": context_columns,
            "imputer_statistics": {
                column: float(value)
                for column, value in zip(context_columns, imputer.statistics_)
            },
            "scaler_mean": {
                column: float(value)
                for column, value in zip(context_columns, scaler.mean_)
            },
            "scaler_scale": {
                column: float(value)
                for column, value in zip(context_columns, scaler.scale_)
            },
            "stage_centroids": stage_centroids,
        },
        "stage_profiles": stage_profiles.to_dict(orient="records"),
        "policy": {
            "enable_threshold": float(policy["enable_threshold"]),
            "max_recommended_lag_minutes": int(policy["max_recommended_lag_minutes"]),
            "ph_enabled_stage_count": int(policy["ph_enabled_stage_count"]),
            "stage_count": int(policy["stage_count"]),
            "ph_enabled_stage_ratio": float(policy["ph_enabled_stage_ratio"]),
            "long_lag_candidate_count": int(policy["long_lag_candidate_count"]),
            "use_ph_in_v2": bool(policy["use_ph_in_v2"]),
            "use_long_lag_over_4h_in_v2": bool(policy["use_long_lag_over_4h_in_v2"]),
            "recommendations": recommendation_map,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build delivery-side T90 v2 assets from private offline data.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--assets-dir", default=str(DEFAULT_ASSETS_DIR), help="Target assets directory.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Dev artifacts directory.")
    parser.add_argument("--skip-parquet", action="store_true", help="Do not export parquet copies.")
    args = parser.parse_args()

    config_path = Path(args.config)
    assets_dir = Path(args.assets_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    dcs, lims, ph, sensors, target_range, config = _load_sources(config_path)
    window_minutes = int(config.get("window", {}).get("minutes", 50))
    time_column = str(config.get("window", {}).get("time_column", "time"))
    ph_feature_window = int(config.get("ph", {}).get("feature_window_minutes", 50))
    tolerance_minutes = int(config.get("ph", {}).get("tolerance_minutes", 2))
    stage_aware = config.get("stage_aware", {})
    if not isinstance(stage_aware, dict):
        stage_aware = {}

    windows, outcomes = build_windows_and_outcomes(dcs, lims, window_minutes, time_column=time_column)
    base_casebase = build_casebase_from_windows(
        windows,
        outcomes,
        target_low=target_range[0],
        target_high=target_range[1],
        include_sensors=sensors,
    )

    stage_count_summary, context_columns, imputer, scaler = evaluate_stage_counts(
        base_casebase,
        stage_counts=DEFAULT_STAGE_COUNTS,
        min_stage_samples=120,
    )
    chosen_stage_count = choose_stage_count(stage_count_summary)
    staged_casebase, stage_model = assign_stage_labels(
        base_casebase,
        stage_count=chosen_stage_count,
        context_columns=context_columns,
        imputer=imputer,
        scaler=scaler,
    )
    stage_profiles = describe_stage_profiles(staged_casebase, context_columns=context_columns)

    ph_features = build_ph_features(ph, feature_window_minutes=ph_feature_window)
    stage_lookup = staged_casebase[["sample_time", "stage_id", "stage_name"]].drop_duplicates()
    staged_augmented_by_lag: dict[int, pd.DataFrame] = {}
    for lag_minutes in DEFAULT_POLICY_LAGS:
        augmented = attach_ph_features(
            base_casebase,
            lims,
            ph_features,
            lag_minutes=lag_minutes,
            tolerance_minutes=tolerance_minutes,
        )
        staged_augmented_by_lag[lag_minutes] = augmented.merge(stage_lookup, on="sample_time", how="inner")

    policy_summary = build_stage_policy_rows(
        staged_casebase,
        staged_augmented_by_lag,
        lags=DEFAULT_POLICY_LAGS,
        limit=0,
        neighbor_count=150,
        local_neighbor_count=80,
        probability_threshold=0.60,
        grid_points=31,
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
        enable_threshold=float(stage_aware.get("enable_threshold", 0.002)),
        max_recommended_lag=int(config.get("ph", {}).get("max_supported_lag_minutes", 240)),
    )
    ph120_casebase = prune_empty_numeric_columns(staged_augmented_by_lag[120].copy())

    stage_policy = _serialize_stage_policy(
        config=config,
        target_range=target_range,
        context_columns=context_columns,
        stage_model=stage_model,
        imputer=imputer,
        scaler=scaler,
        chosen_stage_count=chosen_stage_count,
        stage_count_summary=stage_count_summary,
        stage_profiles=stage_profiles,
        policy=policy,
    )

    base_csv_path = assets_dir / "t90_casebase.csv"
    ph_csv_path = assets_dir / "t90_casebase_ph120.csv"
    json_path = assets_dir / "t90_stage_policy.json"
    save_casebase_dataset(staged_casebase, base_csv_path)
    save_casebase_dataset(ph120_casebase, ph_csv_path)
    if not args.skip_parquet:
        save_casebase_dataset(staged_casebase, assets_dir / "t90_casebase.parquet")
        save_casebase_dataset(ph120_casebase, assets_dir / "t90_casebase_ph120.parquet")
    json_path.write_text(json.dumps(stage_policy, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "window_minutes": window_minutes,
        "target_range": {"low": float(target_range[0]), "high": float(target_range[1])},
        "base_casebase_rows": int(len(staged_casebase)),
        "ph120_casebase_rows": int(len(ph120_casebase)),
        "chosen_stage_count": int(chosen_stage_count),
        "stage_policy": stage_policy["policy"],
        "exported_files": {
            "base_casebase_csv": str(base_csv_path),
            "ph120_casebase_csv": str(ph_csv_path),
            "stage_policy_json": str(json_path),
        },
    }
    summary_path = results_dir / "build_v2_delivery_assets_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
