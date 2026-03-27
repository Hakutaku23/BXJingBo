from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
V3_ROOT = EXPERIMENT_DIR.parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from core import TargetSpec, add_out_of_spec_labels, load_dcs_frame, load_lims_samples, load_ph_frame

DEFAULT_CONFIG_PATH = V3_ROOT / "config" / "phase1_warning.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def resolve_path(config_path: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    return (config_path.parent / relative_path).resolve()


def summarize_intervals(timestamps: pd.Series) -> dict[str, float | None]:
    ts = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return {"median_minutes": None, "p10_minutes": None, "p90_minutes": None}
    deltas = ts.diff().dropna().dt.total_seconds() / 60.0
    return {
        "median_minutes": float(deltas.median()),
        "p10_minutes": float(deltas.quantile(0.10)),
        "p90_minutes": float(deltas.quantile(0.90)),
    }


def evaluate_window_coverage(
    sample_times: pd.Series,
    dcs_times: pd.Series,
    candidate_windows: list[int],
    min_points_per_window: int,
) -> pd.DataFrame:
    dcs_sorted = pd.to_datetime(dcs_times, errors="coerce").dropna().sort_values()
    dcs_values = dcs_sorted.to_numpy(dtype="datetime64[ns]")
    rows: list[dict[str, Any]] = []

    for window_minutes in candidate_windows:
        counts: list[int] = []
        aligned = 0
        for sample_time in pd.to_datetime(sample_times, errors="coerce").dropna():
            left_boundary = (sample_time - pd.Timedelta(minutes=window_minutes)).to_datetime64()
            right_boundary = sample_time.to_datetime64()
            left = np.searchsorted(dcs_values, left_boundary, side="left")
            right = np.searchsorted(dcs_values, right_boundary, side="right")
            count = int(right - left)
            counts.append(count)
            if count >= min_points_per_window:
                aligned += 1

        counts_series = pd.Series(counts, dtype=float) if counts else pd.Series(dtype=float)
        rows.append(
            {
                "window_minutes": int(window_minutes),
                "sample_count": int(len(counts)),
                "aligned_samples": int(aligned),
                "aligned_ratio": float(aligned / len(counts)) if counts else 0.0,
                "median_rows_in_window": float(counts_series.median()) if not counts_series.empty else 0.0,
                "p10_rows_in_window": float(counts_series.quantile(0.10)) if not counts_series.empty else 0.0,
                "p90_rows_in_window": float(counts_series.quantile(0.90)) if not counts_series.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def top_non_null_dcs_columns(dcs: pd.DataFrame, top_k: int = 20) -> list[dict[str, Any]]:
    numeric = dcs.drop(columns=["time"], errors="ignore")
    if numeric.empty:
        return []
    ratios = numeric.notna().mean().sort_values(ascending=False).head(top_k)
    return [
        {
            "column": column,
            "non_null_ratio": float(ratio),
        }
        for column, ratio in ratios.items()
    ]


def build_summary(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    target_spec = TargetSpec(
        center=float(config["target_spec"]["center"]),
        tolerance=float(config["target_spec"]["tolerance"]),
    )
    source_config = config["data_sources"]
    lims_path = resolve_path(config_path, source_config["lims_path"])
    dcs_main_path = resolve_path(config_path, source_config["dcs_main_path"])
    dcs_supplemental_path = resolve_path(config_path, source_config.get("dcs_supplemental_path"))
    ph_path = resolve_path(config_path, source_config.get("ph_path"))

    lims_samples, lims_column_map = load_lims_samples(lims_path)
    lims_labeled = add_out_of_spec_labels(lims_samples, target_spec)
    dcs = load_dcs_frame(dcs_main_path, dcs_supplemental_path)
    ph = load_ph_frame(ph_path)

    labeled = lims_labeled.dropna(subset=["t90"]).copy()
    candidate_windows = [int(value) for value in config["window_search"]["candidate_minutes"]]
    min_points_per_window = int(config["window_search"]["min_points_per_window"])
    window_coverage = evaluate_window_coverage(
        labeled["sample_time"],
        dcs["time"],
        candidate_windows,
        min_points_per_window,
    )

    summary: dict[str, Any] = {
        "phase": config.get("phase", "phase1_warning"),
        "objective": config["objective"]["name"],
        "target_spec": {
            "center": target_spec.center,
            "tolerance": target_spec.tolerance,
            "low": target_spec.low,
            "high": target_spec.high,
        },
        "data_contract": {
            "lims_path": str(lims_path),
            "dcs_main_path": str(dcs_main_path),
            "dcs_supplemental_path": str(dcs_supplemental_path) if dcs_supplemental_path else None,
            "ph_path": str(ph_path) if ph_path else None,
            "lims_column_map": lims_column_map,
        },
        "source_summary": {
            "lims": {
                "grouped_samples": int(len(lims_samples)),
                "samples_with_t90": int(labeled["t90"].notna().sum()),
                "time_min": str(lims_samples["sample_time"].min()),
                "time_max": str(lims_samples["sample_time"].max()),
                "sampling_interval": summarize_intervals(lims_samples["sample_time"]),
            },
            "dcs": {
                "rows": int(len(dcs)),
                "columns_excluding_time": int(max(len(dcs.columns) - 1, 0)),
                "time_min": str(dcs["time"].min()),
                "time_max": str(dcs["time"].max()),
                "sampling_interval": summarize_intervals(dcs["time"]),
                "top_non_null_columns": top_non_null_dcs_columns(dcs),
            },
            "ph": {
                "rows": int(len(ph)),
                "time_min": str(ph["time"].min()) if not ph.empty else None,
                "time_max": str(ph["time"].max()) if not ph.empty else None,
                "sampling_interval": summarize_intervals(ph["time"]) if not ph.empty else {
                    "median_minutes": None,
                    "p10_minutes": None,
                    "p90_minutes": None,
                },
            },
        },
        "label_summary": {
            "samples_with_t90": int(len(labeled)),
            "in_spec_count": int(labeled["is_in_spec"].sum()),
            "out_of_spec_count": int(labeled["is_out_of_spec"].sum()),
            "above_spec_count": int(labeled["is_above_spec"].sum()),
            "below_spec_count": int(labeled["is_below_spec"].sum()),
            "out_of_spec_ratio": float(labeled["is_out_of_spec"].mean()) if len(labeled) else 0.0,
            "above_spec_ratio": float(labeled["is_above_spec"].mean()) if len(labeled) else 0.0,
            "below_spec_ratio": float(labeled["is_below_spec"].mean()) if len(labeled) else 0.0,
        },
        "window_coverage_rows": window_coverage.to_dict(orient="records"),
        "recommendation_for_next_step": {
            "message": (
                "Use these coverage rows to choose the first V3 same-sample warning "
                "baseline windows before any point screening or model comparison."
            ),
            "candidate_windows_ranked_by_alignment": (
                window_coverage.sort_values(
                    ["aligned_ratio", "median_rows_in_window", "window_minutes"],
                    ascending=[False, False, True],
                )["window_minutes"].tolist()
            ),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw data availability for V3 Phase 1 same-sample warning.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-prefix", type=str, default="phase1_warning_data_audit")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path, config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(config_path)
    json_path = artifact_dir / f"{args.output_prefix}_summary.json"
    csv_path = artifact_dir / f"{args.output_prefix}_window_coverage.csv"

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    pd.DataFrame(summary["window_coverage_rows"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
