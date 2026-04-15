"""Build Volatiles-style DCS feature parts for the T90 cleanroom experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = EXPERIMENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cleanroom_cd_soft_v1.dcs_features import FeatureConfig, batched, build_feature_batch

DATA_DIR = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build segment-safe Volatiles-style DCS feature tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "merge_data_volatiles_style_cleaned.csv",
        help="Cleaned DCS CSV. Defaults to the Volatiles-style cleaned DCS output.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DATA_DIR / "merge_data_volatiles_style_column_policy.csv",
        help="Column policy CSV produced by clean_dcs_volatiles_style.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "merge_data_volatiles_style_feature_parts",
        help="Directory where feature part parquet files will be written.",
    )
    parser.add_argument(
        "--time-column",
        default="time",
        help="Timestamp column name.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional explicit DCS columns to expand. Defaults to all policy columns.",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=None,
        help="Optional cap for smoke tests or staged generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of raw DCS columns expanded per parquet part.",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output format for feature parts.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec when --format parquet is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.policy.exists():
        raise FileNotFoundError(args.policy)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy = pd.read_csv(args.policy)
    if args.columns:
        columns = [c for c in args.columns if c in set(policy["column"])]
    else:
        columns = list(policy["column"])
    if args.max_columns is not None:
        columns = columns[: args.max_columns]
    if not columns:
        raise ValueError("No DCS columns selected for feature construction.")

    usecols = [args.time_column] + columns
    frame = pd.read_csv(args.input, usecols=usecols, parse_dates=[args.time_column])
    frame = frame.sort_values(args.time_column).reset_index(drop=True)

    cfg = FeatureConfig()
    manifest: dict[str, object] = {
        "input": str(args.input),
        "policy": str(args.policy),
        "output_dir": str(args.output_dir),
        "format": args.format,
        "time_column": args.time_column,
        "rows": int(frame.shape[0]),
        "raw_columns_selected": int(len(columns)),
        "batch_size": int(args.batch_size),
        "feature_config": {
            "level_window": cfg.level_window,
            "mean_window": cfg.mean_window,
            "dispersion_window": cfg.dispersion_window,
            "activity_window": cfg.activity_window,
            "slope_window": cfg.slope_window,
            "range_window": cfg.range_window,
            "step_count_window": cfg.step_count_window,
            "continuous_lags": list(cfg.continuous_lags),
            "step_lags": list(cfg.step_lags),
            "min_valid_run": cfg.min_valid_run,
        },
        "parts": [],
    }

    reports: list[pd.DataFrame] = []
    for part_idx, batch_cols in enumerate(batched(columns, args.batch_size)):
        features, report = build_feature_batch(
            frame[[args.time_column] + batch_cols],
            time_column=args.time_column,
            policy=policy,
            columns=batch_cols,
            cfg=cfg,
        )
        part_name = f"part_{part_idx:03d}.{args.format}"
        part_path = args.output_dir / part_name
        if args.format == "parquet":
            features.to_parquet(part_path, index=False, compression=args.compression)
        else:
            features.to_csv(part_path, index=False, encoding="utf-8-sig")

        reports.append(report)
        manifest["parts"].append(
            {
                "path": str(part_path),
                "raw_columns": batch_cols,
                "feature_columns": int(features.shape[1] - 1),
            }
        )
        print(
            f"Wrote {part_path} with {len(batch_cols)} raw columns "
            f"and {features.shape[1] - 1} feature columns."
        )

    report_all = pd.concat(reports, ignore_index=True)
    report_path = args.output_dir / "feature_part_report.csv"
    manifest_path = args.output_dir / "manifest.json"
    report_all.to_csv(report_path, index=False, encoding="utf-8-sig")
    manifest["report"] = str(report_path)
    manifest["total_feature_columns"] = int(report_all["feature_columns"].sum())

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
