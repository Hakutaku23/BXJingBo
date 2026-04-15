from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cleanroom_cd_soft_v1.config import load_config, resolve_path


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "base.yaml"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if np.isnan(number) else number
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def asof_values(targets: pd.DataFrame, values: pd.DataFrame, time_col: str, value_cols: list[str], tolerance: pd.Timedelta) -> pd.DataFrame:
    right = values.reset_index().rename(columns={values.index.name or "index": time_col}).sort_values(time_col)
    merged = pd.merge_asof(
        targets.sort_values("asof_time"),
        right[[time_col, *value_cols]],
        left_on="asof_time",
        right_on=time_col,
        direction="backward",
        tolerance=tolerance,
    )
    return merged.drop(columns=[time_col], errors="ignore").sort_values("_sample_order").reset_index(drop=True)


def safe_name(name: str) -> str:
    return str(name).replace("\n", "_").replace("\r", "_").replace(" ", "_")


def load_downstream_frame(path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config["downstream"]
    time_col = str(cfg.get("time_col", "time"))
    header = pd.read_csv(path, nrows=0)
    all_columns = list(header.columns)
    start = int(cfg.get("feature_start_col_1based", 8)) - 1
    end_exclusive = int(cfg.get("feature_end_col_1based", 411))
    selected_columns = all_columns[start:end_exclusive]
    blocked = {"Y", "Lab", "Lab_hold", "Lab_ageMin", "Y_cal"}
    selected_columns = [col for col in selected_columns if col not in blocked and not str(col).startswith("Y_L")]
    if time_col not in all_columns:
        raise ValueError(f"Downstream time column {time_col!r} not found in {path}")
    usecols = [time_col, *selected_columns]
    dtype = {col: "float32" for col in selected_columns}
    frame = pd.read_csv(path, usecols=usecols, dtype=dtype)
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(subset=[time_col], keep="last").reset_index(drop=True)

    numeric_cols: list[str] = []
    for col in selected_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
        numeric_cols.append(col)

    missing_rate = frame[numeric_cols].isna().mean()
    std = frame[numeric_cols].std(skipna=True)
    max_missing = float(cfg.get("max_missing_rate", 0.95))
    min_std = float(cfg.get("min_std", 1.0e-8))
    kept = [col for col in numeric_cols if float(missing_rate[col]) <= max_missing and float(std[col] or 0.0) > min_std]
    dropped_high_missing = [col for col in numeric_cols if float(missing_rate[col]) > max_missing]
    dropped_constant = [col for col in numeric_cols if col not in dropped_high_missing and float(std[col] or 0.0) <= min_std]

    duplicate_groups: list[dict[str, Any]] = []
    dropped_duplicate: list[str] = []
    sample_rows = min(int(cfg.get("duplicate_sample_rows", 20000)), len(frame))
    if sample_rows > 0 and len(kept) > 1:
        sample_index = np.linspace(0, len(frame) - 1, sample_rows, dtype=int)
        sample = frame.iloc[sample_index][kept].round(int(cfg.get("duplicate_round_decimals", 6)))
        signatures: dict[tuple[int, int], list[str]] = {}
        for col in kept:
            hashed = pd.util.hash_pandas_object(sample[col], index=False)
            signatures.setdefault((int(hashed.sum()), int(sample[col].notna().sum())), []).append(col)
        for cols in signatures.values():
            if len(cols) <= 1:
                continue
            representative = sorted(cols, key=lambda item: (len(str(item)), str(item)))[0]
            dropped = [col for col in cols if col != representative]
            duplicate_groups.append({"representative": representative, "dropped": dropped, "columns": cols})
            dropped_duplicate.extend(dropped)
    kept = [col for col in kept if col not in set(dropped_duplicate)]

    result = frame[[time_col, *kept]].copy()
    report = {
        "path": str(path),
        "rows": int(len(frame)),
        "input_columns": int(len(selected_columns)),
        "kept_columns": int(len(kept)),
        "dropped_high_missing_count": int(len(dropped_high_missing)),
        "dropped_constant_count": int(len(dropped_constant)),
        "dropped_duplicate_count": int(len(dropped_duplicate)),
        "dropped_high_missing_columns": dropped_high_missing[:100],
        "dropped_constant_columns": dropped_constant[:100],
        "duplicate_groups": duplicate_groups[:100],
        "time_min": frame[time_col].min().isoformat() if not frame.empty else None,
        "time_max": frame[time_col].max().isoformat() if not frame.empty else None,
        "selected_1based_range": [int(cfg.get("feature_start_col_1based", 8)), int(cfg.get("feature_end_col_1based", 411))],
    }
    return result, report


def build_downstream_features(samples: pd.DataFrame, downstream: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config["downstream"]
    time_col = str(cfg.get("time_col", "time"))
    prefix_root = str(cfg.get("prefix", "dw"))
    lags = [int(v) for v in cfg.get("candidate_lag_minutes", [0, 5, 15, 30])]
    feature_mode = str(cfg.get("feature_mode", "direct_lag_snapshot"))
    tolerance = pd.Timedelta(minutes=int(cfg.get("max_asof_gap_minutes", 5)))

    indexed = downstream.set_index(time_col).sort_index()
    numeric_cols = [col for col in downstream.columns if col != time_col]
    safe_columns = {col: safe_name(col) for col in numeric_cols}

    base = samples[["sample_id", "sample_time"]].copy()
    base["_sample_order"] = np.arange(len(base))
    parts: list[pd.DataFrame] = [base[["sample_id"]].copy()]
    quality_rows: list[dict[str, Any]] = []

    for lag in lags:
        prefix = f"{prefix_root}_lag{lag}"
        targets = base[["sample_id", "sample_time", "_sample_order"]].copy()
        targets["asof_time"] = targets["sample_time"] - pd.Timedelta(minutes=lag)

        values = asof_values(targets, indexed, time_col, numeric_cols, tolerance)[numeric_cols]
        renamed = values.rename(columns={col: f"{prefix}__{safe_columns[col]}__value" for col in numeric_cols})
        parts.append(renamed)

        availability = values.notna().mean(axis=1)
        quality_rows.extend(
            pd.DataFrame(
                {
                    "sample_id": base["sample_id"],
                    "lag_minutes": lag,
                    "feature_availability": availability,
                }
            ).to_dict(orient="records")
        )

    features = pd.concat([part.reset_index(drop=True) for part in parts], axis=1)
    quality = pd.DataFrame(quality_rows)
    report = {
        "feature_mode": feature_mode,
        "lags": lags,
        "input_columns": int(len(numeric_cols)),
        "generated_feature_count": int(features.shape[1] - 1),
        "mean_feature_availability": float(quality["feature_availability"].mean()) if not quality.empty else None,
        "by_lag": json_ready(quality.groupby("lag_minutes")["feature_availability"].mean().to_dict()) if not quality.empty else {},
    }
    return features, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment an existing prepared run with downstream cleaned process features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared-run-dir", type=Path, required=True)
    parser.add_argument("--run-tag", type=str, default="downstream_augmented")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    output_root = resolve_path(config_path, config["paths"]["output_root"])
    downstream_path = resolve_path(config_path, config["paths"].get("downstream_path"))
    if output_root is None or downstream_path is None:
        raise ValueError("paths.output_root and paths.downstream_path are required.")

    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    prepared_run = args.prepared_run_dir.resolve()
    base_feature_path = prepared_run / "feature_table.csv"
    if not base_feature_path.exists():
        raise FileNotFoundError(f"Missing prepared feature table: {base_feature_path}")

    shutil.copy2(config_path, run_dir / "config_snapshot.yaml")
    base = pd.read_csv(base_feature_path)
    samples = base[["sample_id", "sample_time"]].copy()
    samples["sample_time"] = pd.to_datetime(samples["sample_time"], errors="coerce")

    downstream, load_report = load_downstream_frame(downstream_path, config)
    downstream_features, feature_report = build_downstream_features(samples, downstream, config)
    augmented = base.merge(downstream_features, on="sample_id", how="left")

    augmented.to_csv(run_dir / "feature_table.csv", index=False, encoding="utf-8-sig")
    (run_dir / "downstream_load_report.json").write_text(json.dumps(json_ready(load_report), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "downstream_feature_report.json").write_text(json.dumps(json_ready(feature_report), ensure_ascii=False, indent=2), encoding="utf-8")
    if (prepared_run / "label_summary.json").exists():
        shutil.copy2(prepared_run / "label_summary.json", run_dir / "label_summary.json")
    if (prepared_run / "split_audit.json").exists():
        shutil.copy2(prepared_run / "split_audit.json", run_dir / "split_audit.json")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_prepared_run": str(prepared_run),
        "base_shape": list(base.shape),
        "augmented_shape": list(augmented.shape),
        "downstream_load_report": load_report,
        "downstream_feature_report": feature_report,
    }
    (run_dir / "downstream_augmentation_summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
