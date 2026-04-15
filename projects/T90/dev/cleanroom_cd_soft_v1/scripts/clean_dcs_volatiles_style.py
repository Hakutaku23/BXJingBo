from __future__ import annotations

import argparse
import json
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
from cleanroom_cd_soft_v1.data import deduplicate_dcs_columns, load_dcs_frame
from cleanroom_cd_soft_v1.dcs_correction import InvalidRule, causal_hampel, contiguous_segments, invalid_mask


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


def normalize_tag(value: str) -> str:
    text = str(value)
    text = text.replace("B4-", "")
    text = text.replace(".", "_").replace("-", "_")
    for suffix in ["_PV_F_CV", "_PV_CV", "_S_PV_CV", "_S"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.upper()


def load_point_notes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_excel(path)
    if frame.empty or frame.shape[1] < 2:
        return {}
    name_col = frame.columns[0]
    file_col = frame.columns[1]
    notes: dict[str, str] = {}
    for _, row in frame.iterrows():
        tag = row.get(file_col)
        note = row.get(name_col)
        if pd.isna(tag) or pd.isna(note):
            continue
        notes[normalize_tag(str(tag))] = str(note)
    return notes


def infer_column_policy(column: str, note: str | None) -> dict[str, Any]:
    upper_col = column.upper()
    note_text = note or ""
    normalized = normalize_tag(column)
    prefix = normalized.split("_", 1)[0]

    is_temp = upper_col.startswith(("TI_", "TIC_", "TICA_")) or "温度" in note_text
    is_pressure = upper_col.startswith(("PI_", "PIC_")) or "压力" in note_text
    is_flow = upper_col.startswith(("FI_", "FIC_")) or "流量" in note_text or "添加量" in note_text or "总量" in note_text
    is_level = upper_col.startswith(("LI_", "LIC_")) or "液位" in note_text
    is_current = upper_col.startswith("II_") or "电流" in note_text
    is_analyzer = upper_col.startswith(("AI_", "AT_")) or "分析" in note_text
    is_step = any(token in note_text for token in ["开关", "阀位", "开度", "设定"]) or "_SP" in upper_col

    if is_step:
        variable_type = "step"
        fill_method = "previous"
        tau = None
    elif is_temp or is_level or is_analyzer:
        variable_type = "slow_continuous"
        fill_method = "linear"
        tau = 8.0
    elif is_pressure or is_flow or is_current:
        variable_type = "fast_continuous"
        fill_method = "linear"
        tau = 3.0
    else:
        variable_type = "continuous_unknown"
        fill_method = "linear"
        tau = 5.0

    low: float | None = None
    high: float | None = None
    if is_pressure or is_flow or is_level or is_current or is_analyzer:
        low = 0.0
    if is_step and any(token in note_text for token in ["开度", "阀位"]):
        low = 0.0
        high = 100.0

    return {
        "column": column,
        "normalized_tag": normalized,
        "description": note,
        "prefix": prefix,
        "variable_type": variable_type,
        "fill_method": fill_method,
        "low": low,
        "high": high,
        "hampel_window": 11,
        "hampel_n_sigmas": 3.0,
        "ewma_tau": tau,
    }


def fill_short_gaps_series(
    values: pd.Series,
    time_values: pd.Series,
    max_gap_points: int,
    method: str,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    segments = contiguous_segments(x.isna())
    filled = x.copy()
    flag = pd.Series(False, index=x.index)

    for _, segment in segments.iterrows():
        start = int(segment["start_index"])
        end = int(segment["end_index"])
        if int(segment["num_points"]) > int(max_gap_points):
            continue
        positions = np.arange(start, end + 1)
        if method == "previous":
            left = start - 1
            right = end + 1
            fill_value = np.nan
            if left >= 0 and not np.isnan(filled.iloc[left]):
                fill_value = filled.iloc[left]
            elif right < len(filled) and not np.isnan(filled.iloc[right]):
                fill_value = filled.iloc[right]
            if not np.isnan(fill_value):
                filled.iloc[positions] = fill_value
                flag.iloc[positions] = True
            continue

        good = filled.notna()
        if int(good.sum()) < 2:
            continue
        tmp = filled.copy()
        tmp.index = pd.to_datetime(time_values, errors="coerce")
        interpolated = tmp.interpolate(method="time", limit_area="inside")
        interpolated.index = filled.index
        can_fill = interpolated.iloc[positions].notna()
        fill_positions = positions[can_fill.to_numpy()]
        filled.iloc[fill_positions] = interpolated.iloc[fill_positions]
        flag.iloc[fill_positions] = True

    stats = {
        "segments_total": int(len(segments)),
        "segments_short": int((segments["num_points"] <= int(max_gap_points)).sum()) if not segments.empty else 0,
        "segments_long": int((segments["num_points"] > int(max_gap_points)).sum()) if not segments.empty else 0,
    }
    return filled, flag, stats


def segment_safe_ewma(values: pd.Series, tau: float) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    valid_segments = contiguous_segments(x.notna())
    y = pd.Series(np.nan, index=x.index, dtype=float)
    alpha = 1.0 - np.exp(-1.0 / float(tau))
    for _, segment in valid_segments.iterrows():
        start = int(segment["start_index"])
        end = int(segment["end_index"])
        segment_values = x.iloc[start : end + 1].to_numpy(dtype=float)
        if segment_values.size == 0:
            continue
        out = np.empty_like(segment_values, dtype=float)
        out[0] = segment_values[0]
        for idx in range(1, len(segment_values)):
            out[idx] = alpha * segment_values[idx] + (1.0 - alpha) * out[idx - 1]
        y.iloc[start : end + 1] = out
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean T90 DCS data with a Volatiles-style signal correction pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-name", type=str, default="merge_data_volatiles_style_cleaned.csv")
    parser.add_argument("--max-gap-points", type=int, default=6)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    data_dir = resolve_path(config_path, "../../../data")
    if data_dir is None:
        raise ValueError("Could not resolve T90 data directory.")

    dcs_main_path = resolve_path(config_path, config["paths"]["dcs_main_path"])
    dcs_supp_path = resolve_path(config_path, config["paths"].get("dcs_supplemental_path"))
    if dcs_main_path is None:
        raise ValueError("paths.dcs_main_path is required.")
    point_notes = load_point_notes(data_dir / "卤化点位.xlsx")

    dcs_raw = load_dcs_frame(dcs_main_path, dcs_supp_path, config)
    dcs_dedup, dedup_report = deduplicate_dcs_columns(dcs_raw, config)
    time_col = config["dcs"]["time_col"]
    dcs_dedup = dcs_dedup.sort_values(time_col).reset_index(drop=True)
    dcs_dedup[time_col] = pd.to_datetime(dcs_dedup[time_col], errors="coerce")
    numeric_cols = [col for col in dcs_dedup.columns if col != time_col]

    policies = []
    cleaned = pd.DataFrame({time_col: dcs_dedup[time_col]})
    report_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    for col in numeric_cols:
        note = point_notes.get(normalize_tag(col))
        policy = infer_column_policy(col, note)
        policies.append(policy)
        rule = InvalidRule(low=policy["low"], high=policy["high"])

        original = pd.to_numeric(dcs_dedup[col], errors="coerce").astype(float)
        invalid = invalid_mask(original, rule)
        invalid_segments = contiguous_segments(invalid)
        for _, segment in invalid_segments.iterrows():
            segment_rows.append(
                {
                    "column": col,
                    "start_index": int(segment["start_index"]),
                    "end_index": int(segment["end_index"]),
                    "num_points": int(segment["num_points"]),
                    "start_time": dcs_dedup[time_col].iloc[int(segment["start_index"])],
                    "end_time": dcs_dedup[time_col].iloc[int(segment["end_index"])],
                }
            )

        bounded = original.mask(invalid, np.nan)
        filled, fill_flag, fill_stats = fill_short_gaps_series(
            bounded,
            dcs_dedup[time_col],
            max_gap_points=int(args.max_gap_points),
            method=str(policy["fill_method"]),
        )
        despiked = causal_hampel(
            filled,
            window=int(policy["hampel_window"]),
            n_sigmas=float(policy["hampel_n_sigmas"]),
        )
        spike_replaced = int(((filled.notna()) & (despiked.notna()) & ~np.isclose(filled, despiked, equal_nan=True)).sum())

        if policy["ewma_tau"] is not None:
            corrected = segment_safe_ewma(despiked, tau=float(policy["ewma_tau"]))
        else:
            corrected = despiked
        cleaned[col] = corrected.astype("float32")

        report_rows.append(
            {
                "column": col,
                "description": note,
                "variable_type": policy["variable_type"],
                "fill_method": policy["fill_method"],
                "low": policy["low"],
                "high": policy["high"],
                "ewma_tau": policy["ewma_tau"],
                "missing_raw": int(original.isna().sum()),
                "invalid_after_bounds": int(invalid.sum()),
                "filled_short_gap_points": int(fill_flag.sum()),
                "missing_after_short_gap_fill": int(filled.isna().sum()),
                "hampel_replaced_points": spike_replaced,
                "missing_final": int(corrected.isna().sum()),
                **fill_stats,
            }
        )

    output_path = data_dir / args.output_name
    policy_path = data_dir / "merge_data_volatiles_style_column_policy.csv"
    column_report_path = data_dir / "merge_data_volatiles_style_column_report.csv"
    segment_report_path = data_dir / "merge_data_volatiles_style_invalid_segments.csv"
    summary_path = data_dir / "merge_data_volatiles_style_cleaning_report.json"

    cleaned.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.7g")
    pd.DataFrame(policies).to_csv(policy_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(report_rows).to_csv(column_report_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_rows).to_csv(segment_report_path, index=False, encoding="utf-8-sig")

    summary = {
        "output_path": str(output_path),
        "input_main_path": str(dcs_main_path),
        "input_supplemental_path": str(dcs_supp_path) if dcs_supp_path else None,
        "rows": int(len(cleaned)),
        "raw_columns": int(len([col for col in dcs_raw.columns if col != time_col])),
        "deduplicated_columns": int(len(numeric_cols)),
        "dropped_duplicate_columns": dedup_report.get("dropped_columns", []),
        "max_gap_points": int(args.max_gap_points),
        "pipeline": [
            "merge main and supplemental DCS by time",
            "drop exact/near duplicate point columns",
            "mark NaN and conservative engineering-bound violations as invalid",
            "fill only short gaps with linear interpolation or previous hold by point type",
            "replace impulse spikes with causal Hampel median",
            "apply segment-safe EWMA trend correction without crossing long missing gaps",
        ],
        "time_min": cleaned[time_col].min().isoformat() if not cleaned.empty else None,
        "time_max": cleaned[time_col].max().isoformat() if not cleaned.empty else None,
        "policy_path": str(policy_path),
        "column_report_path": str(column_report_path),
        "segment_report_path": str(segment_report_path),
        "deduplication_report": dedup_report,
    }
    summary_path.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
