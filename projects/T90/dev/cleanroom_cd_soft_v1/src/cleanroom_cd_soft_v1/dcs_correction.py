from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


FillMethod = Literal["linear", "pchip", "previous"]


@dataclass(frozen=True)
class InvalidRule:
    low: float | None = None
    high: float | None = None


def invalid_mask(values: pd.Series, rule: InvalidRule | None = None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.isna()
    if rule is not None:
        if rule.low is not None:
            mask = mask | (numeric < float(rule.low))
        if rule.high is not None:
            mask = mask | (numeric > float(rule.high))
    return mask.astype(bool)


def contiguous_segments(mask: pd.Series | np.ndarray) -> pd.DataFrame:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return pd.DataFrame(columns=["start_index", "end_index", "num_points"])
    diff = np.diff(np.r_[False, arr, False].astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return pd.DataFrame(
        {
            "start_index": starts.astype(int),
            "end_index": ends.astype(int),
            "num_points": (ends - starts + 1).astype(int),
        }
    )


def merge_nearby_invalid_segments(
    segments: pd.DataFrame,
    max_gap_points: int = 10,
    min_long_points: int = 10,
) -> pd.DataFrame:
    if segments.empty:
        return segments.copy()
    required = {"start_index", "end_index", "num_points"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"segments missing required columns: {sorted(missing)}")

    ordered = segments.sort_values("start_index").reset_index(drop=True)
    rows: list[dict[str, int]] = []
    current = ordered.iloc[0].to_dict()
    current["gap_points_swallowed"] = 0
    for _, row in ordered.iloc[1:].iterrows():
        gap = int(row["start_index"] - current["end_index"] - 1)
        can_merge = (
            gap >= 0
            and gap <= int(max_gap_points)
            and int(current["num_points"]) > int(min_long_points)
            and int(row["num_points"]) > int(min_long_points)
        )
        if can_merge:
            current["end_index"] = int(row["end_index"])
            current["num_points"] = int(current["num_points"] + gap + row["num_points"])
            current["gap_points_swallowed"] = int(current["gap_points_swallowed"] + gap)
        else:
            rows.append(current)
            current = row.to_dict()
            current["gap_points_swallowed"] = 0
    rows.append(current)
    return pd.DataFrame(rows)


def causal_hampel(values: pd.Series, window: int = 11, n_sigmas: float = 3.0) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    if window % 2 == 0:
        window += 1
    median = x.rolling(window=window, min_periods=max(3, window // 2)).median()
    mad = (x - median).abs().rolling(window=window, min_periods=max(3, window // 2)).median()
    sigma = 1.4826 * mad
    bad = (x - median).abs() > float(n_sigmas) * sigma
    y = x.copy()
    y[bad.fillna(False)] = median[bad.fillna(False)]
    return y


def ewma_trend(values: pd.Series, tau: float = 3.0) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
    y = np.full_like(x, np.nan, dtype=float)
    valid = np.flatnonzero(~np.isnan(x))
    if valid.size == 0:
        return pd.Series(y, index=values.index, dtype=float)
    alpha = 1.0 - np.exp(-1.0 / float(tau))
    first = int(valid[0])
    y[first] = x[first]
    for idx in range(first + 1, len(x)):
        if np.isnan(x[idx]):
            y[idx] = y[idx - 1]
        else:
            y[idx] = alpha * x[idx] + (1.0 - alpha) * y[idx - 1]
    return pd.Series(y, index=values.index, dtype=float)


def fill_short_gaps(
    frame: pd.DataFrame,
    time_col: str,
    columns: list[str],
    max_gap_points: int = 6,
    method_by_column: dict[str, FillMethod] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.sort_values(time_col).copy()
    time_values = pd.to_datetime(result[time_col], errors="coerce")
    method_by_column = method_by_column or {}
    report_rows: list[dict[str, object]] = []

    for col in columns:
        x = pd.to_numeric(result[col], errors="coerce").astype(float)
        segments = contiguous_segments(x.isna())
        filled = x.copy()
        fill_flag = pd.Series(False, index=result.index)
        method = method_by_column.get(col, "linear")

        for _, segment in segments.iterrows():
            start = int(segment["start_index"])
            end = int(segment["end_index"])
            idx_pos = np.arange(start, end + 1)
            if len(idx_pos) > int(max_gap_points):
                continue

            if method == "previous":
                left = start - 1
                right = end + 1
                value = np.nan
                if left >= 0 and not np.isnan(filled.iloc[left]):
                    value = filled.iloc[left]
                elif right < len(filled) and not np.isnan(filled.iloc[right]):
                    value = filled.iloc[right]
                if not np.isnan(value):
                    filled.iloc[idx_pos] = value
                    fill_flag.iloc[idx_pos] = True
                continue

            good = filled.notna()
            if int(good.sum()) < 2:
                continue
            interpolation_method = "pchip" if method == "pchip" else "time"
            tmp = filled.copy()
            tmp.index = time_values
            interpolated = tmp.interpolate(method=interpolation_method, limit_area="inside")
            interpolated.index = result.index
            can_fill = interpolated.iloc[idx_pos].notna()
            fill_positions = idx_pos[can_fill.to_numpy()]
            filled.iloc[fill_positions] = interpolated.iloc[fill_positions]
            fill_flag.iloc[fill_positions] = True

        result[col] = filled
        result[f"{col}_interpFlag"] = fill_flag.astype(bool)
        report_rows.append(
            {
                "column": col,
                "method": method,
                "missing_before": int(x.isna().sum()),
                "missing_after": int(filled.isna().sum()),
                "filled_points": int(fill_flag.sum()),
                "segments_total": int(len(segments)),
                "segments_short": int((segments["num_points"] <= int(max_gap_points)).sum()) if not segments.empty else 0,
            }
        )
    return result, pd.DataFrame(report_rows)


def correct_dcs_trends(
    frame: pd.DataFrame,
    time_col: str,
    continuous_columns: list[str],
    fast_columns: list[str] | None = None,
    hampel_window: int = 11,
    hampel_n_sigmas: float = 3.0,
    tau_fast: float = 3.0,
    tau_slow: float = 8.0,
) -> pd.DataFrame:
    result = frame.sort_values(time_col).copy()
    fast_set = set(fast_columns or [])
    for col in continuous_columns:
        filtered = causal_hampel(result[col], window=hampel_window, n_sigmas=hampel_n_sigmas)
        tau = tau_fast if col in fast_set else tau_slow
        result[f"{col}_trend"] = ewma_trend(filtered, tau=tau)
    return result

