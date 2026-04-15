"""Volatiles-style DCS feature construction for the T90 experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


CONTINUOUS_FEATURES = ("lvl3", "mean5", "mad5", "act5", "slope5", "range5")
STEP_FEATURES = ("val", "ageMin", "lastStep", "stepCnt30")


@dataclass(frozen=True)
class FeatureConfig:
    """Feature construction parameters matching the Volatiles MATLAB style."""

    level_window: int = 3
    mean_window: int = 5
    dispersion_window: int = 5
    activity_window: int = 5
    slope_window: int = 5
    range_window: int = 5
    step_count_window: int = 30
    continuous_lags: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    step_lags: tuple[int, ...] = (0, 1, 2, 3)
    min_valid_run: int = 1


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open index runs where mask is true."""

    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts, ends, strict=True))


def rolling_mad(series: pd.Series, window: int) -> pd.Series:
    """Causal rolling median absolute deviation."""

    def _mad(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan
        med = np.median(values)
        return float(np.median(np.abs(values - med)))

    return series.rolling(window=window, min_periods=1).apply(_mad, raw=True)


def _compute_continuous_base(series: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=series.index)
    out["lvl3"] = series.rolling(cfg.level_window, min_periods=1).median()
    out["mean5"] = series.rolling(cfg.mean_window, min_periods=1).mean()
    out["mad5"] = rolling_mad(series, cfg.dispersion_window)
    out["act5"] = series.diff().abs().rolling(cfg.activity_window, min_periods=1).sum()
    out["slope5"] = (series - series.shift(cfg.slope_window - 1)) / (cfg.slope_window - 1)
    out["range5"] = (
        series.rolling(cfg.range_window, min_periods=1).max()
        - series.rolling(cfg.range_window, min_periods=1).min()
    )
    return out


def _compute_step_base(series: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=series.index)
    out["val"] = series

    values = series.to_numpy(dtype=float)
    changed = np.zeros(values.shape[0], dtype=bool)
    valid_pair = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    changed[1:] = valid_pair & (values[1:] != values[:-1])

    age = np.full(values.shape[0], np.nan, dtype=float)
    last_change: int | None = None
    for i, is_change in enumerate(changed):
        if is_change:
            last_change = i
        if last_change is not None:
            age[i] = i - last_change

    last_step = np.full(values.shape[0], np.nan, dtype=float)
    current_step = np.nan
    for i in range(1, values.shape[0]):
        if changed[i]:
            current_step = values[i] - values[i - 1]
        last_step[i] = current_step

    step_count = (
        pd.Series(changed.astype(float), index=series.index)
        .rolling(cfg.step_count_window, min_periods=1)
        .sum()
    )

    out["ageMin"] = age
    out["lastStep"] = last_step
    out["stepCnt30"] = step_count
    return out


def _compute_segment_safe_base(
    series: pd.Series,
    *,
    variable_type: str,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    base_names = STEP_FEATURES if variable_type == "step" else CONTINUOUS_FEATURES
    out = pd.DataFrame(np.nan, index=series.index, columns=base_names, dtype=float)
    valid = np.isfinite(series.to_numpy(dtype=float))

    for start, end in contiguous_true_runs(valid):
        if end - start < cfg.min_valid_run:
            continue
        idx = series.index[start:end]
        seg = series.iloc[start:end]
        if variable_type == "step":
            seg_base = _compute_step_base(seg, cfg)
        else:
            seg_base = _compute_continuous_base(seg, cfg)
        out.loc[idx, seg_base.columns] = seg_base

    return out


def build_lagged_features_for_column(
    series: pd.Series,
    *,
    column: str,
    variable_type: str,
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build segment-safe Volatiles-style features for one DCS column."""

    cfg = cfg or FeatureConfig()
    variable_type = "step" if variable_type == "step" else "continuous"
    base = _compute_segment_safe_base(series, variable_type=variable_type, cfg=cfg)
    lags = cfg.step_lags if variable_type == "step" else cfg.continuous_lags

    out = pd.DataFrame(index=series.index)
    valid = np.isfinite(series.to_numpy(dtype=float))
    runs = contiguous_true_runs(valid)

    for feat_name in base.columns:
        base_values = base[feat_name]
        for lag in lags:
            out[f"{column}_{feat_name}_L{lag}"] = np.nan
            for start, end in runs:
                if end - start < cfg.min_valid_run:
                    continue
                idx = base.index[start:end]
                seg = base_values.iloc[start:end]
                if lag == 0:
                    lagged = seg
                else:
                    lagged = seg.shift(lag)
                out.loc[idx, f"{column}_{feat_name}_L{lag}"] = lagged.to_numpy()

    return out


def build_feature_batch(
    frame: pd.DataFrame,
    *,
    time_column: str,
    policy: pd.DataFrame,
    columns: Sequence[str],
    cfg: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a feature batch and a compact report for selected DCS columns."""

    cfg = cfg or FeatureConfig()
    features = pd.DataFrame({time_column: frame[time_column]})
    rows: list[dict[str, object]] = []

    policy_by_col = policy.set_index("column") if "column" in policy.columns else pd.DataFrame()

    for column in columns:
        if column not in frame.columns:
            continue
        variable_type = "continuous"
        if not policy_by_col.empty and column in policy_by_col.index:
            raw_type = str(policy_by_col.loc[column, "variable_type"])
            if "step" in raw_type.lower():
                variable_type = "step"

        feat = build_lagged_features_for_column(
            pd.to_numeric(frame[column], errors="coerce"),
            column=column,
            variable_type=variable_type,
            cfg=cfg,
        )
        features = pd.concat([features, feat], axis=1)
        rows.append(
            {
                "column": column,
                "variable_type": variable_type,
                "input_missing": int(frame[column].isna().sum()),
                "feature_columns": int(feat.shape[1]),
                "first_feature": feat.columns[0] if feat.shape[1] else "",
                "last_feature": feat.columns[-1] if feat.shape[1] else "",
            }
        )

    return features, pd.DataFrame(rows)


def batched(values: Sequence[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for i in range(0, len(values), size):
        yield list(values[i : i + size])
