from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def load_lims_samples(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    lims_cfg = config["lims"]
    excel = pd.ExcelFile(path)
    sheet_names = lims_cfg.get("sheet_names") or excel.sheet_names
    frames: list[pd.DataFrame] = []
    for sheet in sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        frame["source_sheet"] = str(sheet)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    time_col = lims_cfg["sample_time_col"]
    target_col = lims_cfg["target_col"]
    context_prefix = str(lims_cfg.get("context_feature_prefix", "lims_ctx"))
    raw["sample_time"] = pd.to_datetime(raw[time_col], errors="coerce")
    raw["t90"] = pd.to_numeric(raw[target_col], errors="coerce")
    valid_range = lims_cfg["valid_target_range"]
    raw = raw.dropna(subset=["sample_time"])
    raw = raw[(raw["t90"].isna()) | ((raw["t90"] >= float(valid_range["min"])) & (raw["t90"] <= float(valid_range["max"])))]

    grouped = raw.groupby("sample_time", dropna=False)
    context_cols: list[str] = []
    if lims_cfg.get("context_feature_mode") == "numeric_non_target":
        blocked = {time_col, target_col, "sample_time", "t90"}
        for col in raw.columns:
            if col in blocked or col == "source_sheet":
                continue
            numeric = pd.to_numeric(raw[col], errors="coerce")
            if numeric.notna().any():
                raw[col] = numeric
                context_cols.append(col)

    rows: list[dict[str, Any]] = []
    for idx, (sample_time, group) in enumerate(grouped, start=1):
        targets = group["t90"].dropna().astype(float)
        if targets.empty:
            continue
        row = {
            "sample_id": f"lims_{idx:06d}",
            "sample_time": sample_time,
            "t90": float(targets.mean()),
            "t90_repeat_count": int(targets.count()),
            "t90_repeat_std": float(targets.std(ddof=0)) if targets.count() > 1 else 0.0,
            "source_sheets": "|".join(sorted(group["source_sheet"].astype(str).unique())),
            "raw_record_count": int(len(group)),
        }
        for col in context_cols:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            safe_col = str(col).replace("\n", "_").replace("\r", "_")
            row[f"{context_prefix}__{safe_col}__mean"] = float(vals.mean()) if not vals.empty else np.nan
            row[f"{context_prefix}__{safe_col}__count"] = int(vals.count())
        rows.append(row)
    samples = pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)
    return samples


def filter_samples_to_dcs_time_range(
    samples: pd.DataFrame,
    dcs: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    boundary_cfg = config.get("data_boundary", {})
    if not bool(boundary_cfg.get("require_lims_within_dcs_time_range", True)):
        return samples.copy(), {"enabled": False, "dropped_samples": 0}
    time_col = config["dcs"]["time_col"]
    margin = pd.Timedelta(minutes=int(boundary_cfg.get("boundary_margin_minutes", 0)))
    dcs_min = dcs[time_col].min() + margin
    dcs_max = dcs[time_col].max() - margin
    mask = (samples["sample_time"] >= dcs_min) & (samples["sample_time"] <= dcs_max)
    filtered = samples.loc[mask].copy().reset_index(drop=True)
    return filtered, {
        "enabled": True,
        "dcs_time_min": dcs[time_col].min().isoformat() if pd.notna(dcs[time_col].min()) else None,
        "dcs_time_max": dcs[time_col].max().isoformat() if pd.notna(dcs[time_col].max()) else None,
        "effective_min": dcs_min.isoformat() if pd.notna(dcs_min) else None,
        "effective_max": dcs_max.isoformat() if pd.notna(dcs_max) else None,
        "input_lims_samples": int(len(samples)),
        "kept_lims_samples": int(len(filtered)),
        "dropped_samples": int((~mask).sum()),
        "dropped_before_dcs": int((samples["sample_time"] < dcs_min).sum()),
        "dropped_after_dcs": int((samples["sample_time"] > dcs_max).sum()),
    }


def load_dcs_frame(main_path: Path, supplemental_path: Path | None, config: dict[str, Any]) -> pd.DataFrame:
    time_col = config["dcs"]["time_col"]
    main = pd.read_csv(main_path)
    main[time_col] = pd.to_datetime(main[time_col], errors="coerce")
    main = main.dropna(subset=[time_col])
    frames = [main]
    if supplemental_path and supplemental_path.exists():
        supp = pd.read_csv(supplemental_path)
        supp[time_col] = pd.to_datetime(supp[time_col], errors="coerce")
        supp = supp.dropna(subset=[time_col])
        suffix = str(config["dcs"]["supplemental_suffix"])
        overlap = [col for col in supp.columns if col != time_col and col in main.columns]
        supp = supp.rename(columns={col: f"{col}{suffix}" for col in overlap})
        frames.append(supp)
    dcs = frames[0]
    for frame in frames[1:]:
        dcs = dcs.merge(frame, on=time_col, how="outer")
    dcs = dcs.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last").reset_index(drop=True)
    numeric_cols = [col for col in dcs.columns if col != time_col]
    for col in numeric_cols:
        dcs[col] = pd.to_numeric(dcs[col], errors="coerce")
    return dcs


def _representative_column(columns: list[str], policy: str) -> str:
    if policy == "shortest_name":
        return sorted(columns, key=lambda item: (len(item), item))[0]
    return sorted(columns)[0]


def _series_equivalent(left: pd.Series, right: pd.Series, min_overlap_fraction: float, max_abs_diff: float) -> bool:
    both = left.notna() & right.notna()
    min_non_null = min(int(left.notna().sum()), int(right.notna().sum()))
    if min_non_null == 0:
        return False
    if float(both.sum() / min_non_null) < min_overlap_fraction:
        return False
    diff = (left[both].astype(float) - right[both].astype(float)).abs()
    return bool(diff.max() <= max_abs_diff)


def deduplicate_dcs_columns(dcs: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dedup_cfg = config["dcs"].get("deduplication", {})
    if not bool(dedup_cfg.get("enabled", False)):
        return dcs.copy(), {"enabled": False, "dropped_columns": []}

    time_col = config["dcs"]["time_col"]
    columns = [col for col in dcs.columns if col != time_col]
    decimals = int(dedup_cfg.get("compare_round_decimals", 6))
    policy = str(dedup_cfg.get("representative_policy", "shortest_name"))
    min_overlap = float(dedup_cfg.get("min_non_null_overlap_fraction", 0.95))
    max_abs_diff = float(dedup_cfg.get("max_abs_diff", 10 ** (-decimals)))

    signatures: dict[tuple[int, int, int], list[str]] = {}
    rounded = dcs[columns].round(decimals)
    for col in columns:
        hashed = pd.util.hash_pandas_object(rounded[col], index=False)
        signature = (int(hashed.sum()), int(hashed.iloc[0] if len(hashed) else 0), int(rounded[col].notna().sum()))
        signatures.setdefault(signature, []).append(col)

    duplicate_groups: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for group_columns in signatures.values():
        if len(group_columns) < 2:
            continue
        remaining = [col for col in group_columns if col not in assigned]
        while remaining:
            seed = remaining[0]
            equivalent = [seed]
            for other in remaining[1:]:
                if _series_equivalent(rounded[seed], rounded[other], min_overlap, max_abs_diff):
                    equivalent.append(other)
            if len(equivalent) > 1:
                representative = _representative_column(equivalent, policy)
                dropped = [col for col in equivalent if col != representative]
                duplicate_groups.append(
                    {
                        "method": "rounded_exact",
                        "representative": representative,
                        "dropped": dropped,
                        "columns": equivalent,
                    }
                )
                assigned.update(equivalent)
            remaining = [col for col in remaining if col not in set(equivalent)]

    near_cfg = dedup_cfg.get("near_duplicate", {})
    if bool(near_cfg.get("enabled", False)):
        active_columns = [col for col in columns if col not in assigned]
        sample_rows = min(int(near_cfg.get("sample_rows", 20000)), len(dcs))
        if sample_rows > 0 and len(active_columns) > 1:
            sample_index = np.linspace(0, len(dcs) - 1, sample_rows, dtype=int)
            sample = dcs.iloc[sample_index][active_columns].astype(float)
            corr = sample.corr().abs()
            corr_threshold = float(near_cfg.get("corr_threshold", 0.999999))
            max_mean_abs_diff = float(near_cfg.get("max_mean_abs_diff", 1.0e-6))
            used: set[str] = set()
            for i, left in enumerate(active_columns):
                if left in used:
                    continue
                equivalent = [left]
                for right in active_columns[i + 1 :]:
                    if right in used:
                        continue
                    corr_value = corr.loc[left, right]
                    if pd.isna(corr_value) or corr_value < corr_threshold:
                        continue
                    both = sample[left].notna() & sample[right].notna()
                    min_non_null = min(int(sample[left].notna().sum()), int(sample[right].notna().sum()))
                    if min_non_null == 0 or float(both.sum() / min_non_null) < min_overlap:
                        continue
                    mean_abs_diff = float((sample.loc[both, left] - sample.loc[both, right]).abs().mean())
                    if mean_abs_diff <= max_mean_abs_diff:
                        equivalent.append(right)
                if len(equivalent) > 1:
                    representative = _representative_column(equivalent, policy)
                    dropped = [col for col in equivalent if col != representative]
                    duplicate_groups.append(
                        {
                            "method": "sample_near_duplicate",
                            "representative": representative,
                            "dropped": dropped,
                            "columns": equivalent,
                        }
                    )
                    used.update(equivalent)

    dropped_columns = sorted({col for group in duplicate_groups for col in group["dropped"]})
    result = dcs.drop(columns=dropped_columns)
    report = {
        "enabled": True,
        "input_columns": int(len(columns)),
        "output_columns": int(len([col for col in result.columns if col != time_col])),
        "dropped_count": int(len(dropped_columns)),
        "dropped_columns": dropped_columns,
        "duplicate_groups": duplicate_groups,
        "config": dedup_cfg,
    }
    return result, report


def apply_dcs_preprocessing(dcs: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    pre_cfg = config["dcs"].get("preprocessing", {})
    method = str(pre_cfg.get("denoise_method", "none")).lower()
    time_col = config["dcs"]["time_col"]
    numeric_cols = [col for col in dcs.columns if col != time_col]
    result = dcs.copy()
    report: dict[str, Any] = {
        "denoise_method": method,
        "input_columns": int(len(numeric_cols)),
        "rows": int(len(dcs)),
    }

    if bool(pre_cfg.get("hold_last_value", False)):
        before_missing = int(result[numeric_cols].isna().sum().sum())
        result[numeric_cols] = result[numeric_cols].ffill().bfill()
        after_missing = int(result[numeric_cols].isna().sum().sum())
        report["hold_last_value"] = True
        report["missing_before_hold_fill"] = before_missing
        report["missing_after_hold_fill"] = after_missing
    else:
        report["hold_last_value"] = False

    if bool(pre_cfg.get("hampel", {}).get("enabled", False)):
        hampel_cfg = pre_cfg["hampel"]
        window = int(hampel_cfg.get("window_rows", 11))
        n_sigmas = float(hampel_cfg.get("n_sigmas", 3.5))
        replacement = str(hampel_cfg.get("replacement", "median"))
        values = result[numeric_cols].astype(float)
        median = values.rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
        mad = (values - median).abs().rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
        threshold = n_sigmas * 1.4826 * mad
        outliers = (values - median).abs() > threshold
        if replacement == "interpolate":
            replaced = values.mask(outliers, np.nan)
            replaced.index = result[time_col]
            replaced = replaced.interpolate(method="time").bfill().ffill()
            replaced.index = result.index
            result[numeric_cols] = replaced
        else:
            result[numeric_cols] = values.mask(outliers, median)
        report["hampel_enabled"] = True
        report["hampel_window_rows"] = window
        report["hampel_n_sigmas"] = n_sigmas
        report["hampel_replacement"] = replacement
        report["hampel_replaced_values"] = int(outliers.sum().sum())
    else:
        report["hampel_enabled"] = False

    if method == "historian_sg":
        sg_cfg = pre_cfg.get("sg_filter", {})
        window_length = int(sg_cfg.get("window_length", 15))
        polyorder = int(sg_cfg.get("polyorder", 2))
        if window_length % 2 == 0:
            window_length += 1
        if window_length <= polyorder:
            window_length = polyorder + 3
            if window_length % 2 == 0:
                window_length += 1
        smoothed_cols: list[str] = []
        for col in numeric_cols:
            series = result[col].astype(float)
            if float(series.std(skipna=True) or 0.0) <= float(sg_cfg.get("min_std", 1.0e-6)):
                continue
            result[col] = savgol_filter(series.to_numpy(dtype=float), window_length=window_length, polyorder=polyorder)
            smoothed_cols.append(col)
        report["sg_filter_enabled"] = True
        report["sg_window_length"] = window_length
        report["sg_polyorder"] = polyorder
        report["sg_smoothed_columns"] = int(len(smoothed_cols))
    elif method == "ema":
        span = int(pre_cfg.get("ema_span_rows", 5))
        result[numeric_cols] = result[numeric_cols].ewm(span=span, adjust=False, ignore_na=True).mean()
        report["ema_span_rows"] = span
    elif method in {"none", "raw"}:
        pass
    elif method in {"mean", "mean_filter"}:
        span = int(pre_cfg.get("mean_window_rows", pre_cfg.get("ema_span_rows", 5)))
        result[numeric_cols] = result[numeric_cols].rolling(window=span, min_periods=1).mean()
        report["mean_window_rows"] = span
    else:
        raise ValueError(f"Unsupported DCS denoise_method: {method}")

    return result, report


def _asof_values(
    targets: pd.DataFrame,
    values: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
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


def build_feature_table(samples: pd.DataFrame, dcs: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_col = config["dcs"]["time_col"]
    stats = list(config["dcs"]["stats"])
    min_points = int(config["dcs"]["min_points_per_window"])
    tolerance = pd.Timedelta(minutes=int(config["dcs"].get("max_asof_gap_minutes", 10)))
    windows = [int(v) for v in config["alignment"]["candidate_window_minutes"]]
    lags = [int(v) for v in config["alignment"]["candidate_lag_minutes"]]
    primary_window = int(config["alignment"]["primary_window_minutes"])
    primary_lag = int(config["alignment"]["primary_lag_minutes"])

    indexed = dcs.set_index(time_col).sort_index()
    numeric_cols = list(indexed.columns)
    marker = pd.DataFrame({"_row_marker": 1.0}, index=indexed.index)
    features = samples[["sample_id", "sample_time"]].copy()
    features["_sample_order"] = np.arange(len(features))
    quality_rows: list[dict[str, Any]] = []

    for window in windows:
        rolling = indexed.rolling(f"{window}min", closed="right", min_periods=1)
        rolling_marker = marker.rolling(f"{window}min", closed="right", min_periods=1).count()
        stat_frames: dict[str, pd.DataFrame] = {}
        if "mean" in stats:
            stat_frames["mean"] = rolling.mean()
        if "median" in stats:
            stat_frames["median"] = rolling.median()
        if "var" in stats:
            stat_frames["var"] = rolling.var(ddof=0).fillna(0.0)
        if "std" in stats:
            stat_frames["std"] = rolling.std(ddof=0).fillna(0.0)
        if "min" in stats or "range" in stats:
            stat_frames["min"] = rolling.min()
        if "max" in stats or "range" in stats:
            stat_frames["max"] = rolling.max()
        count_frame = rolling.count()

        for lag in lags:
            prefix = f"w{window}_lag{lag}"
            targets = features[["sample_id", "sample_time", "_sample_order"]].copy()
            targets["asof_time"] = targets["sample_time"] - pd.Timedelta(minutes=lag)
            start_targets = targets.copy()
            start_targets["asof_time"] = start_targets["asof_time"] - pd.Timedelta(minutes=window)

            row_count = _asof_values(targets, rolling_marker, time_col, ["_row_marker"], tolerance)
            point_count = row_count["_row_marker"].fillna(0.0).astype(float)

            prefix_parts: list[pd.DataFrame] = []
            for stat_name, stat_frame in stat_frames.items():
                if stat_name not in {"min", "max"} or stat_name in stats or "range" in stats:
                    values = _asof_values(targets, stat_frame, time_col, numeric_cols, tolerance)
                    renamed = values[numeric_cols].rename(columns={col: f"{prefix}__{col}__{stat_name}" for col in numeric_cols})
                    if stat_name in stats:
                        prefix_parts.append(renamed)
                    if stat_name == "min":
                        min_values = values[numeric_cols]
                    if stat_name == "max":
                        max_values = values[numeric_cols]

            if "range" in stats and "min" in stat_frames and "max" in stat_frames:
                range_values = max_values - min_values
                prefix_parts.append(range_values.rename(columns={col: f"{prefix}__{col}__range" for col in numeric_cols}))

            if "last" in stats or "delta" in stats:
                last_values = _asof_values(targets, indexed, time_col, numeric_cols, tolerance)[numeric_cols]
                if "last" in stats:
                    prefix_parts.append(last_values.rename(columns={col: f"{prefix}__{col}__last" for col in numeric_cols}))
            if "delta" in stats:
                first_values = _asof_values(start_targets, indexed, time_col, numeric_cols, tolerance)[numeric_cols]
                delta_values = last_values - first_values
                prefix_parts.append(delta_values.rename(columns={col: f"{prefix}__{col}__delta" for col in numeric_cols}))
                if "slope" in stats:
                    slope_values = delta_values / float(window)
                    prefix_parts.append(slope_values.rename(columns={col: f"{prefix}__{col}__slope" for col in numeric_cols}))
            if "cv" in stats and "mean" in stat_frames and "std" in stat_frames:
                mean_values = _asof_values(targets, stat_frames["mean"], time_col, numeric_cols, tolerance)[numeric_cols]
                std_values = _asof_values(targets, stat_frames["std"], time_col, numeric_cols, tolerance)[numeric_cols]
                cv_values = std_values / mean_values.abs().replace(0.0, np.nan)
                prefix_parts.append(cv_values.rename(columns={col: f"{prefix}__{col}__cv" for col in numeric_cols}))

            if "missing_rate" in stats:
                valid_counts = _asof_values(targets, count_frame, time_col, numeric_cols, tolerance)[numeric_cols]
                denominator = point_count.replace(0.0, np.nan).to_numpy()[:, None]
                missing_values = 1.0 - (valid_counts.to_numpy(dtype=float) / denominator)
                missing_values = pd.DataFrame(missing_values, columns=numeric_cols).clip(lower=0.0, upper=1.0)
                prefix_parts.append(missing_values.rename(columns={col: f"{prefix}__{col}__missing_rate" for col in numeric_cols}))
                mean_missing = missing_values.mean(axis=1).fillna(1.0)
            else:
                mean_missing = pd.Series(0.0, index=features.index)

            window_features = pd.concat(prefix_parts, axis=1) if prefix_parts else pd.DataFrame(index=features.index)
            features = pd.concat([features.reset_index(drop=True), window_features.reset_index(drop=True)], axis=1)

            confidence = ((point_count / max(min_points, 1)).clip(upper=1.0) * (1.0 - mean_missing)).clip(lower=0.0, upper=1.0)
            quality = pd.DataFrame(
                {
                    "sample_id": samples["sample_id"].to_numpy(),
                    "window_minutes": window,
                    "lag_minutes": lag,
                    "point_count": point_count.to_numpy(dtype=float),
                    "sufficient_points": (point_count >= min_points).astype(int).to_numpy(),
                    "mean_missing_rate": mean_missing.to_numpy(dtype=float),
                    "window_confidence": confidence.to_numpy(dtype=float),
                }
            )
            quality_rows.extend(quality.to_dict(orient="records"))

    quality = pd.DataFrame(quality_rows)
    if quality.empty:
        features["align_confidence"] = 0.0
        features["state_confidence"] = 0.0
    else:
        align_conf = quality.groupby("sample_id")["window_confidence"].mean().rename("align_confidence")
        primary = quality[
            (quality["window_minutes"] == primary_window)
            & (quality["lag_minutes"] == primary_lag)
        ][["sample_id", "window_confidence"]].rename(columns={"window_confidence": "state_confidence"})
        features = features.merge(align_conf, on="sample_id", how="left")
        features = features.merge(primary, on="sample_id", how="left")
        features["state_confidence"] = features["state_confidence"].fillna(features["align_confidence"])
        features["align_confidence"] = features["align_confidence"].fillna(0.0)
    features = features.drop(columns=["sample_time", "_sample_order"], errors="ignore")
    return features, quality
