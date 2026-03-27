from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


V3_ROOT = Path(__file__).resolve().parents[1]
T90_ROOT = V3_ROOT.parent
DEFAULT_DATA_ROOT = T90_ROOT / "data"


@dataclass(frozen=True)
class TargetSpec:
    center: float
    tolerance: float

    @property
    def low(self) -> float:
        return float(self.center - self.tolerance)

    @property
    def high(self) -> float:
        return float(self.center + self.tolerance)


@dataclass(frozen=True)
class SourcePaths:
    lims_path: Path
    dcs_main_path: Path
    dcs_supplemental_path: Path | None = None
    ph_path: Path | None = None


def normalize_name(name: object) -> str:
    return str(name).strip().replace("\xa0", " ")


def first_non_null(series: pd.Series) -> object:
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    return cleaned.iloc[0]


def infer_lims_column_map(columns: Iterable[object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_name in columns:
        name = normalize_name(raw_name)
        if "采样时间" in name:
            mapping["sample_time"] = name
        elif "样品名称" in name:
            mapping["sample_name"] = name
        elif "在线监测挥发分" in name:
            mapping["volatile_online"] = name
        elif "挥发分" in name and "在线" not in name:
            mapping["volatile_lab"] = name
        elif "90" in name.lower():
            mapping["t90"] = name
        elif "硬脂酸钙含量" in name:
            mapping["calcium_stearate"] = name
        elif "钙含量" in name:
            mapping["calcium"] = name
        elif "溴含量" in name:
            mapping["bromine"] = name
        elif "稳定剂" in name:
            mapping["stabilizer"] = name
        elif "门尼粘度" in name:
            mapping["mooney"] = name
        elif "防老剂" in name:
            mapping["antioxidant"] = name
    return mapping


def _read_excel_all_sheets(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    parts = []
    for sheet_name, frame in sheets.items():
        part = frame.copy()
        part["source_sheet"] = sheet_name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def load_lims_samples(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    raw = _read_excel_all_sheets(path)
    raw.columns = [normalize_name(col) for col in raw.columns]
    column_map = infer_lims_column_map(raw.columns)

    rename_map = {source: target for target, source in column_map.items()}
    frame = raw.rename(columns=rename_map).copy()

    if "sample_time" not in frame.columns:
        raise ValueError(f"Could not infer sample time column from {path}")

    frame["sample_time"] = pd.to_datetime(frame["sample_time"], errors="coerce")

    numeric_columns = [
        "volatile_lab",
        "volatile_online",
        "t90",
        "bromine",
        "calcium_stearate",
        "calcium",
        "stabilizer",
        "mooney",
        "antioxidant",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    aggregation: dict[str, object] = {}
    for column in ["sample_name", "source_sheet", *numeric_columns]:
        if column in frame.columns:
            aggregation[column] = first_non_null

    grouped = (
        frame.dropna(subset=["sample_time"])
        .sort_values("sample_time")
        .groupby("sample_time", as_index=False)
        .agg(aggregation)
        .sort_values("sample_time")
        .reset_index(drop=True)
    )
    return grouped, column_map


def add_out_of_spec_labels(samples: pd.DataFrame, spec: TargetSpec) -> pd.DataFrame:
    result = samples.copy()
    result["t90"] = pd.to_numeric(result.get("t90"), errors="coerce")
    result["is_above_spec"] = result["t90"] > spec.high
    result["is_below_spec"] = result["t90"] < spec.low
    result["is_out_of_spec"] = result["is_above_spec"] | result["is_below_spec"]
    result["is_in_spec"] = result["t90"].between(spec.low, spec.high, inclusive="both")
    return result


def _load_single_dcs_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "time" not in frame.columns:
        raise ValueError(f"DCS file has no 'time' column: {path}")
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    for column in frame.columns:
        if column == "time":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_dcs_frame(main_path: Path, supplemental_path: Path | None = None) -> pd.DataFrame:
    merged = _load_single_dcs_csv(main_path).set_index("time")
    if supplemental_path is not None and supplemental_path.exists():
        supplemental = _load_single_dcs_csv(supplemental_path).set_index("time")
        merged = merged.combine_first(supplemental)
    merged = merged.sort_index().reset_index()
    return merged


def load_ph_frame(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["time", "ph_value"])

    frame = pd.read_excel(path).copy()
    frame.columns = [normalize_name(col) for col in frame.columns]

    if {"col2", "col3"}.issubset(frame.columns):
        time_col = "col2"
        value_col = "col3"
    else:
        time_col = next((col for col in frame.columns if "time" in col.lower()), None)
        value_col = next((col for col in frame.columns if col != time_col), None)
        if time_col is None or value_col is None:
            raise ValueError(f"Could not infer PH columns from {path}")

    result = pd.DataFrame(
        {
            "time": pd.to_datetime(frame[time_col], errors="coerce"),
            "ph_value": pd.to_numeric(frame[value_col], errors="coerce"),
        }
    )
    return result.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def summarize_numeric_window(window: pd.DataFrame, time_column: str = "time") -> dict[str, float]:
    if window.empty:
        return {}

    numeric_columns = [column for column in window.columns if column != time_column]
    result: dict[str, float] = {}

    if time_column in window.columns:
        time_values = pd.to_datetime(window[time_column], errors="coerce")
        time_index = ((time_values - time_values.iloc[0]).dt.total_seconds() / 60.0).to_numpy()
    else:
        time_index = np.arange(len(window), dtype=float)

    for column in numeric_columns:
        series = pd.to_numeric(window[column], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue
        result[f"{column}__last"] = float(valid.iloc[-1])
        result[f"{column}__mean"] = float(valid.mean())
        result[f"{column}__std"] = float(valid.std(ddof=0)) if len(valid) > 1 else 0.0
        result[f"{column}__min"] = float(valid.min())
        result[f"{column}__max"] = float(valid.max())
        result[f"{column}__delta"] = float(valid.iloc[-1] - valid.iloc[0]) if len(valid) > 1 else 0.0
        result[f"{column}__valid_ratio"] = float(valid.notna().mean())

        if len(valid) > 1:
            valid_mask = series.notna().to_numpy()
            valid_x = time_index[valid_mask]
            valid_y = series[valid_mask].to_numpy(dtype=float)
            if np.nanstd(valid_x) > 0:
                result[f"{column}__slope"] = float(np.polyfit(valid_x, valid_y, deg=1)[0])
            else:
                result[f"{column}__slope"] = 0.0
        else:
            result[f"{column}__slope"] = 0.0
    return result


def build_dcs_feature_table(
    labeled_samples: pd.DataFrame,
    dcs: pd.DataFrame,
    window_minutes: int,
    min_points_per_window: int = 5,
) -> pd.DataFrame:
    sample_frame = labeled_samples.copy()
    sample_frame["sample_time"] = pd.to_datetime(sample_frame["sample_time"], errors="coerce")
    sample_frame = sample_frame.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)

    dcs_frame = dcs.copy()
    dcs_frame["time"] = pd.to_datetime(dcs_frame["time"], errors="coerce")
    dcs_frame = dcs_frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    dcs_times = dcs_frame["time"].to_numpy(dtype="datetime64[ns]")

    rows: list[dict[str, object]] = []
    for record in sample_frame.itertuples(index=False):
        sample_time = pd.Timestamp(record.sample_time)
        left_boundary = (sample_time - pd.Timedelta(minutes=window_minutes)).to_datetime64()
        right_boundary = sample_time.to_datetime64()
        left = np.searchsorted(dcs_times, left_boundary, side="left")
        right = np.searchsorted(dcs_times, right_boundary, side="right")
        window = dcs_frame.iloc[left:right]
        if len(window) < min_points_per_window:
            continue

        row = {
            "sample_time": sample_time,
            "t90": getattr(record, "t90", np.nan),
            "is_in_spec": getattr(record, "is_in_spec", False),
            "is_out_of_spec": getattr(record, "is_out_of_spec", False),
            "is_above_spec": getattr(record, "is_above_spec", False),
            "is_below_spec": getattr(record, "is_below_spec", False),
            "window_minutes": int(window_minutes),
            "rows_in_window": int(len(window)),
        }

        for optional_column in [
            "sample_name",
            "volatile_lab",
            "volatile_online",
            "bromine",
            "calcium_stearate",
            "calcium",
            "stabilizer",
            "mooney",
            "antioxidant",
        ]:
            if optional_column in sample_frame.columns:
                row[optional_column] = getattr(record, optional_column, np.nan)

        row.update(summarize_numeric_window(window, time_column="time"))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)
