from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import re

import pandas as pd

from .window_encoder import encode_dcs_window


def _normalize_label(label: object) -> str:
    text = str(label).strip().lower()
    text = text.replace("\xa0", "")
    return re.sub(r"[\s\-_/%(),.]+", "", text)


def _match_column(columns: Sequence[str], *, kind: str) -> str | None:
    for column in columns:
        raw = str(column)
        normalized = _normalize_label(raw)
        if kind == "sample_time":
            if normalized in {"sampletime", "samplingtime", "time", "timestamp"} or "采样时间" in raw:
                return column
        elif kind == "t90":
            if normalized in {"t90", "t90min", "tc90min"} or ("90" in normalized and "min" in normalized):
                return column
        elif kind == "calcium":
            if "hard脂酸钙" in raw or "硬脂酸钙" in raw:
                continue
            if normalized in {"calcium", "ca"} or ("钙" in raw and "量" in raw and "硬脂酸钙" not in raw):
                return column
        elif kind == "bromine":
            if normalized in {"bromine", "br"} or ("溴" in raw and "量" in raw):
                return column
    return None


def normalize_casebase_frame(
    frame: pd.DataFrame,
    *,
    target_low: float,
    target_high: float,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("`frame` must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Casebase is empty.")

    normalized = frame.copy()
    columns = list(normalized.columns)
    rename_map: dict[str, str] = {}

    sample_time_column = _match_column(columns, kind="sample_time")
    target_column = _match_column(columns, kind="t90")
    calcium_column = _match_column(columns, kind="calcium")
    bromine_column = _match_column(columns, kind="bromine")

    if target_column is None:
        raise ValueError("Casebase must contain a T90 column.")
    if calcium_column is None:
        raise ValueError("Casebase must contain a calcium column.")
    if bromine_column is None:
        raise ValueError("Casebase must contain a bromine column.")

    rename_map[target_column] = "t90"
    rename_map[calcium_column] = "calcium"
    rename_map[bromine_column] = "bromine"
    if sample_time_column is not None:
        rename_map[sample_time_column] = "sample_time"

    normalized = normalized.rename(columns=rename_map)

    if "sample_time" in normalized.columns:
        normalized["sample_time"] = pd.to_datetime(normalized["sample_time"], errors="coerce")
    else:
        normalized["sample_time"] = pd.date_range("2000-01-01", periods=len(normalized), freq="min")

    for column in ("t90", "calcium", "bromine"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    numeric_columns = normalized.select_dtypes(include=["number"]).columns.tolist()
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["t90", "calcium", "bromine"]).reset_index(drop=True)
    normalized["is_in_spec"] = normalized["t90"].between(target_low, target_high).astype(int)
    return normalized


def load_casebase_dataset(
    path: str | Path,
    *,
    target_low: float,
    target_high: float,
) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Casebase file not found: {source}")

    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        try:
            frame = pd.read_parquet(source)
        except ImportError as exc:
            raise ImportError(
                "Reading parquet casebase files requires `pyarrow` or `fastparquet`. "
                f"Please install one of them or convert `{source.name}` to CSV for delivery."
            ) from exc
    return normalize_casebase_frame(frame, target_low=target_low, target_high=target_high)


def save_casebase_dataset(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".csv":
        frame.to_csv(target, index=False, encoding="utf-8-sig")
    else:
        try:
            frame.to_parquet(target, index=False)
        except ImportError as exc:
            raise ImportError(
                "Writing parquet casebase files requires `pyarrow` or `fastparquet`. "
                f"Please install one of them or save `{target.name}` as CSV instead."
            ) from exc


def build_casebase_from_windows(
    windows: Sequence[pd.DataFrame],
    outcomes: pd.DataFrame,
    *,
    target_low: float,
    target_high: float,
    include_sensors: Iterable[str] = (),
) -> pd.DataFrame:
    if len(windows) != len(outcomes):
        raise ValueError("`windows` and `outcomes` must have the same length.")

    normalized_outcomes = normalize_casebase_frame(
        outcomes,
        target_low=target_low,
        target_high=target_high,
    )
    if len(normalized_outcomes) != len(windows):
        raise ValueError("Outcome normalization removed rows; please align windows and outcomes first.")

    rows: list[dict[str, object]] = []
    for window, outcome in zip(windows, normalized_outcomes.itertuples(index=False)):
        row = {
            "sample_time": getattr(outcome, "sample_time"),
            "t90": float(getattr(outcome, "t90")),
            "calcium": float(getattr(outcome, "calcium")),
            "bromine": float(getattr(outcome, "bromine")),
            "is_in_spec": int(getattr(outcome, "is_in_spec")),
        }
        for column in normalized_outcomes.columns:
            if column in {"sample_time", "t90", "calcium", "bromine", "is_in_spec"}:
                continue
            value = getattr(outcome, column, None)
            row[column] = value
        row.update(encode_dcs_window(window, include_sensors=include_sensors))
        rows.append(row)
    return pd.DataFrame(rows)
