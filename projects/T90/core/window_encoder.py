from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def encode_dcs_window(
    window: pd.DataFrame,
    *,
    include_sensors: Iterable[str] = (),
) -> dict[str, float]:
    if not isinstance(window, pd.DataFrame):
        raise TypeError("`window` must be a pandas DataFrame.")
    if window.empty:
        raise ValueError("DCS window is empty.")

    frame = window.copy()
    include_set = {str(item) for item in include_sensors}

    time_column = next(
        (col for col in frame.columns if str(col).lower() in {"time", "timestamp", "sample_time"}),
        None,
    )
    if time_column:
        time_index = pd.to_datetime(frame[time_column], errors="coerce")
        frame = frame.drop(columns=[time_column])
    else:
        time_index = None

    numeric = frame.select_dtypes(include=["number"]).copy()
    if include_set:
        numeric = numeric[[col for col in numeric.columns if col in include_set]]
    if numeric.empty:
        raise ValueError("No numeric DCS columns were found in the current window.")

    result: dict[str, float] = {}
    sample_index = np.arange(len(numeric), dtype=float)

    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if series.empty:
            continue

        values = series.to_numpy(dtype=float)
        result[f"{column}__last"] = float(values[-1])
        result[f"{column}__mean"] = float(values.mean())
        result[f"{column}__std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        result[f"{column}__min"] = float(values.min())
        result[f"{column}__max"] = float(values.max())
        result[f"{column}__delta"] = float(values[-1] - values[0])
        result[f"{column}__valid_ratio"] = float(len(values) / len(numeric))

        if len(values) > 1:
            if time_index is not None:
                valid_time = time_index.loc[series.index]
                valid_time = (valid_time - valid_time.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
                x = valid_time / 60.0
            else:
                x = sample_index[: len(values)]
            if np.nanstd(x) > 0:
                result[f"{column}__slope"] = float(np.polyfit(x, values, 1)[0])

    if not result:
        raise ValueError("No usable numeric signal could be encoded from the current DCS window.")
    return result
