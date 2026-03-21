from __future__ import annotations

from pathlib import Path

import pandas as pd

from core import build_t90_recommendation, load_example_dataset


def recommend_t90_controls(
    input_data: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Build a recommendation report for T90 in-spec control using calcium-first adjustments.

    Parameters
    ----------
    input_data:
        Optional dictionary-style interface payload. Supported keys:
        - data: a pandas DataFrame containing at least sample_time, t90, calcium, bromine.
        - data_path: path to a parquet/csv file with the same columns.
        - sample_time: specific sample timestamp to target. Defaults to latest out-of-spec sample.
        - target_range: two-item iterable [low, high], default [7.5, 8.5].
        - top_context_features, neighbor_count, local_neighbor_count,
          probability_threshold, grid_points, include_columns: algorithm parameters.
        - use_example_data: bool, load bundled example parquet when no explicit data is provided.
    """
    payload = input_data or {}
    data = payload.get("data")
    data_path = payload.get("data_path")
    use_example_data = bool(payload.get("use_example_data", not data and not data_path))

    if data is None:
        if data_path:
            path = Path(str(data_path))
            if path.suffix.lower() == ".csv":
                data = pd.read_csv(path)
            else:
                data = pd.read_parquet(path)
        elif use_example_data:
            data = load_example_dataset()
        else:
            raise ValueError("Please provide `data`, `data_path`, or set `use_example_data=True`.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")

    return build_t90_recommendation(
        data=data,
        sample_time=payload.get("sample_time"),
        target_range=payload.get("target_range"),
        top_context_features=int(payload.get("top_context_features", 12)),
        neighbor_count=int(payload.get("neighbor_count", 120)),
        local_neighbor_count=int(payload.get("local_neighbor_count", 80)),
        probability_threshold=float(payload.get("probability_threshold", 0.60)),
        grid_points=int(payload.get("grid_points", 31)),
        include_columns=tuple(payload.get("include_columns", ())),
    )
