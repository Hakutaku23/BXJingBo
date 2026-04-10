from __future__ import annotations

from typing import Any

import pandas as pd


def make_time_purged_split(samples: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_cfg = config["split"]
    ordered = samples.sort_values("sample_time").reset_index(drop=True)
    n = len(ordered)
    train_end_idx = max(1, int(n * float(split_cfg["train_ratio"])))
    valid_end_idx = max(train_end_idx + 1, int(n * (float(split_cfg["train_ratio"]) + float(split_cfg["valid_ratio"]))))
    train_cut = ordered.loc[min(train_end_idx, n - 1), "sample_time"]
    valid_cut = ordered.loc[min(valid_end_idx, n - 1), "sample_time"]
    purge = pd.Timedelta(minutes=int(split_cfg["purge_minutes"]))

    split = pd.Series("purged", index=ordered.index, dtype="object")
    split[ordered["sample_time"] < train_cut - purge] = "train"
    split[(ordered["sample_time"] >= train_cut + purge) & (ordered["sample_time"] < valid_cut - purge)] = "valid"
    split[ordered["sample_time"] >= valid_cut + purge] = "test"

    result = ordered[["sample_id", "sample_time"]].copy()
    result["split"] = split
    result["purge_minutes"] = int(split_cfg["purge_minutes"])

    present_group_cols = [col for col in split_cfg["group_cols"] if col in samples.columns]
    missing_group_cols = [col for col in split_cfg["group_cols"] if col not in samples.columns]
    audit = {
        "strategy": split_cfg["strategy"],
        "sample_count": int(n),
        "train_cut": train_cut.isoformat() if pd.notna(train_cut) else None,
        "valid_cut": valid_cut.isoformat() if pd.notna(valid_cut) else None,
        "purge_minutes": int(split_cfg["purge_minutes"]),
        "split_counts": result["split"].value_counts().to_dict(),
        "present_group_cols": present_group_cols,
        "missing_group_cols": missing_group_cols,
    }
    return result, audit


def leakage_audit(split: pd.DataFrame) -> dict[str, Any]:
    non_purged = split[split["split"].isin(["train", "valid", "test"])].copy()
    sample_cross_split = (
        non_purged.groupby("sample_id")["split"].nunique().gt(1).sum()
        if not non_purged.empty
        else 0
    )
    counts = non_purged["split"].value_counts().to_dict()
    gaps: dict[str, float | None] = {}
    for left, right in [("train", "valid"), ("valid", "test"), ("train", "test")]:
        left_times = non_purged.loc[non_purged["split"] == left, "sample_time"]
        right_times = non_purged.loc[non_purged["split"] == right, "sample_time"]
        if left_times.empty or right_times.empty:
            gaps[f"{left}_to_{right}_gap_minutes"] = None
        else:
            gaps[f"{left}_to_{right}_gap_minutes"] = float((right_times.min() - left_times.max()).total_seconds() / 60.0)
    return {
        "non_purged_split_counts": counts,
        "sample_ids_crossing_splits": int(sample_cross_split),
        **gaps,
    }

