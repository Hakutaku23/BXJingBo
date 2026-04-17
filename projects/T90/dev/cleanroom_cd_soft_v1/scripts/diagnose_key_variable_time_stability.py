from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_KEY_RUN_DIR = PROJECT_DIR / "outputs" / "20260416_123028_exp023_key_variable_change"
DEFAULT_KEY_TABLE = DEFAULT_KEY_RUN_DIR / "key_variable_feature_table.parquet"
DEFAULT_FEATURE_MAP = DEFAULT_KEY_RUN_DIR / "key_variable_feature_map.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs"
TARGET = "is_out_spec_obs"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 20:
        return math.nan
    if pair["x"].nunique(dropna=True) < 2 or pair["y"].nunique(dropna=True) < 2:
        return math.nan
    value = spearmanr(pair["x"], pair["y"]).correlation
    return float(value) if value is not None and not math.isnan(float(value)) else math.nan


def sign(value: float) -> int:
    if value is None or math.isnan(float(value)) or abs(float(value)) < 1.0e-12:
        return 0
    return 1 if float(value) > 0 else -1


def feature_columns_by_variable(feature_map: pd.DataFrame, frame: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for variable, sub in feature_map.groupby("original_variable_wildcard"):
        cols = [col for col in sub["feature"].dropna().astype(str).tolist() if col in frame.columns]
        if cols:
            out[str(variable)] = cols
    return out


def pick_train_top_features(frame: pd.DataFrame, by_variable: dict[str, list[str]], *, top_k: int) -> pd.DataFrame:
    train = frame[frame["split"] == "train"].copy()
    rows: list[dict[str, Any]] = []
    for variable, cols in sorted(by_variable.items()):
        target = train[TARGET]
        scores = []
        for col in cols:
            corr = safe_spearman(train[col], target)
            scores.append((col, corr, abs(corr) if not math.isnan(corr) else -1.0))
        scores = sorted(scores, key=lambda item: item[2], reverse=True)
        for rank, (feature, corr, abs_corr) in enumerate(scores[:top_k], start=1):
            rows.append(
                {
                    "original_variable_wildcard": variable,
                    "rank": rank,
                    "feature": feature,
                    "train_spearman": corr,
                    "train_abs_spearman": abs_corr if abs_corr >= 0 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def build_split_stability(frame: pd.DataFrame, top_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in top_features.itertuples(index=False):
        for split_name, sub in frame.groupby("split"):
            corr_out = safe_spearman(sub[row.feature], sub[TARGET])
            corr_p_pass = safe_spearman(sub[row.feature], sub["p_pass_soft"])
            rows.append(
                {
                    "original_variable_wildcard": row.original_variable_wildcard,
                    "rank": int(row.rank),
                    "feature": row.feature,
                    "split": split_name,
                    "samples": int(len(sub)),
                    "out_spec_count": int(pd.to_numeric(sub[TARGET], errors="coerce").fillna(0).sum()),
                    "out_spec_rate": float(pd.to_numeric(sub[TARGET], errors="coerce").mean()),
                    "t90_mean": float(pd.to_numeric(sub["t90"], errors="coerce").mean()),
                    "spearman_out_spec": corr_out,
                    "spearman_p_pass_soft": corr_p_pass,
                }
            )
    return pd.DataFrame(rows)


def build_monthly_stability(frame: pd.DataFrame, top_features: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["month"] = pd.to_datetime(work["sample_time"], errors="coerce").dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for row in top_features.itertuples(index=False):
        for month, sub in work.dropna(subset=["month"]).groupby("month"):
            corr = safe_spearman(sub[row.feature], sub[TARGET])
            rows.append(
                {
                    "original_variable_wildcard": row.original_variable_wildcard,
                    "rank": int(row.rank),
                    "feature": row.feature,
                    "month": month,
                    "samples": int(len(sub)),
                    "out_spec_count": int(pd.to_numeric(sub[TARGET], errors="coerce").fillna(0).sum()),
                    "out_spec_rate": float(pd.to_numeric(sub[TARGET], errors="coerce").mean()),
                    "t90_mean": float(pd.to_numeric(sub["t90"], errors="coerce").mean()),
                    "spearman_out_spec": corr,
                }
            )
    return pd.DataFrame(rows)


def build_variable_summary(top_features: pd.DataFrame, split_df: pd.DataFrame, monthly_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    first_rank = top_features[top_features["rank"] == 1].copy()
    for row in first_rank.itertuples(index=False):
        split_sub = split_df[(split_df["original_variable_wildcard"] == row.original_variable_wildcard) & (split_df["rank"] == 1)]
        split_corr = split_sub.set_index("split")["spearman_out_spec"].to_dict()
        monthly_sub = monthly_df[
            (monthly_df["original_variable_wildcard"] == row.original_variable_wildcard) & (monthly_df["rank"] == 1)
        ]
        usable_month_corr = monthly_sub["spearman_out_spec"].dropna()
        train_corr = float(split_corr.get("train", math.nan))
        valid_corr = float(split_corr.get("valid", math.nan))
        test_corr = float(split_corr.get("test", math.nan))
        sign_flip_valid = sign(train_corr) != 0 and sign(valid_corr) != 0 and sign(train_corr) != sign(valid_corr)
        sign_flip_test = sign(train_corr) != 0 and sign(test_corr) != 0 and sign(train_corr) != sign(test_corr)
        month_range = (
            float(usable_month_corr.max() - usable_month_corr.min()) if not usable_month_corr.empty else math.nan
        )
        unstable_reasons: list[str] = []
        if sign_flip_valid:
            unstable_reasons.append("valid_sign_flip")
        if sign_flip_test:
            unstable_reasons.append("test_sign_flip")
        if not math.isnan(test_corr) and abs(test_corr) < 0.05:
            unstable_reasons.append("test_abs_corr_lt_0.05")
        if not math.isnan(month_range) and month_range > 0.35:
            unstable_reasons.append("month_corr_range_gt_0.35")
        rows.append(
            {
                "original_variable_wildcard": row.original_variable_wildcard,
                "top_feature": row.feature,
                "train_spearman": train_corr,
                "valid_spearman": valid_corr,
                "test_spearman": test_corr,
                "train_to_test_abs_drop": abs(train_corr) - abs(test_corr)
                if not math.isnan(train_corr) and not math.isnan(test_corr)
                else math.nan,
                "usable_months": int(usable_month_corr.count()),
                "monthly_corr_min": float(usable_month_corr.min()) if not usable_month_corr.empty else math.nan,
                "monthly_corr_max": float(usable_month_corr.max()) if not usable_month_corr.empty else math.nan,
                "monthly_corr_range": month_range,
                "unstable": bool(unstable_reasons),
                "unstable_reasons": ";".join(unstable_reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(["unstable", "train_to_test_abs_drop"], ascending=[False, False])


def build_dataset_profile(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["month"] = pd.to_datetime(work["sample_time"], errors="coerce").dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for key_name, group_col in [("split", "split"), ("month", "month")]:
        for group, sub in work.groupby(group_col):
            rows.append(
                {
                    "group_type": key_name,
                    "group": group,
                    "samples": int(len(sub)),
                    "out_spec_count": int(pd.to_numeric(sub[TARGET], errors="coerce").fillna(0).sum()),
                    "out_spec_rate": float(pd.to_numeric(sub[TARGET], errors="coerce").mean()),
                    "t90_mean": float(pd.to_numeric(sub["t90"], errors="coerce").mean()),
                    "p_pass_soft_mean": float(pd.to_numeric(sub["p_pass_soft"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose time/split stability of key variable features.")
    parser.add_argument("--key-table", type=Path, default=DEFAULT_KEY_TABLE)
    parser.add_argument("--feature-map", type=Path, default=DEFAULT_FEATURE_MAP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", type=str, default="exp024_key_variable_stability")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    frame = pd.read_parquet(args.key_table)
    feature_map = pd.read_csv(args.feature_map)
    by_variable = feature_columns_by_variable(feature_map, frame)
    top_features = pick_train_top_features(frame, by_variable, top_k=int(args.top_k))
    split_df = build_split_stability(frame, top_features)
    monthly_df = build_monthly_stability(frame, top_features)
    summary_df = build_variable_summary(top_features, split_df, monthly_df)
    profile_df = build_dataset_profile(frame)

    top_features.to_csv(run_dir / "train_top_features_by_variable.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(run_dir / "split_stability.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(run_dir / "monthly_stability.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_dir / "variable_stability_summary.csv", index=False, encoding="utf-8-sig")
    profile_df.to_csv(run_dir / "dataset_time_profile.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "key_table": str(args.key_table),
        "feature_map": str(args.feature_map),
        "key_table_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "variable_count": int(len(by_variable)),
        "top_k": int(args.top_k),
        "unstable_variable_count": int(summary_df["unstable"].sum()) if not summary_df.empty else 0,
        "stable_variable_count": int((~summary_df["unstable"]).sum()) if not summary_df.empty else 0,
        "summary_rows": json_ready(summary_df.to_dict(orient="records")),
    }
    (run_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
