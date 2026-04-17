from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = PROJECT_DIR / "outputs" / "20260416_124452_exp025_reassigned_stage_lags_ref10_limsctx"

HAL_RE = re.compile(r"^hal_lag(?P<lag>\d+)__(?P<raw>.+)_(?P<kind>lvl3|mean5|mad5|act5|slope5|range5)_L(?P<L>\d+)$")
REF_RE = re.compile(r"^ref__(?P<root>.+)_(?P<kind>act5|lvl3|mad5|mean5|range5|slope5|ageMin|lastStep|stepCnt30|val)_L(?P<L>\d+)$")
LIMS_RE = re.compile(r"^lims_ctx__(?P<name>.+)__mean$")


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
    return value


def screen_features(train: pd.DataFrame, features: list[str], *, max_missing_rate: float, min_variance: float) -> tuple[list[str], dict[str, Any]]:
    numeric = train[features].apply(pd.to_numeric, errors="coerce")
    missing = numeric.isna().mean()
    kept = [col for col in features if float(missing[col]) <= max_missing_rate]
    variance = numeric[kept].var(skipna=True).fillna(0.0) if kept else pd.Series(dtype=float)
    kept2 = [col for col in kept if float(variance[col]) > min_variance]
    return kept2, {
        "candidate_feature_count": int(len(features)),
        "after_missing_count": int(len(kept)),
        "screened_feature_count": int(len(kept2)),
        "max_missing_rate": float(max_missing_rate),
        "min_variance": float(min_variance),
    }


def select_reduced_features(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    max_features: int,
    candidate_pool_size: int,
    corr_threshold: float,
) -> tuple[list[str], dict[str, Any]]:
    y = pd.to_numeric(train[target], errors="coerce")
    usable = y.notna()
    x = train.loc[usable, features].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    yv = y.loc[usable].astype(float)
    y_centered = yv - float(yv.mean())
    x_centered = x - x.mean(axis=0)
    denom = np.sqrt((x_centered.pow(2).sum(axis=0) * float((y_centered**2).sum())).clip(lower=1.0e-30))
    scores = (x_centered.mul(y_centered, axis=0).sum(axis=0).abs() / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    candidates = scores.sort_values(ascending=False).head(candidate_pool_size).index.tolist()
    corr = x[candidates].corr().abs() if candidates else pd.DataFrame()
    selected: list[str] = []
    for col in candidates:
        if len(selected) >= max_features:
            break
        if not selected:
            selected.append(col)
            continue
        max_corr = corr.loc[col, selected].max()
        if pd.isna(max_corr) or float(max_corr) < corr_threshold:
            selected.append(col)
    return selected, {
        "candidate_pool_count": int(len(candidates)),
        "selected_feature_count": int(len(selected)),
    }


def feature_meta(feature: str, lag_policy: pd.DataFrame) -> dict[str, Any]:
    m = HAL_RE.match(feature)
    if m:
        raw = m.group("raw")
        policy = lag_policy[lag_policy["raw_column"] == raw]
        stage = policy["stage"].iloc[0] if not policy.empty else ""
        return {
            "feature": feature,
            "source": "halogen",
            "stage": stage,
            "original_variable": raw,
            "original_variable_wildcard": f"{raw}_*",
            "lag": f"lag{m.group('lag')}",
            "kind": m.group("kind"),
            "lag_index": f"L{m.group('L')}",
        }
    m = REF_RE.match(feature)
    if m:
        root = m.group("root")
        stage = f"ref_{root.split('_', 1)[0]}"
        return {
            "feature": feature,
            "source": "refining",
            "stage": stage,
            "original_variable": root,
            "original_variable_wildcard": f"{root}_*",
            "lag": "snap10",
            "kind": m.group("kind"),
            "lag_index": f"L{m.group('L')}",
        }
    m = LIMS_RE.match(feature)
    if m:
        name = m.group("name")
        return {
            "feature": feature,
            "source": "lims_context",
            "stage": "lims_context",
            "original_variable": name,
            "original_variable_wildcard": f"{name}_*",
            "lag": "context",
            "kind": "mean",
            "lag_index": "",
        }
    return {
        "feature": feature,
        "source": "unknown",
        "stage": "",
        "original_variable": "",
        "original_variable_wildcard": "",
        "lag": "",
        "kind": "",
        "lag_index": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize input dimensions and reduced selected features for a two-stage run.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-missing-rate", type=float, default=0.95)
    parser.add_argument("--min-variance", type=float, default=1.0e-12)
    parser.add_argument("--reduced-max-features", type=int, default=300)
    parser.add_argument("--reduced-candidate-pool", type=int, default=900)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    args = parser.parse_args()

    feature_table_path = args.run_dir / "two_stage_feature_table.parquet"
    lag_policy_path = args.run_dir / "halogen_lag_policy.csv"
    frame = pd.read_parquet(feature_table_path)
    lag_policy = pd.read_csv(lag_policy_path)
    features = [
        col
        for col in frame.columns
        if col.startswith(("hal__", "hal_lag", "ref__", "lims_ctx__")) and not col.endswith("__count")
    ]
    train = frame[frame["split"] == "train"].copy()
    screened, screen_report = screen_features(
        train,
        features,
        max_missing_rate=args.max_missing_rate,
        min_variance=args.min_variance,
    )

    selected_rows: list[dict[str, Any]] = []
    target_counts: dict[str, int] = {}
    for target in ["cd_mean", "p_pass_soft", "is_out_spec_obs"]:
        selected, _ = select_reduced_features(
            train,
            screened,
            target,
            max_features=args.reduced_max_features,
            candidate_pool_size=args.reduced_candidate_pool,
            corr_threshold=args.corr_threshold,
        )
        target_counts[target] = len(selected)
        for rank, feature in enumerate(selected, start=1):
            selected_rows.append({"target": target, "rank": rank, **feature_meta(feature, lag_policy)})

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(args.run_dir / "selected_two_stage_features.csv", index=False, encoding="utf-8-sig")

    variable_df = (
        selected_df.groupby(["target", "source", "stage", "original_variable_wildcard"], as_index=False)
        .agg(selected_feature_count=("feature", "count"))
        .sort_values(["target", "selected_feature_count"], ascending=[True, False])
    )
    variable_df.to_csv(args.run_dir / "selected_two_stage_variable_summary.csv", index=False, encoding="utf-8-sig")

    input_dimension = {
        "feature_table": str(feature_table_path),
        "feature_table_shape": [int(frame.shape[0]), int(frame.shape[1])],
        **screen_report,
        "reduced_selected_feature_counts": target_counts,
        "split_counts": frame["split"].value_counts(dropna=False).to_dict(),
    }
    pd.DataFrame([input_dimension]).to_csv(args.run_dir / "input_dimension_report.csv", index=False, encoding="utf-8-sig")
    (args.run_dir / "feature_selection_summary.json").write_text(
        json.dumps(json_ready(input_dimension), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(input_dimension), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
