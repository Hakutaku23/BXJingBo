from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FEATURE_TABLE = (
    PROJECT_DIR
    / "outputs"
    / "20260415_165303_exp019_two_stage_stage_candidate_lags_ref10_limsctx"
    / "two_stage_feature_table.parquet"
)
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs"

HAL_RE = re.compile(r"^hal_lag(?P<lag>\d+)__(?P<raw>.+)_(?P<kind>lvl3|mean5|mad5|act5|slope5|range5)_L(?P<L>\d+)$")
REF_RE = re.compile(r"^ref__(?P<root>.+)_(?P<kind>act5|lvl3|mad5|mean5|range5|slope5|ageMin|lastStep|stepCnt30|val)_L(?P<L>\d+)$")

KEY_HALOGEN_VARIABLES = {
    "TICA_C52601",
    "TI_C50604_PV_F_CV",
}
KEY_REFINING_PREFIXES = ("sdu_", "vcu_", "dwd_")
LIMS_CONTEXT_PREFIX = "limsctx__"
META_COLS = [
    "sample_id",
    "sample_time",
    "split",
    "t90",
    "cd_mean",
    "p_pass_soft",
    "is_out_spec_obs",
    "sample_weight",
]

HALOGEN_STAGE_BY_VARIABLE = {
    "TI_C50604_PV_F_CV": "calcium_stearate_additive_feed",
    "TICA_C52601": "m526_post_r514_outlet",
}


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


def parse_feature(col: str) -> dict[str, Any] | None:
    m = HAL_RE.match(col)
    if m:
        raw = m.group("raw")
        if raw not in KEY_HALOGEN_VARIABLES:
            return None
        return {
            "source": "halogen",
            "original_variable": raw,
            "stage": HALOGEN_STAGE_BY_VARIABLE.get(raw, "manual_review"),
            "lag": f"lag{m.group('lag')}",
            "kind": m.group("kind"),
            "lag_index": int(m.group("L")),
            "feature": col,
        }

    m = REF_RE.match(col)
    if m:
        root = m.group("root")
        if not root.startswith(KEY_REFINING_PREFIXES):
            return None
        return {
            "source": "refining",
            "original_variable": root,
            "stage": f"ref_{root.split('_', 1)[0]}",
            "lag": "snap10",
            "kind": m.group("kind"),
            "lag_index": int(m.group("L")),
            "feature": col,
        }
    return None


def build_key_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = frame[META_COLS].copy()
    feature_map_rows: list[dict[str, Any]] = []
    parsed_by_group: dict[tuple[str, str, str, str, str], dict[int, str]] = {}
    selected_base_cols: list[str] = []

    for col in frame.columns:
        parsed = parse_feature(col)
        if parsed is None:
            continue
        selected_base_cols.append(col)
        group_key = (
            parsed["source"],
            parsed["stage"],
            parsed["original_variable"],
            parsed["lag"],
            parsed["kind"],
        )
        parsed_by_group.setdefault(group_key, {})[int(parsed["lag_index"])] = col
        feature_map_rows.append(
            {
                "feature": col,
                "feature_family": "base_point_derived",
                "source": parsed["source"],
                "stage": parsed["stage"],
                "original_variable": parsed["original_variable"],
                "original_variable_wildcard": f"{parsed['original_variable']}_*",
                "lag": parsed["lag"],
                "kind": parsed["kind"],
                "lag_index": f"L{parsed['lag_index']}",
                "formula": "existing cleaned/derived point feature",
            }
        )

    lims_cols = [
        col
        for col in frame.columns
        if col.startswith(LIMS_CONTEXT_PREFIX) and col.endswith("__mean") and "__count" not in col
    ]
    for col in lims_cols:
        selected_base_cols.append(col)
        feature_map_rows.append(
            {
                "feature": col,
                "feature_family": "lims_context",
                "source": "lims_context",
                "stage": "lims_context",
                "original_variable": col.replace(LIMS_CONTEXT_PREFIX, "").replace("__mean", ""),
                "original_variable_wildcard": col.replace(LIMS_CONTEXT_PREFIX, "").replace("__mean", "_*"),
                "lag": "context",
                "kind": "mean",
                "lag_index": "",
                "formula": "LIMS context mean from existing feature table",
            }
        )

    out = pd.concat([out.reset_index(drop=True), frame[selected_base_cols].reset_index(drop=True)], axis=1)

    change_features: dict[str, pd.Series] = {}
    for (source, stage, original_variable, lag, kind), by_lag_index in sorted(parsed_by_group.items()):
        if 0 in by_lag_index and 5 in by_lag_index:
            name = f"keychg__{source}__{stage}__{original_variable}__{lag}__{kind}__L0_minus_L5"
            change_features[name] = frame[by_lag_index[0]] - frame[by_lag_index[5]]
            feature_map_rows.append(
                {
                    "feature": name,
                    "feature_family": "window_change",
                    "source": source,
                    "stage": stage,
                    "original_variable": original_variable,
                    "original_variable_wildcard": f"{original_variable}_*",
                    "lag": lag,
                    "kind": kind,
                    "lag_index": "L0-L5",
                    "formula": f"{by_lag_index[0]} - {by_lag_index[5]}",
                }
            )

        near = [idx for idx in (0, 1) if idx in by_lag_index]
        far = [idx for idx in (4, 5) if idx in by_lag_index]
        if near and far:
            near_cols = [by_lag_index[idx] for idx in near]
            far_cols = [by_lag_index[idx] for idx in far]
            name = f"keychg__{source}__{stage}__{original_variable}__{lag}__{kind}__near_minus_far"
            change_features[name] = frame[near_cols].mean(axis=1) - frame[far_cols].mean(axis=1)
            feature_map_rows.append(
                {
                    "feature": name,
                    "feature_family": "window_change",
                    "source": source,
                    "stage": stage,
                    "original_variable": original_variable,
                    "original_variable_wildcard": f"{original_variable}_*",
                    "lag": lag,
                    "kind": kind,
                    "lag_index": "near-far",
                    "formula": f"mean({','.join(near_cols)}) - mean({','.join(far_cols)})",
                }
            )

        slope_idxs = [idx for idx in (0, 1, 2, 3, 4, 5) if idx in by_lag_index]
        if len(slope_idxs) >= 3:
            cols = [by_lag_index[idx] for idx in slope_idxs]
            y = frame[cols].to_numpy(dtype=float)
            x = np.asarray(slope_idxs, dtype=float)
            x_centered = x - x.mean()
            denom = float(np.sum(x_centered**2))
            valid = np.isfinite(y)
            row_count = valid.sum(axis=1)
            row_sum = np.where(valid, y, 0.0).sum(axis=1)
            row_mean = np.divide(row_sum, row_count, out=np.full_like(row_sum, np.nan, dtype=float), where=row_count > 0)
            centered = np.where(valid, y - row_mean[:, None], 0.0)
            slope = centered.dot(x_centered) / max(denom, 1.0e-12)
            slope[row_count < 2] = np.nan
            name = f"keychg__{source}__{stage}__{original_variable}__{lag}__{kind}__L0_to_L5_slope"
            change_features[name] = pd.Series(slope, index=frame.index)
            feature_map_rows.append(
                {
                    "feature": name,
                    "feature_family": "window_change",
                    "source": source,
                    "stage": stage,
                    "original_variable": original_variable,
                    "original_variable_wildcard": f"{original_variable}_*",
                    "lag": lag,
                    "kind": kind,
                    "lag_index": "L0-L5",
                    "formula": f"linear slope over {','.join(cols)}",
                }
            )

    if change_features:
        out = pd.concat([out, pd.DataFrame(change_features).reset_index(drop=True)], axis=1)

    feature_map = pd.DataFrame(feature_map_rows)
    variable_map = (
        feature_map.groupby(["source", "stage", "original_variable_wildcard"], as_index=False)
        .agg(
            feature_count=("feature", "count"),
            families=("feature_family", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .sort_values(["source", "stage", "original_variable_wildcard"])
    )
    return out, feature_map, variable_map


def screen_features(train: pd.DataFrame, features: list[str], *, max_missing_rate: float, min_variance: float) -> tuple[list[str], dict[str, Any]]:
    numeric = train[features].apply(pd.to_numeric, errors="coerce")
    missing = numeric.isna().mean()
    kept = [col for col in features if float(missing[col]) <= max_missing_rate]
    variance = numeric[kept].var(skipna=True).fillna(0.0) if kept else pd.Series(dtype=float)
    kept2 = [col for col in kept if float(variance[col]) > min_variance]
    return kept2, {
        "input_feature_count": int(len(features)),
        "after_missing_count": int(len(kept)),
        "after_variance_count": int(len(kept2)),
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
        "top_ranked_features": candidates[:50],
    }


def regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    y = y_true.to_numpy(dtype=float)
    p = np.asarray(pred, dtype=float)
    corr = spearmanr(y, p, nan_policy="omit").correlation if len(y) >= 3 else math.nan
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "spearman": float(corr) if corr is not None and not math.isnan(float(corr)) else math.nan,
    }


def binary_metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    y = y_true.to_numpy(dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    out = {
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan
    return out


def best_f1_threshold(y_true: pd.Series, prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), prob)
    if thresholds.size == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1.0e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def threshold_metrics(y_true: pd.Series, prob: np.ndarray, threshold: float) -> dict[str, float]:
    y = y_true.to_numpy(dtype=int)
    pred = (np.asarray(prob, dtype=float) >= float(threshold)).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alarm_rate": float(pred.mean()),
        "false_alarm_rate": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "miss_rate": float(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
    }


def fit_regressor(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> dict[str, Any]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    model = GradientBoostingRegressor(
        loss="huber",
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=10,
        alpha=0.9,
        random_state=20260416,
    )
    sample_weight = train["sample_weight"].to_numpy(dtype=float) if "sample_weight" in train else None
    model.fit(x_train, train[target].to_numpy(dtype=float), sample_weight=sample_weight)
    pred = np.clip(model.predict(x_test), 0.0, 1.0)
    out = regression_metrics(test[target], pred)
    if target == "p_pass_soft":
        out["out_spec_brier_from_p_pass"] = float(brier_score_loss(test["is_out_spec_obs"].astype(int), 1.0 - pred))
    return out


def fit_classifier(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_valid = imputer.transform(valid[features])
    x_test = imputer.transform(test[features])
    model = GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=20260416,
    )
    sample_weight = train["sample_weight"].to_numpy(dtype=float) if "sample_weight" in train else None
    model.fit(x_train, train["is_out_spec_obs"].astype(int), sample_weight=sample_weight)
    valid_prob = model.predict_proba(x_valid)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    threshold = best_f1_threshold(valid["is_out_spec_obs"], valid_prob)
    out = binary_metrics(test["is_out_spec_obs"], test_prob)
    out.update(threshold_metrics(test["is_out_spec_obs"], test_prob, threshold))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate key point-level window-change features.")
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--run-tag", type=str, default="exp023_key_variable_change")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reduced-max-features", type=int, default=300)
    parser.add_argument("--reduced-candidate-pool", type=int, default=900)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    args = parser.parse_args()

    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    raw = pd.read_parquet(args.feature_table)
    key_frame, feature_map, variable_map = build_key_features(raw)
    key_frame_path = run_dir / "key_variable_feature_table.parquet"
    key_frame.to_parquet(key_frame_path, index=False, compression="zstd")
    feature_map.to_csv(run_dir / "key_variable_feature_map.csv", index=False, encoding="utf-8-sig")
    variable_map.to_csv(run_dir / "key_variable_map.csv", index=False, encoding="utf-8-sig")

    features = [col for col in key_frame.columns if col not in META_COLS]
    train = key_frame[key_frame["split"] == "train"].copy()
    valid = key_frame[key_frame["split"] == "valid"].copy()
    test = key_frame[key_frame["split"] == "test"].copy()
    screened, screen_report = screen_features(train, features, max_missing_rate=0.95, min_variance=1.0e-12)

    map_lookup = feature_map.set_index("feature").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for mode in ["full", "reduced"]:
        for target in ["cd_mean", "p_pass_soft", "is_out_spec_obs"]:
            if mode == "full":
                selected = screened
                selection_report = {"mode": mode, **screen_report}
            else:
                selected, reduced_report = select_reduced_features(
                    train,
                    screened,
                    target,
                    max_features=args.reduced_max_features,
                    candidate_pool_size=args.reduced_candidate_pool,
                    corr_threshold=args.corr_threshold,
                )
                selection_report = {"mode": mode, **screen_report, **reduced_report}
                for rank, feature in enumerate(selected, start=1):
                    selected_rows.append(
                        {
                            "mode": mode,
                            "target": target,
                            "rank": rank,
                            "feature": feature,
                            **map_lookup.get(feature, {}),
                        }
                    )
            if not selected:
                continue
            if target == "is_out_spec_obs":
                metrics = fit_classifier(train, valid, test, selected)
                problem = "binary"
            else:
                metrics = fit_regressor(train, test, selected, target)
                problem = "regression"
            rows.append(
                {
                    "mode": mode,
                    "target": target,
                    "problem_type": problem,
                    "samples_train": int(len(train)),
                    "samples_valid": int(len(valid)),
                    "samples_test": int(len(test)),
                    "feature_count": int(len(selected)),
                    "selection_report": json.dumps(json_ready(selection_report), ensure_ascii=False),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected_rows)
    results.to_csv(run_dir / "key_variable_model_results.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(run_dir / "selected_key_variable_features.csv", index=False, encoding="utf-8-sig")

    input_dimension = {
        "source_feature_table": str(args.feature_table),
        "source_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "key_variable_feature_table": str(key_frame_path),
        "key_variable_shape": [int(key_frame.shape[0]), int(key_frame.shape[1])],
        "candidate_feature_count": int(len(features)),
        "screened_feature_count": int(len(screened)),
        "reduced_max_features": int(args.reduced_max_features),
        "split_counts": key_frame["split"].value_counts().to_dict(),
        "selected_variable_count": int(variable_map["original_variable_wildcard"].nunique()),
        "selected_variables": variable_map["original_variable_wildcard"].drop_duplicates().tolist(),
    }
    pd.DataFrame([input_dimension]).to_csv(run_dir / "input_dimension_report.csv", index=False, encoding="utf-8-sig")
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "input_dimension": input_dimension,
        "screen_report": screen_report,
        "results": json_ready(results.to_dict(orient="records")),
        "key_variable_map_path": str(run_dir / "key_variable_map.csv"),
        "selected_features_path": str(run_dir / "selected_key_variable_features.csv"),
        "flow_assignment_note": {
            "TI_C50604_PV_F_CV": "calcium stearate additive/feed-side temperature, not unknown_halogen",
            "TICA_C52601": "M526 outlet temperature after R514 and before flash section, not unknown_halogen",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
