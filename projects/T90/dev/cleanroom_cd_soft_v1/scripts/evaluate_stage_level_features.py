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


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def infer_halogen_stage(raw_col: str) -> str:
    col = str(raw_col).upper()
    if "C50604" in col:
        return "calcium_stearate_additive_feed"
    if "C52601" in col:
        return "m526_post_r514_outlet"
    if any(token in col for token in ["C500", "C510", "CM510", "51004", "51006", "51007"]):
        return "feed_r510"
    if any(token in col for token in ["C511", "CM511"]):
        return "r511"
    if any(token in col for token in ["C512", "CM512", "51204", "C517"]):
        return "r512_buffer"
    if any(token in col for token in ["C513", "CM513"]):
        return "r513_neutralization"
    if any(token in col for token in ["C514", "CM514", "C518", "C516", "C564"]):
        return "r514_additive"
    if any(token in col for token in ["C530", "CM530", "C30501", "53001", "53002", "53003", "53005"]):
        return "v530_flash"
    if any(token in col for token in ["C532", "CM532", "53202", "53203", "53205", "53252"]):
        return "v532_buffer"
    if any(token in col for token in ["C540", "CM540", "54002", "54003", "54051"]):
        return "v540_t300"
    return "unknown_halogen"


def infer_refining_stage(root: str) -> str:
    prefix = str(root).split("_", 1)[0]
    if prefix in {"dwd", "sdu", "vcu"}:
        return f"ref_{prefix}"
    return "ref_unknown"


def feature_source_from_stage_feature(feature: str) -> tuple[str, str, str]:
    parts = feature.split("__")
    if len(parts) < 6:
        return "", "", ""
    source = parts[1]
    stage = parts[2]
    descriptor = "__".join(parts[3:])
    return source, stage, descriptor


def aggregate_group(frame: pd.DataFrame, cols: list[str], prefix: str, aggs: list[str]) -> pd.DataFrame:
    numeric = frame[cols].apply(pd.to_numeric, errors="coerce")
    parts: dict[str, pd.Series] = {}
    if "mean" in aggs:
        parts[f"{prefix}__mean"] = numeric.mean(axis=1, skipna=True)
    if "std" in aggs:
        parts[f"{prefix}__std"] = numeric.std(axis=1, skipna=True).fillna(0.0)
    if "min" in aggs:
        parts[f"{prefix}__min"] = numeric.min(axis=1, skipna=True)
    if "max" in aggs:
        parts[f"{prefix}__max"] = numeric.max(axis=1, skipna=True)
    return pd.DataFrame(parts)


def build_stage_features(frame: pd.DataFrame, *, include_change: bool, aggs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_cols = [
        "sample_id",
        "sample_time",
        "split",
        "t90",
        "cd_mean",
        "p_pass_soft",
        "is_out_spec_obs",
        "sample_weight",
    ]
    out = frame[meta_cols].copy()
    groups: dict[tuple[str, str, str, str, str], list[str]] = {}
    variables: dict[tuple[str, str], set[str]] = {}

    for col in frame.columns:
        m = HAL_RE.match(col)
        if not m:
            continue
        raw = m.group("raw")
        stage = infer_halogen_stage(raw)
        lag = f"lag{m.group('lag')}"
        kind = m.group("kind")
        lidx = f"L{m.group('L')}"
        key = ("halogen", stage, lag, kind, lidx)
        groups.setdefault(key, []).append(col)
        variables.setdefault(("halogen", stage), set()).add(raw)

    for col in frame.columns:
        m = REF_RE.match(col)
        if not m:
            continue
        root = m.group("root")
        stage = infer_refining_stage(root)
        kind = m.group("kind")
        lidx = f"L{m.group('L')}"
        key = ("refining", stage, "snap10", kind, lidx)
        groups.setdefault(key, []).append(col)
        variables.setdefault(("refining", stage), set()).add(root)

    feature_parts: list[pd.DataFrame] = []
    feature_map_rows: list[dict[str, Any]] = []
    base_feature_names_by_key: dict[tuple[str, str, str, str], list[str]] = {}
    for (source, stage, lag, kind, lidx), cols in sorted(groups.items()):
        prefix = f"stage__{source}__{stage}__{lag}__{kind}_{lidx}"
        agg_frame = aggregate_group(frame, cols, prefix, aggs)
        feature_parts.append(agg_frame)
        for feat in agg_frame.columns:
            feature_map_rows.append(
                {
                    "feature": feat,
                    "source": source,
                    "stage": stage,
                    "lag": lag,
                    "kind": kind,
                    "lag_index": lidx,
                    "aggregation": feat.rsplit("__", 1)[-1],
                    "input_column_count": int(len(cols)),
                    "original_variables": ";".join(f"{name}_*" for name in sorted(variables.get((source, stage), []))),
                }
            )
        base_feature_names_by_key.setdefault((source, stage, lag, kind), []).extend(agg_frame.columns)

    if feature_parts:
        out = pd.concat([out.reset_index(drop=True), *[part.reset_index(drop=True) for part in feature_parts]], axis=1)

    change_rows: list[dict[str, Any]] = []
    if include_change:
        change_parts: dict[str, pd.Series] = {}
        feature_map = pd.DataFrame(feature_map_rows)
        for (source, stage, lag, kind), feats in sorted(base_feature_names_by_key.items()):
            for agg in aggs:
                by_lag_index: dict[int, str] = {}
                for feat in feats:
                    if not feat.endswith(f"__{agg}"):
                        continue
                    m = re.search(r"_L(\d+)__", feat)
                    if m:
                        by_lag_index[int(m.group(1))] = feat
                if 0 in by_lag_index and 5 in by_lag_index:
                    name = f"stagechg__{source}__{stage}__{lag}__{kind}__{agg}__L0_minus_L5"
                    change_parts[name] = out[by_lag_index[0]] - out[by_lag_index[5]]
                    vars_text = feature_map.loc[feature_map["feature"] == by_lag_index[0], "original_variables"]
                    change_rows.append(
                        {
                            "feature": name,
                            "source": source,
                            "stage": stage,
                            "lag": lag,
                            "kind": kind,
                            "lag_index": "L0-L5",
                            "aggregation": agg,
                            "input_column_count": None,
                            "original_variables": vars_text.iloc[0] if not vars_text.empty else "",
                        }
                    )
                near = [idx for idx in [0, 1] if idx in by_lag_index]
                far = [idx for idx in [4, 5] if idx in by_lag_index]
                if near and far:
                    name = f"stagechg__{source}__{stage}__{lag}__{kind}__{agg}__near_minus_far"
                    change_parts[name] = out[[by_lag_index[idx] for idx in near]].mean(axis=1) - out[
                        [by_lag_index[idx] for idx in far]
                    ].mean(axis=1)
                    vars_text = feature_map.loc[feature_map["feature"] == by_lag_index[near[0]], "original_variables"]
                    change_rows.append(
                        {
                            "feature": name,
                            "source": source,
                            "stage": stage,
                            "lag": lag,
                            "kind": kind,
                            "lag_index": "near-far",
                            "aggregation": agg,
                            "input_column_count": None,
                            "original_variables": vars_text.iloc[0] if not vars_text.empty else "",
                        }
                    )
        if change_parts:
            out = pd.concat([out, pd.DataFrame(change_parts)], axis=1)
            feature_map_rows.extend(change_rows)

    variable_rows = []
    for (source, stage), names in sorted(variables.items()):
        variable_rows.append(
            {
                "source": source,
                "stage": stage,
                "original_variable_count": int(len(names)),
                "original_variables": ";".join(f"{name}_*" for name in sorted(names)),
            }
        )
    return out, pd.DataFrame(feature_map_rows), pd.DataFrame(variable_rows)


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


def fit_regressor(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> dict[str, Any]:
    del valid
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
    parser = argparse.ArgumentParser(description="Evaluate flow-chart stage-level features.")
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--run-tag", type=str, default="exp022_stage_level_features")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include-change", action="store_true")
    parser.add_argument("--aggs", nargs="+", default=["mean", "std", "min", "max"])
    parser.add_argument("--reduced-max-features", type=int, default=300)
    parser.add_argument("--reduced-candidate-pool", type=int, default=900)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    args = parser.parse_args()

    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    raw = pd.read_parquet(args.feature_table)
    stage_frame, feature_map, variable_map = build_stage_features(raw, include_change=bool(args.include_change), aggs=args.aggs)
    stage_frame_path = run_dir / "stage_feature_table.parquet"
    stage_frame.to_parquet(stage_frame_path, index=False, compression="zstd")
    feature_map.to_csv(run_dir / "stage_feature_map.csv", index=False, encoding="utf-8-sig")
    variable_map.to_csv(run_dir / "stage_variable_map.csv", index=False, encoding="utf-8-sig")

    features = [col for col in stage_frame.columns if col.startswith(("stage__", "stagechg__"))]
    train = stage_frame[stage_frame["split"] == "train"].copy()
    valid = stage_frame[stage_frame["split"] == "valid"].copy()
    test = stage_frame[stage_frame["split"] == "test"].copy()
    screened, screen_report = screen_features(train, features, max_missing_rate=0.95, min_variance=1.0e-12)

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
                map_lookup = feature_map.set_index("feature").to_dict(orient="index")
                for rank, feature in enumerate(selected, start=1):
                    mapped = map_lookup.get(feature, {})
                    selected_rows.append(
                        {
                            "mode": mode,
                            "target": target,
                            "rank": rank,
                            "feature": feature,
                            **mapped,
                        }
                    )
            if not selected:
                continue
            if target == "is_out_spec_obs":
                metrics = fit_classifier(train, valid, test, selected)
                problem = "binary"
            else:
                metrics = fit_regressor(train, valid, test, selected, target)
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
    results.to_csv(run_dir / "stage_model_results.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(run_dir / "selected_stage_features.csv", index=False, encoding="utf-8-sig")

    input_dimension = {
        "source_feature_table": str(args.feature_table),
        "source_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "stage_feature_table": str(stage_frame_path),
        "stage_shape": [int(stage_frame.shape[0]), int(stage_frame.shape[1])],
        "stage_candidate_feature_count": int(len(features)),
        "screened_feature_count": int(len(screened)),
        "include_change_features": bool(args.include_change),
        "aggregations": args.aggs,
        "split_counts": stage_frame["split"].value_counts().to_dict(),
    }
    pd.DataFrame([input_dimension]).to_csv(run_dir / "input_dimension_report.csv", index=False, encoding="utf-8-sig")
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "input_dimension": input_dimension,
        "screen_report": screen_report,
        "results": json_ready(results.to_dict(orient="records")),
        "stage_variable_map_path": str(run_dir / "stage_variable_map.csv"),
        "selected_stage_features_path": str(run_dir / "selected_stage_features.csv"),
    }
    (run_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
