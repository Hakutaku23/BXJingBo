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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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
        "max_features": int(max_features),
        "correlation_threshold": float(corr_threshold),
        "top_ranked_features": candidates[:50],
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
        return {
            "feature": feature,
            "source": "refining",
            "stage": f"ref_{root.split('_', 1)[0]}",
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


def probability_metrics(prob_true_soft: pd.Series, observed: pd.Series, pred: np.ndarray) -> dict[str, float]:
    soft = prob_true_soft.to_numpy(dtype=float)
    y = observed.to_numpy(dtype=int)
    p = np.clip(np.asarray(pred, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    corr = spearmanr(soft, p, nan_policy="omit").correlation if len(soft) >= 3 else math.nan
    out = {
        "soft_mae": float(mean_absolute_error(soft, p)),
        "soft_rmse": float(np.sqrt(mean_squared_error(soft, p))),
        "soft_spearman": float(corr) if corr is not None and not math.isnan(float(corr)) else math.nan,
        "observed_brier": float(brier_score_loss(y, p)),
        "observed_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "predicted_prob_mean": float(p.mean()),
        "observed_out_spec_rate": float(y.mean()),
        "soft_out_prob_mean": float(soft.mean()),
    }
    out["observed_average_precision"] = float(average_precision_score(y, p))
    out["observed_roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan
    return out


def calibration_bins(scored: pd.DataFrame, *, model_name: str, n_bins: int) -> pd.DataFrame:
    work = scored.copy()
    prob_col = f"{model_name}_prob_out_spec"
    work["bin"] = pd.qcut(work[prob_col].rank(method="first"), q=int(n_bins), labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for bin_id, sub in work.groupby("bin"):
        rows.append(
            {
                "model": model_name,
                "bin": int(bin_id),
                "samples": int(len(sub)),
                "pred_prob_mean": float(sub[prob_col].mean()),
                "observed_out_spec_rate": float(sub["is_out_spec_obs"].mean()),
                "soft_out_prob_mean": float(sub["p_out_soft"].mean()),
                "t90_mean": float(sub["t90"].mean()),
                "t90_min": float(sub["t90"].min()),
                "t90_max": float(sub["t90"].max()),
            }
        )
    return pd.DataFrame(rows)


def calibration_errors(calibration: pd.DataFrame) -> dict[str, float]:
    total = max(float(calibration["samples"].sum()), 1.0)
    obs_ece = ((calibration["samples"] / total) * (calibration["pred_prob_mean"] - calibration["observed_out_spec_rate"]).abs()).sum()
    soft_ece = ((calibration["samples"] / total) * (calibration["pred_prob_mean"] - calibration["soft_out_prob_mean"]).abs()).sum()
    return {
        "observed_ece": float(obs_ece),
        "soft_label_ece": float(soft_ece),
    }


def fit_probability_regressor(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    loss: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_valid = imputer.transform(valid[features])
    x_test = imputer.transform(test[features])
    y_train = train["p_out_soft"].to_numpy(dtype=float)
    sample_weight = train["sample_weight"].to_numpy(dtype=float) if "sample_weight" in train else None
    model = GradientBoostingRegressor(
        loss=loss,
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=10,
        alpha=0.9,
        random_state=20260416,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    valid_pred = np.clip(model.predict(x_valid), 0.0, 1.0)
    test_pred = np.clip(model.predict(x_test), 0.0, 1.0)
    metrics = probability_metrics(test["p_out_soft"], test["is_out_spec_obs"], test_pred)
    valid_metrics = probability_metrics(valid["p_out_soft"], valid["is_out_spec_obs"], valid_pred)
    metrics.update({f"valid_{k}": v for k, v in valid_metrics.items()})
    return valid_pred, test_pred, metrics


def calibrate_probability(
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
    valid_target: pd.Series,
) -> np.ndarray:
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.asarray(valid_pred, dtype=float), pd.to_numeric(valid_target, errors="coerce").to_numpy(dtype=float))
    return np.clip(calibrator.transform(np.asarray(test_pred, dtype=float)), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a probability-only T90 out-of-spec model from a two-stage feature table.")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--run-tag", type=str, default="exp026_probability_only")
    parser.add_argument("--loss", choices=["huber", "squared_error"], default="huber")
    parser.add_argument("--max-missing-rate", type=float, default=0.95)
    parser.add_argument("--min-variance", type=float, default=1.0e-12)
    parser.add_argument("--reduced-max-features", type=int, default=300)
    parser.add_argument("--reduced-candidate-pool", type=int, default=900)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args()

    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = args.source_run_dir.parent / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    feature_table_path = args.source_run_dir / "two_stage_feature_table.parquet"
    lag_policy_path = args.source_run_dir / "halogen_lag_policy.csv"
    feature_table = pd.read_parquet(feature_table_path)
    source_shape = [int(feature_table.shape[0]), int(feature_table.shape[1])]
    lag_policy = pd.read_csv(lag_policy_path)
    feature_table["p_out_soft"] = 1.0 - pd.to_numeric(feature_table["p_pass_soft"], errors="coerce").clip(0.0, 1.0)
    features = [
        col
        for col in feature_table.columns
        if col.startswith(("hal__", "hal_lag", "ref__", "lims_ctx__")) and not col.endswith("__count")
    ]
    train = feature_table[feature_table["split"] == "train"].copy()
    valid = feature_table[feature_table["split"] == "valid"].copy()
    test = feature_table[feature_table["split"] == "test"].copy()
    screened, screen_report = screen_features(
        train,
        features,
        max_missing_rate=args.max_missing_rate,
        min_variance=args.min_variance,
    )

    scored = test[
        [
            "sample_id",
            "sample_time",
            "t90",
            "p_pass_soft",
            "p_out_soft",
            "is_out_spec_obs",
            "sample_weight",
        ]
    ].copy()
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []

    for mode in ["full", "reduced"]:
        if mode == "full":
            selected = screened
            selection_report = {"mode": mode, **screen_report}
        else:
            selected, reduced_report = select_reduced_features(
                train,
                screened,
                "p_out_soft",
                max_features=args.reduced_max_features,
                candidate_pool_size=args.reduced_candidate_pool,
                corr_threshold=args.corr_threshold,
            )
            selection_report = {"mode": mode, **screen_report, **reduced_report}
            for rank, feature in enumerate(selected, start=1):
                selected_rows.append(
                    {
                        "mode": mode,
                        "target": "p_out_soft",
                        "rank": rank,
                        **feature_meta(feature, lag_policy),
                    }
                )

        valid_pred, test_pred, metrics = fit_probability_regressor(train, valid, test, selected, loss=args.loss)
        model_variants = {
            f"{mode}_{args.loss}_p_out_soft_raw": (
                test_pred,
                metrics,
                f"GradientBoostingRegressor(loss={args.loss}) on soft out-spec probability",
            ),
            f"{mode}_{args.loss}_p_out_soft_iso_observed": (
                calibrate_probability(valid_pred, test_pred, valid["is_out_spec_obs"]),
                None,
                "raw soft-probability regressor + isotonic calibration on validation observed out-spec labels",
            ),
            f"{mode}_{args.loss}_p_out_soft_iso_soft": (
                calibrate_probability(valid_pred, test_pred, valid["p_out_soft"]),
                None,
                "raw soft-probability regressor + isotonic calibration on validation soft out-spec probability",
            ),
        }
        for model_name, (model_pred, existing_metrics, objective) in model_variants.items():
            model_metrics = existing_metrics or probability_metrics(test["p_out_soft"], test["is_out_spec_obs"], model_pred)
            scored[f"{model_name}_prob_out_spec"] = model_pred
            cal = calibration_bins(scored, model_name=model_name, n_bins=args.calibration_bins)
            calibration_frames.append(cal)
            model_metrics.update(calibration_errors(cal))
            rows.append(
                {
                    "model": model_name,
                    "mode": mode,
                    "target": "p_out_soft",
                    "training_objective": objective,
                    "feature_count": int(len(selected)),
                    "samples_train": int(len(train)),
                    "samples_valid": int(len(valid)),
                    "samples_test": int(len(test)),
                    "selection_report": json.dumps(json_ready(selection_report), ensure_ascii=False),
                    **model_metrics,
                }
            )

    results = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected_rows)
    calibration_df = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    variable_summary = (
        selected_df.groupby(["mode", "target", "source", "stage", "original_variable_wildcard"], as_index=False)
        .agg(selected_feature_count=("feature", "count"))
        .sort_values(["mode", "target", "selected_feature_count"], ascending=[True, True, False])
        if not selected_df.empty
        else pd.DataFrame()
    )

    results.to_csv(run_dir / "probability_model_results.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(run_dir / "probability_scored_test_rows.csv", index=False, encoding="utf-8-sig")
    calibration_df.to_csv(run_dir / "probability_calibration_bins.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(run_dir / "selected_probability_features.csv", index=False, encoding="utf-8-sig")
    variable_summary.to_csv(run_dir / "selected_probability_variable_summary.csv", index=False, encoding="utf-8-sig")

    input_dimension = {
        "source_run_dir": str(args.source_run_dir),
        "source_feature_table": str(feature_table_path),
        "source_feature_table_shape": source_shape,
        "modeling_table_shape_after_target_added": [int(feature_table.shape[0]), int(feature_table.shape[1])],
        **screen_report,
        "split_counts": feature_table["split"].value_counts(dropna=False).to_dict(),
        "target": "p_out_soft = 1 - p_pass_soft",
        "no_classifier_head": True,
        "loss": args.loss,
        "full_feature_count": int(len(screened)),
        "reduced_feature_count": int(
            results.loc[results["mode"] == "reduced", "feature_count"].iloc[0] if (results["mode"] == "reduced").any() else 0
        ),
    }
    pd.DataFrame([input_dimension]).to_csv(run_dir / "input_dimension_report.csv", index=False, encoding="utf-8-sig")
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "input_dimension": input_dimension,
        "results": json_ready(results.to_dict(orient="records")),
    }
    (run_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
