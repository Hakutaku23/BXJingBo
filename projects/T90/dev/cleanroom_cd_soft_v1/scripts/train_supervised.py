from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cleanroom_cd_soft_v1.config import load_config, resolve_path


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "base.yaml"


def latest_prepared_run(output_root: Path) -> Path:
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "feature_table.csv").exists() and path.name != "_pycache_check"
    ]
    if not candidates:
        raise FileNotFoundError(f"No prepared run with feature_table.csv found under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


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
    return value


def feature_columns(
    frame: pd.DataFrame,
    include_lims_context: bool,
    window_minutes: int | None = None,
    lag_minutes: int | None = None,
) -> list[str]:
    dcs_features = [col for col in frame.columns if col.startswith("w") and "__" in col]
    if window_minutes is not None:
        dcs_features = [col for col in dcs_features if col.startswith(f"w{int(window_minutes)}_")]
    if lag_minutes is not None:
        dcs_features = [col for col in dcs_features if f"_lag{int(lag_minutes)}__" in col]
    context_features = [col for col in frame.columns if col.startswith("lims_ctx__")]
    features = dcs_features + (context_features if include_lims_context else [])
    blocked_suffixes = {"__count"}
    return [col for col in features if not any(col.endswith(suffix) for suffix in blocked_suffixes)]


def summarize_feature_groups(features: list[str]) -> dict[str, Any]:
    by_window: dict[str, int] = {}
    by_window_lag: dict[str, int] = {}
    by_stat: dict[str, int] = {}
    lims_context = 0
    for feature in features:
        if feature.startswith("lims_ctx__"):
            lims_context += 1
            continue
        parts = feature.split("__")
        if len(parts) < 3:
            continue
        prefix = parts[0]
        stat = parts[-1]
        window = prefix.split("_lag", 1)[0].lstrip("w")
        lag = prefix.split("_lag", 1)[1] if "_lag" in prefix else "unknown"
        by_window[window] = by_window.get(window, 0) + 1
        by_window_lag[f"w{window}_lag{lag}"] = by_window_lag.get(f"w{window}_lag{lag}", 0) + 1
        by_stat[stat] = by_stat.get(stat, 0) + 1
    return {
        "by_window_minutes": dict(sorted(by_window.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 999999)),
        "by_window_lag": dict(sorted(by_window_lag.items())),
        "by_stat": dict(sorted(by_stat.items())),
        "lims_context_count": lims_context,
    }


def select_model_features(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    selection_cfg = config["modeling"].get("feature_selection", {})
    if not bool(selection_cfg.get("enabled", True)):
        return features, {"enabled": False, "input_feature_count": int(len(features)), "selected_feature_count": int(len(features))}
    ranking_method = str(selection_cfg.get("ranking_method", "target_correlation"))
    max_missing = float(selection_cfg.get("max_missing_rate", 0.8))
    min_variance = float(selection_cfg.get("min_variance", 1.0e-12))
    max_features = int(selection_cfg.get("max_features", len(features)))
    candidate_pool_size = int(selection_cfg.get("candidate_pool_size", max_features))
    corr_threshold = float(selection_cfg.get("correlation_threshold", 0.995))
    always_prefixes = list(selection_cfg.get("always_keep_prefixes", []))
    always_keep = [col for col in features if any(col.startswith(prefix) for prefix in always_prefixes)]
    screened_features = [col for col in features if col not in set(always_keep)]
    numeric = train[screened_features].apply(pd.to_numeric, errors="coerce")
    missing_rate = numeric.isna().mean()
    kept = [col for col in screened_features if float(missing_rate[col]) <= max_missing]
    if not kept:
        selected = always_keep
        return selected, {
            "enabled": True,
            "input_feature_count": int(len(features)),
            "after_missing_filter_count": 0,
            "always_keep_count": int(len(always_keep)),
            "selected_feature_count": int(len(selected)),
            "max_missing_rate": max_missing,
            "min_variance": min_variance,
            "max_features": max_features,
            "candidate_pool_size": candidate_pool_size,
            "correlation_threshold": corr_threshold,
            "ranking_method": ranking_method,
        }
    variance = numeric[kept].var(skipna=True).fillna(0.0)
    non_constant = variance[variance > min_variance]
    if ranking_method == "target_correlation" and target in train.columns:
        y = pd.to_numeric(train[target], errors="coerce")
        usable = y.notna()
        target_numeric = numeric.loc[usable, non_constant.index].copy()
        target_numeric = target_numeric.fillna(target_numeric.median(numeric_only=True))
        y_centered = y.loc[usable].astype(float) - float(y.loc[usable].mean())
        x_centered = target_numeric - target_numeric.mean(axis=0)
        denom = np.sqrt((x_centered.pow(2).sum(axis=0) * float((y_centered**2).sum())).clip(lower=1.0e-30))
        scores = (x_centered.mul(y_centered, axis=0).sum(axis=0).abs() / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        candidate_cols = scores.sort_values(ascending=False).head(candidate_pool_size).index.tolist()
        ranking_score_name = "abs_train_target_corr"
    else:
        scores = non_constant
        candidate_cols = non_constant.sort_values(ascending=False).head(candidate_pool_size).index.tolist()
        ranking_score_name = "train_variance"
    selected_screened: list[str] = []
    if candidate_cols:
        candidate_frame = numeric[candidate_cols].copy()
        candidate_frame = candidate_frame.fillna(candidate_frame.median(numeric_only=True))
        corr = candidate_frame.corr().abs()
        for col in candidate_cols:
            if len(selected_screened) >= max_features:
                break
            if not selected_screened:
                selected_screened.append(col)
                continue
            max_corr = corr.loc[col, selected_screened].max()
            if pd.isna(max_corr) or float(max_corr) < corr_threshold:
                selected_screened.append(col)
    selected = always_keep + [col for col in selected_screened if col not in set(always_keep)]
    return selected, {
        "enabled": True,
        "input_feature_count": int(len(features)),
        "after_missing_filter_count": int(len(kept)),
        "non_constant_count": int(len(non_constant)),
        "candidate_pool_count": int(len(candidate_cols)),
        "always_keep_count": int(len(always_keep)),
        "correlation_pruned_count": int(len(candidate_cols) - len(selected_screened)),
        "selected_feature_count": int(len(selected)),
        "max_missing_rate": max_missing,
        "min_variance": min_variance,
        "max_features": max_features,
        "candidate_pool_size": candidate_pool_size,
        "correlation_threshold": corr_threshold,
        "ranking_method": ranking_method,
        "ranking_score_name": ranking_score_name,
        "top_ranked_features": candidate_cols[:20],
        "selected_feature_groups": summarize_feature_groups(selected),
    }


def regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    y = y_true.to_numpy(dtype=float)
    p = np.asarray(pred, dtype=float)
    corr = spearmanr(y, p, nan_policy="omit").correlation if len(y) >= 3 else math.nan
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "spearman": float(corr) if corr is not None and not math.isnan(float(corr)) else math.nan,
        "soft_brier_as_mse": float(np.mean((p - y) ** 2)),
    }


def binary_metrics(y_true: pd.Series, score: np.ndarray) -> dict[str, float]:
    y = y_true.to_numpy(dtype=int)
    s = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    result = {
        "average_precision": float(average_precision_score(y, s)),
        "brier": float(brier_score_loss(y, s)),
    }
    result["roc_auc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else math.nan
    return result


def fit_simple_baseline(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    problem_type: str,
) -> tuple[np.ndarray, dict[str, float]]:
    if problem_type == "binary":
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=1500, solver="lbfgs")),
            ]
        )
        fit_kwargs = {}
        if "sample_weight" in train.columns:
            fit_kwargs["model__sample_weight"] = train["sample_weight"].to_numpy(dtype=float)
        model.fit(train[features], train[target].astype(int), **fit_kwargs)
        pred = model.predict_proba(test[features])[:, 1]
        return pred, binary_metrics(test[target], pred)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    fit_kwargs = {}
    if "sample_weight" in train.columns:
        fit_kwargs["model__sample_weight"] = train["sample_weight"].to_numpy(dtype=float)
    model.fit(train[features], train[target].astype(float), **fit_kwargs)
    pred = np.clip(model.predict(test[features]), 0.0, 1.0)
    return pred, regression_metrics(test[target], pred)


def fit_autogluon(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    problem_type: str,
    model_dir: Path,
    ag_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float], str]:
    from autogluon.tabular import TabularPredictor

    eval_metric = "average_precision" if problem_type == "binary" else "root_mean_squared_error"
    predictor = TabularPredictor(
        label=target,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=str(model_dir),
        verbosity=int(ag_config["verbosity"]),
    )
    train_df = train[features + [target]].copy()
    valid_df = valid[features + [target]].copy()
    test_x = test[features].copy()
    predictor.fit(
        train_data=train_df,
        tuning_data=valid_df,
        presets=ag_config["presets"],
        time_limit=int(ag_config["time_limit_seconds"]),
        hyperparameters=ag_config.get("hyperparameters"),
        ag_args_fit={"ag.max_memory_usage_ratio": float(ag_config.get("max_memory_usage_ratio", 1.0))},
    )
    if problem_type == "binary":
        proba = predictor.predict_proba(test_x)
        positive_col = 1 if 1 in proba.columns else proba.columns[-1]
        pred = proba[positive_col].to_numpy(dtype=float)
        metrics = binary_metrics(test[target], pred)
    else:
        pred = np.clip(predictor.predict(test_x).to_numpy(dtype=float), 0.0, 1.0)
        metrics = regression_metrics(test[target], pred)
    return pred, metrics, str(predictor.model_best)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Exp-002 快速监督建模验证",
        "",
        f"- prepared_run_dir: `{summary['prepared_run_dir']}`",
        f"- model_run_dir: `{summary['model_run_dir']}`",
        f"- include_lims_context_features: `{summary['include_lims_context_features']}`",
        f"- window_filter: `{summary['window_filter']}`",
        f"- raw_feature_count: `{summary['raw_feature_count']}`",
        "",
        "## 说明",
        "",
        "- 本轮是快速信号验证，不使用 CQDI/RADI 作为门槛。",
        "- 默认只使用 DCS 因果窗口特征；LIMS 其他检测项可通过参数打开，但不作为主实验默认输入。",
        "",
        "## 结果",
        "",
        "```json",
        json.dumps(summary["targets"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick supervised baselines on prepared cleanroom features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared-run-dir", type=Path, default=None)
    parser.add_argument("--run-tag", type=str, default="exp002_quick")
    parser.add_argument("--include-lims-context", action="store_true")
    parser.add_argument("--window-minutes", type=int, default=None)
    parser.add_argument("--lag-minutes", type=int, default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--no-autogluon", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    output_root = resolve_path(config_path, config["paths"]["output_root"])
    if output_root is None:
        raise ValueError("paths.output_root is required.")
    prepared_run = args.prepared_run_dir.resolve() if args.prepared_run_dir else None
    if prepared_run is None:
        configured = config.get("modeling", {}).get("prepared_run_dir")
        prepared_run = resolve_path(config_path, configured) if configured else latest_prepared_run(output_root)
    if prepared_run is None:
        raise ValueError("prepared run directory could not be resolved.")

    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    model_run_dir = output_root / run_id
    model_run_dir.mkdir(parents=True, exist_ok=False)

    frame = pd.read_csv(prepared_run / "feature_table.csv")
    frame = frame[frame["split"].isin(["train", "valid", "test"])].copy()
    include_context = bool(args.include_lims_context or config["modeling"].get("include_lims_context_features", False))
    configured_filter = config["modeling"].get("window_filter", {})
    window_minutes = args.window_minutes if args.window_minutes is not None else configured_filter.get("window_minutes")
    lag_minutes = args.lag_minutes if args.lag_minutes is not None else configured_filter.get("lag_minutes")
    window_minutes = int(window_minutes) if window_minutes is not None else None
    lag_minutes = int(lag_minutes) if lag_minutes is not None else None
    targets = args.targets or list(config["modeling"]["targets"])
    train = frame[frame["split"] == "train"].copy()
    valid = frame[frame["split"] == "valid"].copy()
    test = frame[frame["split"] == "test"].copy()
    raw_features = feature_columns(frame, include_context, window_minutes=window_minutes, lag_minutes=lag_minutes)

    rows: list[dict[str, Any]] = []
    scored = test[["sample_id", "sample_time", "split", "t90", "cd_mean", "p_pass_soft", "is_out_spec_obs", "sample_weight"]].copy()
    target_summary: dict[str, Any] = {}

    for target in targets:
        if target not in frame.columns:
            continue
        problem_type = "binary" if target == "is_out_spec_obs" else "regression"
        train_task = train.dropna(subset=[target])
        valid_task = valid.dropna(subset=[target])
        test_task = test.dropna(subset=[target])
        if train_task.empty or valid_task.empty or test_task.empty:
            continue
        features, feature_selection_audit = select_model_features(train_task, raw_features, target, config)
        if not features:
            raise ValueError(f"No features remain after feature selection for target {target}.")

        baseline_pred, baseline_metrics = fit_simple_baseline(train_task, valid_task, test_task, features, target, problem_type)
        scored.loc[test_task.index, f"simple_{target}_pred"] = baseline_pred
        target_summary.setdefault(target, {})["feature_selection"] = feature_selection_audit
        rows.append(
            {
                "target": target,
                "framework": "simple_baseline",
                "problem_type": problem_type,
                "samples_train": int(len(train_task)),
                "samples_valid": int(len(valid_task)),
                "samples_test": int(len(test_task)),
                "feature_count": int(len(features)),
                "feature_selection": json.dumps(json_ready(feature_selection_audit), ensure_ascii=False),
                **baseline_metrics,
            }
        )
        target_summary.setdefault(target, {})["simple_baseline"] = baseline_metrics

        ag_enabled = bool(config["modeling"]["autogluon"]["enabled"]) and not args.no_autogluon
        if ag_enabled:
            model_dir = model_run_dir / "ag_models" / target
            try:
                ag_pred, ag_metrics, best_model = fit_autogluon(
                    train_task,
                    valid_task,
                    test_task,
                    features,
                    target,
                    problem_type,
                    model_dir,
                    config["modeling"]["autogluon"],
                )
                scored.loc[test_task.index, f"autogluon_{target}_pred"] = ag_pred
                rows.append(
                    {
                        "target": target,
                        "framework": "autogluon",
                        "problem_type": problem_type,
                        "samples_train": int(len(train_task)),
                        "samples_valid": int(len(valid_task)),
                        "samples_test": int(len(test_task)),
                        "feature_count": int(len(features)),
                        "feature_selection": json.dumps(json_ready(feature_selection_audit), ensure_ascii=False),
                        "best_model": best_model,
                        **ag_metrics,
                    }
                )
                target_summary.setdefault(target, {})["autogluon"] = {"best_model": best_model, **ag_metrics}
            except Exception as exc:
                failure = {"error": f"{type(exc).__name__}: {exc}"}
                rows.append(
                    {
                        "target": target,
                        "framework": "autogluon",
                        "problem_type": problem_type,
                        "samples_train": int(len(train_task)),
                        "samples_valid": int(len(valid_task)),
                        "samples_test": int(len(test_task)),
                        "feature_count": int(len(features)),
                        "feature_selection": json.dumps(json_ready(feature_selection_audit), ensure_ascii=False),
                        **failure,
                    }
                )
                target_summary.setdefault(target, {})["autogluon_failed"] = failure

    results = pd.DataFrame(rows)
    results.to_csv(model_run_dir / "quick_model_results.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(model_run_dir / "quick_model_scored_test_rows.csv", index=False, encoding="utf-8-sig")
    summary = {
        "run_id": run_id,
        "prepared_run_dir": str(prepared_run),
        "model_run_dir": str(model_run_dir),
        "include_lims_context_features": include_context,
        "window_filter": {
            "window_minutes": window_minutes,
            "lag_minutes": lag_minutes,
        },
        "raw_feature_count": int(len(raw_features)),
        "split_counts": frame["split"].value_counts().to_dict(),
        "targets": target_summary,
    }
    (model_run_dir / "quick_model_summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(model_run_dir / "exp002_quick_model_report.md", json_ready(summary))
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
