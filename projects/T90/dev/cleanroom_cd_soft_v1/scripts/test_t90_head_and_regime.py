from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FEATURE_TABLE = (
    PROJECT_DIR
    / "outputs"
    / "20260415_165303_exp019_two_stage_stage_candidate_lags_ref10_limsctx"
    / "two_stage_feature_table.parquet"
)
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs"


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


def regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    y = y_true.to_numpy(dtype=float)
    p = np.asarray(pred, dtype=float)
    corr = spearmanr(y, p, nan_policy="omit").correlation if len(y) >= 3 else math.nan
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "spearman": float(corr) if corr is not None and not math.isnan(float(corr)) else math.nan,
    }


def screen_features(
    train: pd.DataFrame,
    features: list[str],
    *,
    max_missing_rate: float,
    min_variance: float,
) -> tuple[list[str], dict[str, Any]]:
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
    if not candidates:
        return [], {"selected_feature_count": 0, "top_ranked_features": []}
    corr = x[candidates].corr().abs()
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
        "top_ranked_features": candidates[:30],
    }


def fit_t90_head(
    frame: pd.DataFrame,
    features: list[str],
    *,
    mode: str,
    use_sample_weight: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["split"] == "train"].copy()
    valid = frame[frame["split"] == "valid"].copy()
    test = frame[frame["split"] == "test"].copy()

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    sample_weight = train["sample_weight"].to_numpy(dtype=float) if use_sample_weight and "sample_weight" in train else None
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
    model.fit(x_train, train["t90"].to_numpy(dtype=float), sample_weight=sample_weight)

    rows: list[dict[str, Any]] = []
    scored_parts: list[pd.DataFrame] = []
    for split_name, split_frame in [("train", train), ("valid", valid), ("test", test)]:
        pred = model.predict(imputer.transform(split_frame[features]))
        metrics = regression_metrics(split_frame["t90"], pred)
        rows.append(
            {
                "mode": mode,
                "use_sample_weight": bool(use_sample_weight),
                "split": split_name,
                "samples": int(len(split_frame)),
                "feature_count": int(len(features)),
                **metrics,
            }
        )
        scored = split_frame[["sample_id", "sample_time", "split", "t90", "cd_mean", "p_pass_soft", "is_out_spec_obs"]].copy()
        scored[f"{mode}_{'weighted' if use_sample_weight else 'unweighted'}_t90_pred"] = pred
        scored_parts.append(scored)
    return pd.DataFrame(rows), pd.concat(scored_parts, ignore_index=True)


def overfit_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (mode, weighted), group in metrics.groupby(["mode", "use_sample_weight"]):
        by_split = group.set_index("split")
        train_rmse = float(by_split.loc["train", "rmse"])
        valid_rmse = float(by_split.loc["valid", "rmse"])
        test_rmse = float(by_split.loc["test", "rmse"])
        train_s = float(by_split.loc["train", "spearman"])
        test_s = float(by_split.loc["test", "spearman"])
        rmse_ratio = test_rmse / max(train_rmse, 1.0e-12)
        spearman_drop = train_s - test_s
        severe = bool(rmse_ratio >= 1.75 or (train_s >= 0.80 and test_s <= 0.35) or spearman_drop >= 0.55)
        rows.append(
            {
                "mode": mode,
                "use_sample_weight": bool(weighted),
                "train_rmse": train_rmse,
                "valid_rmse": valid_rmse,
                "test_rmse": test_rmse,
                "test_to_train_rmse_ratio": float(rmse_ratio),
                "train_spearman": train_s,
                "test_spearman": test_s,
                "spearman_drop": float(spearman_drop),
                "severe_overfit": severe,
            }
        )
    return pd.DataFrame(rows)


def cluster_regimes(
    frame: pd.DataFrame,
    features: list[str],
    *,
    group_name: str,
    output_dir: Path,
    k_min: int = 2,
    k_max: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    use = frame[frame["split"].isin(["train", "valid", "test"])].copy().reset_index(drop=True)
    screened, screen_report = screen_features(
        use,
        features,
        max_missing_rate=0.95,
        min_variance=1.0e-12,
    )
    if not screened:
        empty = pd.DataFrame()
        return empty, empty, empty
    x = use[screened].apply(pd.to_numeric, errors="coerce")
    x = SimpleImputer(strategy="median").fit_transform(x)
    x = StandardScaler().fit_transform(x)
    n_comp = min(20, x.shape[0] - 1, x.shape[1])
    z = PCA(n_components=n_comp, random_state=20260416).fit_transform(x)

    candidates: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in range(k_min, min(k_max, len(use) - 1) + 1):
        model = KMeans(n_clusters=k, n_init=30, random_state=20260416)
        labels = model.fit_predict(z)
        counts = pd.Series(labels).value_counts()
        min_cluster_fraction = float(counts.min() / len(labels))
        sil = float(silhouette_score(z, labels)) if len(counts) > 1 else math.nan
        candidates.append(
            {
                "group": group_name,
                "k": int(k),
                "silhouette": sil,
                "min_cluster_fraction": min_cluster_fraction,
                "feature_count_input": int(len(features)),
                "feature_count_screened": int(len(screened)),
                **{f"screen_{key}": value for key, value in screen_report.items()},
            }
        )
        labels_by_k[k] = labels

    candidates_df = pd.DataFrame(candidates)
    viable = candidates_df[candidates_df["min_cluster_fraction"] >= 0.05]
    best_row = viable.sort_values("silhouette", ascending=False).head(1)
    if best_row.empty:
        best_row = candidates_df.sort_values("silhouette", ascending=False).head(1)
    best_k = int(best_row.iloc[0]["k"])
    labels = labels_by_k[best_k]

    assignments = use[["sample_id", "sample_time", "split", "t90", "cd_mean", "p_pass_soft", "is_out_spec_obs"]].copy()
    assignments[f"{group_name}_regime"] = labels

    summary_rows: list[dict[str, Any]] = []
    for regime, part in assignments.groupby(f"{group_name}_regime"):
        summary_rows.append(
            {
                "group": group_name,
                "best_k": best_k,
                "regime": int(regime),
                "samples": int(len(part)),
                "sample_fraction": float(len(part) / len(assignments)),
                "t90_mean": float(part["t90"].mean()),
                "t90_std": float(part["t90"].std(ddof=0)),
                "cd_mean_mean": float(part["cd_mean"].mean()),
                "p_pass_soft_mean": float(part["p_pass_soft"].mean()),
                "out_spec_rate": float(part["is_out_spec_obs"].mean()),
                "time_min": pd.to_datetime(part["sample_time"]).min().isoformat(),
                "time_max": pd.to_datetime(part["sample_time"]).max().isoformat(),
            }
        )
    summary = pd.DataFrame(summary_rows)

    pd.DataFrame(z[:, : min(5, z.shape[1])], columns=[f"pc{i+1}" for i in range(min(5, z.shape[1]))]).assign(
        sample_id=assignments["sample_id"].to_numpy(),
        **{f"{group_name}_regime": labels},
    ).to_csv(output_dir / f"{group_name}_regime_pca_scores.csv", index=False, encoding="utf-8-sig")
    return candidates_df, summary, assignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a direct T90 regression head and sample-level process regimes.")
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--run-tag", type=str, default="exp020_t90_head_regime_check")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-missing-rate", type=float, default=0.95)
    parser.add_argument("--min-variance", type=float, default=1.0e-12)
    parser.add_argument("--reduced-max-features", type=int, default=300)
    parser.add_argument("--reduced-candidate-pool", type=int, default=900)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    args = parser.parse_args()

    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    frame = pd.read_parquet(args.feature_table)
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], errors="coerce")
    feature_cols = [
        col
        for col in frame.columns
        if col.startswith(("hal__", "hal_lag", "ref__", "lims_ctx__")) and not col.endswith("__count")
    ]
    train = frame[frame["split"] == "train"].copy()
    screened, screen_report = screen_features(
        train,
        feature_cols,
        max_missing_rate=args.max_missing_rate,
        min_variance=args.min_variance,
    )
    reduced, reduced_report = select_reduced_features(
        train,
        screened,
        "t90",
        max_features=args.reduced_max_features,
        candidate_pool_size=args.reduced_candidate_pool,
        corr_threshold=args.corr_threshold,
    )

    metric_parts: list[pd.DataFrame] = []
    scored_parts: list[pd.DataFrame] = []
    for mode, features in [("full", screened), ("reduced", reduced)]:
        for use_weight in [False, True]:
            metrics, scored = fit_t90_head(frame, features, mode=mode, use_sample_weight=use_weight)
            metric_parts.append(metrics)
            scored_parts.append(scored)
    metrics_df = pd.concat(metric_parts, ignore_index=True)
    scored_df = scored_parts[0]
    for part in scored_parts[1:]:
        pred_cols = [col for col in part.columns if col.endswith("_t90_pred")]
        scored_df = scored_df.merge(part[["sample_id", *pred_cols]], on="sample_id", how="left")

    overfit_df = overfit_summary(metrics_df)

    hal_features = [col for col in feature_cols if col.startswith(("hal__", "hal_lag"))]
    ref_features = [col for col in feature_cols if col.startswith("ref__")]
    regime_candidates: list[pd.DataFrame] = []
    regime_summaries: list[pd.DataFrame] = []
    assignments: list[pd.DataFrame] = []
    for group_name, group_features in [("halogen", hal_features), ("refining", ref_features)]:
        candidates, summary, assignment = cluster_regimes(frame, group_features, group_name=group_name, output_dir=run_dir)
        if not candidates.empty:
            regime_candidates.append(candidates)
            regime_summaries.append(summary)
            assignments.append(assignment)

    regime_candidates_df = pd.concat(regime_candidates, ignore_index=True) if regime_candidates else pd.DataFrame()
    regime_summary_df = pd.concat(regime_summaries, ignore_index=True) if regime_summaries else pd.DataFrame()
    regime_assignment = assignments[0]
    for assignment in assignments[1:]:
        regime_cols = [col for col in assignment.columns if col.endswith("_regime")]
        regime_assignment = regime_assignment.merge(assignment[["sample_id", *regime_cols]], on="sample_id", how="left")
    if "halogen_regime" in regime_assignment and "refining_regime" in regime_assignment:
        regime_assignment["combined_regime"] = (
            regime_assignment["halogen_regime"].astype(str) + "_" + regime_assignment["refining_regime"].astype(str)
        )
        combo = (
            regime_assignment.groupby("combined_regime")
            .agg(
                samples=("sample_id", "size"),
                t90_mean=("t90", "mean"),
                t90_std=("t90", lambda s: float(s.std(ddof=0))),
                p_pass_soft_mean=("p_pass_soft", "mean"),
                out_spec_rate=("is_out_spec_obs", "mean"),
            )
            .reset_index()
        )
    else:
        combo = pd.DataFrame()

    metrics_df.to_csv(run_dir / "t90_head_metrics.csv", index=False, encoding="utf-8-sig")
    overfit_df.to_csv(run_dir / "t90_head_overfit_summary.csv", index=False, encoding="utf-8-sig")
    scored_df.to_csv(run_dir / "t90_head_scored_rows.csv", index=False, encoding="utf-8-sig")
    regime_candidates_df.to_csv(run_dir / "regime_k_selection.csv", index=False, encoding="utf-8-sig")
    regime_summary_df.to_csv(run_dir / "regime_summary.csv", index=False, encoding="utf-8-sig")
    regime_assignment.to_csv(run_dir / "regime_assignments.csv", index=False, encoding="utf-8-sig")
    combo.to_csv(run_dir / "combined_regime_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "feature_table": str(args.feature_table),
        "samples": int(len(frame)),
        "split_counts": frame["split"].value_counts(dropna=False).to_dict(),
        "feature_count_raw": int(len(feature_cols)),
        "feature_count_screened": int(len(screened)),
        "feature_count_reduced": int(len(reduced)),
        "screen_report": screen_report,
        "reduced_report": reduced_report,
        "t90_head_metrics_path": str(run_dir / "t90_head_metrics.csv"),
        "t90_head_overfit_summary_path": str(run_dir / "t90_head_overfit_summary.csv"),
        "regime_k_selection_path": str(run_dir / "regime_k_selection.csv"),
        "regime_summary_path": str(run_dir / "regime_summary.csv"),
        "combined_regime_summary_path": str(run_dir / "combined_regime_summary.csv"),
        "any_severe_overfit": bool(overfit_df["severe_overfit"].any()),
    }
    (run_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
