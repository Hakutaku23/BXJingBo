from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
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
        "top_ranked_features": candidates[:30],
    }


class TrainFittedRegimeAssigner:
    def __init__(self, group_name: str, features: list[str], n_clusters: int = 2):
        self.group_name = group_name
        self.features = features
        self.n_clusters = int(n_clusters)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.model: KMeans | None = None
        self.selected_features: list[str] = []
        self.report: dict[str, Any] = {}

    def fit(self, train: pd.DataFrame) -> "TrainFittedRegimeAssigner":
        selected, screen = screen_features(train, self.features, max_missing_rate=0.95, min_variance=1.0e-12)
        self.selected_features = selected
        x = self.imputer.fit_transform(train[selected].apply(pd.to_numeric, errors="coerce"))
        x = self.scaler.fit_transform(x)
        n_comp = min(20, x.shape[0] - 1, x.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=20260416)
        z = self.pca.fit_transform(x)
        self.model = KMeans(n_clusters=self.n_clusters, n_init=30, random_state=20260416)
        labels = self.model.fit_predict(z)
        counts = pd.Series(labels).value_counts().sort_index()
        self.report = {
            "group": self.group_name,
            "n_clusters": int(self.n_clusters),
            "screen": screen,
            "feature_count": int(len(selected)),
            "train_cluster_counts": {str(k): int(v) for k, v in counts.items()},
            "train_cluster_fraction": {str(k): float(v / len(labels)) for k, v in counts.items()},
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pca is None or self.model is None:
            raise RuntimeError("Regime assigner is not fitted.")
        x = self.imputer.transform(frame[self.selected_features].apply(pd.to_numeric, errors="coerce"))
        x = self.scaler.transform(x)
        z = self.pca.transform(x)
        return self.model.predict(z)


def fit_predict_classifier(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    model = GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=20260416,
    )
    fit_kwargs: dict[str, Any] = {}
    if "sample_weight" in train:
        fit_kwargs["sample_weight"] = train["sample_weight"].to_numpy(dtype=float)
    model.fit(x_train, train["is_out_spec_obs"].astype(int), **fit_kwargs)
    out_parts: list[pd.DataFrame] = []
    for split_name, part in [("train", train), ("valid", valid), ("test", test)]:
        prob = model.predict_proba(imputer.transform(part[features]))[:, 1]
        rows = part[["sample_id", "sample_time", "split", "t90", "is_out_spec_obs", "sample_weight", "halogen_regime", "refining_regime", "combined_regime"]].copy()
        rows[f"{name}_prob"] = prob
        out_parts.append(rows)
    report = {"model": name, "feature_count": int(len(features))}
    return pd.concat(out_parts, ignore_index=True), report


def fit_predict_per_regime(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    fallback_prob: pd.DataFrame,
    *,
    min_train_samples: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = ["sample_id", "sample_time", "split", "t90", "is_out_spec_obs", "sample_weight", "halogen_regime", "refining_regime", "combined_regime"]
    out = pd.concat([train[base_cols], valid[base_cols], test[base_cols]], ignore_index=True).copy()
    out = out.merge(fallback_prob[["sample_id", "global_reduced_prob"]], on="sample_id", how="left")
    out["per_combined_model_prob"] = out["global_reduced_prob"]
    rows: list[dict[str, Any]] = []
    for regime, train_part in train.groupby("combined_regime"):
        y = train_part["is_out_spec_obs"].astype(int)
        can_fit = int(len(train_part)) >= min_train_samples and y.nunique() == 2
        row = {
            "combined_regime": regime,
            "train_samples": int(len(train_part)),
            "train_positive": int(y.sum()),
            "fit_model": bool(can_fit),
            "fallback_reason": None,
        }
        if not can_fit:
            row["fallback_reason"] = "too_few_samples_or_single_class"
            rows.append(row)
            continue
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train_part[features])
        model = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=180,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.8,
            min_samples_leaf=8,
            random_state=20260416,
        )
        fit_kwargs: dict[str, Any] = {}
        if "sample_weight" in train_part:
            fit_kwargs["sample_weight"] = train_part["sample_weight"].to_numpy(dtype=float)
        model.fit(x_train, y, **fit_kwargs)
        for split_frame in [train, valid, test]:
            idx = out["sample_id"].isin(split_frame.loc[split_frame["combined_regime"] == regime, "sample_id"])
            if not idx.any():
                continue
            samples = out.loc[idx, "sample_id"]
            source = split_frame.set_index("sample_id").loc[samples, features]
            out.loc[idx, "per_combined_model_prob"] = model.predict_proba(imputer.transform(source))[:, 1]
        rows.append(row)
    return out, pd.DataFrame(rows)


def probability_metrics(y_true: pd.Series, prob: pd.Series) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    p = np.clip(prob.astype(float).to_numpy(), 0.0, 1.0)
    out = {
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan
    return out


def best_f1_threshold(y_true: pd.Series, prob: pd.Series) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), prob.astype(float))
    if thresholds.size == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1.0e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx])


def threshold_metrics(y_true: pd.Series, prob: pd.Series, threshold: float) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    pred = (prob.astype(float).to_numpy() >= float(threshold)).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alarm_rate": float(pred.mean()),
        "false_alarm_rate": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "miss_rate": float(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
    }


def evaluate_threshold_strategy(
    scored: pd.DataFrame,
    *,
    model_name: str,
    prob_col: str,
    strategy: str,
    group_col: str | None = None,
    min_valid_samples: int = 25,
    min_valid_positive: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    valid = scored[scored["split"] == "valid"].copy()
    test = scored[scored["split"] == "test"].copy()
    global_threshold = best_f1_threshold(valid["is_out_spec_obs"], valid[prob_col])
    threshold_rows: list[dict[str, Any]] = []
    if group_col is None:
        test_thresholds = pd.Series(global_threshold, index=test.index)
        threshold_rows.append({"group": "global", "threshold": global_threshold, "valid_samples": int(len(valid)), "valid_positive": int(valid["is_out_spec_obs"].sum()), "fallback": False})
    else:
        thresholds: dict[str, float] = {}
        for group, part in valid.groupby(group_col):
            valid_positive = int(part["is_out_spec_obs"].sum())
            valid_negative = int((part["is_out_spec_obs"] == 0).sum())
            fallback = len(part) < min_valid_samples or valid_positive < min_valid_positive or valid_negative < min_valid_positive
            threshold = global_threshold if fallback else best_f1_threshold(part["is_out_spec_obs"], part[prob_col])
            thresholds[str(group)] = float(threshold)
            threshold_rows.append(
                {
                    "group": str(group),
                    "threshold": float(threshold),
                    "valid_samples": int(len(part)),
                    "valid_positive": valid_positive,
                    "fallback": bool(fallback),
                }
            )
        test_thresholds = test[group_col].astype(str).map(thresholds).fillna(global_threshold)
    pred = (test[prob_col].astype(float).to_numpy() >= test_thresholds.astype(float).to_numpy()).astype(int)
    y = test["is_out_spec_obs"].astype(int).to_numpy()
    metric = {
        "model": model_name,
        "prob_col": prob_col,
        "threshold_strategy": strategy,
        "group_col": group_col or "global",
        "test_samples": int(len(test)),
        **probability_metrics(test["is_out_spec_obs"], test[prob_col]),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alarm_rate": float(pred.mean()),
        "false_alarm_rate": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "miss_rate": float(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "global_valid_threshold": float(global_threshold),
    }
    return metric, pd.DataFrame(threshold_rows).assign(model=model_name, strategy=strategy, group_col=group_col or "global")


def per_group_probability_report(scored: pd.DataFrame, prob_col: str, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test = scored[scored["split"] == "test"].copy()
    for group, part in test.groupby(group_col):
        row = {
            "group_col": group_col,
            "group": str(group),
            "samples": int(len(part)),
            "positive": int(part["is_out_spec_obs"].sum()),
            "out_spec_rate": float(part["is_out_spec_obs"].mean()),
            "prob_mean": float(part[prob_col].mean()),
        }
        if part["is_out_spec_obs"].nunique() > 1:
            row.update(probability_metrics(part["is_out_spec_obs"], part[prob_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate regime-aware alarm thresholds and regime-aware binary models.")
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--run-tag", type=str, default="exp021_regime_alarm_eval")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reduced-max-features", type=int, default=300)
    parser.add_argument("--reduced-candidate-pool", type=int, default=900)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    args = parser.parse_args()

    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    frame = pd.read_parquet(args.feature_table)
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], errors="coerce")
    frame = frame[frame["split"].isin(["train", "valid", "test"])].copy().reset_index(drop=True)
    train = frame[frame["split"] == "train"].copy()
    valid = frame[frame["split"] == "valid"].copy()
    test = frame[frame["split"] == "test"].copy()

    all_features = [
        col
        for col in frame.columns
        if col.startswith(("hal__", "hal_lag", "ref__", "lims_ctx__")) and not col.endswith("__count")
    ]
    screened, screen_report = screen_features(train, all_features, max_missing_rate=0.95, min_variance=1.0e-12)
    reduced, reduce_report = select_reduced_features(
        train,
        screened,
        "is_out_spec_obs",
        max_features=args.reduced_max_features,
        candidate_pool_size=args.reduced_candidate_pool,
        corr_threshold=args.corr_threshold,
    )

    hal_features = [col for col in all_features if col.startswith(("hal__", "hal_lag"))]
    ref_features = [col for col in all_features if col.startswith("ref__")]
    hal_assigner = TrainFittedRegimeAssigner("halogen", hal_features, n_clusters=2).fit(train)
    ref_assigner = TrainFittedRegimeAssigner("refining", ref_features, n_clusters=2).fit(train)
    for part in [train, valid, test]:
        part["halogen_regime"] = hal_assigner.predict(part).astype(int)
        part["refining_regime"] = ref_assigner.predict(part).astype(int)
        part["combined_regime"] = part["halogen_regime"].astype(str) + "_" + part["refining_regime"].astype(str)

    regime_one_hot_cols: list[str] = []
    category_values = {
        "halogen_regime": sorted(train["halogen_regime"].astype(str).unique()),
        "refining_regime": sorted(train["refining_regime"].astype(str).unique()),
        "combined_regime": sorted(train["combined_regime"].astype(str).unique()),
    }
    for part in [train, valid, test]:
        for col, values in category_values.items():
            for value in values:
                new_col = f"regime__{col}__{value}"
                part[new_col] = (part[col].astype(str) == value).astype(float)
                if new_col not in regime_one_hot_cols:
                    regime_one_hot_cols.append(new_col)

    global_scored, global_report = fit_predict_classifier(train, valid, test, reduced, name="global_reduced")
    regime_feature_scored, regime_feature_report = fit_predict_classifier(train, valid, test, reduced + regime_one_hot_cols, name="global_plus_regime")
    per_model_scored, per_model_report = fit_predict_per_regime(train, valid, test, reduced, global_scored, min_train_samples=120)

    scored = global_scored.merge(regime_feature_scored[["sample_id", "global_plus_regime_prob"]], on="sample_id", how="left")
    scored = scored.merge(per_model_scored[["sample_id", "per_combined_model_prob"]], on="sample_id", how="left")

    metric_rows: list[dict[str, Any]] = []
    threshold_frames: list[pd.DataFrame] = []
    eval_specs = [
        ("global_reduced", "global_reduced_prob"),
        ("global_plus_regime", "global_plus_regime_prob"),
        ("per_combined_model", "per_combined_model_prob"),
    ]
    strategies = [
        ("global_threshold", None),
        ("halogen_threshold", "halogen_regime"),
        ("refining_threshold", "refining_regime"),
        ("combined_threshold", "combined_regime"),
    ]
    for model_name, prob_col in eval_specs:
        for strategy_name, group_col in strategies:
            metric, thresholds = evaluate_threshold_strategy(
                scored,
                model_name=model_name,
                prob_col=prob_col,
                strategy=strategy_name,
                group_col=group_col,
            )
            metric_rows.append(metric)
            threshold_frames.append(thresholds)
    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.concat(threshold_frames, ignore_index=True)

    per_group_reports = []
    for prob_col in ["global_reduced_prob", "global_plus_regime_prob", "per_combined_model_prob"]:
        for group_col in ["halogen_regime", "refining_regime", "combined_regime"]:
            per_group_reports.append(per_group_probability_report(scored, prob_col, group_col).assign(prob_col=prob_col))
    per_group = pd.concat(per_group_reports, ignore_index=True)

    scored.to_csv(run_dir / "regime_alarm_scored_rows.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(run_dir / "regime_alarm_metrics.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(run_dir / "regime_alarm_thresholds.csv", index=False, encoding="utf-8-sig")
    per_group.to_csv(run_dir / "regime_alarm_per_group_probability.csv", index=False, encoding="utf-8-sig")
    per_model_report.to_csv(run_dir / "per_combined_model_fit_report.csv", index=False, encoding="utf-8-sig")

    regime_summary = []
    for split_name, part in [("train", train), ("valid", valid), ("test", test)]:
        for group_col in ["halogen_regime", "refining_regime", "combined_regime"]:
            for group, group_part in part.groupby(group_col):
                regime_summary.append(
                    {
                        "split": split_name,
                        "group_col": group_col,
                        "group": str(group),
                        "samples": int(len(group_part)),
                        "positive": int(group_part["is_out_spec_obs"].sum()),
                        "out_spec_rate": float(group_part["is_out_spec_obs"].mean()),
                        "t90_mean": float(group_part["t90"].mean()),
                    }
                )
    pd.DataFrame(regime_summary).to_csv(run_dir / "train_fitted_regime_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "feature_table": str(args.feature_table),
        "split_counts": frame["split"].value_counts().to_dict(),
        "feature_count_raw": int(len(all_features)),
        "feature_count_screened": int(len(screened)),
        "feature_count_reduced": int(len(reduced)),
        "screen_report": screen_report,
        "reduce_report": reduce_report,
        "halogen_regime_report": hal_assigner.report,
        "refining_regime_report": ref_assigner.report,
        "global_model_report": global_report,
        "regime_feature_model_report": regime_feature_report,
        "best_by_f1": json_ready(metrics.sort_values("f1", ascending=False).head(5).to_dict(orient="records")),
        "best_by_brier": json_ready(metrics.sort_values("brier", ascending=True).head(5).to_dict(orient="records")),
    }
    (run_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
