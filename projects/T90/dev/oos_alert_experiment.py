from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from oos_alert_module import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PH_FEATURE_COLUMNS,
    _fit_binary_ensemble,
    _predict_binary_ensemble,
    _predict_stage_aware_probability,
    _prepare_alert_frame,
    _select_dcs_feature_columns,
    _load_assets,
)


DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
RISK_RESULTS_PATH = DEFAULT_RESULTS_DIR / "risk_accuracy_experiment_results.csv"


def _safe_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    normalized = series.where(series.notna(), default)
    return normalized.infer_objects(copy=False).astype(bool)


def _threshold_scan(actual: pd.Series, score: pd.Series, thresholds: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        predicted = (score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(actual, predicted, zero_division=0)),
                "recall": float(recall_score(actual, predicted, zero_division=0)),
                "f1": float(f1_score(actual, predicted, zero_division=0)),
                "predicted_positive_ratio": float(predicted.mean()),
            }
        )
    return rows


def _summarize_classifier(actual: pd.Series, score: pd.Series) -> dict[str, object]:
    thresholds = [round(item, 2) for item in np.arange(0.05, 0.96, 0.05)]
    threshold_rows = _threshold_scan(actual, score, thresholds)
    best_threshold = max(threshold_rows, key=lambda item: item["f1"])
    return {
        "roc_auc": float(roc_auc_score(actual, score)),
        "average_precision": float(average_precision_score(actual, score)),
        "best_f1_threshold": float(best_threshold["threshold"]),
        "best_f1": float(best_threshold["f1"]),
        "best_precision": float(best_threshold["precision"]),
        "best_recall": float(best_threshold["recall"]),
        "threshold_scan": threshold_rows,
    }


def _fit_stage_models(
    train_frame: pd.DataFrame,
    *,
    dcs_features: list[str],
    stage_policy: dict[str, object],
) -> dict[str, dict[str, object]]:
    models: dict[str, dict[str, object]] = {}
    for stage_name, stage_frame in train_frame.groupby("stage_name"):
        stage_bundle: dict[str, object] = {"dcs_model": None, "ph_model": None}
        stage_bundle["dcs_model"] = _fit_binary_ensemble(stage_frame, dcs_features)
        stage_decision = stage_policy["policy"]["recommendations"].get(stage_name, {})
        if bool(stage_decision.get("enable_ph", False)):
            ph_stage = stage_frame.dropna(subset=DEFAULT_PH_FEATURE_COLUMNS).copy()
            if not ph_stage.empty:
                stage_bundle["ph_model"] = _fit_binary_ensemble(ph_stage, dcs_features + DEFAULT_PH_FEATURE_COLUMNS)
        models[stage_name] = stage_bundle
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a stronger dedicated out-of-spec alert classifier for T90.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--splits", type=int, default=5, help="TimeSeriesSplit count.")
    args = parser.parse_args()

    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = results_dir / "oos_alert_experiment_summary.json"
    predictions_csv_path = results_dir / "oos_alert_experiment_predictions.csv"

    config, stage_policy, base_casebase, ph_casebase = _load_assets(Path(args.config))
    frame = _prepare_alert_frame(base_casebase, ph_casebase)
    frame = frame.sort_values("sample_time").reset_index(drop=True)
    dcs_features = _select_dcs_feature_columns(frame)
    actual = frame["is_out_of_spec"].astype(int)

    tscv = TimeSeriesSplit(n_splits=args.splits)
    global_probs = pd.Series(index=frame.index, dtype=float)
    stage_dcs_probs = pd.Series(index=frame.index, dtype=float)
    stage_hybrid_probs = pd.Series(index=frame.index, dtype=float)
    stage_hybrid_meta: list[dict[str, object]] = []

    for fold_index, (train_index, test_index) in enumerate(tscv.split(frame), start=1):
        print(f"evaluating fold {fold_index}/{args.splits}")
        train_frame = frame.iloc[train_index].copy()
        test_frame = frame.iloc[test_index].copy()

        global_model = _fit_binary_ensemble(train_frame, dcs_features)
        if global_model is None:
            raise ValueError("Global DCS alert model could not be fitted.")
        global_probs.iloc[test_index] = _predict_binary_ensemble(global_model, test_frame)

        stage_models = _fit_stage_models(train_frame, dcs_features=dcs_features, stage_policy=stage_policy)
        for stage_name, stage_test in test_frame.groupby("stage_name"):
            stage_bundle = stage_models.get(stage_name, {})
            dcs_model = stage_bundle.get("dcs_model")
            if dcs_model is None:
                stage_dcs_probs.loc[stage_test.index] = _predict_binary_ensemble(global_model, stage_test)
            else:
                stage_dcs_probs.loc[stage_test.index] = _predict_binary_ensemble(dcs_model, stage_test)

            for row_index, row in stage_test.iterrows():
                probability, selection = _predict_stage_aware_probability(
                    row.to_frame().T,
                    stage_name=stage_name,
                    stage_policy=stage_policy,
                    stage_models=stage_models,
                    global_dcs_model=global_model,
                )
                stage_hybrid_probs.loc[row_index] = probability
                stage_hybrid_meta.append(
                    {
                        "index": int(row_index),
                        "selected_model": selection["selected_model"],
                        "ph_policy_enabled": bool(selection["ph_policy_enabled"]),
                        "ph_used": bool(selection["ph_used"]),
                        "ph_fallback_to_dcs_only": bool(selection["ph_fallback_to_dcs_only"]),
                    }
                )

    prediction_table = frame[["sample_time", "stage_name", "t90", "is_in_spec", "is_out_of_spec"]].copy()
    prediction_table["prob_global_dcs"] = global_probs
    prediction_table["prob_stage_dcs"] = stage_dcs_probs
    prediction_table["prob_stage_hybrid"] = stage_hybrid_probs
    meta_frame = pd.DataFrame(stage_hybrid_meta).drop_duplicates(subset=["index"]).set_index("index")
    prediction_table = prediction_table.join(meta_frame, how="left")
    prediction_table["prob_stage_dcs"] = prediction_table["prob_stage_dcs"].fillna(prediction_table["prob_global_dcs"])
    prediction_table["prob_stage_hybrid"] = (
        prediction_table["prob_stage_hybrid"]
        .fillna(prediction_table["prob_stage_dcs"])
        .fillna(prediction_table["prob_global_dcs"])
    )
    prediction_table["selected_model"] = prediction_table["selected_model"].fillna("fallback_global_dcs")
    prediction_table["ph_policy_enabled"] = _safe_bool_series(prediction_table["ph_policy_enabled"])
    prediction_table["ph_used"] = _safe_bool_series(prediction_table["ph_used"])
    prediction_table["ph_fallback_to_dcs_only"] = _safe_bool_series(prediction_table["ph_fallback_to_dcs_only"])
    prediction_table.to_csv(predictions_csv_path, index=False, encoding="utf-8-sig")

    scored_mask = prediction_table[["prob_global_dcs", "prob_stage_dcs", "prob_stage_hybrid"]].notna().all(axis=1)
    scored_table = prediction_table.loc[scored_mask].copy()
    actual_scored = scored_table["is_out_of_spec"].astype(int)
    strategies = {
        "global_dcs_ensemble": scored_table["prob_global_dcs"],
        "stage_dcs_ensemble": scored_table["prob_stage_dcs"],
        "stage_hybrid_optional_ph": scored_table["prob_stage_hybrid"],
    }
    summary = {
        "target_definition": "1 when T90 is outside 8.45 +/- 0.25, otherwise 0",
        "samples": int(len(prediction_table)),
        "scored_samples": int(len(scored_table)),
        "unscored_leading_samples": int(len(prediction_table) - len(scored_table)),
        "out_of_spec_ratio": float(actual.mean()),
        "out_of_spec_ratio_scored_only": float(actual_scored.mean()),
        "stage_policy": stage_policy["policy"]["recommendations"],
        "strategies": {
            name: _summarize_classifier(actual_scored, score.astype(float))
            for name, score in strategies.items()
        },
        "by_stage": {},
        "artifacts": {
            "predictions_csv": str(predictions_csv_path),
            "summary_json": str(summary_json_path),
        },
    }

    for stage_name, stage_frame in scored_table.groupby("stage_name"):
        stage_actual = stage_frame["is_out_of_spec"].astype(int)
        summary["by_stage"][stage_name] = {
            "samples": int(len(stage_frame)),
            "out_of_spec_ratio": float(stage_actual.mean()),
            "strategies": {},
        }
        if stage_actual.nunique() < 2:
            continue
        for name, column in (
            ("global_dcs_ensemble", "prob_global_dcs"),
            ("stage_dcs_ensemble", "prob_stage_dcs"),
            ("stage_hybrid_optional_ph", "prob_stage_hybrid"),
        ):
            summary["by_stage"][stage_name]["strategies"][name] = _summarize_classifier(stage_actual, stage_frame[column].astype(float))

    best_strategy = max(
        summary["strategies"].items(),
        key=lambda item: (item[1]["average_precision"], item[1]["roc_auc"]),
    )
    summary["recommended_strategy"] = {
        "name": best_strategy[0],
        "reason": "Chosen by highest average precision, with ROC AUC used as a tie-breaker.",
        "metrics": best_strategy[1],
    }
    summary["stage_hybrid_runtime_usage"] = {
        "ph_used_ratio": float(_safe_bool_series(prediction_table["ph_used"]).mean()),
        "ph_fallback_ratio": float(_safe_bool_series(prediction_table["ph_fallback_to_dcs_only"]).mean()),
        "selected_model_counts": prediction_table["selected_model"].fillna("missing").value_counts().to_dict(),
    }

    if RISK_RESULTS_PATH.exists():
        risk_frame = pd.read_csv(RISK_RESULTS_PATH)
        risk_frame["is_out_of_spec"] = risk_frame["t90_status"].astype(str).ne("in_spec").astype(int)
        baseline_score = 1.0 - pd.to_numeric(risk_frame["best_point_probability"], errors="coerce").fillna(0.0)
        summary["current_recommender_probability_baseline"] = {
            "samples": int(len(risk_frame)),
            "out_of_spec_ratio": float(risk_frame["is_out_of_spec"].mean()),
            "roc_auc": float(roc_auc_score(risk_frame["is_out_of_spec"], baseline_score)),
            "average_precision": float(average_precision_score(risk_frame["is_out_of_spec"], baseline_score)),
            "meaning": "Uses 1 - best_point_probability from the current recommendation replay as an indirect alert baseline.",
        }

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
