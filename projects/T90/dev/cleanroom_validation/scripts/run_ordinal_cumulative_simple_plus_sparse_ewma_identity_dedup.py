from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from run_ordinal_cumulative_current_head import (
    apply_monotonic_correction,
    assign_labels,
    binary_ap,
    boundary_overconfidence_stats,
    build_feature_rows,
    business_probabilities_from_intervals,
    clip_and_renormalize,
    cumulative_to_interval_probabilities,
    discover_lims_path,
    feature_columns,
    fit_threshold_model,
    format_threshold,
    load_lims_data,
    monotonicity_summary,
    preclean_features,
    predict_threshold_probability,
    probability_argmax,
    select_topk_sensors,
    selected_feature_columns,
)
from run_ordinal_cumulative_paper_faithful_ewma_current_head import build_recursive_ewma_rows
from run_ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head import load_identity_deduped_combined_dcs


THIS_DIR = Path(__file__).resolve().parent
CLEANROOM_DIR = THIS_DIR.parent
PROJECT_DIR = CLEANROOM_DIR.parents[1]
DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "ordinal_cumulative_simple_plus_sparse_ewma_identity_dedup.yaml"
BUSINESS_LABELS = ["acceptable", "warning", "unacceptable"]


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fit_business_probabilities(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    features: list[str],
    thresholds: list[float],
    config: dict[str, object],
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cumulative = pd.DataFrame(index=test_frame.index)
    for threshold in thresholds:
        key = format_threshold(threshold)
        target_key = f"target_lt_{key}"
        model = fit_threshold_model(train_frame, features, target_key, config)
        cumulative[f"lt_{key}"] = predict_threshold_probability(model, test_frame[features])

    if bool(config["models"]["apply_monotonic_correction"]):
        cumulative = apply_monotonic_correction(cumulative)
    business = business_probabilities_from_intervals(clip_and_renormalize(cumulative_to_interval_probabilities(cumulative)))
    cumulative.columns = [f"{prefix}_cum_prob_{column}" for column in cumulative.columns]
    business.columns = [f"{prefix}_business_prob_{column}" for column in business.columns]
    return cumulative, business


def evaluate_predictions(
    frame: pd.DataFrame,
    pred_col: str,
    acceptable_prob_col: str,
    warning_prob_col: str,
    unacceptable_prob_col: str,
) -> dict[str, object]:
    if frame[pred_col].isna().any():
        raise ValueError(f"Prediction column contains missing values: {pred_col}")
    actual = frame["business_label"].astype(str)
    predicted = frame[pred_col].astype(str)
    return {
        "macro_f1": float(f1_score(actual, predicted, labels=BUSINESS_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "core_qualified_average_precision": binary_ap(frame["business_label"].eq("acceptable").astype(int), frame[acceptable_prob_col]),
        "boundary_warning_average_precision": binary_ap(frame["business_label"].eq("warning").astype(int), frame[warning_prob_col]),
        "clearly_unacceptable_average_precision": binary_ap(frame["business_label"].eq("unacceptable").astype(int), frame[unacceptable_prob_col]),
        "boundary_overconfidence": boundary_overconfidence_stats(
            frame,
            pred_col=pred_col,
            acceptable_prob_col=acceptable_prob_col,
            unacceptable_prob_col=unacceptable_prob_col,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a hybrid simple + sparse EWMA validation after sensor-identity de-dup.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    outputs = config["outputs"]
    artifacts_dir = CLEANROOM_DIR / str(outputs["artifacts_dir"])
    reports_dir = CLEANROOM_DIR / str(outputs["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    main_dcs_path = PROJECT_DIR / str(config["data"]["dcs_main_path"])
    supplemental_path = PROJECT_DIR / str(config["data"]["dcs_supplemental_path"])
    lims_path = discover_lims_path(PROJECT_DIR / "data", str(config["data"]["lims_glob"]))

    lims, _ = load_lims_data(lims_path)
    dcs, dcs_audit, alias_frame = load_identity_deduped_combined_dcs(main_dcs_path, supplemental_path, config)

    baseline_rows = build_feature_rows(
        lims,
        dcs,
        lookback_minutes=int(config["baseline"]["lookback_minutes"]),
        stats=list(config["baseline"]["window_statistics"]),
    )
    baseline_rows = assign_labels(baseline_rows, config)
    baseline_rows, preclean_summary = preclean_features(baseline_rows, config)
    baseline_feature_cols = feature_columns(baseline_rows)
    candidate_sensors = sorted({col.split("__", 1)[0] for col in baseline_feature_cols})

    ewma_rows, ewma_audit = build_recursive_ewma_rows(
        lims,
        dcs,
        sensors=candidate_sensors,
        tau_minutes=int(config["ewma"]["tau_minutes"]),
        window_minutes=int(config["ewma"]["window_minutes"]),
        lambda_value=float(config["ewma"]["lambda"]),
        min_rows_per_window=int(config["ewma"]["min_rows_per_window"]),
        min_valid_points_per_sensor=int(config["ewma"]["min_valid_points_per_sensor"]),
    )
    ewma_rows = assign_labels(ewma_rows, config)
    ewma_feature_cols = [col for col in ewma_rows.columns if col.endswith("__ewma_recursive")]

    common = baseline_rows.merge(
        ewma_rows[["sample_time", *ewma_feature_cols]],
        on="sample_time",
        how="inner",
    ).sort_values("sample_time").reset_index(drop=True)
    thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]
    topk = int(config["screening"]["topk_sensors"])

    result_rows: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_predictions: pd.DataFrame | None = None

    for sparse_topm in config["screening"]["sparse_ewma_topm_grid"]:
        tscv = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
        fold_rows: list[dict[str, object]] = []
        predictions = common[
            ["sample_time", "t90", "business_label", "boundary_any_flag", "is_unacceptable"]
            + [column for column in common.columns if column.startswith("target_lt_")]
        ].copy()

        for threshold in thresholds:
            key = format_threshold(threshold)
            predictions[f"base_cum_prob_lt_{key}"] = pd.NA
            predictions[f"hybrid_cum_prob_lt_{key}"] = pd.NA
        for label in BUSINESS_LABELS:
            predictions[f"base_business_prob_{label}"] = pd.NA
            predictions[f"hybrid_business_prob_{label}"] = pd.NA
        predictions["base_business_pred"] = pd.NA
        predictions["hybrid_business_pred"] = pd.NA

        for fold_index, (train_index, test_index) in enumerate(tscv.split(common), start=1):
            train_frame = common.iloc[train_index].copy()
            test_frame = common.iloc[test_index].copy()

            base_sensors, base_scores = select_topk_sensors(train_frame, baseline_feature_cols, topk=topk)
            base_features = selected_feature_columns(baseline_feature_cols, base_sensors)
            restricted_ewma_cols = selected_feature_columns(ewma_feature_cols, base_sensors)
            sparse_sensors, sparse_scores = select_topk_sensors(
                train_frame,
                restricted_ewma_cols,
                topk=min(int(sparse_topm), len(base_sensors)),
            )
            sparse_ewma_features = selected_feature_columns(restricted_ewma_cols, sparse_sensors)
            hybrid_features = sorted(set(base_features + sparse_ewma_features))

            base_cumulative, base_business = fit_business_probabilities(
                train_frame,
                test_frame,
                features=base_features,
                thresholds=thresholds,
                config=config,
                prefix="base",
            )
            hybrid_cumulative, hybrid_business = fit_business_probabilities(
                train_frame,
                test_frame,
                features=hybrid_features,
                thresholds=thresholds,
                config=config,
                prefix="hybrid",
            )

            for column in base_cumulative.columns:
                predictions.loc[test_frame.index, column] = base_cumulative[column]
            for column in hybrid_cumulative.columns:
                predictions.loc[test_frame.index, column] = hybrid_cumulative[column]
            for column in base_business.columns:
                predictions.loc[test_frame.index, column] = base_business[column]
            for column in hybrid_business.columns:
                predictions.loc[test_frame.index, column] = hybrid_business[column]

            base_pred = probability_argmax(base_business.rename(columns=lambda name: name.replace("base_business_prob_", "")), BUSINESS_LABELS)
            hybrid_pred = probability_argmax(hybrid_business.rename(columns=lambda name: name.replace("hybrid_business_prob_", "")), BUSINESS_LABELS)
            predictions.loc[test_frame.index, "base_business_pred"] = base_pred
            predictions.loc[test_frame.index, "hybrid_business_pred"] = hybrid_pred

            fold_rows.append(
                {
                    "fold": fold_index,
                    "train_rows": int(len(train_frame)),
                    "test_rows": int(len(test_frame)),
                    "base_selected_sensor_count": int(len(base_sensors)),
                    "sparse_ewma_sensor_count": int(len(sparse_sensors)),
                    "base_selected_sensors": base_sensors,
                    "base_selected_sensor_scores": base_scores,
                    "sparse_ewma_sensors": sparse_sensors,
                    "sparse_ewma_sensor_scores": sparse_scores,
                    "base_macro_f1": float(f1_score(test_frame["business_label"], base_pred, labels=BUSINESS_LABELS, average="macro", zero_division=0)),
                    "hybrid_macro_f1": float(f1_score(test_frame["business_label"], hybrid_pred, labels=BUSINESS_LABELS, average="macro", zero_division=0)),
                    "base_balanced_accuracy": float(balanced_accuracy_score(test_frame["business_label"], base_pred)),
                    "hybrid_balanced_accuracy": float(balanced_accuracy_score(test_frame["business_label"], hybrid_pred)),
                }
            )

        for label in BUSINESS_LABELS:
            predictions[f"base_business_prob_{label}"] = pd.to_numeric(predictions[f"base_business_prob_{label}"], errors="coerce")
            predictions[f"hybrid_business_prob_{label}"] = pd.to_numeric(predictions[f"hybrid_business_prob_{label}"], errors="coerce")
        for threshold in thresholds:
            key = format_threshold(threshold)
            predictions[f"base_cum_prob_lt_{key}"] = pd.to_numeric(predictions[f"base_cum_prob_lt_{key}"], errors="coerce")
            predictions[f"hybrid_cum_prob_lt_{key}"] = pd.to_numeric(predictions[f"hybrid_cum_prob_lt_{key}"], errors="coerce")

        scored_predictions = predictions.loc[
            predictions["base_business_pred"].notna() & predictions["hybrid_business_pred"].notna()
        ].copy()
        if scored_predictions.empty:
            raise ValueError(f"No outer-test predictions were produced for sparse_topm={sparse_topm}.")

        base_metrics = evaluate_predictions(
            scored_predictions,
            pred_col="base_business_pred",
            acceptable_prob_col="base_business_prob_acceptable",
            warning_prob_col="base_business_prob_warning",
            unacceptable_prob_col="base_business_prob_unacceptable",
        )
        hybrid_metrics = evaluate_predictions(
            scored_predictions,
            pred_col="hybrid_business_pred",
            acceptable_prob_col="hybrid_business_prob_acceptable",
            warning_prob_col="hybrid_business_prob_warning",
            unacceptable_prob_col="hybrid_business_prob_unacceptable",
        )
        hybrid_monotonicity = monotonicity_summary(
            scored_predictions.rename(columns={f"hybrid_cum_prob_lt_{format_threshold(t)}": f"cum_prob_lt_{format_threshold(t)}" for t in thresholds}),
            thresholds,
        )
        row = {
            "sparse_ewma_topm": int(sparse_topm),
            "common_samples_raw": int(len(predictions)),
            "scored_samples": int(len(scored_predictions)),
            "baseline_macro_f1": base_metrics["macro_f1"],
            "hybrid_macro_f1": hybrid_metrics["macro_f1"],
            "baseline_balanced_accuracy": base_metrics["balanced_accuracy"],
            "hybrid_balanced_accuracy": hybrid_metrics["balanced_accuracy"],
            "baseline_core_AP": base_metrics["core_qualified_average_precision"],
            "hybrid_core_AP": hybrid_metrics["core_qualified_average_precision"],
            "baseline_warning_AP": base_metrics["boundary_warning_average_precision"],
            "hybrid_warning_AP": hybrid_metrics["boundary_warning_average_precision"],
            "baseline_boundary_high_conf_non_warning": base_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            "hybrid_boundary_high_conf_non_warning": hybrid_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            "hybrid_any_violation_rate": hybrid_monotonicity["any_violation_rate"],
            "mean_sparse_ewma_sensor_count": float(pd.DataFrame(fold_rows)["sparse_ewma_sensor_count"].mean()),
        }
        result_rows.append(row)

        if best_summary is None or (
            row["hybrid_macro_f1"],
            row["hybrid_balanced_accuracy"],
            row["hybrid_warning_AP"],
        ) > (
            best_summary["candidate"]["hybrid_macro_f1"],
            best_summary["candidate"]["hybrid_balanced_accuracy"],
            best_summary["candidate"]["hybrid_warning_AP"],
        ):
            best_summary = {
                "candidate": row,
                "metrics_summary": {
                    "baseline_simple": base_metrics,
                    "hybrid_treatment": hybrid_metrics,
                    "hybrid_monotonicity": hybrid_monotonicity,
                    "per_fold": fold_rows,
                },
            }
            best_predictions = scored_predictions.copy()

    if best_summary is None or best_predictions is None:
        raise ValueError("No sparse EWMA configuration was scored.")

    results_frame = pd.DataFrame(result_rows).sort_values(
        ["hybrid_macro_f1", "hybrid_balanced_accuracy", "hybrid_warning_AP"],
        ascending=[False, False, False],
    )
    results_csv = artifacts_dir / "simple_plus_sparse_ewma_results.csv"
    summary_json = artifacts_dir / "simple_plus_sparse_ewma_summary.json"
    best_rows_csv = artifacts_dir / "simple_plus_sparse_ewma_best_feature_rows.csv"
    alias_csv = artifacts_dir / "sensor_identity_alias_pairs.csv"
    audit_md = reports_dir / "simple_plus_sparse_ewma_audit.md"

    beat_both_count = int(
        (
            (results_frame["hybrid_macro_f1"] > results_frame["baseline_macro_f1"])
            & (results_frame["hybrid_balanced_accuracy"] > results_frame["baseline_balanced_accuracy"])
        ).sum()
    )
    summary = {
        "experiment_name": config["experiment_name"],
        "data_sources": {
            "dcs_main_path": str(main_dcs_path),
            "dcs_supplemental_path": str(supplemental_path),
            "lims_path": str(lims_path),
        },
        "dcs_audit": dcs_audit,
        "identity_alias_pairs": alias_frame.to_dict(orient="records"),
        "preclean_summary": preclean_summary,
        "ewma_reference": {
            "tau_minutes": int(config["ewma"]["tau_minutes"]),
            "window_minutes": int(config["ewma"]["window_minutes"]),
            "lambda": float(config["ewma"]["lambda"]),
        },
        "best_sparse_configuration": best_summary["candidate"],
        "best_sparse_metrics_summary": best_summary["metrics_summary"],
        "ewma_audit": ewma_audit,
        "beat_both_count": beat_both_count,
        "artifacts": {
            "results_csv": str(results_csv),
            "summary_json": str(summary_json),
            "best_feature_rows_csv": str(best_rows_csv),
            "alias_pairs_csv": str(alias_csv),
            "audit_md": str(audit_md),
        },
    }

    results_frame.to_csv(results_csv, index=False, encoding="utf-8-sig")
    best_predictions.to_csv(best_rows_csv, index=False, encoding="utf-8-sig")
    alias_frame.to_csv(alias_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best = best_summary["candidate"]
    lines = [
        "# Simple Plus Sparse EWMA Audit",
        "",
        "- Baseline: simple 120min monotonic ordinal/cumulative",
        "- Treatment: baseline simple features plus sparse EWMA augmentation on a subset of baseline-selected sensors",
        "- Sensor identity de-dup: on",
        f"- beat_both_count: {beat_both_count}",
        "",
        "## Best Sparse Configuration",
        "",
        json.dumps(best, ensure_ascii=False, indent=2),
    ]
    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
