from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from run_ordinal_cumulative_current_head import (
    build_feature_rows,
    feature_columns,
    format_threshold,
    load_lims_data,
    preclean_features,
    probability_argmax,
    select_topk_sensors,
    selected_feature_columns,
)
from run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup import (
    BUSINESS_LABELS,
    CLEANROOM_DIR,
    PROJECT_DIR,
    aggregate_metrics,
    assign_labels,
    build_candidate_grid,
    evaluate_prediction_set,
    evaluate_predictions,
    fit_cumulative_business_probabilities_inner_thresholds_only,
    load_config,
    monotonicity_summary,
    select_candidate_on_train as select_weight_candidate_on_train,
)
from run_ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head import load_identity_deduped_combined_dcs


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = (
    CLEANROOM_DIR
    / "configs"
    / "ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_ph_lag.yaml"
)


def load_ph_data(path: Path, sensor_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_excel(path, sheet_name=0, usecols=[1, 2])
    raw.columns = ["time", sensor_name]
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    raw[sensor_name] = pd.to_numeric(raw[sensor_name], errors="coerce")
    cleaned = (
        raw.dropna(subset=["time", sensor_name])
        .sort_values("time")
        .groupby("time", as_index=False)
        .agg({sensor_name: "mean"})
        .sort_values("time")
        .reset_index(drop=True)
    )
    audit = {
        "rows": int(len(cleaned)),
        "time_min": str(cleaned["time"].min()),
        "time_max": str(cleaned["time"].max()),
        "value_mean": float(cleaned[sensor_name].mean()),
        "value_std": float(cleaned[sensor_name].std(ddof=0)),
        "value_min": float(cleaned[sensor_name].min()),
        "value_max": float(cleaned[sensor_name].max()),
    }
    return cleaned, audit


def build_ph_candidate_grid(config: dict[str, object]) -> list[dict[str, object]]:
    candidates = [{"candidate_name": "default", "lag_minutes": None, "use_ph": False}]
    for lag_minutes in config["ph_lag"]["lag_grid_minutes"]:
        lag = int(lag_minutes)
        candidates.append(
            {
                "candidate_name": f"ph_lag_{lag:03d}",
                "lag_minutes": lag,
                "use_ph": True,
            }
        )
    return candidates


def build_ph_feature_frame(
    lims: pd.DataFrame,
    ph: pd.DataFrame,
    *,
    sensor_name: str,
    lookback_minutes: int,
    lag_minutes: int,
    stats: list[str],
) -> pd.DataFrame:
    shifted = ph.copy()
    shifted["time"] = shifted["time"] + pd.Timedelta(minutes=int(lag_minutes))
    feature_rows = build_feature_rows(
        lims,
        shifted,
        lookback_minutes=lookback_minutes,
        stats=stats,
    )
    ph_columns = [column for column in feature_rows.columns if column.startswith(f"{sensor_name}__")]
    keep_columns = ["sample_time", *ph_columns]
    return feature_rows[keep_columns].copy()


def ph_feature_columns(frame: pd.DataFrame, sensor_name: str) -> list[str]:
    prefix = f"{sensor_name}__"
    return sorted([column for column in frame.columns if column.startswith(prefix)])


def filter_candidate_ph_columns(
    train_frame: pd.DataFrame,
    columns: list[str],
    config: dict[str, object],
) -> list[str]:
    if not columns:
        return []
    features_cfg = config["features"]
    numeric = train_frame[columns].apply(pd.to_numeric, errors="coerce")
    missing_ratio = numeric.isna().mean()
    kept = missing_ratio[missing_ratio <= float(features_cfg["max_missing_ratio"])].index.tolist()
    if not kept:
        return []
    numeric = numeric[kept]
    nunique = numeric.nunique(dropna=True)
    kept = nunique[nunique > 1].index.tolist()
    if not kept:
        return []
    numeric = numeric[kept]
    final_kept: list[str] = []
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            continue
        dominance = float(series.round(10).value_counts(normalize=True, dropna=False).iloc[0])
        if dominance < float(features_cfg["near_constant_ratio"]):
            final_kept.append(column)
    return sorted(final_kept)


def select_ph_candidate_on_train(
    train_base: pd.DataFrame,
    train_candidate_frames: dict[str, pd.DataFrame],
    *,
    selected_dcs_features: list[str],
    thresholds: list[float],
    config: dict[str, object],
    weight_candidate: dict[str, object],
    ph_candidates: list[dict[str, object]],
    sensor_name: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    ph_cfg = config["ph_lag"]
    inner_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["inner_n_splits"]))
    candidate_rows: list[dict[str, object]] = []

    for candidate in ph_candidates:
        candidate_name = str(candidate["candidate_name"])
        candidate_frame = train_candidate_frames[candidate_name]
        outer_candidate_ph_cols = filter_candidate_ph_columns(
            candidate_frame,
            ph_feature_columns(candidate_frame, sensor_name),
            config,
        )
        inner_rows: list[dict[str, object]] = []
        for inner_train_index, inner_val_index in inner_tscv.split(train_base):
            inner_train = candidate_frame.iloc[inner_train_index].copy()
            inner_val = candidate_frame.iloc[inner_val_index].copy()
            features = selected_dcs_features + outer_candidate_ph_cols
            _cumulative, probabilities = fit_cumulative_business_probabilities_inner_thresholds_only(
                inner_train,
                inner_val,
                features=features,
                thresholds=thresholds,
                config=config,
                candidate=weight_candidate,
            )
            predicted = probability_argmax(probabilities, BUSINESS_LABELS)
            metrics = evaluate_prediction_set(inner_val, probabilities, predicted)
            inner_rows.append(
                {
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "core_qualified_average_precision": metrics["core_qualified_average_precision"],
                    "boundary_warning_average_precision": metrics["boundary_warning_average_precision"],
                    "clearly_unacceptable_average_precision": metrics["clearly_unacceptable_average_precision"],
                    "boundary_high_confidence_non_warning_rate": metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                }
            )

        aggregate = aggregate_metrics(inner_rows)
        candidate_rows.append(
            {
                "candidate_name": candidate_name,
                "lag_minutes": candidate["lag_minutes"],
                "use_ph": candidate["use_ph"],
                "usable_ph_feature_count": int(len(outer_candidate_ph_cols)),
                "usable_ph_features": outer_candidate_ph_cols,
                **aggregate,
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows).sort_values("candidate_name").reset_index(drop=True)
    default_row = candidate_frame.loc[candidate_frame["candidate_name"] == "default"].iloc[0]
    macro_tol = float(ph_cfg["inner_macro_f1_tolerance"])
    bal_tol = float(ph_cfg["inner_balanced_accuracy_tolerance"])
    eligible = candidate_frame[
        (candidate_frame["macro_f1"] >= float(default_row["macro_f1"]) - macro_tol)
        & (candidate_frame["balanced_accuracy"] >= float(default_row["balanced_accuracy"]) - bal_tol)
    ].copy()
    if eligible.empty:
        eligible = candidate_frame.loc[candidate_frame["candidate_name"] == "default"].copy()

    chosen = eligible.sort_values(
        [
            "boundary_high_confidence_non_warning_rate",
            "boundary_warning_average_precision",
            "macro_f1",
            "balanced_accuracy",
        ],
        ascending=[True, False, False, False],
    ).iloc[0]
    return chosen.to_dict(), candidate_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate lag-aware PH augmentation on top of the frozen strongest simple 120min monotonic cumulative baseline."
    )
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
    ph_path = PROJECT_DIR / str(config["data"]["ph_path"])
    lims_path = next((PROJECT_DIR / "data").glob(str(config["data"]["lims_glob"])))

    lookback_minutes = int(config["features"]["lookback_minutes"])
    stats = list(config["features"]["window_statistics"])
    sensor_name = str(config["ph_lag"]["sensor_name"])

    lims, _ = load_lims_data(lims_path)
    dcs, dcs_audit, alias_frame = load_identity_deduped_combined_dcs(main_dcs_path, supplemental_path, config)
    ph, ph_audit = load_ph_data(ph_path, sensor_name)

    base_feature_rows = build_feature_rows(
        lims,
        dcs,
        lookback_minutes=lookback_minutes,
        stats=stats,
    )
    base_feature_rows = assign_labels(base_feature_rows, config)
    base_feature_rows, base_preclean_summary = preclean_features(base_feature_rows, config)
    base_feature_cols = feature_columns(base_feature_rows)
    thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]
    weight_candidates = build_candidate_grid(config)
    ph_candidates = build_ph_candidate_grid(config)

    ph_candidate_frames: dict[str, pd.DataFrame] = {"default": base_feature_rows.copy()}
    ph_candidate_audit_rows: list[dict[str, object]] = [
        {"candidate_name": "default", "lag_minutes": None, "usable_ph_feature_count": 0, "usable_ph_features": []}
    ]
    for candidate in ph_candidates:
        if not bool(candidate["use_ph"]):
            continue
        lag_minutes = int(candidate["lag_minutes"])
        ph_feature_frame = build_ph_feature_frame(
            lims,
            ph,
            sensor_name=sensor_name,
            lookback_minutes=lookback_minutes,
            lag_minutes=lag_minutes,
            stats=stats,
        )
        merged = base_feature_rows.merge(ph_feature_frame, on="sample_time", how="left", sort=False)
        ph_columns = ph_feature_columns(merged, sensor_name)
        ph_candidate_frames[str(candidate["candidate_name"])] = merged
        ph_candidate_audit_rows.append(
            {
                "candidate_name": candidate["candidate_name"],
                "lag_minutes": lag_minutes,
                "usable_ph_feature_count": int(len(ph_columns)),
                "usable_ph_features": ph_columns,
            }
        )

    outer_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    predictions = base_feature_rows[
        ["sample_time", "t90", "business_label", "boundary_any_flag", "is_unacceptable"]
        + [column for column in base_feature_rows.columns if column.startswith("target_lt_")]
    ].copy()
    for threshold in thresholds:
        key = format_threshold(threshold)
        predictions[f"baseline_cum_prob_lt_{key}"] = pd.NA
        predictions[f"ph_cum_prob_lt_{key}"] = pd.NA
    for label in BUSINESS_LABELS:
        predictions[f"baseline_business_prob_{label}"] = pd.NA
        predictions[f"ph_business_prob_{label}"] = pd.NA
    predictions["baseline_business_pred"] = pd.NA
    predictions["ph_business_pred"] = pd.NA
    predictions["selected_ph_candidate"] = pd.NA

    per_fold_rows: list[dict[str, object]] = []
    ph_selected_rows: list[dict[str, object]] = []
    ph_inner_search_rows: list[dict[str, object]] = []
    baseline_eval_rows: list[dict[str, object]] = []
    ph_eval_rows: list[dict[str, object]] = []

    for fold_index, (train_index, test_index) in enumerate(outer_tscv.split(base_feature_rows), start=1):
        train_base = base_feature_rows.iloc[train_index].copy()
        test_base = base_feature_rows.iloc[test_index].copy()
        train_candidate_frames = {
            candidate_name: frame.iloc[train_index].copy()
            for candidate_name, frame in ph_candidate_frames.items()
        }
        test_candidate_frames = {
            candidate_name: frame.iloc[test_index].copy()
            for candidate_name, frame in ph_candidate_frames.items()
        }

        weight_candidate, _weight_search = select_weight_candidate_on_train(
            train_base,
            feature_cols=base_feature_cols,
            thresholds=thresholds,
            config=config,
            candidates=weight_candidates,
        )

        selected_sensors, selected_scores = select_topk_sensors(
            train_base,
            base_feature_cols,
            topk=int(config["screening"]["topk_sensors"]),
        )
        selected_dcs_features = selected_feature_columns(base_feature_cols, selected_sensors)

        selected_ph_candidate, candidate_search = select_ph_candidate_on_train(
            train_base,
            train_candidate_frames,
            selected_dcs_features=selected_dcs_features,
            thresholds=thresholds,
            config=config,
            weight_candidate=weight_candidate,
            ph_candidates=ph_candidates,
            sensor_name=sensor_name,
        )
        candidate_search.insert(0, "fold", fold_index)
        ph_inner_search_rows.extend(candidate_search.to_dict(orient="records"))
        ph_selected_rows.append({"fold": fold_index, **selected_ph_candidate})

        chosen_candidate_name = str(selected_ph_candidate["candidate_name"])
        train_ph = train_candidate_frames[chosen_candidate_name]
        test_ph = test_candidate_frames[chosen_candidate_name]
        chosen_ph_features = filter_candidate_ph_columns(
            train_ph,
            ph_feature_columns(train_ph, sensor_name),
            config,
        )
        treatment_features = selected_dcs_features + chosen_ph_features

        baseline_cumulative, baseline_probabilities = fit_cumulative_business_probabilities_inner_thresholds_only(
            train_base,
            test_base,
            features=selected_dcs_features,
            thresholds=thresholds,
            config=config,
            candidate=weight_candidate,
        )
        ph_cumulative, ph_probabilities = fit_cumulative_business_probabilities_inner_thresholds_only(
            train_ph,
            test_ph,
            features=treatment_features,
            thresholds=thresholds,
            config=config,
            candidate=weight_candidate,
        )

        baseline_pred = probability_argmax(baseline_probabilities, BUSINESS_LABELS)
        ph_pred = probability_argmax(ph_probabilities, BUSINESS_LABELS)

        for threshold in thresholds:
            key = format_threshold(threshold)
            predictions.loc[test_base.index, f"baseline_cum_prob_lt_{key}"] = baseline_cumulative[f"lt_{key}"]
            predictions.loc[test_base.index, f"ph_cum_prob_lt_{key}"] = ph_cumulative[f"lt_{key}"]
        for label in BUSINESS_LABELS:
            predictions.loc[test_base.index, f"baseline_business_prob_{label}"] = baseline_probabilities[label]
            predictions.loc[test_base.index, f"ph_business_prob_{label}"] = ph_probabilities[label]
        predictions.loc[test_base.index, "baseline_business_pred"] = baseline_pred
        predictions.loc[test_base.index, "ph_business_pred"] = ph_pred
        predictions.loc[test_base.index, "selected_ph_candidate"] = chosen_candidate_name

        baseline_metrics = evaluate_prediction_set(test_base, baseline_probabilities, baseline_pred)
        ph_metrics = evaluate_prediction_set(test_base, ph_probabilities, ph_pred)

        baseline_eval_rows.append(
            {
                "fold": fold_index,
                "macro_f1": baseline_metrics["macro_f1"],
                "balanced_accuracy": baseline_metrics["balanced_accuracy"],
                "core_qualified_average_precision": baseline_metrics["core_qualified_average_precision"],
                "boundary_warning_average_precision": baseline_metrics["boundary_warning_average_precision"],
                "clearly_unacceptable_average_precision": baseline_metrics["clearly_unacceptable_average_precision"],
                "boundary_high_confidence_non_warning_rate": baseline_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )
        ph_eval_rows.append(
            {
                "fold": fold_index,
                "macro_f1": ph_metrics["macro_f1"],
                "balanced_accuracy": ph_metrics["balanced_accuracy"],
                "core_qualified_average_precision": ph_metrics["core_qualified_average_precision"],
                "boundary_warning_average_precision": ph_metrics["boundary_warning_average_precision"],
                "clearly_unacceptable_average_precision": ph_metrics["clearly_unacceptable_average_precision"],
                "boundary_high_confidence_non_warning_rate": ph_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )

        per_fold_rows.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_base)),
                "test_rows": int(len(test_base)),
                "selected_sensor_count": int(len(selected_sensors)),
                "selected_sensors": selected_sensors,
                "selected_sensor_scores": selected_scores,
                "weight_candidate_name": weight_candidate["candidate_name"],
                "weight_boundary_weight": weight_candidate["boundary_weight"],
                "weight_warning_weight": weight_candidate["warning_weight"],
                "selected_ph_candidate_name": chosen_candidate_name,
                "selected_ph_lag_minutes": selected_ph_candidate["lag_minutes"],
                "usable_ph_feature_count": int(len(chosen_ph_features)),
                "usable_ph_features": chosen_ph_features,
                "baseline_macro_f1": baseline_metrics["macro_f1"],
                "ph_macro_f1": ph_metrics["macro_f1"],
                "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
                "ph_balanced_accuracy": ph_metrics["balanced_accuracy"],
                "baseline_boundary_high_confidence_non_warning_rate": baseline_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                "ph_boundary_high_confidence_non_warning_rate": ph_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )

    for label in BUSINESS_LABELS:
        predictions[f"baseline_business_prob_{label}"] = pd.to_numeric(predictions[f"baseline_business_prob_{label}"], errors="coerce")
        predictions[f"ph_business_prob_{label}"] = pd.to_numeric(predictions[f"ph_business_prob_{label}"], errors="coerce")
    for threshold in thresholds:
        key = format_threshold(threshold)
        predictions[f"baseline_cum_prob_lt_{key}"] = pd.to_numeric(predictions[f"baseline_cum_prob_lt_{key}"], errors="coerce")
        predictions[f"ph_cum_prob_lt_{key}"] = pd.to_numeric(predictions[f"ph_cum_prob_lt_{key}"], errors="coerce")

    scored_predictions = predictions.loc[
        predictions["baseline_business_pred"].notna() & predictions["ph_business_pred"].notna()
    ].copy()

    baseline_summary = evaluate_predictions(
        scored_predictions,
        pred_col="baseline_business_pred",
        acceptable_prob_col="baseline_business_prob_acceptable",
        warning_prob_col="baseline_business_prob_warning",
        unacceptable_prob_col="baseline_business_prob_unacceptable",
    )
    ph_summary = evaluate_predictions(
        scored_predictions,
        pred_col="ph_business_pred",
        acceptable_prob_col="ph_business_prob_acceptable",
        warning_prob_col="ph_business_prob_warning",
        unacceptable_prob_col="ph_business_prob_unacceptable",
    )
    baseline_monotonicity = monotonicity_summary(
        scored_predictions.rename(columns={f"baseline_cum_prob_lt_{format_threshold(t)}": f"cum_prob_lt_{format_threshold(t)}" for t in thresholds}),
        thresholds,
    )
    ph_monotonicity = monotonicity_summary(
        scored_predictions.rename(columns={f"ph_cum_prob_lt_{format_threshold(t)}": f"cum_prob_lt_{format_threshold(t)}" for t in thresholds}),
        thresholds,
    )
    baseline_fold_mean = aggregate_metrics([{k: v for k, v in row.items() if k != "fold"} for row in baseline_eval_rows])
    ph_fold_mean = aggregate_metrics([{k: v for k, v in row.items() if k != "fold"} for row in ph_eval_rows])
    improvement = {
        "macro_f1_delta": ph_summary["macro_f1"] - baseline_summary["macro_f1"],
        "balanced_accuracy_delta": ph_summary["balanced_accuracy"] - baseline_summary["balanced_accuracy"],
        "core_qualified_average_precision_delta": ph_summary["core_qualified_average_precision"] - baseline_summary["core_qualified_average_precision"],
        "boundary_warning_average_precision_delta": ph_summary["boundary_warning_average_precision"] - baseline_summary["boundary_warning_average_precision"],
        "clearly_unacceptable_average_precision_delta": ph_summary["clearly_unacceptable_average_precision"] - baseline_summary["clearly_unacceptable_average_precision"],
        "boundary_high_confidence_non_warning_rate_delta": ph_summary["boundary_overconfidence"]["high_confidence_non_warning_rate"] - baseline_summary["boundary_overconfidence"]["high_confidence_non_warning_rate"],
    }

    per_fold_csv = artifacts_dir / "ph_lag_per_fold.csv"
    selected_candidate_csv = artifacts_dir / "ph_lag_selected_candidate_per_fold.csv"
    inner_search_csv = artifacts_dir / "ph_lag_inner_search.csv"
    results_csv = artifacts_dir / "ph_lag_results.csv"
    alias_csv = artifacts_dir / "sensor_identity_alias_pairs.csv"
    summary_json = artifacts_dir / "ph_lag_summary.json"
    report_md = reports_dir / "ph_lag_summary.md"

    pd.DataFrame(per_fold_rows).to_csv(per_fold_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(ph_selected_rows).to_csv(selected_candidate_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(ph_inner_search_rows).to_csv(inner_search_csv, index=False, encoding="utf-8-sig")
    scored_predictions.to_csv(results_csv, index=False, encoding="utf-8-sig")
    alias_frame.to_csv(alias_csv, index=False, encoding="utf-8-sig")

    summary = {
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "ph_audit": ph_audit,
        "ph_candidate_grid": ph_candidates,
        "ph_candidate_audit": ph_candidate_audit_rows,
        "data_paths": {
            "dcs_main_path": str(main_dcs_path),
            "dcs_supplemental_path": str(supplemental_path),
            "ph_path": str(ph_path),
            "lims_path": str(lims_path),
        },
        "dcs_audit": dcs_audit,
        "identity_alias_pairs": alias_frame.to_dict(orient="records"),
        "data_summary": {
            "aligned_rows": int(len(base_feature_rows)),
            "raw_common_rows": int(len(predictions)),
            "scored_rows": int(len(scored_predictions)),
            "n_splits_used": int(config["validation"]["n_splits"]),
            "inner_n_splits_used": int(config["validation"]["inner_n_splits"]),
        },
        "base_preclean_summary": base_preclean_summary,
        "baseline_summary": baseline_summary,
        "ph_summary": ph_summary,
        "baseline_monotonicity": baseline_monotonicity,
        "ph_monotonicity": ph_monotonicity,
        "baseline_fold_mean": baseline_fold_mean,
        "ph_fold_mean": ph_fold_mean,
        "improvement_summary": improvement,
        "per_fold": per_fold_rows,
        "selected_candidate_rows": ph_selected_rows,
        "artifacts": {
            "per_fold_csv": str(per_fold_csv),
            "selected_candidate_csv": str(selected_candidate_csv),
            "inner_search_csv": str(inner_search_csv),
            "results_csv": str(results_csv),
            "alias_csv": str(alias_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# PH Lag-Aware Cleanroom Summary",
        "",
        "- Baseline: frozen strongest simple 120min monotonic cumulative main line",
        "- Treatment: append lag-aware PH window stats without changing DCS topk=40 screening",
        "- DCS source: merge_data + merge_data_otr with strict sensor-identity de-dup",
        f"- PH source: {ph_path.name}",
        f"- PH lag grid: {[item['lag_minutes'] for item in ph_candidates if item['use_ph']]}",
        f"- Scored rows: {len(scored_predictions)}",
        f"- Baseline macro_f1: {baseline_summary['macro_f1']:.4f}",
        f"- PH macro_f1: {ph_summary['macro_f1']:.4f}",
        f"- Baseline balanced_accuracy: {baseline_summary['balanced_accuracy']:.4f}",
        f"- PH balanced_accuracy: {ph_summary['balanced_accuracy']:.4f}",
        f"- Baseline warning_AP: {baseline_summary['boundary_warning_average_precision']:.4f}",
        f"- PH warning_AP: {ph_summary['boundary_warning_average_precision']:.4f}",
        f"- Baseline unacceptable_AP: {baseline_summary['clearly_unacceptable_average_precision']:.4f}",
        f"- PH unacceptable_AP: {ph_summary['clearly_unacceptable_average_precision']:.4f}",
        f"- Baseline boundary high-confidence non-warning: {baseline_summary['boundary_overconfidence']['high_confidence_non_warning_rate']:.4f}",
        f"- PH boundary high-confidence non-warning: {ph_summary['boundary_overconfidence']['high_confidence_non_warning_rate']:.4f}",
        "",
        "## Selected PH Candidate Per Fold",
        "",
    ]
    for row in ph_selected_rows:
        lines.append(
            f"- fold {row['fold']}: {row['candidate_name']} "
            f"(lag_minutes={row['lag_minutes']}, usable_ph_feature_count={row['usable_ph_feature_count']})"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
