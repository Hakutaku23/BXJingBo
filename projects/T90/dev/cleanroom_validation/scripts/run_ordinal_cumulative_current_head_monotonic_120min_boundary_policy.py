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
    load_dcs_data,
    load_lims_data,
    predict_threshold_probability,
    preclean_features,
    probability_argmax,
    select_topk_sensors,
    selected_feature_columns,
)


THIS_DIR = Path(__file__).resolve().parent
CLEANROOM_DIR = THIS_DIR.parent
PROJECT_DIR = CLEANROOM_DIR.parents[1]
DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "ordinal_cumulative_current_head_monotonic_120min_boundary_policy.yaml"
BUSINESS_LABELS = ["acceptable", "warning", "unacceptable"]


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_policy_grid(config: dict[str, object]) -> list[dict[str, object]]:
    policy_cfg = config["decision_policy"]
    candidates = [{"policy_name": "default", "warning_floor": None, "margin": None}]
    for warning_floor in policy_cfg["warning_floor_grid"]:
        for margin in policy_cfg["margin_grid"]:
            candidates.append(
                {
                    "policy_name": f"warning_floor_{float(warning_floor):.2f}_margin_{float(margin):.2f}",
                    "warning_floor": float(warning_floor),
                    "margin": float(margin),
                }
            )
    return candidates


def predict_with_boundary_policy(probabilities: pd.DataFrame, candidate: dict[str, object]) -> pd.Series:
    default_pred = probability_argmax(probabilities, BUSINESS_LABELS)
    if candidate["warning_floor"] is None or candidate["margin"] is None:
        return default_pred

    pred = default_pred.copy()
    top_non_warning = probabilities[["acceptable", "unacceptable"]].max(axis=1)
    should_flip = (
        default_pred.ne("warning")
        & probabilities["warning"].ge(float(candidate["warning_floor"]))
        & (top_non_warning - probabilities["warning"]).le(float(candidate["margin"]))
    )
    pred.loc[should_flip] = "warning"
    return pred


def fit_cumulative_business_probabilities(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    features: list[str],
    thresholds: list[float],
    config: dict[str, object],
) -> pd.DataFrame:
    cumulative = pd.DataFrame(index=test_frame.index)
    for threshold in thresholds:
        key = format_threshold(threshold)
        target_key = f"target_lt_{key}"
        model = fit_threshold_model(train_frame, features, target_key, config)
        cumulative[f"lt_{key}"] = predict_threshold_probability(model, test_frame[features])

    if bool(config["models"]["apply_monotonic_correction"]):
        cumulative = apply_monotonic_correction(cumulative)
    intervals = clip_and_renormalize(cumulative_to_interval_probabilities(cumulative))
    return business_probabilities_from_intervals(intervals)


def evaluate_prediction_set(
    frame: pd.DataFrame,
    probabilities: pd.DataFrame,
    predicted: pd.Series,
) -> dict[str, object]:
    scored = pd.DataFrame(index=frame.index)
    scored["business_label"] = frame["business_label"]
    scored["boundary_any_flag"] = frame["boundary_any_flag"]
    scored["pred"] = predicted
    scored["acceptable"] = probabilities["acceptable"]
    scored["warning"] = probabilities["warning"]
    scored["unacceptable"] = probabilities["unacceptable"]
    return {
        "macro_f1": float(f1_score(scored["business_label"], scored["pred"], labels=BUSINESS_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(scored["business_label"], scored["pred"])),
        "core_qualified_average_precision": binary_ap(scored["business_label"].eq("acceptable").astype(int), scored["acceptable"]),
        "boundary_warning_average_precision": binary_ap(scored["business_label"].eq("warning").astype(int), scored["warning"]),
        "clearly_unacceptable_average_precision": binary_ap(scored["business_label"].eq("unacceptable").astype(int), scored["unacceptable"]),
        "boundary_overconfidence": boundary_overconfidence_stats(
            scored,
            pred_col="pred",
            acceptable_prob_col="acceptable",
            unacceptable_prob_col="unacceptable",
        ),
    }


def aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    return {
        "macro_f1": float(frame["macro_f1"].mean()),
        "balanced_accuracy": float(frame["balanced_accuracy"].mean()),
        "core_qualified_average_precision": float(frame["core_qualified_average_precision"].dropna().mean()) if frame["core_qualified_average_precision"].notna().any() else None,
        "boundary_warning_average_precision": float(frame["boundary_warning_average_precision"].dropna().mean()) if frame["boundary_warning_average_precision"].notna().any() else None,
        "clearly_unacceptable_average_precision": float(frame["clearly_unacceptable_average_precision"].dropna().mean()) if frame["clearly_unacceptable_average_precision"].notna().any() else None,
        "boundary_high_confidence_non_warning_rate": float(frame["boundary_high_confidence_non_warning_rate"].mean()),
    }


def select_policy_on_train(
    train_frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    thresholds: list[float],
    config: dict[str, object],
    policy_candidates: list[dict[str, object]],
) -> tuple[dict[str, object], pd.DataFrame]:
    inner_n_splits = int(config["validation"]["inner_n_splits"])
    inner_tscv = TimeSeriesSplit(n_splits=inner_n_splits)
    candidate_rows: list[dict[str, object]] = []

    for candidate in policy_candidates:
        inner_rows: list[dict[str, object]] = []
        for inner_train_index, inner_val_index in inner_tscv.split(train_frame):
            inner_train = train_frame.iloc[inner_train_index].copy()
            inner_val = train_frame.iloc[inner_val_index].copy()
            selected_sensors, _ = select_topk_sensors(
                inner_train,
                feature_cols,
                topk=int(config["screening"]["topk_sensors"]),
            )
            selected_features = selected_feature_columns(feature_cols, selected_sensors)
            probabilities = fit_cumulative_business_probabilities(
                inner_train,
                inner_val,
                features=selected_features,
                thresholds=thresholds,
                config=config,
            )
            predicted = predict_with_boundary_policy(probabilities, candidate)
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
                "policy_name": candidate["policy_name"],
                "warning_floor": candidate["warning_floor"],
                "margin": candidate["margin"],
                **aggregate,
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows).sort_values("policy_name").reset_index(drop=True)
    default_row = candidate_frame.loc[candidate_frame["policy_name"] == "default"].iloc[0]
    macro_tol = float(config["decision_policy"]["inner_macro_f1_tolerance"])
    bal_tol = float(config["decision_policy"]["inner_balanced_accuracy_tolerance"])
    eligible = candidate_frame[
        (candidate_frame["macro_f1"] >= float(default_row["macro_f1"]) - macro_tol)
        & (candidate_frame["balanced_accuracy"] >= float(default_row["balanced_accuracy"]) - bal_tol)
    ].copy()
    if eligible.empty:
        eligible = candidate_frame.loc[candidate_frame["policy_name"] == "default"].copy()

    chosen = eligible.sort_values(
        [
            "boundary_high_confidence_non_warning_rate",
            "macro_f1",
            "balanced_accuracy",
            "boundary_warning_average_precision",
        ],
        ascending=[True, False, False, False],
    ).iloc[0]
    return chosen.to_dict(), candidate_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether a boundary-friendly decision policy can improve the strongest simple baseline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    outputs = config["outputs"]
    artifacts_dir = CLEANROOM_DIR / str(outputs["artifacts_dir"])
    reports_dir = CLEANROOM_DIR / str(outputs["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    lims_path = discover_lims_path(PROJECT_DIR / "data", str(config["data"]["lims_glob"]))
    dcs_path = PROJECT_DIR / str(config["data"]["dcs_path"])

    lims, _ = load_lims_data(lims_path)
    dcs = load_dcs_data(dcs_path)
    feature_rows = build_feature_rows(
        lims,
        dcs,
        lookback_minutes=int(config["features"]["lookback_minutes"]),
        stats=list(config["features"]["window_statistics"]),
    )
    feature_rows = assign_labels(feature_rows, config)
    feature_rows, preclean_summary = preclean_features(feature_rows, config)
    feature_cols = feature_columns(feature_rows)
    thresholds = [float(item) for item in config["labels"]["cumulative_thresholds"]]
    policy_candidates = build_policy_grid(config)

    outer_tscv = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    per_fold_rows: list[dict[str, object]] = []
    default_eval_rows: list[dict[str, object]] = []
    policy_eval_rows: list[dict[str, object]] = []
    selected_policy_rows: list[dict[str, object]] = []
    inner_search_rows: list[dict[str, object]] = []

    for fold_index, (train_index, test_index) in enumerate(outer_tscv.split(feature_rows), start=1):
        train_frame = feature_rows.iloc[train_index].copy()
        test_frame = feature_rows.iloc[test_index].copy()
        chosen_policy, candidate_frame = select_policy_on_train(
            train_frame,
            feature_cols=feature_cols,
            thresholds=thresholds,
            config=config,
            policy_candidates=policy_candidates,
        )
        candidate_frame.insert(0, "fold", fold_index)
        inner_search_rows.extend(candidate_frame.to_dict(orient="records"))
        selected_policy_rows.append({"fold": fold_index, **chosen_policy})

        selected_sensors, selected_scores = select_topk_sensors(
            train_frame,
            feature_cols,
            topk=int(config["screening"]["topk_sensors"]),
        )
        selected_features = selected_feature_columns(feature_cols, selected_sensors)
        probabilities = fit_cumulative_business_probabilities(
            train_frame,
            test_frame,
            features=selected_features,
            thresholds=thresholds,
            config=config,
        )

        default_pred = predict_with_boundary_policy(probabilities, {"warning_floor": None, "margin": None})
        policy_pred = predict_with_boundary_policy(probabilities, chosen_policy)
        default_metrics = evaluate_prediction_set(test_frame, probabilities, default_pred)
        policy_metrics = evaluate_prediction_set(test_frame, probabilities, policy_pred)

        default_eval_rows.append(
            {
                "fold": fold_index,
                "macro_f1": default_metrics["macro_f1"],
                "balanced_accuracy": default_metrics["balanced_accuracy"],
                "core_qualified_average_precision": default_metrics["core_qualified_average_precision"],
                "boundary_warning_average_precision": default_metrics["boundary_warning_average_precision"],
                "clearly_unacceptable_average_precision": default_metrics["clearly_unacceptable_average_precision"],
                "boundary_high_confidence_non_warning_rate": default_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )
        policy_eval_rows.append(
            {
                "fold": fold_index,
                "macro_f1": policy_metrics["macro_f1"],
                "balanced_accuracy": policy_metrics["balanced_accuracy"],
                "core_qualified_average_precision": policy_metrics["core_qualified_average_precision"],
                "boundary_warning_average_precision": policy_metrics["boundary_warning_average_precision"],
                "clearly_unacceptable_average_precision": policy_metrics["clearly_unacceptable_average_precision"],
                "boundary_high_confidence_non_warning_rate": policy_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )

        per_fold_rows.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "selected_sensor_count": int(len(selected_sensors)),
                "selected_sensors": selected_sensors,
                "selected_sensor_scores": selected_scores,
                "chosen_policy_name": chosen_policy["policy_name"],
                "chosen_warning_floor": chosen_policy["warning_floor"],
                "chosen_margin": chosen_policy["margin"],
                "default_macro_f1": default_metrics["macro_f1"],
                "policy_macro_f1": policy_metrics["macro_f1"],
                "default_balanced_accuracy": default_metrics["balanced_accuracy"],
                "policy_balanced_accuracy": policy_metrics["balanced_accuracy"],
                "default_boundary_high_conf_non_warning_rate": default_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                "policy_boundary_high_conf_non_warning_rate": policy_metrics["boundary_overconfidence"]["high_confidence_non_warning_rate"],
            }
        )

    default_summary = aggregate_metrics([{k: v for k, v in row.items() if k != "fold"} for row in default_eval_rows])
    policy_summary = aggregate_metrics([{k: v for k, v in row.items() if k != "fold"} for row in policy_eval_rows])
    improvement = {
        "macro_f1_delta": policy_summary["macro_f1"] - default_summary["macro_f1"],
        "balanced_accuracy_delta": policy_summary["balanced_accuracy"] - default_summary["balanced_accuracy"],
        "core_qualified_average_precision_delta": None if default_summary["core_qualified_average_precision"] is None or policy_summary["core_qualified_average_precision"] is None else policy_summary["core_qualified_average_precision"] - default_summary["core_qualified_average_precision"],
        "boundary_warning_average_precision_delta": None if default_summary["boundary_warning_average_precision"] is None or policy_summary["boundary_warning_average_precision"] is None else policy_summary["boundary_warning_average_precision"] - default_summary["boundary_warning_average_precision"],
        "boundary_high_confidence_non_warning_rate_delta": policy_summary["boundary_high_confidence_non_warning_rate"] - default_summary["boundary_high_confidence_non_warning_rate"],
    }

    per_fold_csv = artifacts_dir / "boundary_policy_per_fold.csv"
    selected_policy_csv = artifacts_dir / "boundary_policy_selected_per_fold.csv"
    inner_search_csv = artifacts_dir / "boundary_policy_inner_search.csv"
    summary_json = artifacts_dir / "boundary_policy_summary.json"
    report_md = reports_dir / "boundary_policy_summary.md"

    pd.DataFrame(per_fold_rows).to_csv(per_fold_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_policy_rows).to_csv(selected_policy_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(inner_search_rows).to_csv(inner_search_csv, index=False, encoding="utf-8-sig")

    summary = {
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "data_paths": {
            "dcs_path": str(dcs_path),
            "lims_path": str(lims_path),
        },
        "data_summary": {
            "aligned_rows": int(len(feature_rows)),
            "dcs_sensor_count": int(len([col for col in dcs.columns if col != "time"])),
            "n_splits_used": int(config["validation"]["n_splits"]),
            "inner_n_splits_used": int(config["validation"]["inner_n_splits"]),
        },
        "preclean_summary": preclean_summary,
        "decision_policy_grid_size": int(len(policy_candidates)),
        "default_summary": default_summary,
        "boundary_policy_summary": policy_summary,
        "improvement_summary": improvement,
        "per_fold": per_fold_rows,
        "selected_policy_rows": selected_policy_rows,
        "artifacts": {
            "per_fold_csv": str(per_fold_csv),
            "selected_policy_csv": str(selected_policy_csv),
            "inner_search_csv": str(inner_search_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Boundary Policy Summary",
        "",
        "- Baseline representation: simple 120min monotonic ordinal/cumulative",
        "- Intervention: decision-layer warning-favoring policy only",
        f"- Policy grid size: {len(policy_candidates)}",
        f"- Default macro_f1: {default_summary['macro_f1']:.4f}",
        f"- Policy macro_f1: {policy_summary['macro_f1']:.4f}",
        f"- Default balanced_accuracy: {default_summary['balanced_accuracy']:.4f}",
        f"- Policy balanced_accuracy: {policy_summary['balanced_accuracy']:.4f}",
        f"- Default boundary high-confidence non-warning: {default_summary['boundary_high_confidence_non_warning_rate']:.4f}",
        f"- Policy boundary high-confidence non-warning: {policy_summary['boundary_high_confidence_non_warning_rate']:.4f}",
        "",
        "## Selected Policy Per Fold",
        "",
    ]
    for row in selected_policy_rows:
        lines.append(
            f"- fold {row['fold']}: {row['policy_name']} "
            f"(warning_floor={row['warning_floor']}, margin={row['margin']})"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
