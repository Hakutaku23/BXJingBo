from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_ordinal_cumulative_current_head import (
    assign_labels,
    build_feature_rows,
    discover_lims_path,
    feature_columns,
    load_dcs_data,
    load_lims_data,
    preclean_features,
)
from run_ordinal_cumulative_paper_faithful_ewma_current_head import build_recursive_ewma_rows
from run_ordinal_cumulative_paper_faithful_ewma_rep_specific_current_head import evaluate_representation


THIS_DIR = Path(__file__).resolve().parent
CLEANROOM_DIR = THIS_DIR.parent
PROJECT_DIR = CLEANROOM_DIR.parents[1]
DEFAULT_CONFIG_PATH = CLEANROOM_DIR / "configs" / "ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head.yaml"


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_canonical_name(main_col: str, supplemental_col: str, prefer_main_column: bool) -> str:
    if prefer_main_column:
        return main_col
    return max([main_col, supplemental_col], key=lambda item: (len(item), item))


def detect_alias_duplicate_pairs(main: pd.DataFrame, supplemental: pd.DataFrame, config: dict[str, object]) -> list[dict[str, object]]:
    dedup_cfg = config["identity_dedup"]
    min_shared_rows = int(dedup_cfg["min_shared_rows"])
    same_ratio_threshold = float(dedup_cfg["same_ratio_threshold"])
    atol = float(dedup_cfg["atol"])
    prefer_main_column = bool(dedup_cfg["prefer_main_column"])

    main_cols = [col for col in main.columns if col != "time"]
    supplemental_cols = [col for col in supplemental.columns if col != "time"]
    pairs: list[dict[str, object]] = []

    for supp_col in supplemental_cols:
        best_match: dict[str, object] | None = None
        supp_frame = supplemental[["time", supp_col]].dropna()
        if supp_frame.empty:
            continue
        for main_col in main_cols:
            aligned = main[["time", main_col]].merge(supp_frame, on="time", how="inner").dropna()
            if len(aligned) < min_shared_rows:
                continue
            same_mask = np.isclose(aligned[main_col].to_numpy(dtype=float), aligned[supp_col].to_numpy(dtype=float), rtol=0.0, atol=atol)
            same_ratio = float(np.mean(same_mask))
            if same_ratio < same_ratio_threshold:
                continue
            candidate = {
                "canonical_sensor": choose_canonical_name(main_col, supp_col, prefer_main_column),
                "main_sensor": main_col,
                "supplemental_sensor": supp_col,
                "shared_rows": int(len(aligned)),
                "same_ratio": same_ratio,
                "max_abs_diff": float(np.max(np.abs(aligned[main_col].to_numpy(dtype=float) - aligned[supp_col].to_numpy(dtype=float)))),
            }
            if best_match is None or candidate["shared_rows"] > best_match["shared_rows"]:
                best_match = candidate
        if best_match is not None:
            pairs.append(best_match)
    return pairs


def load_identity_deduped_combined_dcs(main_path: Path, supplemental_path: Path | None, config: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    main = load_dcs_data(main_path)
    audit = {
        "main_rows": int(len(main)),
        "main_sensor_count": int(len([col for col in main.columns if col != "time"])),
        "supplemental_used": False,
        "supplemental_rows": 0,
        "supplemental_sensor_count": 0,
        "shared_time_count": 0,
        "identity_dedup_applied": False,
        "alias_pair_count": 0,
        "dropped_alias_sensor_count": 0,
    }
    alias_frame = pd.DataFrame(columns=["canonical_sensor", "main_sensor", "supplemental_sensor", "shared_rows", "same_ratio", "max_abs_diff"])
    if supplemental_path is None or not supplemental_path.exists():
        audit["combined_rows"] = int(len(main))
        audit["combined_sensor_count"] = int(len([col for col in main.columns if col != "time"]))
        return main, audit, alias_frame

    supplemental = load_dcs_data(supplemental_path)
    alias_pairs = detect_alias_duplicate_pairs(main, supplemental, config)
    alias_frame = pd.DataFrame(alias_pairs)
    alias_supp_cols = sorted(alias_frame["supplemental_sensor"].unique().tolist()) if not alias_frame.empty else []
    kept_supp_cols = [col for col in supplemental.columns if col == "time" or col not in alias_supp_cols]
    supplemental_kept = supplemental[kept_supp_cols].copy()
    combined = main.merge(supplemental_kept, on="time", how="outer").sort_values("time").reset_index(drop=True)

    audit.update(
        {
            "supplemental_used": True,
            "supplemental_rows": int(len(supplemental)),
            "supplemental_sensor_count": int(len([col for col in supplemental.columns if col != "time"])),
            "shared_time_count": int(len(set(main["time"]) & set(supplemental["time"]))),
            "identity_dedup_applied": True,
            "alias_pair_count": int(len(alias_frame)),
            "dropped_alias_sensor_count": int(len(alias_supp_cols)),
            "dropped_alias_sensors": alias_supp_cols,
            "combined_rows": int(len(combined)),
            "combined_sensor_count": int(len([col for col in combined.columns if col != "time"])),
        }
    )
    return combined, audit, alias_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EWMA validation with strict sensor-identity de-duplication before screening.")
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

    result_rows: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_results: pd.DataFrame | None = None

    for tau in config["ewma"]["tau_grid"]:
        for window_minutes in config["ewma"]["window_grid"]:
            for lambda_value in config["ewma"]["lambda_grid"]:
                ewma_rows, ewma_audit = build_recursive_ewma_rows(
                    lims,
                    dcs,
                    sensors=candidate_sensors,
                    tau_minutes=int(tau),
                    window_minutes=int(window_minutes),
                    lambda_value=float(lambda_value),
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
                if len(common) < 500:
                    continue

                metrics_summary, detailed_results = evaluate_representation(
                    common,
                    baseline_feature_cols=baseline_feature_cols,
                    ewma_feature_cols=ewma_feature_cols,
                    config=config,
                )
                row = {
                    "tau_minutes": int(tau),
                    "window_minutes": int(window_minutes),
                    "lambda": float(lambda_value),
                    "common_samples": int(len(common)),
                    "baseline_macro_f1": metrics_summary["baseline_simple"]["macro_f1"],
                    "ewma_macro_f1": metrics_summary["ewma_treatment"]["macro_f1"],
                    "baseline_balanced_accuracy": metrics_summary["baseline_simple"]["balanced_accuracy"],
                    "ewma_balanced_accuracy": metrics_summary["ewma_treatment"]["balanced_accuracy"],
                    "baseline_core_AP": metrics_summary["baseline_simple"]["core_qualified_average_precision"],
                    "ewma_core_AP": metrics_summary["ewma_treatment"]["core_qualified_average_precision"],
                    "baseline_warning_AP": metrics_summary["baseline_simple"]["boundary_warning_average_precision"],
                    "ewma_warning_AP": metrics_summary["ewma_treatment"]["boundary_warning_average_precision"],
                    "baseline_boundary_high_conf_non_warning": metrics_summary["baseline_simple"]["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                    "ewma_boundary_high_conf_non_warning": metrics_summary["ewma_treatment"]["boundary_overconfidence"]["high_confidence_non_warning_rate"],
                    "ewma_any_violation_rate": metrics_summary["ewma_treatment"]["monotonicity"]["any_violation_rate"],
                    "mean_sensor_overlap_count": float(np.mean([fold["sensor_overlap_count"] for fold in metrics_summary["per_fold"]])),
                }
                result_rows.append(row)

                if best_summary is None or (
                    row["ewma_macro_f1"],
                    row["ewma_balanced_accuracy"],
                    row["ewma_warning_AP"],
                ) > (
                    best_summary["candidate"]["ewma_macro_f1"],
                    best_summary["candidate"]["ewma_balanced_accuracy"],
                    best_summary["candidate"]["ewma_warning_AP"],
                ):
                    best_summary = {
                        "candidate": row,
                        "metrics_summary": metrics_summary,
                        "ewma_audit": ewma_audit,
                    }
                    best_results = detailed_results.copy()

    if best_summary is None or best_results is None:
        raise ValueError("No valid EWMA combination was scored.")

    results_frame = pd.DataFrame(result_rows).sort_values(
        ["ewma_macro_f1", "ewma_balanced_accuracy", "ewma_warning_AP"],
        ascending=[False, False, False],
    )
    results_csv = artifacts_dir / "ordinal_cumulative_ewma_identity_dedup_results.csv"
    summary_json = artifacts_dir / "ordinal_cumulative_ewma_identity_dedup_summary.json"
    best_rows_csv = artifacts_dir / "ordinal_cumulative_ewma_identity_dedup_best_feature_rows.csv"
    alias_csv = artifacts_dir / "sensor_identity_alias_pairs.csv"
    audit_md = reports_dir / "ordinal_cumulative_ewma_identity_dedup_audit.md"

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
        "baseline_reference": {
            "lookback_minutes": int(config["baseline"]["lookback_minutes"]),
            "topk_sensors": int(config["screening"]["topk_sensors"]),
        },
        "screening_policy": {
            "baseline_sensor_subset": "selected from baseline simple-window features inside each training fold after sensor-identity de-dup",
            "ewma_sensor_subset": "selected from EWMA recursive features inside each training fold after sensor-identity de-dup",
        },
        "best_ewma_combination": best_summary["candidate"],
        "best_ewma_metrics_summary": best_summary["metrics_summary"],
        "best_ewma_audit": best_summary["ewma_audit"],
        "artifacts": {
            "results_csv": str(results_csv),
            "summary_json": str(summary_json),
            "best_feature_rows_csv": str(best_rows_csv),
            "alias_pairs_csv": str(alias_csv),
            "audit_md": str(audit_md),
        },
    }

    results_frame.to_csv(results_csv, index=False, encoding="utf-8-sig")
    best_results.to_csv(best_rows_csv, index=False, encoding="utf-8-sig")
    alias_frame.to_csv(alias_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    beat_both_count = int(
        (
            (results_frame["ewma_macro_f1"] > results_frame["baseline_macro_f1"])
            & (results_frame["ewma_balanced_accuracy"] > results_frame["baseline_balanced_accuracy"])
        ).sum()
    )
    best = best_summary["candidate"]
    lines = [
        "# Ordinal / Cumulative EWMA Identity-Dedup Audit",
        "",
        "- Task formulation: ordinal / cumulative current-head",
        "- Sensor identity de-dup: on",
        f"- Alias pairs dropped before feature building: {dcs_audit['alias_pair_count']}",
        f"- Dropped alias sensors: {dcs_audit['dropped_alias_sensor_count']}",
        f"- Combined sensor count after identity de-dup: {dcs_audit['combined_sensor_count']}",
        f"- beat_both_count: {beat_both_count}",
        "",
        "## Best EWMA Combination",
        "",
        json.dumps(best, ensure_ascii=False, indent=2),
    ]
    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
