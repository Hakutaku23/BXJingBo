from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_autogluon_stage2_feature_engineering import load_config, resolve_path
from run_autogluon_stage2_soft_probability import run_soft_probability_stage2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage2_soft_probability_label_family.yaml"


def make_variants(base_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = []
    for rule in base_config["label_family_tuning"]["rules"]:
        for softness in base_config["label_family_tuning"]["boundary_softness_values"]:
            config = copy.deepcopy(base_config)
            config["label_fuzziness"]["rule"] = str(rule)
            config["label_fuzziness"]["boundary_softness"] = float(softness)
            name = f"{rule}_s{float(softness):.2f}"
            variants.append((name, config))
    return variants


def summarize_variant(name: str, config: dict[str, Any], stage_summary: dict[str, Any], snapshot_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant_name": name,
        "rule": str(config["label_fuzziness"]["rule"]),
        "boundary_softness": float(config["label_fuzziness"]["boundary_softness"]),
        "lookback_minutes": int(config["snapshot"]["lookback_minutes"]),
        "top_k": int(config["selection"]["soft_probability_top_k"]),
        "feature_count_after_global_cleaning": int(snapshot_audit["feature_count_after_global_cleaning"]),
        "baseline_mean_soft_mae": float(stage_summary["baseline_mean_soft_mae"]),
        "autogluon_mean_soft_mae": float(stage_summary["autogluon_mean_soft_mae"]),
        "baseline_mean_soft_brier": float(stage_summary["baseline_mean_soft_brier"]),
        "autogluon_mean_soft_brier": float(stage_summary["autogluon_mean_soft_brier"]),
        "baseline_mean_hard_out_ap_diagnostic": float(stage_summary["baseline_mean_hard_out_ap_diagnostic"]),
        "autogluon_mean_hard_out_ap_diagnostic": float(stage_summary["autogluon_mean_hard_out_ap_diagnostic"]),
        "positive_signal": bool(stage_summary["positive_signal"]),
    }


def write_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    best_row = min(rows, key=lambda row: row["autogluon_mean_soft_brier"])
    lines = [
        "# Tabular Framework Validation Audit - Soft Probability Label Family Comparison",
        "",
        "## Purpose",
        "",
        "- Compare multiple fuzzy label mapping families while holding X-side engineering fixed.",
        "- Focus on whether the current logistic boundary mapping is already appropriate.",
        "",
        "## Candidate Summary",
        "",
        json.dumps(rows, ensure_ascii=False, indent=2),
        "",
        "## Best Candidate By AutoGluon Soft Brier",
        "",
        json.dumps(best_row, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run label-family comparison for AutoGluon soft probability branch.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    report_dir = resolve_path(config_path.parent, config["paths"]["report_dir"])
    if artifact_dir is None or report_dir is None:
        raise ValueError("Both artifact_dir and report_dir must be configured.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    detailed_rows: list[pd.DataFrame] = []
    for variant_name, variant_config in make_variants(config):
        results, stage_summary, snapshot_audit, _ = run_soft_probability_stage2(config_path, variant_config)
        rows.append(summarize_variant(variant_name, variant_config, stage_summary, snapshot_audit))
        detailed_rows.append(
            results.assign(
                variant_name=variant_name,
                label_rule=str(variant_config["label_fuzziness"]["rule"]),
                tuned_boundary_softness=float(variant_config["label_fuzziness"]["boundary_softness"]),
            )
        )

    summary_df = pd.DataFrame(rows).sort_values(
        by=["autogluon_mean_soft_brier", "autogluon_mean_soft_mae", "variant_name"],
        ascending=[True, True, True],
    )
    details_df = pd.concat(detailed_rows, ignore_index=True)

    summary_path = artifact_dir / "tabular_framework_validation_soft_probability_label_family_summary.csv"
    details_path = artifact_dir / "tabular_framework_validation_soft_probability_label_family_results.csv"
    json_path = artifact_dir / "tabular_framework_validation_soft_probability_label_family_summary.json"
    audit_path = report_dir / "tabular_framework_validation_soft_probability_label_family_audit.md"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_df.to_dict(orient="records"), stream, ensure_ascii=False, indent=2)
    write_audit(audit_path, summary_df.to_dict(orient="records"))
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "details_path": str(details_path),
                "json_path": str(json_path),
                "audit_path": str(audit_path),
                "best_variant": summary_df.iloc[0].to_dict(),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
