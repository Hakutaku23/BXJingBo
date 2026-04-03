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
DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage2_soft_probability_tuning.yaml"


def make_variant_configs(base_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[int, float, int]] = set()

    lookback_base = int(base_config["snapshot"]["lookback_minutes"])
    softness_base = float(base_config["label_fuzziness"]["boundary_softness"])
    topk_base = int(base_config["selection"]["soft_probability_top_k"])

    def add_variant(name: str, lookback: int, softness: float, topk: int) -> None:
        key = (int(lookback), float(softness), int(topk))
        if key in seen:
            return
        seen.add(key)
        config = copy.deepcopy(base_config)
        config["snapshot"]["lookback_minutes"] = int(lookback)
        config["label_fuzziness"]["boundary_softness"] = float(softness)
        config["selection"]["soft_probability_top_k"] = int(topk)
        variants.append((name, config))

    add_variant("baseline", lookback_base, softness_base, topk_base)
    for lookback in base_config["tuning"]["vary_lookback_minutes"]:
        add_variant(f"lookback_{int(lookback)}", int(lookback), softness_base, topk_base)
    for softness in base_config["tuning"]["vary_boundary_softness"]:
        add_variant(f"softness_{float(softness):.2f}", lookback_base, float(softness), topk_base)
    for topk in base_config["tuning"]["vary_top_k"]:
        add_variant(f"topk_{int(topk)}", lookback_base, softness_base, int(topk))
    return variants


def summarize_variant(name: str, config: dict[str, Any], stage_summary: dict[str, Any], snapshot_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant_name": name,
        "lookback_minutes": int(config["snapshot"]["lookback_minutes"]),
        "boundary_softness": float(config["label_fuzziness"]["boundary_softness"]),
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


def write_tuning_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    best_row = min(rows, key=lambda row: row["autogluon_mean_soft_brier"])
    lines = [
        "# Tabular Framework Validation Audit - Stage 2 Soft Probability Tuning",
        "",
        "## Purpose",
        "",
        "- Controlled local tuning around the already validated soft probability branch.",
        "- Only three knobs are varied: lookback, boundary_softness, and top_k.",
        "- The starting source remains uncleaned-source data with the same stage-2 leakage-controlled workflow.",
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
    parser = argparse.ArgumentParser(description="Run controlled tuning for AutoGluon soft probability branch.")
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
    variant_results: list[pd.DataFrame] = []
    for variant_name, variant_config in make_variant_configs(config):
        results, stage_summary, snapshot_audit, _ = run_soft_probability_stage2(config_path, variant_config)
        result_row = summarize_variant(variant_name, variant_config, stage_summary, snapshot_audit)
        rows.append(result_row)
        variant_results.append(
            results.assign(
                variant_name=variant_name,
                tuned_lookback_minutes=int(variant_config["snapshot"]["lookback_minutes"]),
                tuned_boundary_softness=float(variant_config["label_fuzziness"]["boundary_softness"]),
                tuned_top_k=int(variant_config["selection"]["soft_probability_top_k"]),
            )
        )

    summary_df = pd.DataFrame(rows).sort_values(
        by=["autogluon_mean_soft_brier", "autogluon_mean_soft_mae", "variant_name"],
        ascending=[True, True, True],
    )
    detailed_df = pd.concat(variant_results, ignore_index=True)

    summary_path = artifact_dir / "tabular_framework_validation_soft_probability_tuning_summary.csv"
    details_path = artifact_dir / "tabular_framework_validation_soft_probability_tuning_results.csv"
    json_path = artifact_dir / "tabular_framework_validation_soft_probability_tuning_summary.json"
    audit_path = report_dir / "tabular_framework_validation_soft_probability_tuning_audit.md"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detailed_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_df.to_dict(orient="records"), stream, ensure_ascii=False, indent=2)
    write_tuning_audit(audit_path, summary_df.to_dict(orient="records"))
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
