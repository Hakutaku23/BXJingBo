from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_autogluon_stage2_feature_engineering import (
    load_config,
    resolve_path,
    run_stage2,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "autogluon_stage2_high_risk.yaml"


def write_high_risk_audit(path: Path, config: dict, stage_summary: dict, snapshot_audit: dict) -> None:
    lines = [
        "# Tabular Framework Validation Audit - Stage 2 High-Risk Branch",
        "",
        "## Starting Source Condition",
        "",
        "- The starting source is currently an uncleaned source dataset.",
        "- Uncleaned refers to source state, not a ban on preprocessing.",
        "",
        "## Branch Purpose",
        "",
        "- This branch is split out from the mixed stage-2 line.",
        "- It only keeps the high-risk binary task for further optimization.",
        "",
        "## Label Type",
        "",
        "- The target is a hard binary classification label.",
        "- Definition: target_high_risk = 1 if T90 > 8.7 else 0.",
        "",
        "## X-side Feature Engineering",
        "",
        "- causal 120-minute snapshot",
        "- stats: mean, std, min, max, last, delta, range, slope, valid_ratio",
        "- drop constant features",
        "- drop high-missing features",
        "- drop near-duplicate high-correlation features",
        "- fold-internal supervised feature selection",
        "",
        "## Baseline Definition",
        "",
        "- Baseline is not raw-rule thresholding and not a no-feature baseline.",
        "- Baseline is a simple logistic-regression pipeline on the same engineered X, same split, same selected features.",
        "",
        "## Stage Conclusions",
        "",
        json.dumps(stage_summary, ensure_ascii=False, indent=2),
        "",
        "## Snapshot Audit",
        "",
        json.dumps(snapshot_audit, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoGluon stage 2 high-risk-only validation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_dir = resolve_path(config_path.parent, config["paths"]["artifact_dir"])
    report_dir = resolve_path(config_path.parent, config["paths"]["report_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results, stage_summary, snapshot_audit = run_stage2(config_path, config)
    results_path = artifact_dir / "tabular_framework_validation_high_risk_results.csv"
    summary_path = artifact_dir / "tabular_framework_validation_high_risk_summary.json"
    audit_path = report_dir / "tabular_framework_validation_high_risk_audit.md"

    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "framework": "AutoGluon",
                "phase": "stage2_high_risk_only",
                "stage_summary": stage_summary,
                "snapshot_audit": snapshot_audit,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    write_high_risk_audit(audit_path, config, stage_summary, snapshot_audit)
    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "audit_path": str(audit_path),
                "stage_summary": stage_summary,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
