from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
REPORT_DIR = ROOT_DIR / "reports"

RESULTS_PATH = ARTIFACT_DIR / "tabular_framework_validation_soft_probability_feature_distillation_results.csv"
SUMMARY_PATH = ARTIFACT_DIR / "tabular_framework_validation_soft_probability_feature_distillation_summary.json"
OUTPUT_PNG = ARTIFACT_DIR / "range_position_full_test_performance_dashboard.png"
OUTPUT_MD = REPORT_DIR / "range_position_full_test_performance_note.md"


def load_data() -> tuple[pd.DataFrame, dict]:
    results = pd.read_csv(RESULTS_PATH)
    with SUMMARY_PATH.open("r", encoding="utf-8") as stream:
        summary_rows = json.load(stream)

    summary_lookup = {row["variant_name"]: row for row in summary_rows}
    target = results[results["variant_name"] == "range_position_full"].copy()
    if target.empty:
        raise ValueError("No range_position_full rows found in results file.")
    return target, summary_lookup["range_position_full"]


def build_dashboard(target: pd.DataFrame, summary_row: dict) -> None:
    baseline = target[target["framework"] == "simple_baseline_stage2_soft_probability"].sort_values("fold")
    autogluon = target[target["framework"] == "autogluon_stage2_soft_probability"].sort_values("fold")

    folds = baseline["fold"].to_numpy(dtype=int)
    x = np.arange(len(folds))
    width = 0.36

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("range_position_full Test-Fold Performance", fontsize=16, fontweight="bold")

    metric_specs = [
        ("soft_brier", "Soft Brier", "lower is better", axes[0, 0]),
        ("soft_mae", "Soft MAE", "lower is better", axes[0, 1]),
        ("hard_out_ap_diagnostic", "Hard-Out AP (Diagnostic)", "higher is better", axes[1, 0]),
    ]

    for metric_name, title, note, ax in metric_specs:
        baseline_values = baseline[metric_name].to_numpy(dtype=float)
        ag_values = autogluon[metric_name].to_numpy(dtype=float)
        ax.bar(x - width / 2, baseline_values, width=width, label="Baseline", color="#B0BEC5")
        ax.bar(x + width / 2, ag_values, width=width, label="AutoGluon", color="#1565C0")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {fold}" for fold in folds], rotation=0)
        ax.set_title(f"{title}\n({note})")
        ax.set_ylabel(title)
        ax.legend(loc="best")

    ax_text = axes[1, 1]
    ax_text.axis("off")

    mean_brier_gain = summary_row["baseline_mean_soft_brier"] - summary_row["autogluon_mean_soft_brier"]
    mean_mae_gain = summary_row["baseline_mean_soft_mae"] - summary_row["autogluon_mean_soft_mae"]
    mean_ap_gain = (
        summary_row["autogluon_mean_hard_out_ap_diagnostic"]
        - summary_row["baseline_mean_hard_out_ap_diagnostic"]
    )
    positive_signal = "Yes" if summary_row["positive_signal"] else "No"

    text = "\n".join(
        [
            "Mean Test-Fold Summary",
            "",
            f"Feature variant: range_position_full",
            f"Feature count: {summary_row['feature_count_after_distillation_variant']}",
            "",
            f"Baseline mean Soft Brier: {summary_row['baseline_mean_soft_brier']:.4f}",
            f"AutoGluon mean Soft Brier: {summary_row['autogluon_mean_soft_brier']:.4f}",
            f"Brier improvement: {mean_brier_gain:.4f}",
            "",
            f"Baseline mean Soft MAE: {summary_row['baseline_mean_soft_mae']:.4f}",
            f"AutoGluon mean Soft MAE: {summary_row['autogluon_mean_soft_mae']:.4f}",
            f"MAE improvement: {mean_mae_gain:.4f}",
            "",
            f"Baseline mean Hard-Out AP: {summary_row['baseline_mean_hard_out_ap_diagnostic']:.4f}",
            f"AutoGluon mean Hard-Out AP: {summary_row['autogluon_mean_hard_out_ap_diagnostic']:.4f}",
            f"AP improvement: {mean_ap_gain:.4f}",
            "",
            f"Positive signal: {positive_signal}",
            "",
            "Reading guide:",
            "- Soft metrics are the primary decision metrics.",
            "- Hard-Out AP is diagnostic only.",
            "- Each fold is a held-out future time slice.",
        ]
    )
    ax_text.text(0.02, 0.98, text, va="top", ha="left", fontsize=11, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_note(summary_row: dict) -> None:
    brier_gain = summary_row["baseline_mean_soft_brier"] - summary_row["autogluon_mean_soft_brier"]
    mae_gain = summary_row["baseline_mean_soft_mae"] - summary_row["autogluon_mean_soft_mae"]
    ap_gain = (
        summary_row["autogluon_mean_hard_out_ap_diagnostic"]
        - summary_row["baseline_mean_hard_out_ap_diagnostic"]
    )
    lines = [
        "# range_position_full Test Performance Note",
        "",
        "## What This Figure Shows",
        "",
        "- The figure focuses only on the current best feature variant: `range_position_full`.",
        "- Each fold is a held-out future test slice from time-ordered validation.",
        "- The comparison is always `simple baseline` versus `AutoGluon` on the same fold.",
        "",
        "## Mean Results",
        "",
        f"- Baseline mean Soft Brier: `{summary_row['baseline_mean_soft_brier']:.4f}`",
        f"- AutoGluon mean Soft Brier: `{summary_row['autogluon_mean_soft_brier']:.4f}`",
        f"- Soft Brier improvement: `{brier_gain:.4f}`",
        f"- Baseline mean Soft MAE: `{summary_row['baseline_mean_soft_mae']:.4f}`",
        f"- AutoGluon mean Soft MAE: `{summary_row['autogluon_mean_soft_mae']:.4f}`",
        f"- Soft MAE improvement: `{mae_gain:.4f}`",
        f"- Baseline mean Hard-Out AP diagnostic: `{summary_row['baseline_mean_hard_out_ap_diagnostic']:.4f}`",
        f"- AutoGluon mean Hard-Out AP diagnostic: `{summary_row['autogluon_mean_hard_out_ap_diagnostic']:.4f}`",
        f"- Hard-Out AP improvement: `{ap_gain:.4f}`",
        "",
        "## Decision",
        "",
        "- This is a positive result overall because the primary soft metrics improve consistently enough to justify keeping the current soft-label line.",
        "- The gain is real but not dramatic, so the current conclusion is `promising and worth continuing`, not `finished` or `fully optimized`.",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    target, summary_row = load_data()
    build_dashboard(target, summary_row)
    write_note(summary_row)
    print(
        json.dumps(
            {
                "output_png": str(OUTPUT_PNG),
                "output_note": str(OUTPUT_MD),
                "variant_name": "range_position_full",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
