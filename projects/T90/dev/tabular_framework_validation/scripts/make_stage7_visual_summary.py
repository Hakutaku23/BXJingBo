from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
SUMMARY_PATH = ARTIFACTS / "stage7_final_summary.json"
OUTPUT_PNG = ARTIFACTS / "stage7_final_visual_dashboard.png"
OUTPUT_MD = REPORTS / "stage7_final_visual_note.md"


def _load_summary() -> list[dict]:
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _task_map(summary: list[dict]) -> dict[str, dict]:
    return {row["task_name"]: row for row in summary}


def _extract_fold_values(task: dict, metric_key: str) -> tuple[list[float], list[float]]:
    baseline = [fold["baseline"][metric_key] for fold in task["fold_summaries"]]
    autogluon = [fold["autogluon"][metric_key] for fold in task["fold_summaries"]]
    return baseline, autogluon


def _build_dashboard(centered: dict, five_bin: dict) -> None:
    centered_baseline, centered_ag = _extract_fold_values(centered, "mae")
    five_baseline, five_ag = _extract_fold_values(five_bin, "multiclass_log_loss")

    folds = np.arange(1, 6)
    width = 0.36

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Fold-wise centered_desirability
    ax1.bar(folds - width / 2, centered_baseline, width=width, label="Baseline", color="#b0b7c3")
    ax1.bar(folds + width / 2, centered_ag, width=width, label="AutoGluon", color="#2f6fed")
    ax1.set_title("Centered Desirability\nHeld-out Fold MAE")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("MAE (lower is better)")
    ax1.set_xticks(folds)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)

    # Fold-wise five_bin
    ax2.bar(folds - width / 2, five_baseline, width=width, label="Baseline", color="#b0b7c3")
    ax2.bar(folds + width / 2, five_ag, width=width, label="AutoGluon", color="#2f6fed")
    ax2.set_title("Five-Bin\nHeld-out Fold Multiclass Log Loss")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Log loss (lower is better)")
    ax2.set_xticks(folds)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)

    # Stage0 vs final
    centered_stage0 = centered["stage0_autogluon_ref"]
    centered_final = centered["autogluon_mean_mae"]
    five_stage0 = five_bin["stage0_autogluon_ref"]
    five_final = five_bin["autogluon_mean_multiclass_log_loss"]

    labels = ["Stage0", "Final S7"]
    ax3.bar([0, 1], [centered_stage0, centered_final], color=["#d8dde6", "#2f6fed"], width=0.55)
    ax3.set_xticks([0, 1], labels)
    ax3.set_title("Centered Desirability\nStage0 vs Final S7")
    ax3.set_ylabel("MAE (lower is better)")
    ax3.grid(axis="y", alpha=0.25)
    for x, y in zip([0, 1], [centered_stage0, centered_final]):
        ax3.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=10)

    ax4.bar([0, 1], [five_stage0, five_final], color=["#d8dde6", "#2f6fed"], width=0.55)
    ax4.set_xticks([0, 1], labels)
    ax4.set_title("Five-Bin\nStage0 vs Final S7")
    ax4.set_ylabel("Multiclass log loss (lower is better)")
    ax4.grid(axis="y", alpha=0.25)
    for x, y in zip([0, 1], [five_stage0, five_final]):
        ax4.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle(
        "T90 Tabular Validation: Final S7 Dashboard\n"
        "Held-out performance of the best current solution",
        fontsize=16,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.01,
        "Final recipes: centered_desirability = lag120_win60 + flow_balance + combined_quality | "
        "five_bin = lag120_win120",
        ha="center",
        fontsize=10,
    )

    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_note(centered: dict, five_bin: dict) -> None:
    note = f"""# Stage7 Visual Note

- image:
  - [stage7_final_visual_dashboard.png]({OUTPUT_PNG.as_posix()})
- purpose:
  - provide an easy-to-read summary of the current best solution on held-out folds
- panels:
  - top-left:
    - fold-wise MAE for `centered_desirability`
    - gray = baseline
    - blue = AutoGluon final S7
    - lower is better
  - top-right:
    - fold-wise multiclass log loss for `five_bin`
    - gray = baseline
    - blue = AutoGluon final S7
    - lower is better
  - bottom-left:
    - `centered_desirability` stage0 vs final S7
    - shows whether the final recipe really beats the stage0 reference
  - bottom-right:
    - `five_bin` stage0 vs final S7
    - shows whether the final recipe really beats the stage0 reference
- key takeaways:
  - centered final recipe:
    - `{centered["combo_name"]}`
    - top_k = `{centered["top_k"]}`
    - mean MAE = `{centered["autogluon_mean_mae"]:.4f}`
  - five_bin final recipe:
    - `{five_bin["combo_name"]}`
    - top_k = `{five_bin["top_k"]}`
    - mean multiclass log loss = `{five_bin["autogluon_mean_multiclass_log_loss"]:.4f}`
"""
    OUTPUT_MD.write_text(note, encoding="utf-8")


def main() -> None:
    summary = _load_summary()
    tasks = _task_map(summary)
    centered = tasks["centered_desirability"]
    five_bin = tasks["five_bin"]
    _build_dashboard(centered, five_bin)
    _write_note(centered, five_bin)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
