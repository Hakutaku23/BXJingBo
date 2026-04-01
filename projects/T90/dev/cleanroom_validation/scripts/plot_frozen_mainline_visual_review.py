from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


THIS_DIR = Path(__file__).resolve().parent
CLEANROOM_DIR = THIS_DIR.parent
ARTIFACTS_DIR = (
    CLEANROOM_DIR
    / "artifacts"
    / "monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup"
)
REPORTS_DIR = CLEANROOM_DIR / "reports" / "frozen_mainline_visual_review"

SUMMARY_PATH = ARTIFACTS_DIR / "boundary_weighted_inner_thresholds_summary.json"
RESULTS_PATH = ARTIFACTS_DIR / "boundary_weighted_inner_thresholds_results.csv"
PER_FOLD_PATH = ARTIFACTS_DIR / "boundary_weighted_inner_thresholds_per_fold.csv"

FIGURE_PATH = REPORTS_DIR / "frozen_mainline_visual_review.png"
REPORT_PATH = REPORTS_DIR / "frozen_mainline_online_readiness.md"

LABELS = ["acceptable", "warning", "unacceptable"]


def configure_matplotlib() -> None:
    available_fonts = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for candidate in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        if candidate in available_fonts:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def load_inputs() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    with SUMMARY_PATH.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    results = pd.read_csv(RESULTS_PATH)
    per_fold = pd.read_csv(PER_FOLD_PATH)
    return summary, results, per_fold


def boundary_profile(frame: pd.DataFrame, *, prefix: str) -> dict[str, object]:
    boundary = frame.loc[to_bool(frame["boundary_any_flag"])].copy()
    pred_col = f"{prefix}_business_pred"
    prob_cols = [f"{prefix}_business_prob_{label}" for label in LABELS]
    label_share = boundary[pred_col].value_counts(normalize=True).reindex(LABELS, fill_value=0.0)
    confidence = boundary[prob_cols].max(axis=1)
    high_conf_non_warning = ((boundary[pred_col] != "warning") & (confidence >= 0.6)).sum()
    return {
        "samples": int(len(boundary)),
        "label_share": label_share.to_dict(),
        "high_conf_non_warning_count": int(high_conf_non_warning),
        "high_conf_non_warning_rate": float(high_conf_non_warning / len(boundary)) if len(boundary) else 0.0,
    }


def build_figure(summary: dict[str, object], results: pd.DataFrame, per_fold: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    default_summary = summary["default_summary"]
    weighted_summary = summary["weighted_summary"]

    metrics = [
        ("macro_f1", "Macro-F1"),
        ("balanced_accuracy", "Balanced Acc"),
        ("core_qualified_average_precision", "Core AP"),
        ("boundary_warning_average_precision", "Warning AP"),
        ("clearly_unacceptable_average_precision", "Unacceptable AP"),
    ]

    x = np.arange(len(metrics))
    width = 0.36

    per_fold = per_fold.copy()
    per_fold["fold"] = per_fold["fold"].astype(int)
    per_fold["macro_gain"] = per_fold["weighted_macro_f1"] - per_fold["default_macro_f1"]
    per_fold["balanced_gain"] = per_fold["weighted_balanced_accuracy"] - per_fold["default_balanced_accuracy"]
    per_fold["boundary_caution_gain"] = (
        per_fold["default_boundary_high_conf_non_warning_rate"]
        - per_fold["weighted_boundary_high_conf_non_warning_rate"]
    )

    confusion = confusion_matrix(
        results["business_label"],
        results["weighted_business_pred"],
        labels=LABELS,
        normalize="true",
    )
    confusion_counts = confusion_matrix(
        results["business_label"],
        results["weighted_business_pred"],
        labels=LABELS,
        normalize=None,
    )

    default_boundary = boundary_profile(results, prefix="default")
    weighted_boundary = boundary_profile(results, prefix="weighted")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor("#f7f4ef")
    for ax in axes.flat:
        ax.set_facecolor("#fffdf9")

    ax = axes[0, 0]
    ax.bar(
        x - width / 2,
        [default_summary[key] for key, _ in metrics],
        width=width,
        color="#8ea4b8",
        label="default",
    )
    ax.bar(
        x + width / 2,
        [weighted_summary[key] for key, _ in metrics],
        width=width,
        color="#c96f4a",
        label="frozen mainline",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=15, ha="right")
    ax.set_ylim(0.0, 0.8)
    ax.set_title("Pooled Metrics")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    bar_x = np.arange(len(per_fold))
    bar_w = 0.22
    ax.axhline(0.0, color="#444444", linewidth=1)
    ax.bar(bar_x - bar_w, per_fold["macro_gain"], width=bar_w, color="#cf7c47", label="Macro-F1 gain")
    ax.bar(bar_x, per_fold["balanced_gain"], width=bar_w, color="#3f7f93", label="Balanced Acc gain")
    ax.bar(
        bar_x + bar_w,
        per_fold["boundary_caution_gain"],
        width=bar_w,
        color="#7f9c5e",
        label="Boundary caution gain",
    )
    ax.set_xticks(bar_x)
    ax.set_xticklabels([f"Fold {fold}" for fold in per_fold["fold"]])
    ax.set_title("Fold-Wise Gain vs Default")
    ax.set_ylabel("positive is better")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    image = ax.imshow(confusion, cmap="YlOrBr", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Frozen Mainline Confusion Matrix")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j,
                i,
                f"{confusion[i, j]:.2f}\n({confusion_counts[i, j]})",
                ha="center",
                va="center",
                color="#1f1f1f",
                fontsize=10,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    stack_x = np.arange(2)
    bottom = np.zeros(2)
    colors = {
        "acceptable": "#3f7f93",
        "warning": "#d5a021",
        "unacceptable": "#b55239",
    }
    profiles = [default_boundary, weighted_boundary]
    names = ["default", "frozen mainline"]
    for label in LABELS:
        values = [profile["label_share"][label] for profile in profiles]
        ax.bar(stack_x, values, bottom=bottom, color=colors[label], label=label)
        bottom += np.array(values)
    ax.set_xticks(stack_x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Boundary Samples: Predicted Label Share")
    ax.set_ylabel("share within boundary samples")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    for idx, profile in enumerate(profiles):
        ax.text(
            idx,
            1.02,
            f"HC non-warning\n{profile['high_conf_non_warning_rate']:.3f}\n({profile['high_conf_non_warning_count']}/{profile['samples']})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "T90 Frozen Mainline Visual Review\nsimple 120min + monotonic cumulative + identity de-dup + inner-threshold hard weighting",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict[str, object], results: pd.DataFrame, per_fold: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    weighted_summary = summary["weighted_summary"]
    default_summary = summary["default_summary"]
    weighted_fold = summary["weighted_fold_mean"]

    boundary = results.loc[to_bool(results["boundary_any_flag"])].copy()
    actual_warning_boundary = (boundary["business_label"] == "warning").sum()
    predicted_warning_boundary = (boundary["weighted_business_pred"] == "warning").sum()
    confidence = boundary[[f"weighted_business_prob_{label}" for label in LABELS]].max(axis=1)
    high_conf_non_warning = ((boundary["weighted_business_pred"] != "warning") & (confidence >= 0.6)).sum()

    confusion_counts = confusion_matrix(
        results["business_label"],
        results["weighted_business_pred"],
        labels=LABELS,
        normalize=None,
    )
    acceptable_recall = confusion_counts[0, 0] / confusion_counts[0].sum()
    warning_recall = confusion_counts[1, 1] / confusion_counts[1].sum()
    unacceptable_recall = confusion_counts[2, 2] / confusion_counts[2].sum()

    lines = [
        "# Frozen Mainline Online Readiness",
        "",
        "## Visual Artifact",
        "",
        f"- Figure: `{FIGURE_PATH}`",
        "",
        "## Key Readout",
        "",
        f"- weighted `macro_f1 = {weighted_summary['macro_f1']:.4f}` vs default `{default_summary['macro_f1']:.4f}`",
        f"- weighted `balanced_accuracy = {weighted_summary['balanced_accuracy']:.4f}` vs default `{default_summary['balanced_accuracy']:.4f}`",
        f"- weighted `warning_AP = {weighted_summary['boundary_warning_average_precision']:.4f}`",
        f"- weighted `unacceptable_AP = {weighted_summary['clearly_unacceptable_average_precision']:.4f}`",
        f"- boundary high-confidence non-warning = {weighted_summary['boundary_overconfidence']['high_confidence_non_warning_rate']:.4f}` ({weighted_summary['boundary_overconfidence']['high_confidence_non_warning_count']}/{weighted_summary['boundary_overconfidence']['samples']})",
        f"- fold-mean `macro_f1 = {weighted_fold['macro_f1']:.4f}`, fold-mean `balanced_accuracy = {weighted_fold['balanced_accuracy']:.4f}`",
        "",
        "## Confusion-Matrix Readout",
        "",
        f"- acceptable recall: `{acceptable_recall:.4f}`",
        f"- warning recall: `{warning_recall:.4f}`",
        f"- unacceptable recall: `{unacceptable_recall:.4f}`",
        f"- boundary samples actual warning count: `{int(actual_warning_boundary)}`",
        f"- boundary samples predicted warning count: `{int(predicted_warning_boundary)}`",
        f"- boundary high-confidence non-warning count: `{int(high_conf_non_warning)}`",
        "",
        "## Assessment",
        "",
        "结论：",
        "当前方法**适合进入现场旁路在线测试（shadow / human-in-the-loop）**，但**还不适合直接进入无人值守的自动在线判级或联动控制**。",
        "",
        "理由：",
        "1. 它已经具备进入现场试运行的最低研究条件：",
        "   - 冻结主线清晰，数据口径固定，严格做过 identity de-dup。",
        "   - 概率输出满足单调性，当前 cleanroom 里没有 monotonicity violation。",
        "   - 相对默认基线，它在 pooled 指标和边界谨慎性上有稳定的小幅提升。",
        "2. 但它还没有达到可直接上线替代人工判断的可靠性：",
        "   - 绝对指标仍偏低，尤其 `warning_AP` 和 `unacceptable_AP` 仍不足以支持高风险自动判定。",
        "   - 边界样本里仍有接近一半会被高置信地判成非 `warning`，这与“边界模糊、需谨慎处理”的现场需求仍有张力。",
        "   - 折间稳定性不够强，存在明显弱折，说明对时间段变化的鲁棒性还不够。",
        "3. 因而更合理的现场策略是：",
        "   - 先做只读旁路在线测试，实时出分但不驱动控制。",
        "   - 由工艺员或质检人员对照查看预测、概率和实际化验结果。",
        "   - 在线阶段重点观察：边界批次、明显不合格批次、缺失/漂移点位、以及报警解释是否可读。",
        "",
        "## Recommendation",
        "",
        "- `Go` for shadow online test.",
        "- `No-Go` for autonomous online deployment.",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary, results, per_fold = load_inputs()
    build_figure(summary, results, per_fold)
    write_report(summary, results, per_fold)
    print(
        json.dumps(
            {
                "figure_path": str(FIGURE_PATH),
                "report_path": str(REPORT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
