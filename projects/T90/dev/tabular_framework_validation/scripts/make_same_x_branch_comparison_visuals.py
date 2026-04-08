from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_centered_same_fe_as_soft import build_centered_on_soft_reference
from run_autogluon_soft_probability_weak_compression_search import build_variant_snapshot
from run_autogluon_stage2_feature_engineering import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    resolve_path,
    select_features_fold,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
DEFAULT_CENTERED_CONFIG = WORKSPACE_DIR / "configs" / "autogluon_centered_same_fe_as_soft.yaml"
DEFAULT_SOFT_CONFIG = WORKSPACE_DIR / "configs" / "autogluon_soft_probability_weak_compression_search.yaml"


def _soft_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    truth = np.asarray(y_true, dtype=float)
    return {
        "mae": float(np.mean(np.abs(truth - pred))),
        "brier": float(np.mean((truth - pred) ** 2)),
    }


def _centered_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(y_pred, dtype=float)
    truth = np.asarray(y_true, dtype=float)
    return {
        "mae": float(np.mean(np.abs(truth - pred))),
        "rmse": float(np.sqrt(np.mean((truth - pred) ** 2))),
    }


def collect_centered_oof(centered_config_path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    config = load_config(centered_config_path)
    frame, _ = build_centered_on_soft_reference(centered_config_path, config)
    feature_columns = [column for column in frame.columns if "__" in column]
    label = "target_centered_desirability"
    top_k = int(config["selection"]["top_k"])
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    artifact_dir = resolve_path(centered_config_path.parent, config["paths"]["artifact_dir"])
    if artifact_dir is None:
        raise ValueError("centered artifact_dir must be configured")
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, float | int | str]] = []
    fold_metric_rows: list[dict[str, float | int]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(frame), start=1):
        train = frame.iloc[train_idx].copy().reset_index(drop=True)
        test = frame.iloc[test_idx].copy().reset_index(drop=True)
        selected_features, _ = select_features_fold(
            train_x=train[feature_columns],
            train_y=train[label],
            task_type="regression",
            top_k=top_k,
        )
        train_df = train[selected_features + [label]].copy()
        test_df = test[selected_features + [label]].copy()

        baseline = make_regression_baseline()
        baseline.fit(train_df[selected_features], train_df[label].to_numpy(dtype=float))
        baseline_pred = baseline.predict(test_df[selected_features]).astype(float)

        model_path = artifact_dir / f"ag_same_x_visual_centered_{run_id}_fold{fold_idx}"
        ag_pred, model_best = fit_autogluon_fold(
            train_df=train_df,
            test_df=test_df,
            label=label,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )

        base_metrics = _centered_metrics(test_df[label].to_numpy(dtype=float), baseline_pred)
        ag_metrics = _centered_metrics(test_df[label].to_numpy(dtype=float), ag_pred)
        fold_metric_rows.append(
            {
                "task": "centered_desirability",
                "fold": int(fold_idx),
                "baseline_mae": base_metrics["mae"],
                "autogluon_mae": ag_metrics["mae"],
            }
        )

        for idx in range(len(test_df)):
            rows.append(
                {
                    "task": "centered_desirability",
                    "fold": int(fold_idx),
                    "sample_time": str(pd.Timestamp(test.loc[idx, "sample_time"])),
                    "t90": float(test.loc[idx, "t90"]),
                    "true_label": float(test.loc[idx, label]),
                    "baseline_pred": float(baseline_pred[idx]),
                    "autogluon_pred": float(ag_pred[idx]),
                    "best_model": str(model_best),
                }
            )

    oof = pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    summary = {
        "baseline_mean_mae": float(fold_metrics["baseline_mae"].mean()),
        "autogluon_mean_mae": float(fold_metrics["autogluon_mae"].mean()),
        "relative_improvement_pct": float(
            (fold_metrics["baseline_mae"].mean() - fold_metrics["autogluon_mae"].mean())
            / fold_metrics["baseline_mae"].mean()
            * 100.0
        ),
    }
    return oof, summary


def collect_soft_oof(soft_config_path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    config = load_config(soft_config_path)
    frame, _ = build_variant_snapshot(soft_config_path, config, "current_whole_window_ref")
    feature_columns = [column for column in frame.columns if "__" in column]
    label = str(config["label_fuzziness"]["target_name"])
    top_k = int(config["selection"]["soft_probability_top_k"])
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    artifact_dir = resolve_path(soft_config_path.parent, config["paths"]["artifact_dir"])
    if artifact_dir is None:
        raise ValueError("soft artifact_dir must be configured")
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, float | int | str]] = []
    fold_metric_rows: list[dict[str, float | int]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(frame), start=1):
        train = frame.iloc[train_idx].copy().reset_index(drop=True)
        test = frame.iloc[test_idx].copy().reset_index(drop=True)
        selected_features, _ = select_features_fold(
            train_x=train[feature_columns],
            train_y=train[label],
            task_type="regression",
            top_k=top_k,
        )
        train_df = train[selected_features + [label]].copy()
        test_df = test[selected_features + [label]].copy()

        baseline = make_regression_baseline()
        baseline.fit(train_df[selected_features], train_df[label].to_numpy(dtype=float))
        baseline_pred = np.clip(baseline.predict(test_df[selected_features]).astype(float), 0.0, 1.0)

        model_path = artifact_dir / f"ag_same_x_visual_soft_{run_id}_fold{fold_idx}"
        ag_pred, model_best = fit_autogluon_fold(
            train_df=train_df,
            test_df=test_df,
            label=label,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )
        ag_pred = np.clip(ag_pred.astype(float), 0.0, 1.0)

        base_metrics = _soft_metrics(test_df[label].to_numpy(dtype=float), baseline_pred)
        ag_metrics = _soft_metrics(test_df[label].to_numpy(dtype=float), ag_pred)
        fold_metric_rows.append(
            {
                "task": "soft_probability",
                "fold": int(fold_idx),
                "baseline_brier": base_metrics["brier"],
                "autogluon_brier": ag_metrics["brier"],
            }
        )

        for idx in range(len(test_df)):
            rows.append(
                {
                    "task": "soft_probability",
                    "fold": int(fold_idx),
                    "sample_time": str(pd.Timestamp(test.loc[idx, "sample_time"])),
                    "t90": float(test.loc[idx, "t90"]),
                    "true_label": float(test.loc[idx, label]),
                    "baseline_pred": float(baseline_pred[idx]),
                    "autogluon_pred": float(ag_pred[idx]),
                    "best_model": str(model_best),
                }
            )

    oof = pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    summary = {
        "baseline_mean_brier": float(fold_metrics["baseline_brier"].mean()),
        "autogluon_mean_brier": float(fold_metrics["autogluon_brier"].mean()),
        "relative_improvement_pct": float(
            (fold_metrics["baseline_brier"].mean() - fold_metrics["autogluon_brier"].mean())
            / fold_metrics["baseline_brier"].mean()
            * 100.0
        ),
    }
    return oof, summary


def build_dashboard(centered_oof: pd.DataFrame, soft_oof: pd.DataFrame, centered_summary: dict[str, float], soft_summary: dict[str, float], out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    ax = axes[0, 0]
    labels = ["Centered\n(MAE reduction)", "Soft Target\n(Brier reduction)"]
    improvements = [centered_summary["relative_improvement_pct"], soft_summary["relative_improvement_pct"]]
    bars = ax.bar(labels, improvements, color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Same-X Recipe Relative Gain vs Baseline")
    ax.set_ylabel("Improvement (%)")
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax = axes[0, 1]
    centered_fold = centered_oof.groupby("fold").apply(
        lambda df: pd.Series(
            {
                "baseline": np.mean(np.abs(df["true_label"] - df["baseline_pred"])),
                "autogluon": np.mean(np.abs(df["true_label"] - df["autogluon_pred"])),
            }
        ),
        include_groups=False,
    )
    soft_fold = soft_oof.groupby("fold").apply(
        lambda df: pd.Series(
            {
                "baseline": np.mean((df["true_label"] - df["baseline_pred"]) ** 2),
                "autogluon": np.mean((df["true_label"] - df["autogluon_pred"]) ** 2),
            }
        ),
        include_groups=False,
    )
    ax.plot(centered_fold.index, centered_fold["baseline"], marker="o", color="#7f7f7f", label="Centered baseline MAE")
    ax.plot(centered_fold.index, centered_fold["autogluon"], marker="o", color="#1f77b4", label="Centered AutoGluon MAE")
    ax.plot(soft_fold.index, soft_fold["baseline"], marker="s", linestyle="--", color="#c7c7c7", label="Soft baseline Brier")
    ax.plot(soft_fold.index, soft_fold["autogluon"], marker="s", linestyle="--", color="#ff7f0e", label="Soft AutoGluon Brier")
    ax.set_title("Fold-wise Held-Out Error")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Task-native error")
    ax.legend(fontsize=8, ncol=2)

    def _plot_recent(ax_obj, df: pd.DataFrame, title: str) -> None:
        recent = df.copy()
        recent["sample_time"] = pd.to_datetime(recent["sample_time"])
        recent = recent.sort_values("sample_time").tail(220).reset_index(drop=True)
        x = np.arange(len(recent))
        ax_obj.plot(x, recent["true_label"], color="#2f2f2f", linewidth=2, label="True label")
        ax_obj.plot(x, recent["baseline_pred"], color="#9e9e9e", linewidth=1.6, label="Baseline pred")
        ax_obj.plot(x, recent["autogluon_pred"], color="#1f77b4", linewidth=1.6, label="AutoGluon pred")
        ax_obj.set_title(title)
        ax_obj.set_xlabel("Recent held-out samples (time ordered)")
        ax_obj.set_ylabel("Label / prediction")
        ax_obj.legend(fontsize=8)

    _plot_recent(axes[1, 0], centered_oof, "Centered: True Label vs Predictions")
    _plot_recent(axes[1, 1], soft_oof, "Soft Target: True Label vs Predictions")

    fig.suptitle("Same-X Recipe Comparison: Centered vs Soft Target", fontsize=16, y=1.02)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_note(path: Path, centered_summary: dict[str, float], soft_summary: dict[str, float], image_name: str) -> None:
    lines = [
        "# Same-X Branch Comparison Visual Note",
        "",
        f"![same-x dashboard](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/{image_name})",
        "",
        "## 图的四个部分",
        "",
        "- 左上：在同一套 X 配方下，两条任务相对各自 baseline 的提升幅度。",
        f"  - centered_desirability：MAE 改善 {centered_summary['relative_improvement_pct']:.2f}%",
        f"  - soft target：soft Brier 改善 {soft_summary['relative_improvement_pct']:.2f}%",
        "- 右上：5 个留出时间折上的任务原生误差。线越低越好。",
        "- 左下：centered_desirability 最近一段留出样本中，真实标签、baseline 预测、AutoGluon 预测的折线对比。",
        "- 右下：soft target 最近一段留出样本中，真实软标签、baseline 预测、AutoGluon 预测的折线对比。",
        "",
        "## 解释口径",
        "",
        "- centered 与 soft target 的主指标量纲不同，因此不能直接比较绝对值大小。",
        "- 更合理的比较口径是：在完全相同的 X 配方下，各自相对 baseline 提升了多少。",
        "- 下方两张折线图展示的是“预测结果与真实标签的差距”在样本层面的直观形态。",
        "",
        "## 当前结论",
        "",
        "- 在相同 X 配方下，soft target 对该配方的响应更强。",
        "- centered_desirability 也有稳定正增益，但提升幅度较 soft target 更温和。",
        f"- centered: baseline MAE {centered_summary['baseline_mean_mae']:.5f} -> AutoGluon {centered_summary['autogluon_mean_mae']:.5f}",
        f"- soft target: baseline Brier {soft_summary['baseline_mean_brier']:.5f} -> AutoGluon {soft_summary['autogluon_mean_brier']:.5f}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    centered_config = DEFAULT_CENTERED_CONFIG.resolve()
    soft_config = DEFAULT_SOFT_CONFIG.resolve()
    centered_oof, centered_summary = collect_centered_oof(centered_config)
    soft_oof, soft_summary = collect_soft_oof(soft_config)

    artifact_dir = WORKSPACE_DIR / "artifacts"
    report_dir = WORKSPACE_DIR / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    combined_oof = pd.concat([centered_oof, soft_oof], ignore_index=True)
    (artifact_dir / "same_x_branch_comparison_oof_rows.csv").write_text(combined_oof.to_csv(index=False), encoding="utf-8-sig")
    summary = {
        "centered": centered_summary,
        "soft_target": soft_summary,
    }
    (artifact_dir / "same_x_branch_comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    image_name = "same_x_branch_comparison_dashboard.png"
    build_dashboard(
        centered_oof=centered_oof,
        soft_oof=soft_oof,
        centered_summary=centered_summary,
        soft_summary=soft_summary,
        out_path=artifact_dir / image_name,
    )
    write_note(report_dir / "same_x_branch_comparison_note.md", centered_summary, soft_summary, image_name)


if __name__ == "__main__":
    main()
