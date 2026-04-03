from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from run_autogluon_stage2_feature_engineering import (
    fit_autogluon_fold,
    load_config,
    make_regression_baseline,
    resolve_path,
    select_features_fold,
)
from run_autogluon_stage2_soft_probability_feature_distillation import build_variant_snapshot


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = ROOT_DIR / "configs" / "autogluon_stage2_soft_probability_feature_distillation.yaml"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
REPORT_DIR = ROOT_DIR / "reports"

OUTPUT_CSV = ARTIFACT_DIR / "range_position_full_true_vs_pred_rows.csv"
OUTPUT_PNG = ARTIFACT_DIR / "range_position_full_true_vs_pred_dashboard.png"
OUTPUT_MD = REPORT_DIR / "range_position_full_true_vs_pred_note.md"


def export_rows() -> pd.DataFrame:
    config = load_config(CONFIG_PATH)
    snapshot, _ = build_variant_snapshot(CONFIG_PATH, config, "range_position_full")
    feature_columns = [column for column in snapshot.columns if "__" in column]
    label = str(config["label_fuzziness"]["target_name"])
    top_k = int(config["selection"]["soft_probability_top_k"])
    splitter = TimeSeriesSplit(n_splits=int(config["validation"]["n_splits"]))
    artifact_dir = resolve_path(CONFIG_PATH.parent, config["paths"]["artifact_dir"])
    if artifact_dir is None:
        raise ValueError("artifact_dir must be configured.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, object]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(snapshot), start=1):
        train = snapshot.iloc[train_idx].copy().reset_index(drop=True)
        test = snapshot.iloc[test_idx].copy().reset_index(drop=True)
        selected_features, _ = select_features_fold(
            train_x=train[feature_columns],
            train_y=train[label],
            task_type="regression",
            top_k=top_k,
        )

        baseline = make_regression_baseline()
        baseline.fit(train[selected_features], train[label].to_numpy(dtype=float))
        baseline_pred = baseline.predict(test[selected_features]).astype(float)

        model_path = artifact_dir / f"ag_stage2_soft_probability_visual_{run_id}_fold{fold_idx}"
        framework_pred, model_best = fit_autogluon_fold(
            train_df=train[selected_features + [label]].copy(),
            test_df=test[selected_features + [label]].copy(),
            label=label,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            model_path=model_path,
            ag_config=config["autogluon"],
        )

        for idx, (_, row) in enumerate(test.iterrows()):
            rows.append(
                {
                    "fold": int(fold_idx),
                    "sample_time": pd.to_datetime(row["sample_time"]),
                    "t90": float(row["t90"]),
                    "is_out_of_spec": int(row["is_out_of_spec"]),
                    "true_soft_target": float(row[label]),
                    "baseline_pred_soft": float(np.clip(baseline_pred[idx], 0.0, 1.0)),
                    "autogluon_pred_soft": float(np.clip(framework_pred[idx], 0.0, 1.0)),
                    "autogluon_model_best": str(model_best),
                }
            )

    out = pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    return out


def make_dashboard(rows: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("range_position_full: True vs Predicted", fontsize=16, fontweight="bold")

    true_soft = rows["true_soft_target"].to_numpy(dtype=float)
    baseline_pred = rows["baseline_pred_soft"].to_numpy(dtype=float)
    ag_pred = rows["autogluon_pred_soft"].to_numpy(dtype=float)
    t90 = rows["t90"].to_numpy(dtype=float)
    out_flag = rows["is_out_of_spec"].to_numpy(dtype=int)
    times = pd.to_datetime(rows["sample_time"])

    ax = axes[0, 0]
    ax.scatter(true_soft, baseline_pred, s=18, alpha=0.45, color="#90A4AE", label="Baseline")
    ax.scatter(true_soft, ag_pred, s=18, alpha=0.45, color="#1565C0", label="AutoGluon")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_title("True Soft Target vs Predicted Soft Risk")
    ax.set_xlabel("True soft target")
    ax.set_ylabel("Predicted soft risk")
    ax.legend(loc="best")

    ax = axes[0, 1]
    mask_in = out_flag == 0
    mask_out = out_flag == 1
    ax.scatter(t90[mask_in], ag_pred[mask_in], s=18, alpha=0.4, color="#43A047", label="In spec")
    ax.scatter(t90[mask_out], ag_pred[mask_out], s=18, alpha=0.5, color="#E53935", label="Out of spec")
    ax.axvline(8.2, linestyle="--", color="#6D4C41", linewidth=1)
    ax.axvline(8.7, linestyle="--", color="#6D4C41", linewidth=1)
    ax.set_title("Actual T90 vs AutoGluon Predicted Risk")
    ax.set_xlabel("Actual T90")
    ax.set_ylabel("Predicted soft risk")
    ax.legend(loc="best")

    ax = axes[1, 0]
    order = np.argsort(ag_pred)
    ax.plot(np.arange(len(rows)), true_soft[order], color="#37474F", linewidth=2, label="True soft target")
    ax.plot(np.arange(len(rows)), ag_pred[order], color="#1565C0", linewidth=2, label="AutoGluon pred")
    ax.plot(np.arange(len(rows)), baseline_pred[order], color="#B0BEC5", linewidth=2, label="Baseline pred")
    ax.set_title("Sorted-by-Risk Profile")
    ax.set_xlabel("Samples sorted by AutoGluon predicted risk")
    ax.set_ylabel("Soft value")
    ax.legend(loc="best")

    ax = axes[1, 1]
    recent = rows.tail(min(220, len(rows))).copy()
    ax.plot(pd.to_datetime(recent["sample_time"]), recent["t90"], color="#5D4037", linewidth=1.6, label="Actual T90")
    ax.axhline(8.2, linestyle="--", color="#8D6E63", linewidth=1)
    ax.axhline(8.7, linestyle="--", color="#8D6E63", linewidth=1)
    ax.set_ylabel("Actual T90")
    ax.set_title("Recent Held-Out Samples: T90 and Predicted Risk")
    ax.tick_params(axis="x", rotation=25)
    ax2 = ax.twinx()
    ax2.plot(pd.to_datetime(recent["sample_time"]), recent["autogluon_pred_soft"], color="#1565C0", linewidth=1.6, label="Predicted soft risk")
    ax2.set_ylabel("Predicted soft risk")
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_note(rows: pd.DataFrame) -> None:
    true_soft = rows["true_soft_target"].to_numpy(dtype=float)
    ag_pred = rows["autogluon_pred_soft"].to_numpy(dtype=float)
    baseline_pred = rows["baseline_pred_soft"].to_numpy(dtype=float)
    corr_ag = pd.Series(true_soft).corr(pd.Series(ag_pred), method="spearman")
    corr_base = pd.Series(true_soft).corr(pd.Series(baseline_pred), method="spearman")
    lines = [
        "# range_position_full True-vs-Predicted Note",
        "",
        "## What This Visualization Means",
        "",
        "- This project's current mainline model predicts a soft out-of-spec risk, not a hard class and not raw T90 directly.",
        "- Therefore the most faithful true-vs-predicted comparison is `true soft target` versus `predicted soft risk`.",
        "- A second useful view is `actual T90` versus `predicted soft risk`, because it connects model output back to process understanding.",
        "",
        "## Quick Reading",
        "",
        f"- Spearman correlation between true soft target and AutoGluon prediction: `{corr_ag:.4f}`",
        f"- Spearman correlation between true soft target and baseline prediction: `{corr_base:.4f}`",
        "- If AutoGluon points are visually closer to the diagonal than baseline, the model is learning the soft target structure better.",
        "- If higher actual T90 values concentrate at higher predicted risk, the risk score is process-relevant.",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = export_rows()
    make_dashboard(rows)
    write_note(rows)
    print(
        json.dumps(
            {
                "rows_path": str(OUTPUT_CSV),
                "png_path": str(OUTPUT_PNG),
                "note_path": str(OUTPUT_MD),
                "samples": int(len(rows)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
