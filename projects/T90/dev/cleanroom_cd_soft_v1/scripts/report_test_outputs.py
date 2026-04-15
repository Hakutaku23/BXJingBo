from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_RUN_DIR = PROJECT_DIR / "outputs" / "20260414_140641_exp007_mixed_limsctx"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def build_test_table(scored: pd.DataFrame, spec_low: float, spec_high: float) -> pd.DataFrame:
    frame = scored.copy()
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], errors="coerce")
    frame = frame.sort_values("sample_time").reset_index(drop=True)
    frame["t90_true"] = pd.to_numeric(frame["t90"], errors="coerce")
    frame["t90_pred"] = pd.to_numeric(frame["simple_t90_pred"], errors="coerce") if "simple_t90_pred" in frame.columns else np.nan
    frame["t90_is_pass"] = ((frame["t90_true"] >= spec_low) & (frame["t90_true"] <= spec_high)).astype(int)

    if "simple_p_pass_soft_pred" in frame.columns:
        frame["t90_pass_probability_pred"] = pd.to_numeric(frame["simple_p_pass_soft_pred"], errors="coerce").clip(0.0, 1.0)
    elif "simple_is_out_spec_obs_pred" in frame.columns:
        frame["t90_pass_probability_pred"] = (1.0 - pd.to_numeric(frame["simple_is_out_spec_obs_pred"], errors="coerce")).clip(0.0, 1.0)
    else:
        frame["t90_pass_probability_pred"] = np.nan

    output_cols = [
        "sample_id",
        "sample_time",
        "t90_true",
        "t90_pred",
        "t90_is_pass",
        "t90_pass_probability_pred",
        "cd_mean",
        "simple_cd_mean_pred",
        "p_pass_soft",
        "simple_p_pass_soft_pred",
        "is_out_spec_obs",
        "simple_is_out_spec_obs_pred",
    ]
    return frame[[col for col in output_cols if col in frame.columns]]


def plot_test_labels(table: pd.DataFrame, output_path: Path, spec_low: float, spec_high: float, center: float) -> None:
    plot_frame = table.sort_values("sample_time").reset_index(drop=True)
    x = plot_frame["sample_time"]

    fig, ax_left = plt.subplots(figsize=(16, 6.2), dpi=170)
    ax_right = ax_left.twinx()

    raw_line = ax_left.plot(x, plot_frame["t90_true"], color="#2458a6", linewidth=1.35, label="test raw t90")
    ax_left.scatter(x, plot_frame["t90_true"], color="#2458a6", s=10, alpha=0.55, linewidths=0)

    pred_lines = []
    if plot_frame["t90_pred"].notna().any():
        pred_lines = ax_left.plot(x, plot_frame["t90_pred"], color="#2f7d32", linewidth=1.25, label="test predicted t90")

    cd_line = ax_right.plot(x, plot_frame["cd_mean"], color="#c4472d", linewidth=1.15, alpha=0.88, label="test mapped label cd_mean")

    ax_left.axhline(spec_low, color="#4f7f52", linestyle="--", linewidth=0.9, alpha=0.8, label=f"spec low {spec_low:.2f}")
    ax_left.axhline(center, color="#777777", linestyle=":", linewidth=0.9, alpha=0.8, label=f"center {center:.2f}")
    ax_left.axhline(spec_high, color="#4f7f52", linestyle="--", linewidth=0.9, alpha=0.8, label=f"spec high {spec_high:.2f}")

    ax_left.set_title("Test Set: Raw T90, Mapped Label, and Predicted T90")
    ax_left.set_xlabel("sample time")
    ax_left.set_ylabel("t90")
    ax_right.set_ylabel("mapped label cd_mean")
    ax_right.set_ylim(-0.03, 1.03)
    ax_left.grid(True, alpha=0.22)

    lines = raw_line + pred_lines + cd_line + ax_left.lines[-3:]
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="upper left", ncol=2, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any], table_csv: Path, plot_path: Path) -> None:
    lines = [
        "# 测试集输出报告",
        "",
        f"- model_run_dir: `{summary['model_run_dir']}`",
        f"- test_rows: `{summary['test_rows']}`",
        f"- table_csv: `{table_csv}`",
        f"- plot_png: `{plot_path}`",
        "",
        "## 说明",
        "",
        "- 本报告由传入的 model_run_dir 生成，结果对应当前模型目录内的测试集打分文件。",
        "- 当前模型预测 `cd_mean`、`p_pass_soft` 和 `is_out_spec_obs`，没有直接训练 `t90` 回归头，因此 `t90_pred` 为空。",
        "- `t90_pass_probability_pred` 使用 `simple_p_pass_soft_pred`。",
        "",
        "## 汇总",
        "",
        "```json",
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export test-set T90 table and plot from a model run.")
    parser.add_argument("--model-run-dir", type=Path, default=DEFAULT_MODEL_RUN_DIR)
    parser.add_argument("--spec-low", type=float, default=8.20)
    parser.add_argument("--spec-high", type=float, default=8.70)
    parser.add_argument("--center", type=float, default=8.45)
    args = parser.parse_args()

    model_run_dir = args.model_run_dir.resolve()
    scored_path = model_run_dir / "quick_model_scored_test_rows.csv"
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing scored test rows: {scored_path}")

    scored = pd.read_csv(scored_path)
    table = build_test_table(scored, spec_low=args.spec_low, spec_high=args.spec_high)

    table_csv = model_run_dir / "test_t90_output_table.csv"
    table_xlsx = model_run_dir / "test_t90_output_table.xlsx"
    plot_path = model_run_dir / "test_t90_label_prediction_plot.png"
    report_path = model_run_dir / "test_output_report.md"
    summary_path = model_run_dir / "test_output_summary.json"

    table.to_csv(table_csv, index=False, encoding="utf-8-sig")
    table.to_excel(table_xlsx, index=False)
    plot_test_labels(table, plot_path, spec_low=args.spec_low, spec_high=args.spec_high, center=args.center)

    pass_prob = table["t90_pass_probability_pred"].dropna()
    summary = {
        "model_run_dir": str(model_run_dir),
        "test_rows": int(len(table)),
        "t90_pred_available": bool(table["t90_pred"].notna().any()),
        "observed_pass_rate": float(table["t90_is_pass"].mean()),
        "predicted_pass_probability_mean": float(pass_prob.mean()) if not pass_prob.empty else None,
        "predicted_pass_probability_median": float(pass_prob.median()) if not pass_prob.empty else None,
        "table_csv": str(table_csv),
        "table_xlsx": str(table_xlsx),
        "plot_png": str(plot_path),
    }
    summary_path.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, summary, table_csv, plot_path)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
