from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cleanroom_cd_soft_v1.config import load_config


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "base.yaml"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    return value


def feature_columns(frame: pd.DataFrame, include_lims_context: bool, window_minutes: int | None = None) -> list[str]:
    dcs_features = [col for col in frame.columns if col.startswith("w") and "__" in col]
    if window_minutes is not None:
        dcs_features = [col for col in dcs_features if col.startswith(f"w{int(window_minutes)}_")]
    context_features = [col for col in frame.columns if col.startswith("lims_ctx__") and not col.endswith("__count")]
    return dcs_features + (context_features if include_lims_context else [])


def load_model_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    summary_path = path / "quick_model_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def selected_feature_counts(summary: dict[str, Any] | None) -> dict[str, int]:
    if not summary:
        return {}
    result: dict[str, int] = {}
    for target, target_info in summary.get("targets", {}).items():
        selection = target_info.get("feature_selection", {})
        if "selected_feature_count" in selection:
            result[target] = int(selection["selected_feature_count"])
    return result


def plot_label_distribution(frame: pd.DataFrame, raw_col: str, model_col: str, output_path: Path) -> Path:
    plot_frame = frame[[raw_col, model_col]].dropna().copy()
    plot_frame = plot_frame.sort_values(raw_col).reset_index(drop=True)
    quantile = np.linspace(0.0, 100.0, len(plot_frame))
    raw_sorted = np.sort(plot_frame[raw_col].to_numpy(dtype=float))
    model_sorted = np.sort(plot_frame[model_col].to_numpy(dtype=float))

    fig, ax_left = plt.subplots(figsize=(10, 5.2), dpi=160)
    ax_right = ax_left.twinx()

    left_line = ax_left.plot(quantile, raw_sorted, color="#2458a6", linewidth=2.0, label=f"raw {raw_col}")
    right_line = ax_right.plot(quantile, model_sorted, color="#c4472d", linewidth=2.0, label=f"model {model_col}")

    ax_left.set_xlabel("sample quantile (%)")
    ax_left.set_ylabel(f"raw label: {raw_col}")
    ax_right.set_ylabel(f"model label: {model_col}")
    ax_left.set_title("Raw Label vs Modeling Label Distribution")
    ax_left.grid(True, axis="both", alpha=0.25)

    lines = left_line + right_line
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_label_time_series(frame: pd.DataFrame, raw_col: str, model_col: str, output_path: Path) -> Path:
    plot_frame = frame[["sample_time", "split", raw_col, model_col]].dropna(subset=["sample_time", raw_col, model_col]).copy()
    plot_frame["sample_time"] = pd.to_datetime(plot_frame["sample_time"], errors="coerce")
    plot_frame = plot_frame.dropna(subset=["sample_time"]).sort_values("sample_time").reset_index(drop=True)

    fig, ax_left = plt.subplots(figsize=(16, 6.5), dpi=170)
    ax_right = ax_left.twinx()

    split_colors = {
        "train": "#d9edf7",
        "valid": "#fff1c2",
        "test": "#e4d7f5",
        "purged": "#eeeeee",
    }
    for split, group in plot_frame.groupby("split", sort=False):
        color = split_colors.get(str(split), "#f2f2f2")
        ax_left.axvspan(group["sample_time"].min(), group["sample_time"].max(), color=color, alpha=0.22, linewidth=0)

    raw_line = ax_left.plot(
        plot_frame["sample_time"],
        plot_frame[raw_col].astype(float),
        color="#2458a6",
        linewidth=1.05,
        alpha=0.92,
        label=f"raw {raw_col}",
    )
    model_line = ax_right.plot(
        plot_frame["sample_time"],
        plot_frame[model_col].astype(float),
        color="#c4472d",
        linewidth=1.05,
        alpha=0.90,
        label=f"model {model_col}",
    )

    scatter_colors = plot_frame["split"].map({"train": "#2458a6", "valid": "#8a6f00", "test": "#613f91", "purged": "#666666"}).fillna("#333333")
    ax_left.scatter(plot_frame["sample_time"], plot_frame[raw_col].astype(float), c=scatter_colors, s=6, alpha=0.55, linewidths=0)

    ax_left.axhline(8.20, color="#4f7f52", linestyle="--", linewidth=0.9, alpha=0.75, label="raw spec low 8.20")
    ax_left.axhline(8.45, color="#777777", linestyle=":", linewidth=0.9, alpha=0.75, label="raw center 8.45")
    ax_left.axhline(8.70, color="#4f7f52", linestyle="--", linewidth=0.9, alpha=0.75, label="raw spec high 8.70")

    ax_left.set_title("Complete Label Time Series: Raw t90 vs Modeling cd_mean")
    ax_left.set_xlabel("sample time")
    ax_left.set_ylabel(f"raw label: {raw_col}")
    ax_right.set_ylabel(f"model label: {model_col}")
    ax_right.set_ylim(-0.03, 1.03)
    ax_left.grid(True, axis="both", alpha=0.20)

    split_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.28, label=f"{split} region")
        for split, color in split_colors.items()
        if split in set(plot_frame["split"].astype(str))
    ]
    lines = raw_line + model_line + ax_left.lines[-3:] + split_handles
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="upper left", ncol=2, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def alignment_shapes(alignment: pd.DataFrame, dcs_channel_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (window, lag), group in alignment.groupby(["window_minutes", "lag_minutes"]):
        points = group["point_count"].astype(float)
        n = int(group["sample_id"].nunique())
        rows.append(
            {
                "window_minutes": int(window),
                "lag_minutes": int(lag),
                "sequence_shape_median": [n, int(round(float(points.median()))), int(dcs_channel_count)],
                "sequence_shape_p10": [n, int(round(float(points.quantile(0.10)))), int(dcs_channel_count)],
                "sequence_shape_p90": [n, int(round(float(points.quantile(0.90)))), int(dcs_channel_count)],
                "point_count_mean": float(points.mean()),
                "point_count_min": float(points.min()),
                "point_count_max": float(points.max()),
            }
        )
    return sorted(rows, key=lambda item: (item["window_minutes"], item["lag_minutes"]))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 标签可视化与模型输入维度报告",
        "",
        f"- prepared_run_dir: `{report['prepared_run_dir']}`",
        f"- label_plot: `{report['label_plot']}`",
        f"- label_time_series_plot: `{report['label_time_series_plot']}`",
        "",
        "## 完整标签时间序列图",
        "",
        f"![Complete Label Time Series]({Path(report['label_time_series_plot']).resolve().as_posix()})",
        "",
        "## 标签分布图",
        "",
        f"![Raw Label vs Modeling Label Distribution]({Path(report['label_plot']).resolve().as_posix()})",
        "",
        "## Tabular 模型输入维度",
        "",
        f"- 样本数：`{report['sample_count']}`",
        f"- split：`{report['split_counts']}`",
        f"- DCS 去重后位点数：`{report['dcs_channel_count']}`",
        f"- LIMS context 特征数：`{report['lims_context_feature_count']}`",
        f"- 全部 DCS 窗口统计特征：`{report['all_dcs_feature_count']}`",
        f"- 主窗口 `240 min` DCS-only 候选维度：`{report['main_window']['dcs_only_tabular_shape']}`",
        f"- 主窗口 `240 min` DCS+LIMS-context 候选维度：`{report['main_window']['dcs_lims_context_tabular_shape']}`",
        "",
        "## 已筛选入模维度",
        "",
        f"- 主线 A selected：`{report['selected_feature_counts'].get('main_a_dcs_only', {})}`",
        f"- 主线 B selected：`{report['selected_feature_counts'].get('main_b_lims_context', {})}`",
        "",
        "## 时间窗口序列形态",
        "",
        "这里的 `(n, win, samp)` 按实际原始序列解释为 `(样本数, 窗口内DCS行数, DCS位点数)`；由于 DCS 时间戳可能不完全等间隔，`win` 用窗口内实际点数的 median / p10 / p90 表示。",
        "",
        "```json",
        json.dumps(report["sequence_shapes"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create label plot and model input dimension report.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared-run-dir", type=Path, required=True)
    parser.add_argument("--raw-label-col", type=str, default="t90")
    parser.add_argument("--model-label-col", type=str, default="cd_mean")
    parser.add_argument("--main-window-minutes", type=int, default=240)
    parser.add_argument("--main-a-model-run-dir", type=Path, default=None)
    parser.add_argument("--main-b-model-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    prepared_run = args.prepared_run_dir.resolve()
    output_dir = (args.output_dir or prepared_run).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(prepared_run / "feature_table.csv")
    alignment = pd.read_csv(prepared_run / "alignment_window_quality.csv")
    dcs_report_path = prepared_run / "dcs_deduplication_report.json"
    dcs_report = json.loads(dcs_report_path.read_text(encoding="utf-8")) if dcs_report_path.exists() else {}

    dcs_channel_count = int(dcs_report.get("output_columns", 0))
    if dcs_channel_count <= 0:
        dcs_channel_count = len({col.split("__")[1] for col in frame.columns if col.startswith("w") and "__" in col})

    plot_path = output_dir / "label_distribution_dual_axis.png"
    plot_label_distribution(frame, args.raw_label_col, args.model_label_col, plot_path)
    time_series_plot_path = output_dir / "label_timeseries_raw_vs_model_dual_axis.png"
    plot_label_time_series(frame, args.raw_label_col, args.model_label_col, time_series_plot_path)

    main_window = int(args.main_window_minutes)
    dcs_features = feature_columns(frame, include_lims_context=False, window_minutes=main_window)
    context_features = [col for col in frame.columns if col.startswith("lims_ctx__") and not col.endswith("__count")]
    dcs_context_features = feature_columns(frame, include_lims_context=True, window_minutes=main_window)
    all_dcs_features = feature_columns(frame, include_lims_context=False, window_minutes=None)
    split_counts = frame["split"].value_counts(dropna=False).to_dict() if "split" in frame.columns else {}
    sample_count = int(len(frame))

    main_a_summary = load_model_summary(args.main_a_model_run_dir.resolve() if args.main_a_model_run_dir else None)
    main_b_summary = load_model_summary(args.main_b_model_run_dir.resolve() if args.main_b_model_run_dir else None)

    report = {
        "prepared_run_dir": str(prepared_run),
        "label_plot": str(plot_path),
        "label_time_series_plot": str(time_series_plot_path),
        "sample_count": sample_count,
        "split_counts": split_counts,
        "dcs_channel_count": dcs_channel_count,
        "lims_context_feature_count": int(len(context_features)),
        "all_dcs_feature_count": int(len(all_dcs_features)),
        "main_window": {
            "window_minutes": main_window,
            "dcs_only_candidate_feature_count": int(len(dcs_features)),
            "dcs_lims_context_candidate_feature_count": int(len(dcs_context_features)),
            "dcs_only_tabular_shape": [sample_count, int(len(dcs_features))],
            "dcs_lims_context_tabular_shape": [sample_count, int(len(dcs_context_features))],
        },
        "selected_feature_counts": {
            "main_a_dcs_only": selected_feature_counts(main_a_summary),
            "main_b_lims_context": selected_feature_counts(main_b_summary),
        },
        "sequence_shapes": alignment_shapes(alignment, dcs_channel_count),
        "feature_formula": {
            "stats": list(config["dcs"]["stats"]),
            "main_window_dcs_formula": f"{dcs_channel_count} DCS points * 3 lags * {len(config['dcs']['stats'])} stats = {len(dcs_features)} features",
        },
    }

    (output_dir / "input_dimension_report.json").write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_dir / "input_dimension_report.md", json_ready(report))
    print(json.dumps(json_ready(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
