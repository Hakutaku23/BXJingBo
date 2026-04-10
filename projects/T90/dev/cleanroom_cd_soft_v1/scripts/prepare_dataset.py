from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cleanroom_cd_soft_v1.config import load_config, resolve_path
from cleanroom_cd_soft_v1.data import apply_dcs_preprocessing, build_feature_table, deduplicate_dcs_columns, filter_samples_to_dcs_time_range, load_dcs_frame, load_lims_samples
from cleanroom_cd_soft_v1.labels import add_noise_aware_labels
from cleanroom_cd_soft_v1.split import leakage_audit, make_time_purged_split


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "base.yaml"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def summarize_labels(frame: pd.DataFrame) -> dict[str, Any]:
    summary_cols = [
        "t90",
        "cd_raw",
        "cd_mean",
        "cd_gaussian",
        "p_pass_soft",
        "p_fail_soft",
        "sample_weight",
        "align_confidence",
        "state_confidence",
    ]
    numeric_summary = {
        col: frame[col].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        for col in summary_cols
        if col in frame.columns
    }
    boundary_cols = [col for col in frame.columns if col.startswith("boundary_band_")]
    return {
        "sample_count": int(len(frame)),
        "time_min": frame["sample_time"].min().isoformat() if not frame.empty else None,
        "time_max": frame["sample_time"].max().isoformat() if not frame.empty else None,
        "observed_in_spec_rate": float(frame["is_in_spec_obs"].mean()) if "is_in_spec_obs" in frame else None,
        "observed_out_spec_rate": float(frame["is_out_spec_obs"].mean()) if "is_out_spec_obs" in frame else None,
        "center_band_rate": float(frame["is_center_band_obs"].mean()) if "is_center_band_obs" in frame else None,
        "boundary_band_rates": {col: float(frame[col].mean()) for col in boundary_cols},
        "numeric_summary": json_ready(numeric_summary),
    }


def write_report(
    path: Path,
    run_id: str,
    config: dict[str, Any],
    label_summary: dict[str, Any],
    split_audit: dict[str, Any],
    leak_audit: dict[str, Any],
    alignment_summary: dict[str, Any],
    boundary_audit: dict[str, Any],
    dcs_dedup_report: dict[str, Any],
    dcs_preprocess_report: dict[str, Any],
) -> None:
    lines = [
        "# Exp-001 数据治理与标签构造报告",
        "",
        f"- run_id: `{run_id}`",
        "- 项目线：`projects/T90/dev/cleanroom_cd_soft_v1`，开发支持，不进入稳定交付边界。",
        "- 本轮目标：完成 LIMS 聚合、DCS 因果窗口、噪声感知标签、时间 purge 切分与泄漏检查。",
        "",
        "## CQDI / RADI 说明",
        "",
        "- 既往 CQDI 与 RADI 得分均为 0，当前先判定为部署指标参数/门槛设计待校准问题。",
        "- Exp-001 不使用 CQDI/RADI 作为成败门槛，只输出后续重算所需的标签、边界带、软概率和 split 明细。",
        "",
        "## 标签口径",
        "",
        f"- center: `{config['target_spec']['center']}`",
        f"- spec band: `{config['target_spec']['spec_low']} ~ {config['target_spec']['spec_high']}`",
        f"- centered tolerance half width: `{config['target_spec']['tolerance_half_width']}`",
        f"- cd_method: `{config['labels']['cd_method']}`",
        f"- measurement_error_scenarios: `{config['labels']['measurement_error_scenarios']}`",
        "",
        "## 标签摘要",
        "",
        "```json",
        json.dumps(label_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 切分摘要",
        "",
        "```json",
        json.dumps(split_audit, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 泄漏检查",
        "",
        "```json",
        json.dumps(leak_audit, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 对齐质量",
        "",
        "```json",
        json.dumps(alignment_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 数据边界",
        "",
        "```json",
        json.dumps(boundary_audit, ensure_ascii=False, indent=2),
        "```",
        "",
        "## DCS 位点去重",
        "",
        "```json",
        json.dumps(dcs_dedup_report, ensure_ascii=False, indent=2),
        "```",
        "",
        "## DCS 降噪预处理",
        "",
        "```json",
        json.dumps(dcs_preprocess_report, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 下一步",
        "",
        "- 若 split 中 train/valid/test 样本量均可用，进入 Exp-002 强基线。",
        "- CQDI/RADI 下一步应先改为非硬门槛诊断表，输出子分项和 gate 失败原因，再重新设定阈值网格。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp-001 cleanroom dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--denoise-method", type=str, default=None)
    parser.add_argument("--ema-span-rows", type=int, default=None)
    parser.add_argument("--hampel-enabled", action="store_true")
    parser.add_argument("--hampel-window-rows", type=int, default=None)
    parser.add_argument("--hampel-n-sigmas", type=float, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    preprocessing_cfg = config.setdefault("dcs", {}).setdefault("preprocessing", {})
    if args.denoise_method is not None:
        preprocessing_cfg["denoise_method"] = args.denoise_method
    if args.ema_span_rows is not None:
        preprocessing_cfg["ema_span_rows"] = int(args.ema_span_rows)
    if args.hampel_enabled:
        preprocessing_cfg.setdefault("hampel", {})["enabled"] = True
    if args.hampel_window_rows is not None:
        preprocessing_cfg.setdefault("hampel", {})["window_rows"] = int(args.hampel_window_rows)
    if args.hampel_n_sigmas is not None:
        preprocessing_cfg.setdefault("hampel", {})["n_sigmas"] = float(args.hampel_n_sigmas)
    tag = args.run_tag or str(config["experiment"]["run_tag"])
    run_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{tag}"

    output_root = resolve_path(config_path, config["paths"]["output_root"])
    if output_root is None:
        raise ValueError("paths.output_root is required.")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    lims_path = resolve_path(config_path, config["paths"]["lims_path"])
    dcs_main_path = resolve_path(config_path, config["paths"]["dcs_main_path"])
    dcs_supp_path = resolve_path(config_path, config["paths"].get("dcs_supplemental_path"))
    if lims_path is None or dcs_main_path is None:
        raise ValueError("LIMS and DCS paths are required.")

    shutil.copy2(config_path, run_dir / "config_snapshot.yaml")
    dcs_raw = load_dcs_frame(dcs_main_path, dcs_supp_path, config)
    dcs, dcs_dedup_report = deduplicate_dcs_columns(dcs_raw, config)
    dcs, dcs_preprocess_report = apply_dcs_preprocessing(dcs, config)
    samples = load_lims_samples(lims_path, config)
    samples, boundary_audit = filter_samples_to_dcs_time_range(samples, dcs, config)
    if args.max_samples:
        samples = samples.sort_values("sample_time").tail(int(args.max_samples)).reset_index(drop=True)
        boundary_audit["max_samples_after_boundary_filter"] = int(args.max_samples)
    features, alignment_quality = build_feature_table(samples, dcs, config)
    merged = samples.merge(features, on="sample_id", how="left")
    labeled = add_noise_aware_labels(merged, config)
    split, split_audit = make_time_purged_split(labeled, config)
    leak_audit = leakage_audit(split)
    labeled = labeled.merge(split[["sample_id", "split"]], on="sample_id", how="left")

    feature_table = labeled.drop(columns=["source_sheets"], errors="ignore")
    label_summary = summarize_labels(labeled)
    alignment_summary = {
        "candidate_rows": int(len(alignment_quality)),
        "mean_window_confidence": float(alignment_quality["window_confidence"].mean()) if not alignment_quality.empty else None,
        "sufficient_window_rate": float(alignment_quality["sufficient_points"].mean()) if not alignment_quality.empty else None,
        "mean_missing_rate": float(alignment_quality["mean_missing_rate"].mean()) if not alignment_quality.empty else None,
        "by_window": json_ready(
            alignment_quality.groupby(["window_minutes", "lag_minutes"])[
                ["point_count", "window_confidence", "mean_missing_rate"]
            ]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
        if not alignment_quality.empty
        else [],
    }

    labeled.to_csv(run_dir / "labeled_samples.csv", index=False, encoding="utf-8-sig")
    feature_table.to_csv(run_dir / "feature_table.csv", index=False, encoding="utf-8-sig")
    split.to_csv(run_dir / "split_indices.csv", index=False, encoding="utf-8-sig")
    alignment_quality.to_csv(run_dir / "alignment_window_quality.csv", index=False, encoding="utf-8-sig")
    (run_dir / "label_summary.json").write_text(json.dumps(json_ready(label_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "split_audit.json").write_text(json.dumps(json_ready(split_audit), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "leakage_report.json").write_text(json.dumps(json_ready(leak_audit), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "alignment_quality_report.json").write_text(json.dumps(json_ready(alignment_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "data_boundary_report.json").write_text(json.dumps(json_ready(boundary_audit), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "dcs_deduplication_report.json").write_text(json.dumps(json_ready(dcs_dedup_report), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "dcs_preprocessing_report.json").write_text(json.dumps(json_ready(dcs_preprocess_report), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(run_dir / "exp001_report.md", run_id, config, label_summary, split_audit, leak_audit, alignment_summary, boundary_audit, dcs_dedup_report, dcs_preprocess_report)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "sample_count": int(len(labeled)),
                "split_counts": split["split"].value_counts().to_dict(),
                "boundary_audit": boundary_audit,
                "dcs_deduplication": {
                    "dropped_count": dcs_dedup_report.get("dropped_count"),
                    "dropped_columns": dcs_dedup_report.get("dropped_columns"),
                },
                "dcs_preprocessing": dcs_preprocess_report,
                "label_summary_path": str(run_dir / "label_summary.json"),
                "report_path": str(run_dir / "exp001_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
