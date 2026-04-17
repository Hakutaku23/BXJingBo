from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from test_jitl_probability import safe_metrics
from test_probability_anomaly_guard import json_ready


def parse_neighbor_months(value: Any) -> dict[str, float]:
    if pd.isna(value):
        return {}
    out: dict[str, float] = {}
    for part in str(value).split(";"):
        if not part or ":" not in part:
            continue
        month, weight = part.split(":", 1)
        try:
            out[month.strip()] = float(weight)
        except ValueError:
            continue
    total = sum(out.values())
    if total > 0.0:
        out = {key: val / total for key, val in out.items()}
    return out


def month_shares(row: pd.Series) -> pd.Series:
    weights = parse_neighbor_months(row.get("jitl_neighbor_months"))
    current = str(row.get("month"))
    try:
        current_p = pd.Period(current, freq="M")
    except Exception:
        return pd.Series(
            {
                "neighbor_current_month_weight": math.nan,
                "neighbor_previous_month_weight": math.nan,
                "neighbor_recent_month_weight": math.nan,
                "neighbor_older_month_weight": math.nan,
                "neighbor_dominant_month": "",
                "neighbor_dominant_weight": math.nan,
            }
        )
    current_w = float(weights.get(str(current_p), 0.0))
    prev_w = float(weights.get(str(current_p - 1), 0.0))
    recent_w = current_w + prev_w
    older_w = 0.0
    for month, weight in weights.items():
        try:
            period = pd.Period(month, freq="M")
        except Exception:
            continue
        if period < current_p - 1:
            older_w += float(weight)
    if weights:
        dominant_month, dominant_weight = max(weights.items(), key=lambda item: item[1])
    else:
        dominant_month, dominant_weight = "", math.nan
    return pd.Series(
        {
            "neighbor_current_month_weight": current_w,
            "neighbor_previous_month_weight": prev_w,
            "neighbor_recent_month_weight": recent_w,
            "neighbor_older_month_weight": older_w,
            "neighbor_dominant_month": dominant_month,
            "neighbor_dominant_weight": dominant_weight,
        }
    )


def build_dynamic_route(
    scored: pd.DataFrame,
    *,
    current_hard_min: float,
    current_blend_min: float,
    older_hard_max: float,
    drift_fallback_source: str,
) -> pd.DataFrame:
    out = scored.copy()
    share_df = out.apply(month_shares, axis=1)
    out = pd.concat([out, share_df], axis=1)
    first_month = sorted(out["month"].dropna().astype(str).unique())[0]
    out["first_test_month"] = first_month

    static = out["prob_out_spec_static_iso"].astype(float)
    jitl = out["prob_out_spec_jitl_protected"].astype(float)
    online = out["prob_out_spec_online_global_iso"].astype(float)
    routed = jitl.copy()
    route = pd.Series("jitl", index=out.index, dtype=object)
    weight = pd.Series(1.0, index=out.index, dtype=float)

    drift_mask = out["drift_risk_flag"].astype(str).isin(["high_feature_shift", "month_and_sample_feature_shift", "quality_anomaly"])
    if drift_fallback_source == "online":
        routed.loc[drift_mask] = online.loc[drift_mask]
        route.loc[drift_mask] = "drift_to_online"
    else:
        routed.loc[drift_mask] = static.loc[drift_mask]
        route.loc[drift_mask] = "drift_to_static"
    weight.loc[drift_mask] = 0.0

    current_w = out["neighbor_current_month_weight"].astype(float).fillna(0.0)
    older_w = out["neighbor_older_month_weight"].astype(float).fillna(0.0)
    non_first = out["month"].astype(str) != first_month

    stale_hard = (~drift_mask) & non_first & ((current_w < current_hard_min) | (older_w > older_hard_max))
    routed.loc[stale_hard] = online.loc[stale_hard]
    route.loc[stale_hard] = "stale_to_online"
    weight.loc[stale_hard] = 0.0

    blend_mask = (~drift_mask) & non_first & (~stale_hard) & (current_w < current_blend_min)
    denom = max(current_blend_min - current_hard_min, 1.0e-9)
    blend_weight = ((current_w - current_hard_min) / denom).clip(0.0, 1.0)
    routed.loc[blend_mask] = blend_weight.loc[blend_mask] * jitl.loc[blend_mask] + (1.0 - blend_weight.loc[blend_mask]) * online.loc[blend_mask]
    route.loc[blend_mask] = "blend_jitl_online"
    weight.loc[blend_mask] = blend_weight.loc[blend_mask]

    out["route_policy"] = route
    out["route_jitl_weight"] = weight
    out["prob_out_spec_dynamic_route"] = np.clip(routed.to_numpy(dtype=float), 0.0, 1.0)
    out["prob_pass_dynamic_route"] = 1.0 - out["prob_out_spec_dynamic_route"]
    out["prob_pass_static_iso"] = 1.0 - out["prob_out_spec_static_iso"].astype(float)
    out["prob_pass_online_global_iso"] = 1.0 - out["prob_out_spec_online_global_iso"].astype(float)
    out["prob_pass_jitl_protected"] = 1.0 - out["prob_out_spec_jitl_protected"].astype(float)
    return out


def group_metrics(scored: pd.DataFrame, prob_cols: list[str], group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prob_col in prob_cols:
        for key, sub in scored.groupby(group_col, dropna=False):
            if len(sub) < 2:
                continue
            rows.append({"probability_column": prob_col, group_col: key, **safe_metrics(sub, prob_col)})
    return pd.DataFrame(rows)


def plot_pass_probability(scored: pd.DataFrame, out_path: Path, *, title: str) -> None:
    work = scored.sort_values("sample_time").copy()
    work["sample_time"] = pd.to_datetime(work["sample_time"], errors="coerce")
    work["is_pass_obs"] = 1 - work["is_out_spec_obs"].astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.8), sharex=True, gridspec_kw={"height_ratios": [1.05, 1.0]})

    pass_mask = work["is_pass_obs"] == 1
    fail_mask = ~pass_mask
    axes[0].plot(work["sample_time"], work["t90"].astype(float), color="#355c7d", linewidth=1.8, label="T90 actual")
    axes[0].scatter(work.loc[pass_mask, "sample_time"], work.loc[pass_mask, "t90"].astype(float), s=28, color="#2a9d8f", label="qualified")
    axes[0].scatter(work.loc[fail_mask, "sample_time"], work.loc[fail_mask, "t90"].astype(float), s=36, color="#d62828", marker="x", label="out of spec")
    axes[0].set_ylabel("T90 actual")
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(work["sample_time"], work["prob_pass_dynamic_route"].astype(float), color="#1d3557", linewidth=2.0, label="dynamic routed pass probability")
    axes[1].plot(work["sample_time"], work["prob_pass_online_global_iso"].astype(float), color="#f4a261", linewidth=1.1, alpha=0.75, label="online global pass probability")
    axes[1].scatter(work.loc[pass_mask, "sample_time"], work.loc[pass_mask, "is_pass_obs"], s=18, color="#2a9d8f", alpha=0.65, label="actual qualified")
    axes[1].scatter(work.loc[fail_mask, "sample_time"], work.loc[fail_mask, "is_pass_obs"], s=30, color="#d62828", marker="x", alpha=0.8, label="actual out of spec")
    axes[1].set_ylim(-0.08, 1.08)
    axes[1].set_ylabel("Pass probability / label")
    axes[1].set_xlabel("Sample time")
    axes[1].legend(loc="lower left", ncol=2, frameon=False)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze JITL neighbor-month dynamic routing and plot pass probability.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label-delay-days", type=float, default=1.0)
    parser.add_argument("--current-hard-min", type=float, default=0.08)
    parser.add_argument("--current-blend-min", type=float, default=0.25)
    parser.add_argument("--older-hard-max", type=float, default=0.35)
    parser.add_argument("--drift-fallback-source", choices=["static", "online"], default="static")
    parser.add_argument("--run-tag", type=str, default="dynamic_route")
    args = parser.parse_args()

    scored_path = args.run_dir / "jitl_probability_scored.csv"
    scored = pd.read_csv(scored_path)
    scored = scored[scored["label_delay_days"].astype(float) == float(args.label_delay_days)].copy()
    if scored.empty:
        raise ValueError(f"No rows found for label_delay_days={args.label_delay_days}")
    scored["sample_time"] = pd.to_datetime(scored["sample_time"], errors="coerce")

    routed = build_dynamic_route(
        scored,
        current_hard_min=args.current_hard_min,
        current_blend_min=args.current_blend_min,
        older_hard_max=args.older_hard_max,
        drift_fallback_source=args.drift_fallback_source,
    )

    out_dir = args.run_dir / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    routed_path = out_dir / "jitl_dynamic_routed_scored.csv"
    routed.to_csv(routed_path, index=False, encoding="utf-8-sig")

    prob_cols = [
        "prob_out_spec_static_iso",
        "prob_out_spec_online_global_iso",
        "prob_out_spec_jitl_protected",
        "prob_out_spec_dynamic_route",
    ]
    metrics_df = pd.DataFrame([{"variant": col, **safe_metrics(routed, col)} for col in prob_cols])
    month_rows = []
    for col in prob_cols:
        for month, sub in routed.groupby("month"):
            month_rows.append({"probability_column": col, "month": month, **safe_metrics(sub, col)})
    month_df = pd.DataFrame(month_rows)
    route_df = group_metrics(routed, ["prob_out_spec_dynamic_route"], "route_policy")

    metrics_df.to_csv(out_dir / "jitl_dynamic_route_metrics.csv", index=False, encoding="utf-8-sig")
    month_df.to_csv(out_dir / "jitl_dynamic_route_month_metrics.csv", index=False, encoding="utf-8-sig")
    route_df.to_csv(out_dir / "jitl_dynamic_route_policy_metrics.csv", index=False, encoding="utf-8-sig")

    plot_path = out_dir / "test_t90_and_predicted_pass_probability.png"
    plot_pass_probability(
        routed,
        plot_path,
        title=f"JITL dynamic route, label delay {args.label_delay_days:g} day(s)",
    )

    summary = {
        "run_dir": str(args.run_dir),
        "out_dir": str(out_dir),
        "label_delay_days": float(args.label_delay_days),
        "current_hard_min": float(args.current_hard_min),
        "current_blend_min": float(args.current_blend_min),
        "older_hard_max": float(args.older_hard_max),
        "drift_fallback_source": args.drift_fallback_source,
        "route_policy_counts": routed["route_policy"].value_counts().to_dict(),
        "metrics": metrics_df.to_dict(orient="records"),
        "plot_path": str(plot_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
