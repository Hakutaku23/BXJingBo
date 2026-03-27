from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


BASELINE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASELINE_DIR.parent / "artifacts"


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def policy_row(summary: dict[str, Any], policy_name: str) -> dict[str, Any]:
    for row in summary["policy_rows"]:
        if row["policy_name"] == policy_name:
            return row
    raise ValueError(f"Missing policy row: {policy_name}")


def build_compare_rows(window8: dict[str, Any], window10: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, summary in [("window8", window8), ("window10", window10)]:
        raw = policy_row(summary, "raw_two_level")
        hysteresis = policy_row(summary, "hysteresis_two_level")
        rows.append(
            {
                "candidate": label,
                "window_minutes": int(summary["selected_window_minutes"]),
                "topk_sensors": int(summary["selected_topk_sensors"]),
                "stats": ",".join(summary["selected_stats"]),
                "roc_auc": float(summary["model_level_metrics"]["roc_auc"]),
                "average_precision": float(summary["model_level_metrics"]["average_precision"]),
                "warning_threshold": float(summary["recommended_thresholds"]["warning"]["threshold"]),
                "alarm_threshold": float(summary["recommended_thresholds"]["alarm"]["threshold"]),
                "raw_warning_recall": float(raw["warning_recall"]),
                "raw_warning_specificity": float(raw["warning_specificity"]),
                "raw_alarm_recall": float(raw["alarm_recall"]),
                "raw_alarm_specificity": float(raw["alarm_specificity"]),
                "raw_switch_count": int(raw["switch_count"]),
                "hysteresis_warning_recall": float(hysteresis["warning_recall"]),
                "hysteresis_warning_specificity": float(hysteresis["warning_specificity"]),
                "hysteresis_alarm_recall": float(hysteresis["alarm_recall"]),
                "hysteresis_alarm_specificity": float(hysteresis["alarm_specificity"]),
                "hysteresis_switch_count": int(hysteresis["switch_count"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare short-window warning/alarm policy candidates.")
    parser.add_argument(
        "--window8-summary",
        type=Path,
        default=ARTIFACT_DIR / "phase1_warning_alarm_policy_window8_summary.json",
    )
    parser.add_argument(
        "--window10-summary",
        type=Path,
        default=ARTIFACT_DIR / "phase1_warning_alarm_policy_window10_summary.json",
    )
    parser.add_argument("--output-prefix", type=str, default="phase1_short_window_policy_compare")
    args = parser.parse_args()

    summary8 = load_summary(args.window8_summary)
    summary10 = load_summary(args.window10_summary)
    rows = build_compare_rows(summary8, summary10)
    frame = pd.DataFrame(rows).sort_values(
        ["roc_auc", "average_precision", "raw_alarm_specificity", "raw_warning_recall"],
        ascending=[False, False, False, False],
    )

    result = {
        "rows": frame.to_dict(orient="records"),
        "recommended_candidate_by_model_quality": frame.iloc[0]["candidate"] if not frame.empty else None,
    }

    frame.to_csv(ARTIFACT_DIR / f"{args.output_prefix}_summary.csv", index=False, encoding="utf-8-sig")
    with (ARTIFACT_DIR / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
