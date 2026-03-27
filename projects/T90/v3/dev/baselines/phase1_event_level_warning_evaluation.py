from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASELINE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASELINE_DIR.parent / "artifacts"


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def apply_raw_two_level(probabilities: np.ndarray, warning_threshold: float, alarm_threshold: float) -> np.ndarray:
    states = np.zeros(len(probabilities), dtype=int)
    states[probabilities >= warning_threshold] = 1
    states[probabilities >= alarm_threshold] = 2
    return states


def apply_hysteresis_two_level(
    probabilities: np.ndarray,
    warning_threshold: float,
    alarm_threshold: float,
    warning_release_ratio: float = 0.80,
    alarm_release_ratio: float = 0.85,
) -> np.ndarray:
    states = np.zeros(len(probabilities), dtype=int)
    warning_off = warning_threshold * warning_release_ratio
    alarm_off = alarm_threshold * alarm_release_ratio
    state = 0
    for idx, value in enumerate(probabilities):
        if state == 0:
            if value >= alarm_threshold:
                state = 2
            elif value >= warning_threshold:
                state = 1
        elif state == 1:
            if value >= alarm_threshold:
                state = 2
            elif value < warning_off:
                state = 0
        else:
            if value >= alarm_off:
                state = 2
            elif value >= warning_threshold:
                state = 1
            elif value < warning_off:
                state = 0
        states[idx] = state
    return states


def build_segments(
    frame: pd.DataFrame,
    label_column: str,
    time_column: str = "sample_time",
    gap_limit_minutes: float | None = None,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    work = frame.sort_values(time_column).reset_index(drop=True).copy()
    work[time_column] = pd.to_datetime(work[time_column], errors="coerce")
    labels = work[label_column].astype(int).to_numpy()
    times = work[time_column].to_numpy(dtype="datetime64[ns]")

    segments: list[dict[str, Any]] = []
    start = 0
    for idx in range(1, len(work)):
        label_changed = labels[idx] != labels[idx - 1]
        gap_break = False
        if gap_limit_minutes is not None:
            gap_minutes = (times[idx] - times[idx - 1]) / np.timedelta64(1, "m")
            gap_break = float(gap_minutes) > gap_limit_minutes
        if label_changed or gap_break:
            segments.append(
                {
                    "segment_id": len(segments),
                    "label": int(labels[start]),
                    "start_idx": int(start),
                    "end_idx": int(idx - 1),
                }
            )
            start = idx
    segments.append(
        {
            "segment_id": len(segments),
            "label": int(labels[start]),
            "start_idx": int(start),
            "end_idx": int(len(work) - 1),
        }
    )
    return segments


def run_length_count(mask: np.ndarray) -> tuple[int, float]:
    if len(mask) == 0:
        return 0, 0.0
    run_lengths = []
    run = 0
    for value in mask.astype(int):
        if value:
            run += 1
        elif run:
            run_lengths.append(run)
            run = 0
    if run:
        run_lengths.append(run)
    if not run_lengths:
        return 0, 0.0
    return len(run_lengths), float(np.mean(run_lengths))


def evaluate_policy(
    frame: pd.DataFrame,
    states: np.ndarray,
    gap_limit_minutes: float,
    prewarn_horizon_minutes: float,
) -> dict[str, Any]:
    work = frame.sort_values("sample_time").reset_index(drop=True).copy()
    work["sample_time"] = pd.to_datetime(work["sample_time"], errors="coerce")
    work["state"] = states
    work["warning_or_alarm"] = work["state"] >= 1
    work["alarm_only"] = work["state"] == 2

    segments = build_segments(work, "is_out_of_spec", "sample_time", gap_limit_minutes=gap_limit_minutes)
    bad_segments = [segment for segment in segments if segment["label"] == 1]
    good_segments = [segment for segment in segments if segment["label"] == 0]

    detected_by_warning = 0
    detected_by_alarm = 0
    prewarn_warning = 0
    prewarn_alarm = 0
    warning_delay_samples = []
    alarm_delay_samples = []
    warning_delay_minutes = []
    alarm_delay_minutes = []

    for segment in bad_segments:
        segment_slice = work.iloc[segment["start_idx"] : segment["end_idx"] + 1]
        segment_start_time = segment_slice.iloc[0]["sample_time"]
        warning_positions = np.flatnonzero(segment_slice["warning_or_alarm"].to_numpy(dtype=bool))
        alarm_positions = np.flatnonzero(segment_slice["alarm_only"].to_numpy(dtype=bool))

        if len(warning_positions) > 0:
            detected_by_warning += 1
            first_idx = int(warning_positions[0])
            warning_delay_samples.append(first_idx)
            warning_delay_minutes.append(
                float((segment_slice.iloc[first_idx]["sample_time"] - segment_start_time).total_seconds() / 60.0)
            )
        if len(alarm_positions) > 0:
            detected_by_alarm += 1
            first_idx = int(alarm_positions[0])
            alarm_delay_samples.append(first_idx)
            alarm_delay_minutes.append(
                float((segment_slice.iloc[first_idx]["sample_time"] - segment_start_time).total_seconds() / 60.0)
            )

        history = work.iloc[: segment["start_idx"]]
        if not history.empty:
            history = history[history["sample_time"] >= segment_start_time - pd.Timedelta(minutes=prewarn_horizon_minutes)]
            if history["warning_or_alarm"].any():
                prewarn_warning += 1
            if history["alarm_only"].any():
                prewarn_alarm += 1

    false_warning_good_segments = 0
    false_alarm_good_segments = 0
    for segment in good_segments:
        segment_slice = work.iloc[segment["start_idx"] : segment["end_idx"] + 1]
        if segment_slice["warning_or_alarm"].any():
            false_warning_good_segments += 1
        if segment_slice["alarm_only"].any():
            false_alarm_good_segments += 1

    warning_event_runs, mean_warning_run_length = run_length_count(work["warning_or_alarm"].to_numpy(dtype=bool))
    alarm_event_runs, mean_alarm_run_length = run_length_count(work["alarm_only"].to_numpy(dtype=bool))

    return {
        "samples": int(len(work)),
        "bad_segment_count": int(len(bad_segments)),
        "good_segment_count": int(len(good_segments)),
        "bad_segment_warning_recall": float(detected_by_warning / len(bad_segments)) if bad_segments else 0.0,
        "bad_segment_alarm_recall": float(detected_by_alarm / len(bad_segments)) if bad_segments else 0.0,
        "bad_segment_prewarn_warning_ratio": float(prewarn_warning / len(bad_segments)) if bad_segments else 0.0,
        "bad_segment_prewarn_alarm_ratio": float(prewarn_alarm / len(bad_segments)) if bad_segments else 0.0,
        "good_segment_false_warning_ratio": float(false_warning_good_segments / len(good_segments)) if good_segments else 0.0,
        "good_segment_false_alarm_ratio": float(false_alarm_good_segments / len(good_segments)) if good_segments else 0.0,
        "warning_event_runs": int(warning_event_runs),
        "alarm_event_runs": int(alarm_event_runs),
        "mean_warning_run_length": float(mean_warning_run_length),
        "mean_alarm_run_length": float(mean_alarm_run_length),
        "median_warning_delay_samples": float(np.median(warning_delay_samples)) if warning_delay_samples else None,
        "median_alarm_delay_samples": float(np.median(alarm_delay_samples)) if alarm_delay_samples else None,
        "median_warning_delay_minutes": float(np.median(warning_delay_minutes)) if warning_delay_minutes else None,
        "median_alarm_delay_minutes": float(np.median(alarm_delay_minutes)) if alarm_delay_minutes else None,
        "switch_count": int(np.sum(states[1:] != states[:-1])) if len(states) > 1 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate event-level warning usability for a two-level policy candidate.")
    parser.add_argument(
        "--policy-summary",
        type=Path,
        default=ARTIFACT_DIR / "phase1_warning_alarm_policy_window8_summary.json",
    )
    parser.add_argument(
        "--oof-probabilities",
        type=Path,
        default=ARTIFACT_DIR / "phase1_warning_alarm_policy_window8_oof_probabilities.csv",
    )
    parser.add_argument("--gap-multiplier", type=float, default=2.0)
    parser.add_argument("--prewarn-horizon-minutes", type=float, default=120.0)
    parser.add_argument("--output-prefix", type=str, default="phase1_event_level_window8")
    args = parser.parse_args()

    summary = load_summary(args.policy_summary)
    oof_frame = pd.read_csv(args.oof_probabilities, parse_dates=["sample_time"])
    oof_frame = oof_frame.sort_values("sample_time").reset_index(drop=True)
    intervals = oof_frame["sample_time"].diff().dropna().dt.total_seconds() / 60.0
    median_interval = float(intervals.median()) if not intervals.empty else 0.0
    gap_limit_minutes = median_interval * args.gap_multiplier if median_interval > 0 else None

    warning_threshold = float(summary["recommended_thresholds"]["warning"]["threshold"])
    alarm_threshold = float(summary["recommended_thresholds"]["alarm"]["threshold"])
    probabilities = oof_frame["probability"].to_numpy(dtype=float)

    raw_states = apply_raw_two_level(probabilities, warning_threshold, alarm_threshold)
    hysteresis_states = apply_hysteresis_two_level(probabilities, warning_threshold, alarm_threshold)

    raw_event = evaluate_policy(oof_frame, raw_states, gap_limit_minutes=gap_limit_minutes or 0.0, prewarn_horizon_minutes=args.prewarn_horizon_minutes)
    hysteresis_event = evaluate_policy(oof_frame, hysteresis_states, gap_limit_minutes=gap_limit_minutes or 0.0, prewarn_horizon_minutes=args.prewarn_horizon_minutes)

    result = {
        "selected_window_minutes": int(summary["selected_window_minutes"]),
        "selected_topk_sensors": int(summary["selected_topk_sensors"]),
        "selected_stats": summary["selected_stats"],
        "selected_sensor_names": summary["selected_sensor_names"],
        "warning_threshold": warning_threshold,
        "alarm_threshold": alarm_threshold,
        "median_sample_interval_minutes": median_interval,
        "gap_limit_minutes": gap_limit_minutes,
        "prewarn_horizon_minutes": float(args.prewarn_horizon_minutes),
        "rows": [
            {"policy_name": "raw_two_level", **raw_event},
            {"policy_name": "hysteresis_two_level", **hysteresis_event},
        ],
    }

    with (ARTIFACT_DIR / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    pd.DataFrame(result["rows"]).to_csv(
        ARTIFACT_DIR / f"{args.output_prefix}_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
