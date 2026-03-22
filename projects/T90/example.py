from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from interface import recommend_t90_controls
    from core import get_context_sensors, get_target_range, load_casebase, load_runtime_config
except ModuleNotFoundError:
    from .interface import recommend_t90_controls
    from .core import get_context_sensors, get_target_range, load_casebase, load_runtime_config


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
CASEBASE_PATH = PROJECT_DIR / "assets" / "t90_casebase.csv"


def build_online_request() -> dict[str, object]:
    runtime_config = load_runtime_config(CONFIG_PATH)
    sensors = get_context_sensors(runtime_config)
    casebase = load_casebase(CASEBASE_PATH, target_range=get_target_range(runtime_config))

    template = casebase.iloc[-1]
    window_rows = []
    runtime_time = pd.Timestamp(template["sample_time"]) + pd.Timedelta(minutes=1)
    timestamps = pd.date_range(end=runtime_time, periods=15, freq="min")

    for index, ts in enumerate(timestamps):
        row = {"time": ts}
        offset = index - (len(timestamps) - 1)
        for sensor in sensors:
            last_value = float(template.get(f"{sensor}__last", 0.0))
            slope_value = float(template.get(f"{sensor}__slope", 0.0))
            row[sensor] = last_value + slope_value * offset
        window_rows.append(row)

    current_dcs_window = pd.DataFrame(window_rows)
    online_request = {
        "dcs_window": current_dcs_window,
        "runtime_time": str(runtime_time),
        "skip_feature_ranking": True,
    }
    return online_request


def main() -> None:
    online_request = build_online_request()
    result = recommend_t90_controls(online_request)
    print("online_request", online_request)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
