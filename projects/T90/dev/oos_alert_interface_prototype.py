from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core import get_context_sensors, load_runtime_config, load_stage_aware_example_bundle
from core.window_encoder import encode_dcs_window
from oos_alert_module import (
    DEFAULT_CONFIG_PATH,
    _predict_binary_ensemble,
    fit_global_logistic_alert_model,
    fit_oos_alert_models,
    predict_global_logistic_probability,
)


DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_THRESHOLD_SUMMARY = DEFAULT_RESULTS_DIR / "oos_alert_threshold_sweep_summary.json"
DEFAULT_OUTPUT_JSON = DEFAULT_RESULTS_DIR / "oos_alert_interface_prototype_result.json"


def _parse_path_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    raise ValueError(f"Unsupported tabular file format: {source}")


def _load_threshold_profile(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Threshold summary file not found: {summary_path}. Run oos_alert_threshold_sweep.py first."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_threshold(
    payload: dict[str, object],
    *,
    threshold_profile: dict[str, object],
) -> tuple[str, float, dict[str, object]]:
    if payload.get("alert_threshold") is not None:
        threshold = float(payload["alert_threshold"])
        return "custom", threshold, {
            "source": "input_data.alert_threshold",
            "rule": "manual_override",
        }

    alert_mode = str(payload.get("alert_mode", "low_miss"))
    recommendations = threshold_profile.get("recommended_thresholds", {})
    if alert_mode not in recommendations:
        supported = ", ".join(sorted(recommendations))
        raise ValueError(f"Unsupported alert_mode `{alert_mode}`. Supported modes: {supported}.")
    chosen = recommendations[alert_mode]
    return alert_mode, float(chosen["threshold"]), {
        "source": "threshold_profile",
        "rule": threshold_profile.get("selection_rules", {}).get(alert_mode, {}),
        "historical_metrics": chosen,
    }


def predict_t90_oos_alert_v_next_dev(
    input_data: dict[str, object] | None = None,
    *,
    model_bundle: dict[str, object] | None = None,
    threshold_profile: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = input_data or {}
    config_path = Path(str(payload.get("config_path", DEFAULT_CONFIG_PATH)))
    model_name = str(payload.get("model_name", "logistic_balanced"))

    if model_bundle is None:
        if model_name == "logistic_balanced":
            model_bundle = fit_global_logistic_alert_model(config_path)
        elif model_name == "global_dcs_ensemble":
            model_bundle = fit_oos_alert_models(config_path)
        else:
            raise ValueError("Unsupported model_name. Use `logistic_balanced` or `global_dcs_ensemble`.")

    if model_name == "logistic_balanced":
        if model_bundle.get("model") is None:
            raise ValueError("Global logistic alert model is unavailable.")
    elif model_name == "global_dcs_ensemble":
        if model_bundle.get("global_dcs_model") is None:
            raise ValueError("Global DCS alert model is unavailable.")
    else:
        raise ValueError("Unsupported model_name. Use `logistic_balanced` or `global_dcs_ensemble`.")

    if threshold_profile is None:
        threshold_summary_path = Path(str(payload.get("threshold_summary_path", DEFAULT_THRESHOLD_SUMMARY)))
        threshold_profile = _load_threshold_profile(threshold_summary_path)

    runtime_config = load_runtime_config(config_path)
    sensors = get_context_sensors(runtime_config)
    dcs_window = payload.get("dcs_window")
    runtime_time = payload.get("runtime_time")

    if dcs_window is None and payload.get("dcs_window_path"):
        dcs_window = _parse_path_frame(payload["dcs_window_path"])

    if dcs_window is None:
        example_bundle = load_stage_aware_example_bundle(config_path, project_dir=PROJECT_DIR)
        dcs_window = example_bundle["dcs_window"]
        runtime_time = example_bundle["runtime_time"] if runtime_time is None else runtime_time

    current_context = encode_dcs_window(dcs_window, include_sensors=sensors)
    current_row = pd.DataFrame(
        [{feature: current_context.get(feature, np.nan) for feature in model_bundle["dcs_features"]}]
    )

    if model_name == "logistic_balanced":
        probability = float(predict_global_logistic_probability(model_bundle, current_row)[0])
    else:
        probability = float(_predict_binary_ensemble(model_bundle["global_dcs_model"], current_row)[0])

    alert_mode, threshold, threshold_meta = _resolve_threshold(payload, threshold_profile=threshold_profile)
    predicted_out_of_spec = bool(probability >= threshold)

    return {
        "runtime_context": {
            "runtime_time": str(runtime_time) if runtime_time is not None else None,
            "window_rows": int(len(dcs_window)),
            "window_minutes_expected": int(runtime_config.get("window", {}).get("minutes", 50)),
            "dcs_sensor_count": int(len(sensors)),
        },
        "method": {
            "type": "global DCS-only direct out-of-spec classifier",
            "model_name": model_name,
            "ph_required": False,
            "ph_used": False,
            "stage_required": False,
            "alert_mode": alert_mode,
        },
        "alert": {
            "target": "T90_out_of_spec",
            "definition": "1 when T90 is outside 8.45 +/- 0.25, otherwise 0",
            "probability": probability,
            "threshold": threshold,
            "predicted_out_of_spec": predicted_out_of_spec,
            "threshold_metadata": threshold_meta,
        },
        "warnings": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dev-only future-interface prototype for standalone T90 bad-sample alerting.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--threshold-summary", default=str(DEFAULT_THRESHOLD_SUMMARY), help="Threshold summary JSON path.")
    parser.add_argument("--model-name", default="logistic_balanced", help="One of: logistic_balanced, global_dcs_ensemble.")
    parser.add_argument("--alert-mode", default="low_miss", help="One of: low_miss, low_false_alarm, balanced_f1.")
    parser.add_argument("--alert-threshold", type=float, default=None, help="Optional manual threshold override.")
    parser.add_argument("--use-private-example", action="store_true", help="Use the dev private example bundle when no dcs_window is provided.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Where to write the prototype result JSON.")
    args = parser.parse_args()

    payload: dict[str, object] = {
        "config_path": args.config,
        "threshold_summary_path": args.threshold_summary,
        "model_name": args.model_name,
        "alert_mode": args.alert_mode,
    }
    if args.alert_threshold is not None:
        payload["alert_threshold"] = args.alert_threshold
    if args.use_private_example:
        payload["use_private_example"] = True

    result = predict_t90_oos_alert_v_next_dev(payload)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
